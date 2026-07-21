"""Evaluate Wai baselines on public NOAA CO-OPS observations and predictions.

The live path fetches two NOAA products for each station:

- ``water_level`` verified observations
- ``predictions`` deterministic NOAA tidal predictions

The products are merged by UTC timestamp after datum and unit checks. This
allows the report to compare serious baselines:

- rolling 1-step persistence
- NOAA tidal prediction
- NOAA residual persistence
- HarmonicRidge
- GradBoost
- hybrid residual Ridge: NOAA prediction + learned residual forecast

Output naming is intentionally explicit:

- offline/mock mode writes ``reports/noaa_mock_metrics.{json,md}``
- live mode writes ``reports/noaa_live_metrics.{json,md}``
- live ``--allow-mock`` writes ``reports/noaa_allow_mock_metrics.{json,md}``

Live reports fail before writing if any record contains ``mock_used=true``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.noaa import NOAACoopsAdapter
from src.data.schema import to_model_frame
from src.models.baseline import HarmonicRidgeModel
from src.models.gradient_boost import GradBoostModel
from src.models.metrics import (
    block_bootstrap_ci,
    bootstrap_ci,
    compute_event_metrics,
    compute_metrics,
    save_metrics,
    skill_score,
)


def load_noaa_data(
    station_id: str,
    begin_date: str,
    end_date: str,
    product: str = "water_level",
    datum: str = "MLLW",
    units: str = "metric",
    time_zone: str = "gmt",
) -> pd.DataFrame:
    """Compatibility wrapper over the canonical NOAA adapter."""

    if product != "water_level" or units != "metric" or time_zone != "gmt":
        raise ValueError("NOAA evaluation requires metric GMT water_level data")
    canonical = NOAACoopsAdapter().fetch_observations(
        station_id,
        begin_date,
        end_date,
        latitude=float("nan"),
        longitude=float("nan"),
        datum=datum,
    )
    return to_model_frame(canonical)


def load_noaa_predictions(
    station_id: str,
    begin_date: str,
    end_date: str,
    datum: str = "MLLW",
    units: str = "metric",
    time_zone: str = "gmt",
) -> pd.DataFrame:
    """Compatibility wrapper over the canonical NOAA prediction adapter."""

    if units != "metric" or time_zone != "gmt":
        raise ValueError("NOAA evaluation requires metric GMT predictions")
    canonical = NOAACoopsAdapter().fetch_tide_predictions(
        station_id,
        begin_date,
        end_date,
        latitude=float("nan"),
        longitude=float("nan"),
        datum=datum,
    )
    return to_model_frame(canonical)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
TRAIN_FRAC = 0.75

# Five diverse coastal stations spanning the US coastline.
# Each entry: (noaa_id, display_name, lat, lon)
NOAA_STATIONS = [
    ("9414290", "San Francisco, CA", 37.8065, -122.4659),
    ("1612340", "Honolulu, HI", 21.3067, -157.8675),
    ("8443970", "Boston, MA", 42.3548, -71.0505),
    ("8723214", "Virginia Key, FL", 25.7306, -80.1619),
    ("9410230", "La Jolla, CA", 32.8669, -117.2571),
]

EVAL_BEGIN = "20240101"
EVAL_END = "20240128"

STORM_STATION_ID = "1612340"
STORM_BEGIN = "20240112"
STORM_END = "20240118"


class LiveNOAAFetchError(RuntimeError):
    """Raised when live NOAA fetching fails without explicit mock permission."""


class NOAAReportIntegrityError(RuntimeError):
    """Raised when a live report would contain mock records."""


def _mock_base_signal(n: int) -> np.ndarray:
    t_h = np.arange(n) * (6 / 60)
    return (
        0.5 * np.sin(2 * np.pi * t_h / 12.42)
        + 0.3 * np.sin(2 * np.pi * t_h / 24.0)
    )


def _station_seed(station_id: str) -> int:
    """Stable seed for reproducible mock NOAA fixtures."""
    return int(hashlib.md5(str(station_id).encode("utf-8")).hexdigest()[:8], 16)


def _make_mock_noaa_df(
    station_id: str,
    begin: str,
    end: str,
    lat: float,
    lon: float,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic observation stand-in matching the Wai schema for CI/offline use."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(begin, end, freq="6min", tz="UTC")
    n = len(dates)
    wl = _mock_base_signal(n) + 0.05 * rng.standard_normal(n)
    return pd.DataFrame({
        "timestamp": dates,
        "station_id": station_id,
        "water_level": wl,
        "datum": "MLLW",
        "units": "m",
        "lat": lat,
        "lon": lon,
        "source": "NOAA_COOPS_MOCK",
    })


def _make_mock_noaa_predictions_df(
    station_id: str,
    begin: str,
    end: str,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """Synthetic NOAA-prediction stand-in with the same timestamps as mock obs."""
    dates = pd.date_range(begin, end, freq="6min", tz="UTC")
    wl = _mock_base_signal(len(dates))
    return pd.DataFrame({
        "timestamp": dates,
        "station_id": station_id,
        "water_level": wl,
        "datum": "MLLW",
        "units": "m",
        "lat": lat,
        "lon": lon,
        "source": "NOAA_PREDICTIONS_MOCK",
    })


def fetch_noaa_df(
    station_id: str,
    begin: str,
    end: str,
    lat: float,
    lon: float,
    offline: bool = False,
    allow_mock: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Fetch real or mock NOAA observations for one station."""
    base_meta = {"station_id": station_id, "begin_date": begin, "end_date": end}
    if offline:
        df = _make_mock_noaa_df(
            station_id, begin, end, lat, lon, seed=_station_seed(station_id)
        )
        return df, {
            **base_meta,
            "observation_source": "NOAA_COOPS_MOCK",
            "data_source": "NOAA_COOPS_MOCK",
            "mock_used": True,
            "reason": "offline_mode",
        }

    try:
        df = load_noaa_data(
            station_id=station_id,
            begin_date=begin,
            end_date=end,
            product="water_level",
            datum="MLLW",
            units="metric",
            time_zone="gmt",
        )
        df["lat"] = lat
        df["lon"] = lon
        return df, {
            **base_meta,
            "observation_source": "NOAA_COOPS",
            "data_source": "NOAA_COOPS",
            "mock_used": False,
        }
    except Exception as exc:
        if not allow_mock:
            raise LiveNOAAFetchError(
                f"Live NOAA observation fetch failed for station {station_id} "
                f"({begin}-{end}): {exc}. Re-run with NOAA_OFFLINE=1 or "
                "--allow-mock if synthetic stand-ins are intended."
            ) from exc
        warnings.warn(
            f"NOAA observation fetch failed for station {station_id}: {exc}. "
            "Using mock observations (--allow-mock was set).",
            stacklevel=2,
        )
        df = _make_mock_noaa_df(
            station_id, begin, end, lat, lon, seed=_station_seed(station_id)
        )
        return df, {
            **base_meta,
            "observation_source": "NOAA_COOPS_MOCK",
            "data_source": "NOAA_COOPS_ALLOW_MOCK",
            "mock_used": True,
            "reason": "live_observation_fetch_failed",
            "error": str(exc),
        }


def fetch_noaa_predictions_df(
    station_id: str,
    begin: str,
    end: str,
    lat: float,
    lon: float,
    offline: bool = False,
    allow_mock: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Fetch real or mock NOAA tidal predictions for one station."""
    base_meta = {"station_id": station_id, "begin_date": begin, "end_date": end}
    if offline:
        df = _make_mock_noaa_predictions_df(station_id, begin, end, lat, lon)
        return df, {
            **base_meta,
            "prediction_source": "NOAA_PREDICTIONS_MOCK",
            "mock_used": True,
            "reason": "offline_mode",
        }

    try:
        df = load_noaa_predictions(
            station_id=station_id,
            begin_date=begin,
            end_date=end,
            datum="MLLW",
            units="metric",
            time_zone="gmt",
        )
        df["lat"] = lat
        df["lon"] = lon
        return df, {
            **base_meta,
            "prediction_source": "NOAA_PREDICTIONS",
            "mock_used": False,
        }
    except Exception as exc:
        if not allow_mock:
            raise LiveNOAAFetchError(
                f"Live NOAA prediction fetch failed for station {station_id} "
                f"({begin}-{end}): {exc}. Re-run with NOAA_OFFLINE=1 or "
                "--allow-mock if synthetic stand-ins are intended."
            ) from exc
        warnings.warn(
            f"NOAA prediction fetch failed for station {station_id}: {exc}. "
            "Using mock predictions (--allow-mock was set).",
            stacklevel=2,
        )
        df = _make_mock_noaa_predictions_df(station_id, begin, end, lat, lon)
        return df, {
            **base_meta,
            "prediction_source": "NOAA_PREDICTIONS_MOCK",
            "mock_used": True,
            "reason": "live_prediction_fetch_failed",
            "error": str(exc),
        }


def _single_value(df: pd.DataFrame, col: str, label: str) -> str:
    values = [str(v) for v in df[col].dropna().unique()]
    if not values:
        raise ValueError(f"{label} has no {col} values")
    if len(values) > 1:
        raise ValueError(f"{label} has multiple {col} values: {values}")
    return values[0]


def merge_observations_predictions(
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Merge NOAA observations and predictions by timestamp with unit checks."""
    obs = observations.copy()
    pred = predictions.copy()
    obs["timestamp"] = pd.to_datetime(obs["timestamp"], utc=True)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)

    obs_datum = _single_value(obs, "datum", "observations").upper()
    pred_datum = _single_value(pred, "datum", "predictions").upper()
    if obs_datum != pred_datum:
        raise ValueError(
            f"NOAA observation/prediction datum mismatch: {obs_datum} vs {pred_datum}"
        )

    obs_units = _single_value(obs, "units", "observations").lower()
    pred_units = _single_value(pred, "units", "predictions").lower()
    if obs_units != pred_units:
        raise ValueError(
            f"NOAA observation/prediction unit mismatch: {obs_units} vs {pred_units}"
        )

    obs_cols = [
        "timestamp", "station_id", "water_level", "datum",
        "units", "lat", "lon", "source",
    ]
    obs = obs[obs_cols].rename(columns={
        "water_level": "observed_water_level",
        "source": "observation_source",
    })
    pred = pred[["timestamp", "water_level", "source"]].rename(columns={
        "water_level": "noaa_prediction",
        "source": "prediction_source",
    })

    merged = obs.merge(pred, on="timestamp", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("NOAA observation/prediction merge produced no matching timestamps")
    merged["water_level"] = pd.to_numeric(
        merged["observed_water_level"], errors="coerce"
    )
    merged["noaa_prediction"] = pd.to_numeric(
        merged["noaa_prediction"], errors="coerce"
    )
    merged["source"] = merged["observation_source"]
    return merged.dropna(subset=["water_level", "noaa_prediction"]).sort_values(
        "timestamp"
    ).reset_index(drop=True)


def fetch_noaa_pair(
    station_id: str,
    begin: str,
    end: str,
    lat: float,
    lon: float,
    offline: bool = False,
    allow_mock: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Fetch and merge NOAA observations + predictions for one station."""
    obs, obs_prov = fetch_noaa_df(
        station_id, begin, end, lat, lon, offline=offline, allow_mock=allow_mock
    )
    pred, pred_prov = fetch_noaa_predictions_df(
        station_id, begin, end, lat, lon, offline=offline, allow_mock=allow_mock
    )
    merged = merge_observations_predictions(obs, pred)
    mock_used = bool(obs_prov.get("mock_used") or pred_prov.get("mock_used"))
    data_source = "NOAA_COOPS_MOCK" if offline else (
        "NOAA_COOPS_ALLOW_MOCK" if mock_used else "NOAA_COOPS"
    )
    provenance = {
        "station_id": station_id,
        "begin_date": begin,
        "end_date": end,
        "data_source": data_source,
        "observation_source": obs_prov.get("observation_source"),
        "prediction_source": pred_prov.get("prediction_source"),
        "mock_used": mock_used,
        "n_merged": int(len(merged)),
    }
    if obs_prov.get("error") or pred_prov.get("error"):
        provenance["errors"] = [
            e for e in (obs_prov.get("error"), pred_prov.get("error")) if e
        ]
    return merged, provenance


def rolling_persistence_1step(train_series: pd.Series, test_series: pd.Series) -> np.ndarray:
    """Rolling 1-step persistence: pred[0]=last train, pred[t]=test[t-1]."""
    vals = test_series.to_numpy(dtype=float)
    pred = np.empty(len(vals), dtype=float)
    if len(vals) == 0:
        return pred
    pred[0] = float(train_series.dropna().iloc[-1])
    if len(vals) > 1:
        pred[1:] = vals[:-1]
    return pred


def _metric_record(
    actual: np.ndarray,
    forecast: np.ndarray,
    rolling_reference: np.ndarray,
    noaa_reference: np.ndarray,
) -> dict:
    m = compute_metrics(actual, forecast)
    rolling_m = compute_metrics(actual, rolling_reference)
    noaa_m = compute_metrics(actual, noaa_reference)
    m["mae_skill_vs_rolling_persistence"] = skill_score(m["mae"], rolling_m["mae"])
    m["rmse_skill_vs_rolling_persistence"] = skill_score(m["rmse"], rolling_m["rmse"])
    m["mae_skill_vs_noaa_prediction"] = skill_score(m["mae"], noaa_m["mae"])
    m["rmse_skill_vs_noaa_prediction"] = skill_score(m["rmse"], noaa_m["rmse"])
    return m


def _add_intervals(metrics: dict, actual: np.ndarray, pred: np.ndarray) -> dict:
    block_mae = block_bootstrap_ci(actual, pred, metric="mae", n_boot=500)
    block_rmse = block_bootstrap_ci(actual, pred, metric="rmse", n_boot=500)
    metrics["mae_block_ci_95"] = block_mae
    metrics["rmse_block_ci_95"] = block_rmse
    metrics["mae_iid_ci_95"] = bootstrap_ci(actual, pred, metric="mae", n_boot=500)
    metrics["rmse_iid_ci_95"] = bootstrap_ci(actual, pred, metric="rmse", n_boot=500)
    metrics["mae_ci_95"] = (block_mae["lower"], block_mae["upper"])
    metrics["rmse_ci_95"] = (block_rmse["lower"], block_rmse["upper"])
    return metrics


def _model_input(df: pd.DataFrame, target_col: str = "water_level") -> pd.DataFrame:
    """Return a schema-clean frame so NOAA prediction is not an accidental feature."""
    cols = ["timestamp", "station_id", target_col, "datum", "units", "lat", "lon", "source"]
    out = df[cols].copy()
    if target_col != "water_level":
        out = out.rename(columns={target_col: "water_level"})
    return out


def evaluate_station(
    df: pd.DataFrame,
    station_label: str,
    holdout_type: str = "temporal",
    provenance: Optional[dict] = None,
) -> dict:
    """Evaluate NOAA baselines and residual hybrid models on one merged station."""
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    df["water_level"] = pd.to_numeric(df["water_level"], errors="coerce")
    df["noaa_prediction"] = pd.to_numeric(df["noaa_prediction"], errors="coerce")
    df = df.dropna(subset=["water_level", "noaa_prediction"])

    prov = dict(provenance or {})
    n = len(df)
    if n < 100:
        return {"error": f"Insufficient merged data (n={n})", "station": station_label, **prov}

    n_train = int(n * TRAIN_FRAC)
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy().reset_index(drop=True)

    actual = test["water_level"].to_numpy(dtype=float)
    noaa_pred = test["noaa_prediction"].to_numpy(dtype=float)
    rolling_pred = rolling_persistence_1step(train["water_level"], test["water_level"])

    train_resid = train["water_level"] - train["noaa_prediction"]
    test_resid = test["water_level"] - test["noaa_prediction"]
    resid_persist = rolling_persistence_1step(train_resid, test_resid)
    noaa_resid_persist_pred = noaa_pred + resid_persist

    records = {
        "rolling_persistence": _metric_record(
            actual, rolling_pred, rolling_pred, noaa_pred
        ),
        "noaa_prediction": _metric_record(
            actual, noaa_pred, rolling_pred, noaa_pred
        ),
        "noaa_residual_persistence": _metric_record(
            actual, noaa_resid_persist_pred, rolling_pred, noaa_pred
        ),
    }

    train_clean = _model_input(train)
    test_clean = _model_input(test)

    try:
        harmonic = HarmonicRidgeModel(alpha=1.0).fit(train_clean)
        h_aligned = harmonic.predict_aligned(test_clean)
        rows = h_aligned["_source_row"].to_numpy(dtype=int)
        h_actual = h_aligned["actual"].to_numpy(dtype=float)
        h_pred = h_aligned["prediction"].to_numpy(dtype=float)
        h_metrics = _metric_record(h_actual, h_pred, rolling_pred[rows], noaa_pred[rows])
        records["harmonic_ridge"] = _add_intervals(h_metrics, h_actual, h_pred)
    except Exception as exc:
        records["harmonic_ridge"] = {"error": str(exc)}

    try:
        gradboost = GradBoostModel().fit(train_clean)
        gb_aligned = gradboost.predict_aligned(test_clean)
        rows = gb_aligned["_source_row"].to_numpy(dtype=int)
        gb_actual = gb_aligned["actual"].to_numpy(dtype=float)
        gb_pred = gb_aligned["prediction"].to_numpy(dtype=float)
        gb_metrics = _metric_record(gb_actual, gb_pred, rolling_pred[rows], noaa_pred[rows])
        records["grad_boost"] = _add_intervals(gb_metrics, gb_actual, gb_pred)
    except Exception as exc:
        records["grad_boost"] = {"error": str(exc)}

    try:
        train_resid_df = _model_input(train.assign(residual=train_resid), target_col="residual")
        test_resid_df = _model_input(test.assign(residual=test_resid), target_col="residual")
        resid_model = HarmonicRidgeModel(alpha=1.0).fit(train_resid_df)
        r_aligned = resid_model.predict_aligned(test_resid_df)
        rows = r_aligned["_source_row"].to_numpy(dtype=int)
        final_pred = noaa_pred[rows] + r_aligned["prediction"].to_numpy(dtype=float)
        final_actual = actual[rows]
        hybrid_metrics = _metric_record(
            final_actual, final_pred, rolling_pred[rows], noaa_pred[rows]
        )
        records["hybrid_residual_ridge"] = _add_intervals(
            hybrid_metrics, final_actual, final_pred
        )
    except Exception as exc:
        records["hybrid_residual_ridge"] = {"error": str(exc)}

    train_wl = train["water_level"].dropna()
    event_threshold = float(train_wl.mean() + 2 * train_wl.std())
    event_m = {}
    if isinstance(records.get("harmonic_ridge"), dict) and "error" not in records["harmonic_ridge"]:
        try:
            event_m = compute_event_metrics(h_actual, h_pred, event_threshold)
        except Exception:
            event_m = {}

    return {
        "station": station_label,
        "holdout_type": holdout_type,
        "data_source": prov.get("data_source", "UNKNOWN"),
        "observation_source": prov.get("observation_source"),
        "prediction_source": prov.get("prediction_source"),
        "mock_used": prov.get("mock_used", None),
        "station_id": prov.get("station_id"),
        "begin_date": prov.get("begin_date"),
        "end_date": prov.get("end_date"),
        "n_merged": int(n),
        "n_train": int(n_train),
        "n_test": int(len(test)),
        "train_start": str(train["timestamp"].iloc[0]),
        "train_end": str(train["timestamp"].iloc[-1]),
        "test_start": str(test["timestamp"].iloc[0]),
        "test_end": str(test["timestamp"].iloc[-1]),
        "event_threshold_m": round(event_threshold, 4),
        "event_threshold_source": "train_mean_plus_2std",
        **records,
        "event_metrics_harmonic_ridge": event_m,
    }


def _fmt_float(value: object, digits: int = 4) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    if np.isnan(v):
        return "-"
    return f"{v:.{digits}f}"


def format_results_md(results: dict, report_kind: str = "live") -> str:
    """Render evaluation results as a Markdown report."""
    source_line = {
        "mock": "Offline mock fixtures. These are synthetic sanity checks.",
        "live": "Live NOAA CO-OPS observations merged to NOAA tidal predictions.",
        "allow_mock": "Live attempt with explicit mock fallback allowed.",
    }.get(report_kind, report_kind)
    lines = [
        f"# Wai NOAA {report_kind.replace('_', ' ').title()} Evaluation",
        "",
        f"> Data source: {source_line}",
        "> NOAA tidal predictions are deterministic harmonics; they are a serious baseline, not ground truth.",
        "",
    ]

    model_labels = [
        ("rolling_persistence", "Rolling persistence"),
        ("noaa_prediction", "NOAA prediction"),
        ("noaa_residual_persistence", "NOAA residual persistence"),
        ("harmonic_ridge", "HarmonicRidge"),
        ("grad_boost", "GradBoost"),
        ("hybrid_residual_ridge", "Hybrid residual Ridge"),
    ]

    for station_id, res in results.items():
        label = res.get("station", station_id)
        holdout = res.get("holdout_type", "temporal")
        src = res.get("data_source", "UNKNOWN")
        mock = res.get("mock_used")
        mock_tag = " MOCK" if mock else ""
        lines.append(f"## {label} - {holdout} holdout - {src}{mock_tag}")

        if "error" in res:
            lines.append(f"\n*Error: {res['error']}*\n")
            continue

        lines += [
            "",
            f"- Station: `{res.get('station_id','?')}` window {res.get('begin_date','?')}-{res.get('end_date','?')}",
            f"- Observations: `{res.get('observation_source')}`; predictions: `{res.get('prediction_source')}`; mock_used={mock}",
            f"- Train: {res.get('train_start','')} to {res.get('train_end','')} ({res.get('n_train', '?'):,} obs)",
            f"- Test: {res.get('test_start','')} to {res.get('test_end','')} ({res.get('n_test', '?'):,} obs)",
            f"- Event threshold: train mean + 2 std = {res.get('event_threshold_m', '?')} m",
            "",
            "| Model | MAE | RMSE | R2/NSE | MAE skill vs rolling | MAE skill vs NOAA |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for key, model_label in model_labels:
            m = res.get(key, {})
            if "error" in m:
                lines.append(f"| {model_label} | error | - | - | - | - |")
                continue
            lines.append(
                "| {label} | {mae} | {rmse} | {r2} | {s_roll} | {s_noaa} |".format(
                    label=model_label,
                    mae=_fmt_float(m.get("mae")),
                    rmse=_fmt_float(m.get("rmse")),
                    r2=_fmt_float(m.get("r2")),
                    s_roll=_fmt_float(m.get("mae_skill_vs_rolling_persistence")),
                    s_noaa=_fmt_float(m.get("mae_skill_vs_noaa_prediction")),
                )
            )
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- Mock reports are synthetic CI artifacts and must not be presented as real NOAA performance.",
        "- Live reports hard-fail on mock records; mixed live/mock runs use the `noaa_allow_mock` filename.",
        "- No model includes meteorological forcing, so residual surge skill is limited.",
        "",
    ]
    return "\n".join(lines)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Wai models on NOAA observations and tidal predictions."
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="In live mode, fall back to mock data for failed station fetches. "
             "Writes noaa_allow_mock_metrics.*, never noaa_live_metrics.*.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode (equivalent to NOAA_OFFLINE=1).",
    )
    return parser.parse_args(argv)


def _env_offline() -> bool:
    return os.environ.get("NOAA_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def _report_kind(offline: bool, allow_mock: bool) -> str:
    if offline:
        return "mock"
    if allow_mock:
        return "allow_mock"
    return "live"


def _report_paths(offline: bool, allow_mock: bool) -> tuple[Path, Path, str]:
    kind = _report_kind(offline, allow_mock)
    stem = {
        "mock": "noaa_mock_metrics",
        "live": "noaa_live_metrics",
        "allow_mock": "noaa_allow_mock_metrics",
    }[kind]
    return REPORTS_DIR / f"{stem}.json", REPORTS_DIR / f"{stem}.md", kind


def _assert_live_report_has_no_mock(summary: dict, kind: str) -> None:
    any_mock = any(
        isinstance(v, dict) and v.get("mock_used", False)
        for k, v in summary.items()
        if not k.startswith("_")
    )
    if kind == "live" and any_mock:
        raise NOAAReportIntegrityError(
            "Refusing to write noaa_live_metrics.* because at least one record has mock_used=true"
        )


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    offline = args.offline or _env_offline()
    allow_mock = bool(args.allow_mock)
    json_path, md_path, kind = _report_paths(offline, allow_mock)

    if offline:
        print("NOAA_OFFLINE / --offline: using mock fixtures.")
    else:
        mode = "live with explicit allow-mock fallback" if allow_mock else "live hard-fail"
        print(f"Fetching NOAA observations + predictions: {mode}")

    all_results: dict = {}
    for station_id, label, lat, lon in NOAA_STATIONS:
        print(f"\n  -> {label} ({station_id})")
        merged, prov = fetch_noaa_pair(
            station_id, EVAL_BEGIN, EVAL_END, lat, lon,
            offline=offline, allow_mock=allow_mock,
        )
        res = evaluate_station(
            merged, station_label=label, holdout_type="temporal", provenance=prov
        )
        all_results[station_id] = res
        h = res.get("hybrid_residual_ridge", {})
        if "mae" in h:
            mock_tag = " [MOCK]" if prov.get("mock_used") else ""
            print(
                f"    HybridResidualRidge MAE={h['mae']:.4f} "
                f"RMSE={h['rmse']:.4f} R2={h['r2']:.4f}{mock_tag}"
            )

    print(f"\n  -> Storm-period holdout: Honolulu {STORM_BEGIN}-{STORM_END}")
    storm_id, storm_label, storm_lat, storm_lon = next(
        (s for s in NOAA_STATIONS if s[0] == STORM_STATION_ID),
        (STORM_STATION_ID, "Honolulu, HI", 21.3067, -157.8675),
    )
    storm_merged, storm_prov = fetch_noaa_pair(
        STORM_STATION_ID, STORM_BEGIN, STORM_END, storm_lat, storm_lon,
        offline=offline, allow_mock=allow_mock,
    )
    all_results[f"{storm_id}_storm"] = evaluate_station(
        storm_merged,
        station_label=f"{storm_label} (storm period)",
        holdout_type="event",
        provenance=storm_prov,
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    summary = {
        "_meta": {
            "report_kind": kind,
            "offline_mode": bool(offline),
            "allow_mock": bool(allow_mock),
            "any_mock_used": any(r.get("mock_used", False) for r in all_results.values()),
            "stations_evaluated": len(all_results),
            "output_json": json_path.name,
            "output_markdown": md_path.name,
        },
        **all_results,
    }
    _assert_live_report_has_no_mock(summary, kind)

    save_metrics(summary, json_path)
    md_path.write_text(format_results_md(all_results, report_kind=kind), encoding="utf-8")
    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
