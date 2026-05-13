"""Evaluate Wai models on real public NOAA CO-OPS observations.

Evaluation strategy
-------------------
Three holdout modes are used to stress-test generalisation:

1. **Temporal holdout** — last 25% of a contiguous date range per station.
   Mirrors the synthetic-demo evaluation so metrics are directly comparable.

2. **Station holdout** — train on N-1 stations, evaluate on the held-out
   station. Tests geographic generalisation (not implemented here — requires
   a multi-station dataset fetched at consistent timestamps; left as TODO).

3. **Event / storm-period holdout** — a pre-defined date window known to
   contain a storm or king-tide event. Tests model behaviour during extremes.
   The script reports synthetic, tidecast, and real-observation metrics
   separately so the reader can see degradation under realistic conditions.

Data
----
This script fetches data live from the NOAA CO-OPS public API (no key
required). Each station is limited to a configurable DATE_RANGE_DAYS window
(≤31 per API call). Results are written to:
    reports/noaa_public_metrics.json
    reports/noaa_public_metrics.md

Live vs mock mode
-----------------
- Default ("live") mode requires the NOAA API to be reachable. If a fetch
  fails for any reason the script raises immediately — silent mock fallback
  would falsely advertise "real NOAA evaluation" metrics.
- Mock data is permitted only when:
    NOAA_OFFLINE=1   (environment variable), or
    --allow-mock     (explicit CLI flag).
- Every record in the JSON output carries `data_source` and `mock_used`,
  along with `station_id`, `begin_date`, and `end_date`, so a reader can
  always tell whether a metric was computed against real observations.

Usage
-----
    python -m scripts.evaluate_noaa_public               # live NOAA fetch (hard-fails on error)
    NOAA_OFFLINE=1 python -m scripts.evaluate_noaa_public  # offline mode (mock fixtures)
    python -m scripts.evaluate_noaa_public --allow-mock    # explicit mock fallback on per-station error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import _parse_noaa_response, load_noaa_data
from src.features.engineering import build_feature_matrix
from src.models.baseline import HarmonicRidgeModel
from src.models.gradient_boost import GradBoostModel
from src.models.metrics import (
    block_bootstrap_ci,
    bootstrap_ci,
    compute_event_metrics,
    compute_metrics,
    save_metrics,
)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
TRAIN_FRAC = 0.75

# ── Station registry ─────────────────────────────────────────────────────────
# Five diverse coastal stations spanning the US coastline.
# Each entry: (noaa_id, display_name, lat, lon)
NOAA_STATIONS = [
    ("9414290", "San Francisco, CA",    37.8065, -122.4659),
    ("1612340", "Honolulu, HI",         21.3067, -157.8675),
    ("8443970", "Boston, MA",           42.3548,  -71.0505),
    ("8723214", "Virginia Key, FL",     25.7306,  -80.1619),
    ("9410230", "La Jolla, CA",         32.8669, -117.2571),
]

# Evaluation window: a recent 28-day period (safe within 31-day API limit)
EVAL_BEGIN = "20240101"
EVAL_END   = "20240128"

# Storm-period window — Honolulu had notable high water during this window
STORM_STATION_ID = "1612340"
STORM_BEGIN = "20240112"
STORM_END   = "20240118"

# ── Mock fixtures for offline / CI mode ──────────────────────────────────────

def _make_mock_noaa_df(
    station_id: str,
    begin: str,
    end: str,
    lat: float,
    lon: float,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic stand-in matching the Wai schema for CI/offline use."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(begin, end, freq="6min", tz="UTC")
    n = len(dates)
    t_h = np.arange(n) * (6 / 60)
    wl = (
        0.5 * np.sin(2 * np.pi * t_h / 12.42)
        + 0.3 * np.sin(2 * np.pi * t_h / 24.0)
        + 0.05 * rng.standard_normal(n)
    )
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


class LiveNOAAFetchError(RuntimeError):
    """Raised when a live NOAA fetch fails and mock fallback is not authorised.

    The script must fail hard in this case so reports never silently advertise
    "real NOAA evaluation" while actually showing mock-data metrics.
    """


def fetch_noaa_df(
    station_id: str,
    begin: str,
    end: str,
    lat: float,
    lon: float,
    offline: bool = False,
    allow_mock: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Fetch real or mock observations for one station.

    Returns
    -------
    (df, provenance)
        provenance carries `data_source` (NOAA_COOPS or NOAA_COOPS_MOCK),
        `mock_used` (bool), `station_id`, `begin_date`, `end_date`, and an
        optional `error` string when a live fetch failed and was substituted.

    Raises
    ------
    LiveNOAAFetchError
        When `offline=False` and `allow_mock=False` and the live fetch fails.
    """
    base_meta = {
        "station_id": station_id,
        "begin_date": begin,
        "end_date": end,
    }
    if offline:
        df = _make_mock_noaa_df(station_id, begin, end, lat, lon,
                                seed=hash(station_id) % 2**31)
        return df, {**base_meta, "data_source": "NOAA_COOPS_MOCK",
                    "mock_used": True, "reason": "offline_mode"}

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
        return df, {**base_meta, "data_source": "NOAA_COOPS", "mock_used": False}
    except Exception as exc:
        if not allow_mock:
            # Hard-fail: under live mode we must never silently degrade to mock.
            raise LiveNOAAFetchError(
                f"Live NOAA fetch failed for station {station_id} ({begin}–{end}): {exc}. "
                "Re-run with NOAA_OFFLINE=1 or --allow-mock if synthetic stand-ins are intended."
            ) from exc
        warnings.warn(
            f"NOAA fetch failed for station {station_id}: {exc}. "
            "Using mock fixture (--allow-mock was set).",
            stacklevel=2,
        )
        df = _make_mock_noaa_df(station_id, begin, end, lat, lon)
        return df, {**base_meta, "data_source": "NOAA_COOPS_MOCK",
                    "mock_used": True, "reason": "live_fetch_failed",
                    "error": str(exc)}


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_station(
    df: pd.DataFrame,
    station_label: str,
    holdout_type: str = "temporal",
    provenance: Optional[dict] = None,
) -> dict:
    """Train HarmonicRidge + GradBoost on a real-obs DataFrame, evaluate on holdout.

    Returns a metrics dict ready for JSON serialisation. Every record carries
    `data_source`, `mock_used`, `station_id`, `begin_date`, and `end_date` so
    downstream readers can never mistake mock metrics for real ones.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["water_level"] = pd.to_numeric(df["water_level"], errors="coerce")
    df = df.dropna(subset=["water_level"])

    prov = dict(provenance or {})
    n = len(df)
    if n < 100:
        return {
            "error": f"Insufficient data (n={n})",
            "station": station_label,
            **prov,
        }

    n_train = int(n * TRAIN_FRAC)
    train, test = df.iloc[:n_train], df.iloc[n_train:]

    # --- Rolling 1-step persistence ---
    test_vals = test["water_level"].values
    persist_pred = np.concatenate([
        [float(train["water_level"].dropna().iloc[-1])],
        test_vals[:-1],
    ])
    persist_metrics = compute_metrics(test_vals, persist_pred)

    # --- HarmonicRidge ---
    try:
        harm = HarmonicRidgeModel(alpha=1.0).fit(train)
        harm_metrics = harm.evaluate(test)
        _, y_te = build_feature_matrix(test)
        h_preds = harm.predict_on(test)
        h_preds = h_preds[-len(y_te):]
        block_h = block_bootstrap_ci(y_te.values, h_preds, n_boot=500)
        harm_metrics["mae_block_ci_95"] = block_h
        harm_metrics["mae_iid_ci_95"] = bootstrap_ci(y_te.values, h_preds, n_boot=500)
        harm_metrics["mae_ci_95"] = (block_h["lower"], block_h["upper"])
    except Exception as exc:
        harm_metrics = {"error": str(exc)}

    # --- GradBoost ---
    try:
        gb = GradBoostModel().fit(train)
        gb_metrics = gb.evaluate(test)
        _, y_te = build_feature_matrix(test)
        gb_preds = gb.predict_on(test)
        gb_preds = gb_preds[-len(y_te):]
        block_g = block_bootstrap_ci(y_te.values, gb_preds, n_boot=500)
        gb_metrics["mae_block_ci_95"] = block_g
        gb_metrics["mae_iid_ci_95"] = bootstrap_ci(y_te.values, gb_preds, n_boot=500)
        gb_metrics["mae_ci_95"] = (block_g["lower"], block_g["upper"])
    except Exception as exc:
        gb_metrics = {"error": str(exc)}

    # --- Event metrics at mean+2σ threshold fitted on training data ---
    train_wl = train["water_level"].dropna()
    event_threshold = float(train_wl.mean() + 2 * train_wl.std())
    try:
        _, y_te_aligned = build_feature_matrix(test)
        h_preds_aligned = harm.predict_on(test)[-len(y_te_aligned):]
        event_m = compute_event_metrics(y_te_aligned.values, h_preds_aligned, event_threshold)
    except Exception:
        event_m = {}

    return {
        "station": station_label,
        "holdout_type": holdout_type,
        "data_source": prov.get("data_source", "UNKNOWN"),
        "mock_used": prov.get("mock_used", None),
        "station_id": prov.get("station_id"),
        "begin_date": prov.get("begin_date"),
        "end_date": prov.get("end_date"),
        "n_train": int(n_train),
        "n_test": int(len(test)),
        "train_start": str(train["timestamp"].iloc[0]),
        "train_end": str(train["timestamp"].iloc[-1]),
        "test_start": str(test["timestamp"].iloc[0]),
        "test_end": str(test["timestamp"].iloc[-1]),
        "event_threshold_m": round(event_threshold, 4),
        "persistence_rolling": persist_metrics,
        "harmonic_ridge": harm_metrics,
        "grad_boost": gb_metrics,
        "event_metrics_harmonic_ridge": event_m,
    }


def format_results_md(results: dict) -> str:
    """Render evaluation results as a Markdown report."""
    lines = [
        "# Wai — Real NOAA CO-OPS Observation Evaluation",
        "",
        "> **Data source**: Live NOAA CO-OPS water_level observations (public API).",
        "> Metrics on *real sensor data* will differ from the synthetic-demo results.",
        "> Higher MAE/RMSE is expected due to surge, noise, and datum uncertainty.",
        "",
    ]

    for station_id, res in results.items():
        label = res.get("station", station_id)
        holdout = res.get("holdout_type", "temporal")
        src = res.get("data_source", "UNKNOWN")
        mock = res.get("mock_used")
        mock_tag = " · MOCK" if mock else ""
        lines.append(f"## {label} — {holdout} holdout · {src}{mock_tag}")

        if "error" in res:
            lines.append(f"\n*Error: {res['error']}*\n")
            continue

        lines += [
            "",
            f"- Source: `{src}` · mock_used={mock}",
            f"- Station: `{res.get('station_id','?')}` "
            f"window {res.get('begin_date','?')}–{res.get('end_date','?')}",
            f"- Train: {res.get('train_start','')} → {res.get('train_end','')} "
            f"({res.get('n_train', '?'):,} obs)",
            f"- Test:  {res.get('test_start','')} → {res.get('test_end','')} "
            f"({res.get('n_test', '?'):,} obs)",
            f"- Event threshold (mean+2σ on train): {res.get('event_threshold_m', '?')} m",
            "",
            "| Model | MAE (m) | RMSE (m) | R² | MAE 95% CI |",
            "|-------|---------|----------|----|------------|",
        ]

        for key, label_str in [
            ("persistence_rolling", "Persistence (rolling 1-step)"),
            ("harmonic_ridge", "HarmonicRidge"),
            ("grad_boost", "GradBoost"),
        ]:
            m = res.get(key, {})
            if "error" in m:
                lines.append(f"| {label_str} | error | — | — | — |")
                continue
            mae = m.get("mae", float("nan"))
            rmse = m.get("rmse", float("nan"))
            r2 = m.get("r2", float("nan"))
            ci = m.get("mae_ci_95", ("", ""))
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci and ci[0] != "" else "—"
            lines.append(
                f"| {label_str} | {mae:.4f} | {rmse:.4f} | "
                f"{r2:.4f} | {ci_str} |"
            )

        ev = res.get("event_metrics_harmonic_ridge", {})
        if ev:
            lines += [
                "",
                "**Event metrics (HarmonicRidge, threshold exceedances):**",
                "",
                f"- Precision: {ev.get('precision', '—')}",
                f"- Recall: {ev.get('recall', '—')}",
                f"- F1: {ev.get('f1', '—')}",
                f"- Peak error (on event steps): {ev.get('peak_error_m', '—')} m",
                f"- Side-of-threshold agreement: {ev.get('threshold_agree', '—')}",
            ]
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- Results on real NOAA data are **not comparable** to synthetic-demo metrics.",
        "- Surge and meteorological forcing are not modelled; storm-period errors will be higher.",
        "- Station holdout and multi-station generalisation evaluation are future work.",
        "- This script uses a single 28-day window; longer evaluations require chunked API calls.",
        "",
    ]
    return "\n".join(lines)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Wai models on NOAA CO-OPS public observations."
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="In live mode, fall back to mock data for stations whose live "
             "fetch fails (off by default — script fails hard on fetch errors).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode (equivalent to NOAA_OFFLINE=1).",
    )
    return parser.parse_args(argv)


def _env_offline() -> bool:
    return os.environ.get("NOAA_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    offline = args.offline or _env_offline()
    allow_mock = bool(args.allow_mock)

    if offline:
        print("NOAA_OFFLINE / --offline — using mock fixtures (no network calls).")
    else:
        mode = "live (allow-mock=True)" if allow_mock else "live (hard-fail on fetch errors)"
        print(f"Fetching real NOAA CO-OPS observations · mode={mode} …")

    all_results: dict = {}

    for station_id, label, lat, lon in NOAA_STATIONS:
        print(f"\n  → {label} ({station_id}) …")
        df, prov = fetch_noaa_df(
            station_id, EVAL_BEGIN, EVAL_END, lat, lon,
            offline=offline, allow_mock=allow_mock,
        )
        res = evaluate_station(df, station_label=label, holdout_type="temporal", provenance=prov)
        all_results[station_id] = res

        m_h = res.get("harmonic_ridge", {})
        if "mae" in m_h:
            ci = m_h.get("mae_ci_95", ("", ""))
            ci_str = f" CI=[{ci[0]:.4f},{ci[1]:.4f}]" if ci and ci[0] != "" else ""
            mock_tag = " [MOCK]" if prov.get("mock_used") else ""
            print(f"    HarmonicRidge  MAE={m_h['mae']:.4f}  "
                  f"RMSE={m_h['rmse']:.4f}  R²={m_h['r2']:.4f}{ci_str}{mock_tag}")
        elif "error" in m_h:
            print(f"    HarmonicRidge  error: {m_h['error']}")

    # Storm-period event holdout (Honolulu)
    print(f"\n  → Storm-period holdout: Honolulu {STORM_BEGIN}–{STORM_END}")
    storm_id, storm_label, storm_lat, storm_lon = next(
        (s for s in NOAA_STATIONS if s[0] == STORM_STATION_ID),
        (STORM_STATION_ID, "Honolulu HI (storm)", 21.3067, -157.8675),
    )
    storm_df, storm_prov = fetch_noaa_df(
        STORM_STATION_ID, STORM_BEGIN, STORM_END,
        storm_lat, storm_lon, offline=offline, allow_mock=allow_mock,
    )
    storm_res = evaluate_station(
        storm_df, station_label=f"{storm_label} (storm period)",
        holdout_type="event", provenance=storm_prov,
    )
    all_results[f"{STORM_STATION_ID}_storm"] = storm_res

    REPORTS_DIR.mkdir(exist_ok=True)

    # Top-level run header carries the global flags so readers know whether
    # any record could have been mock without scanning every entry.
    summary = {
        "_meta": {
            "offline_mode": bool(offline),
            "allow_mock": bool(allow_mock),
            "any_mock_used": any(
                r.get("mock_used", False) for r in all_results.values()
            ),
            "stations_evaluated": len(all_results),
        },
        **all_results,
    }

    json_path = REPORTS_DIR / "noaa_public_metrics.json"
    save_metrics(summary, json_path)
    print(f"\nSaved: {json_path}")

    md_path = REPORTS_DIR / "noaa_public_metrics.md"
    md_path.write_text(format_results_md(all_results), encoding="utf-8")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
