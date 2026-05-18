"""Multi-horizon forecast evaluation for Wai.

Evaluates Persistence, HarmonicRidge, and GradBoost at four horizons:
  1 step  (~6 min)
  60 steps (~6 h)
  120 steps (~12 h)
  240 steps (~24 h)

Strategy: direct forecasting — a separate model is trained per horizon with
the target shifted h steps forward. This avoids look-ahead bias: features at
time t are used to predict water_level at time t+h.

WaveGRU operates step-by-step and is only evaluated at horizon 1 (6 min).
Including it at longer horizons would require iterated prediction with
compounding errors, which is outside the current implementation scope.

Output
------
  reports/horizon_metrics.json   — machine-readable results
  reports/horizon_metrics.md     — human-readable Markdown table

Usage
-----
    python -m scripts.evaluate_horizons

Prerequisites
-------------
    python -m scripts.prepare_demo_data
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_demo_data
from src.features.engineering import (
    add_lag_features,
    add_rolling_features,
    add_temporal_covariates,
    add_tidal_harmonics,
    feature_columns,
)
from src.models.metrics import compute_metrics, save_metrics

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
TRAIN_FRAC = 0.75

HORIZONS: Dict[str, int] = {
    "1step_6min": 1,
    "6h": 60,
    "12h": 120,
    "24h": 240,
}


def build_horizon_features(
    df: pd.DataFrame,
    horizon_steps: int,
    target_col: str = "water_level",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build (X, y_h) for direct h-step-ahead forecasting.

    The feature matrix X uses features at time t; the target y_h is the
    water level at time t + horizon_steps. Rows where either is NaN
    (lag warm-up at start, or shifted target missing at end) are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Single-station time series conforming to the Wai schema.
    horizon_steps : int
        Number of 6-minute steps ahead to predict.
    target_col : str

    Returns
    -------
    X : pd.DataFrame of shape (n_valid, n_features)
    y_h : pd.Series of shape (n_valid,)
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["_target_h"] = df[target_col].shift(-horizon_steps)

    df = add_tidal_harmonics(df)
    df = add_temporal_covariates(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    # Preserve original row index — do NOT reset after dropna.
    # The caller splits by `X.index < n_train` where n_train is computed on the
    # pre-dropna DataFrame; resetting here would shift the boundary and leak
    # training-period rows into the test set.
    df = df.dropna()

    feature_cols = feature_columns(df, target_col=target_col)

    return df[feature_cols], df["_target_h"]


def evaluate_persistence_horizon(
    series: pd.Series,
    train_end_idx: int,
    horizon_steps: int,
) -> dict:
    """Evaluate Persistence at horizon h.

    Persistence at horizon h: for each position i in the test window,
    the forecast is the water level observed at position i (the last
    observed value when making a prediction for i+h).

    We align by comparing series[i] to series[i + horizon_steps] for
    all i in the test region where i + horizon_steps is still valid.
    """
    vals = series.values
    n = len(vals)
    # Test window positions that have a valid target h steps ahead
    test_start = train_end_idx
    test_end = n - horizon_steps  # last valid prediction position
    if test_start >= test_end:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                "nse": float("nan"), "corr": float("nan")}
    preds = vals[test_start:test_end]
    actual = vals[test_start + horizon_steps: test_end + horizon_steps]
    return compute_metrics(actual, preds)


def evaluate_sklearn_horizon(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
    model_name: str,
) -> dict:
    """Train and evaluate a sklearn pipeline at a given horizon."""
    if len(train_X) == 0 or len(test_X) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                "nse": float("nan"), "corr": float("nan")}

    if model_name == "harmonic_ridge":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
    elif model_name == "grad_boost":
        pipeline = Pipeline([
            ("model", HistGradientBoostingRegressor(
                max_iter=100, max_depth=5, learning_rate=0.1, random_state=42
            )),
        ])
    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    pipeline.fit(train_X, train_y)
    preds = pipeline.predict(test_X)
    return compute_metrics(test_y.values, preds)


def evaluate_station_horizons(df_station: pd.DataFrame) -> dict:
    """Return horizon metrics for all models and horizons for one station.

    Train/test split (no target leakage)
    ------------------------------------
    For direct h-step forecasting the *target* timestamp is at row i+h while the
    feature row is at i. A clean temporal split therefore must ensure:
      • training rows have target_idx = i + h strictly below the train cutoff,
      • test rows have feature index i at or after the train cutoff.
    Boundary rows whose features fall in the training span but whose targets
    land in the test span are excluded from both sets — they would otherwise
    leak test-period labels into training.
    """
    df_station = df_station.sort_values("timestamp").reset_index(drop=True)
    n = len(df_station)
    n_train = int(n * TRAIN_FRAC)
    series = df_station["water_level"]

    results: dict = {
        "_split": {
            "n_total": int(n),
            "n_train_rows": int(n_train),
            "train_cutoff_ts": str(df_station["timestamp"].iloc[n_train])
            if n_train < n else None,
        }
    }

    for horizon_name, horizon_steps in HORIZONS.items():
        results[horizon_name] = {}

        # Persistence (no feature matrix needed)
        results[horizon_name]["persistence"] = evaluate_persistence_horizon(
            series, n_train, horizon_steps
        )

        # Feature-based models (HarmonicRidge, GradBoost)
        try:
            X, y_h = build_horizon_features(df_station, horizon_steps)
        except Exception as e:
            for m in ("harmonic_ridge", "grad_boost"):
                results[horizon_name][m] = {"error": str(e)}
            continue

        # target_idx is the position of the prediction target (i + h).
        # Excluding boundary rows where target crosses into the test span
        # prevents training rows from seeing test-period labels.
        target_idx = X.index + horizon_steps
        train_mask = target_idx < n_train
        test_mask = X.index >= n_train

        X_train, y_train = X[train_mask], y_h[train_mask]
        X_test, y_test = X[test_mask], y_h[test_mask]

        # Drop any remaining NaN in y_h (defensive — dropna in
        # build_horizon_features already removes shift overshoot at the end).
        valid_train = y_train.notna()
        valid_test = y_test.notna()
        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_test, y_test = X_test[valid_test], y_test[valid_test]

        results[horizon_name]["_split_horizon"] = {
            "n_train_rows": int(len(X_train)),
            "n_test_rows": int(len(X_test)),
            "n_excluded_boundary": int(len(X) - len(X_train) - len(X_test)),
        }

        for model_name in ("harmonic_ridge", "grad_boost"):
            results[horizon_name][model_name] = evaluate_sklearn_horizon(
                X_train, y_train, X_test, y_test, model_name
            )

        # WaveGRU: only at horizon=1; skip others with an honest note
        if horizon_steps == 1:
            from src.models.baseline import WaveGRUModel
            try:
                train_df = df_station.iloc[:n_train]
                test_df = df_station.iloc[n_train:]
                wgru = WaveGRUModel(lookback=24).fit(train_df)
                wgru_metrics = wgru.evaluate(test_df, context_df=train_df)
                results[horizon_name]["wave_gru"] = wgru_metrics
            except Exception as e:
                results[horizon_name]["wave_gru"] = {"error": str(e)}
        else:
            results[horizon_name]["wave_gru"] = {
                "note": "WaveGRU is a 1-step model; not evaluated at this horizon."
            }

    return results


def format_metrics_md(all_results: dict) -> str:
    """Render horizon_metrics as a Markdown table."""
    lines = [
        "# Wai — Multi-Horizon Forecast Evaluation",
        "",
        "Metrics computed on the held-out test split (last 25% by time).",
        "Strategy: **direct forecasting** — a separate model is trained per horizon.",
        "WaveGRU is a 1-step model and is only evaluated at horizon 1 (6 min).",
        "",
    ]

    for station_id, horizons in all_results.items():
        lines.append(f"## Station: {station_id}")
        lines.append("")
        split = horizons.get("_split", {}) if isinstance(horizons, dict) else {}
        if split:
            lines.append(
                f"_train cutoff: {split.get('train_cutoff_ts','?')} "
                f"(n_train={split.get('n_train_rows','?')}, "
                f"n_total={split.get('n_total','?')})._"
            )
            lines.append("")
        lines.append("| Horizon | Model | MAE (m) | RMSE (m) | R² | n_train | n_test |")
        lines.append("|---------|-------|---------|----------|----|---------|--------|")
        for horizon_name, models in horizons.items():
            if horizon_name.startswith("_"):
                continue
            sh = models.get("_split_horizon", {}) if isinstance(models, dict) else {}
            n_tr = sh.get("n_train_rows", "—")
            n_te = sh.get("n_test_rows", "—")
            for model_name, m in models.items():
                if model_name.startswith("_"):
                    continue
                if not isinstance(m, dict) or "mae" not in m:
                    note = m.get("note", m.get("error", "—")) if isinstance(m, dict) else "—"
                    lines.append(f"| {horizon_name} | {model_name} | — | — | {note} | {n_tr} | {n_te} |")
                    continue
                mae = m.get("mae")
                rmse = m.get("rmse")
                r2 = m.get("r2")
                mae_s = f"{mae:.4f}" if mae is not None and not (isinstance(mae, float) and np.isnan(mae)) else "—"
                rmse_s = f"{rmse:.4f}" if rmse is not None and not (isinstance(rmse, float) and np.isnan(rmse)) else "—"
                r2_s = f"{r2:.4f}" if r2 is not None and not (isinstance(r2, float) and np.isnan(r2)) else "—"
                lines.append(f"| {horizon_name} | {model_name} | {mae_s} | {rmse_s} | {r2_s} | {n_tr} | {n_te} |")
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- All metrics are on **synthetic demo data** and cannot be compared to",
        "  published operational benchmarks.",
        "- Direct forecasting trains a separate model for each horizon. This is",
        "  an honest skill assessment but may differ from iterated/recursive approaches.",
        "- Lag features at long horizons (6h, 12h, 24h) reference observations prior",
        "  to the prediction time — no look-ahead bias is introduced.",
        "- **Boundary exclusion:** the split mask uses `target_idx = X.index + h`",
        "  for training and `X.index >= n_train` for test. Rows whose target",
        "  crosses the train/test boundary are dropped from both sets so that no",
        "  training row sees a test-period label.",
        "- Advanced deep learning (LSTM, Transformer) is intentionally excluded to",
        "  keep the repo lightweight and honest.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print("Loading demo data …")
    df = load_demo_data()

    all_results: dict = {}

    for station_id in sorted(df["station_id"].unique()):
        print(f"\n=== {station_id} ===")
        sub = df[df["station_id"] == station_id].copy()
        station_results = evaluate_station_horizons(sub)
        all_results[station_id] = station_results

        for horizon_name, models in station_results.items():
            if horizon_name.startswith("_"):
                continue
            for model_name, m in models.items():
                if model_name.startswith("_"):
                    continue
                if isinstance(m, dict) and "mae" in m:
                    mae = m.get("mae")
                    rmse = m.get("rmse")
                    r2 = m.get("r2")
                    if mae is not None and rmse is not None and r2 is not None:
                        if not (isinstance(mae, float) and np.isnan(mae)):
                            print(
                                f"  {horizon_name:<14} {model_name:<18} "
                                f"MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}"
                            )

    REPORTS_DIR.mkdir(exist_ok=True)

    json_path = REPORTS_DIR / "horizon_metrics.json"
    save_metrics(all_results, json_path)
    print(f"\nSaved: {json_path}")

    md_path = REPORTS_DIR / "horizon_metrics.md"
    md_path.write_text(format_metrics_md(all_results), encoding="utf-8")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
