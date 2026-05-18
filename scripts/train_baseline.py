"""Train and evaluate baseline forecast models on demo data.

Models evaluated
----------------
1. persistence_rolling   — rolling 1-step persistence: predict[t] = observed[t-1].
                           This is the correct 1-step naive baseline for a 6-min
                           forecast (the first test step uses the last training value).
2. persistence_constant  — constant persistence: every test step is predicted as the
                           last training observation. Included for reference; it is a
                           strictly harder baseline and should not be used as the
                           primary comparator for 1-step models.
3. HarmonicRidgeModel    — 8-constituent tidal harmonics + temporal covariates
                           + lags + rolling features, fitted with Ridge.
4. WaveGRUModel          — smoothing heuristic with attention-like weighting
                           (not a real GRU or deep-learning model).
5. GradBoostModel        — HistGradientBoostingRegressor over the same feature
                           matrix as HarmonicRidgeModel (non-linear baseline).

A 75/25 temporal train/test split is used per station to avoid data leakage.
Metrics (MAE, RMSE, R², NSE, correlation) are saved to reports/model_metrics.json.

Usage
-----
    python -m scripts.train_baseline

Prerequisites
-------------
    python -m scripts.prepare_demo_data
"""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.loader import load_demo_data
from src.data.validation import validate
from src.features.engineering import (
    add_lag_features,
    add_rolling_features,
    add_temporal_covariates,
    add_tidal_harmonics,
    feature_columns,
)
from src.models.baseline import HarmonicRidgeModel, PersistenceModel, WaveGRUModel
from src.models.branding import DISPLAY_BY_KEY
from src.models.gradient_boost import GradBoostModel
from src.models.metrics import (
    block_bootstrap_ci,
    bootstrap_ci,
    compute_metrics,
    save_metrics,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_FRAC = 0.75
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
MODELS_DIR  = Path(__file__).resolve().parent.parent / "reports" / "models"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
ABLATION_PATH = REPORTS_DIR / "ablation_metrics.json"
META_PATH    = REPORTS_DIR / "run_metadata.json"


def _git_sha() -> str:
    """Return the current HEAD commit SHA (first 12 chars), or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _data_hash(df: pd.DataFrame) -> str:
    """MD5 hash of the data CSV bytes as a provenance fingerprint."""
    csv_bytes = df.to_csv(index=False).encode()
    return hashlib.md5(csv_bytes).hexdigest()[:16]


def save_model_artifact(model, station_id: str, model_name: str) -> None:
    """Pickle a fitted model to reports/models/{station_id}_{model_name}.pkl."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{station_id}_{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)


def save_run_metadata(meta: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def rolling_persistence_1step(train_series: "pd.Series", test_series: "pd.Series") -> np.ndarray:
    """Rolling 1-step persistence: pred[0]=train[-1], pred[t]=test[t-1] for t>=1.

    This is the correct naive baseline for 1-step-ahead (6 min) forecasting.
    At each position the only information used is the immediately preceding
    observation — no look-ahead.
    """
    test_vals = test_series.values
    last_train = float(train_series.dropna().iloc[-1])
    preds = np.empty(len(test_vals))
    preds[0] = last_train
    if len(test_vals) > 1:
        preds[1:] = test_vals[:-1]
    return preds


def _build_ablation_X(df: pd.DataFrame, config: str, target_col: str = "water_level"):
    """Build feature matrix for a named ablation configuration.

    Configurations
    --------------
    harmonics_only     — 8-constituent sin/cos + temporal covariates; no lags or rolling
    lags_only          — lag features only (no harmonics, no rolling)
    rolling_only       — rolling mean/std only (no harmonics, no lags)
    harmonics_lags     — harmonics + temporal + lags; no rolling
    full               — all features (harmonics + temporal + lags + rolling)
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    if config == "harmonics_only":
        df = add_tidal_harmonics(df)
        df = add_temporal_covariates(df)
    elif config == "lags_only":
        df = add_lag_features(df)
    elif config == "rolling_only":
        df = add_rolling_features(df)
    elif config == "harmonics_lags":
        df = add_tidal_harmonics(df)
        df = add_temporal_covariates(df)
        df = add_lag_features(df)
    elif config == "full":
        df = add_tidal_harmonics(df)
        df = add_temporal_covariates(df)
        df = add_lag_features(df)
        df = add_rolling_features(df)
    else:
        raise ValueError(f"Unknown ablation config: {config!r}")

    df = df.dropna().reset_index(drop=True)
    feature_cols = feature_columns(df, target_col=target_col)
    return df[feature_cols], df[target_col]


def run_ablation(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Train HarmonicRidge under five feature ablation configurations.

    Returns a dict keyed by config name, each with metrics dict plus
    95% bootstrap CI on MAE.
    """
    configs = ["harmonics_only", "lags_only", "rolling_only", "harmonics_lags", "full"]
    results = {}
    for cfg in configs:
        try:
            X_tr, y_tr = _build_ablation_X(train, cfg)
            X_te, y_te = _build_ablation_X(test, cfg)
            # Align to common feature columns
            common = [c for c in X_tr.columns if c in X_te.columns]
            X_tr, X_te = X_tr[common], X_te[common]
            pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_te)
            m = compute_metrics(y_te.values, preds)
            block_mae = block_bootstrap_ci(y_te.values, preds, metric="mae")
            block_rmse = block_bootstrap_ci(y_te.values, preds, metric="rmse")
            m["mae_block_ci_95"] = block_mae
            m["rmse_block_ci_95"] = block_rmse
            m["mae_iid_ci_95"] = bootstrap_ci(y_te.values, preds, metric="mae")
            m["rmse_iid_ci_95"] = bootstrap_ci(y_te.values, preds, metric="rmse")
            m["mae_ci_95"] = (block_mae["lower"], block_mae["upper"])
            m["rmse_ci_95"] = (block_rmse["lower"], block_rmse["upper"])
            m["n_features"] = len(common)
            results[cfg] = m
        except Exception as e:
            results[cfg] = {"error": str(e)}
    return results


def train_station(df_station, station_id: str = ""):
    """Train and evaluate all five model variants for one station."""
    df_station = df_station.sort_values("timestamp").reset_index(drop=True)
    n_train = int(len(df_station) * TRAIN_FRAC)
    train = df_station.iloc[:n_train]
    test = df_station.iloc[n_train:]

    # --- Rolling 1-step persistence (primary naive baseline) ---
    rolling_pred = rolling_persistence_1step(train["water_level"], test["water_level"])
    rolling_metrics = compute_metrics(test["water_level"].values, rolling_pred)

    # --- Constant persistence (reference; harder baseline) ---
    const_persist = PersistenceModel().fit(train["water_level"])
    const_pred = const_persist.predict(len(test))
    const_metrics = compute_metrics(test["water_level"].values, const_pred)

    # --- Harmonic Ridge ---
    harmonic = HarmonicRidgeModel(alpha=1.0).fit(train)
    harmonic_metrics = harmonic.evaluate(test)
    if station_id:
        save_model_artifact(harmonic, station_id, "harmonic_ridge")

    # --- WaveGRU ---
    wavegru = WaveGRUModel(lookback=24).fit(train)
    wavegru_metrics = wavegru.evaluate(test, context_df=train)

    # --- Gradient Boosting ---
    gradboost = GradBoostModel().fit(train)
    gradboost_metrics = gradboost.evaluate(test)
    if station_id:
        save_model_artifact(gradboost, station_id, "grad_boost")

    # Bootstrap 95% CI on MAE/RMSE for the primary supervised models.
    # Headline interval = moving/circular block bootstrap (handles
    # residual autocorrelation honestly). IID bootstrap is retained as a
    # reference baseline (always tighter — useful as a "lower bound").
    from src.features.engineering import build_feature_matrix
    _, y_test_aligned = build_feature_matrix(test)
    h_preds = harmonic.predict_on(test)
    h_preds = h_preds[-len(y_test_aligned):]
    harmonic_metrics["mae_block_ci_95"] = block_bootstrap_ci(
        y_test_aligned.values, h_preds, metric="mae"
    )
    harmonic_metrics["rmse_block_ci_95"] = block_bootstrap_ci(
        y_test_aligned.values, h_preds, metric="rmse"
    )
    # IID bootstrap retained for backwards-compatible reporting + as a
    # transparently-too-tight reference.
    harmonic_metrics["mae_iid_ci_95"] = bootstrap_ci(y_test_aligned.values, h_preds)
    harmonic_metrics["rmse_iid_ci_95"] = bootstrap_ci(y_test_aligned.values, h_preds, metric="rmse")
    # Headline CI for legacy fields uses the block interval.
    harmonic_metrics["mae_ci_95"] = (
        harmonic_metrics["mae_block_ci_95"]["lower"],
        harmonic_metrics["mae_block_ci_95"]["upper"],
    )
    harmonic_metrics["rmse_ci_95"] = (
        harmonic_metrics["rmse_block_ci_95"]["lower"],
        harmonic_metrics["rmse_block_ci_95"]["upper"],
    )

    gb_preds = gradboost.predict_on(test)
    gb_preds = gb_preds[-len(y_test_aligned):]
    gradboost_metrics["mae_block_ci_95"] = block_bootstrap_ci(
        y_test_aligned.values, gb_preds, metric="mae"
    )
    gradboost_metrics["rmse_block_ci_95"] = block_bootstrap_ci(
        y_test_aligned.values, gb_preds, metric="rmse"
    )
    gradboost_metrics["mae_iid_ci_95"] = bootstrap_ci(y_test_aligned.values, gb_preds)
    gradboost_metrics["rmse_iid_ci_95"] = bootstrap_ci(y_test_aligned.values, gb_preds, metric="rmse")
    gradboost_metrics["mae_ci_95"] = (
        gradboost_metrics["mae_block_ci_95"]["lower"],
        gradboost_metrics["mae_block_ci_95"]["upper"],
    )
    gradboost_metrics["rmse_ci_95"] = (
        gradboost_metrics["rmse_block_ci_95"]["lower"],
        gradboost_metrics["rmse_block_ci_95"]["upper"],
    )

    return {
        "persistence": rolling_metrics,           # rolling 1-step (primary)
        "persistence_constant": const_metrics,    # constant holdout (reference)
        "harmonic_ridge": harmonic_metrics,
        "wave_gru": wavegru_metrics,
        "grad_boost": gradboost_metrics,
        "ablation": run_ablation(train, test),
        "split": {
            "train_obs": int(n_train),
            "test_obs": int(len(test)),
            "train_start": str(train["timestamp"].iloc[0]),
            "train_end": str(train["timestamp"].iloc[-1]),
            "test_start": str(test["timestamp"].iloc[0]),
            "test_end": str(test["timestamp"].iloc[-1]),
        },
    }


def main() -> None:
    print("Loading demo data …")
    df = load_demo_data()

    print("Validating …")
    report = validate(df)
    if report.warnings:
        for w in report.warnings:
            print(f"  [WARN] {w}")
    else:
        print("  No validation issues.")

    all_results = {}
    all_ablation = {}
    for station_id in sorted(df["station_id"].unique()):
        print(f"\n=== {station_id} ===")
        sub = df[df["station_id"] == station_id].copy()
        results = train_station(sub, station_id=station_id)
        all_results[station_id] = results
        all_ablation[station_id] = results.pop("ablation", {})

        for model_name, m in results.items():
            if not isinstance(m, dict) or "mae" not in m:
                continue
            label = DISPLAY_BY_KEY.get(model_name, model_name)
            ci = m.get("mae_ci_95", ("", ""))
            ci_str = f"  95%CI=[{ci[0]:.4f}, {ci[1]:.4f}]" if ci and ci[0] != "" else ""
            print(
                f"  {label:<42} MAE={m['mae']:.4f}  "
                f"RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}{ci_str}"
            )

        ablation = all_ablation.get(station_id, {})
        if ablation:
            print(f"\n  Ablation study (HarmonicRidge feature subsets):")
            for cfg, m in ablation.items():
                if not isinstance(m, dict) or "mae" not in m:
                    continue
                n_feat = m.get("n_features", "?")
                print(
                    f"    {cfg:<20} n_feat={n_feat:<3}  "
                    f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}"
                )

    save_metrics(all_results, METRICS_PATH)
    print(f"\nMetrics saved to {METRICS_PATH}")

    save_metrics(all_ablation, ABLATION_PATH)
    print(f"Ablation metrics saved to {ABLATION_PATH}")

    meta = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "data_hash": _data_hash(df),
        "train_frac": TRAIN_FRAC,
        "stations": sorted(df["station_id"].unique().tolist()),
        "model_version": "0.1.0",
    }
    save_run_metadata(meta)
    print(f"Run metadata saved to {META_PATH}")
    print(f"  git_sha={meta['git_sha']}  data_hash={meta['data_hash']}")


if __name__ == "__main__":
    main()
