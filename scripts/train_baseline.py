"""Train and evaluate baseline forecast models on demo data.

Models evaluated
----------------
1. PersistenceModel   — naive last-value baseline
2. HarmonicRidgeModel — 8-constituent tidal harmonics + temporal covariates
                        + lags + rolling features, fitted with Ridge
3. WaveGRUModel       — bidirectional double-EMA with attention (pure Python,
                        adapted from dcablayan/tideformer WaveGRUPrototype)
4. GradBoostModel     — HistGradientBoostingRegressor over the same feature
                        matrix as HarmonicRidgeModel (non-linear baseline)

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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_demo_data
from src.data.validation import validate
from src.models.baseline import HarmonicRidgeModel, PersistenceModel, WaveGRUModel
from src.models.gradient_boost import GradBoostModel
from src.models.metrics import compute_metrics, save_metrics

TRAIN_FRAC = 0.75
METRICS_PATH = Path(__file__).resolve().parent.parent / "reports" / "model_metrics.json"


def train_station(df_station):
    """Train and evaluate all four models for one station."""
    df_station = df_station.sort_values("timestamp").reset_index(drop=True)
    n_train = int(len(df_station) * TRAIN_FRAC)
    train = df_station.iloc[:n_train]
    test = df_station.iloc[n_train:]

    # --- Persistence ---
    persist = PersistenceModel().fit(train["water_level"])
    persist_pred = persist.predict(len(test))
    persist_metrics = compute_metrics(test["water_level"].values, persist_pred)

    # --- Harmonic Ridge ---
    harmonic = HarmonicRidgeModel(alpha=1.0).fit(train)
    harmonic_metrics = harmonic.evaluate(test)

    # --- WaveGRU ---
    wavegru = WaveGRUModel(lookback=24).fit(train)
    wavegru_metrics = wavegru.evaluate(test, context_df=train)

    # --- Gradient Boosting ---
    gradboost = GradBoostModel().fit(train)
    gradboost_metrics = gradboost.evaluate(test)

    return {
        "persistence": persist_metrics,
        "harmonic_ridge": harmonic_metrics,
        "wave_gru": wavegru_metrics,
        "grad_boost": gradboost_metrics,
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
    for station_id in sorted(df["station_id"].unique()):
        print(f"\n=== {station_id} ===")
        sub = df[df["station_id"] == station_id].copy()
        results = train_station(sub)
        all_results[station_id] = results

        for model_name, m in results.items():
            if not isinstance(m, dict) or "mae" not in m:
                continue
            print(
                f"  {model_name:<20} MAE={m['mae']:.4f}  "
                f"RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}"
            )

    save_metrics(all_results, METRICS_PATH)
    print(f"\nMetrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
