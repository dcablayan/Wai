"""Event-holdout evaluation for Wai.

Reports observed-vs-predicted episode metrics on the held-out test span:
event precision, recall, F1, peak-height error, peak-time error, and
lead-time error. Sample-level metrics (MAE/RMSE/R^2) are reported alongside
for context.

The threshold is fit on the *training* window (mean + k*std) so that the
test-period episodes the model is graded on are unseen extremes — never the
training extremes. Records carry both the train-fit threshold and the train
cutoff timestamp so the report is auditable.

Usage
-----
    python -m scripts.evaluate_events
    python -m scripts.evaluate_events --threshold-k 2.5

Output
------
    reports/event_metrics.json
    reports/event_metrics.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_demo_data
from src.features.engineering import build_feature_matrix
from src.models.baseline import HarmonicRidgeModel
from src.models.gradient_boost import GradBoostModel
from src.models.metrics import (
    compute_episode_metrics,
    compute_event_metrics,
    compute_metrics,
    save_metrics,
)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
TRAIN_FRAC = 0.75


def _rolling_persistence(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Same 1-step persistence used in train_baseline (rolling, not constant)."""
    test_vals = test["water_level"].values
    last_train = float(train["water_level"].dropna().iloc[-1])
    pred = np.empty(len(test_vals))
    pred[0] = last_train
    if len(test_vals) > 1:
        pred[1:] = test_vals[:-1]
    return pred


def evaluate_station_events(
    df_station: pd.DataFrame,
    station_id: str,
    threshold_k: float = 2.0,
) -> dict:
    """Train HarmonicRidge + GradBoost; evaluate sample- and episode-level metrics
    against the train-fit alert threshold on the held-out test span."""
    df_station = df_station.sort_values("timestamp").reset_index(drop=True)
    n = len(df_station)
    n_train = int(n * TRAIN_FRAC)
    train = df_station.iloc[:n_train]
    test = df_station.iloc[n_train:]

    train_wl = train["water_level"].dropna()
    threshold = float(train_wl.mean() + threshold_k * train_wl.std())

    # Sample-level baseline (rolling 1-step persistence)
    persist_pred = _rolling_persistence(train, test)
    persist_sample = compute_metrics(test["water_level"].values, persist_pred)
    persist_episode = compute_episode_metrics(
        test["water_level"].values,
        persist_pred,
        threshold=threshold,
        timestamps=test["timestamp"].values,
    )

    # Aligned (X, y) for feature-based models on the test span
    harm = HarmonicRidgeModel(alpha=1.0).fit(train)
    gb = GradBoostModel().fit(train)

    _, y_te = build_feature_matrix(test)
    aligned_test = test.iloc[-len(y_te):]
    aligned_actual = y_te.values
    aligned_ts = aligned_test["timestamp"].values

    h_pred = harm.predict_on(test)[-len(y_te):]
    gb_pred = gb.predict_on(test)[-len(y_te):]

    h_sample = compute_metrics(aligned_actual, h_pred)
    gb_sample = compute_metrics(aligned_actual, gb_pred)

    h_event_sample = compute_event_metrics(aligned_actual, h_pred, threshold)
    gb_event_sample = compute_event_metrics(aligned_actual, gb_pred, threshold)

    h_episode = compute_episode_metrics(aligned_actual, h_pred, threshold,
                                        timestamps=aligned_ts)
    gb_episode = compute_episode_metrics(aligned_actual, gb_pred, threshold,
                                         timestamps=aligned_ts)

    return {
        "station_id": station_id,
        "train_cutoff_ts": str(df_station["timestamp"].iloc[n_train]),
        "train_threshold_m": round(threshold, 4),
        "threshold_k": threshold_k,
        "n_train": int(n_train),
        "n_test": int(len(test)),
        "test_start": str(test["timestamp"].iloc[0]),
        "test_end": str(test["timestamp"].iloc[-1]),
        "test_obs_episodes": int(h_episode["n_obs_episodes"]),
        "persistence_rolling": {
            "sample": persist_sample,
            "event_sample": compute_event_metrics(
                test["water_level"].values, persist_pred, threshold
            ),
            "episode": persist_episode,
        },
        "harmonic_ridge": {
            "sample": h_sample,
            "event_sample": h_event_sample,
            "episode": h_episode,
        },
        "grad_boost": {
            "sample": gb_sample,
            "event_sample": gb_event_sample,
            "episode": gb_episode,
        },
    }


def format_results_md(results: dict, threshold_k: float) -> str:
    lines = [
        "# Wai — Event-Holdout Evaluation",
        "",
        f"Threshold = train mean + {threshold_k}σ (computed on the training "
        "window only). Episodes are contiguous test-period runs at or above "
        "this threshold. Predictions are matched to observations by the "
        "largest temporal overlap; positive lead-time error means the "
        "prediction was **late**, negative means **early**.",
        "",
        "All metrics are computed on the held-out test span. The synthetic "
        "demo data places test-period events near days 80 (surge) and 85 "
        "(king tide) so that this report grades genuinely unseen extremes.",
        "",
    ]
    for station_id, res in results.items():
        if station_id.startswith("_"):
            continue
        lines.append(f"## {station_id}")
        lines.append("")
        lines += [
            f"- Train cutoff: {res['train_cutoff_ts']}",
            f"- Train threshold (mean+{threshold_k}σ): {res['train_threshold_m']} m",
            f"- Test window: {res['test_start']} → {res['test_end']} "
            f"(n_test={res['n_test']:,})",
            f"- Observed episodes in test: {res['test_obs_episodes']}",
            "",
            "| Model | MAE (m) | RMSE (m) | Episode P | Episode R | F1 | "
            "Peak-h err (m) | Peak-t err (s) | Lead err (s) |",
            "|-------|---------|----------|-----------|-----------|----|"
            "----------------|----------------|--------------|",
        ]
        for key, label in [
            ("persistence_rolling", "Persistence (rolling)"),
            ("harmonic_ridge", "HarmonicRidge"),
            ("grad_boost", "GradBoost"),
        ]:
            m = res.get(key, {})
            s = m.get("sample", {})
            ep = m.get("episode", {})
            lines.append(
                f"| {label} | "
                f"{s.get('mae', float('nan')):.4f} | "
                f"{s.get('rmse', float('nan')):.4f} | "
                f"{ep.get('episode_precision', float('nan'))} | "
                f"{ep.get('episode_recall', float('nan'))} | "
                f"{ep.get('episode_f1', float('nan'))} | "
                f"{ep.get('peak_height_error_m', float('nan'))} | "
                f"{ep.get('peak_time_error_s', float('nan'))} | "
                f"{ep.get('lead_time_error_s', float('nan'))} |"
            )
        lines.append("")
    return "\n".join(lines)


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Event-holdout evaluation with episode-level metrics."
    )
    parser.add_argument("--threshold-k", type=float, default=2.0,
                        help="Threshold in train-fit standard deviations (default 2.0).")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = _parse_args(argv)
    print("Loading demo data …")
    df = load_demo_data()

    results: dict = {
        "_meta": {
            "train_frac": TRAIN_FRAC,
            "threshold_k": args.threshold_k,
        }
    }

    for station_id in sorted(df["station_id"].unique()):
        print(f"\n=== {station_id} ===")
        sub = df[df["station_id"] == station_id].copy()
        results[station_id] = evaluate_station_events(
            sub, station_id=station_id, threshold_k=args.threshold_k,
        )
        ep = results[station_id]["harmonic_ridge"]["episode"]
        print(f"  HarmonicRidge episodes: obs={ep['n_obs_episodes']} "
              f"pred={ep['n_pred_episodes']} matched={ep['n_matched']} "
              f"P={ep['episode_precision']} R={ep['episode_recall']}")
        print(f"    peak-height err={ep['peak_height_error_m']} m  "
              f"peak-time err={ep['peak_time_error_s']} s  "
              f"lead err={ep['lead_time_error_s']} s")

    REPORTS_DIR.mkdir(exist_ok=True)
    json_path = REPORTS_DIR / "event_metrics.json"
    save_metrics(results, json_path)
    print(f"\nSaved: {json_path}")

    md_path = REPORTS_DIR / "event_metrics.md"
    md_path.write_text(format_results_md(results, args.threshold_k), encoding="utf-8")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
