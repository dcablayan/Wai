"""Rolling-origin temporal evaluation on synthetic demo stations.

Each fold trains only on observations before the test window and evaluates on
the immediately following time block. This complements the single 75/25 split
with multiple forward-in-time checks and explicit fold metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_demo_data
from src.models.baseline import HarmonicRidgeModel
from src.models.gradient_boost import GradBoostModel
from src.models.metrics import compute_metrics, save_metrics, skill_score
from scripts.evaluate_noaa_public import rolling_persistence_1step

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
N_FOLDS = 3
MIN_TRAIN_FRAC = 0.50
TEST_FRAC = 0.10


def make_rolling_origin_folds(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    min_train_frac: float = MIN_TRAIN_FRAC,
    test_frac: float = TEST_FRAC,
) -> list[tuple[pd.DataFrame, pd.DataFrame, dict]]:
    """Return expanding-window temporal folds with no train/test overlap."""
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    n = len(ordered)
    min_train = int(n * min_train_frac)
    test_size = max(50, int(n * test_frac))
    if min_train <= 0 or min_train + test_size > n:
        return []

    max_start = n - test_size
    starts = np.linspace(min_train, max_start, n_folds, dtype=int)
    folds = []
    seen_starts: set[int] = set()
    for i, test_start_idx in enumerate(starts, start=1):
        if int(test_start_idx) in seen_starts:
            continue
        seen_starts.add(int(test_start_idx))
        train = ordered.iloc[:test_start_idx].copy()
        test = ordered.iloc[test_start_idx:test_start_idx + test_size].copy()
        if train.empty or test.empty:
            continue
        meta = {
            "fold": int(i),
            "train_start": str(train["timestamp"].iloc[0]),
            "train_end": str(train["timestamp"].iloc[-1]),
            "test_start": str(test["timestamp"].iloc[0]),
            "test_end": str(test["timestamp"].iloc[-1]),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        }
        folds.append((train, test, meta))
    return folds


def _with_skill(actual: np.ndarray, forecast: np.ndarray, persistence: np.ndarray) -> dict:
    m = compute_metrics(actual, forecast)
    p = compute_metrics(actual, persistence)
    m["mae_skill_vs_rolling_persistence"] = skill_score(m["mae"], p["mae"])
    m["rmse_skill_vs_rolling_persistence"] = skill_score(m["rmse"], p["rmse"])
    return m


def evaluate_fold(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Evaluate persistence, HarmonicRidge, and GradBoost for one fold."""
    test = test.sort_values("timestamp").reset_index(drop=True)
    actual = test["water_level"].to_numpy(dtype=float)
    persistence = rolling_persistence_1step(train["water_level"], test["water_level"])

    out = {
        "rolling_persistence": _with_skill(actual, persistence, persistence),
    }

    harmonic = HarmonicRidgeModel(alpha=1.0).fit(train)
    h_aligned = harmonic.predict_aligned(test)
    h_rows = h_aligned["_source_row"].to_numpy(dtype=int)
    out["harmonic_ridge"] = _with_skill(
        h_aligned["actual"].to_numpy(dtype=float),
        h_aligned["prediction"].to_numpy(dtype=float),
        persistence[h_rows],
    )

    grad = GradBoostModel().fit(train)
    g_aligned = grad.predict_aligned(test)
    g_rows = g_aligned["_source_row"].to_numpy(dtype=int)
    out["grad_boost"] = _with_skill(
        g_aligned["actual"].to_numpy(dtype=float),
        g_aligned["prediction"].to_numpy(dtype=float),
        persistence[g_rows],
    )
    return out


def evaluate_station_rolling_origin(
    df_station: pd.DataFrame,
    station_id: str,
    n_folds: int = N_FOLDS,
) -> dict:
    folds = make_rolling_origin_folds(df_station, n_folds=n_folds)
    results = []
    for train, test, meta in folds:
        rec = {**meta, **evaluate_fold(train, test)}
        results.append(rec)
    return {"station_id": station_id, "n_folds": len(results), "folds": results}


def format_markdown(results: dict) -> str:
    lines = [
        "# Wai Rolling-Origin Evaluation",
        "",
        "Synthetic demo data only. Each fold trains on the past and tests on the next time block.",
        "",
    ]
    for station, rec in results.items():
        if station.startswith("_"):
            continue
        lines.append(f"## {station}")
        lines.append("")
        lines.append("| Fold | Train End | Test Start | n_train | n_test | Persistence MAE | HarmonicRidge MAE | GradBoost MAE |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for fold in rec.get("folds", []):
            lines.append(
                "| {fold} | {train_end} | {test_start} | {n_train} | {n_test} | {p:.4f} | {h:.4f} | {g:.4f} |".format(
                    fold=fold["fold"],
                    train_end=fold["train_end"],
                    test_start=fold["test_start"],
                    n_train=fold["n_train"],
                    n_test=fold["n_test"],
                    p=fold["rolling_persistence"]["mae"],
                    h=fold["harmonic_ridge"]["mae"],
                    g=fold["grad_boost"]["mae"],
                )
            )
        lines.append("")
    return "\n".join(lines)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling-origin evaluation.")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    df = load_demo_data()
    results = {
        "_meta": {
            "data_source": "DEMO_SYNTHETIC",
            "n_folds_requested": int(args.folds),
            "min_train_frac": MIN_TRAIN_FRAC,
            "test_frac": TEST_FRAC,
        }
    }
    for station_id in sorted(df["station_id"].unique()):
        sub = df[df["station_id"] == station_id]
        results[station_id] = evaluate_station_rolling_origin(
            sub, station_id=station_id, n_folds=args.folds
        )

    REPORTS_DIR.mkdir(exist_ok=True)
    json_path = REPORTS_DIR / "rolling_origin_metrics.json"
    md_path = REPORTS_DIR / "rolling_origin_metrics.md"
    save_metrics(results, json_path)
    md_path.write_text(format_markdown(results), encoding="utf-8")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
