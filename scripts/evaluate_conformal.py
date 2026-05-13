"""Generate conformal interval coverage reports for demo stations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_demo_data
from src.models.baseline import HarmonicRidgeModel
from src.models.conformal import ConformalIntervals
from src.models.gradient_boost import GradBoostModel
from src.models.metrics import save_metrics

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
TRAIN_FRAC = 0.75
CAL_FRAC_OF_TRAIN = 0.15
NOMINAL_COVERAGE = 0.90


def _split_station(
    df_station: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    cal_frac_of_train: float = CAL_FRAC_OF_TRAIN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df_station.sort_values("timestamp").reset_index(drop=True)
    n_train = int(len(ordered) * train_frac)
    n_cal = max(1, int(n_train * cal_frac_of_train))
    train_fit = ordered.iloc[:n_train - n_cal].copy()
    train_cal = ordered.iloc[n_train - n_cal:n_train].copy()
    test = ordered.iloc[n_train:].copy()
    return train_fit, train_cal, test


def _model_report(
    model_name: str,
    model,
    train_fit: pd.DataFrame,
    train_cal: pd.DataFrame,
    test: pd.DataFrame,
    event_threshold: float,
    nominal_coverage: float,
) -> dict:
    fitted = model.fit(train_fit)
    cal = fitted.predict_aligned(train_cal)
    ci = ConformalIntervals(coverage=nominal_coverage).calibrate(
        cal["actual"].to_numpy(dtype=float),
        cal["prediction"].to_numpy(dtype=float),
    )
    test_aligned = fitted.predict_aligned(test)
    actual = test_aligned["actual"].to_numpy(dtype=float)
    pred = test_aligned["prediction"].to_numpy(dtype=float)
    coverage = ci.stratified_coverage(actual, pred, event_threshold=event_threshold)
    lower, upper = ci.intervals(pred)
    return {
        "model": model_name,
        "nominal_coverage": float(nominal_coverage),
        "empirical_coverage": coverage["coverage_overall"],
        "event_coverage": coverage.get("coverage_event"),
        "non_event_coverage": coverage.get("coverage_non_event"),
        "qhat": float(ci.qhat),
        "n_cal": int(ci.n_cal),
        "mean_interval_width": float(np.mean(upper - lower)),
        "n_test": int(len(actual)),
        "n_event_samples": coverage.get("n_event_samples"),
        "n_non_event_samples": coverage.get("n_non_event_samples"),
        "event_threshold_m": float(event_threshold),
    }


def evaluate_station_conformal(
    df_station: pd.DataFrame,
    station_id: str,
    nominal_coverage: float = NOMINAL_COVERAGE,
) -> dict:
    train_fit, train_cal, test = _split_station(df_station)
    train_ref = pd.concat([train_fit, train_cal], ignore_index=True)
    train_wl = train_ref["water_level"].dropna()
    event_threshold = float(train_wl.mean() + 2.0 * train_wl.std())
    models = {
        "harmonic_ridge": HarmonicRidgeModel(alpha=1.0),
        "grad_boost": GradBoostModel(),
    }
    reports = {
        name: _model_report(
            name, model, train_fit, train_cal, test, event_threshold, nominal_coverage
        )
        for name, model in models.items()
    }
    return {
        "station_id": station_id,
        "split": {
            "train_fit_start": str(train_fit["timestamp"].iloc[0]),
            "train_fit_end": str(train_fit["timestamp"].iloc[-1]),
            "cal_start": str(train_cal["timestamp"].iloc[0]),
            "cal_end": str(train_cal["timestamp"].iloc[-1]),
            "test_start": str(test["timestamp"].iloc[0]),
            "test_end": str(test["timestamp"].iloc[-1]),
            "n_train_fit": int(len(train_fit)),
            "n_cal_window": int(len(train_cal)),
            "n_test_window": int(len(test)),
        },
        "threshold_source": "train_fit_plus_cal_mean_plus_2std",
        "event_threshold_m": event_threshold,
        "models": reports,
    }


def format_markdown(results: dict) -> str:
    lines = [
        "# Wai Conformal Coverage Report",
        "",
        "Synthetic demo data only. Calibration is the last 15% of the training window; coverage is measured on the future test split.",
        "",
        "| Station | Model | Nominal | Empirical | Event | Non-event | qhat | n_cal | Mean width |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for station, rec in results.items():
        if station.startswith("_"):
            continue
        for model_name, m in rec.get("models", {}).items():
            def fmt(v: object) -> str:
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    return "-"
                return "-" if np.isnan(x) else f"{x:.4f}"

            lines.append(
                f"| {station} | {model_name} | {fmt(m['nominal_coverage'])} | "
                f"{fmt(m['empirical_coverage'])} | {fmt(m['event_coverage'])} | "
                f"{fmt(m['non_event_coverage'])} | {fmt(m['qhat'])} | "
                f"{m['n_cal']} | {fmt(m['mean_interval_width'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate split-conformal coverage.")
    parser.add_argument("--coverage", type=float, default=NOMINAL_COVERAGE)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    df = load_demo_data()
    results = {
        "_meta": {
            "data_source": "DEMO_SYNTHETIC",
            "nominal_coverage": float(args.coverage),
            "train_frac": TRAIN_FRAC,
            "cal_frac_of_train": CAL_FRAC_OF_TRAIN,
        }
    }
    for station_id in sorted(df["station_id"].unique()):
        sub = df[df["station_id"] == station_id]
        results[station_id] = evaluate_station_conformal(
            sub, station_id=station_id, nominal_coverage=args.coverage
        )

    REPORTS_DIR.mkdir(exist_ok=True)
    json_path = REPORTS_DIR / "conformal_metrics.json"
    md_path = REPORTS_DIR / "conformal_metrics.md"
    save_metrics(results, json_path)
    md_path.write_text(format_markdown(results), encoding="utf-8")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
