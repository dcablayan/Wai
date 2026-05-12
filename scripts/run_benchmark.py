"""Benchmark all prototype models on tidecast and demo data.

Runs TinyTide, HarmonicNet, WaveGRU, and SurgeNet across every station
CSV found in data/demo/tidecast/ (10 real Hohonu station tidal predictions
from dcablayan/tideformer) and writes a markdown report.

Falls back to demo synthetic data if no tidecast files are found.

Usage
-----
    python -m scripts.run_benchmark

Output
------
    reports/benchmark_results.md
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.windowing import load_tidecast_series, make_windows, temporal_split
from src.models.prototypes import (
    HarmonicNetPrototype,
    SurgeNetPrototype,
    TinyTidePrototype,
    WaveGRUPrototype,
    rmse,
)

LOOKBACK = 24
MAX_WINDOWS = 2000
TIDECAST_DIR = Path(__file__).resolve().parent.parent / "data" / "demo" / "tidecast"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def benchmark_station(path: Path) -> dict[str, float]:
    times, series = load_tidecast_series(path)
    windows = make_windows(series, lookback=LOOKBACK, max_samples=MAX_WINDOWS, times=times)
    if len(windows) < 10:
        return {}
    train, val, test = temporal_split(windows)
    fit_windows = train + val

    models = {
        "TinyTide": TinyTidePrototype(lookback=LOOKBACK, lr=0.0005, epochs=2),
        "HarmonicNet": HarmonicNetPrototype(lookback=LOOKBACK),
        "WaveGRU": WaveGRUPrototype(lookback=LOOKBACK),
        "SurgeNet": SurgeNetPrototype(lookback=LOOKBACK),
    }
    return {name: model.fit(fit_windows).evaluate(test) for name, model in models.items()}


def main() -> None:
    station_paths = sorted(TIDECAST_DIR.glob("*.csv")) if TIDECAST_DIR.exists() else []

    if not station_paths:
        print(
            "[WARN] No tidecast CSVs found in data/demo/tidecast/.\n"
            "       Download them with:\n"
            "         python -m scripts.download_tidecast\n"
            "       or copy hohonu-*_tidecast.csv files from dcablayan/tideformer."
        )
        return

    MODEL_NAMES = ["TinyTide", "HarmonicNet", "WaveGRU", "SurgeNet"]
    per_station: dict[str, dict[str, float]] = {}
    overall: dict[str, list[float]] = {m: [] for m in MODEL_NAMES}

    for path in station_paths:
        print(f"  Benchmarking {path.name} …")
        results = benchmark_station(path)
        if not results:
            print(f"    [SKIP] insufficient data")
            continue
        per_station[path.stem] = results
        for name in MODEL_NAMES:
            if name in results:
                overall[name].append(results[name])

    if not per_station:
        print("No stations produced results.")
        return

    lines = [
        "# Wai Benchmark Results",
        "",
        "Models from `src/models/prototypes.py` (ported from dcablayan/tideformer).",
        "Evaluated on tidecast tidal-prediction data (NOAA-derived, Hawaiian stations).",
        f"Lookback: {LOOKBACK} steps (144 min at 6-min cadence) · Max windows: {MAX_WINDOWS}",
        "",
        "| Station | TinyTide RMSE | HarmonicNet RMSE | WaveGRU RMSE | SurgeNet RMSE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for station, scores in per_station.items():
        row = f"| {station} "
        for m in MODEL_NAMES:
            row += f"| {scores.get(m, float('nan')):.3f} "
        row += "|"
        lines.append(row)

    lines += ["", "**Averages**", "| Model | Mean RMSE |", "| --- | --- |"]
    for name in MODEL_NAMES:
        vals = overall[name]
        avg = sum(vals) / len(vals) if vals else float("nan")
        lines.append(f"| {name} | {avg:.3f} |")

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / "benchmark_results.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
