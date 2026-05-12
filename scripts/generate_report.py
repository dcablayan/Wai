"""Generate HTML station reports from demo data and saved model metrics.

Usage
-----
    python -m scripts.generate_report

Prerequisites
-------------
    python -m scripts.prepare_demo_data
    python -m scripts.train_baseline   (optional — report works without metrics)

Output
------
    reports/report_<station_id>.html  for each demo station
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_demo_data
from src.reporting.report import generate_report

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"


def main() -> None:
    print("Loading demo data …")
    df = load_demo_data()

    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            all_metrics = json.load(f)
        print(f"Loaded metrics from {METRICS_PATH}")
    else:
        all_metrics = {}
        print(
            f"[WARN] {METRICS_PATH} not found. "
            "Run python -m scripts.train_baseline first for metric tables."
        )

    REPORTS_DIR.mkdir(exist_ok=True)

    for station_id in sorted(df["station_id"].unique()):
        sub = df[df["station_id"] == station_id]
        station_metrics = all_metrics.get(station_id, {})
        out_path = REPORTS_DIR / f"report_{station_id}.html"
        generate_report(sub, station_metrics, out_path)
        print(f"  Written: {out_path}")

    print("\nDone. Open any report_*.html in your browser.")


if __name__ == "__main__":
    main()
