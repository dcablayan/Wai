"""Enforce per-file coverage floors for high-risk product boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path


CRITICAL_FLOORS = {
    "src/artifacts.py": 80.0,
    "src/data/noaa.py": 60.0,
    "src/evaluation/trajectory_search.py": 75.0,
    "src/evidence.py": 85.0,
    "Hohonu-1/VAR_prediction.py": 40.0,
    "Hohonu-1/api_server.py": 75.0,
}


def check_critical_coverage(report_paths: list[str | Path]) -> dict[str, float]:
    files = {}
    for report_path in report_paths:
        report = json.loads(Path(report_path).read_text())
        files.update(report.get("files", {}))
    measured: dict[str, float] = {}
    failures: list[str] = []
    for required, floor in CRITICAL_FLOORS.items():
        matches = [
            record
            for filename, record in files.items()
            if filename.replace("\\", "/").endswith(required)
        ]
        if len(matches) != 1:
            failures.append(f"{required}: missing from coverage report")
            continue
        percent = float(matches[0]["summary"]["percent_covered"])
        measured[required] = percent
        if percent + 1e-9 < floor:
            failures.append(f"{required}: {percent:.2f}% < required {floor:.2f}%")
    if failures:
        raise RuntimeError("Critical coverage gate failed:\n- " + "\n- ".join(failures))
    return measured


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit(
            "usage: python -m scripts.check_critical_coverage COVERAGE_JSON [...]"
        )
    measured = check_critical_coverage(args)
    for filename, percent in measured.items():
        print(f"{filename}: {percent:.2f}%")


if __name__ == "__main__":
    main()
