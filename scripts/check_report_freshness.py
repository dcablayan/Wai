"""Fail when committed evidence was produced from different source inputs."""

from __future__ import annotations

import json
from pathlib import Path

from src.evidence import evidence_freshness


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_METADATA_PATH = REPO_ROOT / "reports" / "run_metadata.json"
SUMMARY_PATH = REPO_ROOT / "reports" / "summary.json"


def check_report_freshness() -> dict:
    if not RUN_METADATA_PATH.exists():
        raise RuntimeError(f"Missing evidence metadata: {RUN_METADATA_PATH}")
    metadata = json.loads(RUN_METADATA_PATH.read_text())
    freshness = evidence_freshness(metadata, REPO_ROOT)
    if not freshness["fresh_at_verification"]:
        raise RuntimeError(
            "Generated reports are stale: recorded source fingerprint "
            f"{freshness['recorded_source_fingerprint']!r} does not match "
            f"{freshness['current_source_fingerprint']!r}. Run `make demo`."
        )
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())
        recorded = summary.get("staleness", {}).get("recorded_source_fingerprint")
        if recorded != freshness["recorded_source_fingerprint"]:
            raise RuntimeError("reports/summary.json and run_metadata.json disagree")
    return freshness


def main() -> None:
    freshness = check_report_freshness()
    print(
        "Evidence is fresh for source fingerprint "
        f"{freshness['current_source_fingerprint']}."
    )


if __name__ == "__main__":
    main()
