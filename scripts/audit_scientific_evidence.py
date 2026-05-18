"""Write an evidence audit for Wai's remaining scientific claims.

The audit is intentionally stricter than the portfolio prose. It records
whether live NOAA metrics are present, whether those metrics are uncontaminated
by mock data, and whether meteorological forcing is merely supported by the
feature schema or actually validated in checked-in reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.meteorology import supported_meteorological_columns


ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
JSON_PATH = REPORTS_DIR / "scientific_evidence_audit.json"
MD_PATH = REPORTS_DIR / "scientific_evidence_audit.md"


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _station_records(report: dict) -> list[dict]:
    return [
        rec for key, rec in report.items()
        if not key.startswith("_") and isinstance(rec, dict)
    ]


def live_noaa_status(reports_dir: Path = REPORTS_DIR) -> dict:
    """Return a strict status block for the checked-in live NOAA artifact."""
    path = reports_dir / "noaa_live_metrics.json"
    if not path.exists():
        return {
            "status": "missing_live_noaa_metrics",
            "artifact": _display_path(path),
            "verified_live": False,
            "mock_used": None,
            "reason": "No checked-in noaa_live_metrics.json artifact.",
            "next_step": "Run `python -m scripts.evaluate_noaa_public` with network access.",
        }

    report = _read_json(path)
    if report is None:
        return {
            "status": "invalid_live_noaa_metrics_json",
            "artifact": _display_path(path),
            "verified_live": False,
            "mock_used": None,
            "reason": "The live NOAA metrics file is missing or invalid JSON.",
        }

    meta = report.get("_meta", {})
    records = _station_records(report)
    any_mock = bool(meta.get("any_mock_used")) or any(
        bool(rec.get("mock_used")) for rec in records
    )
    if meta.get("report_kind") != "live":
        return {
            "status": "wrong_report_kind",
            "artifact": _display_path(path),
            "verified_live": False,
            "mock_used": any_mock,
            "reason": f"Expected report_kind='live', got {meta.get('report_kind')!r}.",
        }
    if any_mock:
        return {
            "status": "live_report_contains_mock_records",
            "artifact": _display_path(path),
            "verified_live": False,
            "mock_used": True,
            "reason": "A live artifact must not contain mock_used=true records.",
        }
    return {
        "status": "verified_live_noaa_metrics",
        "artifact": _display_path(path),
        "verified_live": True,
        "mock_used": False,
        "stations_evaluated": int(meta.get("stations_evaluated", len(records))),
    }


def meteorological_forcing_status() -> dict:
    """Describe forcing support without claiming forcing validation."""
    cols = supported_meteorological_columns()
    return {
        "status": "supported_not_validated",
        "supported_columns": cols,
        "feature_pipeline": (
            "Numeric supported forcing columns are included by the tabular "
            "feature matrix when present."
        ),
        "checked_in_reports_use_forcing": False,
        "validated_storm_surge_skill": False,
        "reason": (
            "The checked-in synthetic, tidecast, NOAA mock, and NOAA live "
            "artifacts do not include real meteorological covariates."
        ),
    }


def operational_claim_status() -> dict:
    """Record the portfolio claim boundary."""
    return {
        "status": "guarded",
        "operational_noaa_proof": False,
        "allowed_claim": (
            "Synthetic and mock reports are reproducibility and plumbing "
            "evidence only."
        ),
        "disallowed_claim": (
            "Do not present synthetic or NOAA mock metrics as operational "
            "NOAA forecast performance."
        ),
    }


def build_audit(reports_dir: Path = REPORTS_DIR) -> dict:
    live = live_noaa_status(reports_dir)
    forcing = meteorological_forcing_status()
    return {
        "schema_version": 1,
        "generated_by": "scripts/audit_scientific_evidence.py",
        "operational_claim_status": operational_claim_status(),
        "noaa_live_evidence": live,
        "meteorological_forcing": forcing,
        "remaining_scientific_weaknesses": [
            {
                "weakness": "Operational NOAA proof",
                "status": "not_established",
                "current_control": "Mock and live reports have separate filenames and mock flags.",
                "needed_to_close": "Verified live NOAA metrics over representative windows.",
            },
            {
                "weakness": "Meteorological/storm-surge validation",
                "status": "partially_addressed",
                "current_control": "Feature schema accepts external forcing columns.",
                "needed_to_close": "Real wind/pressure/wave/rain covariates and event validation.",
            },
            {
                "weakness": "Checked-in live NOAA artifact",
                "status": (
                    "closed" if live.get("verified_live") else "open"
                ),
                "current_control": live.get("status"),
                "needed_to_close": live.get(
                    "next_step",
                    "Maintain refreshed live metrics with no mock records.",
                ),
            },
        ],
    }


def format_audit_md(audit: dict) -> str:
    live = audit["noaa_live_evidence"]
    forcing = audit["meteorological_forcing"]
    op = audit["operational_claim_status"]
    lines = [
        "# Wai Scientific Evidence Audit",
        "",
        "This report is a guardrail for portfolio claims. It does not add a new",
        "performance result; it records which evidence is present and which gaps",
        "remain open.",
        "",
        "## Claim Boundary",
        "",
        f"- Operational NOAA proof established: `{op['operational_noaa_proof']}`",
        f"- Allowed claim: {op['allowed_claim']}",
        f"- Disallowed claim: {op['disallowed_claim']}",
        "",
        "## Live NOAA Evidence",
        "",
        f"- Status: `{live['status']}`",
        f"- Artifact: `{live['artifact']}`",
        f"- Verified live with no mock records: `{live['verified_live']}`",
        f"- Reason: {live.get('reason', 'Live artifact passed integrity checks.')}",
        "",
        "## Meteorological Forcing",
        "",
        f"- Status: `{forcing['status']}`",
        f"- Checked-in reports use forcing: `{forcing['checked_in_reports_use_forcing']}`",
        f"- Validated storm-surge skill: `{forcing['validated_storm_surge_skill']}`",
        "- Supported columns: "
        + ", ".join(f"`{col}`" for col in forcing["supported_columns"]),
        f"- Reason: {forcing['reason']}",
        "",
        "## Remaining Scientific Weaknesses",
        "",
        "| Weakness | Status | Current control | Needed to close |",
        "| --- | --- | --- | --- |",
    ]
    for rec in audit["remaining_scientific_weaknesses"]:
        lines.append(
            "| {weakness} | {status} | {control} | {needed} |".format(
                weakness=rec["weakness"],
                status=rec["status"],
                control=rec["current_control"],
                needed=rec["needed_to_close"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    audit = build_audit()
    JSON_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    MD_PATH.write_text(format_audit_md(audit), encoding="utf-8")
    print(f"Saved {JSON_PATH.relative_to(ROOT)}")
    print(f"Saved {MD_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
