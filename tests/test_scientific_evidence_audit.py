"""Tests for the scientific evidence audit report."""

from __future__ import annotations

import json

from scripts.audit_scientific_evidence import (
    build_audit,
    format_audit_md,
    live_noaa_status,
)


def test_live_noaa_status_reports_missing_artifact(tmp_path):
    status = live_noaa_status(tmp_path)

    assert status["status"] == "missing_live_noaa_metrics"
    assert status["verified_live"] is False
    assert "evaluate_noaa_public" in status["next_step"]


def test_live_noaa_status_accepts_clean_live_report(tmp_path):
    path = tmp_path / "noaa_live_metrics.json"
    path.write_text(
        json.dumps({
            "_meta": {
                "report_kind": "live",
                "any_mock_used": False,
                "stations_evaluated": 1,
            },
            "9414290": {"mock_used": False},
        }),
        encoding="utf-8",
    )

    status = live_noaa_status(tmp_path)

    assert status["status"] == "verified_live_noaa_metrics"
    assert status["verified_live"] is True
    assert status["mock_used"] is False


def test_live_noaa_status_rejects_mock_contamination(tmp_path):
    path = tmp_path / "noaa_live_metrics.json"
    path.write_text(
        json.dumps({
            "_meta": {"report_kind": "live", "any_mock_used": False},
            "9414290": {"mock_used": True},
        }),
        encoding="utf-8",
    )

    status = live_noaa_status(tmp_path)

    assert status["status"] == "live_report_contains_mock_records"
    assert status["verified_live"] is False


def test_build_audit_keeps_operational_proof_false(tmp_path):
    audit = build_audit(tmp_path)

    assert audit["operational_claim_status"]["operational_noaa_proof"] is False
    assert audit["meteorological_forcing"]["validated_storm_surge_skill"] is False
    assert audit["remaining_scientific_weaknesses"][0]["status"] == "not_established"


def test_format_audit_md_contains_remaining_weaknesses(tmp_path):
    audit = build_audit(tmp_path)
    md = format_audit_md(audit)

    assert "Operational NOAA proof" in md
    assert "Meteorological Forcing" in md
    assert "missing_live_noaa_metrics" in md
