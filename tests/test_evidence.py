"""Tests for content-based report freshness."""

from __future__ import annotations

from src.evidence import evidence_freshness, source_fingerprint


def test_source_fingerprint_ignores_reports_but_changes_with_source(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "reports").mkdir()
    source = tmp_path / "src" / "model.py"
    source.write_text("VALUE = 1\n")
    report = tmp_path / "reports" / "summary.json"
    report.write_text("{}")
    initial = source_fingerprint(tmp_path)
    report.write_text('{"generated": true}')
    assert source_fingerprint(tmp_path) == initial
    source.write_text("VALUE = 2\n")
    assert source_fingerprint(tmp_path) != initial


def test_freshness_requires_matching_recorded_fingerprint(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.py").write_text("VALUE = 1\n")
    fingerprint = source_fingerprint(tmp_path)
    assert evidence_freshness({"source_fingerprint": fingerprint}, tmp_path)[
        "fresh_at_verification"
    ] is True
    assert evidence_freshness({"git_sha": "legacy-only"}, tmp_path)[
        "fresh_at_verification"
    ] is False
