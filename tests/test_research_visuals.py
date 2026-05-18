"""Tests for generated research report visuals."""

from __future__ import annotations

from pathlib import Path

from scripts import generate_research_visuals as visuals


def test_generate_research_visuals_writes_expected_svg_files(tmp_path, monkeypatch):
    monkeypatch.setattr(visuals, "IMAGES_DIR", Path(tmp_path))

    paths = visuals.generate_visuals()

    assert {p.name for p in paths} == {
        "actual_vs_predicted.svg",
        "error_by_horizon.svg",
        "baseline_comparison.svg",
        "residual_plot.svg",
    }
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("<svg")
        assert "</svg>" in text
