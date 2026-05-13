"""Tests for src/reporting/report.py."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.reporting.report import generate_report


def _demo_df():
    ts = pd.date_range("2024-01-01", periods=200, freq="6min", tz="UTC")
    import numpy as np
    wl = 0.5 * np.sin(2 * np.pi * ts.astype("int64") / 1e18 * 3600 / 12.42) + 0.1
    return pd.DataFrame({
        "timestamp": ts,
        "station_id": "TEST-001",
        "water_level": wl,
        "datum": "MLLW",
        "units": "m",
        "lat": 21.3,
        "lon": -157.9,
        "source": "DEMO_SYNTHETIC",
    })


def test_report_creates_file():
    df = _demo_df()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.html"
        result = generate_report(df, {}, out)
        assert result.exists()
        assert result.stat().st_size > 0


def test_report_contains_station_id():
    df = _demo_df()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.html"
        generate_report(df, {}, out)
        content = out.read_text()
        assert "TEST-001" in content


def test_report_contains_disclaimer():
    df = _demo_df()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.html"
        generate_report(df, {}, out)
        content = out.read_text()
        assert "DEMO_SYNTHETIC" in content or "synthetic" in content.lower()


def test_report_with_metrics():
    df = _demo_df()
    metrics = {
        "persistence": {"mae": 0.22, "rmse": 0.28, "r2": 0.45, "nse": 0.45, "corr": 0.72},
        "harmonic_ridge": {"mae": 0.04, "rmse": 0.05, "r2": 0.97, "nse": 0.97, "corr": 0.99},
    }
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.html"
        generate_report(df, metrics, out)
        content = out.read_text()
        assert "persistence" in content
        assert "harmonic_ridge" in content


def test_report_default_threshold_uses_reference_window():
    df = _demo_df()
    reference = df.iloc[:100].copy()
    expected = float(reference["water_level"].mean() + 2 * reference["water_level"].std())
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.html"
        generate_report(df, {}, out, threshold_reference_df=reference)
        content = out.read_text()
        assert "Threshold Source" in content
        assert "Threshold Reference Range" in content
        assert "first_75_percent_reference_window" in content
        assert f"{expected:.2f}" in content
