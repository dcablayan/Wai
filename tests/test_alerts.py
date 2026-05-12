"""Tests for high-water alert detection (src/alerts.py)."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.alerts import (
    AlertConfig,
    compute_threshold,
    detect_alerts,
    generate_alert_summary,
    save_alert_summary,
)


def _synthetic_df(n: int = 200) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=n, freq="6min", tz="UTC")
    rng = np.random.default_rng(7)
    t = np.arange(n) * (6 / 60)
    water_level = np.sin(2 * np.pi * t / 12.42) + 0.05 * rng.standard_normal(n)
    return pd.DataFrame({
        "timestamp": timestamps,
        "station_id": "TEST",
        "water_level": water_level,
        "datum": "MLLW",
        "units": "m",
        "lat": 21.3,
        "lon": -157.8,
        "source": "DEMO_SYNTHETIC",
    })


# ── compute_threshold ─────────────────────────────────────────────────────────

def test_std_threshold_above_mean():
    df = _synthetic_df()
    config = AlertConfig(mode="std", k=2.0)
    t = compute_threshold(df["water_level"], config)
    assert t > df["water_level"].mean()


def test_absolute_threshold():
    df = _synthetic_df()
    config = AlertConfig(mode="absolute", absolute_threshold=0.8)
    assert compute_threshold(df["water_level"], config) == pytest.approx(0.8)


def test_percentile_threshold():
    df = _synthetic_df()
    config = AlertConfig(mode="percentile", percentile=95.0)
    t = compute_threshold(df["water_level"], config)
    expected = float(np.percentile(df["water_level"].dropna(), 95.0))
    assert t == pytest.approx(expected, rel=1e-5)


def test_absolute_threshold_missing_value_raises():
    config = AlertConfig(mode="absolute", absolute_threshold=None)
    with pytest.raises(ValueError, match="absolute_threshold"):
        compute_threshold(pd.Series([1.0, 2.0]), config)


def test_unknown_mode_raises():
    config = AlertConfig(mode="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown alert mode"):
        compute_threshold(pd.Series([1.0, 2.0]), config)


# ── detect_alerts ─────────────────────────────────────────────────────────────

def test_detect_alerts_returns_dataframe():
    df = _synthetic_df()
    config = AlertConfig(mode="std", k=2.0)
    alerts = detect_alerts(df, config)
    assert isinstance(alerts, pd.DataFrame)


def test_detect_alerts_count_above_threshold():
    df = _synthetic_df(n=500)
    config = AlertConfig(mode="std", k=2.0)
    alerts = detect_alerts(df, config)
    threshold = compute_threshold(df["water_level"], config)
    expected = int((df["water_level"] >= threshold).sum())
    assert len(alerts) == expected


def test_detect_alerts_all_above_threshold():
    df = _synthetic_df()
    config = AlertConfig(mode="std", k=2.0)
    alerts = detect_alerts(df, config)
    if not alerts.empty:
        assert (alerts["water_level"] >= alerts["threshold"]).all()


def test_detect_alerts_with_reference_series():
    df = _synthetic_df(n=300)
    train = df.iloc[:200]
    test = df.iloc[200:]
    config = AlertConfig(mode="std", k=2.0)
    alerts = detect_alerts(test, config, reference_series=train["water_level"])
    assert isinstance(alerts, pd.DataFrame)


def test_detect_alerts_zero_at_high_threshold():
    df = _synthetic_df()
    config = AlertConfig(mode="absolute", absolute_threshold=999.0)
    alerts = detect_alerts(df, config)
    assert len(alerts) == 0


# ── generate_alert_summary ────────────────────────────────────────────────────

def test_alert_summary_keys():
    df = _synthetic_df()
    config = AlertConfig(mode="std", k=2.0)
    summary = generate_alert_summary(df, config, station_id="TEST")
    for key in ("station_id", "alert_mode", "threshold", "n_total_obs", "n_alerts",
                "alert_rate_pct", "events"):
        assert key in summary, f"Missing key: {key}"


def test_alert_summary_counts_match():
    df = _synthetic_df(n=300)
    config = AlertConfig(mode="std", k=2.0)
    summary = generate_alert_summary(df, config, station_id="TEST")
    assert summary["n_total_obs"] == len(df)
    alerts = detect_alerts(df, config)
    assert summary["n_alerts"] == len(alerts)


def test_alert_rate_pct_range():
    df = _synthetic_df()
    config = AlertConfig(mode="std", k=2.0)
    summary = generate_alert_summary(df, config, station_id="TEST")
    assert 0.0 <= summary["alert_rate_pct"] <= 100.0


# ── save_alert_summary ────────────────────────────────────────────────────────

def test_save_alert_summary_creates_file():
    df = _synthetic_df()
    config = AlertConfig(mode="std", k=2.0)
    summary = generate_alert_summary(df, config, station_id="TEST")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sub" / "alert_summary.json"
        save_alert_summary(summary, out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["station_id"] == "TEST"
