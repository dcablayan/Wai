"""Tests for src/data/validation.py."""

import pandas as pd
import pytest

from src.data.validation import ValidationReport, validate


def _make_df(**overrides):
    base = {
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="6min", tz="UTC"),
        "station_id": "TEST",
        "water_level": [0.5] * 100,
        "datum": "MLLW",
        "units": "m",
        "lat": 21.0,
        "lon": -157.0,
        "source": "DEMO_SYNTHETIC",
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_clean_data_is_clean():
    df = _make_df()
    report = validate(df)
    assert report.is_clean
    assert report.warnings == []


def test_missing_column_raises():
    df = _make_df()
    df = df.drop(columns=["water_level"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate(df)


def test_nan_values_detected():
    wl = [0.5] * 100
    wl[10] = float("nan")
    df = _make_df(water_level=wl)
    report = validate(df)
    assert report.nan_values == 1


def test_duplicate_timestamps_detected():
    df = _make_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    report = validate(df)
    assert report.duplicate_timestamps >= 1


def test_out_of_range_detected():
    wl = [0.5] * 100
    wl[5] = 999.0  # far outside valid range
    df = _make_df(water_level=wl)
    report = validate(df)
    assert report.out_of_range_values >= 1


def test_timezone_naive_flagged():
    ts = pd.date_range("2024-01-01", periods=100, freq="6min")  # no tz
    df = _make_df(timestamp=ts)
    report = validate(df)
    assert report.timezone_issues > 0
