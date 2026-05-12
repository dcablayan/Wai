"""Tests for src/data/loader.py."""

import pandas as pd
import pytest

from src.data.loader import REQUIRED_COLUMNS, load_demo_data


def test_demo_data_has_required_columns():
    df = load_demo_data()
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_demo_data_not_empty():
    df = load_demo_data()
    assert len(df) > 0


def test_timestamps_are_utc():
    df = load_demo_data()
    assert df["timestamp"].dt.tz is not None


def test_water_level_is_numeric():
    df = load_demo_data()
    assert pd.api.types.is_float_dtype(df["water_level"]) or pd.api.types.is_numeric_dtype(df["water_level"])


def test_source_is_demo():
    df = load_demo_data()
    assert (df["source"] == "DEMO_SYNTHETIC").all(), "All rows must be labeled DEMO_SYNTHETIC"


def test_sorted_by_timestamp():
    df = load_demo_data()
    for station in df["station_id"].unique():
        sub = df[df["station_id"] == station]
        assert sub["timestamp"].is_monotonic_increasing


def test_multiple_stations():
    df = load_demo_data()
    assert df["station_id"].nunique() >= 2
