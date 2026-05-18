"""Tests for optional meteorological forcing feature support."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineering import build_feature_matrix
from src.features.meteorology import (
    audit_meteorological_columns,
    available_meteorological_columns,
    supported_meteorological_columns,
)


def _make_df(n: int = 90) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=n, freq="6min", tz="UTC")
    t = np.arange(n) * (6 / 60)
    wl = np.sin(2 * np.pi * t / 12.42)
    return pd.DataFrame({
        "timestamp": timestamps,
        "station_id": "TEST",
        "water_level": wl,
        "datum": "MLLW",
        "units": "m",
        "lat": 21.3,
        "lon": -157.8,
        "source": "DEMO_SYNTHETIC",
    })


def test_supported_meteorological_columns_are_stable():
    cols = supported_meteorological_columns()
    assert "wind_speed_mps" in cols
    assert "air_pressure_hpa" in cols
    assert len(cols) == len(set(cols))


def test_feature_matrix_includes_numeric_meteorological_covariates():
    df = _make_df()
    df["wind_speed_mps"] = np.linspace(3.0, 9.0, len(df))
    df["air_pressure_hpa"] = np.linspace(1012.0, 1004.0, len(df))

    X, _ = build_feature_matrix(df)

    assert "wind_speed_mps" in X.columns
    assert "air_pressure_hpa" in X.columns


def test_feature_matrix_excludes_noaa_prediction_baseline_column():
    df = _make_df()
    df["noaa_prediction"] = df["water_level"]

    X, _ = build_feature_matrix(df)

    assert "noaa_prediction" not in X.columns


def test_meteorological_audit_flags_missing_and_non_numeric_columns():
    df = _make_df()
    df["wind_speed_mps"] = np.linspace(3.0, 9.0, len(df))
    df["air_pressure_hpa"] = "unknown"

    audit = audit_meteorological_columns(
        df, required=["wind_speed_mps", "air_pressure_hpa", "wave_height_m"]
    )

    assert audit["present_columns"] == ["wind_speed_mps", "air_pressure_hpa"]
    assert audit["numeric_columns"] == ["wind_speed_mps"]
    assert audit["missing_columns"] == ["wave_height_m"]
    assert audit["non_numeric_columns"] == ["air_pressure_hpa"]
    assert audit["usable"] is False


def test_available_meteorological_columns_respects_supported_schema():
    df = _make_df()
    df["wind_speed_mps"] = 4.0
    df["unrecognized_weather"] = 1.0

    assert available_meteorological_columns(df) == ["wind_speed_mps"]
