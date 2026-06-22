"""Tests for canonical Hohonu/NOAA ingestion foundations."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.canonicalize import (
    CANONICAL_COLUMNS,
    DatumMismatchError,
    assert_compatible_datums,
    canonicalize_frame,
)
from src.data.hohonu import HohonuAdapter, mock_hohonu_observations
from src.data.noaa import NOAACoopsAdapter, mock_noaa_observations, mock_noaa_tide_predictions


def test_canonical_schema_validation_and_utc_normalization():
    raw = pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00"],
        "station_id": ["A"],
        "water_level": [1.0],
        "units": ["m"],
        "lat": [21.3],
        "lon": [-157.8],
        "datum": ["mllw"],
    })
    out = canonicalize_frame(
        raw,
        source="TEST",
        record_type="observation",
        retrieved_at="2024-01-01T00:10:00Z",
    )
    assert list(out.columns) == CANONICAL_COLUMNS
    assert str(out["timestamp_utc"].dt.tz) == "UTC"
    assert out.loc[0, "datum"] == "MLLW"
    assert out.loc[0, "latency_seconds"] == 600


def test_unit_handling_converts_feet_to_meters():
    raw = pd.DataFrame({
        "timestamp": ["2024-01-01T00:00:00Z"],
        "station_id": ["A"],
        "water_level": [3.28084],
        "units": ["ft"],
        "lat": [21.3],
        "lon": [-157.8],
        "datum": ["MLLW"],
    })
    out = canonicalize_frame(raw, source="TEST", record_type="observation")
    assert out.loc[0, "water_level_m"] == pytest.approx(1.0, rel=1e-5)


def test_datum_mismatch_rejected():
    a = mock_hohonu_observations(datum="MLLW")
    b = mock_noaa_tide_predictions(datum="NAVD88")
    with pytest.raises(DatumMismatchError, match="incompatible datums"):
        assert_compatible_datums([a, b])


def test_hohonu_mock_ingestion_returns_canonical_observations():
    df = mock_hohonu_observations("HOHONU_TEST", periods=5, qc_status="pass")
    assert set(CANONICAL_COLUMNS).issubset(df.columns)
    assert (df["source"] == "HOHONU_MOCK").all()
    assert (df["record_type"] == "observation").all()
    assert (df["station_id"] == "HOHONU_TEST").all()


def test_hohonu_adapter_canonicalizes_provider_records():
    records = [
        {
            "observed_at": "2024-01-01T00:00:00Z",
            "value": 0.5,
            "units": "m",
            "quality": "pass",
            "flags": ["range_ok"],
        }
    ]
    df = HohonuAdapter().canonicalize_records(
        records,
        station_id="H1",
        latitude=21.3,
        longitude=-157.8,
        datum="MLLW",
        retrieved_at="2024-01-01T00:06:00Z",
    )
    assert df.loc[0, "water_level_m"] == 0.5
    assert df.loc[0, "qc_flags"] == ["range_ok"]


def test_noaa_mock_ingestion_returns_observations_and_predictions():
    obs = mock_noaa_observations("NOAA_TEST", periods=5)
    pred = mock_noaa_tide_predictions("NOAA_TEST", periods=5)
    assert (obs["record_type"] == "observation").all()
    assert (pred["record_type"] == "tide_prediction").all()
    assert (obs["station_id"] == "NOAA_TEST").all()
    assert (pred["station_id"] == "NOAA_TEST").all()


def test_noaa_adapter_canonicalizes_mock_payload():
    payload = {
        "metadata": {"lat": 21.3, "lon": -157.8, "datum": "MLLW"},
        "data": [{"t": "2024-01-01 00:00", "v": "0.42", "q": "verified"}],
    }
    df = NOAACoopsAdapter().canonicalize_payload(
        payload,
        station_id="N1",
        product="water_level",
        record_type="observation",
        latitude=21.3,
        longitude=-157.8,
        retrieved_at="2024-01-01T00:06:00Z",
    )
    assert df.loc[0, "source"] == "NOAA_COOPS_MOCK"
    assert df.loc[0, "water_level_m"] == pytest.approx(0.42)
    assert df.loc[0, "qc_status"] == "verified"
