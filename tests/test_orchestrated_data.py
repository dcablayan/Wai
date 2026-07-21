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
from src.data.noaa import (
    NOAAConfig,
    NOAACoopsAdapter,
    mock_noaa_observations,
    mock_noaa_tide_predictions,
)


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


def test_noaa_adapter_drops_missing_value_sentinels_without_losing_valid_rows():
    payload = {
        "metadata": {"lat": 46.78, "lon": -92.09, "datum": "IGLD"},
        "data": [
            {"t": "2024-01-01 00:00", "v": "", "q": "preliminary"},
            {"t": "2024-01-01 00:06", "v": "183.42", "q": "preliminary"},
            {"t": "2024-01-01 00:12", "v": "-", "q": "preliminary"},
        ],
    }

    frame = NOAACoopsAdapter().canonicalize_payload(
        payload,
        station_id="9099064",
        product="water_level",
        record_type="observation",
        latitude=46.78,
        longitude=-92.09,
        datum="IGLD",
    )

    assert len(frame) == 1
    assert frame.iloc[0]["water_level_m"] == pytest.approx(183.42)


def test_noaa_wind_payload_uses_product_specific_metric_fields():
    payload = {
        "metadata": {"lat": "21.3", "lon": "-157.9"},
        "data": [
            {"t": "2024-01-01 00:00", "s": "5.2", "d": "90", "g": "8.1"}
        ],
    }
    df = NOAACoopsAdapter().canonicalize_payload(
        payload,
        station_id="1612340",
        product="wind",
        record_type="weather_observation",
        latitude=21.3,
        longitude=-157.9,
        datum="METEOROLOGICAL",
        source="NOAA_WIND",
        retrieved_at="2024-01-01T00:06:00Z",
    )
    assert pd.isna(df.loc[0, "water_level_m"])
    assert df.loc[0, "wind_speed_mps"] == pytest.approx(5.2)
    assert df.loc[0, "wind_direction_deg"] == pytest.approx(90.0)
    assert df.loc[0, "wind_gust_mps"] == pytest.approx(8.1)


def test_noaa_pressure_payload_is_not_mislabeled_as_water_level():
    payload = {"data": [{"t": "2024-01-01 00:00", "v": "1013.2"}]}
    df = NOAACoopsAdapter().canonicalize_payload(
        payload,
        station_id="1612340",
        product="air_pressure",
        record_type="weather_observation",
        latitude=21.3,
        longitude=-157.9,
        datum="METEOROLOGICAL",
        source="NOAA_AIR_PRESSURE",
    )
    assert pd.isna(df.loc[0, "water_level_m"])
    assert df.loc[0, "air_pressure_hpa"] == pytest.approx(1013.2)


def test_noaa_fetch_uses_canonical_adapter_and_cache(tmp_path):
    payload = {
        "metadata": {"lat": 21.3, "lon": -157.8, "datum": "MLLW"},
        "data": [{"t": "2024-01-01 00:00", "v": "0.42", "q": "verified"}],
    }

    class Response:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class Session:
        def __init__(self):
            self.calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            return Response()

    session = Session()
    adapter = NOAACoopsAdapter(
        config=NOAAConfig(cache_dir=tmp_path, max_retries=1),
        session=session,
    )
    first = adapter.fetch_observations(
        "1612340",
        "2024-01-01",
        "2024-01-01",
        latitude=21.3,
        longitude=-157.8,
    )
    second = adapter.fetch_observations(
        "1612340",
        "2024-01-01",
        "2024-01-01",
        latitude=21.3,
        longitude=-157.8,
    )
    assert session.calls == 1
    assert first.loc[0, "water_level_m"] == pytest.approx(0.42)
    assert second.loc[0, "water_level_m"] == pytest.approx(0.42)


def test_noaa_operational_guidance_and_product_validation():
    payload = {
        "metadata": {"datum": "MLLW"},
        "data": [{"t": "2024-01-01 00:00", "v": "0.75"}],
    }
    frame = NOAACoopsAdapter().canonicalize_payload(
        payload,
        station_id="9414290",
        product="ofs_water_level",
        record_type="forecast_guidance",
        latitude=37.8,
        longitude=-122.5,
        source="NOAA_OFS_WATER_LEVEL",
    )
    assert frame.loc[0, "record_type"] == "forecast_guidance"
    assert frame.loc[0, "water_level_m"] == pytest.approx(0.75)
    with pytest.raises(ValueError, match="Unsupported NOAA weather product"):
        NOAACoopsAdapter().fetch_weather_observations(
            "9414290",
            "2024-01-01",
            "2024-01-01",
            product="salinity",
            latitude=37.8,
            longitude=-122.5,
        )
