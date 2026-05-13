"""Tests for src/data/loader.py."""

import warnings
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.loader import (
    NOAA_API_URL,
    REQUIRED_COLUMNS,
    _noaa_api_params,
    _parse_noaa_response,
    load_demo_data,
    load_noaa_predictions,
)


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


# ── NOAA predictions loader (mocked — no network calls in tests) ──────────────

def _make_noaa_payload(station_id: str = "9414290", n: int = 5) -> dict:
    """Build a minimal NOAA CO-OPS API response payload."""
    records = []
    for i in range(n):
        records.append({
            "t": f"2024-01-{1 + i // 240:02d} {(i * 6 // 60) % 24:02d}:{(i * 6) % 60:02d}",
            "v": f"{0.5 + i * 0.01:.4f}",
        })
    return {
        "metadata": {"lat": "37.806", "lon": "-122.465"},
        "data": records,
    }


def test_noaa_api_params_structure():
    """_noaa_api_params should return correct keys without making a network call."""
    params = _noaa_api_params(
        "9414290", "20240101", "20240131",
        "predictions", "MLLW", "metric", "gmt",
    )
    assert params["station"] == "9414290"
    assert params["product"] == "predictions"
    assert params["begin_date"] == "20240101"
    assert params["end_date"] == "20240131"
    assert params["datum"] == "MLLW"
    assert params["format"] == "json"


def test_noaa_api_url_constant():
    assert "tidesandcurrents.noaa.gov" in NOAA_API_URL


def test_parse_noaa_response_schema():
    payload = _make_noaa_payload()
    df = _parse_noaa_response(payload, "9414290", "MLLW", "metric")
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_parse_noaa_response_source_label():
    payload = _make_noaa_payload()
    df = _parse_noaa_response(payload, "9414290", "MLLW", "metric", source_label="NOAA_PREDICTIONS")
    assert (df["source"] == "NOAA_PREDICTIONS").all()


def test_parse_noaa_response_error_raises():
    payload = {"error": {"message": "No data available for this station."}}
    with pytest.raises(ValueError, match="NOAA API error"):
        _parse_noaa_response(payload, "9414290", "MLLW", "metric")


def test_parse_noaa_response_empty_data_raises():
    payload = {"metadata": {}, "data": []}
    with pytest.raises(ValueError, match="no data"):
        _parse_noaa_response(payload, "9414290", "MLLW", "metric")


def test_load_noaa_predictions_mocked():
    """load_noaa_predictions makes a single GET request with product=predictions."""
    payload = _make_noaa_payload()
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status.return_value = None

    with patch("src.data.loader.requests.get", return_value=mock_resp) as mock_get:
        df = load_noaa_predictions("9414290", "20240101", "20240131")

    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args
    params_sent = call_kwargs[1].get("params") or call_kwargs[0][1]
    assert params_sent["product"] == "predictions"
    assert df["source"].iloc[0] == "NOAA_PREDICTIONS"
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


# ── predictions-key response fixture (CO-OPS predictions product) ─────────────

def _make_noaa_predictions_payload(station_id: str = "9414290", n: int = 5) -> dict:
    """Payload where records live under 'predictions' key (not 'data')."""
    records = [
        {
            "t": f"2024-01-{1 + i // 240:02d} {(i * 6 // 60) % 24:02d}:{(i * 6) % 60:02d}",
            "v": f"{0.5 + i * 0.01:.4f}",
        }
        for i in range(n)
    ]
    return {
        "metadata": {"lat": "37.806", "lon": "-122.465"},
        "predictions": records,  # <-- CO-OPS predictions product uses this key
    }


def test_parse_noaa_response_predictions_key():
    """_parse_noaa_response must handle records under 'predictions' key."""
    payload = _make_noaa_predictions_payload()
    df = _parse_noaa_response(payload, "9414290", "MLLW", "metric")
    assert len(df) == 5
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_parse_noaa_response_predictions_key_no_data_key():
    """Payload with 'predictions' key but no 'data' key must still parse."""
    payload = _make_noaa_predictions_payload()
    assert "data" not in payload, "Fixture should not have 'data' key"
    df = _parse_noaa_response(payload, "9414290", "MLLW", "metric")
    assert len(df) > 0


def test_parse_noaa_response_empty_both_keys_raises():
    """Payload with empty 'data' and empty 'predictions' must raise ValueError."""
    payload = {"metadata": {}, "data": [], "predictions": []}
    with pytest.raises(ValueError, match="no data"):
        _parse_noaa_response(payload, "9414290", "MLLW", "metric")


def test_parse_noaa_response_datum_mismatch_warns():
    """Datum mismatch between request and response metadata should emit a warning."""
    payload = {
        "metadata": {"lat": "37.806", "lon": "-122.465", "datum": "NAVD"},
        "data": [{"t": "2024-01-01 00:00", "v": "1.234"}],
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _parse_noaa_response(payload, "9414290", "MLLW", "metric")
    texts = [str(w.message) for w in caught]
    assert any("datum" in t.lower() for t in texts), (
        "Expected a datum-mismatch warning but got none"
    )
