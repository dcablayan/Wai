"""Tests for the live NOAA dashboard snapshot assembler."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.noaa import mock_noaa_observations, mock_noaa_tide_predictions
from src.data.noaa_live import (
    fetch_live_noaa_operational_guidance,
    fetch_live_noaa_snapshot,
)


class FakeNOAAAdapter:
    def __init__(self, observations, predictions):
        self.observations = observations
        self.predictions = predictions
        self.calls = []

    def fetch_observations(self, station_id, begin, end, **kwargs):
        self.calls.append(("water_level", station_id, begin, end, kwargs))
        return self.observations.copy()

    def fetch_tide_predictions(self, station_id, begin, end, **kwargs):
        self.calls.append(("predictions", station_id, begin, end, kwargs))
        return self.predictions.copy()


class FakeOFSAdapter:
    def __init__(self, guidance):
        self.guidance = guidance
        self.calls = []

    def fetch_operational_forecast(self, station_id, begin, end, **kwargs):
        self.calls.append(("ofs_water_level", station_id, begin, end, kwargs))
        return self.guidance.copy()


def test_live_snapshot_aligns_observations_and_predictions_without_fallback():
    start = "2026-07-20T00:00:00Z"
    observations = mock_noaa_observations(
        "1612340", start=start, periods=21, residual_m=0.08
    )
    predictions = mock_noaa_tide_predictions(
        "1612340", start=start, periods=21
    )
    adapter = FakeNOAAAdapter(observations, predictions)

    snapshot = fetch_live_noaa_snapshot(
        "1612340",
        lookback_hours=1,
        as_of="2026-07-20T02:00:00Z",
        adapter=adapter,
    )

    assert snapshot.station.name == "Honolulu"
    assert snapshot.station.state == "HI"
    assert snapshot.datum == "MLLW"
    assert len(snapshot.frame) == 11
    assert snapshot.latest_residual_m == pytest.approx(0.08)
    assert snapshot.latest_observed_at == pd.Timestamp("2026-07-20T02:00:00Z")
    assert [call[0] for call in adapter.calls] == ["water_level", "predictions"]
    assert all(call[4]["use_cache"] is False for call in adapter.calls)


def test_live_snapshot_rejects_unsupported_station_and_datum():
    with pytest.raises(ValueError, match="Unsupported NOAA live station"):
        fetch_live_noaa_snapshot("not-a-station")
    with pytest.raises(ValueError, match="Unsupported live datum"):
        fetch_live_noaa_snapshot("1612340", datum="BAD")


def test_live_snapshot_can_fetch_observations_without_tide_predictions():
    observations = mock_noaa_observations(
        "9075014", start="2026-07-20T00:00:00Z", periods=21
    )
    adapter = FakeNOAAAdapter(observations, predictions=pd.DataFrame())

    snapshot = fetch_live_noaa_snapshot(
        "9075014",
        lookback_hours=1,
        datum="IGLD",
        include_tide_predictions=False,
        as_of="2026-07-20T02:00:00Z",
        adapter=adapter,
    )

    assert snapshot.includes_tide_predictions is False
    assert snapshot.frame["predicted_m"].isna().all()
    assert [call[0] for call in adapter.calls] == ["water_level"]


def test_live_operational_guidance_keeps_history_and_future_without_fallback():
    guidance = mock_noaa_tide_predictions(
        "9414290", start="2026-07-19T00:00:00Z", periods=721
    )
    guidance["source"] = "NOAA_OFS_WATER_LEVEL"
    guidance["record_type"] = "forecast_guidance"
    adapter = FakeOFSAdapter(guidance)

    result = fetch_live_noaa_operational_guidance(
        "9414290",
        history_hours=24,
        forecast_hours=48,
        as_of="2026-07-20T00:00:00Z",
        adapter=adapter,
    )

    assert result.frame.iloc[0]["timestamp"] == pd.Timestamp("2026-07-19T00:00:00Z")
    assert result.frame.iloc[-1]["timestamp"] == pd.Timestamp("2026-07-22T00:00:00Z")
    assert result.frame["guidance_m"].notna().all()
    assert [call[0] for call in adapter.calls] == ["ofs_water_level"]
    assert adapter.calls[0][4]["use_cache"] is False
