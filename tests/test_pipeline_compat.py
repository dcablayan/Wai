"""Backward-compatibility of the public ForecastPipeline.run(context) interface."""

from __future__ import annotations

import pytest

from src.data.hohonu import mock_hohonu_observations
from src.data.noaa import mock_noaa_observations, mock_noaa_tide_predictions
from src.data.station_mapping import StationPair
from src.forecasting import ForecastPipeline
from src.orchestration.context import build_forecast_context

STATION, NOAA = "HOHONU_TEST", "NOAA_TEST"


def _context(horizon=360, residual=0.08):
    return build_forecast_context(
        target_station_id=STATION, paired_noaa_station_id=NOAA, horizon_minutes=horizon,
        forecast_time_utc="2024-01-01T18:00:00Z",
        hohonu_observations=mock_hohonu_observations(STATION, periods=300),
        noaa_observations=mock_noaa_observations(NOAA, periods=300, residual_m=residual),
        noaa_tide_predictions=mock_noaa_tide_predictions(NOAA, periods=420),
        station_pair=StationPair(STATION, NOAA),
    )


def test_run_with_only_context_positional_argument_still_works():
    # The historical public call shape: pipeline.run(context).
    result = ForecastPipeline().run(_context())
    payload = result.to_dict()
    assert payload["status"] == "available"
    assert payload["forecast_m"] is not None
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["experts_used"]


def test_legacy_flat_router_mode_matches_old_output_shape():
    result = ForecastPipeline(adaptive=False).run(_context())
    payload = result.to_dict()
    assert payload["status"] == "available"
    assert payload["combination_method"] == "weighted_median"
    assert set(payload) >= {
        "station_id", "forecast_time_utc", "target_time_utc", "horizon_minutes",
        "forecast_m", "lower_m", "upper_m", "confidence", "regime", "experts_used",
        "experts_excluded", "combination_method", "fallback_used", "warnings",
        "diagnostics", "status",
    }


def test_adaptive_and_legacy_both_return_forecast_result():
    for adaptive in (True, False):
        result = ForecastPipeline(adaptive=adaptive).run(_context(horizon=30))
        assert result.status == "available"
        assert result.forecast_m is not None
