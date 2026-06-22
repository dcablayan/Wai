"""Tests for leakage-safe historical replay."""

from __future__ import annotations

import json

import pandas as pd

from src.data.hohonu import mock_hohonu_observations
from src.data.noaa import mock_noaa_observations, mock_noaa_tide_predictions
from src.evaluation import HistoricalReplayConfig, run_historical_replay


def test_historical_replay_generates_router_training_rows_without_future_leakage():
    station_id = "HOHONU_TEST"
    noaa_id = "NOAA_TEST"
    replay = run_historical_replay(
        target_station_id=station_id,
        paired_noaa_station_id=noaa_id,
        hohonu_observations=mock_hohonu_observations(station_id, periods=720),
        noaa_observations=mock_noaa_observations(noaa_id, periods=720, residual_m=0.1),
        noaa_tide_predictions=mock_noaa_tide_predictions(noaa_id, periods=900),
        config=HistoricalReplayConfig(
            horizon_minutes=360,
            min_history_hours=12,
            step_minutes=240,
        ),
    )
    assert len(replay) > 0
    required = {
        "forecast_origin_utc",
        "target_time_utc",
        "context_features",
        "expert_predictions",
        "actual_m",
        "error_by_expert",
        "approx_compute_cost_ms",
        "max_hohonu_input_time_utc",
    }
    assert required.issubset(replay.columns)

    for _, row in replay.iterrows():
        origin = pd.Timestamp(row["forecast_origin_utc"])
        target = pd.Timestamp(row["target_time_utc"])
        max_input = pd.Timestamp(row["max_hohonu_input_time_utc"])
        assert max_input <= origin
        assert target > origin
        expert_predictions = json.loads(row["expert_predictions"])
        assert "local_persistence" in expert_predictions
        assert "actual_m" not in expert_predictions["local_persistence"]


def test_historical_replay_returns_empty_when_history_is_too_short():
    replay = run_historical_replay(
        target_station_id="HOHONU_TEST",
        paired_noaa_station_id="NOAA_TEST",
        hohonu_observations=mock_hohonu_observations(periods=10),
        noaa_observations=mock_noaa_observations(periods=10),
        noaa_tide_predictions=mock_noaa_tide_predictions(periods=20),
        config=HistoricalReplayConfig(horizon_minutes=360, min_history_hours=24),
    )
    assert replay.empty
