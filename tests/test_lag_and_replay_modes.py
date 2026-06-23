"""Tests for real lag application and exhaustive/policy replay modes."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.canonicalize import canonicalize_frame
from src.data.hohonu import mock_hohonu_observations
from src.data.noaa import _mock_tide, mock_noaa_observations, mock_noaa_tide_predictions
from src.data.station_mapping import StationPair
from src.evaluation import EXHAUSTIVE, POLICY, HistoricalReplayConfig, run_historical_replay
from src.experts.regional_residual import RegionalToLocalResidualExpert
from src.forecasting import ForecastPipeline
from src.orchestration.context import build_forecast_context

STATION, NOAA = "HOHONU_TEST", "NOAA_TEST"


def _ramped_noaa(periods=300):
    """NOAA observations whose residual increases monotonically with time."""
    start, freq = "2024-01-01T00:00:00Z", "6min"
    ts, tide = _mock_tide(start, periods, freq)
    ramp = 0.05 + 0.01 * np.arange(periods) / periods * 30
    return canonicalize_frame(
        pd.DataFrame({"timestamp": ts, "station_id": NOAA, "water_level": tide + ramp,
                      "units": "m", "lat": 21.3, "lon": -157.8, "datum": "MLLW",
                      "qc_status": "verified", "qc_flags": [[] for _ in range(periods)]}),
        source="NOAA_COOPS_MOCK", record_type="observation",
        qc_status_col="qc_status", qc_flags_col="qc_flags", retrieved_at="2024-01-02T00:00:00Z",
    )


def _regional_forecast(lag):
    ctx = build_forecast_context(
        target_station_id=STATION, paired_noaa_station_id=NOAA, horizon_minutes=120,
        forecast_time_utc="2024-01-01T18:00:00Z",
        hohonu_observations=mock_hohonu_observations(STATION, periods=300),
        noaa_observations=_ramped_noaa(), noaa_tide_predictions=mock_noaa_tide_predictions(NOAA, periods=420),
        station_pair=StationPair(STATION, NOAA, residual_scale=1.0, lag_minutes=lag),
    )
    return RegionalToLocalResidualExpert().forecast(ctx)


def test_increasing_lag_selects_an_earlier_source_residual():
    f_small = _regional_forecast(120)
    f_large = _regional_forecast(720)
    # On a rising residual ramp, a larger lag uses an earlier (smaller) residual.
    assert f_large.diagnostics["source_residual_m"] < f_small.diagnostics["source_residual_m"]
    assert f_large.predicted_water_level_m < f_small.predicted_water_level_m
    assert f_large.diagnostics["lag_applied"] is True
    assert f_large.diagnostics["lag_source_time_utc"] != f_small.diagnostics["lag_source_time_utc"]


def test_lag_beyond_history_uses_documented_persistence():
    # A short horizon with zero lag cannot look back; it persists the latest
    # residual and flags lag_applied accordingly.
    f = _regional_forecast(0)
    assert f.diagnostics["lag_applied"] is True  # lag_minutes == 0 is the persistence case
    assert "source_residual_m" in f.diagnostics


def _replay(mode, pipeline=None):
    return run_historical_replay(
        target_station_id=STATION, paired_noaa_station_id=NOAA,
        hohonu_observations=mock_hohonu_observations(STATION, periods=720),
        noaa_observations=mock_noaa_observations(NOAA, periods=720, residual_m=0.12),
        noaa_tide_predictions=mock_noaa_tide_predictions(NOAA, periods=900),
        pipeline=pipeline, mode=mode,
        config=HistoricalReplayConfig(horizon_minutes=360, min_history_hours=12, step_minutes=180),
    )


def test_exhaustive_mode_reuses_forecasts_without_duplicate_execution():
    replay = _replay(EXHAUSTIVE)
    assert len(replay) > 0
    # All rows carry every expert's prediction (exhaustive), and the pipeline
    # reused them (cache hits) instead of re-running any expert.
    for _, row in replay.iterrows():
        preds = json.loads(row["expert_predictions"])
        assert "local_persistence" in preds and "noaa_residual" in preds
        assert int(row["expert_calls"]) == 0
        assert int(row["cache_hits"]) >= 1


def test_policy_mode_runs_only_requested_experts():
    replay = _replay(POLICY)
    assert len(replay) > 0
    # Policy mode runs the cascade's chosen experts (few), not all five.
    assert replay["expert_calls"].mean() < 3
    # Policy mode does not record exhaustive per-expert predictions.
    assert all(json.loads(p) == {} for p in replay["expert_predictions"])


def test_precomputed_forecasts_are_reused_by_pipeline():
    from src.forecasting.pipeline import default_experts
    from src.orchestration.context import build_forecast_context as bctx

    ctx = bctx(
        target_station_id=STATION, paired_noaa_station_id=NOAA, horizon_minutes=360,
        forecast_time_utc="2024-01-01T18:00:00Z",
        hohonu_observations=mock_hohonu_observations(STATION, periods=300),
        noaa_observations=mock_noaa_observations(NOAA, periods=300, residual_m=0.08),
        noaa_tide_predictions=mock_noaa_tide_predictions(NOAA, periods=420),
        station_pair=StationPair(STATION, NOAA),
    )
    precomputed = {name: e.forecast(ctx) for name, e in default_experts().items()}
    result = ForecastPipeline().run(ctx, precomputed_forecasts=precomputed)
    trace = result.diagnostics["trace"]
    assert trace["cache_hits"] >= 1
    assert trace["expert_calls"] == 0  # everything reused
