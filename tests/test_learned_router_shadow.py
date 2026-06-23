"""Tests for learned-router shadow mode and forward-time policy evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.hohonu import mock_hohonu_observations
from src.data.noaa import mock_noaa_observations, mock_noaa_tide_predictions
from src.data.station_mapping import StationPair
from src.evaluation import (
    EXHAUSTIVE,
    HistoricalReplayConfig,
    evaluate_router_policies,
    run_historical_replay,
)
from src.evaluation.router_training import RouterTrainingConfig, train_router_from_replay
from src.forecasting import ForecastPipeline
from src.orchestration.context import build_forecast_context
from src.orchestration.learned_router import LearnedRouter

STATION, NOAA = "HOHONU_TEST", "NOAA_TEST"


def _replay(step=180):
    return run_historical_replay(
        target_station_id=STATION, paired_noaa_station_id=NOAA,
        hohonu_observations=mock_hohonu_observations(STATION, periods=720),
        noaa_observations=mock_noaa_observations(NOAA, periods=720, residual_m=0.12),
        noaa_tide_predictions=mock_noaa_tide_predictions(NOAA, periods=900),
        mode=EXHAUSTIVE,
        config=HistoricalReplayConfig(horizon_minutes=360, min_history_hours=12, step_minutes=step),
    )


def _context():
    return build_forecast_context(
        target_station_id=STATION, paired_noaa_station_id=NOAA, horizon_minutes=360,
        forecast_time_utc="2024-01-01T18:00:00Z",
        hohonu_observations=mock_hohonu_observations(STATION, periods=300),
        noaa_observations=mock_noaa_observations(NOAA, periods=300, residual_m=0.12),
        noaa_tide_predictions=mock_noaa_tide_predictions(NOAA, periods=420),
        station_pair=StationPair(STATION, NOAA),
    )


def test_learned_router_shadow_mode_records_without_controlling(tmp_path):
    replay = _replay(step=120)
    model_path = tmp_path / "router.pkl"
    train_router_from_replay(replay, config=RouterTrainingConfig(min_training_rows=4), model_path=model_path)
    router = LearnedRouter.load(model_path)

    pipe = ForecastPipeline(learned_router=router)
    result = pipe.run(_context())
    shadow = result.diagnostics["learned_router_shadow"]
    # The shadow recommendation is recorded but the route is still rule-driven.
    assert shadow is not None
    assert "would_select" in shadow
    assert result.diagnostics["trace"]["route_source"] == "rule_cascade"
    # The cascade's actual primary is unchanged by the shadow recommendation.
    assert result.diagnostics["trace"]["stage_1_expert"] in {
        "local_persistence", "local_tide", "noaa_residual", "regional_to_local_residual",
    }


def test_shadow_recommendation_reports_source(tmp_path):
    replay = _replay(step=120)
    model_path = tmp_path / "router.pkl"
    train_router_from_replay(replay, config=RouterTrainingConfig(min_training_rows=4), model_path=model_path)
    router = LearnedRouter.load(model_path)
    rec = router.shadow_recommend(_context())
    assert rec.source in {"learned", "fallback_low_margin", "fallback_low_support", "fallback_model_error"}


def test_forward_time_router_policy_evaluation_reports_oracle_and_regret():
    replay = _replay(step=120)
    ev = evaluate_router_policies(replay)
    assert ev.validation == "forward_time"
    assert ev.n_test > 0
    assert ev.oracle_mae is not None
    assert ev.learned_router_mae is not None
    # Oracle is the best achievable, so regret versus oracle is non-negative.
    assert ev.routing_regret_m >= -1e-9
    assert ev.oracle_mae <= ev.learned_router_mae + 1e-9
