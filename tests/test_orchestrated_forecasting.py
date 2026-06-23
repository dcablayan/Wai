"""Tests for expert routing, combination, verification, and pipeline output."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.hohonu import mock_hohonu_observations
from src.data.noaa import mock_noaa_observations, mock_noaa_tide_predictions
from src.data.station_mapping import StationPair
from src.experts.base import ExpertForecast
from src.forecasting import ForecastPipeline
from src.orchestration.combiner import ForecastCombiner
from src.orchestration.context import build_forecast_context
from src.orchestration.router import RuleBasedOrchestrator
from src.orchestration.verifier import ForecastVerifier


def _context(
    *,
    horizon_minutes: int = 360,
    forecast_time: str = "2024-01-01T18:00:00Z",
    residual_m: float = 0.08,
    hohonu_qc: str = "pass",
    noaa_periods: int = 300,
):
    station_id = "HOHONU_TEST"
    noaa_id = "NOAA_TEST"
    hohonu = mock_hohonu_observations(station_id, periods=300, qc_status=hohonu_qc)
    noaa_obs = mock_noaa_observations(noaa_id, periods=noaa_periods, residual_m=residual_m)
    noaa_tide = mock_noaa_tide_predictions(noaa_id, periods=420)
    return build_forecast_context(
        target_station_id=station_id,
        paired_noaa_station_id=noaa_id,
        horizon_minutes=horizon_minutes,
        forecast_time_utc=forecast_time,
        hohonu_observations=hohonu,
        noaa_observations=noaa_obs,
        noaa_tide_predictions=noaa_tide,
        station_pair=StationPair(station_id, noaa_id),
    )


def _forecast(name: str, value: float, confidence: float = 0.8, status: str = "success"):
    return ExpertForecast(
        model_name=name,
        forecast_time_utc=pd.Timestamp("2024-01-01T00:00:00Z"),
        target_time_utc=pd.Timestamp("2024-01-01T06:00:00Z"),
        horizon_minutes=360,
        predicted_water_level_m=value if status == "success" else None,
        lower_m=value - 0.1 if status == "success" else None,
        upper_m=value + 0.1 if status == "success" else None,
        confidence=confidence if status == "success" else 0.0,
        status=status,
        message="" if status == "success" else "simulated failure",
    )


def test_router_selects_local_persistence_for_short_fresh_horizon():
    decision = RuleBasedOrchestrator().route(_context(horizon_minutes=30))
    assert decision.regime == "fresh_local_short_horizon"
    assert "local_persistence" in decision.selected_experts


def test_router_selects_tide_and_noaa_residual_for_normal_horizon():
    decision = RuleBasedOrchestrator().route(_context(horizon_minutes=360, residual_m=0.08))
    assert decision.regime == "normal_tide_residual"
    assert decision.selected_experts == ["local_tide", "noaa_residual"]


def test_router_adds_regional_expert_for_large_noaa_residual():
    decision = RuleBasedOrchestrator().route(_context(horizon_minutes=360, residual_m=0.4))
    assert decision.regime == "regional_non_tidal_event"
    assert "noaa_residual" in decision.selected_experts
    assert "regional_to_local_residual" in decision.selected_experts


def test_router_excludes_local_persistence_after_failed_qc():
    decision = RuleBasedOrchestrator().route(_context(horizon_minutes=30, hohonu_qc="fail"))
    assert "local_persistence" in decision.excluded_experts
    assert "failed QC" in decision.excluded_experts["local_persistence"]


def test_router_uses_local_path_when_noaa_data_is_stale():
    context = _context(
        horizon_minutes=60,
        forecast_time="2024-01-02T00:00:00Z",
        noaa_periods=80,
    )
    decision = RuleBasedOrchestrator().route(context)
    assert "noaa_residual" in decision.excluded_experts
    assert "local_persistence" in decision.selected_experts


def test_combiner_ignores_failed_forecasts():
    combined = ForecastCombiner().combine([
        _forecast("good", 1.0),
        _forecast("failed", 5.0, status="failed"),
    ])
    assert combined.forecast_m == pytest.approx(1.0)
    assert combined.experts_used == ["good"]


def test_weighted_median_prefers_higher_weight_expert():
    combined = ForecastCombiner(weights={"a": 0.9, "b": 0.1}).combine([
        _forecast("a", 1.0),
        _forecast("b", 2.0),
    ])
    assert combined.forecast_m == pytest.approx(1.0)


def test_verifier_widens_interval_on_model_disagreement():
    context = _context()
    forecasts = [_forecast("a", 0.0), _forecast("b", 1.0)]
    combined = ForecastCombiner().combine(forecasts, method="simple_median")
    before_width = combined.upper_m - combined.lower_m
    verified, report = ForecastVerifier().verify(combined, context=context, forecasts=forecasts)
    assert verified is not None
    assert verified.upper_m - verified.lower_m > before_width
    assert any("disagreement" in warning for warning in report.warnings)


def test_pipeline_returns_structured_forecast():
    result = ForecastPipeline().run(_context())
    payload = result.to_dict()
    assert payload["status"] == "available"
    assert payload["station_id"] == "HOHONU_TEST"
    assert payload["forecast_m"] is not None
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["experts_used"]


def test_pipeline_excludes_local_persistence_and_uses_safe_tide_path_on_qc_fail():
    # With the adaptive cascade, a failed-local-QC short horizon excludes
    # local_persistence (and stale NOAA experts) and serves the valid tide-based
    # path; it no longer needs to run the redundant safe_fallback expert because
    # local_tide is itself a safe tide-only forecast.
    result = ForecastPipeline().run(_context(horizon_minutes=30, hohonu_qc="fail", noaa_periods=10))
    payload = result.to_dict()
    assert payload["status"] == "available"
    assert "local_persistence" in payload["experts_excluded"]
    assert "failed QC" in payload["experts_excluded"]["local_persistence"]
    assert payload["forecast_m"] is not None
    # The forecast comes from a tide-based safe path.
    assert any(e in payload["experts_used"] for e in ("local_tide", "safe_fallback"))
