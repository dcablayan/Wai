"""Tests for the adaptive forecast cascade."""

from __future__ import annotations

import threading
import time

import pandas as pd
import pytest

from src.data.hohonu import mock_hohonu_observations
from src.data.noaa import mock_noaa_observations, mock_noaa_tide_predictions
from src.data.station_mapping import StationPair
from src.experts.base import ExpertForecast, ForecastExpert
from src.experts.capabilities import LATENCY_INSTANT, ExpertSpec
from src.forecasting import ForecastPipeline
from src.forecasting.pipeline import default_experts
from src.orchestration.cascade import (
    AdaptiveCascade,
    CapabilityGate,
    ExecutionBudget,
)
from src.orchestration.context import build_forecast_context
from src.orchestration.executor import run_experts


def _context(*, horizon_minutes=360, forecast_time="2024-01-01T18:00:00Z",
             residual_m=0.08, hohonu_qc="pass", noaa_periods=300, tide=True):
    station_id, noaa_id = "HOHONU_TEST", "NOAA_TEST"
    hohonu = mock_hohonu_observations(station_id, periods=300, qc_status=hohonu_qc)
    noaa_obs = mock_noaa_observations(noaa_id, periods=noaa_periods, residual_m=residual_m)
    noaa_tide = mock_noaa_tide_predictions(noaa_id, periods=420)
    if not tide:
        noaa_tide = noaa_tide.iloc[0:0]
    return build_forecast_context(
        target_station_id=station_id, paired_noaa_station_id=noaa_id,
        horizon_minutes=horizon_minutes, forecast_time_utc=forecast_time,
        hohonu_observations=hohonu, noaa_observations=noaa_obs,
        noaa_tide_predictions=noaa_tide, station_pair=StationPair(station_id, noaa_id),
    )


class _FixedExpert(ForecastExpert):
    def __init__(self, name, value, *, requires_tide=True, confidence=0.8, half=0.1):
        self.model_name = name
        self.spec = ExpertSpec(model_name=name, requires_tide=requires_tide,
                               latency_class=LATENCY_INSTANT)
        self._value = value
        self._confidence = confidence
        self._half = half

    def forecast(self, context):
        return ExpertForecast(
            model_name=self.model_name, forecast_time_utc=context.forecast_time_utc,
            target_time_utc=context.target_time_utc, horizon_minutes=context.horizon_minutes,
            predicted_water_level_m=self._value, lower_m=self._value - self._half,
            upper_m=self._value + self._half, confidence=self._confidence,
        )


class _ExplodingExpert(ForecastExpert):
    model_name = "exploding"
    spec = ExpertSpec(model_name="exploding", requires_tide=True, latency_class=LATENCY_INSTANT)

    def forecast(self, context):
        raise RuntimeError("boom")


class _SlowExpert(ForecastExpert):
    def __init__(self, name="slow", delay=0.05):
        self.model_name = name
        self.spec = ExpertSpec(model_name=name, requires_tide=True, latency_class=LATENCY_INSTANT)
        self._delay = delay

    def forecast(self, context):
        time.sleep(self._delay)
        tide = context.noaa_tide_prediction
        v = float(tide["water_level_m"])
        return ExpertForecast(
            model_name=self.model_name, forecast_time_utc=context.forecast_time_utc,
            target_time_utc=context.target_time_utc, horizon_minutes=context.horizon_minutes,
            predicted_water_level_m=v, lower_m=v - 0.1, upper_m=v + 0.1, confidence=0.7,
        )


class _BlockingExpert(_SlowExpert):
    def __init__(self, name, release: threading.Event):
        super().__init__(name=name, delay=0.0)
        self._release = release

    def forecast(self, context):
        self._release.wait(timeout=1.0)
        return super().forecast(context)


# --------------------------------------------------------------------------- #
def test_capability_gate_excludes_impossible_experts():
    # No tide -> every tide-requiring expert is excluded before any execution.
    ctx = _context(tide=False)
    gate = CapabilityGate()
    report = gate.evaluate(ctx, default_experts())
    assert "noaa_residual" in report.excluded
    assert "local_tide" in report.excluded
    assert "safe_fallback" in report.excluded
    assert "tide" in report.excluded["local_tide"]
    # local_persistence needs only local obs, which exist.
    assert "local_persistence" in report.eligible


def test_capability_gate_excludes_on_failed_qc_and_staleness():
    ctx = _context(hohonu_qc="fail", noaa_periods=10, forecast_time="2024-01-02T00:00:00Z")
    report = CapabilityGate().evaluate(ctx, default_experts())
    assert "failed QC" in report.excluded["local_persistence"]
    assert "stale" in report.excluded["noaa_residual"]


def test_primary_selection_prefers_residual_at_medium_horizon():
    pipe = ForecastPipeline()
    result = pipe.run(_context(horizon_minutes=360, residual_m=0.08))
    assert result.diagnostics["trace"]["stage_1_expert"] == "noaa_residual"


def test_normal_condition_early_stops_with_single_expert():
    result = ForecastPipeline().run(_context(horizon_minutes=360, residual_m=0.08))
    trace = result.diagnostics["trace"]
    assert trace["early_stop_reason"] is not None
    assert trace["escalated"] is False
    assert trace["expert_calls"] == 1
    assert result.experts_used == ["noaa_residual"]


def test_escalation_after_primary_failure():
    experts = default_experts()
    experts["noaa_residual"] = _ExplodingExpert()
    experts["noaa_residual"].model_name = "noaa_residual"
    pipe = ForecastPipeline(experts=experts)
    result = pipe.run(_context(horizon_minutes=360, residual_m=0.08))
    assert result.status == "available"
    assert result.diagnostics["trace"]["escalated"] is True
    assert "noaa_residual" in result.experts_excluded


def test_escalation_on_large_residual_event():
    result = ForecastPipeline().run(_context(horizon_minutes=360, residual_m=0.5))
    trace = result.diagnostics["trace"]
    assert trace["escalated"] is True
    assert any("event risk" in r or "residual" in r for r in trace["escalation_reasons"])
    assert trace["expert_calls"] >= 2


class _TidePlusExpert(ForecastExpert):
    """Tide-referenced expert that sits a fixed offset above the tide baseline."""

    def __init__(self, name, offset):
        self.model_name = name
        self.spec = ExpertSpec(model_name=name, requires_tide=True, latency_class=LATENCY_INSTANT)
        self._offset = offset

    def forecast(self, context):
        tide = context.noaa_tide_prediction
        v = float(tide["water_level_m"]) + self._offset
        return ExpertForecast(
            model_name=self.model_name, forecast_time_utc=context.forecast_time_utc,
            target_time_utc=context.target_time_utc, horizon_minutes=context.horizon_minutes,
            predicted_water_level_m=v, lower_m=v - 0.05, upper_m=v + 0.05, confidence=0.85,
        )


def test_escalation_after_post_forecast_disagreement_with_baseline():
    # Primary is tide-referenced but lands 0.6 m off the tide baseline while the
    # NOAA residual is small (no event risk). The disagreement is only knowable
    # after the forecast exists, and it alone triggers escalation.
    experts = default_experts()
    experts["noaa_residual"] = _TidePlusExpert("noaa_residual", 0.6)
    pipe = ForecastPipeline(experts=experts)
    result = pipe.run(_context(horizon_minutes=360, residual_m=0.05))
    trace = result.diagnostics["trace"]
    assert trace["escalated"] is True
    assert any("disagreement with tide baseline" in r for r in trace["escalation_reasons"])
    assert not any("event risk" in r for r in trace["escalation_reasons"])


def test_expert_call_budget_is_enforced():
    # Force escalation but cap calls at 1: only the primary may run.
    budget = ExecutionBudget(max_expert_calls=1, reserve_fallback_call=False)
    result = ForecastPipeline().run(_context(horizon_minutes=360, residual_m=0.5), budget=budget)
    assert result.diagnostics["trace"]["expert_calls"] <= 1


def test_fallback_call_is_reserved_when_all_experts_fail():
    # Every non-baseline expert explodes; the reserved safe_fallback still runs.
    experts = default_experts()
    for name in ("local_persistence", "local_tide", "noaa_residual", "regional_to_local_residual"):
        exp = _ExplodingExpert()
        exp.model_name = name
        exp.spec = ExpertSpec(model_name=name, requires_tide=("tide" in name or name != "local_persistence"),
                              latency_class=LATENCY_INSTANT)
        experts[name] = exp
    pipe = ForecastPipeline(experts=experts)
    result = pipe.run(_context(horizon_minutes=360, residual_m=0.08))
    assert result.status == "available"
    assert result.fallback_used is True
    assert result.experts_used == ["safe_fallback"]


def test_timeout_isolation_marks_expert_without_killing_batch():
    ctx = _context()
    release = threading.Event()
    experts = [_BlockingExpert("slow_a", release), _FixedExpert("fast", 0.1)]
    started = time.perf_counter()
    runs = run_experts(experts, ctx, parallel=True, max_parallelism=2, per_expert_timeout_ms=50)
    elapsed = time.perf_counter() - started
    release.set()
    by_name = {r.name: r for r in runs}
    assert by_name["slow_a"].timed_out is True
    assert by_name["fast"].forecast is not None and by_name["fast"].forecast.ok
    assert elapsed < 0.25


def test_single_expert_timeout_returns_without_waiting_for_background_work():
    ctx = _context()
    release = threading.Event()
    started = time.perf_counter()
    runs = run_experts(
        [_BlockingExpert("slow", release)],
        ctx,
        parallel=False,
        per_expert_timeout_ms=50,
    )
    elapsed = time.perf_counter() - started
    release.set()
    assert runs[0].timed_out is True
    assert runs[0].forecast is None
    assert elapsed < 0.25


def test_parallel_results_are_deterministically_ordered():
    ctx = _context()
    experts = [_SlowExpert("a", 0.03), _SlowExpert("b", 0.01), _SlowExpert("c", 0.02)]
    runs = run_experts(experts, ctx, parallel=True, max_parallelism=3)
    assert [r.name for r in runs] == ["a", "b", "c"]  # input order, not completion order


def test_verifier_triggered_fallback_recovers_with_safe_baseline():
    # Primary produces an implausible value; verifier rejects; safe_fallback recovers.
    experts = default_experts()
    bad = _FixedExpert("noaa_residual", 99.0, confidence=0.9)  # outside plausible range
    experts["noaa_residual"] = bad
    pipe = ForecastPipeline(experts=experts, budget=ExecutionBudget(max_expert_calls=1))
    result = pipe.run(_context(horizon_minutes=360, residual_m=0.08))
    assert result.status == "available"
    assert result.fallback_used is True
    assert "rejected by verifier" in (result.diagnostics["trace"]["fallback_reason"] or "")


def test_dependency_aware_staleness_does_not_penalise_unused_source():
    # NOAA stale, but the forecast is local_tide-only: confidence must not drop
    # for the unused stale NOAA source.
    ctx_stale = _context(horizon_minutes=720, hohonu_qc="fail", noaa_periods=20,
                         forecast_time="2024-01-02T06:00:00Z")
    result = ForecastPipeline().run(ctx_stale)
    # local_tide path; no "NOAA input is stale" warning should appear.
    assert all("noaa input is stale" not in w.lower() for w in result.warnings)
