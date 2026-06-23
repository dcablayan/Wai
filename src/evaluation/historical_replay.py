"""Offline historical replay for router-training data and policy evaluation.

Two clearly separated modes:

EXHAUSTIVE
    Run every available expert exactly once per origin, store those outputs, and
    reveal the actual target only after predictions are fixed.  The pipeline
    *reuses* those stored forecasts (``precomputed_forecasts``) instead of
    running any expert a second time, so no expert is ever executed twice at the
    same origin.  This is the dataset for router training/evaluation.

POLICY
    Run only the experts the adaptive cascade actually requests.  This measures
    real production compute savings (mean expert calls, latency) and must not be
    mixed with exhaustive compute cost.

Both modes share one :class:`PreparedStationData` index per station and advance
incrementally through origins (``searchsorted`` slices), instead of re-filtering
and re-sorting the full dataset from the beginning at every origin.  Leakage is
preserved: only observations with ``timestamp_utc <= forecast_origin`` enter
features.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import pandas as pd

from src.forecasting.pipeline import ForecastPipeline, default_experts
from src.orchestration.context import context_from_prepared
from src.orchestration.prepared import PreparedStationData, _nearest_index, _upper_bound

EXHAUSTIVE = "exhaustive"
POLICY = "policy"


@dataclass(frozen=True)
class HistoricalReplayConfig:
    """Configuration for leakage-safe historical replay."""

    horizon_minutes: int = 360
    min_history_hours: float = 24.0
    step_minutes: int = 60
    actual_tolerance_minutes: int = 6


def run_historical_replay(
    *,
    target_station_id: str,
    paired_noaa_station_id: str,
    hohonu_observations: pd.DataFrame,
    noaa_observations: pd.DataFrame,
    noaa_tide_predictions: pd.DataFrame,
    pipeline: ForecastPipeline | None = None,
    config: HistoricalReplayConfig | None = None,
    mode: str = EXHAUSTIVE,
    update_skill: bool = False,
) -> pd.DataFrame:
    """Generate a replay table without using future observations as inputs."""

    cfg = config or HistoricalReplayConfig()
    pipe = pipeline or ForecastPipeline()
    all_experts = default_experts(include_placeholders=False)

    prepared = PreparedStationData.build(
        target_station_id=target_station_id,
        paired_noaa_station_id=paired_noaa_station_id,
        hohonu_observations=hohonu_observations,
        noaa_observations=noaa_observations,
        noaa_tide_predictions=noaa_tide_predictions,
    )
    hohonu = prepared.hohonu
    hohonu_ts = prepared.hohonu_ts
    if hohonu.empty:
        raise ValueError("Historical replay requires local Hohonu observations for actual values")

    start_origin = hohonu.iloc[0]["timestamp_utc"] + pd.Timedelta(hours=cfg.min_history_hours)
    last_origin = hohonu.iloc[-1]["timestamp_utc"] - pd.Timedelta(minutes=cfg.horizon_minutes)
    if last_origin < start_origin:
        return pd.DataFrame()

    rows = []
    origin = start_origin
    while origin <= last_origin:
        target_time = origin + pd.Timedelta(minutes=cfg.horizon_minutes)
        actual_row = _nearest_actual(hohonu, hohonu_ts, target_time, cfg.actual_tolerance_minutes)

        start = time.perf_counter()
        context = context_from_prepared(
            prepared, forecast_time_utc=origin, horizon_minutes=cfg.horizon_minutes
        )
        context_build_ms = (time.perf_counter() - start) * 1000.0

        if mode == EXHAUSTIVE:
            expert_predictions = {
                name: expert.forecast(context) for name, expert in all_experts.items()
            }
            result = pipe.run(
                context, precomputed_forecasts=expert_predictions, context_build_ms=context_build_ms
            )
        else:  # POLICY
            expert_predictions = {}
            result = pipe.run(context, context_build_ms=context_build_ms)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        actual_m = None if actual_row is None else float(actual_row["water_level_m"])
        errors = {}
        if actual_m is not None and expert_predictions:
            for name, forecast in expert_predictions.items():
                if forecast.ok:
                    errors[name] = float(forecast.predicted_water_level_m - actual_m)

        trace = result.diagnostics.get("trace", {})
        rows.append({
            "forecast_origin_utc": str(origin),
            "target_time_utc": str(target_time),
            "target_station_id": target_station_id,
            "paired_noaa_station_id": paired_noaa_station_id,
            "horizon_minutes": cfg.horizon_minutes,
            "mode": mode,
            "regime": result.regime,
            "context_features": json.dumps(_context_features(context), sort_keys=True),
            "selected_experts": json.dumps(result.experts_used),
            "expert_predictions": json.dumps(_expert_predictions(expert_predictions), sort_keys=True),
            "actual_m": actual_m,
            "error_by_expert": json.dumps(errors, sort_keys=True),
            "forecast_m": result.forecast_m,
            "forecast_lower_m": result.lower_m,
            "forecast_upper_m": result.upper_m,
            "confidence": result.confidence,
            "forecast_error_m": None if actual_m is None or result.forecast_m is None else float(result.forecast_m - actual_m),
            "event_severity_m": None if actual_m is None else max(0.0, abs(actual_m) - 0.75),
            "missing_data_conditions": json.dumps(_missing_conditions(context), sort_keys=True),
            "expert_calls": trace.get("expert_calls", 0),
            "cache_hits": trace.get("cache_hits", 0),
            "fallback_used": result.fallback_used,
            "escalated": trace.get("escalated", False),
            "early_stop_reason": trace.get("early_stop_reason"),
            "approx_compute_cost_ms": round(elapsed_ms, 3),
            "context_build_ms": round(context_build_ms, 4),
            "max_hohonu_input_time_utc": context.diagnostics.get("max_hohonu_input_time_utc"),
            "max_noaa_input_time_utc": context.diagnostics.get("max_noaa_input_time_utc"),
            "result_status": result.status,
        })

        if update_skill and actual_m is not None:
            _update_skill_from_predictions(
                pipe, context, result.regime, expert_predictions, actual_m
            )

        origin += pd.Timedelta(minutes=cfg.step_minutes)

    return pd.DataFrame(rows)


def _update_skill_from_predictions(pipe, context, regime, expert_predictions, actual_m) -> None:
    store = getattr(pipe, "skill_store", None)
    if store is None or not expert_predictions:
        return
    for name, fc in expert_predictions.items():
        if not fc.ok:
            store.update(
                expert=name, station=context.target_station_id,
                horizon_minutes=context.horizon_minutes, regime=regime,
                abs_error=None, failed=True, latency_ms=getattr(fc, "latency_ms", 0.0),
            )
            continue
        err = abs(float(fc.predicted_water_level_m) - actual_m)
        covered = float(fc.lower_m) <= actual_m <= float(fc.upper_m)
        store.update(
            expert=name, station=context.target_station_id,
            horizon_minutes=context.horizon_minutes, regime=regime,
            abs_error=err, covered=covered, failed=False,
            latency_ms=getattr(fc, "latency_ms", 0.0),
        )


def _nearest_actual(frame, ts, target_time, tolerance_minutes):
    idx = _nearest_index(ts, target_time)
    if idx is None:
        return None
    row = frame.iloc[idx]
    delta = abs((row["timestamp_utc"] - target_time).total_seconds()) / 60.0
    if delta > tolerance_minutes:
        return None
    return row.to_dict()


def _context_features(context) -> dict:
    return {
        "horizon_minutes": context.horizon_minutes,
        "hohonu_freshness_seconds": context.observation_freshness_seconds.get("hohonu"),
        "noaa_freshness_seconds": context.observation_freshness_seconds.get("noaa"),
        "hohonu_qc_status": context.qc_status.get("hohonu"),
        "noaa_qc_status": context.qc_status.get("noaa"),
        "recent_hohonu_trend_m_per_hour": context.recent_hohonu_trend_m_per_hour,
        "recent_noaa_residual_m": context.recent_noaa_residual_m,
        "noaa_residual_trend_m_per_hour": context.noaa_residual_trend_m_per_hour,
        "tide_phase": context.tide_phase,
    }


def _expert_predictions(forecasts: dict) -> dict:
    return {
        name: {
            "status": forecast.status,
            "prediction_m": forecast.predicted_water_level_m,
            "lower_m": forecast.lower_m,
            "upper_m": forecast.upper_m,
            "confidence": forecast.confidence,
        }
        for name, forecast in forecasts.items()
    }


def _missing_conditions(context) -> dict:
    return {
        "missing_latest_hohonu": context.latest_hohonu_observation is None,
        "missing_latest_noaa": context.latest_noaa_observation is None,
        "missing_tide_prediction": context.noaa_tide_prediction is None,
        "hohonu_qc_ok": context.hohonu_qc_ok,
        "noaa_qc_ok": context.noaa_qc_ok,
    }
