"""Offline historical replay for future router-training datasets."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import pandas as pd

from src.forecasting.pipeline import ForecastPipeline, default_experts
from src.orchestration.context import build_forecast_context


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
) -> pd.DataFrame:
    """Generate a replay table without using future observations as inputs."""

    cfg = config or HistoricalReplayConfig()
    pipe = pipeline or ForecastPipeline()
    all_experts = default_experts(include_placeholders=False)

    hohonu = _sort(hohonu_observations)
    noaa = _sort(noaa_observations)
    tide = _sort(noaa_tide_predictions)

    if hohonu.empty:
        raise ValueError("Historical replay requires local Hohonu observations for actual values")

    start_origin = hohonu["timestamp_utc"].min() + pd.Timedelta(hours=cfg.min_history_hours)
    last_origin = hohonu["timestamp_utc"].max() - pd.Timedelta(minutes=cfg.horizon_minutes)
    if last_origin < start_origin:
        return pd.DataFrame()

    rows = []
    origin = start_origin
    while origin <= last_origin:
        target_time = origin + pd.Timedelta(minutes=cfg.horizon_minutes)
        actual_row = _nearest_actual(hohonu, target_time, cfg.actual_tolerance_minutes)

        hohonu_hist = hohonu[hohonu["timestamp_utc"] <= origin].copy()
        noaa_hist = noaa[noaa["timestamp_utc"] <= origin].copy()

        start = time.perf_counter()
        context = build_forecast_context(
            target_station_id=target_station_id,
            paired_noaa_station_id=paired_noaa_station_id,
            horizon_minutes=cfg.horizon_minutes,
            forecast_time_utc=origin,
            hohonu_observations=hohonu_hist,
            noaa_observations=noaa_hist,
            noaa_tide_predictions=tide,
        )
        expert_predictions = {
            name: expert.forecast(context)
            for name, expert in all_experts.items()
        }
        result = pipe.run(context)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        actual_m = None if actual_row is None else float(actual_row["water_level_m"])
        errors = {}
        if actual_m is not None:
            for name, forecast in expert_predictions.items():
                if forecast.ok:
                    errors[name] = float(forecast.predicted_water_level_m - actual_m)

        rows.append({
            "forecast_origin_utc": str(origin),
            "target_time_utc": str(target_time),
            "target_station_id": target_station_id,
            "paired_noaa_station_id": paired_noaa_station_id,
            "horizon_minutes": cfg.horizon_minutes,
            "context_features": json.dumps(_context_features(context), sort_keys=True),
            "selected_experts": json.dumps(result.experts_used),
            "expert_predictions": json.dumps(_expert_predictions(expert_predictions), sort_keys=True),
            "actual_m": actual_m,
            "error_by_expert": json.dumps(errors, sort_keys=True),
            "forecast_m": result.forecast_m,
            "forecast_error_m": None if actual_m is None or result.forecast_m is None else float(result.forecast_m - actual_m),
            "event_severity_m": None if actual_m is None else max(0.0, abs(actual_m) - 0.75),
            "missing_data_conditions": json.dumps(_missing_conditions(context), sort_keys=True),
            "approx_compute_cost_ms": round(elapsed_ms, 3),
            "max_hohonu_input_time_utc": context.diagnostics.get("max_hohonu_input_time_utc"),
            "max_noaa_input_time_utc": context.diagnostics.get("max_noaa_input_time_utc"),
            "result_status": result.status,
        })
        origin += pd.Timedelta(minutes=cfg.step_minutes)

    return pd.DataFrame(rows)


def _sort(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def _nearest_actual(
    frame: pd.DataFrame,
    target_time: pd.Timestamp,
    tolerance_minutes: int,
) -> dict | None:
    if frame.empty:
        return None
    idx = (frame["timestamp_utc"] - target_time).abs().idxmin()
    row = frame.loc[idx]
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
