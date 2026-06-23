"""Origin-time numerical regime and difficulty diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RegimeAndDifficultyThinker:
    """Estimate regime, forecast difficulty, and event risk without forecasting."""

    expert_id: str = "regime_difficulty_thinker"

    def analyze(self, context: Any, visible_messages: list[Any] | None = None) -> dict[str, Any]:
        horizon_hours = max(0.0, float(context.horizon_minutes) / 60.0)
        residual = abs(float(context.recent_noaa_residual_m or 0.0))
        residual_trend = abs(float(context.noaa_residual_trend_m_per_hour or 0.0))
        local_trend = abs(float(context.recent_hohonu_trend_m_per_hour or 0.0))
        volatility = _recent_volatility(context)
        freshness_penalty = _freshness_penalty(context)
        qc_penalty = 0.0 if context.hohonu_qc_ok and context.noaa_qc_ok else 0.25
        tide_missing_penalty = 0.2 if context.noaa_tide_prediction is None and context.local_tide_prediction is None else 0.0
        weather_signal = min(1.0, abs(float(context.pressure_trend or 0.0)) / 4.0)
        wind_signal = min(1.0, abs(float(context.wind_speed_mps or 0.0)) / 20.0)

        horizon_score = min(1.0, horizon_hours / 24.0)
        residual_score = min(1.0, residual / 0.45)
        trend_score = min(1.0, max(residual_trend, local_trend) / 0.25)
        volatility_score = min(1.0, volatility / 0.25)
        event_risk = _clamp01(
            0.55 * residual_score
            + 0.25 * trend_score
            + 0.10 * volatility_score
            + 0.05 * weather_signal
            + 0.05 * wind_signal
        )
        difficulty = _clamp01(
            0.25 * horizon_score
            + 0.25 * event_risk
            + 0.15 * freshness_penalty
            + 0.15 * qc_penalty
            + 0.10 * volatility_score
            + 0.10 * tide_missing_penalty
        )

        stable = _clamp01(1.0 - event_risk - 0.5 * volatility_score)
        regional = _clamp01(residual_score * (0.6 + 0.4 * trend_score))
        local = _clamp01((local_trend / 0.25) * (1.0 - residual_score))
        missing = _clamp01(freshness_penalty + tide_missing_penalty + qc_penalty)
        total = stable + regional + local + missing
        if total <= 0:
            regime_probabilities = {"stable_tide": 1.0}
        else:
            regime_probabilities = {
                "stable_tide": stable / total,
                "regional_residual": regional / total,
                "local_trend": local / total,
                "data_limited": missing / total,
            }

        recommended_subtasks = []
        recommended_experts = []
        if difficulty >= 0.45:
            recommended_subtasks.append("ANALYZE_RESIDUAL_DYNAMICS")
            recommended_experts.append("residual_dynamics_thinker")
        if event_risk >= 0.35:
            recommended_subtasks.extend(["FORECAST_REGIONAL_RESIDUAL", "TRANSFER_REGIONAL_SIGNAL"])
            recommended_experts.extend(["noaa_residual", "regional_to_local_residual"])
        elif context.horizon_minutes <= 90 and context.hohonu_qc_ok:
            recommended_subtasks.append("FORECAST_LOCAL_LEVEL")
            recommended_experts.append("local_persistence")
        else:
            recommended_subtasks.append("FORECAST_LOCAL_LEVEL")
            recommended_experts.append("local_tide")

        return {
            "regime_probabilities": regime_probabilities,
            "forecast_difficulty": float(difficulty),
            "event_risk": float(event_risk),
            "suspected_forcing_mechanism": _mechanism(regime_probabilities),
            "important_data_gaps": _data_gaps(context),
            "out_of_distribution_score": float(_clamp01(event_risk + qc_penalty + tide_missing_penalty)),
            "expected_value_of_additional_expert_calls": float(_clamp01(difficulty + 0.5 * event_risk)),
            "recommended_next_subtasks": recommended_subtasks,
            "recommended_experts": sorted(set(recommended_experts), key=recommended_experts.index),
            "recommended_verifier_type": "event_risk_verifier" if event_risk >= 0.35 else "physics_datum_verifier",
        }


def _recent_volatility(context: Any) -> float:
    frame = getattr(context, "recent_hohonu_observations", None)
    if frame is None or len(frame) < 3:
        return 0.0
    values = frame["water_level_m"].astype(float).to_numpy()
    diffs = np.diff(values)
    if len(diffs) == 0:
        return 0.0
    return float(np.nanstd(diffs))


def _freshness_penalty(context: Any) -> float:
    penalties = []
    for source, limit in (("hohonu", 60 * 60), ("noaa", 3 * 60 * 60)):
        freshness = float(context.observation_freshness_seconds.get(source, math.inf))
        if math.isinf(freshness):
            penalties.append(1.0)
        else:
            penalties.append(min(1.0, freshness / max(1.0, limit) - 1.0))
    return float(max(0.0, max(penalties) if penalties else 0.0))


def _data_gaps(context: Any) -> list[str]:
    gaps = []
    if context.latest_hohonu_observation is None:
        gaps.append("missing_latest_hohonu")
    if context.latest_noaa_observation is None:
        gaps.append("missing_latest_noaa")
    if context.noaa_tide_prediction is None and context.local_tide_prediction is None:
        gaps.append("missing_tide_prediction")
    if not context.hohonu_qc_ok:
        gaps.append("hohonu_qc_not_good")
    if not context.noaa_qc_ok:
        gaps.append("noaa_qc_not_good")
    if context.pressure_trend is None:
        gaps.append("missing_pressure_trend")
    if context.wind_speed_mps is None:
        gaps.append("missing_wind_speed")
    return gaps


def _mechanism(probabilities: dict[str, float]) -> str:
    return max(probabilities.items(), key=lambda item: item[1])[0]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
