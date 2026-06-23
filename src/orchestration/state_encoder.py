"""Deterministic feature encoder for Wai Ultra coordinator policies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.orchestration.coordination_state import CoordinationState
from src.orchestration.protocol import Role


FEATURE_SCHEMA_VERSION = "wai-ultra-state-v1"


@dataclass(frozen=True)
class FeatureSchema:
    """Versioned fixed-size coordinator-state feature schema."""

    version: str = FEATURE_SCHEMA_VERSION
    feature_names: tuple[str, ...] = field(default_factory=lambda: DEFAULT_FEATURE_NAMES)

    def validate_artifact_schema(self, artifact_schema: dict[str, Any]) -> None:
        version = artifact_schema.get("version")
        names = tuple(artifact_schema.get("feature_names", ()))
        if version != self.version or names != self.feature_names:
            raise ValueError(
                "Coordinator artifact feature schema mismatch: "
                f"expected {self.version}/{len(self.feature_names)} features, "
                f"got {version}/{len(names)} features"
            )


DEFAULT_FEATURE_NAMES = (
    "bias",
    "horizon_hours",
    "horizon_short",
    "horizon_day_fraction",
    "forecast_hour_sin",
    "forecast_hour_cos",
    "forecast_month_sin",
    "forecast_month_cos",
    "tide_phase_rising",
    "tide_phase_falling",
    "tide_phase_slack",
    "hohonu_available",
    "noaa_available",
    "tide_available",
    "hohonu_freshness_hours",
    "noaa_freshness_hours",
    "hohonu_qc_ok",
    "noaa_qc_ok",
    "local_trend_abs",
    "local_volatility",
    "noaa_residual_abs",
    "noaa_residual_trend_abs",
    "pressure_trend_abs",
    "wind_speed_scaled",
    "station_pair_scale",
    "station_pair_lag_hours",
    "model_disagreement",
    "recent_skill_mean",
    "remaining_turns_fraction",
    "remaining_deadline_fraction",
    "recursion_depth",
    "difficulty_estimate",
    "event_risk_estimate",
    "thinker_turns",
    "worker_turns",
    "verifier_turns",
    "unique_worker_count",
    "used_local_persistence",
    "used_local_tide",
    "used_noaa_residual",
    "used_regional_transfer",
    "used_synthesis",
    "used_fallback",
    "candidate_available",
    "candidate_value",
    "candidate_interval_width",
    "candidate_confidence",
    "worker_prediction_mean",
    "worker_prediction_span",
    "worker_interval_mean",
    "last_verifier_accept",
    "last_verifier_replan",
    "last_verifier_reject",
    "last_verifier_continue",
    "last_requested_regional",
    "last_requested_synthesis",
    "last_requested_fallback",
    "last_thinker_recommended_regional",
    "last_thinker_recommended_local",
    "last_thinker_recommended_verifier_event",
)


class StateEncoder:
    """Convert origin-time context and transcript into fixed numerical features."""

    def __init__(self, schema: FeatureSchema | None = None) -> None:
        self.schema = schema or FeatureSchema()

    def encode(self, state: CoordinationState) -> np.ndarray:
        context = state.original_context
        features = {
            "bias": 1.0,
            "horizon_hours": _scale(context.horizon_minutes / 60.0, 24.0),
            "horizon_short": 1.0 if context.horizon_minutes <= 90 else 0.0,
            "horizon_day_fraction": _scale(context.horizon_minutes, 24 * 60),
            "forecast_hour_sin": _time_sin(context.forecast_time_utc.hour, 24),
            "forecast_hour_cos": _time_cos(context.forecast_time_utc.hour, 24),
            "forecast_month_sin": _time_sin(context.forecast_time_utc.month, 12),
            "forecast_month_cos": _time_cos(context.forecast_time_utc.month, 12),
            "tide_phase_rising": 1.0 if context.tide_phase == "rising" else 0.0,
            "tide_phase_falling": 1.0 if context.tide_phase == "falling" else 0.0,
            "tide_phase_slack": 1.0 if context.tide_phase == "slack" else 0.0,
            "hohonu_available": 1.0 if context.latest_hohonu_observation is not None else 0.0,
            "noaa_available": 1.0 if context.latest_noaa_observation is not None else 0.0,
            "tide_available": 1.0 if context.noaa_tide_prediction is not None or context.local_tide_prediction is not None else 0.0,
            "hohonu_freshness_hours": _freshness(context, "hohonu"),
            "noaa_freshness_hours": _freshness(context, "noaa"),
            "hohonu_qc_ok": 1.0 if context.hohonu_qc_ok else 0.0,
            "noaa_qc_ok": 1.0 if context.noaa_qc_ok else 0.0,
            "local_trend_abs": _scale(abs(float(context.recent_hohonu_trend_m_per_hour or 0.0)), 0.3),
            "local_volatility": _scale(_local_volatility(context), 0.25),
            "noaa_residual_abs": _scale(abs(float(context.recent_noaa_residual_m or 0.0)), 0.5),
            "noaa_residual_trend_abs": _scale(abs(float(context.noaa_residual_trend_m_per_hour or 0.0)), 0.3),
            "pressure_trend_abs": _scale(abs(float(context.pressure_trend or 0.0)), 5.0),
            "wind_speed_scaled": _scale(abs(float(context.wind_speed_mps or 0.0)), 25.0),
            "station_pair_scale": _scale(abs(float(context.station_pair.residual_scale)), 2.0),
            "station_pair_lag_hours": _scale(abs(float(context.station_pair.lag_minutes)) / 60.0, 6.0),
            "model_disagreement": _scale(abs(float(context.model_disagreement_m or 0.0)), 1.0),
            "recent_skill_mean": _recent_skill(context),
            "remaining_turns_fraction": _safe_div(
                state.remaining_turn_budget,
                state.budget.coordination_turn_limit,
            ),
            "remaining_deadline_fraction": _safe_div(state.remaining_deadline_ms, state.budget.deadline_ms),
            "recursion_depth": _safe_div(state.recursion_depth, max(1, state.budget.max_recursion_depth)),
            "difficulty_estimate": state.current_difficulty_estimate,
            "event_risk_estimate": state.current_event_risk_estimate,
        }
        features.update(_transcript_features(state))
        vector = np.array([float(features[name]) for name in self.schema.feature_names], dtype=float)
        vector[~np.isfinite(vector)] = 0.0
        return np.clip(vector, -5.0, 5.0)


def _transcript_features(state: CoordinationState) -> dict[str, float]:
    messages = state.full_message_transcript
    worker_forecasts = [
        message.structured_result["forecast"]
        for message in messages
        if message.role is Role.WORKER and message.structured_result.get("forecast")
    ]
    worker_values = [float(forecast["forecast_m"]) for forecast in worker_forecasts]
    interval_widths = [
        float(forecast["upper_m"]) - float(forecast["lower_m"])
        for forecast in worker_forecasts
    ]
    candidate = state.latest_candidate()
    last_verifier = state.verifier_findings[-1] if state.verifier_findings else {}
    last_thinker = _last_thinker_result(state)
    requested = " ".join(last_verifier.get("requested_evidence", []))
    recommended = " ".join(last_thinker.get("recommended_experts", []))
    return {
        "thinker_turns": _count_role(messages, Role.THINKER) / 5.0,
        "worker_turns": _count_role(messages, Role.WORKER) / 5.0,
        "verifier_turns": _count_role(messages, Role.VERIFIER) / 5.0,
        "unique_worker_count": _safe_div(len(state.unique_numerical_forecasters), state.budget.max_distinct_numerical_experts),
        "used_local_persistence": _used(messages, "local_persistence"),
        "used_local_tide": _used(messages, "local_tide"),
        "used_noaa_residual": _used(messages, "noaa_residual"),
        "used_regional_transfer": _used(messages, "regional_to_local_residual"),
        "used_synthesis": _used(messages, "ensemble_synthesis"),
        "used_fallback": _used(messages, "safe_fallback"),
        "candidate_available": 1.0 if candidate else 0.0,
        "candidate_value": 0.0 if candidate is None else _scale(candidate.forecast_m, 5.0),
        "candidate_interval_width": 0.0 if candidate is None else _scale(candidate.interval_width_m, 2.0),
        "candidate_confidence": 0.0 if candidate is None else candidate.confidence,
        "worker_prediction_mean": 0.0 if not worker_values else _scale(float(np.mean(worker_values)), 5.0),
        "worker_prediction_span": 0.0 if len(worker_values) < 2 else _scale(max(worker_values) - min(worker_values), 2.0),
        "worker_interval_mean": 0.0 if not interval_widths else _scale(float(np.mean(interval_widths)), 2.0),
        "last_verifier_accept": 1.0 if last_verifier.get("verdict") == "ACCEPT" else 0.0,
        "last_verifier_replan": 1.0 if last_verifier.get("verdict") == "REPLAN" else 0.0,
        "last_verifier_reject": 1.0 if last_verifier.get("verdict") == "REJECT" else 0.0,
        "last_verifier_continue": 1.0 if last_verifier.get("verdict") == "CONTINUE" else 0.0,
        "last_requested_regional": 1.0 if "regional" in requested else 0.0,
        "last_requested_synthesis": 1.0 if "synthesis" in requested else 0.0,
        "last_requested_fallback": 1.0 if "fallback" in requested else 0.0,
        "last_thinker_recommended_regional": 1.0 if "noaa_residual" in recommended or "regional_to_local" in recommended else 0.0,
        "last_thinker_recommended_local": 1.0 if "local_" in recommended else 0.0,
        "last_thinker_recommended_verifier_event": 1.0 if last_thinker.get("recommended_verifier_type") == "event_risk_verifier" else 0.0,
    }


def _last_thinker_result(state: CoordinationState) -> dict[str, Any]:
    for message in reversed(state.full_message_transcript):
        if message.role is Role.THINKER:
            return message.structured_result
    return {}


def _count_role(messages: list[Any], role: Role) -> int:
    return sum(1 for message in messages if message.role is role)


def _used(messages: list[Any], expert_id: str) -> float:
    return 1.0 if any(message.expert_id == expert_id for message in messages) else 0.0


def _local_volatility(context: Any) -> float:
    frame = context.recent_hohonu_observations
    if frame is None or len(frame) < 3:
        return 0.0
    values = frame["water_level_m"].astype(float).to_numpy()
    return float(np.nanstd(np.diff(values)))


def _recent_skill(context: Any) -> float:
    if not context.recent_model_performance:
        return 0.0
    values = [float(v) for v in context.recent_model_performance.values() if math.isfinite(float(v))]
    if not values:
        return 0.0
    return max(0.0, min(1.0, sum(values) / len(values)))


def _freshness(context: Any, source: str) -> float:
    value = float(context.observation_freshness_seconds.get(source, math.inf))
    if math.isinf(value):
        return 1.0
    return _scale(value / 3600.0, 12.0)


def _scale(value: float, scale: float) -> float:
    return max(0.0, min(1.0, float(value) / max(scale, 1e-6)))


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else max(0.0, min(1.0, float(num) / float(den)))


def _time_sin(value: int, period: int) -> float:
    return math.sin(2 * math.pi * value / period)


def _time_cos(value: int, period: int) -> float:
    return math.cos(2 * math.pi * value / period)
