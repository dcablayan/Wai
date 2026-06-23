"""Live-state Wai Ultra trajectory datasets."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Iterable

import pandas as pd

from src.evaluation.reward import interval_coverage, terminal_loss
from src.forecasting.pipeline import default_experts
from src.orchestration.action_masks import default_expert_specs, feasible_actions_for_state
from src.orchestration.bootstrap_policy import BootstrapCoordinatorPolicy
from src.orchestration.coordinator_head import action_key
from src.orchestration.coordinator_policy import CoordinatorPolicy
from src.orchestration.protocol import CoordinationAction, ExecutionBudget, Role
from src.orchestration.state_encoder import FeatureSchema, StateEncoder
from src.orchestration.ultra_conductor import UltraConductor
from src.orchestration.ultra_executor import UltraExecutor


FORBIDDEN_STATE_KEYS = ("actual", "error", "target_observation", "future", "oracle")


@dataclass
class TrajectoryTransition:
    """One live encoded state/action/result transition for coordinator training."""

    episode_id: str
    forecast_origin: str
    station_id: str
    randomized_condition: str
    turn: int
    encoded_state: list[float]
    state_feature_names: list[str]
    feature_schema_version: str
    available_action_mask: dict[str, bool]
    selected_action: str
    role: str
    expert: str
    subtask: str
    access_list: list[int]
    result_summary: dict[str, Any]
    immediate_cost: float
    terminal: bool
    final_forecast_error: float | None
    interval_coverage: bool | None
    peak_event_loss: float
    failure_or_abstention_status: str | None
    total_calls: int
    total_latency: float
    final_reward: float
    selected_action_feasible: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_coordination_trajectory_dataset(
    replay: pd.DataFrame,
    *,
    max_turns: int = 5,
    keep_alternatives: int = 3,
) -> pd.DataFrame:
    """Build live-schema trajectories from historical replay rows.

    Replay rows are converted back into origin-time context objects, then
    executed through the same Ultra state, encoder, action masks, and executor
    used online. ``keep_alternatives`` controls randomized-pool variants per
    origin in addition to the base episode.
    """

    contexts = []
    actuals = []
    for _, row in replay.iterrows():
        contexts.append(_context_from_replay_row(row))
        actuals.append(None if pd.isna(row.get("actual_m")) else float(row["actual_m"]))
    return build_live_coordination_trajectory_dataset(
        contexts,
        actuals,
        max_turns=max_turns,
        randomized_variants=max(0, keep_alternatives - 1),
    )


def build_live_coordination_trajectory_dataset(
    contexts: Iterable[Any],
    actuals_m: Iterable[float | None],
    *,
    max_turns: int = 5,
    policy: CoordinatorPolicy | None = None,
    randomized_variants: int = 0,
    budget: ExecutionBudget | None = None,
) -> pd.DataFrame:
    """Execute live Ultra states and store live-schema transition rows."""

    rows: list[TrajectoryTransition] = []
    base_policy = policy or BootstrapCoordinatorPolicy()
    for idx, (context, actual_m) in enumerate(zip(contexts, actuals_m)):
        variants = [("base", context)]
        for variant_idx, condition in enumerate(_randomized_conditions(randomized_variants)):
            variants.append((condition, _apply_randomized_condition(context, condition, variant_idx)))
        for condition, variant_context in variants:
            rows.extend(
                _run_episode(
                    variant_context,
                    actual_m,
                    episode_id=_episode_id(variant_context, idx, condition),
                    randomized_condition=condition,
                    policy=base_policy,
                    budget=budget or ExecutionBudget(max_turns=max_turns, max_coordination_turns=max_turns),
                )
            )
    dataset = pd.DataFrame([row.to_dict() for row in rows])
    audit_trajectory_dataset_for_leakage(dataset)
    return dataset


def audit_trajectory_dataset_for_leakage(dataset: pd.DataFrame) -> None:
    """Fail closed if policy-state fields contain future/label-like names."""

    for idx, row in dataset.iterrows():
        names = row.get("state_feature_names", [])
        if isinstance(names, str):
            names = json.loads(names)
        forbidden = [
            name
            for name in names
            if any(token in str(name).lower() for token in FORBIDDEN_STATE_KEYS)
        ]
        if forbidden:
            raise ValueError(f"row {idx}: policy state contains leakage-like features {forbidden}")


def trajectory_data_hash(dataset: pd.DataFrame) -> str:
    payload = dataset.to_json(orient="records", date_format="iso", default_handler=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_episode(
    context: Any,
    actual_m: float | None,
    *,
    episode_id: str,
    randomized_condition: str,
    policy: CoordinatorPolicy,
    budget: ExecutionBudget,
) -> list[TrajectoryTransition]:
    experts = default_experts(include_placeholders=False)
    conductor = UltraConductor(forecast_experts=experts, policy=policy, budget=budget)
    state = conductor.initialize_coordination_state(context)
    executor = UltraExecutor(experts)
    encoder = StateEncoder()
    specs = default_expert_specs()
    rows: list[TrajectoryTransition] = []
    started = time.perf_counter()

    while state.remaining_turn_budget > 0 and not state.timed_out:
        encoded = encoder.encode(state)
        state.encoded_state = encoded.tolist()
        feasible = feasible_actions_for_state(state, specs)
        action = policy.select_action(state, feasible)
        if action is None:
            break
        feasible_keys = {_action_key(item.action): item.feasible for item in feasible}
        message = executor.execute(action, state)
        state.append(action, message)

        terminal = _is_terminal(message) or state.remaining_turn_budget <= 0
        candidate = state.latest_candidate()
        error = None
        coverage = None
        peak_loss = 0.0
        if terminal and candidate is not None and actual_m is not None:
            error = candidate.forecast_m - actual_m
            coverage = interval_coverage(candidate.to_dict(), actual_m)
            peak_loss = abs(error) if abs(actual_m) >= 0.75 else 0.0
        latency = (time.perf_counter() - started) * 1000.0
        reward = -terminal_loss(
            forecast_m=None if candidate is None else candidate.forecast_m,
            lower_m=None if candidate is None else candidate.lower_m,
            upper_m=None if candidate is None else candidate.upper_m,
            actual_m=actual_m,
            total_calls=len(state.full_message_transcript),
            total_latency_ms=latency,
            failed=message.status.value in {"failed", "timeout"},
        )
        rows.append(
            TrajectoryTransition(
                episode_id=episode_id,
                forecast_origin=str(context.forecast_time_utc),
                station_id=str(context.target_station_id),
                randomized_condition=randomized_condition,
                turn=action.turn_id,
                encoded_state=encoded.tolist(),
                state_feature_names=list(FeatureSchema().feature_names),
                feature_schema_version=FeatureSchema().version,
                available_action_mask=feasible_keys,
                selected_action=_action_key(action),
                role=action.role.value,
                expert=action.expert_id,
                subtask=action.subtask_kind.value,
                access_list=list(action.access_list),
                result_summary=_message_summary(message),
                immediate_cost=float(action.expected_cost),
                terminal=terminal,
                final_forecast_error=error,
                interval_coverage=coverage,
                peak_event_loss=peak_loss,
                failure_or_abstention_status=None if message.status.value == "success" else message.status.value,
                total_calls=len(state.full_message_transcript),
                total_latency=latency,
                final_reward=reward,
                selected_action_feasible=bool(feasible_keys.get(_action_key(action), False)),
            )
        )
        if _is_terminal(message):
            break
    return rows


def _is_terminal(message) -> bool:
    verifier = message.structured_result.get("verifier", {})
    return verifier.get("verdict") == "ACCEPT"


def _message_summary(message) -> dict[str, Any]:
    result = {
        "status": message.status.value,
        "expert_id": message.expert_id,
        "visible_prior_turns": list(message.visible_prior_turns),
        "latency_ms": message.latency_ms,
    }
    forecast = message.structured_result.get("forecast")
    if forecast:
        result.update({
            "has_forecast": True,
            "confidence": forecast.get("confidence"),
            "interval_width": (
                None
                if forecast.get("lower_m") is None or forecast.get("upper_m") is None
                else float(forecast["upper_m"]) - float(forecast["lower_m"])
            ),
            "leaf_experts": forecast.get("leaf_experts", forecast.get("experts_used", [])),
        })
    verifier = message.structured_result.get("verifier")
    if verifier:
        result["verifier_verdict"] = verifier.get("verdict")
        result["requested_evidence"] = verifier.get("requested_evidence", [])
    return result


def _action_key(action: CoordinationAction) -> str:
    return action_key(
        action.role.value,
        action.expert_id,
        action.subtask_kind.value,
        action.control_decision.value,
    )


def _context_from_replay_row(row: pd.Series) -> Any:
    features = _parse_json(row["context_features"])
    expert_predictions = _parse_json(row.get("expert_predictions", "{}"))
    origin = pd.Timestamp(row["forecast_origin_utc"])
    if origin.tzinfo is None:
        origin = origin.tz_localize("UTC")
    target = pd.Timestamp(row["target_time_utc"])
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    station_id = str(row.get("target_station_id", "HOHONU_TEST"))
    noaa_id = str(row.get("paired_noaa_station_id", "NOAA_TEST"))
    local_tide = expert_predictions.get("local_tide", {})
    tide_level = local_tide.get("prediction_m", 0.0)
    latest_level = expert_predictions.get("local_persistence", {}).get("prediction_m", tide_level)
    noaa_residual = features.get("recent_noaa_residual_m")
    return SimpleNamespace(
        target_station_id=station_id,
        paired_noaa_station_id=noaa_id,
        forecast_time_utc=origin,
        target_time_utc=target,
        horizon_minutes=int(row.get("horizon_minutes", features.get("horizon_minutes", 360))),
        station_pair=SimpleNamespace(paired_noaa_station_id=noaa_id, residual_scale=1.0, lag_minutes=0),
        datum="MLLW",
        latest_hohonu_observation={"water_level_m": float(latest_level), "timestamp_utc": origin},
        latest_noaa_observation={"water_level_m": float(tide_level) + float(noaa_residual or 0.0), "timestamp_utc": origin},
        noaa_tide_prediction={"water_level_m": float(tide_level), "timestamp_utc": target, "source": "REPLAY"},
        local_tide_prediction=None,
        recent_hohonu_observations=pd.DataFrame(),
        recent_noaa_observations=pd.DataFrame(),
        noaa_tide_predictions=pd.DataFrame(),
        recent_hohonu_trend_m_per_hour=features.get("recent_hohonu_trend_m_per_hour"),
        recent_noaa_residual_m=noaa_residual,
        noaa_residual_trend_m_per_hour=features.get("noaa_residual_trend_m_per_hour"),
        observation_freshness_seconds={
            "hohonu": float(features.get("hohonu_freshness_seconds") or 0.0),
            "noaa": float(features.get("noaa_freshness_seconds") or 0.0),
        },
        qc_status={
            "hohonu": features.get("hohonu_qc_status", "pass"),
            "noaa": features.get("noaa_qc_status", "verified"),
        },
        tide_phase=features.get("tide_phase"),
        pressure_trend=None,
        wind_speed_mps=None,
        wind_direction_deg=None,
        neighboring_station_signals={},
        recent_model_performance={},
        model_disagreement_m=None,
        diagnostics={},
        hohonu_is_fresh=True,
        noaa_is_fresh=True,
        hohonu_qc_ok=features.get("hohonu_qc_status", "pass") in {"pass", "good", "verified", "ok"},
        noaa_qc_ok=features.get("noaa_qc_status", "verified") in {"pass", "good", "verified", "ok"},
    )


def _randomized_conditions(count: int) -> list[str]:
    conditions = [
        "worker_dropout",
        "missing_hohonu",
        "missing_noaa",
        "missing_tide_prediction",
        "stale_sources",
        "failed_qc",
        "worker_exception",
        "timeout",
        "invalid_interval",
        "disabled_synthesis",
        "disabled_verifier",
    ]
    return conditions[: max(0, count)]


def _apply_randomized_condition(context: Any, condition: str, variant_idx: int) -> Any:
    ctx = copy.copy(context)
    disabled = set(getattr(context, "disabled_experts", set()))
    ctx.observation_freshness_seconds = dict(getattr(context, "observation_freshness_seconds", {}) or {})
    ctx.qc_status = dict(getattr(context, "qc_status", {}) or {})
    ctx.forced_worker_exceptions = set(getattr(context, "forced_worker_exceptions", set()))
    ctx.forced_worker_timeouts = set(getattr(context, "forced_worker_timeouts", set()))
    ctx.forced_invalid_intervals = set(getattr(context, "forced_invalid_intervals", set()))
    if condition == "worker_dropout":
        disabled.add("noaa_residual" if variant_idx % 2 == 0 else "local_persistence")
    elif condition == "missing_hohonu":
        ctx.latest_hohonu_observation = None
        ctx.qc_status["hohonu"] = "missing"
        _set_if_mutable(ctx, "hohonu_qc_ok", False)
    elif condition == "missing_noaa":
        ctx.latest_noaa_observation = None
        ctx.recent_noaa_residual_m = None
        ctx.qc_status["noaa"] = "missing"
        _set_if_mutable(ctx, "noaa_qc_ok", False)
    elif condition == "missing_tide_prediction":
        ctx.noaa_tide_prediction = None
        ctx.local_tide_prediction = None
    elif condition == "stale_sources":
        ctx.observation_freshness_seconds = {"hohonu": 99_999.0, "noaa": 99_999.0}
        _set_if_mutable(ctx, "hohonu_is_fresh", False)
        _set_if_mutable(ctx, "noaa_is_fresh", False)
    elif condition == "failed_qc":
        ctx.qc_status = {"hohonu": "fail", "noaa": "fail"}
        _set_if_mutable(ctx, "hohonu_qc_ok", False)
        _set_if_mutable(ctx, "noaa_qc_ok", False)
    elif condition == "worker_exception":
        ctx.forced_worker_exceptions.add("noaa_residual")
    elif condition == "timeout":
        ctx.forced_worker_timeouts.add("regional_to_local_residual")
    elif condition == "invalid_interval":
        ctx.forced_invalid_intervals.add("local_tide")
    elif condition == "disabled_synthesis":
        disabled.add("ensemble_synthesis")
    elif condition == "disabled_verifier":
        disabled.add("event_risk_verifier")
    ctx.disabled_experts = disabled
    return ctx


def _set_if_mutable(target: Any, name: str, value: Any) -> None:
    try:
        setattr(target, name, value)
    except AttributeError:
        pass


def _parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return json.loads(value)


def _episode_id(context: Any, row_idx: int, condition: str) -> str:
    origin = str(context.forecast_time_utc)
    station = str(context.target_station_id)
    return hashlib.sha1(f"{station}:{origin}:{row_idx}:{condition}".encode("utf-8")).hexdigest()[:12]
