"""Expert registry and authoritative action masks for Wai Ultra."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from src.orchestration.coordination_state import CoordinationState
from src.orchestration.protocol import CoordinationAction, ControlDecision, Role, SubtaskKind


@dataclass(frozen=True)
class ExpertSpec:
    """Capability declaration for one Ultra callable component."""

    expert_id: str
    role: Role
    subtasks: tuple[SubtaskKind, ...]
    required_sources: tuple[str, ...] = ()
    thread_safe: bool = True
    placeholder: bool = False
    max_horizon_minutes: int | None = None
    typical_latency_ms: float = 10.0
    checker: Callable[[object], tuple[bool, str]] | None = None


@dataclass(frozen=True)
class FeasibleAction:
    """A masked action candidate exposed to a coordinator policy."""

    action: CoordinationAction
    feasible: bool
    reason: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


def default_expert_specs(include_placeholders: bool = False) -> dict[str, ExpertSpec]:
    """Return the Ultra expert registry derived from numerical expert capabilities."""

    specs = {
        "regime_difficulty_thinker": ExpertSpec(
            "regime_difficulty_thinker",
            Role.THINKER,
            (SubtaskKind.DIAGNOSE_REGIME, SubtaskKind.ESTIMATE_DIFFICULTY),
            typical_latency_ms=2.0,
        ),
        "residual_dynamics_thinker": ExpertSpec(
            "residual_dynamics_thinker",
            Role.THINKER,
            (SubtaskKind.ANALYZE_RESIDUAL_DYNAMICS,),
            required_sources=("noaa_observation", "tide_prediction"),
            typical_latency_ms=2.0,
            checker=_requires_noaa_residual_context,
        ),
        "local_persistence": ExpertSpec(
            "local_persistence",
            Role.WORKER,
            (SubtaskKind.FORECAST_LOCAL_LEVEL,),
            required_sources=("hohonu_observation",),
            max_horizon_minutes=12 * 60,
            typical_latency_ms=8.0,
            checker=_requires_fresh_local,
        ),
        "local_tide": ExpertSpec(
            "local_tide",
            Role.WORKER,
            (SubtaskKind.FORECAST_LOCAL_LEVEL,),
            required_sources=("tide_prediction",),
            typical_latency_ms=5.0,
            checker=_requires_tide,
        ),
        "noaa_residual": ExpertSpec(
            "noaa_residual",
            Role.WORKER,
            (SubtaskKind.FORECAST_REGIONAL_RESIDUAL,),
            required_sources=("noaa_observation", "tide_prediction"),
            typical_latency_ms=10.0,
            checker=_requires_fresh_noaa_residual,
        ),
        "regional_to_local_residual": ExpertSpec(
            "regional_to_local_residual",
            Role.WORKER,
            (SubtaskKind.TRANSFER_REGIONAL_SIGNAL,),
            required_sources=("noaa_observation", "tide_prediction", "station_pair"),
            typical_latency_ms=12.0,
            checker=_requires_fresh_noaa_residual,
        ),
        "safe_fallback": ExpertSpec(
            "safe_fallback",
            Role.WORKER,
            (SubtaskKind.FORECAST_LOCAL_LEVEL,),
            required_sources=("tide_prediction",),
            typical_latency_ms=4.0,
            checker=_requires_tide,
        ),
        "ensemble_synthesis": ExpertSpec(
            "ensemble_synthesis",
            Role.WORKER,
            (SubtaskKind.SYNTHESIZE_FORECASTS, SubtaskKind.ESTIMATE_UNCERTAINTY),
            typical_latency_ms=6.0,
            checker=_requires_worker_candidate,
        ),
        "physics_datum_verifier": ExpertSpec(
            "physics_datum_verifier",
            Role.VERIFIER,
            (SubtaskKind.VERIFY_PHYSICS,),
            typical_latency_ms=4.0,
            checker=_requires_candidate,
        ),
        "cross_source_verifier": ExpertSpec(
            "cross_source_verifier",
            Role.VERIFIER,
            (SubtaskKind.VERIFY_SOURCE_CONSISTENCY,),
            typical_latency_ms=4.0,
            checker=_requires_candidate,
        ),
        "calibration_verifier": ExpertSpec(
            "calibration_verifier",
            Role.VERIFIER,
            (SubtaskKind.VERIFY_CALIBRATION,),
            typical_latency_ms=4.0,
            checker=_requires_candidate,
        ),
        "event_risk_verifier": ExpertSpec(
            "event_risk_verifier",
            Role.VERIFIER,
            (SubtaskKind.VERIFY_EVENT_RISK,),
            typical_latency_ms=4.0,
            checker=_requires_candidate,
        ),
    }
    placeholders = {
        "weather_aware": ExpertSpec(
            "weather_aware",
            Role.WORKER,
            (SubtaskKind.FORECAST_REGIONAL_RESIDUAL,),
            required_sources=("weather",),
            placeholder=True,
            checker=lambda context: (False, "weather-aware expert is not implemented"),
        ),
        "spatial_neighboring_station": ExpertSpec(
            "spatial_neighboring_station",
            Role.WORKER,
            (SubtaskKind.ANALYZE_SPATIAL_PROPAGATION,),
            required_sources=("neighbor_station",),
            placeholder=True,
            checker=lambda context: (False, "neighboring-station expert is not implemented"),
        ),
        "learned_local_residual": ExpertSpec(
            "learned_local_residual",
            Role.WORKER,
            (SubtaskKind.FORECAST_LOCAL_LEVEL,),
            placeholder=True,
            checker=lambda context: (False, "learned local residual expert is not trained"),
        ),
    }
    if include_placeholders:
        specs.update(placeholders)
    return specs


def feasible_actions_for_state(
    state: CoordinationState,
    specs: dict[str, ExpertSpec],
) -> list[FeasibleAction]:
    """Apply safety, capability, budget, and access masks before policy scoring."""

    candidates: list[FeasibleAction] = []
    if state.timed_out:
        return candidates
    if state.remaining_turn_budget <= 0:
        return candidates

    for spec in specs.values():
        for subtask in spec.subtasks:
            access_list = _default_access_list(state, spec, subtask)
            action = CoordinationAction(
                turn_id=state.next_turn_id,
                expert_id=spec.expert_id,
                role=spec.role,
                subtask_kind=subtask,
                access_list=access_list,
                expected_cost=max(0.1, spec.typical_latency_ms / 10.0),
                control_decision=_control_for(spec, subtask),
            )
            ok, reason = _is_feasible(state, spec, action)
            candidates.append(FeasibleAction(action=action, feasible=ok, reason=reason))
    return candidates


def _is_feasible(
    state: CoordinationState,
    spec: ExpertSpec,
    action: CoordinationAction,
) -> tuple[bool, str]:
    context = state.original_context
    if _only_fallback_resources_remain(state) and not _is_reserved_fallback_action(state, action):
        return False, "only reserved fallback resources remain"
    if action.expert_id in set(getattr(context, "disabled_experts", set())):
        return False, f"{action.expert_id} disabled for this randomized episode"
    if spec.placeholder:
        return False, "placeholder capability is unavailable"
    if spec.max_horizon_minutes is not None and context.horizon_minutes > spec.max_horizon_minutes:
        return False, f"horizon exceeds {spec.max_horizon_minutes} minute support"
    if spec.checker is not None:
        ok, reason = spec.checker(context)
        if not ok:
            return False, reason
    if action.expert_id == "safe_fallback" and state.fallback_attempted:
        return False, "safe fallback already attempted"
    if action.role is Role.WORKER and action.expert_id not in {"ensemble_synthesis", "safe_fallback"}:
        if state.remaining_physical_worker_calls <= state.budget.reserved_fallback_worker_calls:
            return False, "physical worker budget reserved for fallback"
        already = state.unique_numerical_forecasters
        would_add = action.expert_id not in already
        if would_add and len(already) >= state.budget.max_distinct_numerical_experts:
            return False, "maximum distinct numerical expert budget reached"
    if action.expert_id == "ensemble_synthesis":
        worker_turns = _successful_base_worker_turns(state)
        if len(worker_turns) < 2:
            return False, "synthesis requires at least two distinct successful base worker outputs"
    if action.role is Role.VERIFIER and state.latest_candidate() is None:
        return False, "verifier requires a current candidate forecast"
    if action.role is Role.VERIFIER and state.remaining_verifier_calls <= 0:
        return False, "verifier budget exhausted"
    if state.remaining_deadline_ms < min(2.0, spec.typical_latency_ms):
        return False, "remaining deadline is too small for action"
    return True, ""


def _default_access_list(
    state: CoordinationState,
    spec: ExpertSpec,
    subtask: SubtaskKind,
) -> list[int]:
    if spec.role is Role.THINKER:
        return []
    if spec.expert_id == "ensemble_synthesis":
        return _successful_base_worker_turns(state)
    if spec.role is Role.VERIFIER:
        latest = state.latest_candidate()
        if latest is None:
            return []
        input_turns = [turn for turn in latest.input_turn_ids if turn != latest.source_turn_id]
        return [*input_turns, latest.source_turn_id]
    if state.verifier_findings:
        requested = state.verifier_findings[-1]
        if requested.get("recommended_next_expert_or_verifier") == spec.expert_id:
            return [int(requested["turn_id"])]
    if state.full_message_transcript and state.full_message_transcript[-1].role is Role.THINKER:
        return [state.full_message_transcript[-1].turn_id]
    return []


def _successful_worker_turns(state: CoordinationState) -> list[int]:
    return [
        message.turn_id
        for message in state.full_message_transcript
        if message.role is Role.WORKER
        and "forecast" in message.structured_result
        and message.structured_result.get("forecast", {}).get("forecast_m") is not None
    ]


def _successful_base_worker_turns(state: CoordinationState) -> list[int]:
    turns = []
    leaf_seen: set[str] = set()
    for message in state.full_message_transcript:
        if message.role is not Role.WORKER or message.expert_id == "ensemble_synthesis":
            continue
        forecast = message.structured_result.get("forecast")
        if not forecast:
            continue
        leaves = forecast.get("leaf_experts", forecast.get("experts_used", [message.expert_id]))
        if "safe_fallback" in leaves:
            continue
        if any(leaf in leaf_seen for leaf in leaves):
            continue
        leaf_seen.update(leaves)
        turns.append(message.turn_id)
    return turns


def _only_fallback_resources_remain(state: CoordinationState) -> bool:
    return state.remaining_turn_budget <= state.budget.reserved_fallback_turns


def _is_reserved_fallback_action(state: CoordinationState, action: CoordinationAction) -> bool:
    if action.expert_id == "safe_fallback":
        return True
    if action.expert_id != "physics_datum_verifier":
        return False
    candidate = state.latest_candidate()
    if candidate is None:
        return False
    return "safe_fallback" in set(candidate.leaf_experts or candidate.experts_used)


def _control_for(spec: ExpertSpec, subtask: SubtaskKind) -> ControlDecision:
    if spec.expert_id == "safe_fallback":
        return ControlDecision.FALLBACK
    if spec.role is Role.VERIFIER:
        return ControlDecision.ACCEPT
    return ControlDecision.CONTINUE


def _requires_tide(context) -> tuple[bool, str]:
    if context.local_tide_prediction is None and context.noaa_tide_prediction is None:
        return False, "tide prediction is missing"
    return True, ""


def _requires_fresh_local(context) -> tuple[bool, str]:
    if context.latest_hohonu_observation is None:
        return False, "latest Hohonu observation is missing"
    if not context.hohonu_qc_ok:
        return False, "latest Hohonu observation failed QC"
    if not context.hohonu_is_fresh:
        return False, "latest Hohonu observation is stale"
    return True, ""


def _requires_noaa_residual_context(context) -> tuple[bool, str]:
    if context.noaa_tide_prediction is None:
        return False, "NOAA tide prediction is missing"
    if context.recent_noaa_residual_m is None:
        return False, "recent NOAA residual is missing"
    if not context.noaa_qc_ok:
        return False, "latest NOAA observation failed QC"
    return True, ""


def _requires_fresh_noaa_residual(context) -> tuple[bool, str]:
    ok, reason = _requires_noaa_residual_context(context)
    if not ok:
        return ok, reason
    if not context.noaa_is_fresh:
        return False, "latest NOAA observation is stale"
    return True, ""


def _requires_candidate(context) -> tuple[bool, str]:
    return True, ""


def _requires_worker_candidate(context) -> tuple[bool, str]:
    return True, ""
