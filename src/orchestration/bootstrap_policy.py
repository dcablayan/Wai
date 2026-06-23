"""Transparent bootstrap policy for Wai Ultra's multi-turn protocol."""

from __future__ import annotations

from dataclasses import replace

from src.orchestration.coordination_state import CoordinationState
from src.orchestration.coordinator_policy import CoordinatorPolicy
from src.orchestration.protocol import CoordinationAction, Role, SubtaskKind
from src.orchestration.action_masks import FeasibleAction


class BootstrapCoordinatorPolicy(CoordinatorPolicy):
    """Heuristic one-action policy that uses Ultra state, roles, graph, and verifiers.

    This is deliberately separate from LearnedCoordinatorPolicy and reports
    policy_source="bootstrap"; it is not a trained coordinator.
    """

    policy_source = "bootstrap"
    artifact_version = "bootstrap-v1"

    def select_action(
        self,
        state: CoordinationState,
        feasible_actions: list[FeasibleAction],
    ) -> CoordinationAction | None:
        feasible = [item.action for item in feasible_actions if item.feasible]
        if not feasible:
            return None
        context = state.original_context

        requested = _last_requested_action(state)
        if requested:
            action = _find(feasible, expert_id=requested)
            if action is not None:
                return _with_reason(action, f"verifier requested {requested}")

        if state.next_turn_id == 0:
            if _is_easy(context):
                action = _choose_first_worker(feasible, context)
                if action:
                    return _with_reason(action, "easy origin: start with one numerical worker")
            action = _find(feasible, expert_id="regime_difficulty_thinker")
            if action:
                return _with_reason(action, "diagnose regime before spending worker budget")

        last = state.full_message_transcript[-1] if state.full_message_transcript else None
        if last is not None and last.role is Role.THINKER:
            recommended = last.structured_result.get("recommended_experts", [])
            for expert_id in recommended:
                action = _find(feasible, expert_id=expert_id)
                if action:
                    return _with_reason(action, "thinker recommendation selected")
            action = _choose_first_worker(feasible, context)
            if action:
                return _with_reason(action, "fallback worker after thinker")

        if last is not None and last.role is Role.WORKER:
            forecast = last.structured_result.get("forecast")
            if forecast and _needs_independent_worker(state):
                action = _next_independent_worker(feasible, state, context)
                if action:
                    return _with_reason(action, "candidate is difficult enough for independent worker evidence")
            if _should_synthesize(state):
                action = _find(feasible, expert_id="ensemble_synthesis")
                if action:
                    return _with_reason(action, "multiple allowed worker forecasts available for synthesis")
            verifier = _choose_verifier(feasible, state, context)
            if verifier:
                return _with_reason(verifier, "verify current candidate")

        if last is not None and last.role is Role.VERIFIER:
            verdict = last.structured_result.get("verifier", {}).get("verdict")
            if verdict == "ACCEPT":
                return None
            if verdict in {"REPLAN", "CONTINUE", "REJECT"}:
                action = _targeted_after_verifier(feasible, state, context, verdict)
                if action:
                    return _with_reason(action, f"verifier verdict {verdict} requested more work")

        if state.latest_candidate() is not None:
            verifier = _choose_verifier(feasible, state, context)
            if verifier:
                return _with_reason(verifier, "default verification path")

        action = _choose_first_worker(feasible, context) or _find(feasible, expert_id="safe_fallback")
        return _with_reason(action, "last feasible bootstrap action") if action else None


def _with_reason(action: CoordinationAction, reason: str) -> CoordinationAction:
    return replace(action, rationale_for_audit=reason)


def _find(
    actions: list[CoordinationAction],
    *,
    expert_id: str | None = None,
    role: Role | None = None,
    subtask: SubtaskKind | None = None,
) -> CoordinationAction | None:
    for action in actions:
        if expert_id is not None and action.expert_id != expert_id:
            continue
        if role is not None and action.role is not role:
            continue
        if subtask is not None and action.subtask_kind is not subtask:
            continue
        return action
    return None


def _is_easy(context) -> bool:
    residual = abs(float(context.recent_noaa_residual_m or 0.0))
    trend = abs(float(context.noaa_residual_trend_m_per_hour or 0.0))
    return (
        context.horizon_minutes <= 90
        and context.hohonu_qc_ok
        and context.hohonu_is_fresh
        and residual < 0.15
        and trend < 0.08
    )


def _choose_first_worker(actions: list[CoordinationAction], context) -> CoordinationAction | None:
    residual = abs(float(context.recent_noaa_residual_m or 0.0))
    if context.horizon_minutes <= 90:
        return _find(actions, expert_id="local_persistence") or _find(actions, expert_id="local_tide")
    if residual >= 0.25:
        return _find(actions, expert_id="noaa_residual") or _find(actions, expert_id="regional_to_local_residual")
    return _find(actions, expert_id="local_tide") or _find(actions, expert_id="noaa_residual")


def _needs_independent_worker(state: CoordinationState) -> bool:
    if state.current_event_risk_estimate >= 0.35 or state.current_difficulty_estimate >= 0.45:
        return len(state.unique_numerical_forecasters) < min(2, state.budget.max_distinct_numerical_experts)
    latest = state.latest_candidate()
    if latest is None:
        return False
    return latest.confidence < 0.55 and len(state.unique_numerical_forecasters) < 2


def _next_independent_worker(
    actions: list[CoordinationAction],
    state: CoordinationState,
    context,
) -> CoordinationAction | None:
    used = state.unique_numerical_forecasters
    preference = ["regional_to_local_residual", "noaa_residual", "local_tide", "local_persistence"]
    if abs(float(context.recent_noaa_residual_m or 0.0)) < 0.2:
        preference = ["local_tide", "noaa_residual", "regional_to_local_residual", "local_persistence"]
    for expert_id in preference:
        if expert_id not in used:
            action = _find(actions, expert_id=expert_id)
            if action:
                return action
    return None


def _should_synthesize(state: CoordinationState) -> bool:
    worker_candidates = [
        message
        for message in state.full_message_transcript
        if message.role is Role.WORKER
        and message.expert_id != "ensemble_synthesis"
        and message.structured_result.get("forecast")
    ]
    return len(worker_candidates) >= 2 and not any(
        message.expert_id == "ensemble_synthesis"
        for message in state.full_message_transcript
    )


def _choose_verifier(
    actions: list[CoordinationAction],
    state: CoordinationState,
    context,
) -> CoordinationAction | None:
    residual = abs(float(context.recent_noaa_residual_m or 0.0))
    if state.current_event_risk_estimate >= 0.35 or residual >= 0.25:
        return _find(actions, expert_id="event_risk_verifier") or _find(actions, expert_id="physics_datum_verifier")
    if _should_use_cross_source(state):
        return _find(actions, expert_id="cross_source_verifier") or _find(actions, expert_id="physics_datum_verifier")
    return _find(actions, expert_id="physics_datum_verifier") or _find(actions, expert_id="calibration_verifier")


def _should_use_cross_source(state: CoordinationState) -> bool:
    workers = {
        message.expert_id
        for message in state.full_message_transcript
        if message.role is Role.WORKER and message.structured_result.get("forecast")
    }
    return bool(workers & {"noaa_residual", "regional_to_local_residual"}) and bool(
        workers & {"local_tide", "local_persistence"}
    )


def _last_requested_action(state: CoordinationState) -> str | None:
    if not state.verifier_findings:
        return None
    requested = state.verifier_findings[-1].get("recommended_next_expert_or_verifier")
    return str(requested) if requested else None


def _targeted_after_verifier(
    actions: list[CoordinationAction],
    state: CoordinationState,
    context,
    verdict: str,
) -> CoordinationAction | None:
    if verdict == "REJECT":
        return _find(actions, expert_id="safe_fallback")
    action = _last_requested_action(state)
    if action:
        found = _find(actions, expert_id=action)
        if found:
            return found
    if _should_synthesize(state):
        return _find(actions, expert_id="ensemble_synthesis")
    return _next_independent_worker(actions, state, context) or _find(actions, expert_id="safe_fallback")
