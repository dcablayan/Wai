"""Multi-turn Wai Ultra conductor."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, replace
from typing import Any

from src.experts import ForecastExpert
from src.orchestration.bootstrap_policy import BootstrapCoordinatorPolicy
from src.orchestration.coordination_state import CoordinationState
from src.orchestration.coordinator_policy import CoordinatorPolicy
from src.orchestration.protocol import (
    CandidateForecast,
    CoordinationAction,
    CoordinationMessage,
    ExecutionBudget,
    MessageStatus,
    Role,
    SubtaskKind,
    VerifierVerdict,
)
from src.orchestration.action_masks import default_expert_specs, feasible_actions_for_state
from src.orchestration.state_encoder import StateEncoder
from src.orchestration.ultra_executor import UltraExecutor


@dataclass
class UltraRunResult:
    """Completed Ultra run returned to ForecastPipeline."""

    status: str
    candidate: CandidateForecast | None
    state: CoordinationState
    warnings: list[str]


class UltraConductor:
    """Learned/bootstrap state-action conductor inspired by public Fugu-like work."""

    def __init__(
        self,
        *,
        forecast_experts: dict[str, ForecastExpert],
        policy: CoordinatorPolicy | None = None,
        budget: ExecutionBudget | None = None,
        encoder: StateEncoder | None = None,
        include_placeholders: bool = False,
    ) -> None:
        self.forecast_experts = forecast_experts
        self.policy = policy or BootstrapCoordinatorPolicy()
        self.budget = budget or ExecutionBudget()
        self.encoder = encoder or StateEncoder()
        self.specs = default_expert_specs(include_placeholders=include_placeholders)
        self.executor = UltraExecutor(forecast_experts)

    def run(self, context: Any, *, recursion_depth: int = 0) -> UltraRunResult:
        state = self.initialize_coordination_state(context, recursion_depth=recursion_depth)
        warnings: list[str] = []
        state.encoded_state = self.encoder.encode(state).tolist()

        while state.remaining_turn_budget > 0:
            if state.timed_out:
                warnings.append("Ultra global deadline expired")
                break

            feasible_actions = feasible_actions_for_state(state, self.specs)
            action = self.policy.select_action(state, feasible_actions)
            if action is None:
                warnings.append("No feasible Ultra action remained")
                self._attempt_safe_fallback(state, warnings)
                break

            message = self.executor.execute(action, state)
            state.append(action, message)
            state.encoded_state = self.encoder.encode(state).tolist()
            warnings.extend(message.warnings)

            if self._is_verifier_acceptance(message):
                candidate = self._accepted_candidate_from_verifier(state, message)
                if candidate is not None:
                    state.accepted_candidate = candidate
                    state.completed_workflow_graph.mark_accepted(message.turn_id)
                    state.termination_reason = "verifier_acceptance"
                    return UltraRunResult("available", candidate, state, warnings)

            verifier = message.structured_result.get("verifier", {})
            if verifier.get("safe_fallback_required"):
                self._attempt_safe_fallback(state, warnings)
                break

            if verifier.get("verdict") == VerifierVerdict.REPLAN.value:
                if not self._enter_bounded_replan(state, warnings):
                    self._attempt_safe_fallback(state, warnings)
                    break
                if state.accepted_candidate is not None:
                    return UltraRunResult("available", state.accepted_candidate, state, warnings)

        if state.accepted_candidate is not None:
            return UltraRunResult("available", state.accepted_candidate, state, warnings)

        if state.termination_reason is None:
            state.termination_reason = "budget_or_deadline_exhausted"
        if not state.fallback_attempted:
            self._attempt_safe_fallback(state, warnings)
            if state.accepted_candidate is not None:
                return UltraRunResult("available", state.accepted_candidate, state, warnings)

        return UltraRunResult("unavailable", None, state, warnings)

    def initialize_coordination_state(self, context: Any, *, recursion_depth: int = 0) -> CoordinationState:
        now = time.monotonic()
        capability_masks = self._initial_capability_masks(context)
        state = CoordinationState(
            original_context=context,
            available_expert_pool=sorted(self.specs),
            capability_masks=capability_masks,
            budget=self.budget,
            started_monotonic=now,
            deadline_monotonic=now + self.budget.deadline_ms / 1000.0,
            recursion_depth=recursion_depth,
            coordinator_policy_source=self.policy.policy_source,
            coordinator_artifact_version=self.policy.artifact_version,
        )
        return state

    def _initial_capability_masks(self, context: Any) -> dict[str, bool]:
        masks = {}
        shell_budget = ExecutionBudget(max_turns=1, deadline_ms=max(100.0, self.budget.deadline_ms))
        now = time.monotonic()
        shell = CoordinationState(
            original_context=context,
            available_expert_pool=sorted(self.specs),
            capability_masks={},
            budget=shell_budget,
            started_monotonic=now,
            deadline_monotonic=now + shell_budget.deadline_ms / 1000.0,
        )
        for item in feasible_actions_for_state(shell, self.specs):
            masks[item.action.expert_id] = masks.get(item.action.expert_id, False) or item.feasible
        return masks

    def _attempt_safe_fallback(self, state: CoordinationState, warnings: list[str]) -> None:
        if state.fallback_attempted or state.timed_out:
            if state.timed_out:
                state.termination_reason = "deadline_expired_before_fallback"
            return
        if state.remaining_turn_budget <= 0:
            state.termination_reason = "turn_budget_exhausted_before_fallback"
            return
        fallback = CoordinationAction(
            turn_id=state.next_turn_id,
            expert_id="safe_fallback",
            role=Role.WORKER,
            subtask_kind=SubtaskKind.FORECAST_LOCAL_LEVEL,
            access_list=[],
            rationale_for_audit="reserved safe fallback attempt",
        )
        message = self.executor.execute(fallback, state)
        state.append(fallback, message)
        warnings.extend(message.warnings)
        state.encoded_state = self.encoder.encode(state).tolist()
        if message.status not in {MessageStatus.SUCCESS, MessageStatus.REUSED}:
            state.termination_reason = "safe_fallback_unavailable"
            return
        if state.remaining_turn_budget <= 0:
            state.termination_reason = "safe_fallback_unverified"
            return
        verifier = CoordinationAction(
            turn_id=state.next_turn_id,
            expert_id="physics_datum_verifier",
            role=Role.VERIFIER,
            subtask_kind=SubtaskKind.VERIFY_PHYSICS,
            access_list=[message.turn_id],
            rationale_for_audit="independent verification of safe fallback",
        )
        verifier_message = self.executor.execute(verifier, state)
        state.append(verifier, verifier_message)
        warnings.extend(verifier_message.warnings)
        state.encoded_state = self.encoder.encode(state).tolist()
        if self._is_verifier_acceptance(verifier_message):
            state.accepted_candidate = self._accepted_candidate_from_verifier(state, verifier_message)
            state.completed_workflow_graph.mark_accepted(verifier_message.turn_id)
            state.termination_reason = "verified_safe_fallback"
        else:
            state.termination_reason = "safe_fallback_rejected"

    def _enter_bounded_replan(self, state: CoordinationState, warnings: list[str]) -> bool:
        if state.recursion_depth >= state.budget.max_recursion_depth:
            warnings.append("Verifier requested replan but recursion depth limit was reached")
            return False
        if state.remaining_turn_budget <= state.budget.reserved_fallback_turns:
            warnings.append("Verifier requested replan but only fallback budget remains")
            return False
        if _last_verifier_requested_fallback(state):
            warnings.append("Recursive fallback is not allowed")
            return False
        child_turns = max(0, state.remaining_turn_budget - state.budget.reserved_fallback_turns)
        if child_turns <= 0:
            warnings.append("Verifier requested replan but no child workflow turns remain")
            return False
        parent_verifier_turn = state.full_message_transcript[-1].turn_id if state.full_message_transcript else -1
        child_context = _child_replan_context(state)
        child_budget = replace(
            state.budget,
            max_turns=child_turns,
            max_coordination_turns=child_turns,
            max_recursion_depth=state.recursion_depth,
            deadline_ms=max(1.0, state.remaining_deadline_ms),
            reserved_fallback_worker_calls=0,
            reserved_fallback_verifier_turns=0,
        )
        child_state = CoordinationState(
            original_context=child_context,
            available_expert_pool=state.available_expert_pool,
            capability_masks=state.capability_masks,
            budget=child_budget,
            started_monotonic=state.started_monotonic,
            deadline_monotonic=state.deadline_monotonic,
            recursion_depth=state.recursion_depth + 1,
            coordinator_policy_source=state.coordinator_policy_source,
            coordinator_artifact_version=state.coordinator_artifact_version,
        )
        child_state.physical_forecast_cache = dict(state.physical_forecast_cache)
        child_accepted_turn: int | None = None
        while child_state.remaining_turn_budget > 0 and not child_state.timed_out:
            feasible_actions = feasible_actions_for_state(child_state, self.specs)
            action = self.policy.select_action(child_state, feasible_actions)
            if action is None:
                break
            if action.expert_id == "safe_fallback":
                break
            message = self.executor.execute(action, child_state)
            child_state.append(action, message)
            child_state.encoded_state = self.encoder.encode(child_state).tolist()
            warnings.extend(message.warnings)
            if self._is_verifier_acceptance(message):
                child_accepted_turn = message.turn_id
                child_state.completed_workflow_graph.mark_accepted(message.turn_id)
                break
            verifier = message.structured_result.get("verifier", {})
            if verifier.get("verdict") in {VerifierVerdict.REPLAN.value, VerifierVerdict.REJECT.value}:
                break
            if verifier.get("safe_fallback_required"):
                break
        if not child_state.full_message_transcript:
            warnings.append("Verifier requested replan but child workflow found no valid action")
            return False
        remapped_accept_turn = _merge_child_transcript_into_parent(state, child_state)
        state.completed_workflow_graph.add_child_workflow(
            parent_verifier_turn_id=parent_verifier_turn,
            child_depth=child_state.recursion_depth,
            child_graph=child_state.topology_dict(),
        )
        state.recursion_depth = max(state.recursion_depth, child_state.recursion_depth)
        if child_accepted_turn is not None and remapped_accept_turn is not None:
            accepted_message = state.full_message_transcript[remapped_accept_turn]
            candidate = self._accepted_candidate_from_verifier(state, accepted_message)
            if candidate is not None:
                state.accepted_candidate = candidate
                state.completed_workflow_graph.mark_accepted(accepted_message.turn_id)
                state.termination_reason = "child_replan_verifier_acceptance"
        return True

    def _is_verifier_acceptance(self, message: CoordinationMessage) -> bool:
        verifier = message.structured_result.get("verifier", {})
        return message.role is Role.VERIFIER and verifier.get("verdict") == VerifierVerdict.ACCEPT.value

    def _accepted_candidate_from_verifier(
        self,
        state: CoordinationState,
        verifier_message: CoordinationMessage,
    ) -> CandidateForecast | None:
        if not verifier_message.visible_prior_turns:
            return None
        candidate_turn = verifier_message.visible_prior_turns[-1]
        candidate = state.current_candidate_forecasts.get(candidate_turn)
        if candidate is None:
            return None
        verifier = verifier_message.structured_result.get("verifier", {})
        confidence = max(
            0.0,
            min(1.0, candidate.confidence + float(verifier.get("confidence_adjustment", 0.0))),
        )
        multiplier = max(1.0, float(verifier.get("interval_adjustment_recommendation", 1.0)))
        half_width = max(candidate.forecast_m - candidate.lower_m, candidate.upper_m - candidate.forecast_m)
        adjusted = replace(
            candidate,
            lower_m=float(candidate.forecast_m - half_width * multiplier),
            upper_m=float(candidate.forecast_m + half_width * multiplier),
            confidence=confidence,
            verifier_adjustments={
                "candidate_turn_verified": candidate_turn,
                "verifier_turn": verifier_message.turn_id,
                "pre_confidence": candidate.confidence,
                "post_confidence": confidence,
                "pre_interval": [candidate.lower_m, candidate.upper_m],
                "post_interval": [
                    float(candidate.forecast_m - half_width * multiplier),
                    float(candidate.forecast_m + half_width * multiplier),
                ],
                "evidence_used": list(verifier_message.visible_prior_turns),
                "verifier": verifier,
            },
        )
        state.current_candidate_forecasts[verifier_message.turn_id] = adjusted
        return adjusted


def _last_verifier_requested_fallback(state: CoordinationState) -> bool:
    if not state.verifier_findings:
        return False
    latest = state.verifier_findings[-1]
    return bool(latest.get("safe_fallback_required")) or latest.get("recommended_next_expert_or_verifier") == "safe_fallback"


def _child_replan_context(state: CoordinationState) -> Any:
    context = copy.copy(state.original_context)
    diagnostics = dict(getattr(context, "diagnostics", {}) or {})
    latest_verifier = state.verifier_findings[-1] if state.verifier_findings else {}
    candidate = state.latest_candidate()
    diagnostics["ultra_replan_request"] = {
        "parent_verifier_turn": latest_verifier.get("turn_id"),
        "verdict": latest_verifier.get("verdict"),
        "requested_evidence": list(latest_verifier.get("requested_evidence", [])),
        "candidate": None if candidate is None else candidate.to_dict(),
    }
    context.diagnostics = diagnostics
    disabled = set(getattr(context, "disabled_experts", set()))
    disabled.add("safe_fallback")
    context.disabled_experts = disabled
    return context


def _merge_child_transcript_into_parent(
    parent: CoordinationState,
    child: CoordinationState,
) -> int | None:
    base_turn = parent.next_turn_id
    accepted_child_turn = child.completed_workflow_graph.final_accepted_node
    remapped_accept_turn = None
    for action, message in zip(child.action_transcript, child.full_message_transcript):
        remapped_action = replace(
            action,
            turn_id=base_turn + action.turn_id,
            access_list=[base_turn + turn for turn in action.access_list],
        )
        remapped_message = CoordinationMessage(
            turn_id=base_turn + message.turn_id,
            expert_id=message.expert_id,
            role=message.role,
            subtask_kind=message.subtask_kind,
            visible_prior_turns=[base_turn + turn for turn in message.visible_prior_turns],
            status=message.status,
            structured_result=copy.deepcopy(message.structured_result),
            latency_ms=message.latency_ms,
            warnings=list(message.warnings),
        )
        parent.append(remapped_action, remapped_message)
        if accepted_child_turn == message.turn_id:
            remapped_accept_turn = remapped_message.turn_id
    parent.physical_forecast_cache.update(child.physical_forecast_cache)
    return remapped_accept_turn
