"""Multi-turn Wai Ultra conductor."""

from __future__ import annotations

import time
from dataclasses import dataclass
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
                candidate = state.latest_candidate()
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
            state.accepted_candidate = state.latest_candidate()
            state.completed_workflow_graph.mark_accepted(verifier_message.turn_id)
            state.termination_reason = "verified_safe_fallback"
        else:
            state.termination_reason = "safe_fallback_rejected"

    def _enter_bounded_replan(self, state: CoordinationState, warnings: list[str]) -> bool:
        if state.recursion_depth >= state.budget.max_recursion_depth:
            warnings.append("Verifier requested replan but recursion depth limit was reached")
            return False
        if state.remaining_turn_budget <= state.budget.reserved_safe_fallback_calls:
            warnings.append("Verifier requested replan but only fallback budget remains")
            return False
        if _last_verifier_requested_fallback(state):
            warnings.append("Recursive fallback is not allowed")
            return False
        state.recursion_depth += 1
        return True

    def _is_verifier_acceptance(self, message: CoordinationMessage) -> bool:
        verifier = message.structured_result.get("verifier", {})
        return message.role is Role.VERIFIER and verifier.get("verdict") == VerifierVerdict.ACCEPT.value


def _last_verifier_requested_fallback(state: CoordinationState) -> bool:
    if not state.verifier_findings:
        return False
    latest = state.verifier_findings[-1]
    return bool(latest.get("safe_fallback_required")) or latest.get("recommended_next_expert_or_verifier") == "safe_fallback"
