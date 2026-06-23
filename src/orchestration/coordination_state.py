"""Mutable state for the Wai Ultra multi-turn conductor."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.orchestration.protocol import (
    CandidateForecast,
    CoordinationAction,
    CoordinationMessage,
    CoordinationTelemetry,
    ExecutionBudget,
    MessageStatus,
    Role,
)
from src.orchestration.workflow_graph import WorkflowGraph


@dataclass
class CoordinationState:
    """Complete origin-time state, transcript, workflow graph, and budgets."""

    original_context: Any
    available_expert_pool: list[str]
    capability_masks: dict[str, bool]
    budget: ExecutionBudget
    started_monotonic: float
    deadline_monotonic: float
    full_message_transcript: list[CoordinationMessage] = field(default_factory=list)
    action_transcript: list[CoordinationAction] = field(default_factory=list)
    completed_workflow_graph: WorkflowGraph = field(default_factory=WorkflowGraph)
    current_candidate_forecasts: dict[int, CandidateForecast] = field(default_factory=dict)
    verifier_findings: list[dict[str, Any]] = field(default_factory=list)
    remaining_turn_budget: int = 0
    remaining_call_budget: int = 0
    recursion_depth: int = 0
    current_difficulty_estimate: float = 0.0
    current_event_risk_estimate: float = 0.0
    encoded_state: list[float] = field(default_factory=list)
    physical_forecast_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    fallback_attempted: bool = False
    termination_reason: str | None = None
    accepted_candidate: CandidateForecast | None = None
    coordinator_policy_source: str = "bootstrap"
    coordinator_artifact_version: str = "bootstrap"

    def __post_init__(self) -> None:
        if self.remaining_turn_budget <= 0:
            self.remaining_turn_budget = self.budget.max_turns
        if self.remaining_call_budget <= 0:
            self.remaining_call_budget = self.budget.max_turns

    @property
    def remaining_deadline_ms(self) -> float:
        return max(0.0, (self.deadline_monotonic - time.monotonic()) * 1000.0)

    @property
    def timed_out(self) -> bool:
        return self.remaining_deadline_ms <= 0.0

    @property
    def next_turn_id(self) -> int:
        return len(self.full_message_transcript)

    @property
    def unique_experts(self) -> set[str]:
        return {message.expert_id for message in self.full_message_transcript}

    @property
    def unique_numerical_forecasters(self) -> set[str]:
        excluded = {
            "ensemble_synthesis",
            "regime_difficulty_thinker",
            "residual_dynamics_thinker",
            "physics_datum_verifier",
            "cross_source_verifier",
            "calibration_verifier",
            "event_risk_verifier",
        }
        return {
            message.expert_id
            for message in self.full_message_transcript
            if message.role is Role.WORKER and message.expert_id not in excluded
        }

    def visible_messages(self, access_list: list[int]) -> list[CoordinationMessage]:
        by_turn = {message.turn_id: message for message in self.full_message_transcript}
        missing = [turn for turn in access_list if turn not in by_turn]
        if missing:
            raise ValueError(f"Action requested unavailable prior turns: {missing}")
        return [by_turn[turn] for turn in access_list]

    def append(self, action: CoordinationAction, message: CoordinationMessage) -> None:
        if action.turn_id != self.next_turn_id:
            raise ValueError(
                f"Action turn_id {action.turn_id} does not match next turn {self.next_turn_id}"
            )
        self.action_transcript.append(action)
        self.full_message_transcript.append(message)
        self.completed_workflow_graph.add_turn(action, message)
        self.remaining_turn_budget = max(0, self.budget.max_turns - len(self.full_message_transcript))
        self.remaining_call_budget = max(0, self.budget.max_turns - len(self.full_message_transcript))

        if action.role is Role.THINKER and message.status in {MessageStatus.SUCCESS, MessageStatus.REUSED}:
            result = message.structured_result
            self.current_difficulty_estimate = float(
                result.get("forecast_difficulty", self.current_difficulty_estimate)
            )
            self.current_event_risk_estimate = float(
                result.get("event_risk", self.current_event_risk_estimate)
            )

        forecast = message.structured_result.get("forecast")
        if forecast and message.status in {MessageStatus.SUCCESS, MessageStatus.REUSED}:
            self.current_candidate_forecasts[action.turn_id] = CandidateForecast(
                forecast_m=float(forecast["forecast_m"]),
                lower_m=float(forecast["lower_m"]),
                upper_m=float(forecast["upper_m"]),
                confidence=float(forecast["confidence"]),
                experts_used=list(forecast.get("experts_used", [action.expert_id])),
                method=str(forecast.get("method", action.expert_id)),
                source_turn_id=action.turn_id,
                diagnostics=dict(forecast.get("diagnostics", {})),
            )

        verifier = message.structured_result.get("verifier")
        if verifier:
            self.verifier_findings.append({
                "turn_id": action.turn_id,
                **verifier,
            })

    def latest_candidate(self) -> CandidateForecast | None:
        if not self.current_candidate_forecasts:
            return None
        latest_turn = max(self.current_candidate_forecasts)
        return self.current_candidate_forecasts[latest_turn]

    def telemetry(self) -> CoordinationTelemetry:
        physical = sum(
            1
            for message in self.full_message_transcript
            if message.role is Role.WORKER
            and message.expert_id not in {"ensemble_synthesis"}
            and not message.structured_result.get("reused", False)
        )
        reused = sum(1 for message in self.full_message_transcript if message.structured_result.get("reused", False))
        fallback = sum(1 for message in self.full_message_transcript if message.expert_id == "safe_fallback")
        return CoordinationTelemetry(
            logical_actions=len(self.full_message_transcript),
            physical_expert_calls=physical,
            reused_expert_outputs=reused,
            unique_experts=len(self.unique_experts),
            verifier_calls=sum(1 for message in self.full_message_transcript if message.role is Role.VERIFIER),
            thinker_calls=sum(1 for message in self.full_message_transcript if message.role is Role.THINKER),
            worker_calls=sum(1 for message in self.full_message_transcript if message.role is Role.WORKER),
            fallback_calls=fallback,
        )

    def topology_dict(self) -> dict[str, Any]:
        return self.completed_workflow_graph.to_dict()
