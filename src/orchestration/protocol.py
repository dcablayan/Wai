"""Typed coordination protocol for Wai Ultra."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Coordination role assigned to an expert on one turn."""

    THINKER = "THINKER"
    WORKER = "WORKER"
    VERIFIER = "VERIFIER"


class ControlDecision(str, Enum):
    """High-level control choice available to the coordinator."""

    CONTINUE = "CONTINUE"
    ACCEPT = "ACCEPT"
    REPLAN = "REPLAN"
    ABSTAIN = "ABSTAIN"
    FALLBACK = "FALLBACK"


class VerifierVerdict(str, Enum):
    """Verifier-specific verdicts."""

    ACCEPT = "ACCEPT"
    CONTINUE = "CONTINUE"
    REPLAN = "REPLAN"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"


class SubtaskKind(str, Enum):
    """Subtasks that can appear in the Ultra workflow graph."""

    DIAGNOSE_REGIME = "DIAGNOSE_REGIME"
    ESTIMATE_DIFFICULTY = "ESTIMATE_DIFFICULTY"
    ANALYZE_RESIDUAL_DYNAMICS = "ANALYZE_RESIDUAL_DYNAMICS"
    ANALYZE_SPATIAL_PROPAGATION = "ANALYZE_SPATIAL_PROPAGATION"
    FORECAST_LOCAL_LEVEL = "FORECAST_LOCAL_LEVEL"
    FORECAST_REGIONAL_RESIDUAL = "FORECAST_REGIONAL_RESIDUAL"
    TRANSFER_REGIONAL_SIGNAL = "TRANSFER_REGIONAL_SIGNAL"
    ESTIMATE_UNCERTAINTY = "ESTIMATE_UNCERTAINTY"
    SYNTHESIZE_FORECASTS = "SYNTHESIZE_FORECASTS"
    VERIFY_PHYSICS = "VERIFY_PHYSICS"
    VERIFY_SOURCE_CONSISTENCY = "VERIFY_SOURCE_CONSISTENCY"
    VERIFY_CALIBRATION = "VERIFY_CALIBRATION"
    VERIFY_EVENT_RISK = "VERIFY_EVENT_RISK"


class MessageStatus(str, Enum):
    """Execution status for one coordination message."""

    SUCCESS = "success"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REUSED = "reused"


@dataclass(frozen=True)
class ExecutionBudget:
    """Global and per-call limits enforced by the Ultra conductor."""

    max_turns: int = 5
    max_distinct_numerical_experts: int = 3
    max_recursion_depth: int = 1
    deadline_ms: float = 2500.0
    per_expert_timeout_ms: float = 750.0
    max_parallel_actions: int = 1
    reserved_safe_fallback_calls: int = 1


@dataclass(frozen=True)
class CoordinationAction:
    """One selected action in the state-action coordination loop."""

    turn_id: int
    expert_id: str
    role: Role
    subtask_kind: SubtaskKind
    subtask_parameters: dict[str, Any] = field(default_factory=dict)
    access_list: list[int] = field(default_factory=list)
    parallel_group: str | None = None
    expected_cost: float = 1.0
    policy_score: float = 0.0
    action_probability: float | None = None
    rationale_for_audit: str = ""
    control_decision: ControlDecision = ControlDecision.CONTINUE

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CoordinationAction":
        values = dict(payload)
        values["role"] = Role(values["role"])
        values["subtask_kind"] = SubtaskKind(values["subtask_kind"])
        values["control_decision"] = ControlDecision(values.get("control_decision", "CONTINUE"))
        return cls(**values)


@dataclass
class CoordinationMessage:
    """Result returned by the selected role-specific expert."""

    turn_id: int
    expert_id: str
    role: Role
    subtask_kind: SubtaskKind
    visible_prior_turns: list[int]
    status: MessageStatus
    structured_result: dict[str, Any]
    latency_ms: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CoordinationMessage":
        values = dict(payload)
        values["role"] = Role(values["role"])
        values["subtask_kind"] = SubtaskKind(values["subtask_kind"])
        values["status"] = MessageStatus(values["status"])
        return cls(**values)


@dataclass(frozen=True)
class CoordinationTelemetry:
    """Accounting emitted with every ForecastResult."""

    logical_actions: int = 0
    physical_expert_calls: int = 0
    reused_expert_outputs: int = 0
    unique_experts: int = 0
    verifier_calls: int = 0
    thinker_calls: int = 0
    worker_calls: int = 0
    fallback_calls: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateForecast:
    """Numerical forecast candidate carried through the coordination state."""

    forecast_m: float
    lower_m: float
    upper_m: float
    confidence: float
    experts_used: list[str]
    method: str
    source_turn_id: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(asdict(self))

    @property
    def interval_width_m(self) -> float:
        return float(self.upper_m - self.lower_m)


@dataclass(frozen=True)
class VerifierResult:
    """Structured verifier output."""

    verdict: VerifierVerdict
    problems_found: list[str] = field(default_factory=list)
    confidence_adjustment: float = 0.0
    interval_adjustment_recommendation: float = 1.0
    requested_evidence: list[str] = field(default_factory=list)
    recommended_next_subtask: SubtaskKind | None = None
    recommended_next_expert_or_verifier: str | None = None
    safe_fallback_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(asdict(self))


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value
