"""Offline oracle workflow search for Wai Ultra trajectories."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any

import pandas as pd

from src.evaluation.reward import TerminalLossConfig, reward_from_loss, terminal_loss
from src.experts.verifiers import (
    CalibrationVerifier,
    CrossSourceVerifier,
    EventRiskVerifier,
    PhysicsAndDatumVerifier,
)
from src.experts.workers import EnsembleSynthesisWorker
from src.orchestration.action_masks import default_expert_specs, feasible_actions_for_state
from src.orchestration.coordination_state import CoordinationState
from src.orchestration.protocol import (
    CoordinationAction,
    CoordinationMessage,
    ExecutionBudget,
    MessageStatus,
    Role,
    RoleInput,
    SubtaskKind,
    VerifierVerdict,
)
from src.orchestration.state_encoder import StateEncoder


@dataclass
class OracleWorkflow:
    """One bounded oracle workflow candidate."""

    workflow_id: str
    actions: list[dict[str, Any]]
    final_candidate: dict[str, Any] | None
    terminal_loss: float
    reward: float
    terminal: bool
    total_calls: int
    total_latency_ms: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def search_oracle_workflows(
    *,
    expert_predictions: dict[str, dict[str, Any]],
    actual_m: float | None,
    max_turns: int = 5,
    config: TerminalLossConfig | None = None,
    keep_alternatives: int = 3,
    beam_width: int = 8,
    context: Any | None = None,
) -> list[OracleWorkflow]:
    """Search bounded valid workflows using live Ultra state transitions."""

    specs = default_expert_specs()
    encoder = StateEncoder()
    budget = ExecutionBudget(max_turns=max_turns, max_coordination_turns=max_turns)
    now = time.monotonic()
    root = CoordinationState(
        original_context=context or _context_from_predictions(expert_predictions),
        available_expert_pool=sorted(specs),
        capability_masks={},
        budget=budget,
        started_monotonic=now,
        deadline_monotonic=now + budget.deadline_ms / 1000.0,
    )
    beam = [root]
    completed: list[CoordinationState] = []

    for _ in range(max_turns):
        next_beam: list[CoordinationState] = []
        for state in beam:
            if _terminal_accept(state) or state.remaining_turn_budget <= 0:
                completed.append(state)
                continue
            state.encoded_state = encoder.encode(state).tolist()
            actions = [
                item.action
                for item in feasible_actions_for_state(state, specs)
                if item.feasible and _search_action_allowed(state, item.action, expert_predictions)
            ]
            for action in actions:
                child = copy.deepcopy(state)
                remapped = _remap_action_for_state(action, child)
                message = _simulate_action(remapped, child, expert_predictions)
                child.append(remapped, message)
                child.encoded_state = encoder.encode(child).tolist()
                if _message_accepts(message):
                    child.completed_workflow_graph.mark_accepted(message.turn_id)
                    completed.append(child)
                else:
                    next_beam.append(child)
        if not next_beam:
            break
        beam = sorted(next_beam, key=lambda state: _state_loss(state, actual_m, config))[:beam_width]

    completed.extend(beam)
    workflows = [_workflow_from_state(state, actual_m, config) for state in completed]
    if not workflows:
        workflows = [_unavailable_workflow(actual_m, config)]
    workflows.sort(key=lambda item: item.terminal_loss)
    return workflows[: max(1, keep_alternatives)]


def _simulate_action(
    action: CoordinationAction,
    state: CoordinationState,
    expert_predictions: dict[str, dict[str, Any]],
) -> CoordinationMessage:
    started = time.perf_counter()
    try:
        visible = state.visible_messages(action.access_list)
    except Exception as exc:
        return _message(action, MessageStatus.FAILED, {"error": str(exc)}, started)
    role_input = RoleInput(
        context=state.original_context,
        subtask_kind=action.subtask_kind,
        subtask_parameters=dict(action.subtask_parameters),
        visible_messages=visible,
        remaining_turn_budget=state.remaining_turn_budget,
        remaining_physical_worker_calls=state.remaining_physical_worker_calls,
        remaining_verifier_calls=state.remaining_verifier_calls,
        remaining_deadline_ms=state.remaining_deadline_ms,
        requested_evidence=list(state.verifier_findings[-1].get("requested_evidence", []))
        if state.verifier_findings
        else [],
    )
    if action.role is Role.THINKER:
        result = _simulated_thinker_result(state.original_context, action)
        return _message(action, MessageStatus.SUCCESS, result, started)
    if action.role is Role.WORKER:
        if action.expert_id == "ensemble_synthesis":
            result = EnsembleSynthesisWorker().run(role_input)
            return _message(action, _status_from_worker_result(result), result, started)
        result = _worker_result_from_prediction(action.expert_id, expert_predictions)
        return _message(action, _status_from_worker_result(result), result, started)
    verifier = _verifier(action).verify(role_input)
    return _message(action, MessageStatus.SUCCESS, {"verifier": verifier.to_dict()}, started)


def _message(
    action: CoordinationAction,
    status: MessageStatus,
    result: dict[str, Any],
    started: float,
) -> CoordinationMessage:
    return CoordinationMessage(
        turn_id=action.turn_id,
        expert_id=action.expert_id,
        role=action.role,
        subtask_kind=action.subtask_kind,
        visible_prior_turns=list(action.access_list),
        status=status,
        structured_result=result,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        warnings=[],
    )


def _worker_result_from_prediction(
    expert_id: str,
    expert_predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = expert_predictions.get(expert_id)
    if payload is None:
        return {
            "forecast": None,
            "worker_status": "unavailable",
            "message": f"no precomputed forecast for {expert_id}",
        }
    status = payload.get("status", "success")
    if status != "success" or payload.get("prediction_m") is None:
        return {
            "forecast": None,
            "worker_status": status,
            "message": payload.get("message", status),
        }
    return {
        "forecast": _candidate_from_prediction(expert_id, payload),
        "worker_status": "success",
        "message": "",
        "assumptions": ["precomputed numerical forecast reused by oracle search"],
    }


def _candidate_from_prediction(expert_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "forecast_m": float(payload["prediction_m"]),
        "lower_m": float(payload["lower_m"]),
        "upper_m": float(payload["upper_m"]),
        "confidence": float(payload.get("confidence", 0.5)),
        "experts_used": [expert_id],
        "leaf_experts": [expert_id],
        "input_turn_ids": [],
        "method": expert_id,
        "diagnostics": dict(payload.get("diagnostics", {})),
    }


def _simulated_thinker_result(context: Any, action: CoordinationAction) -> dict[str, Any]:
    residual = abs(float(getattr(context, "recent_noaa_residual_m", 0.0) or 0.0))
    trend = abs(float(getattr(context, "noaa_residual_trend_m_per_hour", 0.0) or 0.0))
    event_risk = min(1.0, residual / 0.35 + trend / 0.35)
    if event_risk >= 0.35:
        recommended = ["noaa_residual", "regional_to_local_residual"]
    else:
        recommended = ["local_tide", "local_persistence"]
    return {
        "forecast_difficulty": float(min(1.0, 0.2 + event_risk)),
        "event_risk": float(event_risk),
        "recommended_next_subtasks": [_worker_subtask(expert).value for expert in recommended],
        "recommended_experts": recommended,
        "recommended_verifier_type": "event_risk_verifier" if event_risk >= 0.35 else "physics_datum_verifier",
        "oracle_simulated": True,
        "subtask": action.subtask_kind.value,
    }


def _verifier(action: CoordinationAction):
    return {
        "physics_datum_verifier": PhysicsAndDatumVerifier(),
        "cross_source_verifier": CrossSourceVerifier(),
        "calibration_verifier": CalibrationVerifier(),
        "event_risk_verifier": EventRiskVerifier(),
    }[action.expert_id]


def _workflow_from_state(
    state: CoordinationState,
    actual_m: float | None,
    config: TerminalLossConfig | None,
) -> OracleWorkflow:
    candidate = state.latest_candidate()
    candidate_payload = None if candidate is None else candidate.to_dict()
    loss = _state_loss(state, actual_m, config)
    actions = [action.to_dict() for action in state.action_transcript]
    action_path = "/".join(f"{action.role.value}:{action.expert_id}" for action in state.action_transcript)
    return OracleWorkflow(
        workflow_id=f"beam:{action_path or 'unavailable'}",
        actions=actions,
        final_candidate=candidate_payload,
        terminal_loss=loss,
        reward=reward_from_loss(loss),
        terminal=_terminal_accept(state),
        total_calls=len(state.full_message_transcript),
        total_latency_ms=sum(message.latency_ms for message in state.full_message_transcript),
        diagnostics={
            "workflow_graph": state.topology_dict(),
            "verifier_findings": state.verifier_findings,
            "search_type": "bounded_live_state_beam",
        },
    )


def _state_loss(
    state: CoordinationState,
    actual_m: float | None,
    config: TerminalLossConfig | None,
) -> float:
    candidate = state.latest_candidate()
    failed = candidate is None or any(
        message.status in {MessageStatus.FAILED, MessageStatus.TIMEOUT}
        for message in state.full_message_transcript
    )
    return terminal_loss(
        forecast_m=None if candidate is None else candidate.forecast_m,
        lower_m=None if candidate is None else candidate.lower_m,
        upper_m=None if candidate is None else candidate.upper_m,
        actual_m=actual_m,
        total_calls=len(state.full_message_transcript),
        total_latency_ms=sum(message.latency_ms for message in state.full_message_transcript),
        failed=failed,
        config=config,
    )


def _unavailable_workflow(
    actual_m: float | None,
    config: TerminalLossConfig | None,
) -> OracleWorkflow:
    loss = terminal_loss(
        forecast_m=None,
        lower_m=None,
        upper_m=None,
        actual_m=actual_m,
        total_calls=0,
        total_latency_ms=0.0,
        failed=True,
        config=config,
    )
    return OracleWorkflow(
        workflow_id="beam:unavailable",
        actions=[],
        final_candidate=None,
        terminal_loss=loss,
        reward=reward_from_loss(loss),
        terminal=True,
        total_calls=0,
        total_latency_ms=0.0,
        diagnostics={"search_type": "bounded_live_state_beam"},
    )


def _search_action_allowed(
    state: CoordinationState,
    action: CoordinationAction,
    expert_predictions: dict[str, dict[str, Any]],
) -> bool:
    if action.role is Role.WORKER and action.expert_id not in {"ensemble_synthesis", "safe_fallback"}:
        if action.expert_id not in expert_predictions:
            return False
        if any(message.expert_id == action.expert_id for message in state.full_message_transcript):
            return False
    if action.role is Role.THINKER and any(
        message.expert_id == action.expert_id for message in state.full_message_transcript
    ):
        return False
    if action.role is Role.VERIFIER:
        latest = state.verifier_findings[-1] if state.verifier_findings else {}
        if latest.get("verdict") == VerifierVerdict.ACCEPT.value:
            return False
    return True


def _remap_action_for_state(action: CoordinationAction, state: CoordinationState) -> CoordinationAction:
    if action.turn_id == state.next_turn_id:
        return action
    return copy.copy(action)


def _status_from_worker_result(result: dict[str, Any]) -> MessageStatus:
    status = result.get("worker_status")
    if status == "timeout":
        return MessageStatus.TIMEOUT
    if result.get("forecast") is None and status in {"unavailable", "failed", "error"}:
        return MessageStatus.UNAVAILABLE
    if "error" in result:
        return MessageStatus.FAILED
    return MessageStatus.SUCCESS


def _terminal_accept(state: CoordinationState) -> bool:
    return any(_message_accepts(message) for message in state.full_message_transcript)


def _message_accepts(message: CoordinationMessage) -> bool:
    verifier = message.structured_result.get("verifier", {})
    return verifier.get("verdict") == VerifierVerdict.ACCEPT.value


def _worker_subtask(expert_id: str) -> SubtaskKind:
    if expert_id == "noaa_residual":
        return SubtaskKind.FORECAST_REGIONAL_RESIDUAL
    if expert_id == "regional_to_local_residual":
        return SubtaskKind.TRANSFER_REGIONAL_SIGNAL
    return SubtaskKind.FORECAST_LOCAL_LEVEL


def _context_from_predictions(expert_predictions: dict[str, dict[str, Any]]) -> Any:
    baseline = expert_predictions.get("local_tide") or expert_predictions.get("safe_fallback") or {}
    residual_payload = expert_predictions.get("noaa_residual") or {}
    residual = None
    if baseline.get("prediction_m") is not None and residual_payload.get("prediction_m") is not None:
        residual = float(residual_payload["prediction_m"]) - float(baseline["prediction_m"])
    origin = pd.Timestamp("2024-01-01T00:00:00Z")
    return SimpleNamespace(
        target_station_id="ORACLE_CONTEXT",
        paired_noaa_station_id="ORACLE_NOAA",
        forecast_time_utc=origin,
        target_time_utc=origin + pd.Timedelta(hours=6),
        horizon_minutes=360,
        station_pair=SimpleNamespace(paired_noaa_station_id="ORACLE_NOAA", residual_scale=1.0, lag_minutes=0),
        datum="MLLW",
        latest_hohonu_observation={"water_level_m": float(baseline.get("prediction_m", 0.0)), "timestamp_utc": origin},
        latest_noaa_observation={"water_level_m": float(residual_payload.get("prediction_m", baseline.get("prediction_m", 0.0))), "timestamp_utc": origin},
        noaa_tide_prediction={"water_level_m": float(baseline.get("prediction_m", 0.0)), "timestamp_utc": origin},
        local_tide_prediction=None,
        recent_hohonu_observations=pd.DataFrame(),
        recent_noaa_observations=pd.DataFrame(),
        noaa_tide_predictions=pd.DataFrame(),
        recent_hohonu_trend_m_per_hour=0.0,
        recent_noaa_residual_m=residual,
        noaa_residual_trend_m_per_hour=0.0,
        observation_freshness_seconds={"hohonu": 300.0, "noaa": 300.0},
        qc_status={"hohonu": "pass", "noaa": "verified"},
        tide_phase="rising",
        pressure_trend=None,
        wind_speed_mps=None,
        wind_direction_deg=None,
        neighboring_station_signals={},
        recent_model_performance={},
        model_disagreement_m=None,
        diagnostics={},
        hohonu_is_fresh=True,
        noaa_is_fresh=True,
        hohonu_qc_ok=True,
        noaa_qc_ok=True,
    )
