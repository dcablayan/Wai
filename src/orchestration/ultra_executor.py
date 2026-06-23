"""Execution layer for Wai Ultra actions."""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Callable

from src.experts import ForecastExpert
from src.experts.thinkers import RegimeAndDifficultyThinker, ResidualDynamicsThinker
from src.experts.verifiers import (
    CalibrationVerifier,
    CrossSourceVerifier,
    EventRiskVerifier,
    PhysicsAndDatumVerifier,
)
from src.experts.workers import EnsembleSynthesisWorker, ForecastWorkerAdapter
from src.orchestration.coordination_state import CoordinationState
from src.orchestration.protocol import (
    CoordinationAction,
    CoordinationMessage,
    MessageStatus,
    Role,
    RoleInput,
)


class UltraExecutor:
    """Run one role-specific Ultra action with access-list enforcement."""

    def __init__(self, forecast_experts: dict[str, ForecastExpert]) -> None:
        self.thinkers = {
            "regime_difficulty_thinker": RegimeAndDifficultyThinker(),
            "residual_dynamics_thinker": ResidualDynamicsThinker(),
        }
        self.workers: dict[str, Any] = {
            name: ForecastWorkerAdapter(expert)
            for name, expert in forecast_experts.items()
        }
        self.workers["ensemble_synthesis"] = EnsembleSynthesisWorker()
        self.verifiers = {
            "physics_datum_verifier": PhysicsAndDatumVerifier(),
            "cross_source_verifier": CrossSourceVerifier(),
            "calibration_verifier": CalibrationVerifier(),
            "event_risk_verifier": EventRiskVerifier(),
        }

    def execute(self, action: CoordinationAction, state: CoordinationState) -> CoordinationMessage:
        started = time.perf_counter()
        try:
            visible_messages = state.visible_messages(action.access_list)
        except Exception as exc:
            return self._message(
                action,
                status=MessageStatus.FAILED,
                result={"error": str(exc)},
                latency_ms=(time.perf_counter() - started) * 1000.0,
                warnings=[str(exc)],
            )

        if action.role is Role.WORKER and action.expert_id not in {"ensemble_synthesis"}:
            cached = state.physical_forecast_cache.get(action.expert_id)
            if cached is not None:
                result = {**cached, "reused": True}
                return self._message(
                    action,
                    status=MessageStatus.REUSED,
                    result=result,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )

        remaining_timeout_ms = min(
            state.budget.per_expert_timeout_ms,
            max(1.0, state.remaining_deadline_ms),
        )
        result, status, warnings = self._run_with_timeout(
            lambda: self._execute_unbounded(action, self._role_input(action, state, visible_messages)),
            timeout_ms=remaining_timeout_ms,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        if status is MessageStatus.SUCCESS and action.role is Role.WORKER and action.expert_id != "ensemble_synthesis":
            state.physical_forecast_cache[action.expert_id] = dict(result)
        if action.expert_id == "safe_fallback":
            state.fallback_attempted = True

        return self._message(action, status=status, result=result, latency_ms=latency_ms, warnings=warnings)

    def _execute_unbounded(
        self,
        action: CoordinationAction,
        role_input: RoleInput,
    ) -> dict[str, Any]:
        if action.role is Role.THINKER:
            thinker = self.thinkers[action.expert_id]
            return thinker.analyze(role_input)
        if action.role is Role.WORKER:
            worker = self.workers[action.expert_id]
            return worker.run(role_input)
        if action.role is Role.VERIFIER:
            verifier = self.verifiers[action.expert_id]
            return {"verifier": verifier.verify(role_input).to_dict()}
        raise ValueError(f"Unsupported role: {action.role}")

    def _role_input(
        self,
        action: CoordinationAction,
        state: CoordinationState,
        visible_messages: list[CoordinationMessage],
    ) -> RoleInput:
        requested_evidence = []
        if state.verifier_findings:
            requested_evidence = list(state.verifier_findings[-1].get("requested_evidence", []))
        return RoleInput(
            context=state.original_context,
            subtask_kind=action.subtask_kind,
            subtask_parameters=dict(action.subtask_parameters),
            visible_messages=visible_messages,
            remaining_turn_budget=state.remaining_turn_budget,
            remaining_physical_worker_calls=state.remaining_physical_worker_calls,
            remaining_verifier_calls=state.remaining_verifier_calls,
            remaining_deadline_ms=state.remaining_deadline_ms,
            requested_evidence=requested_evidence,
        )

    def _run_with_timeout(
        self,
        fn: Callable[[], dict[str, Any]],
        *,
        timeout_ms: float,
    ) -> tuple[dict[str, Any], MessageStatus, list[str]]:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn)
        try:
            result = future.result(timeout=timeout_ms / 1000.0)
            status = _status_from_result(result)
            executor.shutdown(wait=False, cancel_futures=True)
            return result, status, []
        except concurrent.futures.TimeoutError:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            return (
                {"error": f"action exceeded timeout budget ({timeout_ms:.0f} ms)"},
                MessageStatus.TIMEOUT,
                ["Action timed out; underlying Python work may finish in the background"],
            )
        except Exception as exc:
            executor.shutdown(wait=False, cancel_futures=True)
            return {"error": str(exc)}, MessageStatus.FAILED, [str(exc)]

    def _message(
        self,
        action: CoordinationAction,
        *,
        status: MessageStatus,
        result: dict[str, Any],
        latency_ms: float,
        warnings: list[str] | None = None,
    ) -> CoordinationMessage:
        return CoordinationMessage(
            turn_id=action.turn_id,
            expert_id=action.expert_id,
            role=action.role,
            subtask_kind=action.subtask_kind,
            visible_prior_turns=list(action.access_list),
            status=status,
            structured_result=result,
            latency_ms=float(latency_ms),
            warnings=warnings or [],
        )


def _status_from_result(result: dict[str, Any]) -> MessageStatus:
    if result.get("worker_status") == "timeout":
        return MessageStatus.TIMEOUT
    if result.get("forecast") is None and result.get("worker_status") in {"unavailable", "failed"}:
        return MessageStatus.UNAVAILABLE
    if "error" in result:
        return MessageStatus.FAILED
    return MessageStatus.SUCCESS
