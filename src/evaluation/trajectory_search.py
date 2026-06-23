"""Offline oracle workflow search for Wai Ultra trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from src.evaluation.reward import TerminalLossConfig, reward_from_loss, terminal_loss


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
) -> list[OracleWorkflow]:
    """Search bounded valid workflows using precomputed numerical expert outputs."""

    successful_workers = {
        name: payload
        for name, payload in expert_predictions.items()
        if payload.get("status") == "success" and payload.get("prediction_m") is not None
    }
    workflows: list[OracleWorkflow] = []

    for worker, payload in successful_workers.items():
        actions = [
            _worker_action(worker, 0),
            _verifier_action("physics_datum_verifier", 1, [0]),
        ]
        if len(actions) <= max_turns:
            candidate = _candidate_from_prediction(worker, payload)
            workflows.append(_score_workflow(f"single:{worker}", actions, candidate, actual_m, config))

    for left, right in combinations(successful_workers, 2):
        actions = [
            _worker_action(left, 0),
            _worker_action(right, 1),
            _synthesis_action(2, [0, 1]),
            _verifier_action("cross_source_verifier", 3, [2]),
        ]
        if len(actions) <= max_turns:
            candidate = _synthesize({
                left: successful_workers[left],
                right: successful_workers[right],
            })
            workflows.append(_score_workflow(f"pair:{left}+{right}", actions, candidate, actual_m, config))

    if "safe_fallback" in successful_workers:
        actions = [
            _worker_action("safe_fallback", 0),
            _verifier_action("physics_datum_verifier", 1, [0]),
        ]
        workflows.append(
            _score_workflow(
                "fallback:safe_fallback",
                actions,
                _candidate_from_prediction("safe_fallback", successful_workers["safe_fallback"]),
                actual_m,
                config,
            )
        )

    if not workflows:
        workflows.append(
            _score_workflow(
                "unavailable",
                [],
                None,
                actual_m,
                config,
                failed=True,
            )
        )

    workflows.sort(key=lambda item: item.terminal_loss)
    return workflows[: max(1, keep_alternatives)]


def _score_workflow(
    workflow_id: str,
    actions: list[dict[str, Any]],
    candidate: dict[str, Any] | None,
    actual_m: float | None,
    config: TerminalLossConfig | None,
    *,
    failed: bool = False,
) -> OracleWorkflow:
    loss = terminal_loss(
        forecast_m=None if candidate is None else candidate.get("forecast_m"),
        lower_m=None if candidate is None else candidate.get("lower_m"),
        upper_m=None if candidate is None else candidate.get("upper_m"),
        actual_m=actual_m,
        total_calls=len(actions),
        total_latency_ms=sum(action.get("expected_latency_ms", 0.0) for action in actions),
        failed=failed,
        config=config,
    )
    return OracleWorkflow(
        workflow_id=workflow_id,
        actions=actions,
        final_candidate=candidate,
        terminal_loss=loss,
        reward=reward_from_loss(loss),
        terminal=True,
        total_calls=len(actions),
        total_latency_ms=sum(action.get("expected_latency_ms", 0.0) for action in actions),
    )


def _worker_action(expert_id: str, turn: int) -> dict[str, Any]:
    return {
        "turn": turn,
        "role": "WORKER",
        "expert": expert_id,
        "subtask": _worker_subtask(expert_id),
        "access_list": [],
        "expected_latency_ms": 10.0,
    }


def _synthesis_action(turn: int, access_list: list[int]) -> dict[str, Any]:
    return {
        "turn": turn,
        "role": "WORKER",
        "expert": "ensemble_synthesis",
        "subtask": "SYNTHESIZE_FORECASTS",
        "access_list": access_list,
        "expected_latency_ms": 6.0,
    }


def _verifier_action(expert_id: str, turn: int, access_list: list[int]) -> dict[str, Any]:
    return {
        "turn": turn,
        "role": "VERIFIER",
        "expert": expert_id,
        "subtask": "VERIFY_PHYSICS" if expert_id == "physics_datum_verifier" else "VERIFY_SOURCE_CONSISTENCY",
        "access_list": access_list,
        "expected_latency_ms": 4.0,
    }


def _worker_subtask(expert_id: str) -> str:
    if expert_id == "noaa_residual":
        return "FORECAST_REGIONAL_RESIDUAL"
    if expert_id == "regional_to_local_residual":
        return "TRANSFER_REGIONAL_SIGNAL"
    return "FORECAST_LOCAL_LEVEL"


def _candidate_from_prediction(expert_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "forecast_m": float(payload["prediction_m"]),
        "lower_m": float(payload["lower_m"]),
        "upper_m": float(payload["upper_m"]),
        "confidence": float(payload.get("confidence", 0.5)),
        "experts_used": [expert_id],
        "method": expert_id,
    }


def _synthesize(predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = np.array([float(item["prediction_m"]) for item in predictions.values()], dtype=float)
    confidences = np.array([float(item.get("confidence", 0.5)) for item in predictions.values()], dtype=float)
    weights = np.maximum(confidences, 0.05)
    forecast = float(np.average(values, weights=weights))
    lowers = np.array([float(item["lower_m"]) for item in predictions.values()], dtype=float)
    uppers = np.array([float(item["upper_m"]) for item in predictions.values()], dtype=float)
    lower = float(np.average(lowers, weights=weights))
    upper = float(np.average(uppers, weights=weights))
    span = float(values.max() - values.min()) if len(values) > 1 else 0.0
    half = max(forecast - lower, upper - forecast, 0.04 + 0.5 * span)
    return {
        "forecast_m": forecast,
        "lower_m": forecast - half,
        "upper_m": forecast + half,
        "confidence": float(np.average(confidences, weights=weights) - min(0.2, span * 0.1)),
        "experts_used": list(predictions),
        "method": "oracle_weighted_synthesis",
    }
