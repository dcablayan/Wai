"""Build Wai Ultra multi-turn trajectory datasets from replay rows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.evaluation.reward import interval_coverage
from src.evaluation.trajectory_search import search_oracle_workflows


FORBIDDEN_STATE_KEYS = ("actual", "error", "target_observation", "future", "oracle")


@dataclass
class TrajectoryTransition:
    """One state/action/next-state/reward transition for coordinator training."""

    episode_id: str
    forecast_origin: str
    turn: int
    encoded_state: list[float]
    state_feature_names: list[str]
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_coordination_trajectory_dataset(
    replay: pd.DataFrame,
    *,
    max_turns: int = 5,
    keep_alternatives: int = 3,
) -> pd.DataFrame:
    """Create a transition table from historical replay rows.

    The actual water level is used only after oracle search chooses terminal
    rewards; it is not included in encoded_state.
    """

    transitions: list[TrajectoryTransition] = []
    for row_idx, row in replay.iterrows():
        context_features = _parse_json(row["context_features"])
        expert_predictions = _parse_json(row["expert_predictions"])
        actual_m = None if pd.isna(row.get("actual_m")) else float(row["actual_m"])
        workflows = search_oracle_workflows(
            expert_predictions=expert_predictions,
            actual_m=actual_m,
            max_turns=max_turns,
            keep_alternatives=keep_alternatives,
        )
        episode_id = _episode_id(row, row_idx)
        for alt_idx, workflow in enumerate(workflows):
            state_vector, feature_names = _encode_replay_state(context_features, turn=0, prior_actions=[])
            prior_actions: list[dict[str, Any]] = []
            for action in workflow.actions:
                turn = int(action["turn"])
                if turn > 0:
                    state_vector, feature_names = _encode_replay_state(
                        context_features,
                        turn=turn,
                        prior_actions=prior_actions,
                    )
                terminal = turn == len(workflow.actions) - 1
                candidate = workflow.final_candidate if terminal else None
                error = None
                coverage = None
                peak_loss = 0.0
                if terminal and candidate is not None and actual_m is not None:
                    error = float(candidate["forecast_m"]) - actual_m
                    coverage = interval_coverage(candidate, actual_m)
                    peak_loss = abs(error) if abs(actual_m) >= 0.75 else 0.0
                transitions.append(
                    TrajectoryTransition(
                        episode_id=f"{episode_id}:alt{alt_idx}",
                        forecast_origin=str(row["forecast_origin_utc"]),
                        turn=turn,
                        encoded_state=state_vector,
                        state_feature_names=feature_names,
                        available_action_mask=_available_mask(expert_predictions),
                        selected_action=_action_key(action),
                        role=action["role"],
                        expert=action["expert"],
                        subtask=action["subtask"],
                        access_list=list(action.get("access_list", [])),
                        result_summary=_result_summary(action, expert_predictions, candidate),
                        immediate_cost=float(action.get("expected_latency_ms", 0.0)),
                        terminal=terminal,
                        final_forecast_error=error,
                        interval_coverage=coverage,
                        peak_event_loss=peak_loss,
                        failure_or_abstention_status=None if candidate is not None else "unavailable",
                        total_calls=workflow.total_calls,
                        total_latency=workflow.total_latency_ms,
                        final_reward=workflow.reward,
                    )
                )
                prior_actions.append(action)
    dataset = pd.DataFrame([transition.to_dict() for transition in transitions])
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


def _encode_replay_state(
    context_features: dict[str, Any],
    *,
    turn: int,
    prior_actions: list[dict[str, Any]],
) -> tuple[list[float], list[str]]:
    safe = {
        "horizon_hours": float(context_features.get("horizon_minutes") or 0.0) / 60.0,
        "hohonu_freshness_hours": _hours(context_features.get("hohonu_freshness_seconds")),
        "noaa_freshness_hours": _hours(context_features.get("noaa_freshness_seconds")),
        "hohonu_qc_ok": 1.0 if context_features.get("hohonu_qc_status") in {"pass", "good", "ok"} else 0.0,
        "noaa_qc_ok": 1.0 if context_features.get("noaa_qc_status") in {"pass", "good", "ok"} else 0.0,
        "recent_hohonu_trend_abs": abs(float(context_features.get("recent_hohonu_trend_m_per_hour") or 0.0)),
        "recent_noaa_residual_abs": abs(float(context_features.get("recent_noaa_residual_m") or 0.0)),
        "noaa_residual_trend_abs": abs(float(context_features.get("noaa_residual_trend_m_per_hour") or 0.0)),
        "tide_phase_rising": 1.0 if context_features.get("tide_phase") == "rising" else 0.0,
        "tide_phase_falling": 1.0 if context_features.get("tide_phase") == "falling" else 0.0,
        "turn_fraction": turn / 5.0,
        "prior_worker_count": sum(1 for action in prior_actions if action["role"] == "WORKER") / 5.0,
        "prior_verifier_count": sum(1 for action in prior_actions if action["role"] == "VERIFIER") / 5.0,
        "used_synthesis": 1.0 if any(action["expert"] == "ensemble_synthesis" for action in prior_actions) else 0.0,
    }
    names = list(safe)
    return [float(safe[name]) for name in names], names


def _available_mask(expert_predictions: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {
        f"WORKER:{name}": payload.get("status") == "success"
        for name, payload in expert_predictions.items()
    }


def _result_summary(
    action: dict[str, Any],
    expert_predictions: dict[str, dict[str, Any]],
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    if action["role"] == "WORKER" and action["expert"] in expert_predictions:
        payload = expert_predictions[action["expert"]]
        return {
            "status": payload.get("status"),
            "has_forecast": payload.get("prediction_m") is not None,
            "confidence": payload.get("confidence"),
        }
    if candidate is not None:
        return {"status": "terminal", "method": candidate.get("method")}
    return {"status": "simulated"}


def _action_key(action: dict[str, Any]) -> str:
    control = "ACCEPT" if action["role"] == "VERIFIER" else "CONTINUE"
    return f"{action['role']}:{action['expert']}:{action['subtask']}:{control}"


def _parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(value)


def _episode_id(row: pd.Series, row_idx: int) -> str:
    origin = str(row.get("forecast_origin_utc", row_idx))
    station = str(row.get("target_station_id", "station"))
    return hashlib.sha1(f"{station}:{origin}:{row_idx}".encode("utf-8")).hexdigest()[:12]


def _hours(value: Any) -> float:
    if value is None:
        return 24.0
    try:
        return float(value) / 3600.0
    except (TypeError, ValueError):
        return 24.0
