"""Coordinator policies for Wai Ultra."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.artifacts import load_json_artifact
from src.orchestration.coordination_state import CoordinationState
from src.orchestration.coordinator_head import ActionRegistry, CoordinationHead, action_key
from src.orchestration.protocol import CoordinationAction
from src.orchestration.action_masks import FeasibleAction
from src.orchestration.state_encoder import FeatureSchema, StateEncoder


class CoordinatorPolicy(ABC):
    """Interface for one-turn-at-a-time Ultra policies."""

    policy_source: str = "unknown"
    artifact_version: str = "unversioned"

    @abstractmethod
    def select_action(
        self,
        state: CoordinationState,
        feasible_actions: list[FeasibleAction],
    ) -> CoordinationAction | None:
        """Select one feasible action based on the current encoded state."""


@dataclass
class LearnedCoordinatorPolicy(CoordinatorPolicy):
    """Small learned policy head with hard action masks applied after logits."""

    head: CoordinationHead
    encoder: StateEncoder
    artifact: dict[str, Any]
    policy_source: str = "learned"
    artifact_version: str = "unvalidated"
    min_validation_accuracy: float = 0.01

    @classmethod
    def from_artifact(
        cls,
        artifact: dict[str, Any],
        *,
        encoder: StateEncoder | None = None,
        expected_registry: ActionRegistry | None = None,
        min_validation_accuracy: float = 0.01,
        allow_shadow: bool = False,
    ) -> "LearnedCoordinatorPolicy":
        enc = encoder or StateEncoder()
        FeatureSchema().validate_artifact_schema(artifact["feature_schema"])
        head = CoordinationHead.from_artifact(artifact)
        if expected_registry is not None:
            expected_registry.validate(head.action_registry)
        metadata = artifact.get("training_metadata", {})
        policy_source = _validate_artifact_for_control(
            artifact,
            min_validation_accuracy=min_validation_accuracy,
            allow_shadow=allow_shadow,
        )
        return cls(
            head=head,
            encoder=enc,
            artifact=artifact,
            policy_source=policy_source,
            artifact_version=str(metadata.get("artifact_version", "unversioned")),
            min_validation_accuracy=min_validation_accuracy,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        encoder: StateEncoder | None = None,
        expected_registry: ActionRegistry | None = None,
        min_validation_accuracy: float = 0.01,
        allow_shadow: bool = False,
    ) -> "LearnedCoordinatorPolicy":
        artifact = load_json_artifact(path, expected_kind="coordinator_policy")
        return cls.from_artifact(
            artifact,
            encoder=encoder,
            expected_registry=expected_registry,
            min_validation_accuracy=min_validation_accuracy,
            allow_shadow=allow_shadow,
        )

    def select_action(
        self,
        state: CoordinationState,
        feasible_actions: list[FeasibleAction],
    ) -> CoordinationAction | None:
        feasible = [item.action for item in feasible_actions if item.feasible]
        if not feasible:
            return None
        encoded = self.encoder.encode(state)
        state.encoded_state = encoded.tolist()
        keys = [_key_for_action(action) for action in feasible]
        probabilities = self.head.probabilities(encoded, keys)
        if not probabilities:
            return None
        selected_key = max(probabilities.items(), key=lambda item: item[1])[0]
        selected = feasible[keys.index(selected_key)]
        logits = self.head.logits(encoded)
        registry = list(self.head.action_registry.action_keys)
        score = float(logits[registry.index(selected_key)]) if selected_key in registry else 0.0
        return CoordinationAction(
            **{
                **selected.__dict__,
                "policy_score": score,
                "action_probability": float(probabilities[selected_key]),
                "rationale_for_audit": "learned coordination head selected highest-scoring feasible action",
            }
        )


def build_action_registry_from_specs(specs: dict[str, Any]) -> ActionRegistry:
    keys = []
    for spec in specs.values():
        for subtask in spec.subtasks:
            control = "FALLBACK" if spec.expert_id == "safe_fallback" else "ACCEPT" if spec.role.value == "VERIFIER" else "CONTINUE"
            keys.append(action_key(spec.role.value, spec.expert_id, subtask.value, control))
    return ActionRegistry(version="wai-ultra-actions-v1", action_keys=tuple(sorted(set(keys))))


def _key_for_action(action: CoordinationAction) -> str:
    return action_key(
        action.role.value,
        action.expert_id,
        action.subtask_kind.value,
        action.control_decision.value,
    )


def _validate_artifact_for_control(
    artifact: dict[str, Any],
    *,
    min_validation_accuracy: float,
    allow_shadow: bool,
) -> str:
    metrics = artifact.get("validation_metrics", {})
    metadata = artifact.get("training_metadata", {})
    thresholds = {
        "min_validation_accuracy": min_validation_accuracy,
        "min_heldout_workflow_reward": -10.0,
        "max_routing_regret": 10.0,
        "max_mae": 10.0,
        "max_peak_event_error": 10.0,
        "min_interval_coverage": 0.0,
        "max_unavailable_rate": 1.0,
        "min_fallback_success_rate": 0.0,
        "max_invalid_action_rate": 0.0,
        "min_dropout_reward": -10.0,
        **metadata.get("validation_thresholds", {}),
    }
    thresholds["min_validation_accuracy"] = max(
        float(thresholds["min_validation_accuracy"]),
        float(min_validation_accuracy),
    )
    required = [
        "heldout_workflow_reward",
        "routing_regret",
        "mae",
        "peak_event_error",
        "interval_coverage",
        "unavailable_rate",
        "fallback_success_rate",
        "invalid_action_rate",
        "expert_dropout_reward",
    ]
    missing = [name for name in required if name not in metrics]
    test_accuracy = metrics.get("test_accuracy")
    failed = bool(missing)
    if test_accuracy is not None and float(test_accuracy) < thresholds["min_validation_accuracy"]:
        failed = True
    comparisons = {
        "heldout_workflow_reward": float(metrics.get("heldout_workflow_reward", -float("inf"))) >= thresholds["min_heldout_workflow_reward"],
        "routing_regret": float(metrics.get("routing_regret", float("inf"))) <= thresholds["max_routing_regret"],
        "mae": float(metrics.get("mae", float("inf"))) <= thresholds["max_mae"],
        "peak_event_error": float(metrics.get("peak_event_error", float("inf"))) <= thresholds["max_peak_event_error"],
        "interval_coverage": float(metrics.get("interval_coverage", -float("inf"))) >= thresholds["min_interval_coverage"],
        "unavailable_rate": float(metrics.get("unavailable_rate", float("inf"))) <= thresholds["max_unavailable_rate"],
        "fallback_success_rate": float(metrics.get("fallback_success_rate", -float("inf"))) >= thresholds["min_fallback_success_rate"],
        "invalid_action_rate": float(metrics.get("invalid_action_rate", float("inf"))) <= thresholds["max_invalid_action_rate"],
        "expert_dropout_reward": float(metrics.get("expert_dropout_reward", -float("inf"))) >= thresholds["min_dropout_reward"],
    }
    if not all(comparisons.values()):
        failed = True
    status = str(metadata.get("validation_status", "shadow"))
    if status != "validated":
        failed = True
    if failed:
        if allow_shadow:
            return "learned_shadow"
        raise ValueError("Coordinator artifact did not pass held-out validation thresholds")
    return "learned"
