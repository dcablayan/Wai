"""Coordinator policies for Wai Ultra."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
    min_validation_accuracy: float = 0.0

    @classmethod
    def from_artifact(
        cls,
        artifact: dict[str, Any],
        *,
        encoder: StateEncoder | None = None,
        expected_registry: ActionRegistry | None = None,
        min_validation_accuracy: float = 0.0,
    ) -> "LearnedCoordinatorPolicy":
        enc = encoder or StateEncoder()
        FeatureSchema().validate_artifact_schema(artifact["feature_schema"])
        head = CoordinationHead.from_artifact(artifact)
        if expected_registry is not None:
            expected_registry.validate(head.action_registry)
        metrics = artifact.get("validation_metrics", {})
        if metrics.get("imitation_accuracy", 0.0) < min_validation_accuracy:
            raise ValueError("Coordinator artifact did not pass validation threshold")
        metadata = artifact.get("training_metadata", {})
        return cls(
            head=head,
            encoder=enc,
            artifact=artifact,
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
        min_validation_accuracy: float = 0.0,
    ) -> "LearnedCoordinatorPolicy":
        import pickle

        with open(path, "rb") as handle:
            artifact = pickle.load(handle)
        return cls.from_artifact(
            artifact,
            encoder=encoder,
            expected_registry=expected_registry,
            min_validation_accuracy=min_validation_accuracy,
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
