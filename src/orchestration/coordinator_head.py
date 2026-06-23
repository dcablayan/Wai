"""Small auditable NumPy coordination head for Wai Ultra."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActionRegistry:
    """Versioned action key registry used by learned coordinator artifacts."""

    version: str
    action_keys: tuple[str, ...]

    def validate(self, other: "ActionRegistry") -> None:
        if self.version != other.version or self.action_keys != other.action_keys:
            raise ValueError("Coordinator action-registry mismatch")

    @property
    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass
class CoordinationHead:
    """Linear softmax action scorer kept intentionally below 20k parameters."""

    weights: np.ndarray
    bias: np.ndarray
    action_registry: ActionRegistry
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    training_metadata: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def initialize(
        cls,
        *,
        n_features: int,
        action_registry: ActionRegistry,
        seed: int = 42,
    ) -> "CoordinationHead":
        rng = np.random.default_rng(seed)
        n_actions = len(action_registry.action_keys)
        weights = rng.normal(0.0, 0.01, size=(n_features, n_actions))
        bias = np.zeros(n_actions, dtype=float)
        return cls(weights=weights, bias=bias, action_registry=action_registry)

    @property
    def parameter_count(self) -> int:
        return int(self.weights.size + self.bias.size)

    def logits(self, encoded_state: np.ndarray) -> np.ndarray:
        x = self._normalize(encoded_state)
        return x @ self.weights + self.bias

    def probabilities(self, encoded_state: np.ndarray, feasible_keys: list[str]) -> dict[str, float]:
        logits = self.logits(encoded_state)
        mask = np.array([key in feasible_keys for key in self.action_registry.action_keys], dtype=bool)
        if not mask.any():
            return {}
        masked = np.where(mask, logits, -1e9)
        probs = _softmax(masked)
        return {
            key: float(probs[idx])
            for idx, key in enumerate(self.action_registry.action_keys)
            if mask[idx]
        }

    def fit_imitation(
        self,
        X: np.ndarray,
        y_keys: list[str],
        *,
        learning_rate: float = 0.1,
        epochs: int = 200,
        l2: float = 1e-4,
        seed: int = 42,
    ) -> dict[str, float]:
        """Supervised warm start from oracle next-action labels."""

        if len(X) == 0:
            raise ValueError("Cannot train coordinator head on an empty dataset")
        key_to_idx = {key: idx for idx, key in enumerate(self.action_registry.action_keys)}
        y = np.array([key_to_idx[key] for key in y_keys], dtype=int)
        self.feature_mean = X.mean(axis=0)
        self.feature_scale = X.std(axis=0)
        self.feature_scale[self.feature_scale < 1e-6] = 1.0
        Xn = self._normalize_matrix(X)
        rng = np.random.default_rng(seed)
        order = np.arange(len(Xn))
        for _ in range(epochs):
            rng.shuffle(order)
            logits = Xn[order] @ self.weights + self.bias
            probs = _softmax_2d(logits)
            target = np.zeros_like(probs)
            target[np.arange(len(order)), y[order]] = 1.0
            error = (probs - target) / len(order)
            grad_w = Xn[order].T @ error + l2 * self.weights
            grad_b = error.sum(axis=0)
            self.weights -= learning_rate * grad_w
            self.bias -= learning_rate * grad_b
        pred = np.argmax(Xn @ self.weights + self.bias, axis=1)
        accuracy = float(np.mean(pred == y))
        metrics = {"imitation_accuracy": accuracy, "n_examples": int(len(X)), "parameter_count": self.parameter_count}
        self.validation_metrics.update(metrics)
        return metrics

    def to_artifact(self, *, feature_schema: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "artifact_type": "wai_ultra_coordination_head",
            "feature_schema": feature_schema,
            "action_registry": asdict(self.action_registry),
            "policy_weights": {
                "weights": self.weights.tolist(),
                "bias": self.bias.tolist(),
            },
            "normalization_data": {
                "feature_mean": None if self.feature_mean is None else self.feature_mean.tolist(),
                "feature_scale": None if self.feature_scale is None else self.feature_scale.tolist(),
            },
            "training_metadata": metadata,
            "validation_metrics": self.validation_metrics,
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "CoordinationHead":
        registry_payload = artifact["action_registry"]
        registry = ActionRegistry(
            version=registry_payload["version"],
            action_keys=tuple(registry_payload["action_keys"]),
        )
        weights = np.array(artifact["policy_weights"]["weights"], dtype=float)
        bias = np.array(artifact["policy_weights"]["bias"], dtype=float)
        norm = artifact.get("normalization_data", {})
        mean = norm.get("feature_mean")
        scale = norm.get("feature_scale")
        return cls(
            weights=weights,
            bias=bias,
            action_registry=registry,
            feature_mean=None if mean is None else np.array(mean, dtype=float),
            feature_scale=None if scale is None else np.array(scale, dtype=float),
            training_metadata=dict(artifact.get("training_metadata", {})),
            validation_metrics=dict(artifact.get("validation_metrics", {})),
        )

    def _normalize(self, encoded_state: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None:
            return encoded_state
        return (encoded_state - self.feature_mean) / self.feature_scale

    def _normalize_matrix(self, X: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None:
            return X
        return (X - self.feature_mean) / self.feature_scale


def action_key(role: str, expert_id: str, subtask_kind: str, control: str = "CONTINUE") -> str:
    return f"{role}:{expert_id}:{subtask_kind}:{control}"


def _softmax(values: np.ndarray) -> np.ndarray:
    centered = values - np.max(values)
    exp = np.exp(centered)
    return exp / np.sum(exp)


def _softmax_2d(values: np.ndarray) -> np.ndarray:
    centered = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / np.sum(exp, axis=1, keepdims=True)
