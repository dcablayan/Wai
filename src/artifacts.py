"""Non-executable, integrity-checked model artifact helpers.

Wai artifacts are data, not Python programs.  JSON is deliberately used in
place of pickle so loading a tracked or downloaded artifact cannot execute
arbitrary constructors.  The small learned router uses a validated, portable
decision-tree representation; the NumPy coordinator is already JSON-native.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np


ARTIFACT_FORMAT = "wai.safe-json-artifact"
ARTIFACT_VERSION = 1
MAX_ARTIFACT_BYTES = 25 * 1024 * 1024


class ArtifactValidationError(ValueError):
    """Raised when an artifact is unsafe, corrupt, or schema-incompatible."""


class JsonDecisionTreeClassifier:
    """Minimal inference-only classifier reconstructed from validated arrays."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.classes_ = np.asarray(payload["classes"])
        self.n_features_in_ = int(payload["n_features_in"])
        self.children_left = np.asarray(payload["children_left"], dtype=np.int64)
        self.children_right = np.asarray(payload["children_right"], dtype=np.int64)
        self.feature = np.asarray(payload["feature"], dtype=np.int64)
        self.threshold = np.asarray(payload["threshold"], dtype=float)
        values = np.asarray(payload["value"], dtype=float)
        self.value = values.reshape((len(self.children_left), -1))
        self._validate()

    def _validate(self) -> None:
        n_nodes = len(self.children_left)
        arrays = (
            self.children_right,
            self.feature,
            self.threshold,
            self.value,
        )
        if n_nodes == 0 or any(len(value) != n_nodes for value in arrays):
            raise ArtifactValidationError("decision-tree arrays have inconsistent lengths")
        if len(self.classes_) == 0 or self.value.shape[1] != len(self.classes_):
            raise ArtifactValidationError("decision-tree class/value dimensions differ")
        if self.n_features_in_ <= 0:
            raise ArtifactValidationError("decision-tree feature count must be positive")
        if not np.isfinite(self.threshold).all() or not np.isfinite(self.value).all():
            raise ArtifactValidationError("decision-tree arrays contain non-finite values")
        for node, (left, right, feature) in enumerate(
            zip(self.children_left, self.children_right, self.feature)
        ):
            is_leaf = left == -1 and right == -1
            if is_leaf:
                continue
            if left < 0 or right < 0 or left >= n_nodes or right >= n_nodes:
                raise ArtifactValidationError(f"decision-tree node {node} has invalid children")
            if feature < 0 or feature >= self.n_features_in_:
                raise ArtifactValidationError(f"decision-tree node {node} has invalid feature")

    def predict_proba(self, values: Any) -> np.ndarray:
        rows = _as_feature_matrix(values, self.n_features_in_)
        probabilities = []
        for row in rows:
            node = self._leaf_for(row)
            counts = np.maximum(self.value[node], 0.0)
            total = float(np.sum(counts))
            if total <= 0.0:
                raise ArtifactValidationError("decision-tree leaf has no class mass")
            probabilities.append(counts / total)
        return np.asarray(probabilities, dtype=float)

    def predict(self, values: Any) -> np.ndarray:
        probabilities = self.predict_proba(values)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def _leaf_for(self, row: np.ndarray) -> int:
        node = 0
        visited: set[int] = set()
        while self.children_left[node] != -1:
            if node in visited:
                raise ArtifactValidationError("decision-tree artifact contains a cycle")
            visited.add(node)
            feature = int(self.feature[node])
            node = int(
                self.children_left[node]
                if row[feature] <= self.threshold[node]
                else self.children_right[node]
            )
        return node


def decision_tree_to_payload(model: Any) -> dict[str, Any]:
    """Serialize a fitted sklearn decision tree without executable objects."""

    tree = getattr(model, "tree_", None)
    classes = getattr(model, "classes_", None)
    n_features = getattr(model, "n_features_in_", None)
    if tree is None or classes is None or n_features is None:
        raise ArtifactValidationError("only fitted decision-tree classifiers are supported")
    return {
        "model_type": "decision_tree_classifier_v1",
        "classes": _json_value(np.asarray(classes).tolist()),
        "n_features_in": int(n_features),
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": tree.value.tolist(),
    }


def encode_router_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Convert an in-memory sklearn router artifact to JSON-safe data."""

    payload = {key: value for key, value in artifact.items() if key != "model"}
    payload["artifact_type"] = "wai_learned_router"
    payload["model"] = decision_tree_to_payload(artifact["model"])
    return _json_value(payload)


def decode_router_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and reconstruct an inference-only learned-router artifact."""

    if payload.get("artifact_type") != "wai_learned_router":
        raise ArtifactValidationError("unexpected learned-router artifact type")
    model_payload = payload.get("model")
    if not isinstance(model_payload, dict):
        raise ArtifactValidationError("learned-router artifact has no model payload")
    if model_payload.get("model_type") != "decision_tree_classifier_v1":
        raise ArtifactValidationError("unsupported learned-router model type")
    restored = dict(payload)
    restored["model"] = JsonDecisionTreeClassifier(model_payload)
    return restored


def save_json_artifact(path: str | Path, payload: dict[str, Any], *, kind: str) -> None:
    """Write an integrity-checked JSON envelope for an allowed artifact kind."""

    destination = _require_json_path(path)
    normalized = _json_value(payload)
    canonical = _canonical_json(normalized)
    envelope = {
        "format": ARTIFACT_FORMAT,
        "version": ARTIFACT_VERSION,
        "artifact_kind": str(kind),
        "payload_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "payload": normalized,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")


def load_json_artifact(
    path: str | Path,
    *,
    expected_kind: str,
) -> dict[str, Any]:
    """Load JSON only, verify its digest, and enforce the expected kind."""

    source = _require_json_path(path)
    if source.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ArtifactValidationError("artifact exceeds the 25 MiB safety limit")
    try:
        envelope = json.loads(source.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError("artifact is not valid UTF-8 JSON") from error
    if not isinstance(envelope, dict):
        raise ArtifactValidationError("artifact envelope must be a JSON object")
    if envelope.get("format") != ARTIFACT_FORMAT or envelope.get("version") != ARTIFACT_VERSION:
        raise ArtifactValidationError("unsupported artifact format or version")
    if envelope.get("artifact_kind") != expected_kind:
        raise ArtifactValidationError("artifact kind does not match the requested loader")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ArtifactValidationError("artifact payload must be a JSON object")
    expected_digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    if not _constant_time_equal(str(envelope.get("payload_sha256", "")), expected_digest):
        raise ArtifactValidationError("artifact integrity check failed")
    return payload


def _require_json_path(path: str | Path) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise ArtifactValidationError(
            "executable pickle/joblib artifacts are not accepted; use a .json artifact"
        )
    return destination


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _constant_time_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)


def _as_feature_matrix(values: Any, n_features: int) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    rows = np.asarray(values, dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] != n_features:
        raise ArtifactValidationError(
            f"expected a two-dimensional feature matrix with {n_features} columns"
        )
    if not np.isfinite(rows).all():
        raise ArtifactValidationError("feature matrix contains non-finite values")
    return rows


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(cast(Any, value)))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactValidationError("artifact contains a non-finite number")
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise ArtifactValidationError(f"artifact contains unsupported value {type(value).__name__}")
