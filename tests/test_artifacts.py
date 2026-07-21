"""Security and compatibility tests for non-executable model artifacts."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.artifacts import (
    ArtifactValidationError,
    decode_router_artifact,
    encode_router_artifact,
    load_json_artifact,
    save_json_artifact,
    JsonDecisionTreeClassifier,
)


def _router_artifact() -> dict:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    labels = np.asarray(["low", "low", "high", "high"])
    model = DecisionTreeClassifier(max_depth=2, random_state=42).fit(features, labels)
    return {
        "model": model,
        "feature_columns": ["level"],
        "classes": list(model.classes_),
        "config": {"min_training_rows": 4},
        "n_train": 4,
    }


def test_router_json_round_trip_preserves_predictions(tmp_path):
    artifact = _router_artifact()
    path = tmp_path / "router.json"
    save_json_artifact(path, encode_router_artifact(artifact), kind="learned_router")
    restored = decode_router_artifact(
        load_json_artifact(path, expected_kind="learned_router")
    )
    values = np.asarray([[0.5], [2.5]])
    assert restored["model"].predict(values).tolist() == artifact["model"].predict(values).tolist()
    assert np.allclose(
        restored["model"].predict_proba(values),
        artifact["model"].predict_proba(values),
    )


def test_artifact_loader_rejects_tampering(tmp_path):
    path = tmp_path / "coordinator.json"
    save_json_artifact(path, {"weights": [1.0]}, kind="coordinator_policy")
    envelope = json.loads(path.read_text())
    envelope["payload"]["weights"][0] = 999.0
    path.write_text(json.dumps(envelope))
    with pytest.raises(ArtifactValidationError, match="integrity"):
        load_json_artifact(path, expected_kind="coordinator_policy")


def test_artifact_loader_rejects_pickle_without_deserializing_it(tmp_path):
    path = tmp_path / "router.pkl"
    path.write_bytes(b"not-even-a-real-pickle")
    with pytest.raises(ArtifactValidationError, match="pickle"):
        load_json_artifact(path, expected_kind="learned_router")


def test_artifact_loader_rejects_wrong_kind_and_invalid_json(tmp_path):
    path = tmp_path / "artifact.json"
    save_json_artifact(path, {"safe": True}, kind="coordinator_policy")
    with pytest.raises(ArtifactValidationError, match="kind"):
        load_json_artifact(path, expected_kind="learned_router")
    path.write_text("not-json")
    with pytest.raises(ArtifactValidationError, match="UTF-8 JSON"):
        load_json_artifact(path, expected_kind="coordinator_policy")


def test_router_artifact_validates_tree_shape_and_feature_values():
    payload = encode_router_artifact(_router_artifact())["model"]
    malformed = dict(payload)
    malformed["children_right"] = []
    with pytest.raises(ArtifactValidationError, match="inconsistent"):
        JsonDecisionTreeClassifier(malformed)

    model = JsonDecisionTreeClassifier(payload)
    with pytest.raises(ArtifactValidationError, match="feature matrix"):
        model.predict([[1.0, 2.0]])
    with pytest.raises(ArtifactValidationError, match="non-finite"):
        model.predict([[float("nan")]])


def test_artifact_writer_rejects_non_finite_numbers(tmp_path):
    with pytest.raises(ArtifactValidationError, match="non-finite"):
        save_json_artifact(
            tmp_path / "bad.json",
            {"value": float("inf")},
            kind="coordinator_policy",
        )
