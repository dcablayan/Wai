"""Training utilities for Wai Ultra coordinator policies."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.coordination_trajectories import (
    audit_trajectory_dataset_for_leakage,
    trajectory_data_hash,
)
from src.orchestration.coordinator_head import ActionRegistry, CoordinationHead


@dataclass(frozen=True)
class CoordinatorTrainingConfig:
    """Config for imitation warm-start training."""

    test_fraction: float = 0.25
    random_seed: int = 42
    learning_rate: float = 0.1
    epochs: int = 200
    min_training_rows: int = 4
    artifact_version: str = "wai-ultra-coordinator-v1"


@dataclass
class CoordinatorTrainingReport:
    """Validation summary for trained coordinator artifacts."""

    n_rows: int
    n_train: int
    n_test: int
    train_accuracy: float
    test_accuracy: float | None
    action_counts: dict[str, int]
    forward_time_split: bool
    station_held_out: bool
    event_stratified_counts: dict[str, int]
    randomized_pool_conditions: list[str]
    training_data_hash: str
    artifact_path: str | None = None


def train_coordinator_from_trajectories(
    trajectories: pd.DataFrame,
    *,
    config: CoordinatorTrainingConfig | None = None,
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], CoordinatorTrainingReport]:
    """Train the small coordination head from oracle workflow transitions."""

    cfg = config or CoordinatorTrainingConfig()
    audit_trajectory_dataset_for_leakage(trajectories)
    if len(trajectories) < cfg.min_training_rows:
        raise ValueError(f"Need at least {cfg.min_training_rows} trajectory rows")

    trajectories = trajectories.sort_values(["forecast_origin", "episode_id", "turn"]).reset_index(drop=True)
    X = np.array([np.array(values, dtype=float) for values in trajectories["encoded_state"]])
    y_keys = [str(value) for value in trajectories["selected_action"]]
    split = _forward_time_split(trajectories, cfg.test_fraction)
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    registry = ActionRegistry(
        version="wai-ultra-actions-v1",
        action_keys=tuple(sorted(set(y_keys))),
    )
    head = CoordinationHead.initialize(
        n_features=X.shape[1],
        action_registry=registry,
        seed=cfg.random_seed,
    )
    train_metrics = head.fit_imitation(
        X[train_idx],
        [y_keys[i] for i in train_idx],
        learning_rate=cfg.learning_rate,
        epochs=cfg.epochs,
        seed=cfg.random_seed,
    )
    test_accuracy = None
    if len(test_idx):
        predictions = _predict_keys(head, X[test_idx])
        test_accuracy = float(np.mean([pred == y_keys[i] for pred, i in zip(predictions, test_idx)]))
        head.validation_metrics["test_accuracy"] = test_accuracy

    training_hash = trajectory_data_hash(trajectories)
    metadata = {
        "artifact_version": cfg.artifact_version,
        "random_seed": cfg.random_seed,
        "training_data_hash": training_hash,
        "expert_registry_version": "wai-ultra-actions-v1",
    }
    feature_names = list(trajectories.iloc[0]["state_feature_names"])
    feature_schema = {
        "version": "trajectory-replay-state-v1",
        "feature_names": feature_names,
    }
    artifact = head.to_artifact(feature_schema=feature_schema, metadata=metadata)

    saved = None
    if artifact_path is not None:
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(artifact, handle)
        saved = str(path)

    report = CoordinatorTrainingReport(
        n_rows=int(len(trajectories)),
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
        train_accuracy=float(train_metrics["imitation_accuracy"]),
        test_accuracy=test_accuracy,
        action_counts={str(k): int(v) for k, v in trajectories["selected_action"].value_counts().sort_index().items()},
        forward_time_split=True,
        station_held_out=_has_station_holdout(trajectories, train_idx, test_idx),
        event_stratified_counts=_event_counts(trajectories),
        randomized_pool_conditions=randomized_expert_pool_conditions(),
        training_data_hash=training_hash,
        artifact_path=saved,
    )
    return artifact, report


def randomized_expert_pool_conditions() -> list[str]:
    """Conditions to sample during coordinator training/evaluation."""

    return [
        "hohonu_unavailable",
        "noaa_unavailable",
        "stale_local_data",
        "stale_regional_data",
        "failed_qc",
        "missing_tide_prediction",
        "weather_expert_unavailable",
        "spatial_expert_unavailable",
        "numerical_model_disabled",
        "expert_exception",
        "slow_expert",
        "invalid_interval",
    ]


def sample_randomized_worker_pool(
    available_workers: list[str],
    *,
    seed: int,
    dropout_probability: float = 0.2,
) -> dict[str, bool]:
    """Deterministically sample worker availability masks for robustness training."""

    rng = np.random.default_rng(seed)
    mask = {}
    for worker in available_workers:
        if worker == "safe_fallback":
            mask[worker] = True
        else:
            mask[worker] = bool(rng.random() >= dropout_probability)
    return mask


def _forward_time_split(trajectories: pd.DataFrame, test_fraction: float) -> dict[str, np.ndarray]:
    origins = pd.to_datetime(trajectories["forecast_origin"], utc=True)
    order = np.argsort(origins.to_numpy())
    cutoff = int(max(1, round(len(order) * (1.0 - test_fraction))))
    cutoff = min(cutoff, len(order))
    return {
        "train_idx": np.array(order[:cutoff], dtype=int),
        "test_idx": np.array(order[cutoff:], dtype=int),
    }


def _predict_keys(head: CoordinationHead, X: np.ndarray) -> list[str]:
    logits = X @ head.weights + head.bias
    indices = np.argmax(logits, axis=1)
    return [head.action_registry.action_keys[int(idx)] for idx in indices]


def _has_station_holdout(trajectories: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> bool:
    if "station_id" not in trajectories.columns or len(test_idx) == 0:
        return False
    train_stations = set(trajectories.iloc[train_idx]["station_id"])
    test_stations = set(trajectories.iloc[test_idx]["station_id"])
    return bool(test_stations - train_stations)


def _event_counts(trajectories: pd.DataFrame) -> dict[str, int]:
    if "peak_event_loss" not in trajectories:
        return {}
    event = trajectories["peak_event_loss"].fillna(0.0).astype(float) > 0.0
    return {
        "event": int(event.sum()),
        "ordinary": int((~event).sum()),
    }
