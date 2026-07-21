"""Training utilities for Wai Ultra coordinator policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.artifacts import save_json_artifact
from src.evaluation.coordination_trajectories import (
    audit_trajectory_dataset_for_leakage,
    trajectory_data_hash,
)
from src.orchestration.action_masks import default_expert_specs
from src.orchestration.coordinator_head import ActionRegistry, CoordinationHead
from src.orchestration.coordinator_policy import build_action_registry_from_specs
from src.orchestration.state_encoder import FeatureSchema


@dataclass(frozen=True)
class CoordinatorTrainingConfig:
    """Config for imitation warm-start training."""

    test_fraction: float = 0.25
    random_seed: int = 42
    learning_rate: float = 0.1
    epochs: int = 200
    min_training_rows: int = 4
    artifact_version: str = "wai-ultra-coordinator-v1"
    min_validation_accuracy: float = 0.01
    min_heldout_workflow_reward: float = -10.0
    max_routing_regret: float = 10.0
    max_mae: float = 10.0
    max_peak_event_error: float = 10.0
    min_interval_coverage: float = 0.0
    max_unavailable_rate: float = 1.0
    min_fallback_success_rate: float = 0.0
    max_invalid_action_rate: float = 0.0
    min_dropout_reward: float = -10.0


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
    validation_status: str
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
    _validate_live_feature_schema(trajectories)
    if len(trajectories) < cfg.min_training_rows:
        raise ValueError(f"Need at least {cfg.min_training_rows} trajectory rows")

    trajectories = trajectories.sort_values(["forecast_origin", "episode_id", "turn"]).reset_index(drop=True)
    X = np.array([np.array(values, dtype=float) for values in trajectories["encoded_state"]])
    y_keys = [str(value) for value in trajectories["selected_action"]]
    split = _forward_time_split(trajectories, cfg.test_fraction)
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    registry = build_action_registry_from_specs(default_expert_specs())
    unknown = sorted(set(y_keys) - set(registry.action_keys))
    if unknown:
        raise ValueError(f"Trajectory labels are not in the live action registry: {unknown}")
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
    validation_metrics = _validation_metrics(trajectories, train_idx, test_idx, test_accuracy)
    thresholds = _validation_thresholds(cfg)
    validation_status = _validation_status(validation_metrics, thresholds)
    head.validation_metrics.update(validation_metrics)
    metadata = {
        "artifact_version": cfg.artifact_version,
        "random_seed": cfg.random_seed,
        "training_data_hash": training_hash,
        "expert_registry_version": "wai-ultra-actions-v1",
        "validation_dataset_hash": training_hash,
        "validation_status": validation_status,
        "validation_thresholds": thresholds,
        "time_range": _time_range(trajectories),
        "station_split": _station_split(trajectories, train_idx, test_idx),
        "event_count": _event_counts(trajectories),
        "action_support": {str(k): int(v) for k, v in trajectories["selected_action"].value_counts().sort_index().items()},
        "split": {
            "type": "forward_time_grouped_by_forecast_origin",
            "train_origins": sorted(str(value) for value in trajectories.iloc[train_idx]["forecast_origin"].unique()),
            "test_origins": sorted(str(value) for value in trajectories.iloc[test_idx]["forecast_origin"].unique()),
        },
    }
    feature_names = list(FeatureSchema().feature_names)
    feature_schema = {
        "version": FeatureSchema().version,
        "feature_names": feature_names,
    }
    artifact = head.to_artifact(feature_schema=feature_schema, metadata=metadata)

    saved = None
    if artifact_path is not None:
        path = Path(artifact_path)
        save_json_artifact(path, artifact, kind="coordinator_policy")
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
        validation_status=validation_status,
        artifact_path=saved,
    )
    return artifact, report


def randomized_expert_pool_conditions() -> list[str]:
    """Conditions to sample during coordinator training/evaluation."""

    return [
        "worker_dropout",
        "missing_hohonu",
        "missing_noaa",
        "missing_tide_prediction",
        "stale_sources",
        "failed_qc",
        "worker_exception",
        "timeout",
        "invalid_interval",
        "disabled_synthesis",
        "disabled_verifier",
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
    origins = pd.DataFrame({
        "forecast_origin": pd.to_datetime(trajectories["forecast_origin"], utc=True),
        "row_origin": trajectories["forecast_origin"].astype(str),
    }).drop_duplicates("row_origin").sort_values("forecast_origin")
    if len(origins) <= 1:
        return {
            "train_idx": np.arange(len(trajectories), dtype=int),
            "test_idx": np.array([], dtype=int),
        }
    n_test = max(1, int(round(len(origins) * test_fraction)))
    n_test = min(n_test, len(origins) - 1)
    cutoff = len(origins) - n_test
    train_origins = set(origins.iloc[:cutoff]["row_origin"])
    test_origins = set(origins.iloc[cutoff:]["row_origin"])
    train_idx = np.array(
        [idx for idx, value in enumerate(trajectories["forecast_origin"].astype(str)) if value in train_origins],
        dtype=int,
    )
    test_idx = np.array(
        [idx for idx, value in enumerate(trajectories["forecast_origin"].astype(str)) if value in test_origins],
        dtype=int,
    )
    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
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


def _validate_live_feature_schema(trajectories: pd.DataFrame) -> None:
    schema = FeatureSchema()
    for idx, row in trajectories.iterrows():
        if row.get("feature_schema_version") != schema.version:
            raise ValueError(
                f"row {idx}: expected live feature schema {schema.version}, "
                f"got {row.get('feature_schema_version')}"
            )
        names = tuple(row.get("state_feature_names", ()))
        if names != schema.feature_names:
            raise ValueError(f"row {idx}: live feature names do not match StateEncoder schema")
        if len(row.get("encoded_state", ())) != len(schema.feature_names):
            raise ValueError(f"row {idx}: encoded state length does not match live schema")


def _validation_metrics(
    trajectories: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    test_accuracy: float | None,
) -> dict[str, Any]:
    holdout = trajectories.iloc[test_idx] if len(test_idx) else trajectories.iloc[train_idx]
    terminal = holdout[holdout["terminal"].astype(bool)] if "terminal" in holdout else holdout
    errors = pd.to_numeric(terminal.get("final_forecast_error", pd.Series(dtype=float)), errors="coerce").dropna()
    rewards = pd.to_numeric(terminal.get("final_reward", pd.Series(dtype=float)), errors="coerce").dropna()
    coverage = terminal.get("interval_coverage", pd.Series(dtype=object)).dropna()
    unavailable = terminal.get("final_forecast_error", pd.Series(dtype=float)).isna()
    invalid_actions = ~holdout.get("selected_action_feasible", pd.Series(True, index=holdout.index)).astype(bool)
    terminal_rewards = trajectories[trajectories["terminal"].astype(bool)] if "terminal" in trajectories else trajectories
    best_by_origin = terminal_rewards.groupby("forecast_origin")["final_reward"].max()
    regrets = []
    for _, row in terminal.iterrows():
        best = best_by_origin.get(row["forecast_origin"])
        if best is not None and pd.notna(row.get("final_reward")):
            regrets.append(float(best) - float(row["final_reward"]))
    dropout = terminal[terminal.get("randomized_condition", "").isin(["worker_dropout", "worker_exception", "timeout"])]
    fallback_terminal = terminal[
        terminal["result_summary"].apply(
            lambda value: "safe_fallback" in str(value)
        )
    ] if "result_summary" in terminal else terminal.iloc[0:0]
    fallback_success = (
        float((fallback_terminal["final_forecast_error"].notna()).mean())
        if len(fallback_terminal)
        else 1.0
    )
    peak = terminal[pd.to_numeric(terminal.get("peak_event_loss", pd.Series(dtype=float)), errors="coerce").fillna(0.0) > 0.0]
    peak_errors = pd.to_numeric(peak.get("final_forecast_error", pd.Series(dtype=float)), errors="coerce").abs().dropna()
    return {
        "test_accuracy": None if test_accuracy is None else float(test_accuracy),
        "heldout_workflow_reward": float(rewards.mean()) if len(rewards) else -10.0,
        "routing_regret": float(np.mean(regrets)) if regrets else 0.0,
        "mae": float(errors.abs().mean()) if len(errors) else 10.0,
        "peak_event_error": float(peak_errors.mean()) if len(peak_errors) else 0.0,
        "interval_coverage": float(coverage.astype(bool).mean()) if len(coverage) else 0.0,
        "unavailable_rate": float(unavailable.mean()) if len(terminal) else 1.0,
        "fallback_success_rate": fallback_success,
        "invalid_action_rate": float(invalid_actions.mean()) if len(holdout) else 1.0,
        "expert_dropout_reward": float(pd.to_numeric(dropout.get("final_reward", pd.Series(dtype=float)), errors="coerce").mean())
        if len(dropout)
        else float(rewards.mean()) if len(rewards) else -10.0,
        "n_holdout_rows": int(len(holdout)),
        "n_holdout_origins": int(holdout["forecast_origin"].nunique()) if "forecast_origin" in holdout else 0,
    }


def _validation_thresholds(cfg: CoordinatorTrainingConfig) -> dict[str, float]:
    return {
        "min_validation_accuracy": cfg.min_validation_accuracy,
        "min_heldout_workflow_reward": cfg.min_heldout_workflow_reward,
        "max_routing_regret": cfg.max_routing_regret,
        "max_mae": cfg.max_mae,
        "max_peak_event_error": cfg.max_peak_event_error,
        "min_interval_coverage": cfg.min_interval_coverage,
        "max_unavailable_rate": cfg.max_unavailable_rate,
        "min_fallback_success_rate": cfg.min_fallback_success_rate,
        "max_invalid_action_rate": cfg.max_invalid_action_rate,
        "min_dropout_reward": cfg.min_dropout_reward,
    }


def _validation_status(metrics: dict[str, Any], thresholds: dict[str, float]) -> str:
    test_accuracy = metrics.get("test_accuracy")
    if test_accuracy is not None and float(test_accuracy) < thresholds["min_validation_accuracy"]:
        return "shadow"
    checks = [
        metrics["heldout_workflow_reward"] >= thresholds["min_heldout_workflow_reward"],
        metrics["routing_regret"] <= thresholds["max_routing_regret"],
        metrics["mae"] <= thresholds["max_mae"],
        metrics["peak_event_error"] <= thresholds["max_peak_event_error"],
        metrics["interval_coverage"] >= thresholds["min_interval_coverage"],
        metrics["unavailable_rate"] <= thresholds["max_unavailable_rate"],
        metrics["fallback_success_rate"] >= thresholds["min_fallback_success_rate"],
        metrics["invalid_action_rate"] <= thresholds["max_invalid_action_rate"],
        metrics["expert_dropout_reward"] >= thresholds["min_dropout_reward"],
    ]
    return "validated" if all(checks) else "shadow"


def _time_range(trajectories: pd.DataFrame) -> dict[str, str | None]:
    if "forecast_origin" not in trajectories or trajectories.empty:
        return {"start": None, "end": None}
    origins = pd.to_datetime(trajectories["forecast_origin"], utc=True)
    return {"start": str(origins.min()), "end": str(origins.max())}


def _station_split(
    trajectories: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, list[str]]:
    if "station_id" not in trajectories:
        return {"train": [], "test": [], "held_out": []}
    train = sorted(str(value) for value in trajectories.iloc[train_idx]["station_id"].dropna().unique())
    test = sorted(str(value) for value in trajectories.iloc[test_idx]["station_id"].dropna().unique())
    return {
        "train": train,
        "test": test,
        "held_out": sorted(set(test) - set(train)),
    }
