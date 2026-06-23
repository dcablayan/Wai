"""Evaluation utilities."""

from src.evaluation.coordination_trajectories import (
    audit_trajectory_dataset_for_leakage,
    build_coordination_trajectory_dataset,
)
from src.evaluation.coordinator_training import (
    CoordinatorTrainingConfig,
    train_coordinator_from_trajectories,
)
from src.evaluation.historical_replay import (
    EXHAUSTIVE,
    POLICY,
    HistoricalReplayConfig,
    run_historical_replay,
)
from src.evaluation.router_training import (
    ReplayAuditError,
    RouterPolicyEvaluation,
    RouterTrainingConfig,
    audit_replay_for_router_training,
    build_router_training_frame,
    evaluate_router_policies,
    train_router_from_replay,
)
from src.evaluation.ultra_evaluation import (
    evaluate_modes_on_contexts,
    summarize_mode_evaluation,
)

__all__ = [
    "CoordinatorTrainingConfig",
    "EXHAUSTIVE",
    "POLICY",
    "HistoricalReplayConfig",
    "ReplayAuditError",
    "RouterPolicyEvaluation",
    "RouterTrainingConfig",
    "audit_replay_for_router_training",
    "audit_trajectory_dataset_for_leakage",
    "build_router_training_frame",
    "build_coordination_trajectory_dataset",
    "evaluate_modes_on_contexts",
    "evaluate_router_policies",
    "run_historical_replay",
    "summarize_mode_evaluation",
    "train_coordinator_from_trajectories",
    "train_router_from_replay",
]
