"""Tests for Wai Ultra trajectory data and coordinator training."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.evaluation.coordination_trajectories import (
    audit_trajectory_dataset_for_leakage,
    build_coordination_trajectory_dataset,
)
from src.evaluation.coordinator_training import (
    CoordinatorTrainingConfig,
    sample_randomized_worker_pool,
    train_coordinator_from_trajectories,
)
from src.evaluation.trajectory_search import search_oracle_workflows


def _replay() -> pd.DataFrame:
    rows = []
    for idx, residual in enumerate([0.05, 0.30, 0.10, 0.40]):
        actual = 1.0 + residual
        rows.append({
            "forecast_origin_utc": f"2024-01-0{idx + 1}T00:00:00Z",
            "target_time_utc": f"2024-01-0{idx + 1}T06:00:00Z",
            "target_station_id": "HOHONU_TEST",
            "paired_noaa_station_id": "NOAA_TEST",
            "horizon_minutes": 360,
            "context_features": json.dumps({
                "horizon_minutes": 360,
                "hohonu_freshness_seconds": 300.0,
                "noaa_freshness_seconds": 300.0,
                "hohonu_qc_status": "pass",
                "noaa_qc_status": "verified",
                "recent_hohonu_trend_m_per_hour": 0.01,
                "recent_noaa_residual_m": residual,
                "noaa_residual_trend_m_per_hour": 0.02,
                "tide_phase": "rising",
            }),
            "expert_predictions": json.dumps({
                "local_tide": {
                    "status": "success",
                    "prediction_m": 1.0,
                    "lower_m": 0.85,
                    "upper_m": 1.15,
                    "confidence": 0.75,
                },
                "noaa_residual": {
                    "status": "success",
                    "prediction_m": 1.0 + residual,
                    "lower_m": 0.85 + residual,
                    "upper_m": 1.15 + residual,
                    "confidence": 0.80,
                },
                "regional_to_local_residual": {
                    "status": "success",
                    "prediction_m": 1.0 + 0.9 * residual,
                    "lower_m": 0.82 + 0.9 * residual,
                    "upper_m": 1.18 + 0.9 * residual,
                    "confidence": 0.70,
                },
                "safe_fallback": {
                    "status": "success",
                    "prediction_m": 1.0,
                    "lower_m": 0.7,
                    "upper_m": 1.3,
                    "confidence": 0.55,
                },
            }),
            "actual_m": actual,
            "error_by_expert": json.dumps({}),
            "missing_data_conditions": json.dumps({}),
            "max_hohonu_input_time_utc": f"2024-01-0{idx + 1}T00:00:00Z",
            "max_noaa_input_time_utc": f"2024-01-0{idx + 1}T00:00:00Z",
        })
    return pd.DataFrame(rows)


def test_oracle_search_considers_synthesis_and_peak_weighted_loss():
    row = _replay().iloc[1]
    workflows = search_oracle_workflows(
        expert_predictions=json.loads(row["expert_predictions"]),
        actual_m=float(row["actual_m"]),
        max_turns=5,
        keep_alternatives=3,
    )
    assert workflows[0].terminal is True
    assert workflows[0].final_candidate is not None
    assert any("pair:" in workflow.workflow_id for workflow in workflows)


def test_trajectory_dataset_excludes_future_labels_from_policy_state():
    dataset = build_coordination_trajectory_dataset(_replay(), max_turns=5)
    assert not dataset.empty
    audit_trajectory_dataset_for_leakage(dataset)
    names = set(dataset.iloc[0]["state_feature_names"])
    assert "actual_m" not in names
    assert "final_forecast_error" in dataset.columns


def test_coordinator_training_produces_shadow_artifact_and_report(tmp_path):
    dataset = build_coordination_trajectory_dataset(_replay(), max_turns=5)
    artifact_path = tmp_path / "coordinator.pkl"
    artifact, report = train_coordinator_from_trajectories(
        dataset,
        config=CoordinatorTrainingConfig(min_training_rows=4, epochs=10),
        artifact_path=artifact_path,
    )
    assert artifact_path.exists()
    assert artifact["artifact_type"] == "wai_ultra_coordination_head"
    assert artifact["feature_schema"]["version"] == "trajectory-replay-state-v1"
    assert report.forward_time_split is True
    assert report.training_data_hash
    assert "invalid_interval" in report.randomized_pool_conditions


def test_randomized_worker_pool_is_seeded_and_keeps_fallback():
    workers = ["local_tide", "noaa_residual", "safe_fallback"]
    first = sample_randomized_worker_pool(workers, seed=7, dropout_probability=0.5)
    second = sample_randomized_worker_pool(workers, seed=7, dropout_probability=0.5)
    assert first == second
    assert first["safe_fallback"] is True


def test_trajectory_leakage_audit_fails_closed():
    dataset = build_coordination_trajectory_dataset(_replay(), max_turns=5)
    bad = dataset.copy()
    bad.at[bad.index[0], "state_feature_names"] = ["actual_future_level"]
    with pytest.raises(ValueError, match="leakage-like"):
        audit_trajectory_dataset_for_leakage(bad)
