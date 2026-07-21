"""Tests for Wai Ultra trajectory data and coordinator training."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.evaluation.coordination_trajectories import (
    audit_trajectory_dataset_for_leakage,
    build_coordination_trajectory_dataset,
    build_live_coordination_trajectory_dataset,
)
from src.evaluation.coordinator_training import (
    CoordinatorTrainingConfig,
    sample_randomized_worker_pool,
    train_coordinator_from_trajectories,
)
from src.evaluation.trajectory_search import search_oracle_workflows
from src.forecasting import ForecastPipeline
from src.orchestration.coordinator_policy import LearnedCoordinatorPolicy
from src.orchestration.state_encoder import FeatureSchema


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
    assert workflows[0].workflow_id.startswith("beam:")
    assert workflows[0].diagnostics["search_type"] == "bounded_live_state_beam"

    broad = search_oracle_workflows(
        expert_predictions=json.loads(row["expert_predictions"]),
        actual_m=float(row["actual_m"]),
        max_turns=7,
        keep_alternatives=500,
        beam_width=500,
    )
    assert any(
        any(action["expert_id"] == "ensemble_synthesis" for action in workflow.actions)
        for workflow in broad
    )


def test_oracle_search_is_independent_of_wall_clock(monkeypatch):
    row = _replay().iloc[1]
    predictions = json.loads(row["expert_predictions"])
    monkeypatch.setattr("src.evaluation.trajectory_search.time.monotonic", lambda: 10.0)
    monkeypatch.setattr("src.orchestration.coordination_state.time.monotonic", lambda: 10_000.0)
    first = search_oracle_workflows(
        expert_predictions=predictions,
        actual_m=float(row["actual_m"]),
        max_turns=7,
        keep_alternatives=50,
        beam_width=100,
    )
    monkeypatch.setattr("src.evaluation.trajectory_search.time.monotonic", lambda: 50_000.0)
    monkeypatch.setattr("src.orchestration.coordination_state.time.monotonic", lambda: 1_000_000.0)
    second = search_oracle_workflows(
        expert_predictions=predictions,
        actual_m=float(row["actual_m"]),
        max_turns=7,
        keep_alternatives=50,
        beam_width=100,
    )
    assert [workflow.workflow_id for workflow in first] == [
        workflow.workflow_id for workflow in second
    ]


def test_trajectory_dataset_excludes_future_labels_from_policy_state():
    dataset = build_coordination_trajectory_dataset(_replay(), max_turns=5)
    assert not dataset.empty
    audit_trajectory_dataset_for_leakage(dataset)
    names = set(dataset.iloc[0]["state_feature_names"])
    assert "actual_m" not in names
    assert "final_forecast_error" in dataset.columns


def test_coordinator_training_produces_shadow_artifact_and_report(tmp_path):
    dataset = build_coordination_trajectory_dataset(_replay(), max_turns=5)
    artifact_path = tmp_path / "coordinator.json"
    artifact, report = train_coordinator_from_trajectories(
        dataset,
        config=CoordinatorTrainingConfig(min_training_rows=4, epochs=10),
        artifact_path=artifact_path,
    )
    assert artifact_path.exists()
    assert artifact["artifact_type"] == "wai_ultra_coordination_head"
    assert artifact["feature_schema"]["version"] == FeatureSchema().version
    assert artifact["feature_schema"]["feature_names"] == list(FeatureSchema().feature_names)
    assert artifact["training_metadata"]["validation_status"] == "validated"
    assert report.forward_time_split is True
    assert report.validation_status == "validated"
    assert report.training_data_hash
    assert "worker_dropout" in dataset["randomized_condition"].unique()
    assert "invalid_interval" in report.randomized_pool_conditions


def test_train_to_live_artifact_round_trip_runs_ultra_pipeline(tmp_path):
    dataset = build_coordination_trajectory_dataset(_replay(), max_turns=5)
    artifact_path = tmp_path / "coordinator.json"
    train_coordinator_from_trajectories(
        dataset,
        config=CoordinatorTrainingConfig(min_training_rows=4, epochs=20),
        artifact_path=artifact_path,
    )
    policy = LearnedCoordinatorPolicy.load(artifact_path)
    replay_row = _replay().iloc[0:1]
    live_dataset = build_coordination_trajectory_dataset(replay_row, max_turns=5, keep_alternatives=1)
    # The pipeline run exercises the artifact through the live Ultra conductor.
    from src.evaluation.coordination_trajectories import _context_from_replay_row

    context = _context_from_replay_row(replay_row.iloc[0])
    result = ForecastPipeline(mode="ultra", ultra_policy=policy).run(context)
    assert result.mode == "ultra"
    assert result.status == "available"
    assert result.coordinator_policy_source == "learned"

    replay_with_policy = build_live_coordination_trajectory_dataset(
        [context],
        [float(replay_row.iloc[0]["actual_m"])],
        policy=policy,
        randomized_variants=0,
    )
    assert not replay_with_policy.empty
    assert replay_with_policy["selected_action_feasible"].all()
    assert live_dataset["feature_schema_version"].eq(FeatureSchema().version).all()


def test_forward_time_split_keeps_complete_origins_in_one_split(tmp_path):
    dataset = build_coordination_trajectory_dataset(_replay(), max_turns=5, keep_alternatives=3)
    artifact, report = train_coordinator_from_trajectories(
        dataset,
        config=CoordinatorTrainingConfig(min_training_rows=4, epochs=5),
        artifact_path=tmp_path / "coordinator.json",
    )
    assert report.forward_time_split is True
    split = artifact["training_metadata"]["split"]
    assert not set(split["train_origins"]) & set(split["test_origins"])


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
