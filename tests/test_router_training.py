"""Tests for advisory learned-router training from replay rows."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.hohonu import mock_hohonu_observations
from src.data.noaa import mock_noaa_observations, mock_noaa_tide_predictions
from src.evaluation import HistoricalReplayConfig, run_historical_replay
from src.evaluation.router_training import (
    ReplayAuditError,
    RouterTrainingConfig,
    audit_replay_for_router_training,
    best_expert_label,
    build_router_training_frame,
    train_router_from_replay,
)
from src.orchestration.learned_router import LearnedRouter


def _replay_row(
    *,
    origin: str = "2024-01-01T00:00:00Z",
    target: str = "2024-01-01T06:00:00Z",
    tide_phase: str = "rising",
    residual: float = 0.1,
    errors: dict | None = None,
) -> dict:
    return {
        "forecast_origin_utc": origin,
        "target_time_utc": target,
        "target_station_id": "HOHONU_TEST",
        "paired_noaa_station_id": "NOAA_TEST",
        "horizon_minutes": 360,
        "context_features": json.dumps({
            "horizon_minutes": 360,
            "recent_noaa_residual_m": residual,
            "recent_hohonu_trend_m_per_hour": 0.01,
            "hohonu_freshness_seconds": 0.0,
            "noaa_freshness_seconds": 0.0,
            "hohonu_qc_status": "pass",
            "noaa_qc_status": "verified",
            "tide_phase": tide_phase,
        }),
        "selected_experts": json.dumps(["local_tide", "noaa_residual"]),
        "expert_predictions": json.dumps({}),
        "actual_m": 0.1,
        "error_by_expert": json.dumps(errors or {
            "local_tide": 0.2,
            "noaa_residual": 0.05,
            "safe_fallback": 0.2,
        }),
        "forecast_m": 0.15,
        "forecast_error_m": 0.05,
        "event_severity_m": 0.0,
        "missing_data_conditions": json.dumps({
            "missing_latest_hohonu": False,
            "missing_latest_noaa": False,
            "missing_tide_prediction": False,
            "hohonu_qc_ok": True,
            "noaa_qc_ok": True,
        }),
        "approx_compute_cost_ms": 1.0,
        "max_hohonu_input_time_utc": origin,
        "max_noaa_input_time_utc": origin,
        "result_status": "available",
    }


def test_best_expert_label_uses_smallest_absolute_error_with_stable_tie_break():
    assert best_expert_label({
        "safe_fallback": -0.1,
        "local_tide": 0.1,
        "noaa_residual": 0.3,
    }) == "local_tide"


def test_replay_audit_accepts_leakage_safe_rows():
    replay = pd.DataFrame([_replay_row()])
    report = audit_replay_for_router_training(replay)
    assert report.n_rows == 1
    assert report.n_violations == 0
    assert "context__recent_noaa_residual_m" in report.feature_columns_checked


def test_replay_audit_rejects_future_inputs():
    replay = pd.DataFrame([
        _replay_row(
            origin="2024-01-01T00:00:00Z",
            target="2024-01-01T06:00:00Z",
        )
    ])
    replay.loc[0, "max_hohonu_input_time_utc"] = "2024-01-01T00:06:00Z"
    with pytest.raises(ReplayAuditError, match="after forecast origin"):
        audit_replay_for_router_training(replay)


def test_replay_audit_rejects_leakage_like_feature_names():
    row = _replay_row()
    features = json.loads(row["context_features"])
    features["actual_m"] = 0.1
    row["context_features"] = json.dumps(features)
    with pytest.raises(ReplayAuditError, match="forbidden"):
        audit_replay_for_router_training(pd.DataFrame([row]))


def test_build_router_training_frame_derives_labels_and_features():
    replay = pd.DataFrame([
        _replay_row(errors={"local_tide": 0.2, "noaa_residual": 0.01}),
        _replay_row(tide_phase="falling", errors={"local_tide": 0.02, "noaa_residual": 0.3}),
    ])
    X, y, metadata = build_router_training_frame(replay)
    assert len(X) == 2
    assert list(y) == ["noaa_residual", "local_tide"]
    assert "context__recent_noaa_residual_m" in X.columns
    assert metadata.loc[0, "best_abs_error_m"] == pytest.approx(0.01)


def test_train_router_from_generated_replay_and_load_artifact(tmp_path):
    station_id = "HOHONU_TEST"
    noaa_id = "NOAA_TEST"
    replay = run_historical_replay(
        target_station_id=station_id,
        paired_noaa_station_id=noaa_id,
        hohonu_observations=mock_hohonu_observations(station_id, periods=720),
        noaa_observations=mock_noaa_observations(noaa_id, periods=720, residual_m=0.1),
        noaa_tide_predictions=mock_noaa_tide_predictions(noaa_id, periods=900),
        config=HistoricalReplayConfig(
            horizon_minutes=360,
            min_history_hours=12,
            step_minutes=240,
        ),
    )
    model_path = tmp_path / "router.json"
    report_path = tmp_path / "router_report.json"
    _, report = train_router_from_replay(
        replay,
        config=RouterTrainingConfig(min_training_rows=4),
        model_path=model_path,
        report_path=report_path,
    )
    assert model_path.exists()
    assert report_path.exists()
    assert report.n_rows >= 4
    assert report.train_accuracy >= 0.0

    router = LearnedRouter.load(model_path)
    context_features = json.loads(replay.iloc[0]["context_features"])
    missing_conditions = json.loads(replay.iloc[0]["missing_data_conditions"])
    prediction = router.predict_from_features(context_features, missing_conditions)
    assert prediction.recommended_expert in report.label_counts
    assert isinstance(prediction.probabilities, dict)
