"""Architecture tests for Wai Ultra coordination."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data.hohonu import mock_hohonu_observations
from src.data.noaa import mock_noaa_observations, mock_noaa_tide_predictions
from src.data.station_mapping import StationPair
from src.experts.base import ExpertForecast, ForecastExpert
from src.experts.thinkers.residual_dynamics import ResidualDynamicsThinker
from src.forecasting import ForecastPipeline, default_experts
from src.orchestration.context import build_forecast_context
from src.orchestration.coordinator_policy import LearnedCoordinatorPolicy
from src.orchestration.action_masks import default_expert_specs, feasible_actions_for_state
from src.orchestration.bootstrap_policy import BootstrapCoordinatorPolicy
from src.orchestration.protocol import (
    CoordinationAction,
    CoordinationMessage,
    ExecutionBudget,
    MessageStatus,
    Role,
    RoleInput,
    SubtaskKind,
)
from src.orchestration.state_encoder import FeatureSchema, StateEncoder
from src.orchestration.ultra_conductor import UltraConductor
from src.orchestration.ultra_executor import UltraExecutor


def _context(
    *,
    horizon_minutes: int = 360,
    residual_m: float = 0.08,
    forecast_time: str = "2024-01-01T18:00:00Z",
    hohonu_qc: str = "pass",
):
    station_id = "HOHONU_TEST"
    noaa_id = "NOAA_TEST"
    return build_forecast_context(
        target_station_id=station_id,
        paired_noaa_station_id=noaa_id,
        horizon_minutes=horizon_minutes,
        forecast_time_utc=forecast_time,
        hohonu_observations=mock_hohonu_observations(station_id, periods=520, qc_status=hohonu_qc),
        noaa_observations=mock_noaa_observations(noaa_id, periods=520, residual_m=residual_m),
        noaa_tide_predictions=mock_noaa_tide_predictions(noaa_id, periods=620),
        station_pair=StationPair(station_id, noaa_id),
    )


def test_coordination_action_serializes_round_trip():
    action = CoordinationAction(
        turn_id=2,
        expert_id="ensemble_synthesis",
        role=Role.WORKER,
        subtask_kind=SubtaskKind.SYNTHESIZE_FORECASTS,
        access_list=[0, 1],
        policy_score=1.2,
        action_probability=0.7,
        rationale_for_audit="test",
    )
    restored = CoordinationAction.from_dict(action.to_dict())
    assert restored == action


def test_ultra_easy_conditions_use_worker_then_verifier_graph():
    result = ForecastPipeline(mode="ultra").run(
        _context(horizon_minutes=60, residual_m=0.08, forecast_time="2024-01-01T12:00:00Z")
    )
    assert result.status == "available"
    assert result.mode == "ultra"
    assert result.coordinator_policy_source == "bootstrap"
    assert result.role_sequence == ["WORKER", "VERIFIER"]
    assert result.termination_reason == "verifier_acceptance"
    assert result.executed_topology["final_accepted_node"] == 1
    assert result.physical_expert_calls == 1


def test_ultra_difficult_conditions_build_multi_turn_access_graph():
    result = ForecastPipeline(
        mode="ultra",
        ultra_budget=ExecutionBudget(max_turns=7, max_coordination_turns=7),
    ).run(_context(residual_m=0.40))
    assert result.status == "available"
    assert result.role_sequence == ["THINKER", "WORKER", "WORKER", "WORKER", "VERIFIER"]
    assert "ensemble_synthesis" in [node["expert_id"] for node in result.executed_topology["nodes"]]
    synthesis = [
        message for message in result.diagnostics["ultra"]["transcript"]
        if message["expert_id"] == "ensemble_synthesis"
    ][0]
    assert synthesis["structured_result"]["allowed_input_turns"] == synthesis["visible_prior_turns"]
    assert result.executed_topology["access_edges"]


def test_turn_two_changes_when_turn_one_thinker_output_changes():
    normal = ForecastPipeline(mode="ultra").run(_context(residual_m=0.08))
    event = ForecastPipeline(mode="ultra").run(_context(residual_m=0.40))
    assert normal.role_sequence[0] == "THINKER"
    assert event.role_sequence[0] == "THINKER"
    normal_second = normal.executed_topology["nodes"][1]["expert_id"]
    event_second = event.executed_topology["nodes"][1]["expert_id"]
    assert normal_second == "local_tide"
    assert event_second == "noaa_residual"


def test_verifier_choice_changes_with_worker_outputs_and_event_risk():
    easy = ForecastPipeline(mode="ultra").run(
        _context(horizon_minutes=60, residual_m=0.08, forecast_time="2024-01-01T12:00:00Z")
    )
    event = ForecastPipeline(
        mode="ultra",
        ultra_budget=ExecutionBudget(max_turns=7, max_coordination_turns=7),
    ).run(_context(residual_m=0.40))
    assert easy.executed_topology["nodes"][-1]["expert_id"] == "physics_datum_verifier"
    assert event.executed_topology["nodes"][-1]["expert_id"] == "event_risk_verifier"


def test_synthesis_consumes_only_explicit_access_list():
    context = _context(residual_m=0.40)
    experts = default_experts()
    conductor = UltraConductor(forecast_experts=experts)
    state = conductor.initialize_coordination_state(context)
    executor = UltraExecutor(experts)

    first = CoordinationAction(
        turn_id=0,
        expert_id="noaa_residual",
        role=Role.WORKER,
        subtask_kind=SubtaskKind.FORECAST_REGIONAL_RESIDUAL,
    )
    first_msg = executor.execute(first, state)
    state.append(first, first_msg)
    second = CoordinationAction(
        turn_id=1,
        expert_id="regional_to_local_residual",
        role=Role.WORKER,
        subtask_kind=SubtaskKind.TRANSFER_REGIONAL_SIGNAL,
    )
    second_msg = executor.execute(second, state)
    state.append(second, second_msg)
    synthesis = CoordinationAction(
        turn_id=2,
        expert_id="ensemble_synthesis",
        role=Role.WORKER,
        subtask_kind=SubtaskKind.SYNTHESIZE_FORECASTS,
        access_list=[1],
    )
    synth_msg = executor.execute(synthesis, state)
    assert synth_msg.structured_result["allowed_input_turns"] == [1]
    assert synth_msg.structured_result["forecast"] is None
    assert synth_msg.structured_result["worker_status"] == "unavailable"

    valid_synthesis = CoordinationAction(
        turn_id=2,
        expert_id="ensemble_synthesis",
        role=Role.WORKER,
        subtask_kind=SubtaskKind.SYNTHESIZE_FORECASTS,
        access_list=[0, 1],
    )
    valid_msg = executor.execute(valid_synthesis, state)
    assert valid_msg.structured_result["allowed_input_turns"] == [0, 1]
    assert sorted(valid_msg.structured_result["forecast"]["leaf_experts"]) == [
        "noaa_residual",
        "regional_to_local_residual",
    ]


def test_result_conditioned_state_changes_turn_two_action():
    context = _context(residual_m=0.08)
    conductor = UltraConductor(forecast_experts=default_experts())
    policy = BootstrapCoordinatorPolicy()
    encoder = StateEncoder()

    low_state = conductor.initialize_coordination_state(context)
    high_state = conductor.initialize_coordination_state(context)
    action = CoordinationAction(
        turn_id=0,
        expert_id="regime_difficulty_thinker",
        role=Role.THINKER,
        subtask_kind=SubtaskKind.DIAGNOSE_REGIME,
    )
    low_state.append(
        action,
        CoordinationMessage(
            turn_id=0,
            expert_id="regime_difficulty_thinker",
            role=Role.THINKER,
            subtask_kind=SubtaskKind.DIAGNOSE_REGIME,
            visible_prior_turns=[],
            status=MessageStatus.SUCCESS,
            structured_result={
                "forecast_difficulty": 0.05,
                "event_risk": 0.05,
                "recommended_experts": ["local_tide"],
            },
            latency_ms=0.0,
        ),
    )
    high_state.append(
        action,
        CoordinationMessage(
            turn_id=0,
            expert_id="regime_difficulty_thinker",
            role=Role.THINKER,
            subtask_kind=SubtaskKind.DIAGNOSE_REGIME,
            visible_prior_turns=[],
            status=MessageStatus.SUCCESS,
            structured_result={
                "forecast_difficulty": 0.9,
                "event_risk": 0.9,
                "recommended_experts": ["noaa_residual"],
            },
            latency_ms=0.0,
        ),
    )
    assert encoder.encode(low_state).tolist() != encoder.encode(high_state).tolist()
    low_action = policy.select_action(low_state, feasible_actions_for_state(low_state, default_expert_specs()))
    high_action = policy.select_action(high_state, feasible_actions_for_state(high_state, default_expert_specs()))
    assert low_action is not None
    assert high_action is not None
    assert low_action.expert_id == "local_tide"
    assert high_action.expert_id == "noaa_residual"


def test_residual_dynamics_handles_non_contiguous_indexes_and_alignment_gaps():
    base = pd.Timestamp("2024-01-01T00:00:00Z")
    context = SimpleNamespace(
        recent_noaa_residual_m=0.2,
        noaa_residual_trend_m_per_hour=0.04,
        recent_hohonu_trend_m_per_hour=0.0,
        station_pair=SimpleNamespace(residual_scale=1.0, lag_minutes=0),
        recent_noaa_observations=pd.DataFrame(
            {
                "timestamp_utc": [base, base + pd.Timedelta(minutes=6), base + pd.Timedelta(minutes=12)],
                "water_level_m": [1.0, 1.1, 1.2],
            },
            index=[10, 20, 30],
        ),
        noaa_tide_predictions=pd.DataFrame(
            {
                "timestamp_utc": [base, base + pd.Timedelta(minutes=6), base + pd.Timedelta(minutes=12)],
                "water_level_m": [0.9, 0.95, 1.0],
            },
            index=[100, 110, 120],
        ),
    )
    role_input = RoleInput(
        context=context,
        subtask_kind=SubtaskKind.ANALYZE_RESIDUAL_DYNAMICS,
        subtask_parameters={},
        visible_messages=[],
        remaining_turn_budget=5,
        remaining_physical_worker_calls=3,
        remaining_verifier_calls=3,
        remaining_deadline_ms=1000.0,
    )
    result = ResidualDynamicsThinker().analyze(role_input)
    assert result["alignment_support"] == 3
    assert result["status"] != "unavailable" if "status" in result else True

    context.noaa_tide_predictions["timestamp_utc"] = context.noaa_tide_predictions["timestamp_utc"] + pd.Timedelta(hours=1)
    unavailable = ResidualDynamicsThinker().analyze(role_input)
    assert unavailable["status"] == "unavailable"
    assert unavailable["alignment_support"] == 0


class SlowPersistenceExpert(ForecastExpert):
    model_name = "local_persistence"

    def forecast(self, context):
        time.sleep(0.05)
        return ExpertForecast(
            model_name=self.model_name,
            forecast_time_utc=context.forecast_time_utc,
            target_time_utc=context.target_time_utc,
            horizon_minutes=context.horizon_minutes,
            predicted_water_level_m=0.0,
            lower_m=-0.1,
            upper_m=0.1,
            confidence=0.5,
        )


def test_per_expert_timeout_is_reported_without_claiming_cancellation():
    context = _context(horizon_minutes=60, forecast_time="2024-01-01T12:00:00Z")
    experts = default_experts()
    experts["local_persistence"] = SlowPersistenceExpert()
    state = UltraConductor(
        forecast_experts=experts,
        budget=ExecutionBudget(per_expert_timeout_ms=5, deadline_ms=100),
    ).initialize_coordination_state(context)
    action = CoordinationAction(
        turn_id=0,
        expert_id="local_persistence",
        role=Role.WORKER,
        subtask_kind=SubtaskKind.FORECAST_LOCAL_LEVEL,
    )
    message = UltraExecutor(experts).execute(action, state)
    assert message.status.value == "timeout"
    assert any("may finish in the background" in warning for warning in message.warnings)


def test_duplicate_worker_action_reuses_physical_output_accounting():
    context = _context(residual_m=0.40)
    experts = default_experts()
    state = UltraConductor(forecast_experts=experts).initialize_coordination_state(context)
    executor = UltraExecutor(experts)
    first = CoordinationAction(
        turn_id=0,
        expert_id="noaa_residual",
        role=Role.WORKER,
        subtask_kind=SubtaskKind.FORECAST_REGIONAL_RESIDUAL,
    )
    state.append(first, executor.execute(first, state))
    second = CoordinationAction(
        turn_id=1,
        expert_id="noaa_residual",
        role=Role.WORKER,
        subtask_kind=SubtaskKind.FORECAST_REGIONAL_RESIDUAL,
    )
    second_msg = executor.execute(second, state)
    state.append(second, second_msg)
    telemetry = state.telemetry()
    assert second_msg.status.value == "reused"
    assert telemetry.physical_expert_calls == 1
    assert telemetry.reused_expert_outputs == 1


def test_pipeline_modes_remain_compatible():
    context = _context()
    mini = ForecastPipeline(mode="mini").run(context)
    legacy = ForecastPipeline(mode="legacy").run(context)
    ultra = ForecastPipeline(mode="ultra").run(context)
    assert mini.status == "available"
    assert legacy.status == "available"
    assert ultra.status == "available"
    assert mini.mode == "mini"
    assert legacy.mode == "legacy"
    assert ultra.mode == "ultra"


def test_learned_artifact_feature_schema_mismatch_fails_closed():
    artifact = {
        "feature_schema": {
            "version": "different",
            "feature_names": ["bias"],
        },
        "action_registry": {
            "version": "wai-ultra-actions-v1",
            "action_keys": ["WORKER:local_tide:FORECAST_LOCAL_LEVEL:CONTINUE"],
        },
        "policy_weights": {
            "weights": [[0.0]],
            "bias": [0.0],
        },
        "normalization_data": {
            "feature_mean": None,
            "feature_scale": None,
        },
        "training_metadata": {},
        "validation_metrics": {"imitation_accuracy": 1.0},
    }
    with pytest.raises(ValueError, match="feature schema mismatch"):
        LearnedCoordinatorPolicy.from_artifact(artifact)

    with pytest.raises(ValueError, match="feature schema mismatch"):
        FeatureSchema().validate_artifact_schema(artifact["feature_schema"])
