"""Tests for the rolling SkillStore and skill-aware combination."""

from __future__ import annotations

import pandas as pd
import pytest

from src.experts.base import ExpertForecast
from src.orchestration.combiner import ForecastCombiner
from src.orchestration.skill_store import SkillStore, horizon_bucket


def _fc(name, value, *, lower=None, upper=None, confidence=0.8):
    return ExpertForecast(
        model_name=name, forecast_time_utc=pd.Timestamp("2024-01-01T00:00:00Z"),
        target_time_utc=pd.Timestamp("2024-01-01T06:00:00Z"), horizon_minutes=360,
        predicted_water_level_m=value,
        lower_m=value - 0.1 if lower is None else lower,
        upper_m=value + 0.1 if upper is None else upper,
        confidence=confidence,
    )


def test_horizon_buckets():
    assert horizon_bucket(30) == "short"
    assert horizon_bucket(360) == "medium"
    assert horizon_bucket(1440) == "day"
    assert horizon_bucket(5000) == "long"


def test_sparse_data_falls_back_to_coarser_levels_and_prior():
    store = SkillStore(min_samples=5)
    # No data at all -> prior is returned with source "prior".
    est = store.estimate(expert="noaa_residual", station="S", horizon_minutes=360, regime="normal")
    assert est.source_level == "prior"
    assert est.mae == pytest.approx(0.12, abs=1e-9)  # noaa_residual prior
    assert est.sample_count == 0


def test_single_observation_is_not_treated_as_strong_evidence():
    store = SkillStore(min_samples=5, prior_strength=4.0)
    store.update(expert="noaa_residual", station="S", horizon_minutes=360, regime="normal",
                 abs_error=1.0, covered=False)
    est = store.estimate(expert="noaa_residual", station="S", horizon_minutes=360, regime="normal")
    # One huge error must be shrunk strongly toward the prior, not believed.
    assert est.mae < 0.4
    assert est.source_level in {"prior", "global", "horizon", "station_horizon"}


def test_skill_converges_with_enough_samples():
    store = SkillStore(min_samples=5, decay=0.7)
    for _ in range(40):
        store.update(expert="noaa_residual", station="S", horizon_minutes=360, regime="normal",
                     abs_error=0.05, covered=True)
    est = store.estimate(expert="noaa_residual", station="S", horizon_minutes=360, regime="normal")
    assert est.mae < 0.10
    assert est.sample_count >= 5
    assert est.source_level == "station_horizon_regime"


def test_skill_store_json_roundtrip(tmp_path):
    store = SkillStore()
    for i in range(10):
        store.update(expert="local_tide", station="S", horizon_minutes=120, regime="normal",
                     abs_error=0.1 + 0.01 * i, covered=True, latency_ms=0.5)
    path = tmp_path / "skill.json"
    store.save(path)
    loaded = SkillStore.load(path)
    a = store.estimate(expert="local_tide", station="S", horizon_minutes=120, regime="normal")
    b = loaded.estimate(expert="local_tide", station="S", horizon_minutes=120, regime="normal")
    assert a.mae == pytest.approx(b.mae)
    assert a.sample_count == b.sample_count


def test_skill_aware_combination_weights_stronger_expert():
    # Give 'good' much higher skill weight than 'bad'; weighted median should pick it.
    weights = {"good": 50.0, "bad": 0.5}
    combined = ForecastCombiner().combine(
        [_fc("good", 1.0), _fc("bad", 2.0)], method="weighted_median", weights=weights,
    )
    assert combined.forecast_m == pytest.approx(1.0)


def test_combination_drops_safe_fallback_from_valid_ensemble():
    combined = ForecastCombiner().combine(
        [_fc("noaa_residual", 1.0), _fc("safe_fallback", 5.0)], method="weighted_average",
    )
    assert "safe_fallback" not in combined.experts_used
    assert combined.forecast_m == pytest.approx(1.0)


def test_interval_floor_widens_single_expert_interval():
    base = ForecastCombiner().combine([_fc("noaa_residual", 1.0)])
    floored = ForecastCombiner().combine([_fc("noaa_residual", 1.0)], min_half_width_m=0.5)
    assert (floored.upper_m - floored.lower_m) > (base.upper_m - base.lower_m)
    assert floored.upper_m - floored.forecast_m == pytest.approx(0.5)
