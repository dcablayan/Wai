"""Dashboard forecast alignment tests."""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

from app import run_forecast  # noqa: E402


def test_dashboard_forecast_arrays_align_exactly():
    forecast = run_forecast("DEMO-HNL")
    keys = [
        "timestamps",
        "actual",
        "persistence_pred",
        "harmonic_pred",
        "gradboost_pred",
        "harmonic_lower",
        "harmonic_upper",
        "gradboost_lower",
        "gradboost_upper",
    ]
    lengths = {key: len(forecast[key]) for key in keys}
    assert len(set(lengths.values())) == 1, lengths
    assert lengths["timestamps"] > 0
    assert forecast["timestamps"].is_monotonic_increasing
    assert np.all(forecast["harmonic_lower"] <= forecast["harmonic_pred"])
    assert np.all(forecast["harmonic_upper"] >= forecast["harmonic_pred"])
    assert forecast["harmonic_coverage"]["n_samples"] == lengths["actual"]
    assert forecast["gradboost_coverage"]["n_samples"] == lengths["actual"]
