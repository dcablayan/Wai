"""Tests for generated summary helpers."""

from __future__ import annotations

from scripts.build_summary import _ablation_claims


def test_ablation_claims_are_derived_from_metrics():
    metrics = {
        "A": {
            "harmonics_only": {"r2": 0.95, "mae": 0.2},
            "full": {"r2": 0.98, "mae": 0.1},
        },
        "B": {
            "harmonics_only": {"r2": 0.99, "mae": 0.3},
            "full": {"r2": 0.97, "mae": 0.4},
        },
    }
    claims = _ablation_claims(metrics)
    assert claims["harmonics_only_r2_min"] == 0.95
    assert claims["harmonics_only_r2_max"] == 0.99
    assert claims["harmonics_only_all_ge_0_98"] is False
    assert claims["full_mae_better_than_harmonics_only_all"] is False
    assert "does not support" in claims["statement"]


def test_ablation_claims_change_when_metrics_change():
    metrics = {
        "A": {
            "harmonics_only": {"r2": 0.981, "mae": 0.2},
            "full": {"r2": 0.99, "mae": 0.1},
        },
        "B": {
            "harmonics_only": {"r2": 0.982, "mae": 0.3},
            "full": {"r2": 0.99, "mae": 0.2},
        },
    }
    claims = _ablation_claims(metrics)
    assert claims["harmonics_only_all_ge_0_98"] is True
    assert claims["full_mae_better_than_harmonics_only_all"] is True
    assert "0.9810 to 0.9820" in claims["statement"]
