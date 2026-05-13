"""Tests proving that features at row t do not include water_level[t].

These tests guard against target leakage in feature engineering:
- add_rolling_features must not include water_level[t] in the window at row t
- build_feature_matrix must not pass water_level[t] to X at the row predicting t
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    add_rolling_features,
    build_feature_matrix,
)


def _make_df(n: int = 100) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=n, freq="6min", tz="UTC")
    t = np.arange(n) * (6 / 60)
    wl = np.sin(2 * np.pi * t / 12.42) + 0.02 * np.random.default_rng(99).standard_normal(n)
    return pd.DataFrame({
        "timestamp": timestamps,
        "station_id": "TEST",
        "water_level": wl,
        "datum": "MLLW",
        "units": "m",
        "lat": 21.3,
        "lon": -157.8,
        "source": "DEMO_SYNTHETIC",
    })


# ── add_rolling_features: shift(1) guarantees no leakage ─────────────────────

def test_rolling_feature_excludes_current_row():
    """Perturbing water_level[t] must NOT change rmean at row t."""
    df = _make_df(n=80)
    target_row = 50

    df_perturbed = df.copy()
    df_perturbed.loc[target_row, "water_level"] = 1e6

    feat_orig = add_rolling_features(df.copy())
    feat_pert = add_rolling_features(df_perturbed)

    for col in ("water_level_rmean10", "water_level_rmean40", "water_level_rstd10"):
        assert feat_orig.loc[target_row, col] == pytest.approx(
            feat_pert.loc[target_row, col]
        ), f"{col} at row {target_row} changed when water_level[{target_row}] was perturbed — leakage!"


def test_rolling_feature_propagates_to_next_row():
    """Perturbing water_level[t] MUST change rmean at row t+1 (window includes t)."""
    df = _make_df(n=80)
    target_row = 50

    df_perturbed = df.copy()
    df_perturbed.loc[target_row, "water_level"] = 1e6

    feat_orig = add_rolling_features(df.copy())
    feat_pert = add_rolling_features(df_perturbed)

    assert feat_orig.loc[target_row + 1, "water_level_rmean10"] != pytest.approx(
        feat_pert.loc[target_row + 1, "water_level_rmean10"]
    ), "rmean at t+1 should reflect the perturbed value at t"


def test_rolling_std_excludes_current_row():
    """Perturbing water_level[t] must NOT change rstd at row t."""
    df = _make_df(n=80)
    target_row = 60

    df_perturbed = df.copy()
    df_perturbed.loc[target_row, "water_level"] = 1e6

    feat_orig = add_rolling_features(df.copy())
    feat_pert = add_rolling_features(df_perturbed)

    assert feat_orig.loc[target_row, "water_level_rstd10"] == pytest.approx(
        feat_pert.loc[target_row, "water_level_rstd10"]
    ), "rstd at row t changed when water_level[t] was perturbed — leakage!"


# ── build_feature_matrix: target leakage at the matrix level ─────────────────

def test_build_feature_matrix_no_target_leakage():
    """Perturbing water_level[t] must not alter features X at the same output row.

    After dropna + reset_index, original row 50 maps to row 10 in X
    (first 40 rows dropped due to lag-40 NaN warmup).
    """
    df = _make_df(n=100)
    target_orig_row = 50

    df2 = df.copy()
    df2.loc[target_orig_row, "water_level"] = 1e6

    X1, y1 = build_feature_matrix(df.copy())
    X2, y2 = build_feature_matrix(df2)

    max_lag = 40
    x_row = target_orig_row - max_lag  # row in X after warmup dropped

    # The target must have changed at that output row
    assert abs(y2.iloc[x_row] - y1.iloc[x_row]) > 1e5, (
        "Perturbing water_level at original row did not change y — wrong row mapping"
    )

    # All features at that output row must be IDENTICAL
    assert np.allclose(X1.iloc[x_row].values, X2.iloc[x_row].values), (
        "Features at output row changed when only the target water_level[t] was perturbed — leakage!"
    )


def test_build_feature_matrix_features_do_not_contain_target_col():
    """Ensure water_level is never a column in X."""
    df = _make_df()
    X, _ = build_feature_matrix(df)
    assert "water_level" not in X.columns


def test_build_feature_matrix_lag_features_are_shifted():
    """Lag-1 feature at row t must equal water_level[t-1], never water_level[t]."""
    df = _make_df(n=60)
    X, y = build_feature_matrix(df.copy())
    # Row 0 in X corresponds to original row 40 (after dropna)
    # lag1 at X-row 0 should equal water_level at original row 39 = y at X-row -1 (doesn't exist)
    # Check via direct value comparison across consecutive X rows
    for i in range(1, min(10, len(X))):
        assert X["water_level_lag1"].iloc[i] == pytest.approx(
            y.iloc[i - 1], abs=1e-9
        ), f"lag1[{i}] != y[{i-1}] — lag features are not properly shifted"
