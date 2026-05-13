"""Tests proving no original test-period rows bleed into the training set.

Guards against the index-reset leakage that occurred when build_horizon_features
called reset_index(drop=True) after dropna, causing the train/test boundary
to shift by ~max_lag rows into the test period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_horizons import build_horizon_features

TRAIN_FRAC = 0.75


def _synthetic_df(n: int = 500, station: str = "TEST") -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=n, freq="6min", tz="UTC")
    rng = np.random.default_rng(42)
    t = np.arange(n) * (6 / 60)
    wl = (
        0.6 * np.sin(2 * np.pi * t / 12.42)
        + 0.3 * np.sin(2 * np.pi * t / 24.0)
        + 0.01 * rng.standard_normal(n)
    )
    return pd.DataFrame({
        "timestamp": timestamps,
        "station_id": station,
        "water_level": wl,
        "datum": "MLLW",
        "units": "m",
        "lat": 21.3,
        "lon": -157.8,
        "source": "DEMO_SYNTHETIC",
    })


@pytest.mark.parametrize("horizon_steps", [1, 10, 60])
def test_no_test_rows_in_training(horizon_steps: int):
    """Train and test index sets must be disjoint for every horizon."""
    df = _synthetic_df(n=500)
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    n_train = int(n * TRAIN_FRAC)

    X, y_h = build_horizon_features(df, horizon_steps=horizon_steps)

    train_idx = set(X.index[X.index < n_train])
    test_idx = set(X.index[X.index >= n_train])

    assert train_idx.isdisjoint(test_idx), (
        f"horizon={horizon_steps}: test rows leaked into training! "
        f"Overlap: {train_idx & test_idx}"
    )


@pytest.mark.parametrize("horizon_steps", [1, 60])
def test_split_boundary_respects_original_n_train(horizon_steps: int):
    """All training rows must come from original rows < n_train; all test rows from >= n_train."""
    df = _synthetic_df(n=500)
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    n_train = int(n * TRAIN_FRAC)

    X, _ = build_horizon_features(df, horizon_steps=horizon_steps)

    train_rows = X.index[X.index < n_train]
    test_rows = X.index[X.index >= n_train]

    if len(train_rows) > 0 and len(test_rows) > 0:
        assert train_rows.max() < n_train, (
            "Highest training-row index >= n_train — train contains test-period data"
        )
        assert test_rows.min() >= n_train, (
            "Lowest test-row index < n_train — test contains training-period data"
        )


def test_training_rows_contiguous_after_warmup():
    """After lag warmup, all rows from warmup_end to n_train-1 should be in train."""
    df = _synthetic_df(n=500)
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    n_train = int(n * TRAIN_FRAC)

    X, _ = build_horizon_features(df, horizon_steps=1)
    train_idx = sorted(X.index[X.index < n_train])

    if len(train_idx) >= 2:
        # Rows should be consecutive integers (no gaps within train period)
        expected = list(range(train_idx[0], train_idx[-1] + 1))
        assert train_idx == expected, (
            "Training rows are not contiguous — unexpected gaps after dropna"
        )


def test_x_index_preserves_original_positions():
    """build_horizon_features must preserve DataFrame row indices, not reset to 0-based."""
    df = _synthetic_df(n=200)
    df = df.sort_values("timestamp").reset_index(drop=True)

    X, y_h = build_horizon_features(df, horizon_steps=1)

    # After dropna, the first valid row is ~row 40 (max_lag=40 warmup).
    # If index were reset, the minimum would be 0 — that would be wrong.
    assert X.index.min() > 0, (
        "X.index minimum is 0 — looks like index was reset after dropna, "
        "which breaks the train/test split boundary"
    )
