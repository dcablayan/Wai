"""Tests for src/data/windowing.py."""

import pytest

from src.data.windowing import make_windows, temporal_split


def _series(n=100):
    return [float(i) * 0.1 for i in range(n)]


def test_make_windows_length():
    series = _series(50)
    windows = make_windows(series, lookback=10)
    assert len(windows) == 50 - 10  # n - lookback (horizon=1)


def test_make_windows_values_shape():
    series = _series(50)
    w = make_windows(series, lookback=10)[0]
    assert len(w["values"]) == 10
    assert isinstance(w["target_value"], float)
    assert isinstance(w["target_time"], float)


def test_make_windows_max_samples():
    series = _series(200)
    windows = make_windows(series, lookback=24, max_samples=50)
    assert len(windows) == 50


def test_make_windows_too_short_returns_empty():
    series = _series(5)
    windows = make_windows(series, lookback=10)
    assert windows == []


def test_temporal_split_sizes():
    windows = make_windows(_series(500), lookback=24)
    train, val, test = temporal_split(windows, train_frac=0.70, val_frac=0.15)
    total = len(train) + len(val) + len(test)
    assert total == len(windows)
    assert len(train) > len(val)
    assert len(val) > 0
    assert len(test) > 0


def test_temporal_split_order_preserved():
    series = list(range(200))
    windows = make_windows([float(x) for x in series], lookback=10)
    train, val, test = temporal_split(windows)
    # Last value in train should come before first in val
    assert train[-1]["target_time"] < val[0]["target_time"]
    assert val[-1]["target_time"] < test[0]["target_time"]
