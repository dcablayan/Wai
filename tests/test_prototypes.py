"""Tests for src/models/prototypes.py."""

import math
import pytest

from src.data.windowing import make_windows, temporal_split
from src.models.prototypes import (
    HarmonicNetPrototype,
    SurgeNetPrototype,
    TinyTidePrototype,
    TsunamiSentinelPrototype,
    WaveGRUPrototype,
    rmse,
)


def _sine_windows(n=300, lookback=24):
    import math as m
    series = [m.sin(2 * m.pi * i / 12.42) for i in range(n)]
    return make_windows(series, lookback=lookback, max_samples=200)


def test_rmse_perfect():
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_rmse_known_error():
    assert rmse([0.0, 0.0], [1.0, 1.0]) == pytest.approx(1.0)


def test_wavegru_fits_and_predicts():
    windows = _sine_windows()
    train, _, test = temporal_split(windows)
    model = WaveGRUPrototype(lookback=24)
    model.fit(train)
    score = model.evaluate(test)
    assert not math.isnan(score)
    assert score >= 0.0


def test_wavegru_beats_trivial_bound():
    windows = _sine_windows(n=500)
    train, val, test = temporal_split(windows)
    model = WaveGRUPrototype(lookback=24).fit(train + val)
    score = model.evaluate(test)
    # Sine wave is predictable; RMSE should be below 2 (amplitude is 1)
    assert score < 2.0


def test_harmonic_net_fits_and_predicts():
    windows = _sine_windows()
    train, _, test = temporal_split(windows)
    model = HarmonicNetPrototype(lookback=24).fit(train)
    score = model.evaluate(test)
    assert not math.isnan(score)


def test_tiny_tide_fits_and_predicts():
    windows = _sine_windows()
    train, _, test = temporal_split(windows)
    model = TinyTidePrototype(lookback=24, lr=0.001, epochs=1).fit(train)
    score = model.evaluate(test)
    assert not math.isnan(score)


def test_surge_net_returns_tuple():
    windows = _sine_windows(n=100)
    train, _, test = temporal_split(windows)
    model = SurgeNetPrototype(lookback=24).fit(train)
    result = model.predict(test[0])
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_tsunami_sentinel_returns_tuple():
    windows = _sine_windows(n=100)
    train, _, test = temporal_split(windows)
    model = TsunamiSentinelPrototype(lookback=24).fit(train)
    pred, flag = model.predict(test[0])
    assert isinstance(flag, bool)
    assert isinstance(pred, float)
