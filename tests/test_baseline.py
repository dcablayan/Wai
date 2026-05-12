"""Tests for WaveGRUModel adapter (src/models/baseline.py)."""

import math
import pandas as pd
import numpy as np
import pytest

from src.models.baseline import WaveGRUModel


def _synthetic_df(n: int = 300, station: str = "TEST") -> pd.DataFrame:
    """Small synthetic tidal series matching the Wai schema."""
    timestamps = pd.date_range("2024-01-01", periods=n, freq="6min", tz="UTC")
    t = np.arange(n) * (6 / 60)  # hours
    water_level = np.sin(2 * np.pi * t / 12.42) + 0.02 * np.random.default_rng(0).standard_normal(n)
    return pd.DataFrame({
        "timestamp": timestamps,
        "station_id": station,
        "water_level": water_level,
        "datum": "MLLW",
        "units": "m",
        "lat": 21.3,
        "lon": -157.8,
        "source": "DEMO_SYNTHETIC",
    })


def test_wavegru_model_fit_returns_self():
    df = _synthetic_df()
    model = WaveGRUModel(lookback=12)
    result = model.fit(df)
    assert result is model


def test_wavegru_model_predict_on_shape():
    df = _synthetic_df(n=200)
    train, test = df.iloc[:150], df.iloc[150:]
    model = WaveGRUModel(lookback=12).fit(train)
    preds = model.predict_on(test)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(test)


def test_wavegru_model_predict_requires_fit():
    model = WaveGRUModel(lookback=12)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_on(_synthetic_df(n=50))


def test_wavegru_model_evaluate_returns_metrics():
    df = _synthetic_df(n=200)
    train, test = df.iloc[:150], df.iloc[150:]
    model = WaveGRUModel(lookback=12).fit(train)
    metrics = model.evaluate(test)
    assert isinstance(metrics, dict)
    for key in ("mae", "rmse", "r2"):
        assert key in metrics
        assert not math.isnan(metrics[key])


def test_wavegru_model_evaluate_with_context():
    df = _synthetic_df(n=200)
    train, test = df.iloc[:150], df.iloc[150:]
    model = WaveGRUModel(lookback=12).fit(train)
    metrics = model.evaluate(test, context_df=train)
    assert "rmse" in metrics
    assert metrics["rmse"] >= 0.0


def test_wavegru_model_finite_predictions():
    df = _synthetic_df(n=200)
    train, test = df.iloc[:150], df.iloc[150:]
    model = WaveGRUModel(lookback=12).fit(train)
    preds = model.predict_on(test)
    assert np.all(np.isfinite(preds))
