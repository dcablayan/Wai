"""Tests for GradBoostModel (src/models/gradient_boost.py)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.models.gradient_boost import GradBoostModel


def _synthetic_df(n: int = 300, station: str = "TEST") -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=n, freq="6min", tz="UTC")
    rng = np.random.default_rng(42)
    t = np.arange(n) * (6 / 60)
    water_level = (
        0.6 * np.sin(2 * np.pi * t / 12.42)
        + 0.3 * np.sin(2 * np.pi * t / 24.0)
        + 0.02 * rng.standard_normal(n)
    )
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


def test_gradboost_fit_returns_self():
    df = _synthetic_df()
    model = GradBoostModel()
    result = model.fit(df)
    assert result is model


def test_gradboost_predict_on_shape():
    df = _synthetic_df(n=200)
    train, test = df.iloc[:150], df.iloc[150:]
    model = GradBoostModel().fit(train)
    preds = model.predict_on(test)
    assert isinstance(preds, np.ndarray)
    # Feature matrix drops some rows due to lag NaN — length may be < len(test)
    assert len(preds) <= len(test)
    assert len(preds) > 0


def test_gradboost_predict_requires_fit():
    model = GradBoostModel()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_on(_synthetic_df(n=50))


def test_gradboost_evaluate_returns_metrics():
    df = _synthetic_df(n=200)
    train, test = df.iloc[:150], df.iloc[150:]
    model = GradBoostModel().fit(train)
    metrics = model.evaluate(test)
    assert isinstance(metrics, dict)
    for key in ("mae", "rmse", "r2"):
        assert key in metrics
        assert not math.isnan(metrics[key])


def test_gradboost_finite_predictions():
    df = _synthetic_df(n=200)
    train, test = df.iloc[:150], df.iloc[150:]
    model = GradBoostModel().fit(train)
    preds = model.predict_on(test)
    assert np.all(np.isfinite(preds))


def test_gradboost_beats_persistence_on_tidal_signal():
    """GradBoost should achieve lower MAE than naive persistence on a tidal series."""
    df = _synthetic_df(n=500)
    n_train = 375
    train, test = df.iloc[:n_train], df.iloc[n_train:]

    model = GradBoostModel().fit(train)
    gb_metrics = model.evaluate(test)

    from src.models.baseline import PersistenceModel
    from src.models.metrics import compute_metrics
    last_val = float(train["water_level"].iloc[-1])
    persist_pred = np.full(len(test), last_val)
    persist_metrics = compute_metrics(test["water_level"].values, persist_pred)

    assert gb_metrics["mae"] < persist_metrics["mae"], (
        f"GradBoost MAE {gb_metrics['mae']:.4f} should be < "
        f"Persistence MAE {persist_metrics['mae']:.4f}"
    )


def test_gradboost_evaluate_requires_fit():
    model = GradBoostModel()
    with pytest.raises(RuntimeError, match="fit"):
        model.evaluate(_synthetic_df(n=50))
