"""Baseline and improved forecast models for Wai.

Models
------
PersistenceModel
    Naive baseline: predict the last observed value for all future steps.
    Useful as a floor to beat.

HarmonicRidgeModel
    Harmonic regression over eight tidal constituents (M2, S2, K1, O1, N2,
    M4, M6, Mm) plus temporal covariates, lags, and rolling statistics,
    fitted with Ridge regression.

WaveGRUModel
    DataFrame adapter wrapping WaveGRUPrototype (dcablayan/tideformer).
    Smoothing heuristic with attention-like weighting, not a real GRU.
    Operates on raw values; no feature engineering required.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.engineering import build_feature_frame, build_feature_matrix
from src.models.metrics import compute_metrics


class PersistenceModel:
    """Predict the last observed value for every future step."""

    def __init__(self) -> None:
        self._last: Optional[float] = None

    def fit(self, series: pd.Series) -> "PersistenceModel":
        self._last = float(series.dropna().iloc[-1])
        return self

    def predict(self, steps: int) -> np.ndarray:
        if self._last is None:
            raise RuntimeError("Call fit() before predict()")
        return np.full(steps, self._last)


class HarmonicRidgeModel:
    """Tidal harmonic regression with Ridge regularisation.

    Feature set: tidal constituent sin/cos pairs (M2, S2, K1, O1, N2),
    short-to-medium lag observations, and rolling mean/std windows.  These
    capture most of the predictable tidal signal for a 6-minute-resolution
    series without requiring deep learning infrastructure.

    Notes
    -----
    - This is NOT an advanced deep-learning model; it is a strong linear
      baseline that is honest about what it can predict.
    - MAE / RMSE on demo data are reported in reports/model_metrics.json.
    - The current project intentionally keeps this as a lightweight
      scikit-learn baseline.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._pipeline: Optional[Pipeline] = None
        self._feature_cols: Optional[list] = None

    def fit(self, df: pd.DataFrame, target_col: str = "water_level") -> "HarmonicRidgeModel":
        X, y = build_feature_matrix(df, target_col)
        self._feature_cols = list(X.columns)
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha)),
        ])
        self._pipeline.fit(X, y)
        return self

    def predict_on(self, df: pd.DataFrame, target_col: str = "water_level") -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Call fit() before predict_on()")
        X, _ = build_feature_matrix(df, target_col)
        X = X[self._feature_cols]
        return self._pipeline.predict(X)

    def predict_aligned(self, df: pd.DataFrame, target_col: str = "water_level") -> pd.DataFrame:
        """Return timestamps, actual values, and predictions on valid feature rows."""
        if self._pipeline is None:
            raise RuntimeError("Call fit() before predict_aligned()")
        feat = build_feature_frame(df, target_col=target_col)
        X = feat[self._feature_cols]
        pred = self._pipeline.predict(X)
        return pd.DataFrame({
            "timestamp": feat["timestamp"].values,
            "actual": feat[target_col].values,
            "prediction": pred,
            "_source_row": feat["_source_row"].astype(int).values,
        })

    def evaluate(self, df: pd.DataFrame, target_col: str = "water_level") -> dict:
        """Return metrics dict for this model on the given DataFrame."""
        X, y = build_feature_matrix(df, target_col)
        X = X[self._feature_cols]
        pred = self._pipeline.predict(X)
        return compute_metrics(y.values, pred)


class WaveGRUModel:
    """DataFrame adapter for WaveGRUPrototype (dcablayan/tideformer).

    Wraps the pure-Python double-exponential smoothing prototype
    so it fits the same DataFrame API as PersistenceModel and HarmonicRidgeModel.

    This is not a real GRU or deep-learning model. It operates on raw values
    only — no tidal feature engineering is applied.  It provides a useful
    complementary baseline: strong for short
    horizons, interpretable, dependency-free in its core implementation.
    """

    LOOKBACK = 24  # 24 × 6min = 144 min of context (matches tideformer benchmark)

    def __init__(self, lookback: int = LOOKBACK) -> None:
        self.lookback = lookback
        self._proto = None

    def fit(self, df: pd.DataFrame, target_col: str = "water_level") -> "WaveGRUModel":
        from src.data.windowing import make_windows
        from src.models.prototypes import WaveGRUPrototype

        series = df.sort_values("timestamp")[target_col].dropna().tolist()
        windows = make_windows(series, lookback=self.lookback)
        self._proto = WaveGRUPrototype(lookback=self.lookback)
        self._proto.fit(windows)
        self._train_series = series
        return self

    def predict_on(
        self,
        df: pd.DataFrame,
        target_col: str = "water_level",
        context_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Predict on df, prepending lookback context from context_df if given."""
        if self._proto is None:
            raise RuntimeError("Call fit() before predict_on()")
        from src.data.windowing import make_windows

        context = (
            context_df.sort_values("timestamp")[target_col].dropna().tolist()[-self.lookback:]
            if context_df is not None
            else self._train_series[-self.lookback:]
        )
        test_vals = df.sort_values("timestamp")[target_col].dropna().tolist()
        combined = context + test_vals

        preds = []
        for i in range(len(context), len(combined)):
            window = {
                "values": combined[i - self.lookback: i],
                "times": list(range(i - self.lookback, i)),
                "target_time": float(i),
                "target_value": combined[i],
            }
            preds.append(self._proto.predict(window))
        return np.array(preds)

    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = "water_level",
        context_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        pred = self.predict_on(df, target_col, context_df)
        actual = df.sort_values("timestamp")[target_col].dropna().values[:len(pred)]
        return compute_metrics(actual, pred)
