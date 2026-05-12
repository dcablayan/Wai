"""Baseline and improved forecast models for Wai.

Models
------
PersistenceModel
    Naive baseline: predict the last observed value for all future steps.
    Useful as a floor to beat.

HarmonicRidgeModel
    Harmonic regression over the five major tidal constituents (M2, S2, K1,
    O1, N2) plus lagged observations and rolling statistics, fitted with
    Ridge regression.  Captures the dominant semi-diurnal/diurnal tidal
    signal and is easy to interpret and extend.

The code is structured so that substituting an LSTM or Transformer encoder
for the Ridge estimator requires only swapping the estimator inside
HarmonicRidgeModel.fit() — the feature pipeline stays the same.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.engineering import build_feature_matrix
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
    - To extend to LSTM/Transformer: replace the Pipeline with a torch/keras
      model and keep build_feature_matrix() for feature extraction.
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

    def evaluate(self, df: pd.DataFrame, target_col: str = "water_level") -> dict:
        """Return metrics dict for this model on the given DataFrame."""
        X, y = build_feature_matrix(df, target_col)
        X = X[self._feature_cols]
        pred = self._pipeline.predict(X)
        return compute_metrics(y.values, pred)
