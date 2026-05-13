"""Gradient boosting baseline model for Wai.

Uses scikit-learn HistGradientBoostingRegressor over the same 8-constituent
tidal harmonic feature matrix as HarmonicRidgeModel. No extra dependencies.

HistGradientBoostingRegressor captures non-linear interactions between tidal
constituents, lag features, and rolling statistics that Ridge cannot model.
It is still a supervised ML baseline, not a deep-learning model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.features.engineering import build_feature_frame, build_feature_matrix
from src.models.metrics import compute_metrics


class GradBoostModel:
    """HistGradientBoostingRegressor over the tidal feature matrix.

    Drop-in replacement for HarmonicRidgeModel — same DataFrame API,
    same feature pipeline, different estimator. Use for comparing
    linear vs non-linear skill on the same feature set.

    Notes
    -----
    - Requires scikit-learn >= 1.0 (HistGradientBoostingRegressor is stable).
    - No external libraries (XGBoost / LightGBM) required.
    - Default hyperparameters are reasonable but not grid-searched.
    """

    def __init__(
        self,
        max_iter: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
    ) -> None:
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._model: Optional[HistGradientBoostingRegressor] = None
        self._feature_cols: Optional[list] = None

    def fit(self, df: pd.DataFrame, target_col: str = "water_level") -> "GradBoostModel":
        X, y = build_feature_matrix(df, target_col)
        self._feature_cols = list(X.columns)
        self._model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
        )
        self._model.fit(X, y)
        return self

    def predict_on(self, df: pd.DataFrame, target_col: str = "water_level") -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict_on()")
        X, _ = build_feature_matrix(df, target_col)
        X = X[self._feature_cols]
        return self._model.predict(X)

    def predict_aligned(self, df: pd.DataFrame, target_col: str = "water_level") -> pd.DataFrame:
        """Return timestamps, actual values, and predictions on valid feature rows."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict_aligned()")
        feat = build_feature_frame(df, target_col=target_col)
        X = feat[self._feature_cols]
        pred = self._model.predict(X)
        return pd.DataFrame({
            "timestamp": feat["timestamp"].values,
            "actual": feat[target_col].values,
            "prediction": pred,
            "_source_row": feat["_source_row"].astype(int).values,
        })

    def evaluate(self, df: pd.DataFrame, target_col: str = "water_level") -> dict:
        if self._model is None:
            raise RuntimeError("Call fit() before evaluate()")
        X, y = build_feature_matrix(df, target_col)
        X = X[self._feature_cols]
        pred = self._model.predict(X)
        return compute_metrics(y.values, pred)
