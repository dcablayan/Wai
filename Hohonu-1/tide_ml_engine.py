"""Model-comparison and digital-environment utilities for tide prediction.

This module provides an "auto-ML" style search over several candidate
feature/pipeline configurations and produces a single-step-to-horizon forecast
using the best performer on rolling time-series validation.
"""

from __future__ import annotations

# Standard lib
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence

# Third-party
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

try:
    import tensorflow as tf
    from tensorflow import keras

    _HAS_TENSORFLOW = True
except Exception:
    tf = None
    keras = None
    _HAS_TENSORFLOW = False

try:
    import torch
    import torch.nn as torch_nn
    import torch.optim as torch_optim

    _HAS_TORCH = True
except Exception:
    torch = None
    torch_nn = None
    torch_optim = None
    _HAS_TORCH = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    _HAS_TS_MODELS = True
except Exception:
    ARIMA = None
    ExponentialSmoothing = None
    _HAS_TS_MODELS = False
from VAR_prediction import forecast_accuracy


DEFAULT_QA_METRIC_WEIGHTS = {
    "rmse_target": 0.40,
    "mae_target": 0.20,
    "mape_target": 0.10,
    "me_target": 0.07,
    "mpe_target": 0.03,
    "r2_target": 0.10,
    "corr_target": 0.05,
    "minmax_target": 0.05,
    "nse_target": 0.10,
}

MODEL_FAMILY_OPTIONS = (
    "ridge",
    "lasso",
    "elastic",
    "huber",
    "knn",
    "svr",
    "hgb",
    "mlp",
    "rf",
    "gbr",
    "extra",
    "arima",
    "ets",
    "exp_smooth",
    "lstm",
    "pinn",
)

MODEL_FAMILY_HELP_TEXT = (
    "Comma- or space-separated model families to include "
    "(ridge, lasso, elastic, huber, knn, svr, hgb, mlp, rf, gbr, extra, arima, ets, exp_smooth, lstm, pinn)."
)


@dataclass
class ModelCandidateResult:
    name: str
    lags: int
    model: str
    params: Dict
    rmse_target: float
    rmse_all: float
    mae_target: float
    mae_all: float
    rmse_colwise: Dict[str, float]
    mae_colwise: Dict[str, float]
    mape_target: float
    r2_target: float
    corr_target: float
    mpe_target: float
    me_target: float
    minmax_target: float
    nse_target: float
    estimator: object


def _forecast_series_metrics(forecast, actual):
    """Compute standardized target-only forecast metrics."""
    try:
        return forecast_accuracy(
            np.asarray(forecast).reshape(-1),
            np.asarray(actual).reshape(-1),
        )
    except Exception:
        return {
            "mape": float("nan"),
            "me": float("nan"),
            "mae": float("nan"),
            "mpe": float("nan"),
            "rmse": float("nan"),
            "r_square": float("nan"),
            "corr": float("nan"),
            "minmax": float("nan"),
            "nse": float("nan"),
        }


def _metric_or_nan(*values):
    for value in values:
        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
            return float(value)
    return float("nan")


def _coerce_qa_metric_weights(raw_weights=None):
    if not raw_weights:
        return dict(DEFAULT_QA_METRIC_WEIGHTS)

    alias = {
        "rmse": "rmse_target",
        "mae": "mae_target",
        "mape": "mape_target",
        "r2": "r2_target",
        "corr": "corr_target",
        "mpe": "mpe_target",
        "me": "me_target",
        "minmax": "minmax_target",
        "nse": "nse_target",
        "selected_rmse": "rmse_target",
        "selected_mae": "mae_target",
        "selected_mape": "mape_target",
        "selected_r2": "r2_target",
        "selected_corr": "corr_target",
        "selected_mpe": "mpe_target",
        "selected_me": "me_target",
        "selected_minmax": "minmax_target",
        "selected_nse": "nse_target",
        "qa_score": "selected_qa_score",
    }

    merged = dict(DEFAULT_QA_METRIC_WEIGHTS)
    for key, value in raw_weights.items():
        if key is None:
            continue
        normalized_key = str(key).strip().lower().replace("-", "_")
        normalized_key = alias.get(normalized_key, normalized_key)
        if isinstance(value, (int, float, np.number)):
            merged[normalized_key] = float(value)
    return merged


def _qa_score_from_metrics(
    metrics: Dict,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    score_parts = 0.0
    total_weight = 0.0

    metric_weights = _coerce_qa_metric_weights(weights)

    def read_metric(name):
        value = metrics.get(name)
        if value is None:
            value = metrics.get(f"selected_{name}")
        return _metric_or_nan(value)

    def add_lower(name, weight_key):
        value = read_metric(name)
        weight = metric_weights.get(weight_key, 0.0)
        if not np.isfinite(value) or weight <= 0:
            return
        score_parts += weight * (1.0 / (1.0 + abs(float(value))))
        nonlocal_total_weight(weight)

    def add_higher(name, weight_key):
        value = read_metric(name)
        weight = metric_weights.get(weight_key, 0.0)
        if not np.isfinite(value) or weight <= 0:
            return
        bounded = float(np.clip(value, -1.0, 1.0))
        score_parts += weight * ((bounded + 1.0) / 2.0)
        nonlocal_total_weight(weight)

    def nonlocal_total_weight(weight):
        nonlocal total_weight
        total_weight += weight

    add_lower("rmse_target", "rmse_target")
    add_lower("mae_target", "mae_target")
    add_lower("mape_target", "mape_target")
    add_lower("me_target", "me_target")
    add_lower("mpe_target", "mpe_target")
    add_higher("r2_target", "r2_target")
    add_higher("corr_target", "corr_target")
    add_higher("minmax_target", "minmax_target")
    add_higher("nse_target", "nse_target")

    if total_weight <= 0:
        return float("nan")
    return float(score_parts / total_weight)


def _weighted_candidate_metrics(
    items: List[ModelCandidateResult],
    weights=None,
    qa_metric_weights=None,
):
    """Aggregate selected metrics from model candidates using optional weights."""
    names = [
        ("rmse_target", "selected_rmse"),
        ("mae_target", "selected_mae"),
        ("mape_target", "selected_mape"),
        ("r2_target", "selected_r2"),
        ("corr_target", "selected_corr"),
        ("mpe_target", "selected_mpe"),
        ("me_target", "selected_me"),
        ("minmax_target", "selected_minmax"),
        ("nse_target", "selected_nse"),
    ]

    if not items:
        payload = {name: float("nan") for _, name in names}
        payload["selected_qa_score"] = _qa_score_from_metrics(payload, qa_metric_weights)
        return payload

    if weights is None:
        w = np.ones(len(items), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(items):
            w = np.ones(len(items), dtype=float)

    finite_mask = np.isfinite(w)
    if not finite_mask.any():
        w = np.ones(len(items), dtype=float)
        finite_mask = np.ones(len(items), dtype=bool)
    if not finite_mask.all():
        w = np.where(finite_mask, w, 0.0)
    if w.sum() <= 0:
        w = np.ones(len(items), dtype=float)
    w = w / np.sum(w)

    payload = {}
    for attr, out_key in names:
        vals = []
        vals_w = []
        for item, weight in zip(items, w):
            val = getattr(item, attr, float("nan"))
            if isinstance(val, (int, float, np.number)) and np.isfinite(val):
                vals.append(float(val))
                vals_w.append(weight)
        if not vals:
            payload[out_key] = float("nan")
        else:
            vals = np.asarray(vals, dtype=float)
            vals_w = np.asarray(vals_w, dtype=float)
            payload[out_key] = float(np.average(vals, weights=vals_w))
    payload["selected_qa_score"] = _qa_score_from_metrics(payload, qa_metric_weights)
    return payload


def _to_numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").astype(float)


def _coerce_ts_index(index_like):
    idx = pd.to_datetime(index_like)
    if idx.empty:
        return idx
    return idx


def _build_time_features(ts: pd.DatetimeIndex) -> np.ndarray:
    """Build cyclic time-of-day/week/year features from timestamps."""
    # Timestamps are in UTC or local timezone as provided.
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    doy = ts.dayofyear.to_numpy()
    month = ts.month.to_numpy()

    hour_rad = 2 * np.pi * (hour / 24.0)
    dow_rad = 2 * np.pi * (dow / 7.0)
    doy_rad = 2 * np.pi * (doy / 365.25)
    month_rad = 2 * np.pi * (month / 12.0)

    return np.column_stack(
        [
            np.sin(hour_rad),
            np.cos(hour_rad),
            np.sin(dow_rad),
            np.cos(dow_rad),
            np.sin(doy_rad),
            np.cos(doy_rad),
            np.sin(month_rad),
            np.cos(month_rad),
        ]
    )


def _build_sequence_matrix(
    values: np.ndarray,
    index: pd.DatetimeIndex,
    n_lags: int,
    include_cyclic_features: bool = True,
):
    """Create 3D lag tensor for recurrent models: (samples, lags, features)."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        values = values.reshape(-1, 1)

    cyc = _build_time_features(_coerce_ts_index(index)) if include_cyclic_features else None

    X = []
    y = []
    for t in range(n_lags, len(values)):
        lag_block = values[t - n_lags : t]
        if include_cyclic_features:
            if cyc is None or len(cyc) == 0:
                return None, None
            cyc_block = cyc[t - n_lags : t]
            lag_features = np.concatenate([lag_block, cyc_block], axis=1)
        else:
            lag_features = lag_block
        X.append(lag_features)
        y.append(values[t])

    if len(X) == 0:
        return np.empty((0, max(n_lags, 1), values.shape[1])), np.empty((0, values.shape[1]))

    return np.asarray(X), np.asarray(y)


def _coerce_arima_order(raw):
    if raw is None:
        return (1, 0, 0)
    if isinstance(raw, str):
        try:
            parts = [
                part.strip()
                for part in raw.replace(";", ",").replace(" ", ",").split(",")
                if part.strip()
            ]
            raw = tuple(int(float(part)) for part in parts[:3])
        except Exception:
            return (1, 0, 0)
    if (
        isinstance(raw, (tuple, list))
        and len(raw) == 3
        and all(isinstance(v, (int, np.integer, float, np.floating)) for v in raw)
    ):
        return tuple(int(v) for v in raw)
    return (1, 0, 0)


def _extract_lag_matrix(
    row: np.ndarray,
    lags: int,
    target_dim: int,
    include_cyclic_features: bool,
):
    row = np.asarray(row, dtype=float).reshape(-1)
    if row.size == 0 or target_dim <= 0 or lags <= 0:
        return None

    feature_width = lags * target_dim
    if include_cyclic_features and row.size > feature_width:
        row = row[:feature_width]

    if row.size < feature_width:
        row = np.pad(row, (feature_width - row.size, 0), constant_values=np.nan)

    row = row[:feature_width]
    matrix = row.reshape(-1, target_dim)
    if matrix.size == 0:
        return None
    return matrix[-lags:, :]


if _HAS_TS_MODELS:
    class _ARIMAForecaster:
        _model_type = "arima"

        def __init__(
            self,
            lags: int,
            target_dim: int,
            order=(1, 0, 0),
            include_cyclic_features: bool = False,
            random_state: int = 0,
        ):
            self.lags = int(lags)
            self.target_dim = int(target_dim)
            self.order = tuple(_coerce_arima_order(order))
            self.include_cyclic_features = bool(include_cyclic_features)
            self.random_state = int(random_state)
            self._fallback_last = np.zeros(self.target_dim, dtype=float) if self.target_dim > 0 else np.array([])
            self._models = []

        def fit(self, X, y):
            y = np.asarray(y, dtype=float)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if y.size == 0:
                raise ValueError("ARIMA forecaster needs non-empty training data.")
            self.target_dim = int(y.shape[1])
            self._fallback_last = np.zeros(self.target_dim, dtype=float)
            self._models = []

            for col in range(self.target_dim):
                series = pd.Series(y[:, col]).replace([np.inf, -np.inf], np.nan)
                series = series.ffill().bfill().to_numpy(dtype=float)
                if series.size == 0:
                    self._fallback_last[col] = 0.0
                    self._models.append(None)
                    continue

                last = series[np.isfinite(series)]
                if last.size == 0:
                    self._fallback_last[col] = 0.0
                    self._models.append(None)
                    continue
                self._fallback_last[col] = float(last[-1])
                if len(series) < max(self.order) + 3:
                    self._models.append(None)
                    continue

                try:
                    model = ARIMA(series, order=self.order).fit(disp=False)
                except Exception:
                    model = None
                self._models.append(model)
            return self

        def _predict_with_tail(self, models, tail, col):
            model = models[col]
            fallback = self._fallback_last[col]
            if model is None:
                return fallback

            try:
                state = model
                if tail is not None and tail.size > 0:
                    tail_vec = tail[:, col]
                    tail_vec = np.asarray(tail_vec, dtype=float)
                    tail_valid = tail_vec[np.isfinite(tail_vec)]
                    if len(tail_valid) > 0:
                        state = state.append(tail_valid.tolist(), refit=False)
                pred = state.forecast(1)
                return float(np.asarray(pred).reshape(-1)[0])
            except Exception:
                return fallback

        def predict(self, X):
            X = np.asarray(X, dtype=float)
            single = X.ndim == 1
            if single:
                X = X.reshape(1, -1)
            predictions = []
            for row in X:
                tail = _extract_lag_matrix(
                    row,
                    lags=self.lags,
                    target_dim=self.target_dim,
                    include_cyclic_features=self.include_cyclic_features,
                )
                row_pred = []
                for col in range(self.target_dim):
                    value = self._predict_with_tail(self._models, tail, col)
                    if not np.isfinite(value):
                        value = self._fallback_last[col]
                    row_pred.append(float(value))
                predictions.append(row_pred)
            return np.asarray(predictions, dtype=float) if not single else np.asarray(predictions[0], dtype=float)


if _HAS_TS_MODELS:
    class _ExpSmoothingForecaster:
        _model_type = "exp_smooth"

        def __init__(
            self,
            lags: int,
            target_dim: int,
            seasonal_periods: Optional[int] = None,
            trend: Optional[str] = None,
            seasonal: Optional[str] = None,
            include_cyclic_features: bool = False,
            random_state: int = 0,
        ):
            self.lags = int(lags)
            self.target_dim = int(target_dim)
            self.seasonal_periods = None if not seasonal_periods else int(seasonal_periods)
            self.trend = trend
            self.seasonal = seasonal
            self.include_cyclic_features = bool(include_cyclic_features)
            self.random_state = int(random_state)
            self._models = []
            self._fallback_last = np.zeros(self.target_dim, dtype=float) if self.target_dim > 0 else np.array([])

        def fit(self, X, y):
            y = np.asarray(y, dtype=float)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if y.size == 0:
                raise ValueError("Exponential smoothing forecaster needs non-empty training data.")

            self.target_dim = int(y.shape[1])
            self._models = []
            self._fallback_last = np.zeros(self.target_dim, dtype=float)
            for col in range(self.target_dim):
                series = pd.Series(y[:, col]).replace([np.inf, -np.inf], np.nan)
                series = series.ffill().bfill().to_numpy(dtype=float)
                if series.size == 0:
                    self._fallback_last[col] = 0.0
                    self._models.append(None)
                    continue

                finite = series[np.isfinite(series)]
                if finite.size == 0:
                    self._fallback_last[col] = 0.0
                    self._models.append(None)
                    continue
                self._fallback_last[col] = float(finite[-1])

                seasonal_periods = self.seasonal_periods
                if seasonal_periods is not None and seasonal_periods > len(finite) // 3:
                    seasonal_periods = None

                try:
                    model = ExponentialSmoothing(
                        finite,
                        trend=self.trend,
                        seasonal=self.seasonal,
                        seasonal_periods=seasonal_periods,
                    ).fit(optimized=True)
                except Exception:
                    model = None
                self._models.append(model)
            return self

        def predict(self, X):
            X = np.asarray(X, dtype=float)
            single = X.ndim == 1
            if single:
                X = X.reshape(1, -1)
            preds = []
            for _ in X:
                row_pred = []
                for col, model in enumerate(self._models):
                    if model is None:
                        value = self._fallback_last[col]
                    else:
                        try:
                            value = float(np.asarray(model.forecast(1)).reshape(-1)[0])
                        except Exception:
                            value = self._fallback_last[col]
                    if not np.isfinite(value):
                        value = self._fallback_last[col]
                    row_pred.append(float(value))
                preds.append(row_pred)
            arr = np.asarray(preds, dtype=float)
            if single:
                return arr[0]
            return arr


if _HAS_TENSORFLOW:
    class _LSTMForecaster:
        """Small wrapped Keras sequence model for one-step iterative forecasting."""

        _model_type = "lstm"

        def __init__(
            self,
            lags: int,
            features: int,
            target_dim: int,
            include_cyclic_features: bool = False,
            hidden_units: int = 48,
            epochs: int = 8,
            batch_size: int = 32,
            random_state: int = 0,
        ):
            self.lags = int(lags)
            self.features = int(features)
            self.target_dim = int(target_dim)
            self.include_cyclic_features = bool(include_cyclic_features)
            self.hidden_units = int(hidden_units)
            self.epochs = int(epochs)
            self.batch_size = int(batch_size)
            self.random_state = int(random_state)
            self.model = None

        def fit(self, X, y):
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
            if X.ndim != 3:
                raise ValueError("LSTM forecaster expects 3D sequence input.")

            if X.shape[2] != self.features:
                raise ValueError("LSTM input feature mismatch.")

            keras.backend.clear_session()
            if hasattr(tf.random, "set_seed"):
                tf.random.set_seed(self.random_state)

            layer_units = max(8, int(self.hidden_units))
            model = keras.Sequential(
                [
                    keras.layers.Input(shape=(self.lags, self.features)),
                    keras.layers.LSTM(layer_units),
                    keras.layers.Dense(self.target_dim),
                ]
            )
            model.compile(optimizer="adam", loss="mse")
            model.fit(
                X,
                y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0,
                shuffle=False,
            )
            self.model = model
            return self

        def predict(self, X):
            if self.model is None:
                raise RuntimeError("Model not fitted.")
            X = np.asarray(X, dtype=float)
            if X.ndim == 2:
                # Allow a flattened row by reshaping from flattened lags to sequence
                X = X.reshape(1, self.lags, max(1, int(X.shape[1] / self.lags)))
            return self.model.predict(X, verbose=0)


if _HAS_TORCH:
    class _PinnForecaster:
        """Minimal PINN-style single-output regularized regressor via differentiable model."""

        _model_type = "pinn"

        def __init__(
            self,
            lags: int,
            target_idx: int = -1,
            hidden_units: int = 64,
            hidden_layers: int = 2,
            physics_weight: float = 0.2,
            epochs: int = 80,
            lr: float = 1e-3,
            random_state: int = 0,
        ):
            self.lags = int(lags)
            self.target_idx = int(target_idx)
            self.hidden_units = int(hidden_units)
            self.hidden_layers = int(hidden_layers)
            self.physics_weight = float(physics_weight)
            self.epochs = int(epochs)
            self.lr = float(lr)
            self.random_state = int(random_state)
            self.model = None
            self.n_vars = None
            self.with_cyclic = False

        class _MLP(torch_nn.Module):
            def __init__(self, in_dim, out_dim, hidden=64, layers=2):
                super().__init__()
                body = []
                dim = in_dim
                for _ in range(max(1, layers)):
                    body.extend([torch_nn.Linear(dim, hidden), torch_nn.Tanh()])
                    dim = hidden
                body.append(torch_nn.Linear(dim, out_dim))
                self.net = torch_nn.Sequential(*body)

            def forward(self, x):
                return self.net(x)

        def fit(self, X, y):
            if torch is None:
                raise RuntimeError("Torch missing.")

            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            n_features = X.shape[1]
            n_vars = y.shape[1]
            if n_vars <= 0:
                raise ValueError("PINN requires at least one target variable.")
            if self.lags <= 0:
                raise ValueError("PINN lags must be > 0.")
            target_idx = self.target_idx
            if target_idx < 0:
                target_idx = n_vars + target_idx
            if target_idx < 0 or target_idx >= n_vars:
                raise ValueError("PINN target_idx is outside target dimension.")

            torch.manual_seed(self.random_state)
            net = self._MLP(n_features, n_vars)
            optimizer = torch_optim.Adam(net.parameters(), lr=self.lr)
            loss_fn = torch_nn.MSELoss()

            x = torch.tensor(X, dtype=torch.float32)
            t = torch.tensor(y, dtype=torch.float32)

            lag_target_idx = (self.lags - 1) * n_vars + target_idx
            if lag_target_idx >= n_features:
                lag_target_idx = n_features - 1
            for _ in range(self.epochs):
                pred = net(x)
                mse = loss_fn(pred, t)

                # Physics-style smoothness prior on target trajectory.
                lag_term = x[:, lag_target_idx]
                target_pred = pred[:, target_idx]
                phys = torch.mean((target_pred - lag_term) ** 2)
                loss = mse + self.physics_weight * phys
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model = net
            self.n_vars = int(n_vars)
            self._target_idx = target_idx
            return self

        def predict(self, X):
            if self.model is None:
                raise RuntimeError("Model not fitted.")
            X = np.asarray(X, dtype=float)
            with torch.no_grad():
                pred = self.model(torch.tensor(X, dtype=torch.float32)).numpy()
            return pred


def _build_supervised_matrix(
    values: np.ndarray,
    index: pd.DatetimeIndex,
    n_lags: int,
    include_cyclic_features: bool = True,
):
    """Create X,y matrices for one-step-ahead forecasting.

    y_t predicts all series at time t using data from times t-1...t-n_lags.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        values = values.reshape(-1, 1)

    if include_cyclic_features:
        cyc = _build_time_features(index)

    X = []
    y = []
    for t in range(n_lags, len(values)):
        lag_block = values[t - n_lags : t].reshape(-1)
        if include_cyclic_features:
            feats = np.concatenate([lag_block, cyc[t]])
        else:
            feats = lag_block
        X.append(feats)
        y.append(values[t])

    return np.asarray(X), np.asarray(y)


def _fit_model(model_key: str, random_state: int, **kwargs):
    """Construct candidate estimator for one model family."""
    if model_key == "ridge":
        alpha = kwargs.get("alpha", 1.0)
        base = Ridge(alpha=alpha)
        return MultiOutputRegressor(Pipeline([("scale", StandardScaler(with_mean=False)), ("model", base)]))
    if model_key == "lasso":
        alpha = float(kwargs.get("alpha", 0.001))
        max_iter = int(kwargs.get("max_iter", 2500))
        base = Lasso(alpha=alpha, max_iter=max_iter, random_state=random_state)
        return MultiOutputRegressor(
            Pipeline([("scale", StandardScaler(with_mean=False)), ("model", base)])
        )
    if model_key in {"elastic", "elasticnet"}:
        alpha = float(kwargs.get("alpha", 0.001))
        l1_ratio = float(kwargs.get("l1_ratio", 0.5))
        max_iter = int(kwargs.get("max_iter", 2500))
        base = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=random_state
        )
        return MultiOutputRegressor(
            Pipeline([("scale", StandardScaler(with_mean=False)), ("model", base)])
        )
    if model_key == "huber":
        epsilon = float(kwargs.get("epsilon", 1.35))
        max_iter = int(kwargs.get("max_iter", 2500))
        alpha = float(kwargs.get("alpha", 0.0001))
        base = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=max_iter)
        return MultiOutputRegressor(
            Pipeline([("scale", StandardScaler(with_mean=False)), ("model", base)])
        )
    if model_key == "knn":
        n_neighbors = int(kwargs.get("n_neighbors", 8))
        weights = str(kwargs.get("weights", "distance"))
        base = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        return MultiOutputRegressor(
            Pipeline([("scale", StandardScaler(with_mean=False)), ("model", base)])
        )
    if model_key == "svr":
        c = float(kwargs.get("C", 4.0))
        epsilon = float(kwargs.get("epsilon", 0.1))
        gamma = kwargs.get("gamma", "scale")
        kernel = str(kwargs.get("kernel", "rbf"))
        base = SVR(C=c, epsilon=epsilon, gamma=gamma, kernel=kernel)
        return MultiOutputRegressor(
            Pipeline([("scale", StandardScaler(with_mean=False)), ("model", base)])
        )
    if model_key == "hgb":
        max_iter = int(kwargs.get("n_estimators", 220))
        learning_rate = float(kwargs.get("learning_rate", 0.1))
        max_depth = kwargs.get("max_depth", None)
        if max_depth is not None:
            max_depth = int(max_depth)
        base = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
        return MultiOutputRegressor(base)
    if model_key == "arima":
        if not _HAS_TS_MODELS:
            raise ValueError("ARIMA candidate requires statsmodels.")
        return _ARIMAForecaster(
            lags=kwargs.get("lags", 24),
            target_dim=int(kwargs.get("target_dim", 2)),
            order=_coerce_arima_order(kwargs.get("order")),
            include_cyclic_features=bool(kwargs.get("include_cyclic_features", True)),
            random_state=random_state,
        )
    if model_key in {"exp_smooth", "ets", "holt_winters"}:
        if not _HAS_TS_MODELS:
            raise ValueError("Exponential-smoothing candidate requires statsmodels.")
        seasonal_periods = kwargs.get("seasonal_periods")
        try:
            seasonal_periods = int(seasonal_periods) if seasonal_periods is not None else None
        except Exception:
            seasonal_periods = None
        return _ExpSmoothingForecaster(
            lags=kwargs.get("lags", 24),
            target_dim=int(kwargs.get("target_dim", 2)),
            seasonal_periods=seasonal_periods,
            trend=kwargs.get("trend"),
            seasonal=kwargs.get("seasonal"),
            include_cyclic_features=bool(kwargs.get("include_cyclic_features", True)),
            random_state=random_state,
        )
    if model_key == "rf":
        n_estimators = kwargs.get("n_estimators", 250)
        max_depth = kwargs.get("max_depth", None)
        return MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=int(n_estimators),
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=1,
            )
        )
    if model_key == "gbr":
        learning_rate = float(kwargs.get("learning_rate", 0.05))
        n_estimators = int(kwargs.get("n_estimators", 300))
        max_depth = int(kwargs.get("max_depth", 3))
        return MultiOutputRegressor(
            GradientBoostingRegressor(
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
        )
    if model_key == "extra":
        n_estimators = int(kwargs.get("n_estimators", 220))
        max_depth = kwargs.get("max_depth", None)
        min_samples_leaf = int(kwargs.get("min_samples_leaf", 2))
        return MultiOutputRegressor(
            ExtraTreesRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=1,
            )
        )
    if model_key == "mlp":
        hidden_units = int(kwargs.get("hidden_units", 64))
        hidden_layers = max(1, int(kwargs.get("hidden_layers", 2)))
        hidden_layer_sizes = tuple([hidden_units] * hidden_layers)
        max_iter = int(kwargs.get("max_iter", 500))
        alpha = float(kwargs.get("alpha", 0.0001))
        learning_rate_init = float(kwargs.get("learning_rate_init", 1e-3))
        base = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=max_iter,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
        )
        return MultiOutputRegressor(
            Pipeline([("scale", StandardScaler(with_mean=False)), ("model", base)])
        )
    if model_key == "lstm":
        if not _HAS_TENSORFLOW:
            raise ValueError("LSTM candidate requires TensorFlow/Keras.")
        return _LSTMForecaster(
            lags=int(kwargs["lags"]),
            features=int(kwargs["feature_dim"]),
            target_dim=int(kwargs.get("target_dim", kwargs.get("n_series", 2))),
            include_cyclic_features=bool(kwargs.get("include_cyclic_features", False)),
            hidden_units=int(kwargs.get("hidden_units", 48)),
            epochs=int(kwargs.get("epochs", 8)),
            batch_size=int(kwargs.get("batch_size", 32)),
            random_state=random_state,
        )
    if model_key == "pinn":
        if not _HAS_TORCH:
            raise ValueError("PINN candidate requires PyTorch.")
        return _PinnForecaster(
            lags=int(kwargs["lags"]),
            target_idx=int(kwargs.get("target_idx", -1)),
            hidden_units=int(kwargs.get("hidden_units", 64)),
            hidden_layers=int(kwargs.get("hidden_layers", 2)),
            physics_weight=float(kwargs.get("physics_weight", 0.2)),
            epochs=int(kwargs.get("epochs", 80)),
            lr=float(kwargs.get("lr", 1e-3)),
            random_state=random_state,
        )
    raise ValueError(f"Unknown model family: {model_key}")


def _infer_step_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) >= 2:
        inferred = pd.infer_freq(index)
        if inferred is not None:
            return pd.tseries.frequencies.to_offset(inferred).delta

    # Default to 6-minute observations for the Hohonu/NOAA cadence.
    return pd.Timedelta(minutes=6)


def simulate_noaa_environment(
    noaa_history: np.ndarray,
    times: pd.DatetimeIndex,
    steps: int,
    random_state: int = 0,
):
    """Simple digital-tide simulator from recent tidal rhythm + AR residuals."""
    if noaa_history.size == 0 or steps <= 0:
        return np.array([])

    noaa_history = np.asarray(noaa_history, dtype=float)
    rng = np.random.default_rng(random_state)

    if times.dtype.kind == "O":
        times = pd.to_datetime(times)

    noaa = pd.Series(noaa_history)
    slot = times.hour * 60 + times.minute
    cyc_profile = noaa.groupby(slot).mean()
    cyc_profile = cyc_profile.reindex(range(24 * 60), method="ffill").fillna(method="bfill")

    residual = noaa - noaa.index.to_series().map(lambda i: cyc_profile.iloc[int(slot.iloc[i])]).to_numpy()
    residual = pd.Series(residual).replace([np.inf, -np.inf], np.nan).ffill().bfill().to_numpy()

    phi = 0.0
    if len(residual) > 2:
        x = residual[:-1]
        y = residual[1:]
        denom = np.dot(x, x)
        if denom > 1e-12:
            phi = float(np.clip(np.dot(x, y) / denom, 0.0, 0.95))

    # Mean residual scale controls synthetic variability.
    noise_scale = float(np.nanstd(residual)) if residual.size > 1 else 0.0
    if not np.isfinite(noise_scale) or noise_scale <= 0:
        noise_scale = 0.01 * (float(np.nanmax(np.abs(noaa)) + 1.0)
                             if np.isfinite(np.nanmax(np.abs(noaa))) else 1.0)

    base_start = slot.iloc[-1] if len(slot) else 0
    residual_state = residual[-1] if len(residual) else 0.0
    simulated = []
    for i in range(steps):
        cyc_slot = int((base_start + ((i + 1) * 6)) % (24 * 60))
        cyc_val = float(cyc_profile.iloc[cyc_slot])
        residual_state = phi * residual_state + rng.normal(scale=noise_scale * 0.3)
        simulated.append(cyc_val + residual_state)

    return np.asarray(simulated, dtype=float)


def _normalize_model_families(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []

    if isinstance(values, str):
        values = [values]

    aliases = {
        "randomforest": "rf",
        "random_forest": "rf",
        "rf": "rf",
        "arima": "arima",
        "auto_arima": "arima",
        "sarima": "arima",
        "ridge": "ridge",
        "lasso": "lasso",
        "elastic": "elastic",
        "elasticnet": "elastic",
        "en": "elastic",
        "huber": "huber",
        "knn": "knn",
        "knearest": "knn",
        "supportvector": "svr",
        "support_vector_regressor": "svr",
        "svr": "svr",
        "hgb": "hgb",
        "histgradientboosting": "hgb",
        "hist_gradient_boosting": "hgb",
        "mlp": "mlp",
        "mlpregressor": "mlp",
        "neural": "mlp",
        "gradientboosting": "gbr",
        "gradient_boosting": "gbr",
        "gbr": "gbr",
        "gbm": "gbr",
        "extratrees": "extra",
        "extra_trees": "extra",
        "extra": "extra",
        "ets": "exp_smooth",
        "exp_smooth": "exp_smooth",
        "exponential_smoothing": "exp_smooth",
        "holt_winters": "exp_smooth",
        "lstm": "lstm",
        "pinn": "pinn",
    }
    normalized: List[str] = []
    for raw in values:
        if raw is None:
            continue
        token = str(raw).strip().lower()
        if not token:
            continue
        parts = [p.strip() for p in token.split(",") if p.strip()]
        for part in parts:
            norm = aliases.get(part, part)
            normalized.append(norm)

    # Deduplicate while preserving order.
    return list(dict.fromkeys([x for x in normalized if x]))


def _score_candidate(
    series_df: pd.DataFrame,
    candidate: Dict,
    target_col: Optional[int] = None,
    random_state: int = 0,
) -> Optional[ModelCandidateResult]:
    model_key = candidate["model"]
    lags = int(candidate["lags"])
    data = _to_numeric_matrix(series_df)
    data = data.ffill().bfill()
    times = pd.to_datetime(series_df.index)
    include_cyclic = bool(candidate.get("include_cyclic_features", True))
    if model_key == "lstm":
        X, y = _build_sequence_matrix(
            data.to_numpy(),
            times,
            n_lags=lags,
            include_cyclic_features=include_cyclic,
        )
    else:
        X, y = _build_supervised_matrix(
            data.to_numpy(),
            times,
            n_lags=lags,
            include_cyclic_features=include_cyclic,
        )
    if X is None or y is None or len(X) == 0:
        return None
    if X.shape[0] < 40:
        return None

    # Keep last column as primary target (typically Hohonu cleaned series).
    if target_col is None:
        target_col = -1

    n_splits = min(4, max(2, X.shape[0] // 160))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    try:
        est = _fit_model(
            model_key,
            random_state=random_state,
            alpha=candidate.get("alpha", 1.0),
            n_estimators=candidate.get("n_estimators", 100),
            max_depth=candidate.get("max_depth", None),
            learning_rate=candidate.get("learning_rate", 0.05),
            min_samples_leaf=candidate.get("min_samples_leaf", 2),
            hidden_units=candidate.get("hidden_units", 64),
            batch_size=candidate.get("batch_size", 32),
            epochs=candidate.get("epochs", 8),
            hidden_layers=candidate.get("hidden_layers", 2),
            physics_weight=candidate.get("physics_weight", 0.2),
            lr=candidate.get("lr", 1e-3),
            target_idx=candidate.get("target_idx", -1),
            target_dim=int(data.shape[1]),
            feature_dim=int(X.shape[2]) if model_key == "lstm" else int(X.shape[1] / lags) if X.shape[1] % lags == 0 else int(data.shape[1]),
            n_series=int(data.shape[1]),
            include_cyclic_features=include_cyclic,
        )
    except Exception:
        return None

    rmse_scores = []
    mae_scores = []
    pred_blocks = []
    true_blocks = []
    for train_idx, test_idx in splitter.split(X):
        if len(test_idx) == 0:
            continue
        est.fit(X[train_idx], y[train_idx])
        pred = est.predict(X[test_idx])

        # Optional penalty for unstable candidates.
        if not np.isfinite(pred).all():
            return None

        pred_blocks.append(np.asarray(pred))
        true_blocks.append(np.asarray(y[test_idx]))
        fold_rmse = []
        fold_mae = []
        for c in range(y.shape[1]):
            fold_rmse.append(
                float(mean_squared_error(y[test_idx, c], pred[:, c], squared=False))
            )
            fold_mae.append(float(np.mean(np.abs(y[test_idx, c] - pred[:, c]))))
        rmse_scores.append(fold_rmse)
        mae_scores.append(fold_mae)

    if not rmse_scores:
        return None

    rmse_cols = np.mean(np.asarray(rmse_scores), axis=0)
    mae_cols = np.mean(np.asarray(mae_scores), axis=0)

    try:
        all_true = np.concatenate(true_blocks, axis=0)
        all_pred = np.concatenate(pred_blocks, axis=0)
        target_true = all_true[:, target_col]
        target_pred = all_pred[:, target_col]
        raw_metrics = _forecast_series_metrics(target_pred, target_true)
        r2_target = raw_metrics.get("r_square", float("nan"))
        corr_target = raw_metrics.get("corr", float("nan"))
        mape_target = raw_metrics.get("mape", float("nan"))
        mpe_target = raw_metrics.get("mpe", float("nan"))
        me_target = raw_metrics.get("me", float("nan"))
        minmax_target = raw_metrics.get("minmax", float("nan"))
        nse_target = raw_metrics.get("nse", float("nan"))
    except Exception:
        r2_target = float("nan")
        corr_target = float("nan")
        mape_target = float("nan")
        mpe_target = float("nan")
        me_target = float("nan")
        minmax_target = float("nan")
        nse_target = float("nan")

    return ModelCandidateResult(
        name=candidate.get("name", f"{model_key}_{lags}"),
        lags=lags,
        model=f"{model_key}",
        params=dict(candidate),
        rmse_target=float(rmse_cols[target_col]),
        rmse_all=float(np.mean(rmse_cols)),
        mae_target=float(mae_cols[target_col]),
        mae_all=float(np.mean(mae_cols)),
        rmse_colwise={
            series_df.columns[i]: float(rmse_cols[i]) for i in range(len(rmse_cols))
        },
        mae_colwise={series_df.columns[i]: float(mae_cols[i]) for i in range(len(mae_cols))},
        mape_target=_metric_or_nan(mape_target),
        r2_target=_metric_or_nan(r2_target),
        corr_target=_metric_or_nan(corr_target),
        mpe_target=_metric_or_nan(mpe_target),
        me_target=_metric_or_nan(me_target),
        minmax_target=_metric_or_nan(minmax_target),
        nse_target=_metric_or_nan(nse_target),
        estimator=est,
    )


def _forecast_with_estimator(
    estimator,
    series_df: pd.DataFrame,
    steps: int,
    lags: int,
    include_cyclic_features: bool = True,
    use_digital_twin: bool = False,
    random_state: int = 0,
) -> np.ndarray:
    if steps <= 0:
        return np.empty((0, series_df.shape[1]))

    hist = _to_numeric_matrix(series_df).ffill().bfill().to_numpy()
    hist_idx = pd.to_datetime(series_df.index)
    hist_times = pd.DatetimeIndex(hist_idx)

    if use_digital_twin and hist.shape[1] >= 2:
        sim_noaa = simulate_noaa_environment(hist[:, 0], hist_idx, steps, random_state)
    else:
        sim_noaa = None

    freq = _infer_step_delta(hist_idx)
    future_index = pd.date_range(
        start=hist_idx[-1] + freq, periods=steps, freq=freq
    )

    preds = []
    for i in range(steps):
        if hist.shape[0] < lags:
            raise ValueError("History shorter than required lag window.")
        if getattr(estimator, "_model_type", None) == "lstm":
            cyc_enabled = bool(
                getattr(estimator, "include_cyclic_features", include_cyclic_features)
            )
            if cyc_enabled:
                cyc = _build_time_features(hist_times)
                seq = np.concatenate(
                    [hist[-lags:], cyc[-lags:]], axis=1
                )
            else:
                seq = hist[-lags:]
            row = seq.reshape(1, lags, -1)
        else:
            parts = [hist[-lags:].reshape(-1)]
            if include_cyclic_features:
                parts.append(_build_time_features(future_index[: i + 1])[-1])
            row = np.concatenate(parts)
            row = row.reshape(1, -1)
        y_next = estimator.predict(row)[0]

        # Blend in digital twin for NOAA trajectory (column 0) if available.
        if sim_noaa is not None:
            y_next[0] = sim_noaa[i]

        preds.append(y_next)
        hist = np.vstack([hist, y_next])
        hist_times = hist_times.append(pd.DatetimeIndex([future_index[i]]))

    return np.asarray(preds)


def _fit_candidate_model(
    candidate: ModelCandidateResult,
    series_df: pd.DataFrame,
    random_state: int = 0,
) -> object:
    """Create and fit an estimator from a scored candidate."""
    model_key = candidate.model
    include_cyclic = bool(candidate.params.get("include_cyclic_features", True))
    history = _to_numeric_matrix(series_df).ffill().bfill()
    feature_dim = history.shape[1] + (8 if (model_key == "lstm" and include_cyclic) else 0)
    estimator = _fit_model(
        model_key,
        random_state=random_state,
        alpha=candidate.params.get("alpha", 1.0),
        n_estimators=candidate.params.get("n_estimators", 100),
        max_depth=candidate.params.get("max_depth", None),
        learning_rate=candidate.params.get("learning_rate", 0.05),
        min_samples_leaf=candidate.params.get("min_samples_leaf", 2),
        hidden_units=candidate.params.get("hidden_units", 64),
        batch_size=candidate.params.get("batch_size", 32),
        epochs=candidate.params.get("epochs", 8),
        hidden_layers=candidate.params.get("hidden_layers", 2),
        physics_weight=candidate.params.get("physics_weight", 0.2),
        lr=candidate.params.get("lr", 1e-3),
        target_idx=candidate.params.get("target_idx", -1),
        target_dim=int(series_df.shape[1]),
        feature_dim=feature_dim,
        n_series=int(series_df.shape[1]),
        include_cyclic_features=include_cyclic,
    )
    X, y = _build_supervised_matrix(
        _to_numeric_matrix(series_df).ffill().bfill().to_numpy(),
        pd.to_datetime(series_df.index),
        n_lags=candidate.lags,
        include_cyclic_features=include_cyclic,
    )
    if model_key == "lstm":
        X, y = _build_sequence_matrix(
            _to_numeric_matrix(series_df).ffill().bfill().to_numpy(),
            pd.to_datetime(series_df.index),
            n_lags=candidate.lags,
            include_cyclic_features=include_cyclic,
        )
    estimator.fit(X, y)
    return estimator


def _fit_meta_ensemble(
    combined_data: pd.DataFrame,
    scored: List[ModelCandidateResult],
    target_col: int = -1,
    meta_top_k: int = 4,
    random_state: int = 0,
    holdout_ratio: float = 0.2,
):
    """Fit a ridge-stacked meta-learner over top candidates."""
    data = _to_numeric_matrix(combined_data).ffill().bfill()
    if data.shape[0] < 120 or len(scored) < 2:
        return None

    try:
        holdout_ratio = float(holdout_ratio)
    except Exception:
        holdout_ratio = 0.2
    holdout_ratio = min(max(holdout_ratio, 0.1), 0.5)

    ranked = sorted(scored, key=lambda s: s.rmse_target)
    candidates = ranked[: max(2, min(int(meta_top_k), len(ranked)))]

    if target_col is None:
        target_col = -1
    target_col = int(target_col)
    if target_col < 0:
        target_col = data.shape[1] + target_col
    if target_col < 0 or target_col >= data.shape[1]:
        target_col = data.shape[1] - 1

    split_idx = int((1.0 - holdout_ratio) * len(data))
    split_idx = max(len(data) // 2, split_idx)
    if split_idx < 30 or split_idx >= len(data) - 1:
        return None

    train_df = data.iloc[:split_idx]
    holdout_df = data.iloc[split_idx:]
    hold_steps = len(holdout_df)
    if hold_steps <= 8:
        return None

    forecast_pool = []
    used_candidates = []
    for idx, candidate in enumerate(candidates):
        try:
            est = _fit_candidate_model(
                candidate,
                train_df,
                random_state=random_state + idx,
            )
            pred = _forecast_with_estimator(
                est,
                train_df,
                steps=hold_steps,
                lags=candidate.lags,
                include_cyclic_features=bool(
                    candidate.params.get("include_cyclic_features", True)
                ),
                use_digital_twin=False,
                random_state=random_state + idx,
            )
            if pred is None or len(pred) != hold_steps:
                continue
            if pred.shape[1] != data.shape[1]:
                continue
            forecast_pool.append(np.asarray(pred))
            used_candidates.append(candidate)
        except Exception:
            continue

    if len(forecast_pool) < 2:
        return None

    stack = np.stack(forecast_pool, axis=2)  # [steps, cols, models]
    hold_true = holdout_df.to_numpy()
    coeffs = []
    intercepts = []
    hold_pred = []

    for col in range(data.shape[1]):
        x = stack[:, col, :]
        x = np.nan_to_num(
            x,
            nan=np.nan,
            posinf=np.nan,
            neginf=np.nan,
        )
        fill_vals = np.zeros(x.shape[1], dtype=float)
        for m in range(x.shape[1]):
            col_vals = x[:, m]
            finite_vals = col_vals[np.isfinite(col_vals)]
            if finite_vals.size > 0:
                fill_vals[m] = float(np.mean(finite_vals))
        x = np.where(np.isfinite(x), x, fill_vals)

        y = hold_true[:hold_steps, col]
        finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if finite.sum() < max(8, x.shape[1] + 1):
            coef = np.ones((x.shape[1],), dtype=float)
            coef /= coef.size
            intercept = 0.0
            pred = x.dot(coef)
        else:
            x_clean = x[finite]
            y_clean = y[finite]
            if x_clean.shape[1] == 1:
                coef = np.array([1.0], dtype=float)
                intercept = 0.0
                pred = x_clean[:, 0]
            else:
                reg = Ridge(alpha=1e-3)
                reg.fit(x_clean, y_clean)
                coef = reg.coef_.astype(float)
                intercept = float(reg.intercept_)
                pred = reg.predict(x_clean)

            # Predict over full holdout window for reporting.
            pred = x.dot(coef) + intercept

        coeffs.append(coef)
        intercepts.append(float(intercept))
        hold_pred.append(pred)

    hold_pred = np.column_stack(hold_pred)
    target_true = hold_true[:hold_steps, target_col]
    target_pred = hold_pred[:, target_col]
    if target_true.size == 0:
        return None
    meta_stats = _forecast_series_metrics(target_pred, target_true)
    holdout_mae = float(np.mean(np.abs(target_true - target_pred)))
    holdout_rmse = float(mean_squared_error(target_true, target_pred, squared=False))

    return {
        "coeffs": coeffs,
        "intercepts": intercepts,
        "used": used_candidates,
        "selected_rmse": holdout_rmse,
        "holdout_mae": holdout_mae,
        "holdout_mape": _metric_or_nan(meta_stats.get("mape")),
        "holdout_r2": _metric_or_nan(meta_stats.get("r_square")),
        "holdout_corr": _metric_or_nan(meta_stats.get("corr")),
        "holdout_mpe": _metric_or_nan(meta_stats.get("mpe")),
        "holdout_me": _metric_or_nan(meta_stats.get("me")),
        "holdout_minmax": _metric_or_nan(meta_stats.get("minmax")),
        "holdout_nse": _metric_or_nan(meta_stats.get("nse")),
        "hold_pred": hold_pred,
    }


def get_default_candidate_grid(
    include_lstm: bool = False,
    include_pinn: bool = False,
    profile: str = "compact",
) -> List[Dict]:
    """Return candidate model/specification combinations for auto-search.

    profile:
      - compact: lightweight defaults for fast iterations
      - broad: expanded search space for stronger model-combination tests
      - auto: adaptive subset of broad candidates for balanced runtime/search tradeoff
    """
    profile = (profile or "compact").lower()
    if profile not in {"compact", "broad", "auto"}:
        profile = "compact"

    ridge_alphas = [1.0]
    lag_grid = [24]
    include_extremes = profile in {"broad", "auto"}

    if include_extremes:
        ridge_alphas = [0.5, 1.0, 2.0]
        lag_grid = [12, 24, 30, 36, 48]

    candidates = [
        {"name": f"ridge_l{lag}_a{alpha}", "model": "ridge", "lags": lag, "alpha": alpha}
        for lag in lag_grid
        for alpha in ridge_alphas
    ]
    candidates.extend(
        [
            {
                "name": "lasso_l24",
                "model": "lasso",
                "lags": 24,
                "alpha": 0.001,
                "max_iter": 2500,
            },
            {
                "name": "elastic_l24",
                "model": "elastic",
                "lags": 24,
                "alpha": 0.002,
                "l1_ratio": 0.6,
                "max_iter": 2500,
            },
            {
                "name": "huber_l24",
                "model": "huber",
                "lags": 24,
                "epsilon": 1.5,
                "max_iter": 2500,
            },
            {
                "name": "knn_l24",
                "model": "knn",
                "lags": 24,
                "n_neighbors": 8,
                "weights": "distance",
            },
            {
                "name": "svr_l24",
                "model": "svr",
                "lags": 24,
                "C": 4.0,
                "gamma": "scale",
                "epsilon": 0.12,
            },
            {
                "name": "hgb_l24",
                "model": "hgb",
                "lags": 24,
                "n_estimators": 260,
                "learning_rate": 0.08,
                "max_depth": 8,
            },
            {
                "name": "mlp_l24",
                "model": "mlp",
                "lags": 24,
                "hidden_units": 64,
                "hidden_layers": 2,
                "max_iter": 350,
                "alpha": 0.0008,
            },
        ]
    )

    candidates.extend(
        [
            {"name": "rf_l24", "model": "rf", "lags": 24, "n_estimators": 220},
            {"name": "rf_l36", "model": "rf", "lags": 36, "n_estimators": 240},
            {"name": "rf_l48", "model": "rf", "lags": 48, "n_estimators": 280},
            {"name": "gbr_l24", "model": "gbr", "lags": 24, "n_estimators": 260},
            {"name": "extra_l24", "model": "extra", "lags": 24, "n_estimators": 220},
            {"name": "rf_cyc_l24", "model": "rf", "lags": 24, "n_estimators": 180, "include_cyclic_features": True},
            {"name": "gbr_cyc_l36", "model": "gbr", "lags": 36, "n_estimators": 220, "include_cyclic_features": True},
        ]
    )

    if _HAS_TS_MODELS:
        candidates.extend(
            [
                {
                    "name": "arima_l24",
                    "model": "arima",
                    "lags": 24,
                    "order": "1,0,1",
                    "include_cyclic_features": False,
                },
                {
                    "name": "ets_l24",
                    "model": "exp_smooth",
                    "lags": 24,
                    "trend": "add",
                    "seasonal": None,
                    "seasonal_periods": None,
                    "include_cyclic_features": False,
                },
            ]
        )

    if include_extremes:
        candidates.extend(
            [
                {
                    "name": f"lasso_l{lag}_a{alpha}",
                    "model": "lasso",
                    "lags": lag,
                    "alpha": alpha,
                    "max_iter": 2500,
                }
                for lag in (12, 24, 30)
                for alpha in (0.0005, 0.001, 0.0025)
            ]
        )
        candidates.extend(
            [
                {
                    "name": f"elastic_l{lag}_a{alpha}_r{l1_ratio}",
                    "model": "elastic",
                    "lags": lag,
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "max_iter": 3000,
                }
                for lag in (12, 24, 30)
                for alpha in (0.0005, 0.0015)
                for l1_ratio in (0.3, 0.7)
            ]
        )
        candidates.extend(
            [
                {
                    "name": f"huber_l{lag}_e{epsilon}",
                    "model": "huber",
                    "lags": lag,
                    "epsilon": epsilon,
                    "max_iter": 2500,
                }
                for lag in (18, 24, 30)
                for epsilon in (1.1, 1.8)
            ]
        )
        candidates.extend(
            [
                {
                    "name": f"knn_l{lag}_k{k}_w{weights}",
                    "model": "knn",
                    "lags": lag,
                    "n_neighbors": k,
                    "weights": weights,
                }
                for lag in (12, 24, 30)
                for k in (5, 9, 13)
                for weights in ("distance",)
            ]
        )
        candidates.extend(
            [
                {
                    "name": f"svr_l{lag}_c{C}_g{gamma}_e{eps}",
                    "model": "svr",
                    "lags": lag,
                    "C": C,
                    "gamma": gamma,
                    "epsilon": eps,
                    "kernel": "rbf",
                }
                for lag in (12, 24, 30)
                for C in (2.0, 6.0)
                for gamma in ("scale",)
                for eps in (0.05, 0.15)
            ]
        )
        candidates.extend(
            [
                {
                    "name": f"hgb_l{lag}_n{n}_lr{lr}",
                    "model": "hgb",
                    "lags": lag,
                    "n_estimators": n,
                    "learning_rate": lr,
                    "max_depth": 8,
                }
                for lag in (12, 24)
                for n in (180, 260)
                for lr in (0.07, 0.1)
            ]
        )
        candidates.extend(
            [
                {
                    "name": f"mlp_l{lag}_u{units}_d{depth}_i{iters}",
                    "model": "mlp",
                    "lags": lag,
                    "hidden_units": units,
                    "hidden_layers": depth,
                    "max_iter": iters,
                    "alpha": 0.0008,
                }
                for lag in (12, 24)
                for units in (48, 64)
                for depth in (1, 2)
                for iters in (250, 450)
            ]
        )
        candidates.extend(
            [
                {
                    "name": f"rf_l{lag}_n{n}",
                    "model": "rf",
                    "lags": lag,
                    "n_estimators": n,
                }
                for lag in (12, 30, 48)
                for n in (180, 260)
            ]
        )
        candidates.extend(
            [
                {
                    "name": f"extra_l{lag}_n{n}",
                    "model": "extra",
                    "lags": lag,
                    "n_estimators": n,
                }
                for lag in (12, 30)
                for n in (180, 260)
            ]
        )
        candidates.extend(
            [
                {
                    "name": f"gbr_l{lag}_n{n}_lr{lr}",
                    "model": "gbr",
                    "lags": lag,
                    "n_estimators": n,
                    "learning_rate": lr,
                    "max_depth": 2,
                }
                for lag in (18, 24, 30, 36)
                for n in (200, 260)
                for lr in (0.03, 0.06)
            ]
        )
        candidates.extend(
            [
                {
                    "name": f"gbr_cyc_l{lag}_n{n}_lr{lr}",
                    "model": "gbr",
                    "lags": lag,
                    "n_estimators": n,
                    "learning_rate": lr,
                    "include_cyclic_features": True,
                    "max_depth": 3,
                }
                for lag in (18, 30)
                for n in (180, 240)
                for lr in (0.04, 0.08)
            ]
        )
        if _HAS_TS_MODELS:
            candidates.extend(
                [
                    {
                        "name": f"arima_l{lag}_o{order}",
                        "model": "arima",
                        "lags": lag,
                        "order": order,
                        "include_cyclic_features": False,
                    }
                    for lag in (12, 24, 30)
                    for order in ("1,0,0", "1,1,0", "2,0,1", "2,1,2")
                ]
            )
            candidates.extend(
                [
                    {
                        "name": f"ets_l{lag}_t{trend}_s{seasonal or 'none'}_p{seasonal_periods}_inc",
                        "model": "exp_smooth",
                        "lags": lag,
                        "trend": trend,
                        "seasonal": seasonal,
                        "seasonal_periods": seasonal_periods,
                        "include_cyclic_features": True,
                    }
                    for lag in (12, 24, 30)
                    for trend, seasonal, seasonal_periods in (
                        ("add", None, None),
                        ("add", "add", 240),
                        ("add", "add", 480),
                    )
                ]
            )

    if include_lstm and _HAS_TENSORFLOW:
        candidates.extend(
            [
                {
                    "name": "lstm_l24",
                    "model": "lstm",
                    "lags": 24,
                    "hidden_units": 24,
                    "epochs": 6,
                    "batch_size": 32,
                    "include_cyclic_features": False,
                },
                {
                    "name": "lstm_cyc_l24",
                    "model": "lstm",
                    "lags": 24,
                    "hidden_units": 32,
                    "epochs": 8,
                    "batch_size": 32,
                    "include_cyclic_features": True,
                },
            ]
        )
        if include_extremes:
            candidates.extend(
                [
                    {
                        "name": f"lstm_l{lag}_u{units}_e{epochs}_c{int(bool(cyc))}",
                        "model": "lstm",
                        "lags": lag,
                        "hidden_units": units,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "include_cyclic_features": cyc,
                    }
                    for lag in (12, 18, 24, 36)
                    for units in (16, 24, 32)
                    for epochs in (4, 8)
                    for batch_size in (16, 32)
                    for cyc in (False, True)
                ]
            )
    if include_pinn and _HAS_TORCH:
        candidates.extend(
            [
                {
                    "name": "pinn_l24_1",
                    "model": "pinn",
                    "lags": 24,
                    "hidden_units": 64,
                    "hidden_layers": 2,
                    "epochs": 80,
                    "physics_weight": 0.20,
                    "include_cyclic_features": False,
                },
                {
                    "name": "pinn_l36_smooth",
                    "model": "pinn",
                    "lags": 36,
                    "hidden_units": 72,
                    "hidden_layers": 3,
                    "epochs": 120,
                    "physics_weight": 0.30,
                    "include_cyclic_features": False,
                },
            ]
        )
        if include_extremes:
            candidates.extend(
                [
                    {
                        "name": f"pinn_l{lag}_u{hidden}_d{layers}_e{epochs}_w{physics_weight}",
                        "model": "pinn",
                        "lags": lag,
                        "hidden_units": hidden,
                        "hidden_layers": layers,
                        "epochs": epochs,
                        "physics_weight": physics_weight,
                        "include_cyclic_features": False,
                    }
                    for lag in (12, 24, 36)
                    for hidden in (48, 64)
                    for layers in (2, 3)
                for epochs in (60, 80)
                    for physics_weight in (0.10, 0.25)
                ]
            )
    if profile == "auto":
        max_auto_candidates = 24
        if len(candidates) > max_auto_candidates:
            keep_idx = np.sort(
                np.random.RandomState(0).choice(
                    np.arange(len(candidates)),
                    size=max_auto_candidates,
                    replace=False,
                )
            )
            candidates = [candidates[i] for i in keep_idx]

    return candidates


def run_auto_ml_search(
    combined_data: pd.DataFrame,
    steps: int = 960,
    candidate_grid: Optional[Sequence[Dict]] = None,
    random_state: int = 0,
    use_digital_twin: bool = True,
    target_col: Optional[int] = None,
    strategy: str = "best",
    meta_top_k: int = 4,
    meta_holdout_ratio: float = 0.2,
    ensemble_size: int = 2,
    candidate_model_families: Optional[Sequence[str]] = None,
    candidate_profile: str = "compact",
    candidate_mix_max_size: int = 4,
    qa_metric_weights: Optional[Dict[str, float]] = None,
):
    """Select the best model/lag combination and return full forecast bundle."""
    data = _to_numeric_matrix(combined_data)
    data = data.ffill().bfill()

    if target_col is None:
        target_col = -1

    strategy = (strategy or "best").lower()
    if strategy not in {"best", "ensemble", "meta", "mix"}:
        raise ValueError("strategy must be one of {'best', 'ensemble', 'meta', 'mix'}")

    if candidate_grid is None:
        candidate_grid = get_default_candidate_grid(
            include_lstm=_HAS_TENSORFLOW,
            include_pinn=_HAS_TORCH,
            profile=str(candidate_profile),
        )

    model_families = _normalize_model_families(candidate_model_families)
    if model_families:
        candidate_grid = [
            candidate
            for candidate in candidate_grid
            if str(candidate.get("model", "")).lower() in model_families
        ]
        if len(candidate_grid) == 0:
            raise ValueError(
                "No candidates remain after filtering by candidate_model_families."
            )

    scored: List[ModelCandidateResult] = []
    for candidate in candidate_grid:
        result = _score_candidate(
            data,
            candidate,
            target_col=target_col,
            random_state=random_state,
        )
        if result is not None:
            scored.append(result)

    if not scored:
        # Fallback: naive persistence baseline.
        last = np.asarray(data.iloc[-1].values)
        forecast = np.tile(last, (steps, 1))
        return {
            "forecast": forecast,
            "model_name": "naive_persistence",
            "selected_rmse": np.nan,
            "selected_mae": float("nan"),
            "selected_mape": float("nan"),
            "selected_r2": float("nan"),
            "selected_corr": float("nan"),
            "selected_mpe": float("nan"),
            "selected_me": float("nan"),
            "selected_minmax": float("nan"),
            "selected_nse": float("nan"),
            "selected_qa_score": float("nan"),
            "selected_metrics": {
                "selected_rmse": float("nan"),
                "selected_mae": float("nan"),
                "selected_mape": float("nan"),
                "selected_r2": float("nan"),
                "selected_corr": float("nan"),
                "selected_mpe": float("nan"),
                "selected_me": float("nan"),
                "selected_minmax": float("nan"),
                "selected_nse": float("nan"),
                "selected_qa_score": float("nan"),
            },
            "scores": [],
            "columns": list(combined_data.columns),
            "digital_twin_used": bool(use_digital_twin),
            "strategy": "single_best",
            "ensemble_size": 1,
            "meta_payload": None,
        }

    ordered = sorted(scored, key=lambda s: s.rmse_target)
    meta_payload = None
    selected_metrics = None

    if strategy == "ensemble":
        k = min(max(2, ensemble_size), len(ordered))
        top = ordered[:k]
        forecasts = []
        weights = []
        for idx, item in enumerate(top):
            est = _fit_candidate_model(item, data, random_state=random_state)
            pred = _forecast_with_estimator(
                est,
                data,
                steps=steps,
                lags=item.lags,
                include_cyclic_features=bool(item.params.get("include_cyclic_features", True)),
                use_digital_twin=use_digital_twin,
                random_state=random_state,
            )
            w = 1.0 / (item.rmse_target + 1e-6)
            forecasts.append(pred)
            weights.append(w)
        weights = np.asarray(weights, dtype=float)
        forecast = np.average(np.asarray(forecasts), axis=0, weights=weights)
        selected_name = "+".join([item.name for item in top])
        selected_rmse = float(np.average([item.rmse_target for item in top], weights=weights))
        selected_metrics = _weighted_candidate_metrics(
            top,
            weights=weights,
            qa_metric_weights=qa_metric_weights,
        )
    elif strategy == "mix":
        max_mix = max(2, int(candidate_mix_max_size))
        base = ordered[:]

        if model_families:
            family_best = {}
            for item in ordered:
                if item.model not in family_best:
                    family_best[item.model] = item
            filtered = [family_best[m] for m in model_families if m in family_best]
            if filtered:
                base = filtered

        max_candidates = max(2, min(len(base), max_mix))
        if len(base) < 2:
            if len(ordered) >= 2:
                base = ordered[:max_candidates]
            else:
                best = ordered[0]
                forecast = _forecast_with_estimator(
                    _fit_candidate_model(best, data, random_state=random_state),
                    data,
                    steps=steps,
                    lags=best.lags,
                    include_cyclic_features=bool(best.params.get("include_cyclic_features", True)),
                    use_digital_twin=use_digital_twin,
                    random_state=random_state,
                )
                selected_name = best.name
                selected_rmse = float(best.rmse_target)
                selected_metrics = _weighted_candidate_metrics(
                    [best], weights=None, qa_metric_weights=qa_metric_weights
                )
                selected_metrics["selected_rmse"] = float(selected_rmse)
                selected_metrics["selected_qa_score"] = _qa_score_from_metrics(
                    selected_metrics, qa_metric_weights
                )
        if len(base) >= 2:
            best_combo = base[:2]
            best_combo_score = float("inf")
            best_combo_weights = np.array([1.0, 1.0], dtype=float)
            best_combo_size = 2

            # Evaluate mixed combinations of top models by CV score proxy.
            for size in range(2, max_candidates + 1):
                for idx_combo in combinations(range(len(base)), size):
                    combo = [base[idx] for idx in idx_combo]
                    combo_rmses = np.array([item.rmse_target for item in combo], dtype=float)
                    weights = 1.0 / (combo_rmses + 1e-6)
                    weights_sum = float(weights.sum())
                    if weights_sum <= 0:
                        continue
                    weights = weights / weights_sum
                    score = float(np.average(combo_rmses, weights=weights))
                    if score < best_combo_score:
                        best_combo_score = score
                        best_combo = combo
                        best_combo_weights = weights
                        best_combo_size = len(combo)

            forecasts = []
            valid_weights = []
            used_combo = []
            for idx, item in enumerate(best_combo):
                estimator = _fit_candidate_model(
                    item,
                    data,
                    random_state=random_state + idx,
                )
                pred = _forecast_with_estimator(
                    estimator,
                    data,
                    steps=steps,
                    lags=item.lags,
                    include_cyclic_features=bool(item.params.get("include_cyclic_features", True)),
                    use_digital_twin=use_digital_twin,
                    random_state=random_state,
                )
                if pred is None or len(pred) == 0:
                    continue
                forecasts.append(np.asarray(pred))
                valid_weights.append(best_combo_weights[idx])
                used_combo.append(item)

            if len(forecasts) < 2:
                best = ordered[0]
                forecast = _forecast_with_estimator(
                    _fit_candidate_model(best, data, random_state=random_state),
                    data,
                    steps=steps,
                    lags=best.lags,
                    include_cyclic_features=bool(
                        best.params.get("include_cyclic_features", True)
                    ),
                    use_digital_twin=use_digital_twin,
                    random_state=random_state,
                )
                selected_name = best.name
                selected_rmse = float(best.rmse_target)
                selected_metrics = _weighted_candidate_metrics(
                    [best], weights=None, qa_metric_weights=qa_metric_weights
                )
                selected_metrics["selected_rmse"] = float(selected_rmse)
                selected_metrics["selected_qa_score"] = _qa_score_from_metrics(
                    selected_metrics, qa_metric_weights
                )
            else:
                valid_weights = np.asarray(valid_weights, dtype=float)
                weight_sum = float(valid_weights.sum())
                if weight_sum <= 0:
                    valid_weights = np.ones(len(forecasts), dtype=float)
                    weight_sum = float(valid_weights.sum())
                valid_weights = valid_weights / weight_sum
                forecast = np.average(np.asarray(forecasts), axis=0, weights=valid_weights)
                selected_name = (
                    f"mix[{'+'.join([item.name for item in best_combo])}]"
                )
                selected_rmse = float(best_combo_score / max(1, best_combo_size))
                selected_metrics = _weighted_candidate_metrics(
                    used_combo,
                    weights=valid_weights,
                    qa_metric_weights=qa_metric_weights,
                )
                selected_metrics["selected_rmse"] = float(selected_rmse)
    elif strategy == "meta":
        meta = _fit_meta_ensemble(
            data,
            ordered,
            target_col=target_col,
            meta_top_k=meta_top_k,
            random_state=random_state,
            holdout_ratio=meta_holdout_ratio,
        )
        if meta is None:
            best = ordered[0]
            forecast = _forecast_with_estimator(
                _fit_candidate_model(best, data, random_state=random_state),
                data,
                steps=steps,
                lags=best.lags,
                include_cyclic_features=bool(
                    best.params.get("include_cyclic_features", True)
                ),
                use_digital_twin=use_digital_twin,
                random_state=random_state,
            )
            selected_name = best.name
            selected_rmse = float(best.rmse_target)
            selected_metrics = _weighted_candidate_metrics(
                [best], weights=None, qa_metric_weights=qa_metric_weights
            )
            selected_metrics["selected_rmse"] = float(selected_rmse)
            selected_metrics["selected_qa_score"] = _qa_score_from_metrics(
                selected_metrics, qa_metric_weights
            )
        else:
            top = meta["used"]
            coeffs = meta["coeffs"]
            intercepts = meta["intercepts"]
            forecasts = []
            for idx, item in enumerate(top):
                est = _fit_candidate_model(item, data, random_state=random_state + idx)
                pred = _forecast_with_estimator(
                    est,
                    data,
                    steps=steps,
                    lags=item.lags,
                    include_cyclic_features=bool(
                        item.params.get("include_cyclic_features", True)
                    ),
                    use_digital_twin=use_digital_twin,
                    random_state=random_state,
                )
                pred = np.asarray(pred) if pred is not None else None
                if pred is None or len(pred) == 0:
                    continue
                if pred.shape[0] != steps or pred.shape[1] != data.shape[1]:
                    continue
                forecasts.append(np.asarray(pred))
            if len(forecasts) < 2:
                best = ordered[0]
                forecast = _forecast_with_estimator(
                    _fit_candidate_model(best, data, random_state=random_state),
                    data,
                    steps=steps,
                    lags=best.lags,
                    include_cyclic_features=bool(
                        best.params.get("include_cyclic_features", True)
                    ),
                    use_digital_twin=use_digital_twin,
                    random_state=random_state,
                )
                selected_name = best.name
                selected_rmse = float(best.rmse_target)
                selected_metrics = _weighted_candidate_metrics(
                    [best], weights=None, qa_metric_weights=qa_metric_weights
                )
                selected_metrics["selected_rmse"] = float(selected_rmse)
                selected_metrics["selected_qa_score"] = _qa_score_from_metrics(
                    selected_metrics, qa_metric_weights
                )
            else:
                n_cols = data.shape[1]
                forecast = np.zeros((steps, n_cols), dtype=float)
                for c in range(n_cols):
                    x = np.column_stack([p[:, c] for p in forecasts])
                    coef = np.asarray(coeffs[c], dtype=float)
                    if coef.size == 0:
                        coef = np.ones((x.shape[1],), dtype=float)
                        coef /= coef.size
                        intercept = 0.0
                    else:
                        intercept = float(intercepts[c])
                    if coef.shape[0] != x.shape[1]:
                        coef = coef[: x.shape[1]]
                        if coef.size < x.shape[1]:
                            coef = np.concatenate(
                                [coef, np.zeros((x.shape[1] - coef.size,), dtype=float)]
                            )
                    forecast[:, c] = x.dot(coef) + intercept
                selected_name = "meta_stack[" + "+".join([item.name for item in top]) + "]"
                selected_rmse = float(meta["selected_rmse"])
                selected_metrics = {
                    "selected_rmse": float(meta["selected_rmse"]),
                    "selected_mae": _metric_or_nan(meta.get("holdout_mae")),
                    "selected_mape": _metric_or_nan(meta.get("holdout_mape")),
                    "selected_r2": _metric_or_nan(meta.get("holdout_r2")),
                    "selected_corr": _metric_or_nan(meta.get("holdout_corr")),
                    "selected_mpe": _metric_or_nan(meta.get("holdout_mpe")),
                    "selected_me": _metric_or_nan(meta.get("holdout_me")),
                    "selected_minmax": _metric_or_nan(meta.get("holdout_minmax")),
                    "selected_nse": _metric_or_nan(meta.get("holdout_nse")),
                }
                selected_metrics["selected_qa_score"] = _qa_score_from_metrics(
                    selected_metrics, qa_metric_weights
                )
                meta_payload = {
                    "top_models": [item.name for item in top],
                    "coeffs": [np.asarray(coef).tolist() for coef in coeffs],
                    "intercepts": [float(v) for v in intercepts],
                    "top_k": len(top),
                    "holdout_ratio": float(meta_holdout_ratio),
                    "holdout_rmse": float(meta["selected_rmse"]),
                    "holdout_mae": selected_metrics.get("selected_mae"),
                    "holdout_mape": selected_metrics.get("selected_mape"),
                    "holdout_r2": selected_metrics.get("selected_r2"),
                    "holdout_corr": selected_metrics.get("selected_corr"),
                    "holdout_nse": selected_metrics.get("selected_nse"),
                }
    else:
        best = ordered[0]
        forecast = _forecast_with_estimator(
            _fit_candidate_model(best, data, random_state=random_state),
            data,
            steps=steps,
            lags=best.lags,
            include_cyclic_features=bool(best.params.get("include_cyclic_features", True)),
            use_digital_twin=use_digital_twin,
            random_state=random_state,
        )
        selected_name = best.name
        selected_rmse = best.rmse_target
        selected_metrics = _weighted_candidate_metrics(
            [best], weights=None, qa_metric_weights=qa_metric_weights
        )
        selected_metrics["selected_rmse"] = float(selected_rmse)
        selected_metrics["selected_qa_score"] = _qa_score_from_metrics(
            selected_metrics, qa_metric_weights
        )
        meta_payload = None

    if selected_metrics is None:
        selected_metrics = {
            "selected_rmse": float("nan"),
            "selected_mae": float("nan"),
            "selected_mape": float("nan"),
            "selected_r2": float("nan"),
            "selected_corr": float("nan"),
            "selected_mpe": float("nan"),
            "selected_me": float("nan"),
            "selected_minmax": float("nan"),
            "selected_nse": float("nan"),
        }
        selected_metrics["selected_qa_score"] = _qa_score_from_metrics(
            selected_metrics, qa_metric_weights
        )

    if isinstance(selected_metrics.get("selected_rmse"), float) and np.isnan(
        selected_metrics["selected_rmse"]
    ):
        selected_metrics["selected_rmse"] = _metric_or_nan(selected_rmse)
    selected_metrics["selected_qa_score"] = _qa_score_from_metrics(
        selected_metrics, qa_metric_weights
    )

    return {
        "forecast": forecast,
        "model_name": selected_name,
        "selected_rmse": selected_rmse,
        "selected_mae": selected_metrics.get("selected_mae"),
        "selected_mape": selected_metrics.get("selected_mape"),
        "selected_r2": selected_metrics.get("selected_r2"),
        "selected_corr": selected_metrics.get("selected_corr"),
        "selected_mpe": selected_metrics.get("selected_mpe"),
        "selected_me": selected_metrics.get("selected_me"),
        "selected_minmax": selected_metrics.get("selected_minmax"),
        "selected_nse": selected_metrics.get("selected_nse"),
        "selected_qa_score": selected_metrics.get("selected_qa_score"),
        "selected_metrics": selected_metrics,
        "scores": [
            {
                "name": item.name,
                "model": item.model,
                "lags": item.lags,
                "rmse_target": item.rmse_target,
                "rmse_all": item.rmse_all,
                "mae_target": item.mae_target,
                "mae_all": item.mae_all,
                "mape_target": item.mape_target,
                "r2_target": item.r2_target,
                "corr_target": item.corr_target,
                "mpe_target": item.mpe_target,
                "me_target": item.me_target,
                "minmax_target": item.minmax_target,
                "nse_target": item.nse_target,
                "rmse_colwise": item.rmse_colwise,
                "mae_colwise": item.mae_colwise,
            }
            for item in scored
        ],
        "columns": list(combined_data.columns),
        "digital_twin_used": bool(use_digital_twin),
        "strategy": "ensemble"
        if strategy == "ensemble"
        else "meta_stacking"
        if strategy == "meta"
        else "model_mix"
        if strategy == "mix"
        else "single_best",
        "ensemble_size": int(ensemble_size) if strategy == "ensemble" else 1,
        "meta_payload": meta_payload,
    }


def benchmark_best_model(
    combined_data: pd.DataFrame,
    candidate_grid: Optional[Sequence[Dict]] = None,
    holdout_ratio: float = 0.1,
    random_state: int = 0,
    candidate_model_families: Optional[Sequence[str]] = None,
    candidate_profile: str = "compact",
    qa_metric_weights: Optional[Dict[str, float]] = None,
):
    """Evaluate candidate models on an explicit holdout window.

    Returns a compact metrics table for quick inspection.
    """
    data = _to_numeric_matrix(combined_data).ffill().bfill()
    if data.shape[0] < 200:
        raise ValueError("Not enough history for benchmarking; need at least 200 observations.")

    split_idx = int((1 - holdout_ratio) * len(data))
    if split_idx < 20:
        split_idx = int(0.8 * len(data))
    train_df = data.iloc[:split_idx]
    holdout_df = data.iloc[split_idx:]

    if candidate_grid is None:
        candidate_grid = get_default_candidate_grid(
            include_lstm=_HAS_TENSORFLOW,
            include_pinn=_HAS_TORCH,
            profile=str(candidate_profile),
        )

    model_families = _normalize_model_families(candidate_model_families)
    if model_families:
        candidate_grid = [
            candidate
            for candidate in candidate_grid
            if str(candidate.get("model", "")).lower() in model_families
        ]
        if len(candidate_grid) == 0:
            raise ValueError(
                "No candidates remain after filtering by candidate_model_families."
            )

    metrics = []
    for candidate in candidate_grid:
        result = _score_candidate(
            train_df,
            candidate,
            target_col=-1,
            random_state=random_state,
        )
        if result is None:
            continue

        # Train on train data and forecast holdout horizon.
        include_cyclic = bool(candidate.get("include_cyclic_features", True))
        history = train_df.to_numpy()
        if result.model == "lstm":
            X, y = _build_sequence_matrix(
                history,
                pd.to_datetime(train_df.index),
                n_lags=result.lags,
                include_cyclic_features=include_cyclic,
            )
        else:
            X, y = _build_supervised_matrix(
                history,
                pd.to_datetime(train_df.index),
                n_lags=result.lags,
                include_cyclic_features=include_cyclic,
            )
        if X is None or len(X) == 0:
            continue

        feature_dim = train_df.shape[1] + (8 if (result.model == "lstm" and include_cyclic) else 0)
        try:
            model = _fit_model(
                result.model,
                random_state=random_state,
                alpha=result.params.get("alpha", 1.0),
                n_estimators=result.params.get("n_estimators", 100),
                max_depth=result.params.get("max_depth", None),
                learning_rate=result.params.get("learning_rate", 0.05),
                min_samples_leaf=result.params.get("min_samples_leaf", 2),
                hidden_units=result.params.get("hidden_units", 64),
                batch_size=result.params.get("batch_size", 32),
                epochs=result.params.get("epochs", 8),
                hidden_layers=result.params.get("hidden_layers", 2),
                physics_weight=result.params.get("physics_weight", 0.2),
                lr=result.params.get("lr", 1e-3),
                target_idx=result.params.get("target_idx", -1),
                target_dim=int(train_df.shape[1]),
                feature_dim=int(feature_dim),
                n_series=int(train_df.shape[1]),
                include_cyclic_features=include_cyclic,
            )
            model.fit(X, y)
        except Exception:
            continue
        hold_pred = _forecast_with_estimator(
            model,
            train_df,
            steps=len(holdout_df),
            lags=result.lags,
            include_cyclic_features=include_cyclic,
            use_digital_twin=False,
            random_state=random_state,
        )
        # Align target: last column
        target_idx = -1
        hold_true = holdout_df.to_numpy()
        target_true = hold_true[: hold_pred.shape[0], target_idx]
        target_pred = hold_pred[: len(target_true), target_idx]
        raw_metrics = _forecast_series_metrics(target_pred, target_true)

        metrics.append(
            {
                "name": result.name,
                "rmse_target": float(mean_squared_error(target_true, target_pred, squared=False)),
                "mae_target": float(np.mean(np.abs(target_true - target_pred)),),
                "mape_target": _metric_or_nan(raw_metrics.get("mape")),
                "r2_target": _metric_or_nan(raw_metrics.get("r_square")),
                "corr_target": _metric_or_nan(raw_metrics.get("corr")),
                "mpe_target": _metric_or_nan(raw_metrics.get("mpe")),
                "me_target": _metric_or_nan(raw_metrics.get("me")),
                "minmax_target": _metric_or_nan(raw_metrics.get("minmax")),
                "nse_target": _metric_or_nan(raw_metrics.get("nse")),
            }
        )
        metrics[-1]["selected_qa_score"] = _qa_score_from_metrics(
            metrics[-1], qa_metric_weights
        )

    return metrics


def benchmark_model_health_from_rolling(
    candidate_rows: List[Dict], qa_metric_weights: Optional[Dict[str, float]] = None
) -> List[Dict]:
    """Summarize model-family health from rolling backtest rows."""
    if not candidate_rows:
        return []

    by_model = {}
    by_family = {}
    for row in candidate_rows:
        family = str(row.get("model", "")).strip()
        name = str(row.get("name", "unknown"))
        if not family:
            family = "unknown"
        by_family.setdefault(family, {"rows": [], "rank_series": {}})
        by_model.setdefault((family, name), []).append(row)
        by_family[family]["rows"].append(row)

    # Build family-level rank summary from per-fold candidate ranks.
    for row in candidate_rows:
        family = str(row.get("model", "unknown")).strip() or "unknown"
        name = str(row.get("name", "unknown"))
        fold = row.get("fold")
        if fold is None:
            continue
        family_bucket = by_family.get(family)
        if family_bucket is None:
            continue
        family_bucket["rank_series"].setdefault(name, []).append(row.get("rank"))

    health = []
    for family, info in sorted(by_family.items(), key=lambda item: item[0]):
        rows = info["rows"]
        if not rows:
            continue

        # Collect family-level mean/std metrics using only numeric rows.
        def _collect(metric_key):
            values = [
                float(r.get(metric_key))
                for r in rows
                if isinstance(r.get(metric_key), (int, float, np.number))
                and np.isfinite(float(r.get(metric_key)))
            ]
            return values

        rmse_vals = _collect("rmse_target")
        mae_vals = _collect("mae_target")
        mape_vals = _collect("mape_target")
        r2_vals = _collect("r2_target")
        corr_vals = _collect("corr_target")
        me_vals = _collect("me_target")
        mpe_vals = _collect("mpe_target")
        nse_vals = _collect("nse_target")
        minmax_vals = _collect("minmax_target")
        qa_score_vals = _collect("selected_qa_score")

        qa_rows = []
        for row in rows:
            qa_value = _qa_score_from_metrics(row, qa_metric_weights=qa_metric_weights)
            if isinstance(qa_value, (int, float, np.number)) and np.isfinite(float(qa_value)):
                qa_rows.append(float(qa_value))

        family_row_ranks = [
            float(r)
            for series in info["rank_series"].values()
            for r in series
            if isinstance(r, (int, float, np.number)) and np.isfinite(float(r))
        ]
        model_count = len({str(r.get("name")) for r in rows})
        fold_count = len({r.get("fold") for r in rows if r.get("fold") is not None})

        health_row = {
            "family": family,
            "candidate_count": int(model_count),
            "folds": int(fold_count),
            "success_rows": int(len(rows)),
            "avg_rmse": _metric_or_nan(rmse_vals),
            "std_rmse": _metric_or_nan(np.std(rmse_vals, ddof=0)) if rmse_vals else float("nan"),
            "avg_mae": _metric_or_nan(mae_vals),
            "std_mae": _metric_or_nan(np.std(mae_vals, ddof=0)) if mae_vals else float("nan"),
            "avg_mape": _metric_or_nan(mape_vals),
            "std_mape": _metric_or_nan(np.std(mape_vals, ddof=0)) if mape_vals else float("nan"),
            "avg_r2": _metric_or_nan(r2_vals),
            "avg_corr": _metric_or_nan(corr_vals),
            "avg_me": _metric_or_nan(me_vals),
            "avg_mpe": _metric_or_nan(mpe_vals),
            "avg_minmax": _metric_or_nan(minmax_vals),
            "avg_nse": _metric_or_nan(nse_vals),
            "std_nse": _metric_or_nan(np.std(nse_vals, ddof=0)) if nse_vals else float("nan"),
            "avg_rank": _metric_or_nan(family_row_ranks),
            "rank_stability": _metric_or_nan(np.std(family_row_ranks, ddof=0))
            if len(family_row_ranks) > 1
            else float("nan"),
            "avg_qa_score": _metric_or_nan(qa_rows),
        }

        health_row["leaderboard_score"] = _metric_or_nan(
            _qa_score_from_metrics(
                {
                    "rmse_target": health_row["avg_rmse"],
                    "mae_target": health_row["avg_mae"],
                    "mape_target": health_row["avg_mape"],
                    "me_target": health_row["avg_me"],
                    "mpe_target": health_row["avg_mpe"],
                    "r2_target": health_row["avg_r2"],
                    "corr_target": health_row["avg_corr"],
                    "minmax_target": health_row["avg_minmax"],
                    "nse_target": health_row["avg_nse"],
                },
                qa_metric_weights=qa_metric_weights,
            )
        )
        health.append(
            health_row
        )

    def _health_sort_key(item):
        score = item.get("leaderboard_score")
        if isinstance(score, (int, float, np.number)) and np.isfinite(float(score)):
            primary = -float(score)
        else:
            primary = float("inf")

        rmse = item.get("avg_rmse")
        if isinstance(rmse, (int, float, np.number)) and np.isfinite(float(rmse)):
            secondary = float(rmse)
        else:
            secondary = float("inf")
        return primary, secondary

    return sorted(health, key=_health_sort_key)


def benchmark_best_model_rolling(
    combined_data: pd.DataFrame,
    candidate_grid: Optional[Sequence[Dict]] = None,
    holdout_ratio: float = 0.15,
    rolling_folds: int = 4,
    random_state: int = 0,
    candidate_model_families: Optional[Sequence[str]] = None,
    candidate_profile: str = "compact",
    qa_metric_weights: Optional[Dict[str, float]] = None,
    backtest_window: str = "auto",
    holdout_steps: Optional[int] = None,
):
    """Run rolling holdout evaluation to estimate model performance stability over time."""
    data = _to_numeric_matrix(combined_data).ffill().bfill()
    if data.shape[0] < 280:
        raise ValueError("Not enough history for rolling backtesting; need at least 280 observations.")

    if candidate_grid is None:
        candidate_grid = get_default_candidate_grid(
            include_lstm=_HAS_TENSORFLOW,
            include_pinn=_HAS_TORCH,
            profile=str(candidate_profile),
        )

    model_families = _normalize_model_families(candidate_model_families)
    if model_families:
        candidate_grid = [
            candidate
            for candidate in candidate_grid
            if str(candidate.get("model", "")).lower() in model_families
        ]
        if len(candidate_grid) == 0:
            raise ValueError("No candidates remain after filtering by candidate_model_families.")

    # Infer holdout length from desired window.
    step_seconds = float(_infer_step_delta(data.index).total_seconds())
    rows_per_day = int(max(1, np.ceil(24 * 60 * 60 / max(step_seconds, 60.0)))
    if str(backtest_window).lower() == "month":
        default_steps = 30 * rows_per_day
    elif str(backtest_window).lower() == "quarter":
        default_steps = 90 * rows_per_day
    else:
        default_steps = int(max(1, data.shape[0] * holdout_ratio))

    holdout_steps = int(holdout_steps if holdout_steps is not None else default_steps)
    holdout_steps = max(12, holdout_steps)

    max_start = data.shape[0] - holdout_steps
    if max_start <= 0:
        raise ValueError("Holdout window is larger than available history.")

    rolling_folds = max(2, int(rolling_folds))
    end = data.shape[0] - holdout_steps
    min_train_rows = 120
    if end <= min_train_rows:
        raise ValueError(
            "Not enough observations before rolling fold start. Reduce holdout_steps or choose a shorter backtest window."
        )

    start_idx = max(min_train_rows, int(0.5 * end))
    start_idx = min(start_idx, end - 1)
    fold_starts = np.linspace(start_idx, end - 1, num=rolling_folds)
    fold_starts = np.unique(np.round(fold_starts).astype(int))

    # Keep starts within available training/holdout bounds.
    fold_starts = np.array(
        [
            idx
            for idx in fold_starts
            if min_train_rows <= idx < end and idx + holdout_steps <= data.shape[0]
        ]
    )

    rolling_rows = []
    evaluated_folds = 0
    for fold_idx, fold_start in enumerate(fold_starts, start=1):
        train_df = data.iloc[:fold_start]
        holdout_df = data.iloc[fold_start : fold_start + holdout_steps]
        if train_df.shape[0] < 120 or holdout_df.shape[0] < 12:
            continue

        evaluated_folds += 1
        # Rank candidates by fold RMSE for stability tracking.
        fold_scored: List[Dict] = []
        for candidate in candidate_grid:
            result = _score_candidate(
                train_df,
                candidate,
                target_col=-1,
                random_state=random_state,
            )
            if result is None:
                continue

            include_cyclic = bool(candidate.get("include_cyclic_features", True))
            history = train_df.to_numpy()
            if result.model == "lstm":
                X, y = _build_sequence_matrix(
                    history,
                    pd.to_datetime(train_df.index),
                    n_lags=result.lags,
                    include_cyclic_features=include_cyclic,
                )
            else:
                X, y = _build_supervised_matrix(
                    history,
                    pd.to_datetime(train_df.index),
                    n_lags=result.lags,
                    include_cyclic_features=include_cyclic,
                )
            if X is None or len(X) == 0:
                continue

            feature_dim = train_df.shape[1] + (
                8 if (result.model == "lstm" and include_cyclic) else 0
            )
            try:
                model = _fit_model(
                    result.model,
                    random_state=random_state,
                    alpha=result.params.get("alpha", 1.0),
                    n_estimators=result.params.get("n_estimators", 100),
                    max_depth=result.params.get("max_depth", None),
                    learning_rate=result.params.get("learning_rate", 0.05),
                    min_samples_leaf=result.params.get("min_samples_leaf", 2),
                    hidden_units=result.params.get("hidden_units", 64),
                    batch_size=result.params.get("batch_size", 32),
                    epochs=result.params.get("epochs", 8),
                    hidden_layers=result.params.get("hidden_layers", 2),
                    physics_weight=result.params.get("physics_weight", 0.2),
                    lr=result.params.get("lr", 1e-3),
                    target_idx=result.params.get("target_idx", -1),
                    target_dim=int(train_df.shape[1]),
                    feature_dim=int(feature_dim),
                    n_series=int(train_df.shape[1]),
                    include_cyclic_features=include_cyclic,
                )
                model.fit(X, y)
                hold_pred = _forecast_with_estimator(
                    model,
                    train_df,
                    steps=len(holdout_df),
                    lags=result.lags,
                    include_cyclic_features=include_cyclic,
                    use_digital_twin=False,
                    random_state=random_state,
                )
            except Exception:
                continue
            if hold_pred is None or len(hold_pred) == 0:
                continue
            hold_pred = np.asarray(hold_pred)
            target_true = holdout_df.to_numpy()[: hold_pred.shape[0], -1]
            target_pred = hold_pred[: len(target_true), -1]
            if target_true.size == 0:
                continue

            rmse_target = float(mean_squared_error(target_true, target_pred, squared=False))
            mae_target = float(np.mean(np.abs(target_true - target_pred)))
            raw_metrics = _forecast_series_metrics(target_pred, target_true)
            fold_scored.append(
                {
                    "fold": int(fold_idx),
                    "fold_start": int(fold_start),
                    "name": result.name,
                    "model": result.model,
                    "lags": int(result.lags),
                    "rmse_target": rmse_target,
                    "mae_target": mae_target,
                    "mape_target": _metric_or_nan(raw_metrics.get("mape")),
                    "r2_target": _metric_or_nan(raw_metrics.get("r_square")),
                    "corr_target": _metric_or_nan(raw_metrics.get("corr")),
                    "mpe_target": _metric_or_nan(raw_metrics.get("mpe")),
                    "me_target": _metric_or_nan(raw_metrics.get("me")),
                    "minmax_target": _metric_or_nan(raw_metrics.get("minmax")),
                    "nse_target": _metric_or_nan(raw_metrics.get("nse")),
                    "selected_qa_score": _metric_or_nan(
                        _qa_score_from_metrics(
                            {
                                "rmse_target": rmse_target,
                                "mae_target": mae_target,
                                "mape_target": raw_metrics.get("mape"),
                                "r2_target": raw_metrics.get("r_square"),
                                "corr_target": raw_metrics.get("corr"),
                                "mpe_target": raw_metrics.get("mpe"),
                                "me_target": raw_metrics.get("me"),
                                "minmax_target": raw_metrics.get("minmax"),
                                "nse_target": raw_metrics.get("nse"),
                            },
                            qa_metric_weights=qa_metric_weights,
                        )
                    ),
                }
            )

        fold_scored = [row for row in fold_scored if np.isfinite(row.get("rmse_target", np.nan))]
        if not fold_scored:
            continue

        fold_scored = sorted(fold_scored, key=lambda row: row["rmse_target"])
        for rank, row in enumerate(fold_scored, start=1):
            row["rank"] = float(rank)
            rolling_rows.append(row)

    if not rolling_rows:
        return {
            "fold_count": 0,
            "holdout_steps": holdout_steps,
            "backtest_window": backtest_window,
            "requested_folds": int(rolling_folds),
            "evaluated_folds": int(evaluated_folds),
            "rows": [],
            "family_health": [],
        }

    return {
        "fold_count": int(len(fold_starts)),
        "requested_folds": int(rolling_folds),
        "evaluated_folds": int(evaluated_folds),
        "holdout_steps": int(holdout_steps),
        "backtest_window": str(backtest_window),
        "rows": rolling_rows,
        "family_health": benchmark_model_health_from_rolling(
            rolling_rows, qa_metric_weights=qa_metric_weights
        ),
    }
