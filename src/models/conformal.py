"""Split-conformal prediction intervals for Wai pipeline models.

Split-conformal is a distribution-free method that provides marginal coverage
guarantees over exchangeable calibration and test samples.

Procedure
---------
1. Train model on training data.
2. Predict on a held-out calibration split of the training data.
3. Compute nonconformity scores: s_i = |y_i - yhat_i|
4. Compute qhat = ceil((1-alpha)(n+1))/n -th quantile of {s_1, ..., s_n}.
5. At test time: prediction interval = [yhat - qhat, yhat + qhat].

Theoretical coverage (1 - alpha) holds in expectation for exchangeable samples.
For non-stationary time series, empirical coverage may fall below nominal —
the reported coverage should always be checked against the calibration set.

Honest limitations
------------------
- Coverage guarantee assumes exchangeability; tidal series are non-stationary.
- Intervals are symmetric and do not adapt to local variance.
- A calibration set of at least ~50 samples is recommended for stable qhat.

References
----------
Angelopoulos & Bates (2021), "A Gentle Introduction to Conformal Prediction."
"""

from __future__ import annotations

import numpy as np


class ConformalIntervals:
    """Split-conformal prediction intervals with symmetric absolute residuals."""

    def __init__(self, coverage: float = 0.90) -> None:
        if not 0 < coverage < 1:
            raise ValueError(f"coverage must be in (0, 1), got {coverage}")
        self.coverage = coverage
        self._qhat: float | None = None
        self._n_cal: int = 0

    def calibrate(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> "ConformalIntervals":
        """Fit qhat from calibration residuals.

        Parameters
        ----------
        actual : array-like of shape (n_cal,)
            Ground-truth values on the calibration set.
        predicted : array-like of shape (n_cal,)
            Model predictions on the calibration set.
        """
        a = np.asarray(actual, dtype=float).ravel()
        p = np.asarray(predicted, dtype=float).ravel()
        mask = ~(np.isnan(a) | np.isnan(p))
        a, p = a[mask], p[mask]
        n = len(a)
        if n == 0:
            raise ValueError("No valid calibration samples after NaN removal")
        scores = np.abs(a - p)
        level = min(1.0, np.ceil(self.coverage * (n + 1)) / n)
        self._qhat = float(np.quantile(scores, level))
        self._n_cal = n
        return self

    @property
    def qhat(self) -> float:
        if self._qhat is None:
            raise RuntimeError("Call calibrate() before accessing qhat")
        return self._qhat

    @property
    def n_cal(self) -> int:
        return self._n_cal

    def intervals(
        self,
        predictions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) prediction bounds.

        Parameters
        ----------
        predictions : array-like of shape (n,)

        Returns
        -------
        lower, upper : np.ndarray, np.ndarray
        """
        p = np.asarray(predictions, dtype=float)
        return p - self.qhat, p + self.qhat

    def empirical_coverage(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> float:
        """Fraction of actual values contained within the prediction interval."""
        a = np.asarray(actual, dtype=float).ravel()
        p = np.asarray(predicted, dtype=float).ravel()
        mask = ~(np.isnan(a) | np.isnan(p))
        a, p = a[mask], p[mask]
        if len(a) == 0:
            return float("nan")
        lower, upper = self.intervals(p)
        return float(np.mean((a >= lower) & (a <= upper)))
