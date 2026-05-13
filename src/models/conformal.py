"""Split-conformal prediction intervals for Wai pipeline models.

Split-conformal is a distribution-free method that provides marginal coverage
guarantees over exchangeable calibration and test samples.

Procedure
---------
1. Train model on training data.
2. Predict on a held-out calibration split of the training data.
3. Compute nonconformity scores: s_i = |y_i - yhat_i|.
4. With ``k = ceil((1 - alpha) * (n + 1))``, set qhat to the **k-th smallest
   score** (i.e. ``np.quantile(scores, ..., method='higher')``). The k-th-order
   statistic is exactly the finite-sample threshold from the Vovk/Angelopoulos
   construction — linear-interpolated quantiles undercover by ~1 / (2 n).
5. At test time: prediction interval = [yhat - qhat, yhat + qhat].

Theoretical coverage (1 - alpha) holds in expectation for exchangeable samples.
For non-stationary time series, empirical coverage may fall below nominal —
always report empirical coverage on a forward-in-time test split (not the
calibration set) and stratify by station, horizon, and event vs. non-event.

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

from typing import Dict, Optional

import numpy as np


class ConformalIntervals:
    """Split-conformal prediction intervals with symmetric absolute residuals."""

    def __init__(self, coverage: float = 0.90) -> None:
        if not 0 < coverage < 1:
            raise ValueError(f"coverage must be in (0, 1), got {coverage}")
        self.coverage = coverage
        self._qhat: float | None = None
        self._n_cal: int = 0
        self._k: Optional[int] = None

    def calibrate(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> "ConformalIntervals":
        """Fit qhat as the k-th smallest absolute calibration residual.

        ``k = ceil((1 - alpha) * (n + 1))`` where ``alpha = 1 - coverage``.
        If ``k > n`` (calibration set too small for the requested coverage),
        qhat is set to the maximum residual (the most conservative finite-
        sample bound available) and a flag is recorded.

        Parameters
        ----------
        actual, predicted : 1-D arrays of identical length (calibration set).
        """
        a = np.asarray(actual, dtype=float).ravel()
        p = np.asarray(predicted, dtype=float).ravel()
        mask = ~(np.isnan(a) | np.isnan(p))
        a, p = a[mask], p[mask]
        n = len(a)
        if n == 0:
            raise ValueError("No valid calibration samples after NaN removal")
        scores = np.sort(np.abs(a - p))

        k = int(np.ceil(self.coverage * (n + 1)))
        self._k = k
        if k > n:
            # Coverage too high for the calibration set: fall back to the max.
            self._qhat = float(scores[-1])
        else:
            # k-th smallest residual (1-indexed).
            self._qhat = float(scores[k - 1])
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

    @property
    def k(self) -> Optional[int]:
        """Order statistic used for qhat (1-indexed, returns None pre-calibration)."""
        return self._k

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
        """Fraction of actual values contained within the prediction interval.

        Note
        ----
        The intended call site is the *temporal test set* (data strictly
        after the calibration window), not the calibration set itself.
        Reporting coverage on calibration-like data overstates real-world
        skill.
        """
        a = np.asarray(actual, dtype=float).ravel()
        p = np.asarray(predicted, dtype=float).ravel()
        mask = ~(np.isnan(a) | np.isnan(p))
        a, p = a[mask], p[mask]
        if len(a) == 0:
            return float("nan")
        lower, upper = self.intervals(p)
        return float(np.mean((a >= lower) & (a <= upper)))

    def stratified_coverage(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        event_threshold: Optional[float] = None,
    ) -> Dict[str, object]:
        """Empirical coverage on future test data, split by event/non-event.

        Computes overall coverage plus, if ``event_threshold`` is given,
        coverage restricted to (a) samples where ``actual >= threshold``
        (event period) and (b) samples below the threshold (non-event).
        This is the breakdown referenced by the model card: non-event
        coverage should be near the nominal level while event coverage is
        typically lower — a known limitation of marginal split-conformal
        on non-stationary signals.
        """
        a = np.asarray(actual, dtype=float).ravel()
        p = np.asarray(predicted, dtype=float).ravel()
        mask = ~(np.isnan(a) | np.isnan(p))
        a, p = a[mask], p[mask]
        overall = self.empirical_coverage(a, p) if len(a) else float("nan")
        out: Dict[str, object] = {
            "n_samples": int(len(a)),
            "coverage_overall": overall,
            "nominal_coverage": float(self.coverage),
            "qhat": float(self._qhat) if self._qhat is not None else float("nan"),
            "k": self._k,
            "n_cal": int(self._n_cal),
        }
        if event_threshold is not None:
            ev = a >= event_threshold
            out["event_threshold"] = float(event_threshold)
            out["n_event_samples"] = int(ev.sum())
            out["n_non_event_samples"] = int((~ev).sum())
            out["coverage_event"] = (
                self.empirical_coverage(a[ev], p[ev]) if ev.any() else float("nan")
            )
            out["coverage_non_event"] = (
                self.empirical_coverage(a[~ev], p[~ev]) if (~ev).any() else float("nan")
            )
        return out
