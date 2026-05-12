"""Pure-Python tidal forecasting prototypes.

Ported from dcablayan/tideformer (prototypes.py) with minor adaptations:
  - Attribution comments added
  - Unused architectures (TidalTransformer, CoastalGNN, HybridVAE,
    NeuroHarmonic, TideFoundation) omitted; originals remain in source repo
  - TinyTidePrototype retained as the best-performing benchmark model
  - No external dependencies — stdlib + math only

All prototypes share the same window-dict interface:
    window = {
        "values"      : List[float],  # lookback observations
        "times"       : List[float],  # fractional hours from series start
        "target_value": float,        # next-step ground truth (for training)
        "target_time" : float,        # fractional hour of prediction target
    }

Use src/data/windowing.make_windows() to build windows from a raw series.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

Window = Dict[str, object]

# ── Helpers ───────────────────────────────────────────────────────────────────

def rmse(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return float("nan")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def rolling_mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def grid_search(scorer, candidates):
    best_score, best = None, None
    for c in candidates:
        s = scorer(c)
        if best_score is None or s < best_score:
            best_score, best = s, c
    return best, best_score


def hour_of_day(time_hours: float) -> float:
    return (time_hours % 24.0) / 24.0


def lunar_phase(time_hours: float) -> float:
    return (time_hours % (29.53 * 24.0)) / (29.53 * 24.0)


# ── TinyTidePrototype ─────────────────────────────────────────────────────────

class TinyTidePrototype:
    """Two-layer MLP-style forecaster with tidal covariates and skip connection.

    Best performer in the dcablayan/tideformer benchmark (mean RMSE 0.222).
    Features hour-of-day and lunar-phase covariates plus a direct skip from
    the last observation. Trains via lightweight gradient descent.
    """

    def __init__(self, lookback: int, lr: float = 0.001, epochs: int = 3):
        self.lookback = lookback
        self.lr = lr
        self.epochs = epochs
        self.hidden_weights = [0.0] * (lookback + 2)
        self.output_weights = [0.0] * lookback
        self.bias = 0.0

    def _encode(self, values: List[float], times: List[float]) -> float:
        h = sum(w * x for w, x in zip(self.hidden_weights, values))
        h += self.hidden_weights[-2] * hour_of_day(times[-1])
        h += self.hidden_weights[-1] * lunar_phase(times[-1])
        return math.tanh(h)

    def fit(self, windows: List[Window]) -> "TinyTidePrototype":
        if not windows:
            return self
        scale = 1.0 / max(len(windows), 1)
        for _ in range(self.epochs):
            random.shuffle(windows)
            for w in windows:
                values, times, target = w["values"], w["times"], w["target_value"]
                hidden = self._encode(values, times)
                skip = values[-1]
                pred = self.predict(values, times, hidden, skip)
                err = pred - target
                for i, x in enumerate(values):
                    self.hidden_weights[i] -= self.lr * 2 * err * x * scale
                    self.output_weights[i] -= self.lr * 2 * err * hidden * x * scale
                self.hidden_weights[-2] -= self.lr * 2 * err * hour_of_day(times[-1]) * scale
                self.hidden_weights[-1] -= self.lr * 2 * err * lunar_phase(times[-1]) * scale
                self.bias -= self.lr * 2 * err * scale
        return self

    def predict(self, values, times=None, hidden=None, skip=None) -> float:
        times = times or list(range(len(values)))
        hidden = self._encode(values, times) if hidden is None else hidden
        skip = values[-1] if skip is None else skip
        return sum(w * hidden for w in self.output_weights) + self.bias + skip

    def evaluate(self, windows: List[Window]) -> float:
        return rmse(
            [w["target_value"] for w in windows],
            [self.predict(w["values"], w["times"]) for w in windows],
        )


# ── HarmonicNetPrototype ──────────────────────────────────────────────────────

class HarmonicNetPrototype:
    """Physics-informed harmonic projection over 8 tidal constituents.

    Periods (hours): M2=12.42, S2=12.0, K1=23.93, O1=25.82, Mm=327.9,
    MSf=661.3, M4=6.21, M6=4.14.  A causal residual smoothing head
    captures non-harmonic variance.
    """

    def __init__(self, lookback: int):
        self.lookback = lookback
        self.periods = [12.42, 12.0, 23.93, 25.82, 327.9, 661.3, 6.21, 4.14]
        self.coeffs: List[Tuple[float, float]] = [(0.0, 0.0)] * len(self.periods)
        self.residual_mean = 0.0

    def _fit_harmonics(self, times: List[float], values: List[float]) -> None:
        n = len(values)
        if n == 0:
            return
        self.coeffs = []
        for period in self.periods:
            omega = 2 * math.pi / period
            a = (2 / n) * sum(v * math.sin(omega * t) for v, t in zip(values, times))
            b = (2 / n) * sum(v * math.cos(omega * t) for v, t in zip(values, times))
            self.coeffs.append((a, b))

    def _harmonic_value(self, t: float) -> float:
        return sum(
            a * math.sin(2 * math.pi / p * t) + b * math.cos(2 * math.pi / p * t)
            for (a, b), p in zip(self.coeffs, self.periods)
        )

    def fit(self, windows: List[Window]) -> "HarmonicNetPrototype":
        if not windows:
            return self
        values, times = [], []
        for w in windows:
            values.extend(w["values"])
            times.extend(w["times"])
            if len(values) >= self.lookback * 6:
                break
        self._fit_harmonics(times, values)
        residuals = [
            v - self._harmonic_value(t)
            for w in windows
            for v, t in zip(w["values"], w["times"])
        ]
        if residuals:
            self.residual_mean = rolling_mean(residuals[-self.lookback:])
        return self

    def predict(self, window: Window) -> float:
        return self._harmonic_value(window["target_time"]) + self.residual_mean

    def evaluate(self, windows: List[Window]) -> float:
        return rmse(
            [w["target_value"] for w in windows],
            [self.predict(w) for w in windows],
        )


# ── WaveGRUPrototype ──────────────────────────────────────────────────────────

class WaveGRUPrototype:
    """Bidirectional double-exponential smoothing with attention-like weighting.

    Emulates bidirectional GRU gating by combining forward and backward EMA
    passes.  A lightweight attention mechanism upweights steps with larger
    residual energy.  Second-best performer in the tideformer benchmark
    (mean RMSE 0.911).
    """

    def __init__(self, lookback: int):
        self.lookback = lookback
        self.alpha = 0.5
        self.beta = 0.1
        self.attention_temperature = 2.0

    def _double_exp(self, values: List[float], alpha: float, beta: float) -> float:
        level = values[0]
        trend = (values[1] - values[0]) if len(values) > 1 else 0.0
        for val in values[1:]:
            prev = level
            level = alpha * val + (1 - alpha) * (level + trend)
            trend = beta * (level - prev) + (1 - beta) * trend
        return level + trend

    def _attentive(self, values: List[float], alpha: float, beta: float) -> float:
        fwd = self._double_exp(values, alpha, beta)
        bwd = self._double_exp(list(reversed(values)), alpha, beta)
        residuals = [
            abs(values[i] - rolling_mean(values[max(0, i - 2): i + 1]))
            for i in range(len(values))
        ]
        weights = [math.exp(r * self.attention_temperature) for r in residuals]
        norm = sum(weights) or 1.0
        attention = sum(wt * v for wt, v in zip(weights, values)) / norm
        return 0.4 * fwd + 0.4 * bwd + 0.2 * attention

    def fit(self, windows: List[Window]) -> "WaveGRUPrototype":
        if not windows:
            return self
        candidates = [(0.3, 0.05), (0.5, 0.1), (0.7, 0.15), (0.85, 0.2)]

        def score(candidate):
            a, b = candidate
            return rmse(
                [w["target_value"] for w in windows],
                [self._attentive(w["values"], a, b) for w in windows],
            )

        best, _ = grid_search(score, candidates)
        if best:
            self.alpha, self.beta = best
        return self

    def predict(self, window: Window) -> float:
        return self._attentive(window["values"], self.alpha, self.beta)

    def evaluate(self, windows: List[Window]) -> float:
        return rmse(
            [w["target_value"] for w in windows],
            [self.predict(w) for w in windows],
        )


# ── SurgeNetPrototype ─────────────────────────────────────────────────────────

class SurgeNetPrototype:
    """Dual-head tide + surge estimator.

    Uses HarmonicNetPrototype as the tide head and estimates surge magnitude
    from the harmonic residual.  External values (e.g. NOAA wind/pressure
    proxy) can be passed in each window dict as "external_values".

    Returns (prediction, surge_magnitude) from predict().
    """

    def __init__(self, lookback: int):
        self.lookback = lookback
        self.harmonic_head = HarmonicNetPrototype(lookback)
        self.surge_sensitivity = 0.5

    def fit(self, windows: List[Window]) -> "SurgeNetPrototype":
        self.harmonic_head.fit(windows)
        if not windows:
            return self
        residuals = [
            abs(w["target_value"] - self.harmonic_head.predict(w))
            for w in windows
        ]
        self.surge_sensitivity = rolling_mean(residuals) or 0.5
        return self

    def predict(self, window: Window) -> Tuple[float, float]:
        harmonic = self.harmonic_head.predict(window)
        external = window.get("external_values", [])
        surge_bias = rolling_mean(external) if external else 0.0
        surge_mag = self.surge_sensitivity + surge_bias
        return harmonic + surge_mag * 0.2, surge_mag

    def evaluate(self, windows: List[Window]) -> float:
        return rmse(
            [w["target_value"] for w in windows],
            [self.predict(w)[0] for w in windows],
        )


# ── TsunamiSentinelPrototype ──────────────────────────────────────────────────

class TsunamiSentinelPrototype:
    """High-pass multi-scale anomaly detector for rapid water-level spikes.

    Computes residual energy relative to a local rolling baseline across
    multiple scales and flags windows exceeding a learned energy threshold.

    Returns (next_value_prediction, tsunami_flag) from predict().
    """

    def __init__(self, lookback: int):
        self.lookback = lookback
        self.threshold = 0.5

    def fit(self, windows: List[Window]) -> "TsunamiSentinelPrototype":
        if not windows:
            return self
        energies = []
        for w in windows:
            baseline = rolling_mean(w["values"])
            hp = [v - baseline for v in w["values"]]
            ms = [max(hp[max(0, i - 2): i + 1]) for i in range(len(hp))]
            energies.append(math.sqrt(rolling_mean([h ** 2 for h in ms])))
        self.threshold = rolling_mean(energies) * 2.0
        return self

    def predict(self, window: Window) -> Tuple[float, bool]:
        baseline = rolling_mean(window["values"])
        hp = [v - baseline for v in window["values"]]
        ms = [max(hp[max(0, i - 2): i + 1]) for i in range(len(hp))]
        energy = math.sqrt(rolling_mean([h ** 2 for h in ms]))
        flag = energy > self.threshold
        return window["values"][-1] + hp[-1], flag

    def evaluate(self, windows: List[Window]) -> float:
        return rmse(
            [w["target_value"] for w in windows],
            [self.predict(w)[0] for w in windows],
        )
