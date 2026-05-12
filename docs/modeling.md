# Wai Modeling Guide

## Problem Statement

Coastal water levels are driven by a combination of:
- **Astronomical tides** — predictable harmonic cycles from gravitational forcing
  by the Moon and Sun (M2, S2, K1, O1, N2 constituents and ~35 others).
- **Meteorological effects** — storm surge, wind setup, pressure-driven changes.
- **Seasonal / sea-level trends** — thermal expansion, freshwater runoff, ENSO.
- **Sensor noise** — measurement error, fouling, communication dropouts.

The goal is to forecast water level at a coastal station for the next N steps
(6-minute intervals) given recent observations and tidal knowledge.

---

## Part A — Production-style pipeline models

These models share the same DataFrame API and are evaluated against synthetic
demo data on a temporal hold-out (last 25% per station).

### 1. Persistence Baseline (`PersistenceModel`)

**What it does**: predicts the last observed water level for every future step.

**Why it matters**: A persistence model is often surprisingly competitive for
very short horizons (< 1 hour). If any model cannot beat it, it is not useful.

**Limitations**: Falls apart quickly beyond a tidal half-cycle (~6 hours).

---

### 2. Harmonic Ridge (`HarmonicRidgeModel`)

**What it does**: Fits a Ridge-regularised linear regression over:

- Sin/cos pairs for **8 tidal constituents**:
  - M2 (12.42 h) — principal lunar semi-diurnal
  - S2 (12.00 h) — principal solar semi-diurnal
  - K1 (23.93 h) — lunisolar diurnal
  - O1 (25.82 h) — principal lunar diurnal
  - N2 (12.66 h) — larger lunar elliptic semi-diurnal
  - M4 (6.21 h) — shallow-water overtide of M2
  - M6 (4.14 h) — shallow-water overtide (3rd harmonic)
  - Mm (327.9 h) — lunar monthly
- **Temporal covariates**: hour-of-day and lunar-phase sin/cos (adapted from
  `dcablayan/tideformer` prototype helpers)
- Lagged water-level observations at 6, 12, 24, 60, 120, 240 minutes
- Rolling mean and std at 1-hour, 4-hour, and 24-hour windows

**Why it works**: The dominant tidal signal is well-approximated by a small
set of sinusoids. Lags capture short-range autocorrelation that the harmonics
miss (e.g. storm surge). Temporal covariates improve diurnal phasing.
Ridge shrinks noisy feature coefficients.

**Limitations**:
- Purely linear — cannot model interaction effects.
- Lag features introduce data-leakage risk if horizon > lag depth.
- Cannot forecast storm surge from first principles (no atmospheric input).
- Demo data is synthetic; real-world RMSE will differ significantly.

---

### 3. WaveGRU Adapter (`WaveGRUModel`)

**What it does**: Wraps `WaveGRUPrototype` (from `src/models/prototypes.py`) in
the standard DataFrame API. Internally uses bidirectional double-exponential
smoothing with an attention-like weighting over residual energy. Operates on
raw water-level values only — no tidal feature engineering is applied.

**Why it's included**: Provides a complementary baseline that requires no
domain-specific feature engineering. Useful for comparing tidal-feature-based
models against a general time-series smoother.

**Limitations**:
- This is a pure-Python smoothing algorithm, not a trained GRU network. The
  "GRU" name reflects the bidirectional gating concept it emulates, not a
  deep learning implementation.
- Performance on multi-step horizons degrades faster than HarmonicRidge.
- Alpha/beta hyperparameters are selected via a 4-candidate grid search, not
  gradient-based optimisation.

---

## Part B — Prototype benchmark models

These are pure-Python research prototypes ported from
[dcablayan/tideformer](https://github.com/dcablayan/tideformer) (see
`src/models/prototypes.py`). They use stdlib math only — no PyTorch,
TensorFlow, numpy, or scikit-learn. They accept and return window dicts
(see `src/data/windowing.make_windows()`).

They are evaluated on NOAA-derived tidecast predictions (smooth, deterministic
signal) — see [`docs/metrics_interpretation.md`](metrics_interpretation.md)
for why this matters.

### TinyTidePrototype

Two-layer MLP-style forecaster with hour-of-day and lunar-phase covariates
plus a direct skip connection from the last observation. Trains via lightweight
gradient descent (no autograd). **Best performer** on tidecast data (mean RMSE
0.222 ft).

### HarmonicNetPrototype

Physics-informed projection over 8 tidal constituents (M2, S2, K1, O1, Mm,
MSf, M4, M6) using least-squares amplitude fitting on the training windows,
plus a causal residual smoothing head. Underperforms on short windows because
the long-period constituents (Mm, MSf) need months of data to fit reliably.

### WaveGRUPrototype

Bidirectional double-exponential smoothing with attention-like weighting.
Alpha/beta selected via a 4-candidate grid search on training windows.
Second-best performer on tidecast (mean RMSE 0.911 ft). This is the algorithm
wrapped by `WaveGRUModel` in the pipeline tier.

### SurgeNetPrototype

Dual-head model: `HarmonicNetPrototype` as the tide head, plus a surge
magnitude estimated from the harmonic residual. Accepts optional
`external_values` in the window dict (e.g. wind/pressure proxy). Returns
`(prediction, surge_magnitude)`. Currently underperforms because the surge
head amplifies harmonic fitting error.

### TsunamiSentinelPrototype

High-pass multi-scale anomaly detector. Computes residual energy relative to
a local rolling baseline across multiple scales and flags windows exceeding a
learned energy threshold. Returns `(next_value_prediction, tsunami_flag)`.
Not evaluated with RMSE; designed for detection, not point forecasting.

**Important**: All prototype model names ("GRU", "Net", "AI") are descriptive
of the algorithmic concept they implement, not claims about deep learning
infrastructure. These are benchmark baselines — they demonstrate ideas, not
production performance.

---

## Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAE    | mean\|y − ŷ\| | Average absolute error |
| RMSE   | √mean(y − ŷ)² | Penalises large errors more than MAE |
| R²     | 1 − SS_res / SS_tot | Fraction of variance explained; 1 = perfect |
| NSE    | = R² for point forecasts | Nash-Sutcliffe efficiency (hydrology standard) |
| Corr   | Pearson r | Linear correlation; does not capture bias |

Pipeline model metrics are in **meters** (synthetic demo data).
Prototype benchmark metrics are in **feet** (tidecast NOAA predictions).

Metrics are computed on a held-out **temporal test set** (no shuffling) to
avoid look-ahead bias.

See [`docs/metrics_interpretation.md`](metrics_interpretation.md) for a full
discussion of what these numbers do and do not mean.

---

## Honest Limitations

- All pipeline results are from **synthetic data** and cannot be compared to
  real-world forecasting benchmarks.
- All prototype benchmark results are from **NOAA-derived tidal predictions**
  (smooth, deterministic). Real sensor data would yield substantially higher
  RMSE.
- No hyperparameter search is performed for pipeline models; Ridge alpha=1.0
  is a reasonable default but not optimal for every station.
- Prototype models have no multi-step forecasting capability in the current
  implementation (horizon=1 only).

---

## Next Steps / Roadmap

| Idea | Complexity | Value |
|------|-----------|-------|
| Gradient boosting (XGBoost / LightGBM) | Low | Medium — captures non-linear interactions |
| LSTM encoder | Medium | High — learns long-range tidal patterns automatically |
| Transformer (Informer / Autoformer) | High | High — state-of-the-art for long-horizon forecasting |
| Harmonic constituent fitting (t_tide / utide) | Low | High — true tidal analysis, not regression |
| NOAA tidal predictions as a feature | Low | Medium — free, high-quality tidal signal |
| Storm-surge covariate (NWS forecast) | Medium | High — critical for flood early warning |
| Uncertainty quantification (conformal) | Medium | High — actionable for coastal managers |
| Multi-step evaluation (6h, 12h, 24h horizons) | Low | High — honest skill assessment |

The codebase is structured so that replacing `HarmonicRidgeModel` with any of
the above requires only changing `src/models/baseline.py` — the data pipeline,
feature engineering, and reporting layers remain unchanged.
