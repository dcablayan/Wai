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

### 1. ʻAle Kūpaʻa (Steady Wave) — `PersistenceModel`

**What it does**: predicts the last observed water level for every future step.

**Why it matters**: A persistence model is often surprisingly competitive for
very short horizons (< 1 hour). If any model cannot beat it, it is not useful.

**Limitations**: Falls apart quickly beyond a tidal half-cycle (~6 hours).

---

### 2. Nalu Hoʻokani Ridge (Harmonic Wave Ridge) — `HarmonicRidgeModel`

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
- Optional numeric external forcing columns when supplied, including
  `wind_speed_mps`, `wind_direction_deg`, `air_pressure_hpa`, `rainfall_mm`,
  and `wave_height_m`

**Why it works**: The dominant tidal signal is well-approximated by a small
set of sinusoids. Lags capture short-range autocorrelation that the harmonics
miss (e.g. non-tidal residual excursions). Temporal covariates improve diurnal
phasing.
Ridge shrinks noisy feature coefficients.

**Limitations**:
- Purely linear — cannot model interaction effects.
- Lag features introduce data-leakage risk if horizon > lag depth.
- Checked-in reports cannot validate storm-surge skill because they do not
  include real atmospheric or wave inputs.
- Demo data is synthetic; real-world RMSE will differ significantly.

---

### 3. Nalu Holo Adapter (Fast Wave Adapter) — `WaveGRUModel`

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

> **Naming note**: Each prototype carries a Hawaiian wave display name
> (e.g. `ʻAle Iki (Ripple)`) used in reports and the dashboard. Internal Python
> class names (`TinyTidePrototype`, `HarmonicNetPrototype`, etc.) remain
> unchanged for import stability. The full mapping is in
> `src/models/branding.py`.

### TinyTidePrototype

Two-layer MLP-style forecaster with hour-of-day and lunar-phase covariates
plus a direct skip connection from the last observation. Trains via
lightweight gradient descent (no autograd). Compare it to the persistence
row in `reports/benchmark_results.md`; do not assume it always wins.

### HarmonicNetPrototype

Physics-informed projection over 8 tidal constituents (M2, S2, K1, O1, Mm,
MSf, M4, M6) using least-squares amplitude fitting on the training windows,
plus a causal residual smoothing head. Underperforms on short windows because
the long-period constituents (Mm, MSf) need months of data to fit reliably.

### WaveGRUPrototype

Bidirectional double-exponential smoothing with attention-like weighting.
Alpha/beta selected via a 4-candidate grid search on training windows.
This is a smoothing heuristic, not a real GRU and not a deep-learning model.
It is the algorithm wrapped by `WaveGRUModel` in the pipeline tier.

### SurgeNetPrototype

Residual heuristic: `HarmonicNetPrototype` as the tide head plus a residual
magnitude estimated from harmonic error. Accepts optional `external_values` in
the window dict, but it is not meteorological surge modeling and should not be
described as storm-surge prediction.

### TsunamiSentinelPrototype

High-pass multi-scale anomaly detector. Computes residual energy relative to
a local rolling baseline across multiple scales and flags windows exceeding a
learned energy threshold. Returns `(next_value_prediction, tsunami_flag)`.
This is an anomaly toy, not a validated tsunami detector.

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

## Implemented Capabilities

Beyond the three original pipeline models, Wai now includes:

### 4. Gradient Boosting Baseline (`GradBoostModel`)

**What it does**: `HistGradientBoostingRegressor` (scikit-learn) over the same
8-constituent tidal feature matrix as `HarmonicRidgeModel`. Replaces the Ridge
estimator with a gradient-boosted decision tree ensemble, capturing non-linear
interactions between tidal constituents, lags, and rolling statistics.

**Why it's useful**: Allows direct comparison of linear vs. non-linear skill
on an identical feature set. No additional dependencies required.

**Limitations**: Default hyperparameters; not grid-searched. On a periodic
tidal signal the gain over Ridge is modest. Does not generalise across stations
without retraining.

---

### 5. Multi-Horizon Evaluation (`scripts/evaluate_horizons.py`)

Evaluates Persistence, HarmonicRidge, and GradBoost at four horizons:
1 step (6 min), 60 steps (6 h), 120 steps (12 h), 240 steps (24 h).

**Strategy**: direct forecasting — a separate model is trained per horizon
with the target shifted h steps forward. This avoids look-ahead bias.

**WaveGRU** is only evaluated at horizon 1 because it is a 1-step algorithm.

**Limitations**: Direct forecasting is an optimistic estimate of skill at
longer horizons (it is trained for exactly that horizon). Lag features at long
horizons still reference observations prior to the prediction time, so no
leakage is introduced, but the feature relevance degrades with horizon length.

---

### 6. Conformal Prediction Intervals (`src/models/conformal.py`)

**What it does**: Split-conformal prediction intervals using symmetric
absolute residuals as nonconformity scores. Given a calibration set, computes
qhat (the corrected quantile of calibration residuals) and returns
`[ŷ − qhat, ŷ + qhat]` for any model's predictions.

**Coverage guarantee**: Marginal coverage ≥ (1 − α) in expectation over
exchangeable calibration and test samples.

**Limitations**: Assumes exchangeability; tidal series are non-stationary, so
empirical coverage may fall below the nominal level, especially for longer
forecast horizons. Intervals are symmetric (constant width) and do not adapt
to local variance.

---

### 7. NOAA Tidal Predictions Loader (`src/data/loader.py`)

`load_noaa_predictions()` fetches NOAA CO-OPS deterministic tidal predictions
(the `predictions` product) for any gauged station and date range. No API key
required. Useful as a high-quality tidal signal for additional feature
construction or direct comparison.

**Limitations**: Predictions are deterministic harmonics — no meteorological
forcing or surge component.

---

### 8. Optional Meteorological Forcing Columns (`src/features/meteorology.py`)

The tabular feature builder accepts numeric external forcing columns when a
caller supplies them: `wind_speed_mps`, `wind_direction_deg`,
`air_pressure_hpa`, `rainfall_mm`, and `wave_height_m`.

**Limitations**: This is feature support, not validation. The checked-in
synthetic, tidecast, NOAA mock, and NOAA live reports do not include real
forcing covariates, so storm-surge performance remains unproven.

---

### 9. High-Water Alert Detection (`src/alerts.py`)

Three configurable threshold modes:
- `'std'` — mean + k × std_dev (configurable k; default 2.0)
- `'absolute'` — fixed water-level value in series units
- `'percentile'` — p-th percentile of a reference distribution

Alert events are summarised in `reports/alert_summary.json` and shown in the
dashboard Alerts tab.

**Limitations**: Static thresholds computed on training data; does not adapt
to seasonal sea-level variation or long-term trends.

---

### 10. Spatial Interpolation (`src/features/spatial.py`)

Inverse-distance weighting (IDW) across stations with valid lat/lon
coordinates. Only applicable when two or more stations are available.

**Limitations**: Deterministic method; no uncertainty quantification. Not
designed for extrapolation beyond the station network. Coastal geometry
(along-shore vs. cross-shore) is not modelled.

---

## Honest Limitations

- All pipeline results are from **synthetic data** and cannot be compared to
  real-world forecasting benchmarks.
- All prototype benchmark results are from **NOAA-derived tidal predictions**
  (smooth, deterministic). Real sensor data would yield substantially higher
  RMSE.
- No hyperparameter search is performed for pipeline models; defaults are
  reasonable starting points, not optimised values.
- Prototype models (WaveGRU, TinyTide, etc.) are 1-step algorithms; they are
  not evaluated at multi-step horizons in the current implementation.
- Advanced deep learning (LSTM, Transformer) and validated meteorological
  surge modeling are intentionally excluded to keep the repo lightweight and
  honest.
