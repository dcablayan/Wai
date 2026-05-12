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

## Models Implemented

### 1. Persistence Baseline (`PersistenceModel`)

**What it does**: predicts the last observed water level for every future step.

**Why it matters**: A persistence model is often surprisingly competitive for
very short horizons (< 1 hour). If any model cannot beat it, it is not useful.

**Limitations**: Falls apart quickly beyond a tidal half-cycle (~6 hours).

---

### 2. Harmonic Ridge (`HarmonicRidgeModel`)

**What it does**: Fits a Ridge-regularised linear regression over:
- Sin/cos pairs for 5 tidal constituents: M2 (12.42 h), S2 (12.00 h),
  K1 (23.93 h), O1 (25.82 h), N2 (12.66 h)
- Lagged water-level observations at 6, 12, 24, 60, 120, 240 minutes
- Rolling mean and std at 1-hour, 4-hour, and 24-hour windows

**Why it works**: The dominant tidal signal is well-approximated by a small
set of sinusoids. Lags capture short-range autocorrelation that the harmonics
miss (e.g. storm surge).  Ridge shrinks noisy feature coefficients.

**Limitations**:
- Purely linear — cannot model interaction effects.
- Lag features introduce data-leakage risk if horizon > lag depth.
- Cannot forecast storm surge from first principles (no atmospheric input).
- Demo data is synthetic; real-world RMSE will differ.

---

## Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAE    | mean\|y − ŷ\| | Average absolute error in meters |
| RMSE   | √mean(y − ŷ)² | Penalises large errors more than MAE |
| R²     | 1 − SS_res / SS_tot | Fraction of variance explained; 1 = perfect |
| NSE    | = R² for point forecasts | Nash-Sutcliffe efficiency (hydrology standard) |
| Corr   | Pearson r | Linear correlation; does not capture bias |

Metrics are computed on a held-out **temporal test set** (last 25% of data
per station) to avoid look-ahead bias. No shuffling.

---

## Honest Limitations

- All results on the demo dataset are from **synthetic data** and cannot
  be compared to real-world forecasting benchmarks.
- The harmonic model does not account for sea-level rise, salinity, or
  meteorological forcing beyond what the lagged observations encode.
- No hyperparameter search is performed; Ridge alpha=1.0 is a reasonable
  default but not optimal for every station.

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

The codebase is structured so that replacing `HarmonicRidgeModel` with any of
the above requires only changing `src/models/baseline.py` — the data pipeline,
feature engineering, and reporting layers remain unchanged.
