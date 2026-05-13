# Model Card — Wai Coastal Water-Level Forecasting

> **Status: Research Demo — Not an Operational System**
>
> Wai is a portfolio-grade research demonstration. It is not a deployed,
> validated operational flood-warning system and must not be used to make
> emergency-management, evacuation, or infrastructure decisions.

---

## Model Details

| Field | Value |
|-------|-------|
| Version | 0.1.0 |
| Authors | Dylan Cablayan |
| Type | Supervised regression (Ridge, HistGradientBoosting) + rule-based prototype models |
| Task | Multi-step-ahead coastal water-level forecasting |
| Horizons | 1 step (6 min), 6 h, 12 h, 24 h |
| Inputs | Tidal harmonic sin/cos features (8 constituents), temporal covariates, lag observations, rolling statistics |
| Outputs | Predicted water level (metres, MLLW datum) + optional conformal prediction intervals |

---

## Intended Use

**In scope:**
- Portfolio demonstration of an end-to-end time-series forecasting pipeline
- Reproducing the feature engineering, evaluation, and reporting methodology described in the codebase
- Learning resource for tidal harmonic modelling and conformal uncertainty quantification
- Research baseline for comparing harmonic regression to more complex architectures

**Out of scope:**
- Real-time operational flood forecasting or emergency alerts
- Insurance underwriting or infrastructure risk assessment
- Replacing or supplementing official NOAA products (CO-OPS tidal predictions, NWS storm-surge forecasts)
- Any context where model errors could cause harm

---

## Scientific Hypotheses

The following claims are testable within this codebase and constitute the scientific contribution:

1. **Tidal harmonics dominate predictable variance.** Eight-constituent harmonic features (M2, S2, K1, O1, N2, M4, M6, Mm) alone account for ≥ 98% of R² on the synthetic demo series. *(Supported by ablation study — see below.)*

2. **Adding lag and rolling features provides incremental improvement over harmonics alone.** The full feature set (harmonics + lags + rolling) achieves marginally lower MAE than harmonics-only. *(Supported by ablation study.)*

3. **Rolling 1-step persistence is the correct 6-minute naive baseline.** Constant holdout persistence degrades rapidly at longer horizons and artificially inflates the apparent advantage of learned models. *(Supported by the two-variant persistence comparison in `model_metrics.json`.)*

4. **Direct multi-horizon forecasting avoids compounding error.** Training a separate model per horizon is a principled alternative to iterated (recursive) prediction. *(Evaluated in `horizon_metrics.json`.)*

5. **Split-conformal intervals achieve ≥ 90% empirical coverage on the in-distribution test set.** Coverage may fall below the nominal level when the series is non-stationary or when evaluated on out-of-distribution periods (e.g. storm surge). *(Tested in `test_conformal.py`.)*

---

## Ablation Study Results (synthetic demo data)

The ablation varies which feature groups are given to HarmonicRidge.
All results on the temporal holdout (last 25% of data per station).

### DEMO-HNL (Honolulu synthetic)

| Feature set | n features | MAE (m) | RMSE (m) | R² |
|-------------|-----------|---------|----------|----|
| harmonics_only | 20 | 0.0178 | 0.0223 | 0.9902 |
| lags_only | 6 | 0.0194 | 0.0243 | 0.9883 |
| rolling_only | 6 | 0.0198 | 0.0248 | 0.9878 |
| harmonics_lags | 26 | 0.0175 | 0.0219 | 0.9905 |
| **full** | **32** | **0.0168** | **0.0211** | **0.9912** |

### DEMO-SFO (San Francisco synthetic)

| Feature set | n features | MAE (m) | RMSE (m) | R² |
|-------------|-----------|---------|----------|----|
| harmonics_only | 20 | 0.0181 | 0.0227 | 0.9980 |
| lags_only | 6 | 0.0198 | 0.0248 | 0.9976 |
| rolling_only | 6 | 0.0306 | 0.0380 | 0.9944 |
| harmonics_lags | 26 | 0.0175 | 0.0219 | 0.9981 |
| **full** | **32** | **0.0168** | **0.0211** | **0.9983** |

**Takeaway**: Tidal harmonics are the strongest single feature group; lags and rolling statistics provide a small but consistent improvement. Rolling features alone are the weakest single group, and their benefit over lags is station-dependent.

---

## Training Data

| Dataset | Source | Status |
|---------|--------|--------|
| `data/demo/demo_water_levels.csv` | Fully synthetic (M2/S2/K1/O1 + Gaussian noise + synthetic surge) | **Not real sensor data** |
| `data/demo/tidecast/*.csv` | NOAA-derived tidal predictions (harmonic model output, not raw observations) | **Not raw sensor data** |
| NOAA CO-OPS via API | Public, no key required | Real observations — not stored in this repo |

All pipeline metrics reported in `reports/model_metrics.json` and `reports/horizon_metrics.json` are computed on **synthetic demo data**. They are not directly comparable to published operational benchmark results.

---

## Evaluation Protocol

### Temporal holdout
- Split: last 25% of each station's time series (by timestamp)
- No shuffling; temporal order is preserved
- Feature lag warm-up rows are excluded from both train and test

### Leakage controls
- Rolling features are computed on `shift(1)` to exclude `water_level[t]` when predicting `water_level[t]`
- Horizon features preserve original row indices after `dropna()` to prevent boundary shift
- Tests in `tests/test_leakage.py` and `tests/test_split_integrity.py` enforce these invariants

### Multi-horizon strategy
- Direct forecasting: a separate model is trained per horizon
- Persistence at horizon h: rolling h-step (pred[t] = obs[t-h])
- WaveGRU is evaluated at 1-step only; iterated prediction is out of scope

### NOAA real-observation evaluation
- Run `python -m scripts.evaluate_noaa_public` (or `NOAA_OFFLINE=1 ...` for CI)
- Temporal holdout on 28-day windows for 5 geographically diverse stations
- Storm-period holdout for Honolulu (Jan 2024)
- Results saved to `reports/noaa_public_metrics.json`

---

## Metrics

| Metric | Definition | Notes |
|--------|-----------|-------|
| MAE | Mean absolute error (m) | Primary metric |
| RMSE | Root mean squared error (m) | Penalises large errors more |
| R² | Coefficient of determination | 1 = perfect; negative = worse than mean |
| NSE | Nash-Sutcliffe efficiency = R² for point forecasts | Hydrology convention |
| Precision/Recall | Threshold exceedance classification | Event-level; uses training-split threshold |
| Peak error | Max \|actual - forecast\| on event steps | Operational relevance |
| Bootstrap 95% CI | 1000-replicate percentile bootstrap on MAE/RMSE | Reports uncertainty in estimates |

---

## Known Limitations and Failure Modes

1. **Synthetic training data.** All pipeline metrics are on a synthetic series with known tidal structure and Gaussian noise. Real sensor data includes datum uncertainty, communication dropouts, biofouling drift, and unmodelled surge. MAE on real data will be substantially higher.

2. **No meteorological forcing.** Storm surge (wind stress + inverse barometer) is not modelled. The model cannot predict surge events from first principles — it can only extrapolate the tidal signal.

3. **No datum / datum-mismatch handling.** The loader issues a warning when the API response datum differs from the requested datum, but does not automatically convert. Users must ensure consistent datum use.

4. **Conformal intervals assume exchangeability.** Tidal series are non-stationary; coverage may fall below 90% nominal during regime changes or storm events.

5. **Prototype model names are misleading.** "ʻAle Piʻi (Rising Wave)" and "Kai Eʻe (Tsunami)" are pure-Python research baselines, not production surge or tsunami detectors. The names are aesthetic choices, not capability claims.

6. **Tidecast benchmark metrics are on smooth data.** NOAA tidal predictions are deterministic harmonic output — inherently easier than real sensor data. Prototype model RMSE on tidecast data cannot be compared to literature results on real observations.

7. **Short evaluation windows.** 28-day NOAA API windows are insufficient to characterise seasonal variability, multi-year climate signals, or rare extreme events.

8. **No spatial generalisation test.** All models are trained and tested at the same station. Station-holdout (train on N-1 stations, test on held-out station) has not been evaluated.

---

## Not an Operational System

Wai does not:
- Issue real-time flood warnings or emergency advisories
- Connect to live tide-gauge infrastructure
- Incorporate NWS storm-surge or wave-height forecasts
- Hold any operational certification (e.g. NOAA Weather-Ready Nation)
- Replace or supplement NOAA CO-OPS official products

Anyone requiring operational coastal flood forecasts should consult:
- [NOAA CO-OPS Tidal Predictions](https://tidesandcurrents.noaa.gov/)
- [NWS Storm Surge Unit](https://www.nhc.noaa.gov/surge/)
- [NOAA Weather-Ready Nation](https://www.weather.gov/wrn/)

---

## Citation

If you reference this work in academic or professional writing, please use:

```
Cablayan, D. (2026). Wai: A coastal water-level forecasting research demo.
GitHub: https://github.com/dcablayan/Wai
```
