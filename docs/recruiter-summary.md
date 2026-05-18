# Recruiter Summary

## Resume Bullets

- Built Wai, a reproducible water-level prediction research demo comparing
  tide-aware statistical models against rolling persistence and NOAA tidal
  prediction baselines.
- Implemented leakage-safe temporal evaluation across synthetic holdouts,
  rolling-origin folds, multi-horizon direct forecasts, conformal intervals,
  event metrics, and NOAA mock/live evaluation paths.
- Added generated research reporting that separates synthetic, tidecast, NOAA
  mock, and NOAA live evidence so demo metrics are not overstated as
  operational forecast proof.
- Verified strongest result: HarmonicRidge reduced average synthetic MAE
  versus rolling persistence at all evaluated horizons, including 0.0280 m vs
  0.5329 m at 6 h and 0.0297 m vs 0.4050 m at 12 h.
- Designed a Streamlit dashboard for forecast overlays, model comparison,
  high-water alerts, conformal coverage, and prototype benchmark review.

## Screenshot Recommendations

Use captions that state the evidence track.

| Screenshot | Recommended caption |
| --- | --- |
| `docs/images/actual_vs_predicted.svg` | Synthetic held-out actual vs HarmonicRidge and rolling persistence predictions |
| `docs/images/error_by_horizon.svg` | Synthetic MAE by forecast horizon, showing when tide-aware features beat persistence |
| `docs/images/baseline_comparison.svg` | 1-step synthetic baseline comparison across persistence and lightweight models |
| `docs/images/residual_plot.svg` | HarmonicRidge residuals on synthetic holdout, including remaining error structure |
| Dashboard Forecasts tab | Local Streamlit view of synthetic 1-step forecasts and conformal interval |
| Dashboard Model Comparison tab | Metrics table showing baseline, HarmonicRidge, GradBoost, and horizon results |
| Dashboard Uncertainty tab | Split-conformal coverage, including event and non-event coverage caveats |

## Interview Framing

Lead with the scientific question: "Did the hybrid tide-aware model beat a
baseline, and where?" Then give the narrow verified answer: yes on the
synthetic benchmark against rolling persistence, no live operational NOAA
claim, and NOAA mock does not beat NOAA prediction. That honesty is the point:
the repo demonstrates evaluation discipline, not inflated forecasting claims.
