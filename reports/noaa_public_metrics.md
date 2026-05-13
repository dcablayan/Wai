# Wai — Real NOAA CO-OPS Observation Evaluation

> **Data source**: Live NOAA CO-OPS water_level observations (public API).
> Metrics on *real sensor data* will differ from the synthetic-demo results.
> Higher MAE/RMSE is expected due to surge, noise, and datum uncertainty.

## San Francisco, CA — temporal holdout · NOAA_COOPS_MOCK · MOCK

- Source: `NOAA_COOPS_MOCK` · mock_used=True
- Station: `9414290` window 20240101–20240128
- Train: 2024-01-01 00:00:00+00:00 → 2024-01-21 05:54:00+00:00 (4,860 obs)
- Test:  2024-01-21 06:00:00+00:00 → 2024-01-28 00:00:00+00:00 (1,621 obs)
- Event threshold (mean+2σ on train): 0.8295 m

| Model | MAE (m) | RMSE (m) | R² | MAE 95% CI |
|-------|---------|----------|----|------------|
| Persistence (rolling 1-step) | 0.0577 | 0.0717 | 0.9709 | — |
| HarmonicRidge | 0.0403 | 0.0503 | 0.9851 | [0.0388, 0.0417] |
| GradBoost | 0.0457 | 0.0571 | 0.9808 | [0.0438, 0.0476] |

**Event metrics (HarmonicRidge, threshold exceedances):**

- Precision: nan
- Recall: nan
- F1: nan
- Peak error (on event steps): nan m
- Side-of-threshold agreement: 1.0

## Honolulu, HI — temporal holdout · NOAA_COOPS_MOCK · MOCK

- Source: `NOAA_COOPS_MOCK` · mock_used=True
- Station: `1612340` window 20240101–20240128
- Train: 2024-01-01 00:00:00+00:00 → 2024-01-21 05:54:00+00:00 (4,860 obs)
- Test:  2024-01-21 06:00:00+00:00 → 2024-01-28 00:00:00+00:00 (1,621 obs)
- Event threshold (mean+2σ on train): 0.829 m

| Model | MAE (m) | RMSE (m) | R² | MAE 95% CI |
|-------|---------|----------|----|------------|
| Persistence (rolling 1-step) | 0.0592 | 0.0742 | 0.9689 | — |
| HarmonicRidge | 0.0414 | 0.0519 | 0.9842 | [0.0400, 0.0427] |
| GradBoost | 0.0471 | 0.0591 | 0.9795 | [0.0451, 0.0488] |

**Event metrics (HarmonicRidge, threshold exceedances):**

- Precision: nan
- Recall: nan
- F1: nan
- Peak error (on event steps): nan m
- Side-of-threshold agreement: 1.0

## Boston, MA — temporal holdout · NOAA_COOPS_MOCK · MOCK

- Source: `NOAA_COOPS_MOCK` · mock_used=True
- Station: `8443970` window 20240101–20240128
- Train: 2024-01-01 00:00:00+00:00 → 2024-01-21 05:54:00+00:00 (4,860 obs)
- Test:  2024-01-21 06:00:00+00:00 → 2024-01-28 00:00:00+00:00 (1,621 obs)
- Event threshold (mean+2σ on train): 0.8309 m

| Model | MAE (m) | RMSE (m) | R² | MAE 95% CI |
|-------|---------|----------|----|------------|
| Persistence (rolling 1-step) | 0.0582 | 0.0733 | 0.9698 | — |
| HarmonicRidge | 0.0399 | 0.0502 | 0.9852 | [0.0385, 0.0412] |
| GradBoost | 0.0453 | 0.0570 | 0.9810 | [0.0437, 0.0470] |

**Event metrics (HarmonicRidge, threshold exceedances):**

- Precision: nan
- Recall: 0.0
- F1: nan
- Peak error (on event steps): 0.0849 m
- Side-of-threshold agreement: 0.9994

## Virginia Key, FL — temporal holdout · NOAA_COOPS_MOCK · MOCK

- Source: `NOAA_COOPS_MOCK` · mock_used=True
- Station: `8723214` window 20240101–20240128
- Train: 2024-01-01 00:00:00+00:00 → 2024-01-21 05:54:00+00:00 (4,860 obs)
- Test:  2024-01-21 06:00:00+00:00 → 2024-01-28 00:00:00+00:00 (1,621 obs)
- Event threshold (mean+2σ on train): 0.8293 m

| Model | MAE (m) | RMSE (m) | R² | MAE 95% CI |
|-------|---------|----------|----|------------|
| Persistence (rolling 1-step) | 0.0587 | 0.0742 | 0.9690 | — |
| HarmonicRidge | 0.0405 | 0.0511 | 0.9847 | [0.0390, 0.0420] |
| GradBoost | 0.0472 | 0.0591 | 0.9795 | [0.0452, 0.0492] |

**Event metrics (HarmonicRidge, threshold exceedances):**

- Precision: nan
- Recall: 0.0
- F1: nan
- Peak error (on event steps): 0.0856 m
- Side-of-threshold agreement: 0.9987

## La Jolla, CA — temporal holdout · NOAA_COOPS_MOCK · MOCK

- Source: `NOAA_COOPS_MOCK` · mock_used=True
- Station: `9410230` window 20240101–20240128
- Train: 2024-01-01 00:00:00+00:00 → 2024-01-21 05:54:00+00:00 (4,860 obs)
- Test:  2024-01-21 06:00:00+00:00 → 2024-01-28 00:00:00+00:00 (1,621 obs)
- Event threshold (mean+2σ on train): 0.8325 m

| Model | MAE (m) | RMSE (m) | R² | MAE 95% CI |
|-------|---------|----------|----|------------|
| Persistence (rolling 1-step) | 0.0598 | 0.0752 | 0.9683 | — |
| HarmonicRidge | 0.0409 | 0.0514 | 0.9846 | [0.0392, 0.0426] |
| GradBoost | 0.0465 | 0.0580 | 0.9804 | [0.0448, 0.0482] |

**Event metrics (HarmonicRidge, threshold exceedances):**

- Precision: nan
- Recall: nan
- F1: nan
- Peak error (on event steps): nan m
- Side-of-threshold agreement: 1.0

## Honolulu, HI (storm period) — event holdout · NOAA_COOPS_MOCK · MOCK

- Source: `NOAA_COOPS_MOCK` · mock_used=True
- Station: `1612340` window 20240112–20240118
- Train: 2024-01-12 00:00:00+00:00 → 2024-01-16 11:54:00+00:00 (1,080 obs)
- Test:  2024-01-16 12:00:00+00:00 → 2024-01-18 00:00:00+00:00 (361 obs)
- Event threshold (mean+2σ on train): 0.8681 m

| Model | MAE (m) | RMSE (m) | R² | MAE 95% CI |
|-------|---------|----------|----|------------|
| Persistence (rolling 1-step) | 0.0603 | 0.0741 | 0.9634 | — |
| HarmonicRidge | 0.0460 | 0.0572 | 0.9768 | [0.0425, 0.0494] |
| GradBoost | 0.0502 | 0.0631 | 0.9717 | [0.0455, 0.0550] |

**Event metrics (HarmonicRidge, threshold exceedances):**

- Precision: nan
- Recall: nan
- F1: nan
- Peak error (on event steps): nan m
- Side-of-threshold agreement: 1.0

## Notes

- Results on real NOAA data are **not comparable** to synthetic-demo metrics.
- Surge and meteorological forcing are not modelled; storm-period errors will be higher.
- Station holdout and multi-station generalisation evaluation are future work.
- This script uses a single 28-day window; longer evaluations require chunked API calls.
