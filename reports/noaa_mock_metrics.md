# Wai NOAA Mock Evaluation

> Data source: Offline mock fixtures. These are synthetic sanity checks.
> NOAA tidal predictions are deterministic harmonics; they are a serious baseline, not ground truth.

## San Francisco, CA - temporal holdout - NOAA_COOPS_MOCK MOCK

- Station: `9414290` window 20240101-20240128
- Observations: `NOAA_COOPS_MOCK`; predictions: `NOAA_PREDICTIONS_MOCK`; mock_used=True
- Train: 2024-01-01 00:00:00+00:00 to 2024-01-21 05:54:00+00:00 (4,860 obs)
- Test: 2024-01-21 06:00:00+00:00 to 2024-01-28 00:00:00+00:00 (1,621 obs)
- Event threshold: train mean + 2 std = 0.8273 m

| Model | MAE | RMSE | R2/NSE | MAE skill vs rolling | MAE skill vs NOAA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rolling persistence | 0.0590 | 0.0745 | 0.9690 | 0.0000 | -0.4703 |
| NOAA prediction | 0.0402 | 0.0505 | 0.9857 | 0.3199 | 0.0000 |
| NOAA residual persistence | 0.0572 | 0.0720 | 0.9710 | 0.0312 | -0.4243 |
| HarmonicRidge | 0.0407 | 0.0511 | 0.9848 | 0.3074 | -0.0177 |
| GradBoost | 0.0465 | 0.0581 | 0.9803 | 0.2075 | -0.1645 |
| Hybrid residual Ridge | 0.0406 | 0.0510 | 0.9848 | 0.3086 | -0.0158 |

## Honolulu, HI - temporal holdout - NOAA_COOPS_MOCK MOCK

- Station: `1612340` window 20240101-20240128
- Observations: `NOAA_COOPS_MOCK`; predictions: `NOAA_PREDICTIONS_MOCK`; mock_used=True
- Train: 2024-01-01 00:00:00+00:00 to 2024-01-21 05:54:00+00:00 (4,860 obs)
- Test: 2024-01-21 06:00:00+00:00 to 2024-01-28 00:00:00+00:00 (1,621 obs)
- Event threshold: train mean + 2 std = 0.8289 m

| Model | MAE | RMSE | R2/NSE | MAE skill vs rolling | MAE skill vs NOAA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rolling persistence | 0.0584 | 0.0727 | 0.9701 | 0.0000 | -0.4746 |
| NOAA prediction | 0.0396 | 0.0501 | 0.9858 | 0.3218 | 0.0000 |
| NOAA residual persistence | 0.0561 | 0.0703 | 0.9721 | 0.0395 | -0.4164 |
| HarmonicRidge | 0.0402 | 0.0508 | 0.9848 | 0.3120 | -0.0132 |
| GradBoost | 0.0462 | 0.0577 | 0.9803 | 0.2092 | -0.1645 |
| Hybrid residual Ridge | 0.0402 | 0.0508 | 0.9848 | 0.3119 | -0.0134 |

## Boston, MA - temporal holdout - NOAA_COOPS_MOCK MOCK

- Station: `8443970` window 20240101-20240128
- Observations: `NOAA_COOPS_MOCK`; predictions: `NOAA_PREDICTIONS_MOCK`; mock_used=True
- Train: 2024-01-01 00:00:00+00:00 to 2024-01-21 05:54:00+00:00 (4,860 obs)
- Test: 2024-01-21 06:00:00+00:00 to 2024-01-28 00:00:00+00:00 (1,621 obs)
- Event threshold: train mean + 2 std = 0.8343 m

| Model | MAE | RMSE | R2/NSE | MAE skill vs rolling | MAE skill vs NOAA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rolling persistence | 0.0585 | 0.0736 | 0.9696 | 0.0000 | -0.4567 |
| NOAA prediction | 0.0401 | 0.0500 | 0.9860 | 0.3135 | 0.0000 |
| NOAA residual persistence | 0.0564 | 0.0711 | 0.9716 | 0.0348 | -0.4061 |
| HarmonicRidge | 0.0404 | 0.0504 | 0.9852 | 0.3114 | -0.0040 |
| GradBoost | 0.0456 | 0.0572 | 0.9809 | 0.2230 | -0.1328 |
| Hybrid residual Ridge | 0.0411 | 0.0516 | 0.9845 | 0.2986 | -0.0226 |

## Virginia Key, FL - temporal holdout - NOAA_COOPS_MOCK MOCK

- Station: `8723214` window 20240101-20240128
- Observations: `NOAA_COOPS_MOCK`; predictions: `NOAA_PREDICTIONS_MOCK`; mock_used=True
- Train: 2024-01-01 00:00:00+00:00 to 2024-01-21 05:54:00+00:00 (4,860 obs)
- Test: 2024-01-21 06:00:00+00:00 to 2024-01-28 00:00:00+00:00 (1,621 obs)
- Event threshold: train mean + 2 std = 0.8302 m

| Model | MAE | RMSE | R2/NSE | MAE skill vs rolling | MAE skill vs NOAA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rolling persistence | 0.0572 | 0.0727 | 0.9699 | 0.0000 | -0.4135 |
| NOAA prediction | 0.0405 | 0.0505 | 0.9854 | 0.2926 | 0.0000 |
| NOAA residual persistence | 0.0553 | 0.0702 | 0.9719 | 0.0331 | -0.3667 |
| HarmonicRidge | 0.0408 | 0.0510 | 0.9845 | 0.2866 | -0.0080 |
| GradBoost | 0.0458 | 0.0577 | 0.9802 | 0.1994 | -0.1312 |
| Hybrid residual Ridge | 0.0410 | 0.0512 | 0.9844 | 0.2835 | -0.0123 |

## La Jolla, CA - temporal holdout - NOAA_COOPS_MOCK MOCK

- Station: `9410230` window 20240101-20240128
- Observations: `NOAA_COOPS_MOCK`; predictions: `NOAA_PREDICTIONS_MOCK`; mock_used=True
- Train: 2024-01-01 00:00:00+00:00 to 2024-01-21 05:54:00+00:00 (4,860 obs)
- Test: 2024-01-21 06:00:00+00:00 to 2024-01-28 00:00:00+00:00 (1,621 obs)
- Event threshold: train mean + 2 std = 0.8312 m

| Model | MAE | RMSE | R2/NSE | MAE skill vs rolling | MAE skill vs NOAA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rolling persistence | 0.0579 | 0.0724 | 0.9705 | 0.0000 | -0.4573 |
| NOAA prediction | 0.0397 | 0.0498 | 0.9861 | 0.3138 | 0.0000 |
| NOAA residual persistence | 0.0559 | 0.0699 | 0.9725 | 0.0336 | -0.4083 |
| HarmonicRidge | 0.0401 | 0.0502 | 0.9852 | 0.3085 | -0.0120 |
| GradBoost | 0.0455 | 0.0572 | 0.9808 | 0.2155 | -0.1481 |
| Hybrid residual Ridge | 0.0399 | 0.0501 | 0.9853 | 0.3123 | -0.0066 |

## Honolulu, HI (storm period) - event holdout - NOAA_COOPS_MOCK MOCK

- Station: `1612340` window 20240112-20240118
- Observations: `NOAA_COOPS_MOCK`; predictions: `NOAA_PREDICTIONS_MOCK`; mock_used=True
- Train: 2024-01-12 00:00:00+00:00 to 2024-01-16 11:54:00+00:00 (1,080 obs)
- Test: 2024-01-16 12:00:00+00:00 to 2024-01-18 00:00:00+00:00 (361 obs)
- Event threshold: train mean + 2 std = 0.8663 m

| Model | MAE | RMSE | R2/NSE | MAE skill vs rolling | MAE skill vs NOAA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rolling persistence | 0.0641 | 0.0785 | 0.9596 | 0.0000 | -0.5246 |
| NOAA prediction | 0.0420 | 0.0527 | 0.9818 | 0.3441 | 0.0000 |
| NOAA residual persistence | 0.0620 | 0.0765 | 0.9617 | 0.0334 | -0.4737 |
| HarmonicRidge | 0.0686 | 0.0872 | 0.9472 | -0.0533 | -0.6223 |
| GradBoost | 0.0525 | 0.0654 | 0.9704 | 0.1935 | -0.2421 |
| Hybrid residual Ridge | 0.0552 | 0.0693 | 0.9667 | 0.1521 | -0.3058 |

## Notes

- Mock reports are synthetic CI artifacts and must not be presented as real NOAA performance.
- Live reports hard-fail on mock records; mixed live/mock runs use the `noaa_allow_mock` filename.
- No model includes meteorological forcing, so residual surge skill is limited.
