# Wai — Multi-Horizon Forecast Evaluation

Metrics computed on the held-out test split (last 25% by time).
Strategy: **direct forecasting** — a separate model is trained per horizon.
WaveGRU is a 1-step model and is only evaluated at horizon 1 (6 min).

## Station: DEMO-HNL

| Horizon | Model | MAE (m) | RMSE (m) | R² |
|---------|-------|---------|----------|----|
| 1step_6min | persistence | 0.0233 | 0.0293 | 0.9830 |
| 1step_6min | harmonic_ridge | 0.0170 | 0.0212 | 0.9911 |
| 1step_6min | grad_boost | 0.0195 | 0.0244 | 0.9883 |
| 1step_6min | wave_gru | 0.0776 | 0.0974 | 0.8124 |
| 6h | persistence | 0.2784 | 0.3598 | -1.5882 |
| 6h | harmonic_ridge | 0.0184 | 0.0230 | 0.9894 |
| 6h | grad_boost | 0.0338 | 0.0451 | 0.9593 |
| 6h | wave_gru | — | — | WaveGRU is a 1-step model; not evaluated at this horizon. |
| 12h | persistence | 0.2965 | 0.3680 | -1.7624 |
| 12h | harmonic_ridge | 0.0182 | 0.0228 | 0.9894 |
| 12h | grad_boost | 0.0334 | 0.0454 | 0.9579 |
| 12h | wave_gru | — | — | WaveGRU is a 1-step model; not evaluated at this horizon. |
| 24h | persistence | 0.0681 | 0.0833 | 0.8582 |
| 24h | harmonic_ridge | 0.0178 | 0.0223 | 0.9898 |
| 24h | grad_boost | 0.0210 | 0.0268 | 0.9854 |
| 24h | wave_gru | — | — | WaveGRU is a 1-step model; not evaluated at this horizon. |

## Station: DEMO-SFO

| Horizon | Model | MAE (m) | RMSE (m) | R² |
|---------|-------|---------|----------|----|
| 1step_6min | persistence | 0.0289 | 0.0360 | 0.9950 |
| 1step_6min | harmonic_ridge | 0.0168 | 0.0211 | 0.9983 |
| 1step_6min | grad_boost | 0.0215 | 0.0270 | 0.9972 |
| 1step_6min | wave_gru | 0.2224 | 0.2593 | 0.7407 |
| 6h | persistence | 0.7731 | 0.9174 | -2.2654 |
| 6h | harmonic_ridge | 0.0179 | 0.0224 | 0.9981 |
| 6h | grad_boost | 0.0486 | 0.0631 | 0.9846 |
| 6h | wave_gru | — | — | WaveGRU is a 1-step model; not evaluated at this horizon. |
| 12h | persistence | 0.4896 | 0.6115 | -0.5038 |
| 12h | harmonic_ridge | 0.0181 | 0.0227 | 0.9979 |
| 12h | grad_boost | 0.0422 | 0.0534 | 0.9885 |
| 12h | wave_gru | — | — | WaveGRU is a 1-step model; not evaluated at this horizon. |
| 24h | persistence | 0.1626 | 0.1923 | 0.8529 |
| 24h | harmonic_ridge | 0.0186 | 0.0232 | 0.9979 |
| 24h | grad_boost | 0.0295 | 0.0382 | 0.9942 |
| 24h | wave_gru | — | — | WaveGRU is a 1-step model; not evaluated at this horizon. |

## Notes

- All metrics are on **synthetic demo data** and cannot be compared to
  published operational benchmarks.
- Direct forecasting trains a separate model for each horizon. This is
  an honest skill assessment but may differ from iterated/recursive approaches.
- Lag features at long horizons (6h, 12h, 24h) reference observations prior
  to the prediction time — no look-ahead bias is introduced.
- Advanced deep learning (LSTM, Transformer) is intentionally excluded to
  keep the repo lightweight and honest.
