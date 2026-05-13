# Wai — Multi-Horizon Forecast Evaluation

Metrics computed on the held-out test split (last 25% by time).
Strategy: **direct forecasting** — a separate model is trained per horizon.
WaveGRU is a 1-step model and is only evaluated at horizon 1 (6 min).

## Station: DEMO-HNL

_train cutoff: 2024-03-08 12:00:00+00:00 (n_train=16200, n_total=21600)._

| Horizon | Model | MAE (m) | RMSE (m) | R² | n_train | n_test |
|---------|-------|---------|----------|----|---------|--------|
| 1step_6min | persistence | 0.0234 | 0.0294 | 0.9847 | 16159 | 5399 |
| 1step_6min | harmonic_ridge | 0.0179 | 0.0229 | 0.9908 | 16159 | 5399 |
| 1step_6min | grad_boost | 0.0231 | 0.0412 | 0.9699 | 16159 | 5399 |
| 1step_6min | wave_gru | 0.0795 | 0.1004 | 0.8214 | 16159 | 5399 |
| 6h | persistence | 0.2876 | 0.3723 | -1.4801 | 16100 | 5340 |
| 6h | harmonic_ridge | 0.0282 | 0.0569 | 0.9422 | 16100 | 5340 |
| 6h | grad_boost | 0.0457 | 0.0808 | 0.8831 | 16100 | 5340 |
| 6h | wave_gru | — | — | WaveGRU is a 1-step model; not evaluated at this horizon. | 16100 | 5340 |
| 12h | persistence | 0.3064 | 0.3862 | -1.7178 | 16040 | 5280 |
| 12h | harmonic_ridge | 0.0297 | 0.0664 | 0.9198 | 16040 | 5280 |
| 12h | grad_boost | 0.0448 | 0.0789 | 0.8865 | 16040 | 5280 |
| 12h | wave_gru | — | — | WaveGRU is a 1-step model; not evaluated at this horizon. | 16040 | 5280 |
| 24h | persistence | 0.0871 | 0.1228 | 0.7262 | 15920 | 5160 |
| 24h | harmonic_ridge | 0.0295 | 0.0668 | 0.9189 | 15920 | 5160 |
| 24h | grad_boost | 0.0358 | 0.0744 | 0.8996 | 15920 | 5160 |
| 24h | wave_gru | — | — | WaveGRU is a 1-step model; not evaluated at this horizon. | 15920 | 5160 |

## Station: DEMO-SFO

_train cutoff: 2024-03-08 12:00:00+00:00 (n_train=16200, n_total=21600)._

| Horizon | Model | MAE (m) | RMSE (m) | R² | n_train | n_test |
|---------|-------|---------|----------|----|---------|--------|
| 1step_6min | persistence | 0.0290 | 0.0361 | 0.9952 | 16159 | 5399 |
| 1step_6min | harmonic_ridge | 0.0179 | 0.0229 | 0.9980 | 16159 | 5399 |
| 1step_6min | grad_boost | 0.0224 | 0.0297 | 0.9967 | 16159 | 5399 |
| 1step_6min | wave_gru | 0.2234 | 0.2611 | 0.7460 | 16159 | 5399 |
| 6h | persistence | 0.7783 | 0.9258 | -2.2128 | 16100 | 5340 |
| 6h | harmonic_ridge | 0.0277 | 0.0576 | 0.9876 | 16100 | 5340 |
| 6h | grad_boost | 0.0587 | 0.0919 | 0.9683 | 16100 | 5340 |
| 6h | wave_gru | — | — | WaveGRU is a 1-step model; not evaluated at this horizon. | 16100 | 5340 |
| 12h | persistence | 0.5037 | 0.6329 | -0.5523 | 16040 | 5280 |
| 12h | harmonic_ridge | 0.0297 | 0.0666 | 0.9828 | 16040 | 5280 |
| 12h | grad_boost | 0.0536 | 0.0843 | 0.9725 | 16040 | 5280 |
| 12h | wave_gru | — | — | WaveGRU is a 1-step model; not evaluated at this horizon. | 16040 | 5280 |
| 24h | persistence | 0.1761 | 0.2136 | 0.8251 | 15920 | 5160 |
| 24h | harmonic_ridge | 0.0304 | 0.0679 | 0.9823 | 15920 | 5160 |
| 24h | grad_boost | 0.0487 | 0.0902 | 0.9688 | 15920 | 5160 |
| 24h | wave_gru | — | — | WaveGRU is a 1-step model; not evaluated at this horizon. | 15920 | 5160 |

## Notes

- All metrics are on **synthetic demo data** and cannot be compared to
  published operational benchmarks.
- Direct forecasting trains a separate model for each horizon. This is
  an honest skill assessment but may differ from iterated/recursive approaches.
- Lag features at long horizons (6h, 12h, 24h) reference observations prior
  to the prediction time — no look-ahead bias is introduced.
- **Boundary exclusion:** the split mask uses `target_idx = X.index + h`
  for training and `X.index >= n_train` for test. Rows whose target
  crosses the train/test boundary are dropped from both sets so that no
  training row sees a test-period label.
- Advanced deep learning (LSTM, Transformer) is intentionally excluded to
  keep the repo lightweight and honest.
