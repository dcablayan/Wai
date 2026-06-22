# Wai — Event-Holdout Evaluation

Threshold = train mean + 2.0σ (computed on the training window only). Episodes are contiguous test-period runs at or above this threshold. Predictions are matched to observations by the largest temporal overlap; positive lead-time error means the prediction was **late**, negative means **early**.

All metrics are computed on the held-out test span. The synthetic demo data places test-period events near days 80 (surge) and 85 (king tide) so that this report grades genuinely unseen extremes.

## DEMO-HNL

- Train cutoff: 2024-03-08 12:00:00+00:00
- Train threshold (mean+2.0σ): 0.493 m
- Test window: 2024-03-08 12:00:00+00:00 → 2024-03-30 23:54:00+00:00 (n_test=5,400)
- Observed episodes in test: 2

| Model | MAE (m) | RMSE (m) | Episode P | Episode R | F1 | Peak-h err (m) | Peak-t err (s) | Lead err (s) |
|-------|---------|----------|-----------|-----------|----|----------------|----------------|--------------|
| Persistence (rolling) | 0.0234 | 0.0294 | 1.0 | 1.0 | 1.0 | 0.0 | 360.0 | 360.0 |
| HarmonicRidge | 0.0177 | 0.0225 | 1.0 | 0.5 | 0.6667 | 0.043536 | 1800.0 | 1080.0 |
| GradBoost | 0.0223 | 0.0399 | nan | 0.0 | nan | nan | nan | nan |

## DEMO-SFO

- Train cutoff: 2024-03-08 12:00:00+00:00
- Train threshold (mean+2.0σ): 1.0317 m
- Test window: 2024-03-08 12:00:00+00:00 → 2024-03-30 23:54:00+00:00 (n_test=5,400)
- Observed episodes in test: 13

| Model | MAE (m) | RMSE (m) | Episode P | Episode R | F1 | Peak-h err (m) | Peak-t err (s) | Lead err (s) |
|-------|---------|----------|-----------|-----------|----|----------------|----------------|--------------|
| Persistence (rolling) | 0.0290 | 0.0361 | 0.7692 | 0.7692 | 0.7692 | 0.0 | 360.0 | 360.0 |
| HarmonicRidge | 0.0178 | 0.0226 | 1.0 | 0.7692 | 0.8696 | 0.036162 | 792.0 | 144.0 |
| GradBoost | 0.0215 | 0.0286 | 1.0 | 0.7692 | 0.8696 | 0.034056 | 900.0 | 108.0 |
