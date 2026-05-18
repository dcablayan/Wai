# Wai Results Summary

Small summary table for portfolio review. Lower error is better.

| Baseline | Hybrid model | Horizon | Improvement direction | Evidence track | Caveat |
| --- | --- | --- | --- | --- | --- |
| Rolling persistence, 0.0262 m avg MAE | HarmonicRidge, 0.0179 m avg MAE | 1-step / 6 min | Improved, 31.6% lower MAE | Synthetic direct holdout | Synthetic sanity check only |
| Rolling persistence, 0.5329 m avg MAE | HarmonicRidge, 0.0280 m avg MAE | 6 h | Improved, 94.7% lower MAE | Synthetic direct holdout | Direct horizon model on synthetic data |
| Rolling persistence, 0.4050 m avg MAE | HarmonicRidge, 0.0297 m avg MAE | 12 h | Improved, 92.7% lower MAE | Synthetic direct holdout | Direct horizon model on synthetic data |
| Rolling persistence, 0.1316 m avg MAE | HarmonicRidge, 0.0299 m avg MAE | 24 h | Improved, 77.2% lower MAE | Synthetic direct holdout | Persistence is phase-sensitive at this horizon |
| Rolling persistence | HarmonicRidge | 1-step rolling-origin folds | Improved in all 6 folds | Synthetic rolling-origin | Only 2 synthetic stations |
| Harmonics-only features | Full tidal + lag + rolling features | 1-step | Improved MAE on both stations | Synthetic ablation | Does not prove real-station generalization |
| NOAA tidal prediction, 0.0404 avg MAE | Hybrid residual Ridge, 0.0430 avg MAE | 1-step mock holdout | Did not improve; worse than NOAA prediction | NOAA mock eval | Mock fixture, not live NOAA proof |
| Tidecast last-value persistence, 0.222 avg RMSE | TinyTide, 0.222 avg RMSE | 1-step tidecast | Tied, no clear improvement | Tidecast prototype benchmark | Smooth NOAA-derived predictions, not observations |
| 90% nominal conformal interval | HarmonicRidge split-conformal | 1-step | Overall coverage slightly below nominal | Synthetic conformal | Event coverage degraded, especially on `DEMO-HNL` |
