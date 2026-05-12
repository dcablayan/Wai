# Wai Benchmark Results

Models from `src/models/prototypes.py` (ported from dcablayan/tideformer).
Evaluated on tidecast tidal-prediction data (NOAA-derived, Hawaiian stations).
Lookback: 24 steps (144 min at 6-min cadence) · Max windows: 2000

| Station | ʻAle Iki (Ripple) RMSE | Nalu Hoʻokani (Harmonic Wave) RMSE | Nalu Holo (Fast Wave) RMSE | ʻAle Piʻi (Rising Wave) RMSE |
| --- | --- | --- | --- | --- |
| hohonu-12_tidecast | 0.493 | 4.772 | 0.682 | 4.715 |
| hohonu-142_tidecast | 0.169 | 6.160 | 0.245 | 5.654 |
| hohonu-160_tidecast | 0.101 | 9.372 | 1.188 | 8.478 |
| hohonu-168_tidecast | 0.127 | 1.342 | 0.296 | 1.246 |
| hohonu-16_tidecast | 0.210 | 6.378 | 1.156 | 7.144 |
| hohonu-186_tidecast | 0.609 | 6.143 | 2.265 | 5.454 |
| hohonu-191_tidecast | 0.267 | 16.605 | 1.799 | 14.894 |
| hohonu-207_tidecast | 0.015 | 2.708 | 0.094 | 2.854 |
| hohonu-37_tidecast | 0.083 | 11.037 | 0.966 | 9.702 |
| hohonu-74_tidecast | 0.151 | 3.317 | 0.416 | 2.849 |

**Averages**
| Model | Mean RMSE |
| --- | --- |
| ʻAle Iki (Ripple) | 0.222 |
| Nalu Hoʻokani (Harmonic Wave) | 6.783 |
| Nalu Holo (Fast Wave) | 0.911 |
| ʻAle Piʻi (Rising Wave) | 6.299 |
