# Wai Benchmark Results

Models from `src/models/prototypes.py` (ported from dcablayan/tideformer).
Evaluated on tidecast tidal-prediction data (NOAA-derived, Hawaiian stations).
Lookback: 24 steps (144 min at 6-min cadence) · Max windows: 2000

| Station | ʻAle Iki (Ripple) RMSE | Nalu Hoʻokani (Harmonic Wave) RMSE | Nalu Holo (Fast Wave) RMSE | ʻAle Piʻi (Rising Wave) RMSE |
| --- | --- | --- | --- | --- |
| hohonu-12_tidecast | 0.493 | 4.248 | 0.682 | 4.578 |
| hohonu-142_tidecast | 0.169 | 3.652 | 0.245 | 3.384 |
| hohonu-160_tidecast | 0.101 | 6.274 | 1.188 | 5.326 |
| hohonu-168_tidecast | 0.127 | 2.347 | 0.296 | 2.088 |
| hohonu-16_tidecast | 0.210 | 9.597 | 1.156 | 8.623 |
| hohonu-186_tidecast | 0.609 | 26.015 | 2.265 | 23.729 |
| hohonu-191_tidecast | 0.267 | 13.718 | 1.799 | 12.314 |
| hohonu-207_tidecast | 0.015 | 4.302 | 0.094 | 4.795 |
| hohonu-37_tidecast | 0.083 | 6.178 | 0.966 | 5.669 |
| hohonu-74_tidecast | 0.151 | 9.950 | 0.416 | 9.193 |

**Averages**
| Model | Mean RMSE |
| --- | --- |
| ʻAle Iki (Ripple) | 0.222 |
| Nalu Hoʻokani (Harmonic Wave) | 8.628 |
| Nalu Holo (Fast Wave) | 0.911 |
| ʻAle Piʻi (Rising Wave) | 7.970 |
