# Wai Benchmark Results

Models from `src/models/prototypes.py` (ported from dcablayan/tideformer) plus a last-value persistence comparator.
Evaluated on tidecast tidal-prediction data (NOAA-derived, Hawaiian stations).
WaveGRUPrototype is a smoothing heuristic, not a real GRU. SurgeNetPrototype is a residual heuristic, not meteorological surge modeling.
Lookback: 24 steps (144 min at 6-min cadence) · Max windows: 2000

| Station | Persistence (last value) RMSE | ʻAle Iki (Ripple) RMSE | Nalu Hoʻokani (Harmonic Wave) RMSE | Nalu Holo (Fast Wave) RMSE | ʻAle Piʻi (Rising Wave) RMSE |
| --- | --- | --- | --- | --- | --- |
| hohonu-12_tidecast | 0.493 | 0.493 | 5.754 | 0.682 | 6.607 |
| hohonu-142_tidecast | 0.169 | 0.169 | 2.147 | 0.245 | 1.956 |
| hohonu-160_tidecast | 0.101 | 0.101 | 3.929 | 1.188 | 4.905 |
| hohonu-168_tidecast | 0.127 | 0.127 | 1.668 | 0.296 | 2.141 |
| hohonu-16_tidecast | 0.210 | 0.210 | 3.965 | 1.156 | 4.957 |
| hohonu-186_tidecast | 0.609 | 0.609 | 9.339 | 2.265 | 11.314 |
| hohonu-191_tidecast | 0.267 | 0.267 | 10.351 | 1.799 | 11.519 |
| hohonu-207_tidecast | 0.015 | 0.015 | 3.506 | 0.094 | 3.542 |
| hohonu-37_tidecast | 0.083 | 0.083 | 4.279 | 0.966 | 5.407 |
| hohonu-74_tidecast | 0.151 | 0.151 | 4.116 | 0.416 | 3.611 |

**Averages**
| Model | Mean RMSE |
| --- | --- |
| Persistence (last value) | 0.222 |
| ʻAle Iki (Ripple) | 0.222 |
| Nalu Hoʻokani (Harmonic Wave) | 4.905 |
| Nalu Holo (Fast Wave) | 0.911 |
| ʻAle Piʻi (Rising Wave) | 5.596 |
