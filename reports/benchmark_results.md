# Wai Benchmark Results

Models from `src/models/prototypes.py` (ported from dcablayan/tideformer).
Evaluated on tidecast tidal-prediction data (NOAA-derived, Hawaiian stations).
Lookback: 24 steps (144 min at 6-min cadence) · Max windows: 2000

| Station | TinyTide RMSE | HarmonicNet RMSE | WaveGRU RMSE | SurgeNet RMSE |
| --- | --- | --- | --- | --- |
| hohonu-12_tidecast | 0.493 | 11.064 | 0.682 | 10.117 |
| hohonu-142_tidecast | 0.169 | 2.258 | 0.245 | 2.045 |
| hohonu-160_tidecast | 0.101 | 22.224 | 1.188 | 20.160 |
| hohonu-168_tidecast | 0.127 | 0.857 | 0.296 | 0.797 |
| hohonu-16_tidecast | 0.210 | 9.175 | 1.156 | 8.107 |
| hohonu-186_tidecast | 0.609 | 21.838 | 2.265 | 20.277 |
| hohonu-191_tidecast | 0.267 | 13.148 | 1.799 | 12.280 |
| hohonu-207_tidecast | 0.015 | 2.351 | 0.094 | 2.305 |
| hohonu-37_tidecast | 0.083 | 7.738 | 0.966 | 6.722 |
| hohonu-74_tidecast | 0.151 | 3.240 | 0.416 | 3.007 |

**Averages**
| Model | Mean RMSE |
| --- | --- |
| TinyTide | 0.222 |
| HarmonicNet | 9.389 |
| WaveGRU | 0.911 |
| SurgeNet | 8.582 |
