# Wai Conformal Coverage Report

Synthetic demo data only. Calibration is the last 15% of the training window; coverage is measured on the future test split.

| Station | Model | Nominal | Empirical | Event | Non-event | qhat | n_cal | Mean width |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DEMO-HNL | harmonic_ridge | 0.9000 | 0.8754 | 0.6515 | 0.8782 | 0.0340 | 2390 | 0.0679 |
| DEMO-HNL | grad_boost | 0.9000 | 0.8823 | 0.0455 | 0.8927 | 0.0392 | 2390 | 0.0783 |
| DEMO-SFO | harmonic_ridge | 0.9000 | 0.8905 | 0.7631 | 0.8977 | 0.0356 | 2390 | 0.0712 |
| DEMO-SFO | grad_boost | 0.9000 | 0.8972 | 0.8084 | 0.9022 | 0.0437 | 2390 | 0.0874 |
