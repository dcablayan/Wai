# Wai Conformal Coverage Report

Synthetic demo data only. Calibration is the last 15% of the training window; coverage is measured on the future test split.

| Station | Model | Nominal | Empirical | Event | Non-event | qhat | n_cal | Mean width |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DEMO-HNL | harmonic_ridge | 0.9000 | 0.8754 | 0.6515 | 0.8782 | 0.0340 | 2390 | 0.0679 |
| DEMO-HNL | grad_boost | 0.9000 | 0.8774 | 0.0152 | 0.8882 | 0.0386 | 2390 | 0.0772 |
| DEMO-SFO | harmonic_ridge | 0.9000 | 0.8905 | 0.7631 | 0.8977 | 0.0356 | 2390 | 0.0712 |
| DEMO-SFO | grad_boost | 0.9000 | 0.8944 | 0.8084 | 0.8993 | 0.0434 | 2390 | 0.0869 |
