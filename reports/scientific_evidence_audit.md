# Wai Scientific Evidence Audit

This report is a guardrail for portfolio claims. It does not add a new
performance result; it records which evidence is present and which gaps
remain open.

## Claim Boundary

- Operational NOAA proof established: `False`
- Allowed claim: Synthetic and mock reports are reproducibility and plumbing evidence only.
- Disallowed claim: Do not present synthetic or NOAA mock metrics as operational NOAA forecast performance.

## Live NOAA Evidence

- Status: `missing_live_noaa_metrics`
- Artifact: `reports/noaa_live_metrics.json`
- Verified live with no mock records: `False`
- Reason: No checked-in noaa_live_metrics.json artifact.

## Meteorological Forcing

- Status: `supported_not_validated`
- Checked-in reports use forcing: `False`
- Validated storm-surge skill: `False`
- Supported columns: `wind_speed_mps`, `wind_direction_deg`, `air_pressure_hpa`, `rainfall_mm`, `wave_height_m`
- Reason: The checked-in synthetic, tidecast, NOAA mock, and NOAA live artifacts do not include real meteorological covariates.

## Remaining Scientific Weaknesses

| Weakness | Status | Current control | Needed to close |
| --- | --- | --- | --- |
| Operational NOAA proof | not_established | Mock and live reports have separate filenames and mock flags. | Verified live NOAA metrics over representative windows. |
| Meteorological/storm-surge validation | partially_addressed | Feature schema accepts external forcing columns. | Real wind/pressure/wave/rain covariates and event validation. |
| Checked-in live NOAA artifact | open | missing_live_noaa_metrics | Run `python -m scripts.evaluate_noaa_public` with network access. |
