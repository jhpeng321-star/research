# Learned physical parameters: development-only registration

Owner: S-RB2-LPHYS. Base FULL-v4 physics22cc32f6, repository HEAD03c81f91.
Fixed rule forms: SUMO IDM, LC2013, insertion checks and source accounting.
Corrections remain OFF. No SUMO execution, neural residuals or black-box retraining.

FIT/selection allowlist:969400–969431 only; the development panel is969420–969431.
Reserved969432–969439 excluded from every candidate-selection reader.
WMDATA2 quarantined by INT and excluded. Observable fields only.
All outcomes used for fitting/selection will be listed in an exposure receipt.

Coordinate ownership: see `E:/agent-work/rb-shared/RB2_LPHYS_OWNERSHIP.md`.
FIT/LANE own per-driver IDM/LC population/posteriors. Do not fit them twice.
Start with documented departure parameters and retain exact safety/queue rules.
Use gradient-free Bayesian optimisation, paired futures/common random numbers,
and distributional losses for delay,10s speeds, breakdown/onset, lane counts,
ramp delay and expected controller shifts. Freeze the objective before search.
Report development comparisons as selection evidence; no final-panel claims.
