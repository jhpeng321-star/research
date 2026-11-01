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

## Frozen first screen / autonomous comparison (before outcome inference)
Three global probabilities select only SUMO `departLane`free/first and
`departSpeed`max/last; identity-keyed samples are persistent across retries/actions.
GP expected-improvement search uses57evaluations on1068observable basePos births
in240..300s native FIT32. Existing joint IDM population stays fixed. Six observable
files per native tape only. Later native development12 conditional births are a
separately labelled diagnostic, not a free-running qualification.
Autonomous comparison: all12development worlds969420..969431, five registered
arms,600s, first2of the SAME frozenK8 futures (not relabelled asK8).
Compare off vs learned_v1 with identical driver/arrival/controller streams.
Report H120/300/600 delay CRPS/bias, upstream rate distance/onset IBS,10s speed
CRPS, lane counts/timing, ramp-lane road loss plus ramp source wait separately,
and controller-minus-native expected signs. No reserved or WMDATA2 reads.
Keep candidate experimental if delay CRPS fails to improve or300s upstream
rate-distance worsens. Cell singleton CIs remain NA. These are development
screening criteria, not the integrator's registered distributional gate.
