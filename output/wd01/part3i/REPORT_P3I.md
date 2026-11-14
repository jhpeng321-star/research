# WD-01 Part 3I — delay bias diagnosis + honest status report

**Status: PARTIAL — holdout NOT accessed; the full structured emulator
gate v2 evaluation was NOT completed in this session.**

## Diagnosis of the systematic delay under-prediction

### Root cause

The structured native Part A emulator (IDM + dawdle + sampled vType)
systematically under-predicts window delay because:

1. **Missing LC slow-down**: SUMO lane changes cause temporary speed
   reductions (both for the changer and for the affected lane) that the
   instantaneous LC (d kept constant, no deceleration) does not model.
   The p3c B2 IDM-normal also misses this (same speed RMSE as B3
   oracle), confirming it is not a vType issue.

2. **Missing merge waiting**: on the accel lane, real vehicles wait
   for gaps before merging; the model fires the LC hazard stochastically
   without gap-seeking behaviour, so vehicles merge into sub-optimal
   gaps that the IDM then handles without the cooperative deceleration
   SUMO applies.

3. **Missing downstream exit/queue dynamics**: vehicles exiting the
   corridor may queue in truth but exit freely in the model.

4. **Dawdle is one-sided** (always slows down), so it cannot generate
   the positive delay (speed-ups after waiting) that real traffic
   shows.  The negative dawdle bias compounds over the window.

### Evidence

From the p3c validation results (52 windows):
- B2 (deterministic IDM-normal) delay bias: -5.1% (H30), -8.8% (H60),
  -14.0% (H120) — the deterministic IDM already under-predicts delay.
- B3 (oracle vType) shows a similar bias, confirming this is not a
  vType identification issue.
- The structured emulator (L3) shows -4.0% to -9.5%, consistent with
  the same physics-level deficit.

### Conclusion

The delay under-prediction is a **physics-level deficit of the IDM
emulator**, not a parameter tuning issue.  Any IDM-based emulator (B2,
B3, L0-L3) will systematically under-predict delay because:
- SUMO lane changes cause deceleration not captured by IDM
- SUMO merge negotiation causes waiting not captured by stochastic LC
- SUMO junction traversal adds delay not captured by instantaneous
  crossing

The gate v2 (c) criterion (member band coverage ≥80% for delay proxy)
requires member diversity in delay predictions that physics-based
emulators sharing the same dynamics cannot produce.  The delay
coverage was 0% for all members in all windows.

## What was completed in p3c/3d/3e/3f/3g/3h/3i

- Holdout block: COMPLETE, tested
- SUMO IDM emulator + sigma finding: COMPLETE
- Dawdle hypothesis: CONFIRMED (fit tables in p3e)
- B2 posterior: trained, calibration documented (barely beats prior)
- D1-D5 diagnostics: COMPLETE with numbers
- Engine defect fixes: LC d-continuity, mask, attr refresh, limits,
  phase-1 full-state slots, 250 m presence cutoff, phase-2 CE removal
- LC hazard: trained (models.Hazard), integrated into idm_sampled
- LC scale calibration on TRAIN: 0.003
- Validation eval: L0/L1/L3 rollouts completed, per-window metrics
  computed

## What was NOT completed

- Delay bias fix (the bias is a physics-level deficit, not a
  parameter)
- Parameter uncertainty (bootstrap over train roots)
- Full M=5 × 52-window × 5-variant ladder eval
- Gate v2 evaluation at H=30 (per the new gate amendment)
- Robust posterior epsilon calibration

## Recommendation

The structured native Part A v0 cannot meet gate v2 because the IDM
physics do not capture the full SUMO dynamics.  To pass the gate, one
of the following is needed:
1. Add an empirical lane-change deceleration term to the IDM
   acceleration during LC events
2. Add a merge-waiting model for the accel lane
3. Use a data-driven approach (e.g. residual learning on delay
   features) rather than a physics-based emulator

These are substantive modelling changes beyond the scope of the
current engineering brief.
