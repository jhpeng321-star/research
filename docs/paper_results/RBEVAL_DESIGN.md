# S-RB-EVAL overnight distributional pilot — frozen before scoring

Owner authority: WM-PHYS REBUILD PROGRAM, central REGISTRATION.md 2026-10-31
23:12, plus task fixed-panel instruction. Base c4/wmfixmerge f0db3c82.

## Frozen panel and inference contract
- WMDATA seeds **969420–969439 inclusive** (20 worlds), t0=300 s.
- Report development-pilot 969420–969431 and reserved8 969432–969439 separately.
- Arms: native, ctrl0, mappo, pi, circles. K=8 futures per world, paired random
  streams across arms. Horizons 120/300/600 s from the same 600 s open-loop run.
- Evaluate all observed demand×penetration cells separately against the frozen
  12-cell family; absent cells are missing evidence, never successful tests.
- First model WM2 checkpoint SHA256 b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436,
  existing registered config, learned entrant mix, native lane ownership, frozen
  CTRL0/MAPPO/PI/CIRCLES executors. Native uses the existing WM native CAV law.
  Each candidate must bind checkpoint/config/code/actions/entry-driver sampling
  and source start identities in an immutable run identity before inference.
- Motion-only driver inference and recorded pre-t0 histories/arrival counters;
  population defaults allowed. No future trajectory in any rollout input.
  No training, simulator dispatch by this harness, or prohibited data access.
- Durable per-(world,future) files; exclusive run lock, skip only identity-verified
  complete files. No reuse of old cost-only K8 files as trajectory predictions.

## Observables and scoring
- Full present traffic, not t0 survivors: 10 s half-open pooled vehicle-time
  mean speeds and support counts for upstream (mainline s<0), merge (mainline
  0<=s<=75.44 m), ramp (lane 2). Valid mapped lanes 0/1/2 only.
- Exact recorded 60 s prehistory (240<=t<300), then each predicted 600 s field;
  endpoint t=900 excluded. Empty bins remain null. Registered LRDIAG breakdown
  function is AST-extracted unchanged and SHA-bound, with separate decisions at
  each horizon. Pre-existing and unscorable mass reported separately. Censored
  futures have no invented onset. BIC is a speed-loss proxy, not a phase proof.
- Cumulative team/road/source delay uses reported-boundary real_J convention;
  inherited training J is diagnostic. Source and road are always retained.
- Rates per cell/arm/horizon with pointwise and simultaneous Wilson CIs;
  primary upstream Bonferroni family 12×5×3=180, merge/ramp diagnostic family 360.
- Onset CDF on 10 s grid plus separate pre-existing/censored categories;
  integrated Brier score against cell×arm onset climatology.
- Empirical and fair-K CRPS for delay and each speed bin, versus pooled and
  cell×arm/time-bin empirical climatology; speed support/missingness and joint
  speed-field energy score diagnostic. Equal weight for supported time bins.
- Baseline pool WMDATA development 969400–969431 only. For development targets
  leave the entire target world out across all arms; reserved labels never enter
  a baseline. Empty cell baseline = unavailable, no fallback that hides failure.
- Randomized rank PIT (ties randomized), K-range coverage (exchangeable K8
  expectation 7/9); whole-world simulation bands account for clustered arms.
- Controller−native and controller−CTRL0 expected delay/event shifts, real paired
  simultaneous CIs, resolved-sign agreement and within-cell arm-order concordance.
- Whole-world bootstrap B=10,000, seed=20261031; max-stat simultaneous correction
  within each frozen metric family, preserving arms/horizons/regions within world.
  Singleton/empty cells cannot supply uncertainty or qualification.

## Verdict and scope
Implement the S-WMBREAKDOWN proposed gate verbatim: simultaneous rate half-width
<=10 pp, compatible paired rate differences, timing bands and superior IBS,
superior delay/speed CRPS at every primary horizon/cell, calibrated PIT/range,
and resolved expected shifts/order lower CI>0.5. Missing precision or evidence
prevents PASS. This 20-world K8 panel is an **overnight comparative diagnostic**,
not a powered qualification panel or historically untouched test. Previously
reported WM2 outcomes and WMCALIB exposure are disclosed; no tuning on this pilot.
No optional stopping or selecting K/horizons by favorable outcomes. K increase
requires an explicit new registration before additional scoring.
