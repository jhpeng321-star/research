# S-RB2-CAL v1: development-only physical uncertainty calibration

Frozen before fitting, 2026-11-01. Base FULL-v4 `22cc32f6`.

## Admission and exposure

- Fit inputs: native motion and causal request counters from FIT 969400–969431
  only. Development outcome diagnostics: 969420–969431 only, all five arms.
- Never open reserved raw prediction tasks, assignment/type/parameter truth or
  prohibited paths. The owner-required morning reports contain aggregate reserved
  results; reading those is disclosed exposure, not independent test evidence.
- All 32 FIT worlds are already exposed to population fitting. Scalar
  leave-world-out diagnostics reuse that fixed population and are internal
  empirical-Bayes checks, not independent validation or final gate evidence.

## Learned quantities (no neural training, no trajectory residual)

1. Likelihood power in [0,1], retaining FIT's learned shrinkage strength. Learn
   by physical acceleration-profile CRPS: motion 240–270 predicts 270–300 using
   the actual deployed SUMO CF step and lane-chain geometry. Zero is population;
   one is the frozen likelihood. Driver settings remain joint support draws.
2. Gamma-rate variance multiplier >=0, holding conditional mean fixed. Learn
   by negative-binomial log score: causal request counts 0–150 predict 150–300.
   The zero endpoint is a point rate; one reproduces Jeffreys Gamma-Poisson.
3. Entrant mixture concentration >=0, including the fixed-population infinity
   endpoint. Learn by CRPS predicting a driver's 270–300 motion from OTHER
   drivers' prefix-motion posterior counts in the same world. One future-owned
   Dirichlet mixture is shared by all entrant draws, preserving ordinal coupling.
   No new independent-coordinate driver perturbation or per-step noise.

Use compact one-dimensional bounded optimization with explicit endpoints. Freeze
full-data parameters; report whole-world excluded scalar fits. Do not select on
reserved8. Population support/weights and learned corrections stay frozen/OFF.

## Outcome diagnostic and delivery

Reuse S-RB-DECIDE metrics for development absolute/paired-delay CRPS,
50/80/90% coverage, strict preference Brier and selected 0.8-call reliability.
An outcome-spread-only diagnostic may establish limitations but must never be
shipped as fake simulated futures. Physics candidates require fresh rollouts;
input calibration alone does not establish outcome coverage or reliable calls.
Integrator owns fresh recorded-world exposure audit, replicated strata, valid
CIs and final fixed-source comparison. Versioned artifacts and source hashes
are published in HANDOFF.md.
