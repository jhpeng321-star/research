# IDM observation-v2 Phase-A screen results

Status: **executed development diagnosis; scale-up no-go because the frozen
v5.3 Stage-1 treatment-integrity gate failed. No predictor, thresholds,
probabilities, calibration, or Phase-B data generation are authorized.**

Date: 2026-08-30

## Controlling protocol delta

The owner-approved change is recorded verbatim:

> The requirement changes from ">=90% safe among ALL raw feasible
> activations" to ">=90% safe among GATE-SELECTED activations", with the
> full-horizon hard-brake outcome retained as a label for the conditional
> Gate-B model. Paper claim becomes: "a two-stage RSU gate selectively
> escalates to a high-authority wall when physical delivery, safety, and
> conditional benefit are jointly supported."

The treatment remains damping plus a wall only during its declared override;
the matched arm remains frozen analytical IDM damping. Version 5.3 adds
pre-latched same-lane reserves with at most one handoff per lane, compactness
enforcement with no partial-lane HOLD, and a fixed five-second in-horizon
linear taper to the identical damping command. These are treatment-integrity
repairs. The taper is not a claimed repair for delayed downstream safety harm.

## v5.3 implementation and bounded Stage-1 replay

The frozen profile is `adaptive_compact_wall_v5_3_integrity`, contract
`idm_compact_moving_wall/5.3`. The controller implements
`DAMPING -> ASSEMBLING -> HOLDING -> TAPERING -> RELEASED`, ranked same-lane
reserves, one handoff per lane, two-sample compactness enforcement, and blend
fractions `0.8, 0.6, 0.4, 0.2, 0.0`. Safety projection remains downstream of
the blend, and zero blend returns to analytical damping rather than
uncontrolled SUMO.

The seven prescribed development cases were replayed; no new seed cohort was
opened. Positive benefit means beyond the frozen replay/noise deadband.

| Seed / decision | v5.3 dose | Safety | Benefit | Integrity behavior | Comparison with v5.2 |
|---|---:|---:|---:|---|---|
| 4301 / 660 | fail | pass | yes | one same-lane handoff, then compactness-loss taper | hard-brake and low-TTC totals did not worsen |
| 4302 / 620 | pass | pass | yes | planned five-step taper | hard-brake and low-TTC totals did not worsen |
| 4307 / 680 | pass | fail | yes | planned five-step taper | safety totals unchanged, but the frozen budget still fails |
| 4308 / 680 | pass | fail | no | planned five-step taper | hard braking improved but remains outside budget |
| 4309 / 620 | pass | fail | no | planned five-step taper | safety totals unchanged and outside budget |
| 4311 / 700 | pass | fail | yes | planned five-step taper | safety totals unchanged, but the frozen budget still fails |
| 4312 / 660 | fail | pass | no | compactness-loss taper | absolute hard braking worsened by one vehicle-step; paired excess changed from -7 to -6 |

Integrity results:

- exact branch-prefix identity: pass;
- exact treatment-repeat identity on 4301/660 and 4302/620: pass;
- no partial-lane HOLD: pass;
- taper reaches zero and returns to damping: pass;
- safety not worse than matched v5.2 in every case: **fail** because of
  4312/660;
- at least two of three prior safe-beneficial cases retained: **fail**.

There is a protocol/cohort inconsistency in the last gate. The three
safe-beneficial completed v5.2 cases are 4302/700, 4307/660, and 4311/660, but
the prescribed seven-case list contains different decision starts for those
seeds. Within the prescribed list, 4307/680 and 4311/700 were beneficial but
already unsafe under v5.2, while 4312/660 was the only safe-beneficial start
and was not dose-qualified. The runner did not substitute cases after reading
outcomes. Even without that ambiguity, the explicit per-case safety
non-worsening gate fails, so Stage 1 is a no-go.

Machine-readable evidence is in
`runs/probe/idm_wall_v5_3_stage1_replay/summary.json`.

## Observation-v2 contract

Observation v2 adds 27 strict predecision fields to the preserved v1 contract,
for 101 expanded fields and 56 primary fields. Every new feature uses only the
common control prefix through the declared decision, windows are measured in
simulation seconds, and future-sample count is zero.

Post-merge RSU state and causal history:

- vehicle count, density per lane-km, occupancy proxy;
- arithmetic and harmonic speed, speed variance;
- hard-brake count and low-TTC pair count;
- 20-second and 60-second discharge rates;
- 60-second density, speed, and speed-variance slopes;
- 60-second hard-brake and low-TTC active fractions;
- short-minus-long discharge rate.

Pre-merge early-warning signals:

- 20-second speed variance;
- speed lag-1 autocorrelation over 20 and 60 seconds;
- 20-second same-lane gap availability and variance;
- gap lag-1 autocorrelation over 20 and 60 seconds.

Decision-time dose context:

- proposed wall speed;
- selected-member margin to the release boundary;
- 60-second upstream flow;
- upstream-flow by wall-speed product.

Forbidden fields remain seed, arm identity, postdecision state, realized dose,
treatment phase, and outcome. The exact schema and per-feature provenance are
in `counterfactual/idm_predictor_contract.py`.

## Exact-prefix recapture

All ten completed v5.2 IDM cases were recaptured from their control prefixes.
Every preserved v1 feature matched its stored value, all 27 observation-v2
features were produced, and the capture created zero treatment branches and
zero new outcomes. Result: 10/10 cases across seven seeds pass exact identity.

Evidence:
`runs/probe/idm_observation_v2_capture_v5_2_completed/summary.json`.

## Existing-artifact descriptive screen

This screen computed standardized mean differences and grouped
leave-one-seed-out sign stability only. It did not fit a classifier, choose a
threshold, estimate probabilities, or calculate deployable policy value.

Artifact accounting:

- Gate F: 10 dose-qualified IDM starts across seven seeds and 38 failed-dose
  starts across 12 seeds;
- Gate B joint label: 3 positive starts across three seeds and 7 negative
  starts across seven seeds among the ten dose-qualified IDM cases;
- safety alone: 5 safe and 5 unsafe cases, each side spanning five seeds;
- meaningful benefit alone: 5 positive cases across three seeds and 5
  nonbeneficial cases across five seeds;
- the seven diagnosis cases are references to existing starts and add no
  duplicate rows;
- the legacy v2 headroom artifact contains 36 Krauss cases, 16 engaged across
  ten seeds. It is screened separately and is not pooled with IDM because the
  HDV model, damping checkpoint, wall treatment, dose contract, and feature
  schema differ.

All descriptive headroom checks pass: both classes span at least three seeds,
no duplicate starts or forbidden feature names were found, grouped folds keep
seeds intact, exact observation recapture passes, and realized effects exceed
replay noise.

The strongest stable observation-v2 joint-label differences in this ten-case
IDM sample are:

- post-merge speed-variance slope: standardized difference +1.502, grouped
  leave-one-seed-out range +1.419 to +1.687;
- current post-merge speed variance: +1.362, range +1.019 to +1.808;
- post-merge density slope: -1.302, range -1.515 to -1.208;
- short-minus-long post-merge discharge: -1.174, range -1.368 to -1.069.

For safety alone, post-merge speed-variance slope has standardized difference
+1.200 with grouped leave-one-seed-out range +0.977 to +1.432. These are
descriptive signals from a very small opened development sample, not estimated
generalization or calibrated treatment-effect prediction.

Machine-readable evidence is in
`runs/probe/idm_observation_v2_screen/screen.json`.

## Decision

**No-go for scale-up in this Phase-A state.** Observation v2 has plausible,
strictly causal descriptive headroom, but the frozen v5.3 Stage-1 treatment
gate fails. Per the approved order, an observation signal cannot authorize a
predictor around a treatment version that did not pass its integrity/safety
replay. `predictor_authorized=false` and `phase_b_authorized=false` remain
locked.

The next scientific decision must be made before further execution: either
declare v5.3 a mechanism no-go and retain v5.2 as negative treatment evidence,
or approve a separately versioned treatment/case-contract correction. This
report does not authorize a treatment-blind feasibility scan, new seeds,
fitting, threshold selection, or predictor-scale generation.

## Verification

- Focused Phase-A tests: 61 passed.
- Full repository suite: **812 passed, 0 failed** in 65.74 seconds.
