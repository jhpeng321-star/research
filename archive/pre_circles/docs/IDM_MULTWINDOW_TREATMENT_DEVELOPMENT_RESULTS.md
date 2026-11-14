# IDM multi-window damping-plus-wall treatment development

Status: **causal treatment heterogeneity demonstrated; predictor not yet authorized**.

Date: 2026-08-27

## Frozen causal arms

- Control: analytical selected-absorber IDM damping with the frozen
  `disturbance_p30` reference and zero learned residual.
- Treatment: identical damping plus adaptive compact wall v5.1.
- Wall: 20 s pulse, full occupied-lane coverage, 50 m compact span,
  trajectory-coherent members, adaptive servo, 20% common-target reduction,
  1.0 m/s2 assembly/servo deceleration envelope.
- Decision windows: controller steps 620, 660, 680 and 700 at 2700 veh/h
  mainline plus 400 veh/h ramp, 20% scattered CAV penetration, IDM HDVs.
- Development seeds 4101--4106 are inspected and excluded from future model
  validation and locked evaluation.

All branches use an identical 960-controller-step environment horizon. An
initial multi-window run used `decision_step + 260` as the environment horizon
and caused step-1 physical divergence for decisions before step 700. Those
artifacts were atomically replaced and are not evidence. A regression test now
requires identical total horizons.

## Verification

- Complete repository suite: 514 passed; two unrelated warnings.
- Damping-versus-damping exact repeat: passed on seeds 4101 and 4103.
- Engaged wall-versus-wall exact repeat: passed on seed 4101/step 660 and seed
  4103/step 700.
- Corrected predecision physical identity: passed for every replayed wall
  branch; first divergence occurs only at the first requested treatment step.
- In-band-only override, direct out-of-band damping, and return-to-damping
  audits pass.

## Corrected pilot result

There are 24 state-time opportunities. The damping-only timing scan admits 12
as geometrically and instantaneously safe. Ten of those 12 are dose qualified.

- Dose-qualified fraction: 10/12 = 83.3% (passes the frozen 80% threshold).
- Dose-qualified seeds: five; seed 4104 has no qualified treatment.
- Safety-admissible completed cases: 7/10 = 70% (fails the frozen 90% threshold).
- Safe beneficial completed treatments span four seeds: 4101, 4102, 4105,
  4106.
- Nonbeneficial completed treatments span four seeds: 4101, 4102, 4103, 4105.
- Zero excess collisions or teleports.
- No collapse-prevention case. Collapse prevention cannot be the primary claim.

Representative causally valid heterogeneity:

- seed 4101/step 660: safe and selected, about 769 s aggregate delay benefit;
- seed 4101/steps 620 and 680: full-dose but unsafe and strongly harmful;
- seed 4102/step 700: safe and selected, about 370 s delay benefit;
- seed 4102/steps 660 and 680: safe but nonbeneficial;
- seed 4105/step 620: safe and selected, about 1201 s delay benefit;
- seed 4105/step 660: safe but nonbeneficial.

This is conditional treatment-effect headroom rather than wall-feasibility
signal: both positive and negative labels remain after successful dose and
geometry qualification, including opposite labels within the same seed.

## Current treatment limitations

### Prospective safety

The three unsafe completed cases have minimum selected-member front/rear gaps
of 6.4 m, 7.2 m and 18.0 m. All seven safe completed cases have minimum gaps of
at least 25.6 m. Excess hard braking occurs mostly after release, so the
current closing-TTC screen does not reject tightly packed states that later
amplify the wall disturbance.

A development-derived 20 m front-and-rear activation gap floor would reject
all three observed unsafe completed cases and retain all seven safe completed
cases. It is the smallest round threshold above the largest observed unsafe
minimum gap. It is also directionally IDM-consistent: desired spacing grows
with standstill gap plus speed times desired headway. SUMO documents `minGap`
and `tau` as the standing-space and desired-time-headway components of
car-following safety.

This is not validated evidence. It is a material, post-development change to
the feasibility/safety shield and must be explicitly versioned, frozen, and
prospectively tested on untouched seeds before use in the paper.

### Assembly realization

Seed 4104/step 680 receives 10/20 s and 50% lane coverage, then aborts with
`assembly_speed_mismatch_timeout`. One member is constrained by traffic and
falls from 23.75 to 14.87 m/s while the other converges to 22.75 m/s. Extending
the timer alone would reduce the required holding fraction and does not solve
the asymmetric traffic constraint. The next repair should improve the
predecision trajectory-feasibility test or refuse this cohort, not increase
braking authority.

Seed 4106/step 620 receives a full-duration override but only 68% aggregate
frozen-lane coverage and is therefore excluded.

## Required scientific decision

Recommended primary estimand: **safety/service escalation**. Collapse
prevention remains a secondary rare outcome.

Recommended next treatment version:

1. keep wall geometry, 20% terminal reduction, 20 s duration, 1.0 m/s2
   authority, and the existing assembly lifecycle unchanged;
2. add only a versioned 20 m front-and-rear activation-gap floor;
3. retain assembly failures in the prospective dose denominator rather than
   adding another development-fitted refusal rule;
4. run a corrected 12--20-seed prospective treatment-dose/headroom pilot on
   untouched wall-development seeds;
5. train a treatment-effect model only if prospective dose, safety, positive,
   and negative diversity gates pass.

The alternative is to retain the v5.1 TTC-only screen and train a learned
safety-risk head. That is not recommended: only ten completed development
cases exist, raw safety is 70%, and the model would again risk learning a
small-sample feasibility/safety boundary instead of treatment benefit.

## Evidence artifacts

- `runs/eval/idm_wall_timing_scan_v1_fine/summary.json`
- `runs/eval/idm_wall_multiwindow_pilot_v1/manifest.json`
- `runs/eval/idm_wall_multiwindow_pilot_v1/summary.json`
- `runs/eval/idm_wall_multiwindow_pilot_v1/seeds/seed4101.json` through
  `seed4106.json`