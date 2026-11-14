# Adaptive compact wall v5.1 development audit

Status: **treatment delivery repaired; paper gate failed; predictor not
authorized**.

Date: 2026-08-26

## Frozen treatment

Control remains the exact frozen decentralized damping checkpoint. Treatment
is identical damping plus a 20 s full-occupied-lane adaptive compact wall.
Only latched wall CAVs receive overrides; out-of-band CAVs retain identical
damping; release returns all members to damping.

The approved v5.1 change is a separately frozen adaptive-servo deceleration
bound of 1.0 m/s2 throughout ASSEMBLING and HOLDING. Front/rear TTC
projection, influence-zone monitoring, collision/teleport/hard-brake aborts,
20% maximum common-target reduction, 50 m compact span and full occupied-lane
coverage remain unchanged.

## Verification evidence

- Focused causal/controller tests: 41 passed.
- Complete repository suite: 482 passed, two unrelated warnings.
- Exact development twins:
  runs/probe/counterfactual_adaptive_compact_v5_1_exact_s504_506_step180.json
- Excluded development audit:
  runs/probe/counterfactual_adaptive_compact_v5_1_development_hyper_p20_s601_613

Both exact twins pass damping replay identity, predecision identity, treatment
repeat identity, in-band-only override, out-of-band damping identity and
release-to-damping.

Seed 504 demonstrates the intended repair:

- 20/20 s dose and 100% lane coverage;
- 100% of geometry samples within 50 m;
- maximum span 30.57 m, down from 82.82 m under v5;
- the artifact records the 1.0 m/s2 configured and realized command bound;
- delay benefit 485.69 s, queue-area benefit 677 vehicle-s and discharge
  benefit 12 vehicles;
- no collision or teleport increment and low-TTC exposure improves from two
  steps to one;
- hard-brake exposure increases from six to ten vehicle-steps, all after
  release and none recorded as following a currently controlled wall CAV.

Seed 506 remains compact and near-neutral:

- 20/20 s dose and 100% lane coverage;
- maximum span 11.64 m;
- hard-brake exposure decreases from ten to nine vehicle-steps;
- low-TTC exposure is unchanged;
- delay cost 12.74 s, queue-area benefit 49 vehicle-s and no discharge change.

## Bounded excluded-development cohort

The audit uses seeds 601--613, hyper demand, 20% CAV penetration, decision
steps 120/180/240, 20 s treatment and 180 s outcome horizon. These seeds were
previously inspected and remain excluded from prospective evaluation.

There are 39 paired cases and nine feasible activations.

### Treatment delivery

- Eight of nine feasible activations are dose/geometry qualified: 88.9%.
- All nine engaged cases receive the full 20 s override and 100% frozen-lane
  coverage.
- One case, seed 612 step 120, is within 50 m for only 20% of treatment and is
  correctly excluded from the strict completed denominator.
- No excess collision or teleport occurs.

This passes the predeclared 80% treatment-dose criterion and repairs the
dominant failure in the earlier dataset.

### Safety

Among eight strict completed treatments:

- six are safety-admissible;
- two fail the hard-brake budget;
- safety-admissible fraction is 75%, below the frozen 90% criterion.

The unsafe cases are:

- seed 606 step 180: +4 hard-brake vehicle-steps and +2 low-TTC steps;
- seed 609 step 120: +5 hard-brake vehicle-steps while low-TTC exposure
  improves by 11 steps.

In both cases the number of treatment-window hard-brake events is unchanged
from damping. The excess occurs after release over the full 180 s outcome
horizon. This does not prove the events are noncausal: the treatment changes
later traffic trajectories. Therefore the frozen full-horizon hard-brake
budget is retained and the cases remain unsafe.

### Safety/service outcome diversity

Strict completed outcomes span both decisions:

- beneficial completed treatments occur across four independent seeds;
- nonselected completed treatments occur across three independent seeds;
- seed 611 step 120 is safety-admissible but service-inferior, with 266.98 s
  delay cost and three fewer discharged vehicles;
- seeds 606 step 180 and 609 step 120 are beneficial in some service/safety
  components but are rejected by the safety budget.

This is useful diversity for a future multi-output safety/service gate, but it
does not override the failed raw-treatment safety criterion.

### Collapse-prevention headroom

Among eight strict completed pre-collapse treatments:

- collapse is prevented once, at seed 607 step 180;
- nonprevention occurs in seven cases across five seeds;
- prevention therefore spans one seed, not the required three.

Seed 607 meets the minimum 80% geometry fraction exactly but later reaches an
88.68 m maximum span. It is development evidence, not a robust
collapse-prevention claim.

The primary collapse-prevention gate fails.

## Model-observation limitation

During hyper-density replay, more than 32 active CAVs can be present. The
frozen checkpoint logs that overflow peers are truncated from the centralized
critic observation. Counterfactual identity remains valid because both arms
use the same frozen controller and deterministic truncation. Nevertheless,
the paper must disclose that the checkpoint does not provide full-fleet
centralized observability above 32 active CAVs.

## Scientific conclusion

The project is no longer blocked by a wall that terminates after a few
seconds. v5.1 reliably delivers a compact, causal intervention in most
feasible cases and produces effects above replay noise.

It is still not valid to claim:

- a verified predictor;
- calibrated treatment-effect probabilities;
- a generally safe raw wall treatment;
- robust collapse prevention;
- generalization to IDM.

## Required decision before more data

### Option A: safety/service escalation as the primary paper estimand

Recommended based on current evidence.

Define the primary decision as whether a physically feasible wall is expected
to be safety-admissible, service-noninferior and meaningfully beneficial.
Model feasibility, safety, service and collapse as separate outputs. Keep
collapse prevention as a secondary mechanism/rare high-value outcome. Freeze
the existing full-horizon hard-brake criterion rather than weakening it after
seeing development results.

The next implementation would freeze a prospective safety/service protocol
and determine whether a conservative gate can raise activated-case safety
above 90% on untouched seeds while preserving both action classes.

### Option B: retain collapse prevention as the primary estimand

Do not add seeds or train a predictor. Redesign treatment timing, duration or
geometry to produce collapse prevention on at least three independent
development seeds, then repeat treatment verification. This is higher risk
because only one prevention seed appears despite repaired treatment dose.

### Option C: redesign for raw-treatment safety before either predictor

Investigate a smoother release or lower-intensity/shorter pulse while keeping
the 90% full-horizon safety target. This materially changes the treatment and
requires a new frozen definition and exact twins. Post-release attribution
suggests this is plausible, but the current evidence does not prove release
discontinuity is the cause.

No fresh pilot or predictor-scale dataset should run until one option is
explicitly frozen.
