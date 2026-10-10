# IDM v5.4 observation-v2 review protocol

**Status:** review-only proposal; not authorized for implementation, feature
recapture, fitting, or new-seed execution  
**Date:** 2026-08-31  
**Trigger:** consider only after the frozen v1 prospective validation is reported
without reinterpretation and the owner approves an observation repair  
**Scientific role:** causal decision-time observation repair for the two-stage
safety-and-benefit escalation gate

## 1. Scope and invariants

This protocol does not change the treatment, labels or deployment thresholds.

- Control remains the frozen decentralized IDM damping checkpoint.
- Treatment remains identical damping plus the frozen v5.4 P30 moving wall.
- Gate-F admission, wall membership, assemble--hold--release behavior, safety
  projection and smooth return to damping remain frozen.
- Safety target remains `safety_admissible`.
- Benefit remains `wall_selected` conditional on observed safety and full dose.
- Duration-dose target remains realized duration fraction at least 0.80.
- Safety and benefit thresholds remain 0.70 and 0.75.
- Delay remains a noninferiority guardrail; queue area is not a selection target.
- No realized treatment field, outcome, oracle, seed identity or vehicle identity
  may enter the learned vector.

The first execution is development-only on already-opened seeds 4401--4520.
Seeds 4601--4660 and 5001--5060 are prohibited. New validation seeds require a
separate freeze and approval.

## 2. Proposed smallest observation contract

### 2.1 Compact measured context

Keep exactly these ten nonredundant saved-prefix summaries unless the pre-fit
audit finds an implementation mismatch:

1. `pattern60_pre_merge_speed_std_mps_change`
2. `pattern60_pre_merge_speed_mps_change`
3. `local_lane_1_speed_std_mps`
4. `obs_v2_premerge_gap_lag1_autocorr_60s`
5. `traffic_lag_30_soft_band_lane_coverage_fraction`
6. `traffic_lag_5_density_scaled`
7. `preflight_min_front_ttc_s_capped`
8. `preflight_min_rear_ttc_s_capped`
9. `preflight_min_acceleration_mps2`
10. `obs_v2_dose_release_boundary_margin_m`

The first six describe wave state, persistence and CAV placement. The last four
describe t0 treatment headroom. They are retained as context, not asserted to be
sufficient.

### 2.2 New shadow outputs

The learned vector receives scalar or fixed-length lane summaries, preferably
as treatment-minus-control deltas:

- projected duration fraction and assembly success;
- assembly time and speed-convergence time in seconds;
- maximum member span, minimum lane coverage and speed spread;
- minimum front and rear TTC at 1, 2, 5 and 10 seconds;
- minimum front/rear gap and release safety margin;
- minimum follower acceleration, required deceleration and compression rate;
- number of followers/vehicles predicted to encounter the wall;
- per-required-lane gap and TTC quantiles (minimum, 10th, 25th percentile);
- command-authority delta and projection-survival fraction;
- acceleration clip, conflict and saturation fractions;
- same-lane reserve count and replacement availability by required lane;
- damping-versus-wall crossing-count delta and predicted inflow reduction;
- excess-demand and downstream-storage delta;
- estimated shock-gradient location, direction/speed and persistence.

No variable-length identity list enters the model. Lane summaries use the frozen
required-lane ordering and fixed absent-lane masks.

## 3. Exact predecision capture

Capture once from the common damping trajectory after the base damping action is
computed and before any real wall request is applied.

### 3.1 Metadata and fingerprints

- simulation time, time step and declared decision step;
- demand, penetration, lane geometry and IDM model contract;
- frozen damping checkpoint/controller fingerprint;
- v5.4 wall profile and treatment fingerprint;
- extractor/shadow schema and source hash;
- route/demand and RNG-state fingerprints needed for deterministic replay.

### 3.2 Vehicle state in the causal influence corridor

For every vehicle on premerge, ramp, merge and downstream lanes plus a buffer
long enough that no unrecorded vehicle can reach the proposed wall in ten
seconds, capture:

- internal linkage ID, road, lane, lane position and distance to merge;
- route index, type, CAV/HDV status and wall eligibility;
- speed, acceleration, length, minGap, tau, accel, decel, emergencyDecel,
  allowed/max speed and speedFactor;
- sorted lane adjacency plus leader/follower IDs, net gaps and relative speeds.

Internal IDs are used only to maintain simulated identity and are dropped before
feature serialization.

### 3.3 Controller and wall proposal state

- active CAV IDs and selected wall members;
- every same-lane eligible reserve candidate, rank and available headroom;
- required/occupied lanes and release-boundary distance;
- damping step, lane references, prior acceleration, filtered residual,
  structured lane action, release/projected-release state and cooldowns;
- each member's nominal, desired and safety-projected acceleration, target speed,
  constraint reasons and projection-survival flag.

## 4. Deterministic shadow equations and execution

From one immutable capture, create two in-memory twins:

1. frozen decentralized damping;
2. identical damping plus the frozen v5.4 wall proposal.

Advance both for 20 steps of 0.5 seconds (ten seconds total). Use frozen per-type
IDM dynamics, identical boundary assumptions, fixed-lane motion for this short
projection, and the exact v5.4 assembly, hold, handoff and safety kernels. The
shadow must be side-effect-free: it may not mutate the real simulator, damping
controller, wall state or RNG.

If the capture corridor cannot exclude a new arrival within ten seconds, create
deterministic boundary ghosts from prefix-only detector flow/speed. Do not read
the actual future trace. Report an uncertainty/boundary flag and exclude cases
whose shadow depends materially on an unmodelled boundary arrival.

Record outputs at 1, 2, 5 and 10 seconds. Use wall-minus-damping deltas for flow,
storage, compression and shock quantities so the model estimates incremental
treatment consequences rather than merely reconstructing congestion level.

## 5. Causality and implementation tests

The extractor/shadow implementation must pass before any model screen:

1. identical common-prefix capture and fingerprints across control/treatment;
2. deterministic byte-equivalent features on exact replay;
3. order invariance when input vehicle enumeration is permuted;
4. unchanged features when actual dose, completion/release reason, arm outcomes,
   oracle labels and effects are mutated after capture;
5. no seed, absolute ID, realized branch state or postdecision timestamp in the
   model vector;
6. target-only wall commands and identical out-of-band damping in the shadow;
7. zero side effects on the real branch/controller/RNG;
8. monotonic toy tests: closing follower/reduced rear gap cannot improve the
   projected rear-safety margin; more saturation cannot improve projected dose;
9. exact schema, units, missing masks and feature-name hash;
10. fail-closed behavior on an incomplete capture or uncapped nonfinite value.

## 6. Development-only screen

After the implementation tests pass:

1. Recapture all 480 decision states on seeds 4401--4520. Compute shadows for
   the 199 Gate-F admissions and join by the frozen case ID to existing labels.
2. Do not regenerate labels or modify treatment outcomes.
3. Compare three prespecified blocks: compact current context; shadow only;
   compact context plus shadow.
4. Compare only regularized logistic regression and the existing shallow
   booster.
5. Use five outer folds grouped by complete seed and inner grouped folds for all
   block/model selection.
6. Report safety, conditional-benefit and duration-dose AUC/AP/Brier; fixed
   0.70/0.75 policy precision, recall, commands and command seeds; folds,
   seed-removal sensitivity, held-out permutation importance, family ablation,
   nearest-neighbor consistency and shuffled-label controls.

### Development gate

The observation repair passes only if all of the following hold:

- at least one primary target (safety or conditional benefit) improves grouped
  out-of-fold AUC by at least approximately 0.03 over its frozen v1 comparator;
- the same candidate materially improves fixed-threshold policy behavior rather
  than obtaining AUC alone;
- selected safety precision is at least 0.70 and joint-benefit precision at
  least 0.75 at the unchanged thresholds;
- coverage does not collapse below the frozen minimum command-case and
  independent-command-seed support;
- improvement survives shuffled-label controls, every outer fold, and
  leave-one-seed metric sensitivity without one seed dominating;
- no leakage, replay, dose, collision/teleport or service-noninferiority gate
  fails.

Failure means the current wall outcome is not sufficiently observable for a
learned online escalation gate under this experiment. Do not lower thresholds
or add identical seeds after seeing that failure. Report the fixed conservative
supervisor and treatment boundary honestly.

## 7. Prospective v2 test after a pass

Only after owner review may one feature contract, model family and bundle be
frozen. Allocate 60--120 new independent validation seeds that exclude:

- development seeds 4401--4520;
- frozen v1 prospective seeds 4601--4660;
- reserved final-paper seeds 5001--5060;
- every other seed inspected during mechanism development.

The validation protocol must be written and hashed before outcomes are opened.
The v2 result cannot be combined with v1 to rescue either prospective test.

## 8. Runtime estimate

At four workers:

- recapture of the 120 development prefixes: approximately 1.5--2.5 hours;
- pure 10-second shadows for 199 admissions: under approximately 15 minutes;
- grouped screen: approximately 5--10 minutes;
- later 60-seed paired prospective test: approximately 3--6 hours.

Benchmark four seeds first and update only the runtime estimate, not the frozen
scientific criteria.
