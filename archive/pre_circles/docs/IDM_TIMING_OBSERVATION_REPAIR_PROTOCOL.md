# IDM Sequential P30 Timing — Causal Observation-Repair Protocol

**Status:** frozen 2026-08-30 before any SUMO run. Development-only (seeds
4121–4132); no Path A, no seeds ≥4201, no new treatment branches, no RL.
The retrospective timing oracle reference remains +0.04784.
**Purpose:** reconstruct compact causal B–D observations at the exact
existing t1/t2/t3 decision states (native-IDM schedule trajectories only)
and re-run the causal observation/value screen on family increments
A, A+B, A+B+C, A+B+C+D.

## 1. Cohort and identity contract (unchanged, frozen)

- Seeds 4121–4132; penetration 0.05/0.10; demands 1900/2000/2100; the 69
  existing timing states; the frozen t1/t2/t3 schedules
  (`runs/eval/idm_timing_schedules_v1`) and P30 treatment outcomes
  (`runs/eval/idm_timing_pilot_v1`).
- The reconstruction runs **only deterministic native-IDM trajectories**;
  no P30 treatment arm is executed. At every reconstructed landmark the
  capture must show exact agreement with the stored schedule record in:
  seed, demand, penetration, decision step, recurrence, target vehicle
  identity, lane, schedule/config hash, candidate observation fingerprint
  (nonclassic hash of the 134-field candidate observation) and
  physical-state fingerprint
  (`raw_decision_context_fingerprint`). Without exact identity the
  observation is **not** joined to the treatment outcome; a mismatch report
  is written and the phase stops.
- t2/t3 observations are captured from the native-IDM waiting trajectory
  (the causal state available to a controller that abstained earlier).
- Schedules, treatment mechanics, outcomes, rewards, thresholds, candidate
  selection and oracle calculations are not modified.

## 2. Frozen feature set (40 features, compact, machine-readable audit)

Every feature is computed from samples at or before the decision instant.
Provenance: name, family, source quantity, window, and allowed inputs are
recorded per feature in the artifact provenance table; forbidden inputs are
listed in §3.

### Family A — recurrence/local metadata (8)
`stage_index` (1/2/3); `landmark_wave` (wave_onset indicator);
`persistence_steps`; `runway_m`; `predicted_dose_s`; `current_instant_s`;
`lane_1`; `lane_0`.

### Family B — target-local and temporal state (14)
Instantaneous (6): `ego_speed`; `ego_accel`; `leader_gap`; `leader_rel_speed`
(leader − ego); `follower_gap`; `follower_closing` (follower − ego).
Histories (8, causal windows ending exactly at the decision): `speed_std_5s`,
`speed_std_20s` (std of target speed over the last 5/20 s);
`min_gap_5s`, `min_gap_20s` (min leader gap); `max_closing_5s`,
`max_closing_20s` (max leader closing rate in the target lane);
`severe_count_5s`, `severe_count_20s` (count of sub-−3 m/s² events of the
target lane). Windows use only samples with sample time ≤ decision instant;
a window with <2 samples marks `available=False` for that feature (missing
imputation is training-fold-only, §5).

### Family C — merge-zone RSU state (10)
`merge_lane0_mean_speed`; `merge_lane1_mean_speed`;
`merge_lane0_speed_std`; `merge_lane1_speed_std`; `merge_zone_density`
(vehicles per km in the 0–300 m zone); `ramp_count` (vehicles on the ramp
edge at the instant); `ramp_slow_count` (ramp vehicles < 3 m/s);
`merge_discharge_rate_10s` (mean steps-per-second discharge over the last
10 s); `zone_severe_count_10s`; `zone_low_ttc_count_10s` (same-lane
sub-1.5 s pair count in the zone).

### Family D — corridor state (8)
`corridor_mean_speed`; `corridor_speed_std`; `corridor_fraction_slow`
(speed < 15 m/s fraction of pre-edge vehicles); `gradient_upstream_merge`
(mean speed at [300,800) m − mean speed at [0,300) m);
`gradient_merge_downstream` (downstream-edge mean speed − [0,300) m mean
speed); `idm_wave_energy_level`; `idm_severe_short_ema`; `cav_upstream_count`
(CAVs on the pre-edge; fleet-deployment knowledge).

Excluded (never computed or exposed): seed ID; future landmarks or future
observations; selected oracle action; realized dose; treatment outcome;
terminal safety/delay/discharge/reward; artifact names or hashes as model
inputs; any quantity using samples after the decision instant.

## 3. Validation and model contract (frozen)

- Leave-one-seed-out outer validation (12 folds); all stages of a seed stay
  in one fold.
- All preprocessing, feature selection, scaling, missing-value imputation,
  regularization, thresholds and model selection happen strictly inside the
  training seeds of each outer fold (seed-grouped inner folds for selection).
- Models (no neural network; no tree library is available in this
  environment and 69 states do not support it — frozen):
  1. always abstain; 2. fixed timing rules (now/r2/r3, penalties retained);
  3. simple per-stage threshold rule; 4. ridge value model; 5. logistic
  advantage classifier with the frozen threshold grid; 6. physically
  interpretable threshold rules as reported baselines.

## 4. Sequential policy evaluation (frozen, unchanged from the prior protocol)

Cross-fitted sequential policy replayed against the existing paired library;
attempted actions retain realized penalties; never replace a failed or
mistaken predicted action retrospectively with abstention; pre-action
feasibility is the only deployment mask.

## 5. Frozen gate set (unchanged from the prior screen)

1. ≥ +0.02 mean value over abstention and over the best fixed timing policy;
2. ≥ 40% recovery of +0.04784;
3. positive aggregate value after removing each individual seed;
4. meaningful held-out benefit (> +0.02) in ≥ 3 seeds;
5. activation and wait/abstention diversity (≥ 5 each);
6. no single demand, penetration, recurrence or seed dominating (positive in
   both penetrations, ≥ 2 demands, ≥ 2 recurrences; top state < 50%);
7. above shuffled-label control (+0.01) and above the majority-action
   control (best fixed timing rule);
8. effects beyond replay/numerical noise;
9. no frozen safety/service violation among selected actions.

Required reports: activation count; helpful/harmful/neutral; dose and
constraint failures; value by seed/demand/penetration/recurrence;
family ablation A → A+B → A+B+C → A+B+C+D; selected feature coefficients and
their stability across outer folds; seed-level uncertainty; oracle recovery
per stage.

## 6. Execution order

1. this frozen protocol; 2. causal state reconstruction (native-IDM only,
   4 workers, resumable, identity-verified, mismatch → stop); 3. integrity
   tests + `python -m pytest -q`; 4. screen rerun on families A, A+B,
   A+B+C, A+B+C+D; 5. stop and report.

If the smallest passing family is A+B, A+B+C or A+B+C+D, recommend it for
the binary sequential option environment/reward audit — no RL automatically.
If every family fails: no added model complexity or development seeds;
recommend the rule-based damping boundary plus coordinated moving-wall
escalation as the paper path (fixed-P30 timing prediction unsupported).
