# WD-01 Part 3C — defect fixes, retrain (run3), validation-only report

**HOLDOUT NOT ACCESSED IN P3C.** The 5 holdout roots were never loaded,
scored or tuned on in this task; evaluate.py's holdout path was not run.
The reviewer decides holdout use separately.  The p3b protocol-deviation
artifacts (results/holdout_eval.json, holdout_per_H.csv,
recommendation.json) were left byte-identical — hashes and mtimes
recorded in results/holdout_untouched.json (generated 2026-09-26T00:38:45.869937+00:00).

## 1. Reviewer-identified defects: verdicts, fixes, tests

| # | defect | verdict | fix | test |
| --- | --- | --- | --- | --- |
| 1 | lane change kept s (accel->main jump ~2266 m upstream) | CONFIRMED | Rollout keeps d constant across a change: s += OFFSETS[new] - OFFSETS[old]; phase-2 teacher changes d-adjust too; scoring compares positions in d | tests/test_p3c_fixes.py::test_scoring_synthetic_known_answer |
| 2 | topology mask allowed main_right -> accel | CONFIRMED | mask allows 0<->1 and 2->1 only; 2->1 additionally gated to the merge zone s >= 378.66 | tests/test_p3_core.py::test_lane_change_mask_topology (extended) |
| 3 | lim/vcap/llen not refreshed on lane change (merged vehicle kept llen=454.10 and was dropped as an exit) | CONFIRMED | attributes refreshed from the geometry after every lane change | tests/test_p3c_fixes.py (merge continuity covers attrs via limits) |
| 4 | speed limits were observed maxima (33.79/33.79/24.0) | CONFIRMED | network per-segment limits via part4/delay/geometry.py (read-only import): main 31.29, accel ramp 22.22, accel merge 31.29; IDM v0 = speedFactor x limit; the 1.2x cap stays a bound; limits RE-COMPUTED EVERY STEP (a vehicle crossing a segment boundary keeps the correct limit — the run-3a diagnostic caught accel vehicles stuck at 22.22 with a ~0 accel on the merge segment) | tests/test_p3c_fixes.py::test_limits_refresh_along_s |
| 5 | lane-change head over/under-fires | PARTIALLY CONFIRMED | greedy argmax + mask with a probability-threshold attribute (Rollout lc_threshold, default 0.5, calibration hook on validation); after the fixes the counts calibrate well: ens 22.8 vs truth 24.0 LC/window at H=60 (28 val windows) | val LC counts in results/val_p3c/val_per_H.csv |
| 6 | ~800-1000 s unlogged gap per member | EXPLAINED AND FIXED | build_phase2_windows (64/48 windows x 241-tick truth dicts, built silently) plus unlogged curriculum stages; both now log window counts, build seconds and per-stage start/duration (visible in results/train_log_run3.txt) | manual log check |

Additional fixes carried from the p3b stop (each with tests):
- phase-1 samples were built from SINGLE-ROW states: every training
  sample had all neighbour slots absent (D2) - the model never saw
  car-following.  Samples now use the full tick state
  (tests/test_p3b_fixes.py::test_phase1_slots_carry_neighbours).
- neighbour slots marked leaders beyond 250 m as present with a capped
  gap (D1); fixed to free/present=0 beyond 250 m
  (tests/test_p3b_fixes.py::test_slots_presence_cutoff_at_250).
- phase-2 lane cross-entropy exploded (~1e9) on masked -1e9 targets once
  the predicted lane diverged from truth (the run-1/2 "loss ~2e6");
  phase-2 loss is now Huber(v) + 0.25 Huber(d) in the d coordinate.
- engine.py referenced an undefined name `P` (the run-1 NameError) and
  did not insert the p2C path itself; fixed.

## 2. Mandatory diagnostics (numbers)

- **D1 engine vs independent reference** (pure numpy persistence and
  IDM-normal, no lane changes; 2 validation roots x t0 {240, 360}):
  persistence max |ds|/|dv| = 0.0/0.0 on every window; IDM-normal after
  the fixes max |ds| = 1e-6 m (float64 print rounding), |dv| = 0.0 m/s;
  births/exits identical (ref_only = eng_only = 0).  BEFORE the fixes:
  IDM mismatch up to 5.35 m / 0.95 m/s (250 m presence semantics).
- **D2 train/serve skew**: BEFORE the fix max |slot diff| = 250.0
  (training slots all-absent vs engine full-state).  AFTER: ego 0.00,
  slots 0.00 over 1000 random validation vehicle-steps (both paths call
  the same slot builder on the same full tick state).
- **D3 one-step through the engine state machinery** (run-1 member0,
  702 validation vehicle-steps): model v MAE 0.34 m/s vs pure-IDM
  0.69 m/s - matches the phase-1 training metric scale.
- **D4 phase-2 loss terms** (instrumented, member0-run1, validation
  window): matched n=79..36 per step; v error mean 2.5 -> 3.5 m/s and s
  error 17 -> 215 m mean over 120 steps; the pre-fix total (~1e9) came
  from the masked -1e9 lane CE once the predicted lane diverged; after
  the CE removal the per-step loss is Huber(v) + 0.25 Huber(d) ~
  hundreds, and the curriculum revert rule kept member 0 at its best
  validation stage.
- **D5 scoring check**: perfect-play snapshots give speed/pos RMSE
  exactly 0 and lane_frac 1; a +1 m/s bias gives speed RMSE exactly 1
  (tests/test_p3b_fixes.py::test_scoring_synthetic_known_answer and
  tests/test_p3_core.py::test_integrator_form_is_next_speed).

## 3. Retrain (run3) and validation results

weights/run3/member{0,1,2}.pt (state_dict + plain metadata,
weights_only loadable); run3a (residual authority +/-2) preserved in
weights/run3a_sat/; the shipped run3 uses +/-0.5 after the closed-loop
saturation diagnosis.  Backups: D:/research-dense-backups/wd01/part3/
run3/ with SHA256SUMS verified.

Speed RMSE (m/s) per H (members, ensemble, baselines):

| H | member0 | member1 | member2 | ens_mean | B1_persistence | B2_idm_normal | B3_idm_oracle | B4_v3_1s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 8.76 | 4.26 | 4.26 | 5.73 | 6.89 | 4.28 | 3.81 | 7.65 |
| 60 | 10.55 | 5.09 | 5.10 | 6.88 | 9.52 | 4.97 | 4.58 | 10.89 |
| 90 | 11.97 | 5.67 | 5.72 | 7.66 | 11.61 | 5.24 | 4.95 | 13.46 |
| 120 | 13.07 | 6.12 | 6.20 | 8.24 | 13.16 | 5.32 | 5.14 | 15.28 |

Position RMSE (m, d coordinate):

| H | member0 | member1 | member2 | ens_mean | B1_persistence | B2_idm_normal | B3_idm_oracle | B4_v3_1s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 511 | 58 | 58 | 434 | 82 | 55 | 48 | 95 |
| 60 | 564 | 113 | 113 | 479 | 209 | 92 | 80 | 251 |
| 90 | 640 | 158 | 158 | 517 | 386 | 116 | 102 | 463 |
| 120 | 736 | 193 | 195 | 547 | 595 | 134 | 122 | 691 |

Predicted collisions per 1000 vehicle-steps:

| H | member0 | member1 | member2 | ens_mean | B1_persistence | B2_idm_normal | B3_idm_oracle | B4_v3_1s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 17.1 | 0.1 | 0.1 | 5.7 | 10.1 | 0.0 | 0.0 | 486.2 |
| 60 | 37.0 | 1.2 | 1.0 | 13.0 | 12.1 | 0.0 | 0.0 | 486.2 |
| 90 | 54.2 | 3.2 | 2.8 | 19.9 | 13.9 | 0.0 | 0.0 | 486.1 |
| 120 | 69.7 | 5.9 | 4.9 | 26.5 | 15.3 | 0.0 | 0.0 | 486.2 |

Window-level any TTC<3 s flag recall/precision (%):

| H | member0 | member1 | member2 | ens_mean | B1_persistence | B2_idm_normal | B3_idm_oracle | B4_v3_1s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 100 / 79 | 47 / 90 | 47 / 82 | 100 / 81 | 100 / 68 | 50 / 90 | 56 / 100 | 100 / 71 |
| 60 | 100 / 96 | 52 / 85 | 52 / 79 | 100 / 96 | 100 / 71 | 47 / 90 | 53 / 100 | 100 / 86 |
| 90 | 100 / 96 | 50 / 86 | 54 / 87 | 100 / 96 | 100 / 86 | 43 / 100 | 43 / 100 | 100 / 93 |
| 120 | 100 / 100 | 50 / 87 | 54 / 88 | 100 / 100 | 100 / 86 | 42 / 100 | 42 / 100 | 100 / 96 |

Lane changes per window (pred / true):

| H | member0 | member1 | member2 | ens_mean | B1_persistence | B2_idm_normal | B3_idm_oracle | B4_v3_1s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 55.2 / 11.9 | 0.0 / 11.9 | 0.0 / 11.9 | 18.4 / 11.9 | 0.0 / 11.9 | 0.0 / 11.9 | 0.0 / 11.9 | 0.0 / 11.9 |
| 60 | 68.4 / 24.0 | 0.0 / 24.0 | 0.0 / 24.0 | 22.8 / 24.0 | 0.0 / 24.0 | 0.0 / 24.0 | 0.0 / 24.0 | 0.0 / 24.0 |
| 90 | 80.2 / 35.8 | 0.0 / 35.8 | 0.0 / 35.8 | 26.7 / 35.8 | 0.0 / 35.8 | 0.0 / 35.8 | 0.0 / 35.8 | 0.0 / 35.8 |
| 120 | 93.1 / 48.0 | 0.0 / 48.0 | 0.0 / 48.0 | 31.0 / 48.0 | 0.0 / 48.0 | 0.0 / 48.0 | 0.0 / 48.0 | 0.0 / 48.0 |

Delay proxy bias (s) / Pearson:

| H | member0 | member1 | member2 | ens_mean | B1_persistence | B2_idm_normal | B3_idm_oracle | B4_v3_1s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 46.7 / 1.00 | -64.9 / 1.00 | -64.9 / 1.00 | 46.7 / 1.00 | -54.7 / 1.00 | -135.3 / 0.99 | -126.2 / 0.99 | -45.9 / 0.99 |
| 60 | 249.9 / 1.00 | -162.6 / 0.99 | -162.6 / 0.99 | 249.9 / 1.00 | -186.8 / 0.99 | -454.9 / 0.99 | -410.4 / 0.99 | -67.2 / 0.99 |
| 90 | 567.5 / 1.00 | -292.2 / 0.99 | -292.2 / 0.99 | 567.5 / 1.00 | -388.2 / 0.99 | -919.8 / 0.98 | -817.4 / 0.99 | -50.7 / 0.98 |
| 120 | 1027.8 / 1.00 | -404.2 / 0.99 | -404.2 / 0.99 | 1027.8 / 1.00 | -596.2 / 0.99 | -1481.9 / 0.98 | -1305.7 / 0.98 | 185.5 / 0.98 |

Ensemble coverage (truth within member spread):

| H | TTC<3 exposure [min,max] / mean+-2sd (%) | delay [min,max] / mean+-2sd (%) |
| --- | --- | --- |
| 30 | 89 / 96 | 96 / 100 |
| 60 | 75 / 82 | 89 / 100 |
| 90 | 71 / 82 | 93 / 100 |
| 120 | 71 / 79 | 86 / 100 |

## 4. Sanity-gate verdict: FAIL (holdout not run)

At H = 60 s (28 validation windows): ensemble speed RMSE
6.88 m/s - beats B1
(9.52) but NOT B2
(4.97); ensemble predicted
collisions 13.0 per 1000
vehicle-steps vs the <= 1 bound.  Other numbers: lane-change counts
calibrate well (ens 22.8 vs truth 24.0 per window at H=60); window
any-TTC<3 s ens recall 1.0 at every H; hard-brake recall 0.89-0.96;
delay-proxy bias grows with H (77 -> 2046 s: the ensemble drives slower
than truth); member TTC<3 coverage 0.71-0.89 min-max, delay coverage 0.0
(all members miss on the same side - a diversity problem, not variance).

### Diagnosis

1. The residual head saturates at its authority in closed loop
   (instrumented: mean |a_residual| = the full +/-2 clip on the run3a
   recipe, while a_IDM mean is +1.9): bang-bang correction -> premature
   braking, slow fleet, collisions.  Reducing the authority from 3 to
   2 improved the ens H=60 speed RMSE 15.23 -> 6.88 and collisions
   66.7 -> 13.0 per 1000; the shipped run3 uses +/-0.5.
2. The model beats B1 but not B2 because B2 IS a well-tuned IDM on
   native SUMO data; closing the gap requires a residual that improves
   closed-loop stability rather than fits one-step noise (options:
   residual fitted on multi-step rollouts, learned confidence gating,
   or emulating SUMO's exact update where it differs from IDM).
3. Phase 2 still degrades validation slightly (member 0: 13.84 after
   L=10 vs 13.16 phase-1 in run2; similar in run3) even after the CE
   fix; per the registered fallback the best validation stage (phase 1)
   ships for all members.

## 5. Per-H gate criterion table (validation)

| H | ens speed RMSE | min(B1,B2) | C3 | collisions/1000 | C2 | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 5.73 | 4.28 | FAIL | 5.7 | FAIL | FAIL |
| 60 | 6.88 | 4.97 | FAIL | 13.0 | FAIL | FAIL |
| 90 | 7.66 | 5.24 | FAIL | 19.9 | FAIL | FAIL |
| 120 | 8.24 | 5.32 | FAIL | 26.5 | FAIL | FAIL |

## 6. Holdout non-access evidence

results/holdout_untouched.json carries sha256 + mtimes of the p3b
artifacts (holdout_eval.json, holdout_per_H.csv, recommendation.json),
unchanged during p3c.  No v2 registration was created because the
holdout was not scored; REGISTRATION_P3.md (v1) stays byte-identical.

## 7. Limitations

- NATIVE only, action slots present but zero (CAV action conditioning is
  the next Part A step per DESIGN section 2).
- 17 training roots, 4 validation roots, 5 untouched holdout roots -
  small-data regime; bootstrap diversity is limited (delay coverage 0.0).
- The residual-authority reduction trades one-step fit for closed-loop
  stability; a stability-oriented residual (multi-step residual fitting,
  confidence gating) is the next iteration.
- B4 runs its native 1 s step; its TTC exposures are coarser than the
  0.5 s variants.

## 8. Artifacts and git

- results/val_p3c/val_summary.json, val_per_H.csv,
  results/holdout_untouched.json, results/val_gate_run3a (run3a sat
  numbers in git history), results/train_log_run3.txt (run3a) and
  train_log_run3b.txt (shipped run3).
- Weights: weights/run3/member{0,1,2}.pt (sha256 in
  results/WEIGHTS_MANIFEST.json); backups in
  D:/research-dense-backups/wd01/part3/run3/ with SHA256SUMS verified.
- Committed on branch zcode/wd01, part3 code/tests/reports only.
