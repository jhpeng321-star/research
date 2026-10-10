# CHANGELOG_P3C — part 3 defect fixes, run3 retrain, validation-only report

Task: fix the reviewer-identified Part A v0 engine/training defects,
retrain, and report VALIDATION results.  The HOLDOUT was not read, scored
or tuned on in p3c (evidence: results/holdout_untouched.json; the p3b
deviation artifacts results/holdout_eval.json, holdout_per_H.csv,
recommendation.json were left byte-identical).

Run-1 weights (stopped p3 run) preserved: results/run1_weights_sha256.txt
(originals) and results/run1_weights_sha256_after_move.txt (copies in
weights/run1_stopped/).

## Fixes (all with tests; see tests/)

- F1 (defect 1, lane-change coordinate): Rollout keeps d constant across
  a lane change: s += OFFSETS[new] - OFFSETS[old] (accel->main no longer
  teleports ~2266 m upstream; main_left<->main_right no longer shifts
  1.45 m).  Phase-2 teacher lane changes d-adjust too; scoring compares
  positions in d.  Test: tests/test_p3c_fixes.py (scorer/merge tests).
- F2 (defect 2, topology mask): mask allows only 0<->1 and 2->1; 2->1 is
  additionally gated to the accel merge zone s >= 378.66.
  Test: tests/test_p3_core.py::test_lane_change_mask_topology.
- F3 (defect 3, lane attributes): lim/vcap/llen refreshed after every
  lane change (a merged vehicle no longer keeps llen=454.10 and is no
  longer dropped as a false exit).
  Test: tests/test_p3c_fixes.py (merge test asserts limits).
- F4 (defect 4, speed limits): network per-segment limits (main 31.29;
  accel ramp 22.22, accel merge 31.29) via part4/delay/geometry.py
  (imported read-only); IDM v0 = speedFactor x limit; vcap = 1.2 x limit.
  Plus F4b found in D1: limits are re-computed EVERY step (a vehicle
  crossing a segment boundary kept the old limit: accel vehicle stuck at
  limit 22.22 with ~0 accel on the merge segment).
  Tests: tests/test_p3c_fixes.py::test_limits_refresh_along_s,
  test_speed_limits_network.
- F5 (defect 5, lane-change rule): greedy argmax + topology mask with a
  probability-threshold attribute (lc_threshold, default 0.5;
  calibration hook on validation).  Post-fix validation counts: ens
  22.8 LC/window vs truth 24.0 at H=60 - calibrated, no extra threshold
  needed.  Counts per H in results/val_p3c/val_per_H.csv.
- F6 (defect 6, unlogged ~800-1000 s gap): build_phase2_windows (64/48
  windows x 241-tick truth dicts, built silently) plus unlogged
  curriculum stages; both now log (window count, build seconds, stage
  start/duration).
- F7 (from D1, the 250 m presence semantics): neighbour slots marked
  leaders beyond 250 m present with a capped gap; fixed to
  free/present=0 beyond 250 m.
  Test: tests/test_p3b_fixes.py::test_slots_presence_cutoff_at_250.
- F8 (from D2, the critical training defect): phase-1 samples were built
  from SINGLE-ROW states, so every training sample had ALL neighbour
  slots absent (max |slot diff| = 250 vs the engine) - the model never
  learned car-following.  Samples now use the full tick state (per-tick
  slot cache for speed).
  Test: tests/test_p3b_fixes.py::test_phase1_slots_carry_neighbours.
- F9 (from D4): phase-2 lane cross-entropy exploded (~1e9 per step) on
  masked -1e9 targets once the predicted lane diverged from truth (the
  reported loss ~2e6).  Phase-2 loss is now Huber(v) + 0.25 Huber(d) in
  the d coordinate (lane head trained by CE in phase 1 only, evaluated
  greedily).
- F10: phase-1/phase-2 supervise position in the d coordinate (y_d
  target; comparable across lane changes).
- F11: engine.py defines P2C/HERE before use and imports p3lib as P (the
  run-1 NameError); engine also inserts the p2C path itself.

## Diagnostics

- D1 (engine vs independent numpy reference, 2 val roots x t0 {240,360},
  persistence + IDM-normal): persistence max |ds|/|dv| = 0.0/0.0 on
  every window; IDM-normal after F4b/F7 max |ds| = 1e-6 m (float64 print
  rounding), |dv| = 0.0 m/s; births/exits identical.  BEFORE the fixes
  the IDM mismatch reached 5.35 m / 0.95 m/s.  Script: diag_d1.py.
- D2 (train/serve skew, 1000 random validation vehicle-steps): BEFORE
  max |slot diff| = 250.0 (training slots all-absent vs engine
  full-state); AFTER 0.00 ego / 0.00 slots (hist is built by the same
  code path).  Script: diag_d2d5.py.
- D3 (one-step through the engine state machinery, run-1 member0, 702
  validation vehicle-steps): model v MAE 0.34 m/s vs pure-IDM 0.69 m/s
  - consistent with the phase-1 training metric.
- D4 (phase-2 loss terms, instrumented): matched n=79..36 per step; v
  error mean 2.5 -> 3.5 m/s and s error 17 -> 215 m mean over 120
  steps; the pre-fix ~1e9-per-step explosion came from the masked -1e9
  lane CE once the predicted lane diverged.  After F9 the per-step loss
  is Huber(v) + 0.25 Huber(d) ~ hundreds at L=120.
- D5 (scorer): perfect-play snapshots -> speed/pos RMSE exactly 0,
  lane_frac 1; +1 m/s bias -> speed RMSE exactly 1.
  Tests: tests/test_p3b_fixes.py::test_scoring_synthetic_known_answer,
  tests/test_p3_core.py::test_integrator_form_is_next_speed.

## Defect (5) calibration numbers (validation, H=60)

- ensemble lane changes per window: 22.8 (pred) vs 24.0 (truth);
  per H in results/val_p3c/val_per_H.csv.  The argmax+mask rule with the
  corrected topology is adequately calibrated; lc_threshold stays 0.5.

## Retrain run3

- Recipe: phase-1 (4 epochs, <=20000 samples/root, Adam lr 1e-3
  wd 1e-4) + curriculum (10,200)/(40,120)/(120,60) with per-member val
  revert; residual authority +/-0.5 m/s^2 after the saturation
  diagnosis (run3a at +/-2 preserved in weights/run3a_sat/ with
  results/run3a_sat_weights_sha256.txt; its gate numbers are in
  results/val_gate_run2.json-era logs and the first val_report pass:
  ens H=60 speed RMSE 15.23, collisions 66.7/1000).
- Wall clock: ~1974 s training + ~700 s validation report.

## Validation gate: FAIL (holdout not run)

28 validation windows (4 roots x t0 {120..480 step 60}), H=60:
- ensemble speed RMSE 6.88 m/s - beats B1 (9.52) but NOT B2 (4.97);
- ensemble predicted collisions 13.0 per 1000 vehicle-steps (> 1);
- window any-TTC<3 s recall 1.0 (base rate 0.786-1.0), hard-brake
  recall 0.89-0.96; delay coverage within member min-max 0.0 at H=60
  (members miss on the same side).

Per the brief, the holdout was therefore NOT evaluated.  Diagnosis and
next steps (residual stability: smaller authority/gating/multi-step
residual fitting) in REPORT_P3C.md.

## Final checks

- pytest tests/: 13 passed.
- vehicle_dynamics.pt sha256 ded83907...d4a8f5 unchanged (start/end).
- p2C MANIFEST.json sha256 2439004d...c82dd27 unchanged.
- p3b holdout artifacts byte-identical (results/holdout_untouched.json).
