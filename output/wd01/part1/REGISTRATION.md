# WD-01 Part 1 — T2 rollout-drift protocol registration

- Experiment ID: WD-01, part 1, task T2.
- Registered (this file written and sha256-hashed): 2026-09-25, UTC timestamp
  recorded in REPORT_PART1.md and results/registration_record.json.
- Status at registration: T2 has NOT been scored. No T2 number in this file.
- Model: surviving pinned checkpoint
  `D:/research-dense/output/worldmodel_v3_vehicle/vehicle_dynamics.pt`
  (sha256 `ded83907b8d218574d3f5359c332408f17167da055574afbe29eaecb59d4a8f5`),
  MLP 12->128->128->1, dv over 1 s. Action-blind native-dynamics baseline
  (in-window physics arm-independent; action one-hot constant per run).
- Data: the 21 holdout runs of `build_tracks_manifest.json`, evaluated on the
  regenerated 602 s prefixes (`runs/regenerated_v1`, retry1 where present).
  T1a identity: all 88 rebuilt tracks are array-exact equal to the surviving
  npz (max abs diff 0.0), so regenerated prefixes stand in for the deleted
  original runs exactly.
- Zero simulation. No SUMO/TraCI. CPU only (torch 2.12.1+cpu). Device resolved
  once via `WD01_DEVICE` (default: cuda if available else cpu).

## 1. Rollout protocol

- Start ticks t0 in {120, 180, 240, 300, 360, 420, 480}; horizons H in
  {30, 60, 90, 120} s; every t0 satisfies t0 + H <= 600, so each H uses all
  7 ticks: 21 runs x 7 t0 = 147 evals per H, 588 eval total per variant.
  No subsampling (CPU budget: measured throughput reported; estimated << 2 h).
- Vehicle set: every vehicle present in the true state at t0 with a valid
  lane group (the builder's state definition; junction/internal and unmapped
  lanes are excluded at t0 already). No births after t0. No lane changes:
  each vehicle keeps its t0 lane group for the whole rollout.
- State per vehicle: merge-relative d (m) and speed v (m/s); d decreases
  forward. Step dt = 1 s (the model's native step).
- Leaders: recomputed EVERY step within each lane group from the PREDICTED
  states (sort by d, rearmost first; leader = next vehicle ahead, i.e. next
  smaller d; gap = d_self - d_leader; free iff no leader in group, gap > 250 m
  or non-finite gap, in which case v_leader = v_self and gap = 250 m, exactly
  the training featurisation).
- Model input: features (v_self, v_leader, gap, lane one-hot(4), free flag,
  action one-hot(4) with the run's arm index), standardised by the
  checkpoint's x_mean/x_std; dv = model(x); one forward pass per step over
  all vehicles (batched).

### 1a. RAW variant (original propagate)

    v_raw = v + dv;  v' = max(v_raw, 0);  d' = d - (v + v')/2

### 1b. CONSTRAINED variant (hard-constraint integrator, DESIGN Part A §2.3)

    a  = clip(dv, -9.0, +3.0)                      # m/s^2 over the 1 s step
    v' = clip(v + a * 1 s, 0, v_cap[lane_group])   # v_cap = 1.2 x train max
    d' = d - (v + v')/2 * 1 s
    per lane group, processed FRONT to BACK (ascending d, leader first):
      if d'_self <= d'_leader:  overlap event counted (not hidden);
                                d'_self := d'_leader + 5.0   # 5 m behind

with v_cap[g] = 1.2 x max{ v_self : TRAIN split, lane_group = g } computed
from the surviving train npz (the model's own training data). Cap values are
recorded in results/t2_summary.json when scoring runs.

## 2. Truth comparison

- Per vehicle id. A vehicle-step h in [1, H] is comparable iff the vehicle is
  present in the true state at t0 + h AND its true lane group equals its t0
  lane group. The fraction of vehicle-steps lost (truth lane change or exit)
  is reported per H (and per run).
- Predicted tracks are scored on the identical tick grid.

## 3. Events (C1)

- TTC event (per vehicle-second): at tick t, the vehicle has an in-group
  leader (free flag 0), gap = d_self - d_leader >= 0, closing speed
  dv_c = v_self - v_leader > 0, and gap / dv_c < 3.0 s. Evaluated at ticks
  t0+1 .. t0+H. Predicted events use predicted states and predicted leaders;
  truth events use truth states and truth leaders.
- Hard-brake event (per vehicle-second): v(t+1) - v(t) < -3 m/s over the
  1 s step, vehicle present at t and t+1 (truth: in truth; predicted: in the
  rollout; lane group change allowed). Transitions t in [t0, t0+H-1].
- Event universe per eval: the window (t0, t0+H].
- Matching: an event is (vehicle id, tick). A predicted event matches a truth
  event iff same vehicle id and |t_pred - t_truth| <= 2 s; greedy nearest
  matching, each event used at most once, per vehicle.
- recall = matched_truth_events / truth_events; precision =
  matched_pred_events / pred_events; pooled over all 147 evals of that H
  (pooled counts, not per-eval averaging). Per-run recall/precision also
  reported.

## 4. Runaway (C2)

- Speed cap violation: vehicle-step with v' > 1.2 x v_cap_base[g] where
  v_cap_base[g] is the train-max above (i.e. v' > v_cap of §1b). Counted for
  BOTH variants (CONSTRAINED must be 0 by construction; a nonzero count is a
  bug and fails the run).
- Negative speed: v' < 0, counted for both (RAW can produce it only via
  v_raw < 0 which clamps to 0; reported for completeness).
- Overlap (pass-through) events: from the CONSTRAINED integrator flags; for
  RAW the same geometric rule (d'_self <= d'_leader after the step, within
  the lane group) is applied diagnostically. Rate reported per 1000
  vehicle-steps.
- Max predicted speed over the whole rollout grid (m/s).

## 5. Accuracy (C3)

- Speed RMSE at step H: per (run, t0): RMSE over comparable vehicles at
  step H of (v_pred - v_truth); headline = mean over the 147 evals.
  Pooled (all comparable vehicle-steps of all evals in one RMSE) reported
  too. The same form for position RMSE (metres).
- Baselines: (i) constant-speed persistence: v' = v, d' = d - v (RAW form);
  passed through the CONSTRAINED integrator with dv = 0 for the CONSTRAINED
  comparison; (ii) dv = 0 baseline: numerically the identical trajectory to
  persistence in both variants (dv = 0 => v' = v); reported once and noted as
  identical.
- Per-step curves (step 1..120, per variant) go to CSV; H-points are the
  steps 30/60/90/120 of the same curves.

## 6. Registered PASS rules (verbatim from the WD-01 part 1 brief)

- C1: recall >= 0.80 for both event types (TTC<3 s and hard braking), per
  horizon and variant.
- C2: zero vehicle-steps above 1.2 x lane-group train max AND overlap events
  <= 1 per 1000 vehicle-steps.
- C3: speed RMSE at horizon H <= persistence speed RMSE at H.

A horizon H "passes" in a variant iff all three hold in that variant.

## 7. Recommendation rule

part_a_horizon_s = the LONGEST H in {30, 60, 90, 120} that passes C1, C2 and
C3 in the CONSTRAINED variant (RAW results reported alongside). If no H
passes, the recommendation states so and identifies the first failing
criterion and the horizon at which it fails (largest H whose criteria all
hold is reported as the fallback bound). This rule and every threshold above
are frozen as of this file's hash; any later change is a registered DEVIATION
reported in REPORT_PART1.md.

## 8. Deliverables of T2

results/t2_summary.json (headline + verdicts + recommendation + throughput),
results/t2_per_horizon.csv (H x variant table), results/t2_per_run.csv,
results/t2_per_step.csv, results/t2_c1_events.json (counts per H x variant),
capped raw per-step metrics. REPORT_PART1.md section-by-section write-up.

## 9. Known limitations (also restated in the report)

Action-blind 60-600 s window (fleet switch at 600 s); no lane-change or birth
model; 1 s step is coarse for TTC<1.5 s discrimination; data provenance is
the regenerated prefixes (T1a-verified array-exact); vehicle set frozen at t0
so late-window exits are censored, not predicted.
