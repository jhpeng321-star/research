# C4 RSU belief state (`src/wms/control/belief.py`)

Purpose: the Option B world-model check (docs/WM_C4_DESIGN.md 2.6) needs a full WM-S state, but at
deployment the RSU only has (a) CAV reports and (b) its own sensors in fixed zones. `BeliefTracker`
assimilates both into a WM-frame state + driver history (Dreamer-like belief; EKF tracking with a
physics/WM propagator). Zero simulation, zero training; CPU only.

## API
- `BeliefTracker(config=BeliefConfig(), propagate=None)`; `update(t_s, reports, rsu_detections=None)` per RSU tick.
  - `CavReport(cav_id, s_m, lane, v_mps, a_mps2, age_s, length_m, front=Detection|None, rear=Detection|None)`;
    `Detection(gap_m, dv_mps)` = anonymous nearest same-lane neighbour within 100 m (bumper gap).
  - `RsuDetection(s_m, lane, v_mps, a_mps2, age_s)` = anonymous vehicle (HDV or CAV) in an RSU zone.
    `None` = no RSU frame; `[]` = RSU saw its zones empty.
  - `debug_id` fields are evaluation-only (truth ids for scoring); the tracker never reads them for association.
- Outputs: `state()` -> `[1,K]` `s_m, v_mps, a_mps2, lane, present, is_cav, length_m, exit_s_m` (contract keys,
  rollout-ready); `uncertainty()` -> `pos_std_m, speed_std_mps, last_seen_age_s, track_id`;
  `history()` -> slot-indexed `kin [T,K,7] / mask / target_mask / ids / history_s / dt_s / available_span_s /
  run_start_limited` (schema of `wms.eval.history_inputs`); `driver_history()` -> `[1,K,10,6]` + mask (identical to
  `wms.model.history.history_step_features`, tested); `rollout_inputs()` -> `state, history, history_mask` for
  `WmSModel.rollout`; `coverage(window_s, zone)` -> fraction of lane length observed recently (5 m cells), per-lane
  CAV/RSU-only density and flow (optional entry prior; not used to spawn vehicles).
- Propagators: `idm_propagate` (default; SUMO-vType IDM, ramp end obstacle, simple mandatory merge) or
  `wms_step_propagator(model, **step_kwargs)` (wraps `WmSModel.step`; CAVs default to `hold_speed_cav_controller`).
- Ablations: `BeliefConfig(rsu_sensors=False)` = CAV reports only; `oracle_mask_state(true_state, detection_mask(true_state))`
  = "sensed-only" truth (CAVs + currently detected HDVs). Synth helpers: `reports_from_state`, `rsu_detections_from_state`.

## Design
- CAV tracks keyed by CAV id. HDV detections: absolute (s, v) from gap/dv (front s = s_cav + gap + 5 m assumed length;
  rear s = s_cav - L_cav - gap), extrapolated over receipt age (noise inflated with age), removed if at a CAV position
  (detected CAVs are those CAVs), clustered across reporters/RSU, then Hungarian assignment (scipy; greedy fallback) on
  gated Mahalanobis (s, v) cost, adjacent-lane moves (0<->1, 2->1) allowed with a penalty. Unmatched -> new track.
- Prediction: whole belief (compact occupied-track state) through `propagate`; covariance by constant-velocity
  Jacobian + white-acceleration noise (q = 0.6 m^2/s^3).
- Drops: exit (`exit_s_m`), position std > 40 m, negative information (an unmatched HDV track predicted inside space a
  CAV sensor or RSU zone currently sees empty), CAV report timeout 5 s.
- History mask marks samples seen within 1 s, so the driver encoder never learns from the belief's own IDM guesses.
- Layer rule: `wms.control` is a root-layer package (core/config only), so physics is re-implemented minimally in
  `idm_propagate` and the WM is wrapped by duck typing.

## RSU zones (contract constants, c4/base 004181a)
`RSU_MAINLINE_S_M = (-800, 225.44)` lanes 0/1, `RSU_RAMP_S_M = (-378.66, 75.44)` lane 2, half-open `[lo, hi)`.
Checked against the frozen recon_v6 RSU zone in `output/wd01/partA_v2g/params_nominal.json` (`rsu_zone_default`:
MAIN_ZONE_D (0,800) on both mainline lanes, merge lanes 1487606179_{0,1,2}, whole ramp 11553048#1_0, DOWN_ZONE_P (0,150)):
physical `[1845.67, 2871.11)`, `[1844.22, 2869.66)`, `[0, 454.1)` minus the lane offsets 2645.67 / 2644.22 / 378.66 =
exactly the contract values. No mismatch. The recon_v6 RSU sensor code itself was not found in this repo
(only the frozen zone), so whether it supplies cross-frame ids is unverified; the belief assumes anonymous RSU detections.

## Synthetic results (tests/unit/test_c4_belief.py `main`, no SUMO)
2 mainline lanes over the WM road (-2645.67..553.33 m), heterogeneous IDM HDVs (v0 27-33, T 0.9-1.7, a, b, s0, length
random), inflow upstream, merge-zone slowdown (v0 = 8 m/s for t in [20, 70) s) sending a wave upstream, no lane
changes. Reports every 1 s: 5 % missing, age 0 or 0.5 s, gap noise 0.3 m, dv 0.1 m/s; RSU noise 0.3 m / 0.1 m/s.
Mean of 3 seeds. recall = truth HDVs with a track within 10 m (whole road / RSU mainline zone); err = |s_track - s_truth|
over HDV tracks; false = HDV tracks >= 10 m off; calib = err <= 2 sigma + 1 m; cover = cells seen within 5 s.

| pen | RSU | recall all @60/@120 | recall zone @120 | med err m @120 | p90 err m @120 | false @120 | cover @120 |
|---|---|---|---|---|---|---|---|
| 2 %  | off | 0.05 / 0.05 | 0.07 | 0.55 | 105.3 | 0.22 | 0.11 |
| 2 %  | on  | 0.48 / 0.50 | 1.00 | 0.15 | 5.3   | 0.07 | 0.44 |
| 5 %  | off | 0.14 / 0.14 | 0.14 | 0.39 | 70.2  | 0.19 | 0.27 |
| 5 %  | on  | 0.54 / 0.54 | 1.00 | 0.14 | 3.2   | 0.05 | 0.54 |
| 10 % | off | 0.23 / 0.25 | 0.22 | 0.36 | 98.0  | 0.21 | 0.45 |
| 10 % | on  | 0.59 / 0.62 | 1.00 | 0.19 | 0.6   | 0.04 | 0.69 |
| 20 % | off | 0.35 / 0.39 | 0.32 | 0.32 | 26.1  | 0.13 | 0.64 |
| 20 % | on  | 0.64 / 0.72 | 1.00 | 0.16 | 0.5   | 0.03 | 0.84 |
| 40 % | off | 0.67 / 0.64 | 0.61 | 0.26 | 1.1   | 0.05 | 0.84 |
| 40 % | on  | 0.81 / 0.84 | 1.00 | 0.16 | 0.5   | 0.01 | 0.94 |

Identity-switch rate <= 0.05 % of associations everywhere. Errors do not grow over 30 -> 120 s (no drift: the belief is
re-anchored each tick). CAV-only: tracks that exist are accurate (median < 0.6 m), but most HDVs are missing and the
p90 is dominated by stale propagated tracks (up to the 40 m std threshold). With RSU sensors the 1 km zone is complete
and, at 2 %, the whole-road recall rises 0.05 -> 0.50; upstream of -800 m recall still equals CAV-only.

## Limitations
- A vehicle never seen by a CAV or the RSU does not exist in the belief (upstream of -800 m at low penetration most
  HDVs are missing; WM checks there under-count demand). The density/flow estimate is reported but no vehicles are spawned.
- Synthetic test has no lane changes, no ramp traffic and noise-free CAV accelerations; SUMO lane changes, occlusion
  beyond the nearest neighbour and real comm delays are untested. Default IDM propagator merges ramp vehicles as soon as
  lane 1 has 3 m gaps (crude); the WM-S propagator wrapper does not thread driver history.
- HDV length is assumed 5 m for front detections; RSU/CAV noise levels and q are hand-set, not calibrated on SUMO.
- Stale unobserved tracks are kept until their std exceeds 40 m; a WM check should weight/ensemble them using
  `uncertainty()` (not done here).
