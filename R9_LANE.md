# R9_LANE - ramp lane-end stop (physics.ramp_lane_end_stop) and residual diagnosis

Branch c4/v10r9lane (merged c4/v10r9launch 78b93f16). Status block at the bottom is refreshed as results arrive.

## Change (file:line)
- `src/wms/physics/config.py` `PhysicsConfig.ramp_lane_end_stop: bool = False` (+ bool validation, excluded from `from_params_file`).
- `src/wms/model/merge_pivot.py` `ramp_lane_end_stop(kin, next_s, next_v, present, physics, route_shift)`: a present ramp-lane car (lane one-hot of the PRE snapshot) whose longitudinal update puts its front past `physics.merge_zone_s1_m` (the network lane end, 75.44 m; cast to the tensor dtype, no tolerance) is held at the end with speed 0. Route-coordinate aware (`x = s + shift`).
- `src/wms/model/wms.py` `pivot_decision_geometry` (shared by recorded replay, fitter, runtime audit and the native step): decision geometry sees the pinned front (idempotent). `step()`: the pinned state goes into `_apply_merge_pivot`; afterwards cars that CHANGED lane that step get their free (unpinned) motion back - cars that stay on the ramp lane keep position = lane end, v = 0.
- `src/wms/train/losses/merge_pivot.py` `build_recorded_geometry`: pins `moved` before the geometry so the replay observers (runtime cache `proposals`, fitter sequences) see the same motion as the audit.
- Plumbing: `config/train.py` (`PhysicsOptions.ramp_lane_end_stop`, inference switch), `cli/assemble.py` `build_physics`, `config/legacy_flat.py`, `train/settings.py`.
- Round-9 pin (owner-approved 2026-10-22): `configs/c4_wm_v10round9.toml [physics] ramp_lane_end_stop = true`; `configs/r9_physics_round9.json` (fitter physics) regenerated = `asdict(build_physics(toml))` (only that key added). Code default stays false; round-8 preset unchanged.
- `tools/r9_support_reconcile.py --ramp-lane-end-stop on|off|preset` (default preset). Fitter honours the flag through the physics JSON; runtime cache digest already hashes `str(physics)` and the source files.
- Not a tolerance/padding: no clamp slack, no threshold; the end is the geometry value already in `physics`. Flag off = identical code path (guarded by `getattr(..., False)`).

## Tests
`tests/test_r9_lane_end_stop.py` (11): end==75.44; stopped car at the end stays (pinned, v 0); approaching car stops exactly at the end; cars not reaching it / lane-1 cars / absent slots untouched; route shift honoured; flag off = unchanged (car carried past end in legacy); step holds a staying car and leaves other slots bit-identical; a merging car is decided at the end and keeps its free motion (next_s/next_v equal the flag-off free motion); decision geometry pinned + idempotent; round-9 preset + physics JSON pin; bool validation.

## Tests run
- `python -m pytest -q --basetemp=results/pt_tmp tests/test_r9_lane_end_stop.py tests/test_merge_pivot_training.py tests/test_merge_pivot_pre_window.py tests/test_merge_pivot_runtime.py tests/test_merge_pivot.py tests/test_merge_pivot_hybrid.py tests/test_merge_pivot_memory.py tests/test_r9_support_audit.py tests/test_r9_runtime.py tests/test_r9_fit_exact.py tests/test_r9_exact.py tests/test_merge_fit_stage2a.py tests/test_merge_fit_v2.py tests/test_route_frame.py tests/test_merge_pivot_support.py tests/test_merge_core.py tests/test_merge_core_v2_config.py tests/test_merge_core_v2_seqtf.py tests/test_merge_core_v2_onpolicy.py tests/test_merge_context.py tests/unit/config`: 637 passed, 10 failed. The 10 are environment-only (the guard refuses pytest's default `E:\Temp` tmp_path outside the workspace; the real-checkpoint / TRAIN-episode tests need `data/r9_train/{wm_s_stage300.pt,v5_native/episodes/w963004...}`, copied untracked into this worktree). Re-run of exactly those files with `--basetemp` inside the workspace: `tests/test_r9_runtime.py tests/test_r9_exact.py` 35 passed; `tests/test_r9_fit_exact.py tests/test_merge_fit_stage2a.py` 66 passed => all 647 pass, none skipped/weakened.
- `tests/unit/config tests/unit/physics tests/golden tests/equivalence`: 217 passed, 26 skipped (pre-existing skips: data_cache C1 checkpoints absent / WMS_EQ_FULL unset).
- `ruff check` on the touched files: clean.
- Flag OFF on real data: new code with `--ramp-lane-end-stop off` on w963486/490/494 reproduces the S-SUP v3 events (support verdict, F1/M0/P1-P3 mass, gaps) with 0 differences over 335 events.

## Diagnosis of the 2 fast "short of the merge edge" cases (not lane-end; NOT fixed by the flag)
Both are a ramp car crossing the 2.93 m ramp junction into the merge edge and merging in its first row there (zone start x = 0, `merge_zone_s0_m`).

| | w963494 veh 783 (t 1181.0->1181.5) | w963526 veh 280 (t 439.0->439.5) |
|---|---|---|
| recorded pre route x / v | -7.820 / 16.349 | -6.851 / 14.388 |
| recorded a, v_post | -1.1685, 15.7648 | -1.0497, 13.8634 |
| recorded post x (lane 1) | 0.0624 | 0.0803 |
| SUMO Euler x_pre + v_post*dt | 0.0624 (exact) | 0.0803 (exact) |
| trapezoid with the RECORDED v/a (WM convention) | +0.146 m too far | +0.131 m too far |
| WM accel / v_next | about -3.31 / 14.696 | about -2.78 / 12.997 |
| WM predicted x (trapezoid) | -0.0587 | -0.0052 |
| miss | -0.121 = -0.267 (WM braking error) + 0.146 (integration) | -0.0855 = -0.2167 + 0.1312 |

Finding 1 (contract mismatch, size known): SUMO's recorded positions follow Euler `x_{t+1} = x_t + v_{t+1} dt` exactly (median |error| 1.5e-5 m over 198,865 same-lane rows of w963494, vs 3.0e-2 m for the WM's trapezoid `0.5 (v_t + v_{t+1}) dt`; `a` in a row is the acceleration that produced that row's speed). `advance_longitudinal` uses the trapezoid, so every moving car has a systematic bias of 0.5 |a| dt^2 = 0.125 |a| m (0.15 m at -1.2 m/s^2, 1.1 m at -9). Finding 2 (genuine model error): in both cases the WM brakes about 1.7-2.2 m/s^2 harder than SUMO as the car approaches the junction/merge edge. The trapezoid bias partly masks it (net -0.12/-0.09 m); a switch to Euler alone would make these two cases WORSE (predicted x -0.472 / -0.353, i.e. 0.53 / 0.43 m short) because the learned acceleration was trained against the trapezoid. Verdict: the two cases are model error (acceleration), amplified/compensated by a known trapezoid-vs-Euler convention difference. NOT implemented: an opt-in `euler_position_update` would change every rollout and needs a recalibration/retrain of the acceleration; recorded here as a follow-up, not as a contract fix for round 9.

## The 12 F1-unsupported cars of the 48-episode partial fit (part48b, 6847020a, 3689/3701)
Cross-referenced with this run (ON) and, where available, the flag-off replay of the same episodes.

| # | episode / veh / event row | fit label (F1 unsupported) | flag OFF | flag ON | class |
|---|---|---|---|---|---|
| 1 | w963054 / 727 / 2239 | physical veto: outside_merge_zone | not replayed | supported, F1 positive | lane-end overshoot (fixed by flag; inferred from fit label + ON result) |
| 2 | w963089 / 420 / 1335 | physical veto: outside_merge_zone | vetoed, WM front 75.531 vs recorded 75.440, v 0 | supported, F1 positive | lane-end overshoot (fixed) |
| 3 | w963098 / 526 / 1845 | zero mass, no physical veto | not replayed | supported but F1 zero, absorbed_fullpath | lifecycle absorption (not lane end), see below |
| 4 | w963113 / 495 / 1571 | zero mass, no physical veto | supported, F1 zero, absorbed_fullpath | same | lifecycle absorption |
| 5 | w963133 / 538 / 1712 | physical veto: target_interval_intersection | not replayed | still vetoed (wm_motion_only_intersection): lane-1 lead veh 427 predicted 64.362 vs recorded 64.877 (-0.51 m), overlap with the stopped ego at 59.385 is 0.023 m; recorded successor is clear (427 left to lane 0) | neighbour one-step motion error (genuine model error) |
| 6 | w963142 / 292 / 971 | physical veto: outside_merge_zone | not replayed | supported, F1 positive (ego front 75.440 vs recorded 75.439) | lane-end overshoot (fixed) |
| 7 | w963276 / 737 / 2335 | physical veto: outside_merge_zone | not replayed | supported, F1 positive | lane-end overshoot (fixed) |
| 8 | w963486 / 549 / 1651 | zero mass, no physical veto | supported, F1 zero, absorbed | same (front 6.445 vs 6.675, v 0.63) | lifecycle absorption |
| 9 | w963501 / 162 / 614 | physical veto: outside_merge_zone | vetoed, WM front 75.567 vs 75.439 | supported, F1 positive | lane-end overshoot (fixed) |
| 10 | w963633 / 688 / 2248 | physical veto: target_interval_intersection | pending | pending | likely like #5 (check) |
| 11 | w963683 / 488 / 1584 | physical veto: outside_merge_zone | vetoed, WM front 75.524 vs 75.440 | pending | lane-end overshoot (OFF verified) |
| 12 | w963727 / 196 / 779 | zero mass, no physical veto | pending | pending | likely lifecycle absorption (check) |

Lifecycle absorption (#3, #4, #8, #12): the car queues at the head of the ramp with an UNKNOWN lane one-hot for 5-23 rows (inside the 2.93 m junction, pack `s` frozen at about -0.6 .. -1.4), then crawls onto the merge edge at v < 0.6 m/s. For #8 the replay cache shows the risk-membership flag `ramp` = 1,0,1 at rows 1603-1605: the route coordinate `x` hovers around the zone start (0.04, -0.004, 0.024 in the WM's moved position), the car "starts" at 1603, leaves risk at 1604 and the absorbing exit makes the later merge zero mass. This is NOT the lane-end problem and the flag does not change it; the physical support is fine (verdict `supported`), only the strict full-path lifecycle is zero. Not a defect of F1/M0 laws; a lifecycle-at-zone-start-boundary effect for crawling queue heads (suggested follow-up: not treat a junction/zone-start boundary flicker of a stopped queue head as an absorbing exit; not implemented, as it changes the lifecycle contract). For #3, #4, #12 I verified only the unknown-lane/crawl signature, not the per-row ramp flicker.
Tally of the 12: lane-end overshoot 6 (#1, #2, #6, #7, #9, #11) - fixed by the flag (5 verified ON, #11 verified OFF only so far); lifecycle absorption 4 (#3, #4, #8, #12); neighbour motion error 2 (#5, #10 likely); the flag fixes 6/12 and leaves 6/12 (they are not lane-end).
