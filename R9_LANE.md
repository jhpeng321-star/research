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

Update (all 28 episodes finished): #10 (w963633/688) is still vetoed with the same signature as #5 (wm_motion_only_intersection, WM neighbour motion; ego front 68.967 = recorded); #11 (w963683/488) is supported with F1 positive under the flag (front 75.440 = recorded 75.440); #12 (w963727/196) is supported but F1 zero, absorbed_fullpath (lifecycle absorption). Final tally of the 12: lane-end overshoot 6 (#1, #2, #6, #7, #9, #11) all fixed by the flag; lifecycle absorption 4 (#3, #4, #8, #12) unchanged; neighbour-motion intersection 2 (#5, #10) unchanged.

## Before / after (flag OFF vs ON), route frame, round-9 physics, TRAIN only
Per HDV recorded merge in the audit window. BEFORE = S-SUP `results/r9_support_v3` (14 genrand_plain) + `r9_dev4` (4 v5_native) at the old code, which I re-verified: new code with the flag OFF reproduces it with 0 differences (3 episodes, 335 events). AFTER = this worktree, `--ramp-lane-end-stop on`, `results/on` (local, not tracked).

| set | HDV events | supported | explained exclusions | unexplained vetoes | F1 zero | M0 zero |
|---|---|---|---|---|---|---|
| 18 S-SUP episodes, OFF | 1328 | 1311 | 11 | 6 | 5 | 5 |
| 18 S-SUP episodes, ON | 1328 | 1315 | 11 | 2 | 1 | 1 |
| all 28 (18 + 10 episodes holding the 12 part48b cars), ON | 2076 | 2058 | 14 | 4 | 6 | 6 |

Residual under ON in the 18: 2 vetoes = the two fast short-of-edge cases (w963494/783, w963526/280, WM acceleration error, see above) and 1 F1/M0 zero = w963486/549 (lifecycle absorption, physically supported). Extra 10 episodes: vetoes #5 and #10 (neighbour motion), F1 zero #3, #4, #12 (lifecycle absorption). `mismatched_events` (audit vs fitter) are the two fast cases, as before. P1-P3 zero-mass 593/2060 (deprecated strict law, unchanged character).

## Review summary (self-review; the astra reviewer was not available in this session)
The stop is a physical constraint, not a tolerance: the end is the network value already in `PhysicsConfig` (75.44), the update only removes motion beyond it and sets v = 0 (SUMO behaviour), cars that change lane keep their free motion, nothing is padded, floored or forced, the default is false and the flag-off path is unchanged (0 differences on real data, unit-tested). The decision geometry uses the pinned front, which is what the recorded data show (a car that merges at the end is recorded at 75.44 = the lane end). Limits: (1) the stop is applied to all ramp-lane cars incl. CAV motion; (2) on-policy rollouts use the junction-free line, so there the end is `merge_zone_s1_m` directly; (3) 2 fast cases remain (WM braking error, plus a known 0.125 |a| m trapezoid-vs-Euler convention gap that I did not change); (4) lifecycle absorption of crawling queue heads at the zone-start boundary (4 of 12) is a separate, unaddressed issue.

---
# Part 2 - lifecycle absorption of crawling queue heads at the zone start (`physics.ramp_risk_retain`)

## Exact rule that produced the zero mass (549 = w963486 veh 549, tick 1603-1605)
The car reaches the head of the ramp queue and crawls through the 2.93 m ramp junction (lane one-hot unknown on rows 1598-1605, pack `s` frozen at -0.6224, then 0.0406 on row 1606). SUMO's own route position follows Euler exactly: x(1605) = -0.001 m, x(1606) = 0.0406 m (zone start `merge_zone_s0_m` = 0). The decision-zone membership `geometry()['ramp']` is, by the P1 contract, evaluated on the WM-PREDICTED next position (moved route x), not on the recorded position. For 549 the moved route x is +0.043 (tick 1603), -0.004 (1604), +0.024 (1605): the WM's one-step displacement of a car whose true x is -0.14 ... -0.001 jitters around the bound 0. So `ramp` = 1,0,1 on ticks 1603-1605 (cache `results/cache_on`, ramp row sequence `[... 1 0 1 1 ...]`). The first 1 starts the risk chain (entry tick 1603), the 0 at 1604 is an absorbing exit (`terminal = started & ~ramp`), so the real merge 48 ticks later has no running maximum to raise = zero mass although physically supported. True route x is monotone, so a start-side dip of the PREDICTED position is not a zone exit. The same signature (long unknown-lane junction crawl) holds for the other three; all four are fixed by the same rule (below). The consumers of that bookkeeping were: `MergePivotTerm.prepare` (training loss), `certify_pre_window` (pre-window history), `WmSModel._apply_merge_pivot` -> `update_state` (native rollout), `tools/r9_support_audit.py` (audit) and `tools/r9_fit_exact.attach_episode` (fitter `lifecycle_active`).

## Change
- `geometry()` now also returns `ramp_open` = present, lane == ramp, finite, route x <= `merge_zone_s1_m` (the same lane/end test as `ramp`, WITHOUT the start bound). It never makes a car eligible.
- `merge_pivot.close_risk(risk, ramp_open, reset)`: a risk chain that has started stays at risk on later ticks while the car is still `ramp_open` (on the ramp lane before its lane end) and the identity is unchanged. Real exits still end the chain: lane change (open false), passing the lane end (open false), absence/identity change (reset). `update_state(..., retain=)` is the online equivalent (a started, non-terminal chain is kept while `retain`). The bridged tick has `eligible` false, so c = -inf: it adds no mass and no running maximum, it just does not terminate the chain. No constants, no hysteresis, no padding.
- Applied identically in `MergePivotTerm.prepare`, `certify_pre_window(retain=)`, `_apply_merge_pivot`, `r9_support_audit.audit_episode` and `r9_fit_exact.attach_episode` (`lifecycle_active`), all gated by `physics.ramp_risk_retain` (default false = bit-identical; plumbed like `ramp_lane_end_stop`: `PhysicsConfig`, `PhysicsOptions`, `build_physics`, `legacy_flat`, `TrainSettings`). Round-9 preset and `configs/r9_physics_round9.json` set it true.
- Tests `tests/test_r9_risk_retain.py` (5): close_risk semantics (bridges only a started open chain; real exit, new identity and never-started cars are not bridged), `ramp_open` has no start bound but keeps lane and end and is never eligible, `update_state(retain=)` (kept / never starts / real exit terminates), `certify_pre_window` chain and history mask with and without the flag, default-off + preset/JSON pin. Run: `python -m pytest -q --basetemp=results/pt_tmp3 <the 21 touched files of part 1 + tests/test_r9_risk_retain.py + tests/unit/config>` -> 652 passed, 0 failed (647 + 5).

## Verification (replay, flag ON = both R9LANE flags, TRAIN only)
Episodes w963098, w963113, w963486, w963727 (the cars 526, 495, 549, 196) plus w963004 and w963490 as regression: 640 HDV events compared with the previous ON run (`results/on`): exactly 4 events changed, the 4 target cars: F1 / M0 zero -> positive, `absorbed_fullpath` True -> False (549 and 526/495 also P1-P3 positive). Audit and fitter agree (no `mismatched_events`, 0 strict-window-vs-fullpath disagreements). All other 636 events are identical (verdict, F1/M0/P1-P3 mass, absorbed flag). F1/M0 zero per episode: 486 1->0, 727 1->0, 098 1->0, 113 1->0, 004 0, 490 0.
