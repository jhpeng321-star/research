# R9 physical-support reconciliation (S-SUP, branch c4/v10r9fix)

Status at 16:35 (the owner machine restarted once at ~16:10; two 192/40-episode runs were lost
with it). Evidence below = completed runs only. **Not all 192 TRAIN episodes were replayed**
(one episode = 12.5 min single core; see "cpuN receipt"). Numbers are per HDV recorded merge in the
audit window (warm-up excluded).

## Result

| set | physics | geometry | HDV events | supported | explained exclusions | unexplained vetoes | F1 zero | M0 zero |
|---|---|---|---|---|---|---|---|---|
| BEFORE: 16 episodes (8 v5_native + 8 genrand_plain), `results/r9_support` 11:32 | unverified flat-config defaults | pack frame (junctions collapsed) | 1246 | 1120 (89.9 %) | 0 | 126 | 126 | 126 |
| AFTER: v5_native w963004/007/015/016 (`results/r9_dev4`) | round-9 preset (`build_physics`) | route frame | 288 | 285 | 3 | **0** | **0** | **0** |
| AFTER, round-8 preset physics (`results/r9_dev_r8phys`, w963004) | round-8 preset | route frame | 77 | 77 | 0 | 0 | 0 | 0 |
| SAME 4 episodes, old pack-frame verdict on the same moved states | round-9 | pack frame | 288 | 256 | - | 32 | - | - |

Before, per source (16 episodes): v5_native 613 events / 543 supported / 70 F1-M0 zero;
genrand_plain 633 / 577 / 56. v3 (genrand_plain, current code) is appended in
"Additional runs" if it finished. F1 "other 3" in dev4 are the three dropped explained sequences.
P1-P3 (strict hard-threshold law) still has 70/285 zero-mass events (event does not raise the running
maximum): a property of that deprecated law, not a support defect; F1 and M0 are the live laws.

S1 on these episodes: 100 % of retained recorded merges physically supported, 0 zero-mass events under
F1 and M0, every exclusion (3/288 = 1.0 %) explained below. S1 is NOT demonstrated on all 192 episodes.

## Root causes found (cause table, 16 old episodes -> fix)

| old cause (count) | class | root cause | fix |
|---|---|---|---|
| junction_projection_unresolved (107) | (c) contract geometry | the pack frame collapses the internal junction lanes (3.00 m main / 2.93 m ramp before the merge edge, 8.00 m after it; `:1780739017_*`, `:13638027693_*` in the net) and freezes a junction occupant's `s` at its last on-lane value. Leaders just past the 8 m junction therefore appeared 8 m too close; occupants stood still. | `src/wms/data/route_frame.py`: `x = s + shift`; on-lane shift = edge constant; junction rows dead-reckoned `x_t = x_{t-1} + v_t dt` (SUMO Euler). Verified on all 13,503 TRAIN junction exits (max error 7.6e-6 m) and 25,684 junction rows (0 missed, 0 false) |
| zero-mass "absorbed_before_event" (3 of 4 F1/M0 zeros in w963004) + lifecycle flicker | (c) | zone membership used stale/collapsed `s`: a ramp vehicle crossing the 2.93 m junction left the zone for 1-2 ticks (absorbing risk exit), so its later merge had zero mass | `geometry()` zone membership in the route coordinate (`src/wms/model/merge_pivot.py`); fitter lifecycle uses the same `geo["ramp"]` (`tools/r9_fit_exact.py` attach_episode); test `test_zone_membership_follows_the_route_coordinate_through_the_ramp_junction` |
| prior event blocks fresh entry (latent) | (c) | `certify_pre_window` fresh-entry path ignored `terminal` | `fresh_entry = ~seen & ~terminal & ...` (`merge_pivot.py` loss file) |
| outgoing_target_substep_unverified (8) | (b) recorded timing/ordering | SUMO changes lanes sequentially, front-most first: a target-lane leader that leaves in the same step is gone when the ego decides, but P1 keeps pre-change lanes. All 8 TRAIN cases: the overlapping vehicle is AHEAD of the ego | classified and excluded with its whole sequence: `src/wms/train/losses/merge_pivot_support.py` (same function in the likelihood `MergePivotTerm.prepare`, fitter `drop_explained_exclusions`, audit) |
| junction-follower overlap (2 of 1522 route-frame recorded merges) | (c)/(b) | a target-lane vehicle inside the upstream internal lane overlaps the ego's rear in the recorded successor; SUMO did not treat it as blocking | same exclusion class `junction_internal_target_vehicle` |
| wm_motion_only_intersection (7), wm_motion_outside_decision_zone (3) | (a) WM motion | recorded successor is clear, the WM's own predicted motion creates the overlap / leaves the zone | NOT explained away (see residual) |
| physics (summary.json `physics_confirmed=false`) | config | flat checkpoint config drops `sumo_fidelity` (training: true) and the round-9 preset sets `merge_end_wait=false` | `tools/r9_exact.training_physics(preset)`; reconcile/audit take `--train-config` (default `configs/c4_wm_v10round9.toml`) |

Truth check (recorded rows only, no WM, 1522 HDV events incl. warm-up, 16 episodes): overlap with
pre-change target-lane neighbours at the decision (successor) row = 211 in the pack frame, **10 in the
route frame** (8 same-step vacated leaders ahead of the ego + 2 junction followers) - so the contract
geometry now reproduces SUMO's own decisions up to those documented 0.66 %.

## Fixes (file:line at HEAD)

- `src/wms/data/route_frame.py`: `static_shift`, `junction_span`, `route_shift` (causal; only rows <= t).
- `src/wms/model/merge_pivot.py` `geometry(..., route_shift)`: overlap veto, gaps, zone in route x; returns `route_s`; on-policy states pass no shift (junction-free line; documented).
- `src/wms/train/losses/merge_pivot.py`: `build_recorded_geometry` computes shifts once (all consumers share it), adds `label_*` successor rows (never model inputs; leakage test updated to assert it), `prepare` applies the explained-exclusion and logs `support_*` counts; fresh-entry fix.
- `src/wms/train/losses/merge_pivot_support.py`: `classify_vetoed_events`, `sequence_rows`.
- `tools/r9_exact.py` (`training_physics`), `r9_runtime.py`, `r9_support_audit.py`, `r9_support_reconcile.py` (`--train-config --cache --resume --interleave --aggregate-only`), `r9_fit_exact.py` (`drop_explained_exclusions`, identity keys), `merge_fit_stage2a.py` (`--runtime-cache`), `r9_support_report.py` (tables).
- Forbidden items NOT used: no gap padding/inflation, no veto relaxation, no likelihood floor, no forced merges, no thresholds. Excluded events are only those whose recorded successor really intersects and every intersecting body is a vacated leader ahead or an internal-lane occupant; events whose recorded successor is clear (WM-motion class) are never excluded.

## Residual (class a, WM motion) and S1 verdict

In the completed route-frame runs there are 0 unexplained vetoes (0 of 288 + 77), so no class-(a) event
remains there. The old 16-episode run had 10 class-(a) events (7 wm_motion_only_intersection + 3
outside_decision_zone, 0.8 %), measured under the old physics/pack frame; whether they persist under
the corrected frame/physics is not measured at scale. If they do, F1/F2 cannot repair them (frozen WM motion).
**S1 can pass on the replayed episodes; it is not proven for all 192.** On-policy (rollout) geometry
still uses the junction-free WM line, so on-policy support near the zone ends is not covered by this change.

## Tests

`python -m pytest -q tests/test_merge_pivot_training.py tests/test_merge_pivot_pre_window.py tests/test_merge_pivot_runtime.py tests/test_merge_pivot.py tests/test_merge_pivot_hybrid.py tests/test_r9_support_audit.py tests/test_r9_runtime.py tests/test_r9_fit_exact.py tests/test_r9_exact.py tests/test_merge_fit_stage2a.py tests/test_merge_fit_v2.py tests/test_route_frame.py tests/test_merge_pivot_support.py tests/test_merge_core.py tests/test_merge_core_v2_config.py tests/test_merge_core_v2_seqtf.py tests/test_merge_core_v2_onpolicy.py tests/test_merge_context.py` -> **468 passed, 0 failed/skipped** (at commit before the last two small commits; the final targeted rerun is listed below). Fixtures that violated the new junction contract (speeds/positions across the 2.93 m lane) were made physical; parity/leakage tests were adapted (same function, same shift; `label_*` excluded and asserted to change), none weakened or skipped. Ruff clean on new/changed files (3 pre-existing F-errors in `agg_horizon.py`).

## Commands / receipts

- Dev (local, single core 12.6 min/episode): `python tools/r9_support_reconcile.py --source v5_native=data/r9_train/v5_native --workers 1 --max-episodes 1 --cache none --output results/r9_dev1`.
- `results/r9_dev4`: `... --source v5_native=... --source genrand_plain=... --workers 4 --max-episodes 4 --cache none --output results/r9_dev4` (15:22-16:02, local PC, contended).
- `results/r9_dev_r8phys`: same with `--train-config configs/c4_wm_v10round8.toml`.
- Physics for the fitter: `results/r9_physics_round9.json` (= `asdict(build_physics(load_train_config("configs/c4_wm_v10round9.toml")))`; regenerate with `tools/r9_exact.training_physics`).
- Prior agent's reconcile (PIDs 24960/22672/14732) had already exited before this session; nothing of it left running.
- cpuN receipt: Linux 6.8 Ubuntu 24.04, 48 cores, `~/miniconda3/bin/python` 3.10 + torch 2.12.1+cpu, 37 GB free. Not usable for this job: the PC->cpuN link measured 265 KB/s (20 MB in 75 s; 4 parallel streams: same aggregate), the 3.7 GB TRAIN set would take ~4 h; ssh resets after ~2 min and banner timeouts while another agent transferred. A partial `~/r9fix_sup/data/r9_train` (empty dir) was left by me and is harmless. So the scale run was local (22 logical cores shared with other agents, ~32 GB RAM).
