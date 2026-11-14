# HORIZON-CHECK (2026-10-13): how far ahead can the C4 world model be trusted for decisions?

Branch `c4/horizon` (from `c4/int` 95b92fe). Scripts: `scripts/horizon_check/`. Small results:
`scripts/horizon_check/horizon_results/` (`A_agg_results.json`, `B_analysis.json`,
`B_analysis_sampling_vs_point_8pts.json`). Raw outputs (not committed, ~0.1-0.2 MB per episode):
`runs/active/C4-HORIZON-20261013/{A,B}` in this worktree.

Models: **C4** = `WM-S-C4-20261013/fetched/gpuW/c4_wm_main/wm_s_best.pt`; **C4R** (refreshed) =
`.../loop_it1_refresh/wm_s_best.pt`; **R5b** = `WM-S-R5B-20261012/box/gpuA/r5b_main_a1/wm_s_best.pt`;
**C1** = `data_cache/ckpt_c1fast/wm_s.pt`. C4 policy = `cpuT_mappo_main/policy_latest.pt` ("mappo").
Sign convention in part B: `D_X` = improvement of arm X over all-native B (cost_B - cost_X);
positive = better.

## Bottom line

| quantity | trust up to | evidence |
|---|---|---|
| per-vehicle position (HDV median abs ds) | ~30 s | recorded states 5-6 m @30 s, 12 m @60 s, 76-117 m @120 s; SUMO saved states (imagined all-native) 5 m @30 s, 21 m @60 s, 136-148 m @120 s |
| aggregate **levels** of the t0 roster (zone delay, vehicle count, mean speed) | 60 s (120 s with caveats) | cumulative zone delay rel. err 3-8 % @60 s, 7-12 % @120 s (constant-speed baseline 10-15 % / 17-22 %) |
| aggregate levels of the **full** traffic | 60 s | at 120 s 61 % of vehicles in the RSU zone arrived after t0; WM zone count rel. err 0.51 vs 0.18 for "nothing changes" persistence; zone speed MAE 10.3 vs 3.8 m/s |
| decision: fleet-wide controller vs native (ALL vs B), damping, delay | 60 s | sign agreement 0.94 @30 s, 0.94 @60 s, 0.75 @120 s |
| decision: which CAVs / small group (GROUP, ONLY_i), damping | 30 s at most | sign agreement 0.68-0.75 @30 s, 0.50-0.60 @60 s, 0.38-0.62 @120 s; regret/spread 0.05 / 0.25 / 0.41 |
| decision: the C4 (MAPPO) policy, any arm | **not at any horizon** | sign agreement 0.34-0.38 (below chance); the WM predicts a gain in 79-100 % of MAPPO arms, SUMO in 28-62 %; WM ALL-arm gain 79 / 158 / 342 veh-s vs SUMO 9 / 16 / 28 veh-s @30/60/120 s |

**Recommended maximum imagination horizon for decisions: 30 s** for per-CAV / small-group
cooperation credit, and **60 s** only for fleet-level aggregate delay comparisons (ALL vs B with
a frozen controller). 120 s imagination is not decision grade: aggregate levels at 120 s are
dominated by arrivals the WM does not simulate, and arm differences lose their sign. A wall
(60 s+ payoff) cannot be validated by current WM imagination; use SUMO branching for that.
Reward-based decisions are worse than delay-based ones (reward sign agreement 0.75 / 0.68 / 0.53).
Hard-brake predictions are not usable at any horizon (part A).

## A. Aggregate accuracy on recorded unseen-wall states (clean-scorer rollout)

Rollout = the clean scorer's `WmsPredictor.predict_states` (CAVs teacher-forced, entries off,
seed 0, 120 s). Integrity: C1 on wall_same gives 8.7 / 99.8 m at 60 / 120 s (scorer baseline gate:
8.71 / 99.84). The recorded futures hold only the t0 roster (no entrants), so both sides are
roster-restricted. The roster's present fraction is 0.92 / 0.78 / 0.55 / 0.17-0.23 at 10/30/60/120 s.
The recorded exits also reach the model (`present_seq`, `junction_seq`), so this setting is
optimistic (see B). Fields come from `wms.control.rsu_field.build_rsu_field` (52 cells x 50 m).
Team-reward components come from `wms.control.reward.team_reward` (default weights) on every
0.5 s step over the RSU zone, with acceleration taken as the speed finite difference.
**CV** = constant-speed baseline (HDVs hold their t0 speed and lane; CAVs copied from the record).
Cells: rel. err = MAE / mean abs(recorded); r = Pearson across states.

wall_same_real (120 states; wall_same 67 / wall_pilot 53 in the JSON give the same picture):

| model | h | HDV med abs ds (m) | zone count rel.err / r | zone mean speed MAE / r | cum. zone delay rel.err / r | cum. zone reward rel.err / r | cum. hard-brake rel.err |
|---|---|---|---|---|---|---|---|
| C4 | 10 | 1.5 | 0.07 / 0.91 | 1.03 / 0.91 | 0.02 / 1.00 | 0.03 / 0.97 | 0.6 |
| C4 | 30 | 5.0 | 0.14 / 0.88 | 2.35 / 0.76 | 0.05 / 0.99 | 0.04 / 0.97 | 0.7 |
| C4 | 60 | 11.8 | 0.09 / 0.95 | 2.11 / 0.84 | 0.08 / 0.98 | 0.09 / 0.91 | 2.6 |
| C4 | 120 | 79.9 | 0.18 / 0.95 | 2.62 / 0.93 | 0.12 / 0.98 | 0.22 / 0.88 | 9.1 |
| C4R | 30 / 60 / 120 | 5.4 / 13.3 / 76.2 | 0.13 / 0.10 / 0.19 | 2.30 / 2.33 / 3.06 | 0.05 / 0.08 / 0.12 | 0.04 / 0.10 / 0.23 | 0.8 / 2.8 / 9.5 |
| R5b | 30 / 60 / 120 | 4.5 / 10.9 / 84.0 | 0.12 / 0.08 / 0.15 | 2.23 / 1.50 / 2.70 | 0.04 / 0.06 / 0.06 | 0.07 / 0.11 / 0.20 | 1.8 / 3.5 / 9.1 |
| C1 | 30 / 60 / 120 | 5.6 / 13.0 / 87.8 | 0.10 / 0.07 / 0.16 | 1.71 / 1.38 / 2.57 | 0.04 / 0.05 / 0.05 | 0.33 / 0.44 / 1.42 | 21 / 33 / 123 |
| CV | 30 / 60 / 120 | 22.6 / 91.7 / 793 | 0.12 / 0.21 / 0.52 | 2.87 / 3.89 / 6.84 | 0.10 / 0.15 / 0.22 | 0.08 / 0.10 / 0.19 | 0.7-0.8 |

Cell level, C4, wall_same_real: density MAE is 4.0 / 6.9 / 5.9 / 3.6 veh/km (r 0.84 / 0.64 / 0.76 / 0.72).
Queue-flag recall / precision is 0.83/0.88, 0.75/0.69, 0.74/0.71 and 0.80/0.70 at 10/30/60/120 s.

Reading:
- Aggregates degrade much more slowly than per-vehicle positions. Between 60 and 120 s the
  median abs ds grows 7x (12 to 80-117 m), but the cumulative zone delay error only goes from
  3-8 % to 6-12 %, and the zone count / mean speed keep r of about 0.9.
- Much of this is persistence. The CV baseline also reaches r 0.90-0.96 on delay, but its error
  is about 2x the WM's at 60-120 s. The WM adds real skill on fields (cell density r 0.76 vs 0.40
  @60 s) and on queue flags (recall 0.74 vs 0.52).
- C4, C4R, R5b and C1 are close on fields and delay. C1's reward error comes from spurious hard
  brakes. Hard-brake totals are wrong by 2.5-9x (C4 / C4R / R5b) at 60 s and beyond.
- The refreshed WM (C4R) is not better than C4 on any aggregate (per-vehicle 108 vs 94 m @120 s on wall_same).

## B. Decision accuracy vs horizon (SUMO saved-state branching vs WM imagination)

Design (`branch_horizon.py`):
- Demand 2700 veh/h with a native base episode, so every controller is branched from IDENTICAL
  start states.
- Decision points: t = 360 s (and 480 s for 4121-4124 @0.2); 16 points total. Seeds 4121-4128
  @0.2, plus 4121-4124 @0.4. 4125-4128 @0.4 were dropped to cap SUMO use.
- Arms per controller: ALL, GROUP (close chain, 2-6 CAVs), and ONLY_i (up to 3 group members),
  plus one shared all-native B. Each arm runs 120 s in a fresh SUMO process and is read at
  30 / 60 / 120 s (prefix of the same branch).
- Determinism: replaying B from the saved state reproduced the recorded B delay exactly (5/5 points checked).
- Imagination follows `wms.control.coop_imagine` semantics: oracle state, members driven by the
  controller, other CAVs as identified drivers, entries off. Drivers use the encoder point
  estimate (C4, C4R; = the scorer's path).
- 237 arm pairs per WM.

| WM | h | delay sign agree (all arms) / larger-half abs D | delay Pearson / Spearman | reward sign agree | damping ALL | damping GROUP / ONLY_i | mappo all arms | random all arms | regret/spread damping / mappo |
|---|---|---|---|---|---|---|---|---|---|
| C4 | 30 | 0.64 / 0.79 | 0.95 / 0.67 | 0.75 | 0.94 | 0.75 / 0.68 | 0.38 | 0.78 | 0.05 / 0.49 |
| C4 | 60 | 0.62 / 0.73 | 0.93 / 0.53 | 0.68 | 0.94 | 0.50 / 0.60 | 0.34 | 0.86 | 0.25 / 0.64 |
| C4 | 120 | 0.62 / 0.76 | 0.85 / 0.54 | 0.53 | 0.75 | 0.38 / 0.57 | 0.38 | 0.90 | 0.41 / 0.56 |
| C4R | 30 | 0.65 / 0.81 | 0.95 / 0.67 | 0.74 | 0.94 | 0.75 / 0.68 | 0.38 | 0.81 | 0.05 / 0.49 |
| C4R | 60 | 0.63 / 0.73 | 0.93 / 0.54 | 0.70 | 0.94 | 0.56 / 0.60 | 0.38 | 0.86 | 0.26 / 0.63 |
| C4R | 120 | 0.64 / 0.78 | 0.85 / 0.54 | 0.53 | 0.75 | 0.50 / 0.62 | 0.39 | 0.90 | 0.35 / 0.55 |

How to read this:
- The pooled Pearson (0.85-0.95) is carried by random acceleration's large harms. Within an arm
  kind and controller it is around 0, e.g. damping ONLY_i r = 0.07 / -0.18 / 0.04.
- "Regret/spread" is the SUMO loss from taking the WM's best arm among {B, ALL, GROUP, ONLY_i},
  divided by the SUMO best-worst spread (0 = perfect, ~0.5 = random pick). Random's regret is
  0.02-0.04: the answer is always "do not use it".
- MAPPO is systematically over-credited. For ONLY_i the WM predicts a gain in 91 / 89 / 79 % of
  cases, SUMO in 28 / 30 / 36 %. For ALL the WM predicts +79 / +158 / +342 veh-s, SUMO +9 / +16 / +28.
  This is model exploitation by a policy trained in this WM. Imagined credit for the C4 policy is
  not evidence of a gain.
- Damping at 120 s: SUMO says GROUP / ONLY_i still hurt (frac>0 0.38 / 0.28). The WM flips to
  "helps" (0.62 / 0.53).

Imagined all-native B vs SUMO B (C4; C4R is similar; full traffic includes arrivals):

| h | HDV med abs ds (m) | cohort share of zone vehicles | zone count rel.err: WM vs all / persistence | zone speed MAE (m/s): WM / persistence | cum. road delay: WM vs SUMO-cohort bias | SUMO cohort vs all (delay missed by dropping arrivals) |
|---|---|---|---|---|---|---|
| 30 | 5.3 | 0.94 | 0.12 / 0.13 | 1.78 / 1.87 | +42 veh-s (4 %) | -109 veh-s (10 %) |
| 60 | 21.2 | 0.87 | 0.12 / 0.14 | 2.29 / 2.37 | +227 veh-s (11 %) | -401 veh-s (17 %) |
| 120 | 148 | 0.39 | 0.51 / 0.18 | 10.3 / 3.8 | +1646 veh-s (43 %) | -1397 veh-s (27 %) |

### How much of the error comes from missing arrivals?

- **Levels.** Arrivals matter a lot from 60 s and dominate at 120 s. 13 % of zone vehicles at
  60 s and 61 % at 120 s are post-t0 arrivals, so the entries-off WM is worse than persistence on
  zone count and speed at 120 s. The WM's cumulative delay looks right against full traffic
  (rel. err 0.07-0.10) only because two errors cancel. It over-predicts the roster's delay
  (+4 / +11 / +43 %, queues that persist too long) and omits the arrivals' delay (-10 / -17 / -27 %).
- **Decisions.** Arrivals are NOT the main source of error. Sign agreement against the SUMO
  cohort-only difference is 0.64 / 0.64 / 0.61 (C4), the same as against all vehicles
  (0.64 / 0.62 / 0.62). Arrivals add only about 10 % to the SUMO arm differences
  (median abs D 156 vs 141 veh-s @120 s). Decision errors come from the WM's dynamics of the
  vehicles it does simulate: the merge queue, and the response to non-native CAV behaviour.
- Part A is roster-only on both sides by construction (no entrant data in the state sets), so A
  measures dynamics error only.

### Driver sampling in `coop_imagine` / `rsu_check` biases imagination

`coop_imagine.imagined_cooperation` (and `rsu_check`) draw driver parameters as
`estimate + N(0,1) x uncertainty`, clamped to [0.5, 1.5] x estimate. This slows free-flow vehicles
systematically. On the same 8 points, imagined-B HDV median abs ds is:

| imagination | 30 s | 60 s | 120 s |
|---|---|---|---|
| C4, sampled drivers (2 samples) | 27.8 m | 78.0 m | 103 m |
| C4, point estimate | 5.3 m | 19.3 m | 71 m |

With sampled drivers, roster delay is over-predicted by 12 / 29 / 60 % (vs 4 / 10 / 37 %), and
delay sign agreement drops to 0.66 / 0.58 / 0.57 (vs 0.68 / 0.68 / 0.64).

## C. WM improvements implied

1. **Fix driver sampling** in `wms/control/coop_imagine.py` and `wms/control/rsu_check.py`. Use the
   point estimate, or a calibrated, unclamped spread. As it stands it is the largest single error
   source at 30 s (5x the per-vehicle error).
2. **Entries in imagination**, including policy training. The predictor has entry processes
   (v1/v2), but no code under `src/wms/control/` (coop_imagine, dreamer, mappo, rsu_check) uses one. This is required for any 60-120 s aggregate:
   61 % of zone vehicles at 120 s are arrivals.
3. **Endogenous exits / no recorded `present_seq` in the scorer**. The clean scorer feeds the
   recorded presence and junction sequences, so its 120 s numbers (76-117 m) flatter the model
   relative to imagination from SUMO states (136-148 m). Add a scorer mode without them and
   train for it.
4. **Merge-queue / lane timing.** Imagined roster delay is +11 % @60 s and +43 % @120 s too high,
   so imagined merge queues persist too long. Ramp-merge acceptance timing and lane-change
   timing near the merge end are the targets. Queue-flag precision is about 0.7.
5. **120 s training stage plus an aggregate loss.** The per-vehicle error grows 7x from 60 to
   120 s. A cumulative-delay / RSU-field loss at 60-120 s would target what decisions consume.
6. **Policy exploitation guard.** Shorten MAPPO / dreamer imagination to 30 s or less, and add
   pessimism (ensemble/sample disagreement penalty) or periodic SUMO-branch validation. The WM
   over-credits the current policy about 10x.
7. **Hard-brake term.** WM hard-brake totals are wrong by 2.5-9x from 60 s. Down-weight this
   term in imagined planning scores, or calibrate it, before using it in regret.
8. `coop_imagine`'s damping/PI path is broken on c4/int: it imports `TimelineCandidate` /
   `classic_dependencies` from `wms.control.deploy`, which were removed in 1ea2409. Restored
   locally in `branch_horizon.py`.

## Cost, budget and caveats

- **SUMO use exceeded the 40-minute budget**, depending on how it is counted:
  - Branch-phase wall time summed over processes: 318 min for 16 points (about 17 x 120 s
    branches per point). Most of it is Python-side tracking and controllers on a shared, saturated
    22-core box.
  - Estimated sumo.exe CPU: about 10 min (sampled 3 s CPU per ~400 simulated s).
  - Add base episodes, 8 reimagine B replays, and about 12 episodes aborted during setup (no branches).
- Episodes 4125-4128 @0.4 were not run. Episodes 4125-4128 @0.2 and 4121-4124 @0.4 were stopped
  after one decision point.
- Sample size: 16 points and 237 pairs per WM, with points not independent within episodes.
  There are no confidence intervals. ONLY_i differences are small and partly chaotic in SUMO
  itself; no SUMO noise floor was measured.
- MAPPO runs accel-only on both sides (`PolicyExecutor(state)` without `with_lanes`), so its lane head is not exercised.
- Imagination wall time is 100-2000 s per (controller, WM, point) on the contended CPU.
