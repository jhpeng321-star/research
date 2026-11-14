# GENRAND - undirected random CAV-action data + paired branches (branch c4/genrand, 2026-10-14)

Owner rule: the WM learns action effects from UNDIRECTED data only (native, generic random
direct-output commands, the learned policy's own rollouts). No designed controllers or
scripted behaviours. Only random STRUCTURE is allowed.

Code:
- `traffic_shepherd/c4/genrand.py`: `GenericRandomPolicy`, `run_genrand_job`, branch episode writer.
- `traffic_shepherd/c4/branching.py`: `plan_seed_arms` / `reseeded` / `seed_for_arm`, plus
  `measure_episode(seed_arms=M)`.
- `traffic_shepherd/c4/collect.py`: `--policy generic_random --gr-mode --case-plan
  weighted-cycle --pen-weights --branch-*`.
- `traffic_shepherd/c4/genrand_driver.py` + `scripts/genrand/collect_genrand.sh`: box driver.
- Tests: `tests/unit/test_c4_genrand.py`.
- Registration draft: `GENRAND_REGISTRATION.md`.

## 1. Generator (`GenericRandomPolicy`, seeded per episode)

Everything is per CAV. CAVs are tracked by vehicle id, and every hold is log-uniform 2-60 s in
whole control seconds. The command approaches its target at a per-segment jerk U[0.5, 3] m/s^3.
A new CAV starts from its measured acceleration. Commands are NOT pre-clipped: the env's
speed-bound clip gives `issued`, and SUMO speedMode 27 then gives `realized`.

**`--gr-mode speed` (default).** Each hold draws a target speed relative to the speed limit
(31.29 m/s):
- 65 % U[0.8, 1.15];
- 25 % U[0.3, 0.8];
- 10 % "free" (command +2.6; SUMO caps it at the allowed speed).

The segment also draws an acceleration limit U[0.5, 2.6] and a deceleration limit U[1, 3]. The
command is `clip(v_target - v, -dlim, alim)` per 1 s. A CAV below 1 m/s for more than 20 s ends
its hold early.

A first version scaled targets by the local-flow speed instead of the limit. That compounded
(CAV mean speed 8.2 vs native 20.5 m/s) and was replaced (commit 1d965a5).

**`--gr-mode accel`.** Target acceleration U[-3, 2.6] held 2-60 s: the literal first spec. In
the smoke it parked CAVs: 58 % of CAV-steps were below 1 m/s (native: 5 %).

**Shared latents (both modes).** Per episode, rho ~ U(0, 1) and 1-3 latent processes are
drawn. Each latent is a random target (speed factor or acceleration) held 2-60 s. At every
latent hold boundary, each present CAV independently joins with probability rho: its new
target is `w * latent + (1 - w) * own draw` with w ~ U(0, 1), and its hold matches the latent's
hold. Nothing about position, lane or traffic enters the choice.

**Lane intents.** The per-episode rate is log-uniform in [1/600, 1/20] s^-1 per CAV (Poisson).
Each intent goes left or right (+-1 WM lane) and is re-requested for a log-uniform 1-10 s or
until the CAV reaches that lane.
- Every tick passes a `lanes` array, so CAVs stay under the env's policy lane control
  (laneChangeMode 513, safety-gapped `changeLane`).
- The env refuses impossible targets: ramp lane, non-adjacent lanes, edge ends, junctions.
- With the lanes array, SUMO's own discretionary lane changes are off for CAVs, as for the
  lane-head policy.

## 2. Channels (episode + conversion)

`episode.npz` per control tick:
- `requested_acceleration_mps2`: policy output.
- `issued_acceleration_mps2`: after the env's speed-bound clip (`[-3, 3]`, `v + a dt` in
  `[0, allowed]`).
- `lane_request_target`: NEW. The policy's lane request, before the safety mask.
- `lane_command_target`: the issued `changeLane`.

The realized acceleration is in the state (`a_mps2`, SUMO).

`wms.data.c4_episodes` changes:
- It copies `requested_acceleration_mps2`, `lane_command_target` and `lane_request_target` as
  extra per-episode arrays. They are stored for later use and the loader does not read them.
  The WM schema has no lane-command slot, and the model is unchanged.
- The loader's `action_timeline` is still the ISSUED command, and `states[..., a]` is still
  the REALIZED acceleration. A unit test checks that they differ.
- Manifest entries carry `genrand` (episode parameters), `branch`, `pair_key`, `t_start_s` and
  `extra_arrays`. Records carry `meta.pair_key`.

## 3. Paired branches (`run_genrand_job`, `--branch-cadence 60`)

1. **Base episode.** It runs uninterrupted and is recorded like every collector episode. At
   each decision time t_k = warmup + 60 k with t_k + H <= end, the run calls `saveState` and
   snapshots the env's Python state (the recorder is detached) and the policy. Saving does not
   perturb the run.
2. **Branches.** After the base finishes, each saved state is loaded into a FRESH SUMO
   (`branching.load_state`, speedMode 27 re-asserted). The arms are:
   - `SEED:j` for j < M (default M = 4): the snapshotted policy deep-copied and re-seeded with
     `seed_for_arm(policy_seed, t_k, j)`. Episode parameters and current command levels are
     kept, every CAV starts a new segment, and pending lane intents are dropped.
   - `B`: all CAVs native. SUMO's lane changing is back on for CAVs.

   Each arm runs H = 60 s. Common random numbers come from the identical saved SUMO state,
   including the RNG (`--save-state.rng true`). Once trajectories diverge, the RNG draws can
   diverge too.
3. **Branch episodes.** Each branch is written to
   `<out>/branches/generic_random/<job>/t<tk>_{seed<j>,native}/` as a `c4_sumo_episode/1`
   episode:
   - it holds the base's last 60 s before t_k, plus the branch;
   - `warmup_s = t_k` (windows start at the branch), `world_t0_s` = base warmup (same world id,
     so the same TRAIN/val split side);
   - a unique `arm` (`generic_random_b0360_seed0`, ...), and
     `branch = {pair_key, base_world, base_episode, branch_t_s, arm, kind, arm_seed, ...}`;
   - `entry_first_seen_s` / `entry_first_entrance` from the FULL base run, so the converter's
     300 s entry counter is exact. Unit test and SUMO smoke: equal to the base at t_k.

   `BRANCHES_COMPLETE.json` marks the job complete. Resume re-runs the base and all branches if
   it is missing (both directories are preserved as `.incomplete-N`). The saved-state XMLs are
   scratch and are deleted per point.
4. **Determinism.** `--branch-repeat-check N` re-runs `SEED:0` at the first N points.

## 4. Collection driver

```
scripts/genrand/collect_genrand.sh runs/active/C4-GENRAND-20261014 961000-961479 961480-961515 46 [--speed-factor 1.3] [--no-physics-log]
# == python -m traffic_shepherd.c4.genrand_driver --out ... --plain-seeds ... --branch-seeds ... --workers 46
```

- Runs `plain` and `branch` collect processes concurrently, splitting workers by estimated work.
- Thread env = 1 per worker.
- Resumable: re-run the same command.
- Writes `GENRAND_MANIFEST.json` (arguments, git commit, estimate, return codes, conversion) and
  an aggregate `COLLECT_SUMMARY.json` (plus one per phase).
- Converts into `c4_episodes/`, or `c4_episodes.<n>` if that already exists and is non-empty.
- Defaults: penetrations 0.05/0.1/0.2/0.4 with weights 1/1/2/2, demands 2400/2700, warmup
  300, horizon 900, cadence 60, M = 4, H = 60, repeat check 1.
- Plain collection without the driver:
  `python -m traffic_shepherd.c4.collect --policy generic_random --seeds 961000-961479
  --case-plan weighted-cycle --penetrations 0.05,0.1,0.2,0.4 --pen-weights 1,1,2,2 --demands
  2400,2700 --warmup 300 --horizon 900 --workers 46 --out runs/active/C4-GENRAND-20261014/collect`
  (with PYTHONPATH=src).

## 5. Smoke evidence (laptop, exposed dev seeds; raw data git-ignored under `runs/active/C4-GENRAND-SMOKE-20261014/`)

Same scenario for every column: seed 4107, m2700, p0.2, warmup 300, 300 s window. Stats are
over the window.

| | native | speed (default) | speed v1 (local flow) | accel |
|---|---|---|---|---|
| CAV-steps < 1 m/s | 4.9 % | 2.2 % | 6.2 % | 58.4 % |
| HDV-steps < 1 m/s | 5.0 % | 3.2 % | 5.5 % | 27.1 % |
| CAV / HDV mean speed (m/s) | 20.5 / 20.2 | 19.3 / 20.5 | 8.2 / 11.5 | 5.0 / 11.6 |
| arrivals | 233 | 242 | 142 | 128 |
| issued != requested (> 0.05) | - | 3.9 % | 0.6 % | 44 % |
| realized != issued (> 0.2) | - | 61 % | 23 % | 24 % |
| collisions / teleports | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| sim-s per wall-s (episode / window) | 4.8 / 3.3 | 5.9 / 4.1 | 4.3 / 2.8 | 5.4 / 3.9 |

Further smokes:
- **p0.4 speed mode** (seed 4108, m2700, 300 s):
  - CAV / HDV mean speed 17.8 / 19.9 m/s; CAV < 1 m/s 4.3 %.
  - Lane commands: 381 issued and 36 completed within the 1 s request; 309 refused as not
    allowed (ramp / non-adjacent targets).
  - 0 collisions.
- **Branch** (seed 4107, p0.2, warmup 300 + 120, point t = 360, M = 2 + native, repeat check 1):
  - 3 branches, 0 collisions / teleports.
  - SEED:0 repeat is identical (max |ds| = 0, 0 mismatches).
  - Each 60 s branch takes 7.5-8.0 s wall including the fresh SUMO + loadState.
  - Conversion of base + 3 branches: the branch 60 s pre-history is identical to the base,
    and the branch counter at t_k equals the base counter.
  - Windows: 30 / 22 / 10 at 10 / 30 / 60 s horizons; at 60 s, one window per arm with the
    same t0.
- **Driver** end to end: 1 plain + 1 branched episode, then conversion. Result in
  `driver/GENRAND_MANIFEST.json`.

**Cost.** In speed mode a 1200 s episode takes about 260 s on one laptop core (about 0.25
CPU-s per simulated second in the window, about 0.17 including warmup). A branched episode
(14 points x 5 arms) takes about 800 s. 10,260 windows (516 episodes, 36 of them branched)
take about 45 core-h, i.e. about 1.0 h on 46 workers at laptop core speed (about 1.3-1.5 h
with slower cores).

## 6. Limitations

- Lane-intent completion is counted within the 1 s request: SUMO usually needs longer, so the
  intent is re-requested for its hold.
- The branch arms' lane intents start empty, and native arms restore SUMO's discretionary lane
  changing.
- The `branching.py` CLI path (`measure_episode(seed_arms=M)`) calls the policy accel-only
  (no lanes). The GENRAND driver uses `run_genrand_job`, which passes lanes.
- f32 converted arrays: about 20 MB per 1200 s episode and 2 MB per branch.
