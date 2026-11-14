# C4 SUMO side — notes (branch c4/sumo, 2026-10-12)

Code: `traffic_shepherd/c4/` (env, conversion, baselines, recorder, collector, smoke).
Tests: `tests/unit/test_c4_sumo.py` (15; one SUMO smoke, skipped without `sumo`). Full suite
`PYTHONPATH=src python -m pytest -q tests/unit`: 648 passed, 1 skipped, 1 failed (pre-existing
`test_drive_registry` registry-text mismatch, also failing without this branch).
Design: [docs/WM_C4_DESIGN.md](docs/WM_C4_DESIGN.md) §2.7; contract `src/wms/control/contract.py` (unchanged).

## 1. API

```python
from traffic_shepherd.c4.env import CavSumoEnv
from traffic_shepherd.c4.baselines import run_policy_episode, RandomAccelPolicy, make_policy

env = CavSumoEnv(k_slots=192)                      # one TraCI connection per process
state = env.reset(seed=4107, demand_vph=2700, penetration=0.2, warmup_s=300, horizon_s=600,
                  controller="direct",              # direct|native|damping|pi|wall
                  episode_dir="runs/active/X/ep1")  # optional recorder
state, info = env.step(accel_by_slot)              # [1,K] torch/numpy, absolute m/s^2
```

* `reset(seed, demand_vph, penetration, warmup_s, horizon_s)` = contract `CavEnv.reset`
  (+ optional `controller`, `episode_dir`, `meta`).  Scenario = `traffic_shepherd.scenarios.
  build_scenario` (Poisson demand, ramp 400 veh/h, driver types, nested CAV draws), preserved
  network bytes, 0.5 s physics, 1 s control.  The warmup always runs native; for damping/PI the
  frozen `FleetSwitchSession` switches native -> controller at `t = warmup_s`.
* `step(accel_by_slot)`: one 1 s control interval (2 physics steps).  `direct`: each live CAV slot's
  value becomes `CommandOwnership.step(wall_commands={vid: a})` -> `direct_target` (clip to
  [max(-3, -v/dt), min(1.5, (v_allowed - v)/dt)]) -> `slowDown(target, 1 s)`; NaN / missing ->
  that CAV is released to native for the tick.  HDVs are never commanded.  `CommandOwnership` is
  the sole writer; CAV speedMode **27** (campaign mode, `cav_speed_mode=27`; the adapter config
  still declares 31 — 27 is the explicit emergency-braking candidate used by every WM campaign).
* State dict (batch 1, contract): `s_m, v_mps, a_mps2` f32, `lane` long (0 left, 1 right,
  2 ramp/accel, -1 absent **or junction transit**), `present`, `is_cav` bool, `length_m`
  (nominal 5 m, as the packs), plus `junction` bool (extra key).
* `info`: `time_s`, `interval` (per-tick SUMO metrics below), `actions` {vid: controller},
  `rsu_detections` (every vehicle in the RSU zones: vehicle_id, slot, s_m, lane, v, a, is_cav,
  length_m, junction, timestamp), `dropped_vehicles`, `done`; on the last step `summary`.
* Metrics per tick / episode / post-warmup window: requested-not-arrived vehicle-seconds incl.
  source waiting (`RequestLedger`), `source_wait_vehicle_seconds`, `speed_deficit_vehicle_seconds`
  (Σ dt·max(0, 1 - v/31.29)), arrivals/departures, boundary crossings (mainline entry/exit, ramp
  entry), `braking_vehicle_seconds` (a < -3) and `acceleration_energy` (`braking_effort`),
  `ttc_below_3_vehicle_seconds` + minimum closing TTC (same 200 m leader query and minGap
  restoration as `metrics.ttc_exposure`, from subscriptions), collisions, teleports.
* History for the WM driver encoder: `env.history_state(history_s)` -> slot-indexed
  `state["history"]` dict of the `wms.eval` schema (`kin [T,K,7]`, `mask`, `target_mask`,
  `history_s`, `history_stride_s`, `dt_s`, `available_span_s`, `run_start_limited`);
  `env.driver_history(history_s=None|H, history_stride_s=1, history_len=10)` ->
  `(driver_history [1,K,H,F], mask)` for `WmSModel.rollout` (long: `long_history_features`, F=9;
  short: `history_step_features`, F=6).  Buffer = last 241 ticks (120 s), configurable.
* `run_policy_episode(env, policy, seed=..., ...)`: `policy` = callable `state -> accel [1,K]`
  (direct) or `"native" | "damping" | "pi" | "wall"`.  `ContractPolicyAdapter(policy,
  obs_builder, group_encoder=, rsu_field=, safety=)` composes the `wms.control` contract pieces
  into such a callable.  `RandomAccelPolicy(seed)` = smooth random direct accelerations
  (U[-2.5, 1.5] targets held 2-8 s, 1 m/s^3 jerk) for WM training data.
* Wall baseline (`WallLaw`): `wms.physics.wall_exec.NumpyWallExecutor`, law v2.2 merge-release;
  formation rule of this harness: one CAV per mainline lane, s in [-2000, -300] m, offset
  <= 75 m, wall speed clip(min v - 3, 6, 25); other CAVs damping; 30 s cooldown after handback.
  Check (seed 4108, m2600, p0.2, 120 s after a 120 s warmup): 2 walls formed, 1 assembled and held
  to the merge release, 0 collisions. Baselines native/damping/PI also ran clean (0 collisions).
  It is a harness approximation, NOT the registered wall-v2 SUMO supervisor
  (`output/wall_v2_*`, not in this tree).

## 2. SUMO -> WM-frame conversion (parity verified)

Same encoding as every WM-S pack: frozen WM-N1 featurizer (`output/wd01/wm_n1/features`):
`LANE_TABLE` + `HERE_OFFSETS` (copied to `c4/geometry.py`, unit test asserts equality), s = s_down
(merge zone 0..75.44; mainline upstream to -2645 m, ramp from -378.66 m, downstream > 75.44),
persistent slot pool (sorted ids, freed slots first, lowest free slot, drop when full), junction
rows: lane -1, s carried.  **Every vehicle on the network is converted** (all of the RSU zones).
Verified on the smoke episode (seed 4107, m2700, p0.2, 420 s, 72,983 vehicle rows):
`parity.verify_episode` re-featurizes the recorded physics log with the frozen `parse_run` and
finds identical slot assignment, lanes, junction flags, v, a; |Δs| <= 1.2e-4 m (float32; the
packs store float16).  RSU 50 m cell counts of the converted state == an independent reference
from SUMO lane lengths on every step (0 failures); the table sensor == `SumoLocalSensor.capture`
(1,403 sampled CAV frames, 0 mismatches).

## 3. Episode format (WM refresh / Dreamer data loop)

`<episode>/`: `manifest.json` (status running/complete/failed), `summary.json`,
`episode.npz` (`c4_sumo_episode/1`), `run/physics_steps.jsonl.gz` (wm41_logging/1 rows, the input of
the WM-N1 featurizer / hetpack pack builders), `run/action_timeline.jsonl.gz`
(`traffic_shepherd_action_timeline/1`: per 1 s per CAV `owner` in {native, damping,
pi-external-u, direct}, `issued_command` accel, target speed, requested accel, measured speed),
`scenario/`.

`episode.npz` (T states at 0.5 s incl. t = 0, C control ticks, K slots): `time_s [T]`,
`s_m, v_mps, a_mps2, length_true_m [T,K]`, `lane, lane_carry [T,K] i8`,
`present, is_cav, junction [T,K]`, `vid_index [T,K]` -> `vehicle_ids`, `cav_vehicle_ids`,
`control_time_s [C]`, `controller [C,K]` (wms `CONTROLLERS` ids: native 0, damping 1, pi 2,
direct 3), `issued_acceleration_mps2`, `issued_target_speed_mps`, `requested_acceleration_mps2`
`[C,K]`, `controller_vocab`, `meta_json`.  Control row c is decided on state 2c.

Converter: `recorder.episode_windows(ep, window_s=30, stride_s=10, history_s=60)` ->
`TrajectoryDataset.from_arrays` arrays (`states [N,T+1,K,7]`, `present`, `identities`, `is_cav`,
`junction`, `controller_timeline [N,C,K]`, `issued_*`, `history_kin [N,H,K,7]`, `history_mask`,
`history_target_mask`, `controller_vocab`); `write_trajectory_npz(path)`;
`TrajectoryDataset.from_pack(path, history_s=60)` loads it (tested + smoke: 4 windows, driver
history [K,61,9], CAV teacher, per-CAV direct action timeline).  Open for the WM workstream: the
trainer currently mixes only A11 packs (`from_real_packs` / `MixedTrajectoryDataset`); feeding
these npz windows needs a mixdata source hook, or run the physics logs through the existing
WM-N1 featurizer -> pack -> `ACTION_TIMELINE_INDEX` sidecar path (hetpack, branch wms/swpack).

## 4. Collector

`python -m traffic_shepherd.c4.collect --policy random|native|damping|pi|wall|zero|module:factory|<ckpt>
--penetrations 0.05,0.1,0.2,0.4 --demands 2400,2600,2700 --seeds 4107-4148 --warmup 300
--horizon 600 --workers N --out runs/active/<ID>` (`--policy-loader` default
`wms.control.policy:load_policy_fn`, `--check-parity`, `--no-ttc`, `--no-physics-log`).
Spawn pool, one SUMO per worker, write-once episode dirs `<out>/episodes/<policy>/<s.._m.._p....>`,
resumable (complete skipped; incomplete kept as `.incomplete-N`, attempt recorded), `INDEX.json`,
`collect_log.jsonl`, `COLLECT_SUMMARY.json` (speed).  Seeds: only the exposed controller-development
seeds 4107-4148 (`c4/seeds.py`); sealed, final-test, chunk-5, TEST_REPL, planner, WM-28/34 and
holdout blocks are refused explicitly.  Fresh seeds for the final comparison need a registered,
exposure-audited block.

Evidence: `_out/c4_sumo/SMOKE_20261012.json` (`python -m traffic_shepherd.c4.smoke`, PASS) and
`_out/c4_sumo/SPEED_4W_20261012.json` (collector speed run); raw episodes stay local under
`runs/active/C4-SUMO-{SMOKE,SPEED}-20261012/` (git-ignored).

## 5. Measured SUMO speed (this laptop, 22 logical CPUs, SUMO 1.27.1, TraCI over loopback)

| setting | sim-s / wall-s |
|---|---|
| 1 worker, steady state, 420 s episode (m2700 p0.2, random direct, <=128 veh, recorder+TTC) | 10.0 per episode (7.4 in the post-warmup window) |
| 4 workers in parallel, 4 x 300 s episodes (m2700 p0.2/0.4, random), incl. worker start (torch/ONNX import) | 5.4 per worker (window 4.1); aggregate 16.9 |
| smoke with every parity check on (420 s) | 4.2 (window 2.6) |

Per-step cost is dominated by TraCI Python parsing of subscription results (~0.5 ms per TraCI
call on Windows loopback, so every per-vehicle getter was replaced by one subscription read).
Sizing: ~5-7 sim-s/wall-s per worker under load => one worker does a 900 s episode (300 warmup
+ 600 horizon) in ~2.5-3 min; 1 core-hour ~ 20-25 episodes.  A 48-core box at ~40 workers ~
800-1000 episodes/h (assumes Linux scales similarly; not measured there).  libsumo is not installed
(would remove the socket cost).

## 6. Limitations / notes for the coordinator

* Contract: lane = -1 also for junction transits (pack parity); an extra `junction` key and
  `lane_carry` in detections are provided.  `length_m` is the pack nominal 5 m; true SUMO lengths
  are in `episode.npz` / detections.
* K = 192 slot cap (pack format): overflow vehicles are dropped from the state for that tick
  (counted: `dropped_vehicle_steps`, `dropped_cav_steps`; dropped CAVs run native).  None observed
  (max 130 present at m2700).
* Layering: `tests/unit/test_import_graph.py` forbids `traffic_shepherd -> wms` imports.  All wms
  use is confined to `traffic_shepherd/c4/wms_glue.py` (lazy: wall executor, history featurisers,
  contract constants), added as the one sanctioned exception in that test — please confirm.
  Also: `wms/control/*` currently maps to the "root" layer of that test (allowed: core, config,
  cli), so control modules that import model/physics/data will fail the layer test until
  `control` gets an `ALLOWED` entry.
* PI evidence rows (`PISession` receipts) are not persisted by this harness (committed and dropped
  each tick); PI command records carry owner "damping" (adapter identity) and are logged as
  `pi-external-u`.
* Commands: SUMO's safe-speed check (speedMode 27 keeps it) can make a CAV slower than commanded
  (8% of direct commands in the random-policy smoke, never faster); issued vs realised is in
  `episode.npz`.
* Pre-existing unit failure, unrelated: `tests/unit/data/test_drive_registry.py::
  test_committed_registry_matches_the_generator_output` (fails on c4/base too).
