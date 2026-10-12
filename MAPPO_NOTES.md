# MAPPO baseline for WM-S C4 - fair universal MAPPO (branch c4/mappo2, 2026-10-13)

Model-free comparator for the C4 method (docs/WM_C4_DESIGN.md section 3, step 4b): MAPPO trained
directly in SUMO. Everything except the learning method is the C4 code, so the comparison isolates
"analytic gradients through the WM" vs "on-policy SUMO samples".

## What is identical to the C4 policy (fixed in code, not configurable)
- Actor: `wms.control.policy.CavPolicy(lane_head=True)` + `RsuGroupEncoder` at default sizes (the
  sizes `load_policy_fn` rebuilds). Per CAV and 1 s decision: tanh-Gaussian acceleration + masked
  categorical stay/left/right.
- Execution: `mappo.agent.ActorStep` is `PolicyExecutor` step for step: `LaneDwell` clock ->
  `build_observations(CommandMemory.augment(state))` (reports incl. the other CAVs' last command and
  lane intent) -> `group_encoder(obs, state["s_m"], build_rsu_field(state))` -> `lane_feasibility`
  -> `policy(..., lane_mask, noise)` -> `apply_lane_choices` -> `safe_accel` (min with the target-lane
  bound when a CAV changes lane) -> `CommandMemory.record` -> `env.step(accel, lanes=...)`. Unit test:
  deterministic `ActorStep` == `PolicyExecutor.act_with_lane` over 8 steps incl. lane changes.
- Lane rules `[lanes]` = `configs/c4_policy.toml [lanes]` (test asserts equality).
- Reward: `wms.control.reward.team_reward` with `RewardWeights` (incl. fairness CVaR) per 0.5 s
  PHYSICS step, summed over the two physics steps of each control step (as the imagination loop).
  The t+0.5 s frame comes from `CavSumoEnv.history`; vehicle identity per physics step is checked by
  vehicle index; a vehicle that leaves (or whose slot is re-used) is carried at constant speed for
  the reward, and TTC is evaluated on the actual next occupants.
- TTC constraint: per-control-step cost = sum of the 2 physics-step `cost_ttc`; budget 0.01 and dual
  lr 0.01 (= C4 `ttc_budget`, `dual_lr`), projected dual ascent.

## Learning method (MAPPO, CTDE)
- Joint per-CAV action: log pi = log pi(accel latent) + log pi(lane | feasible mask) (0 when only
  stay is feasible); entropy bonus on both heads (pre-tanh Gaussian entropy, exact masked lane
  entropy; coefs 0.01/0.01). Stored per frame: decision state (+ relayed command/intent), lane mask,
  pre-tanh latent, lane choice, old log-probs; observations / RSU field are rebuilt in the update.
  Log-probs are re-evaluated by running the frozen `CavPolicy.forward` with forward hooks on
  `head`/`lane_head` (no actor code duplicated; tested equal to the rollout log-probs).
- Centralised critic `TeamCritic` on the privileged SUMO state; one team reward stream, team GAE
  (0.99, 0.95) broadcast to every acting CAV; return normalisation (Welford); value clipping.
- PPO: clip 0.2, 4 epochs, minibatches of FRAMES (1024; the RSU encoder couples one frame's CAVs),
  KL stop 1.5 x 0.03, grad clip 0.5, lr 3e-4 (actor + encoder) / 1e-3 (critic).
  Cost: `(L_reward + alpha * L_cost) / (1 + alpha)` with a pessimistic clipped surrogate of the
  batch-normalised discounted cost-to-go (no cost critic).
- Exact speed-ups: absent slots and then non-CAV slots are dropped before the update forward passes
  (set functions with masks; stable order keeps the report tie-break; tested).

## Universal training and budget
- Per episode: penetration uniform over {0.05, 0.1, 0.2, 0.4} (`penetration_mode = "uniform"`:
  U(0.05, 0.4)), demand over {2400, 2600, 2700}, seed over the exposed block 4107-4148 (`seeds`; the
  env re-checks every seed), optional `warmup_choices_s`. Noise seed = f(run seed, episode index).
- Budget: charged SUMO-seconds (controlled; `charge_warmup` adds warmup) and env-steps, plus
  simulated seconds incl. warmup and wasted seconds of failed episodes, cumulative and per iteration.

## Gradient-noise estimator (paper: "noisy learning signal")
Every `grad_noise_every` iterations, before the update and at the rollout weights (ratio = 1), the
policy-gradient loss `-mean(A * log pi)` is differentiated separately on the M disjoint frame
minibatches of the update partition (no optimizer step). With g_m the flattened actor+encoder
gradient and g = mean g_m: `pg_grad_mean_norm` = |g|, `pg_grad_norm_mean`/`_std` over |g_m|,
`pg_grad_var_trace` = sum_m |g_m - g|^2 / (M-1) (trace of the covariance of a one-minibatch
estimate), `pg_grad_var_trace_batch` = that / M (full-batch estimate), `pg_grad_snr` =
|g|^2 / pg_grad_var_trace. Also per update: pre-clip actor/critic grad norms (mean, max).

## Workers, robustness, resume
- `concurrent.futures.ProcessPoolExecutor`, `spawn` context (Windows + Linux), one SUMO env per
  process (built once), weights shipped per job. Episode exceptions return failure records; a dead
  worker (`BrokenProcessPool`) or a wave deadline (`episode_timeout_s` x waves) stops only this
  pool's own processes and rebuilds it. A failed episode is re-run once (`retries`) with a new seed
  (logged `failure` rows incl. traceback and the retry). Each worker deletes only the scenario dirs
  its own env created (`<out>/scenarios/pid<N>/`, plain dirs only).

## Spot-instance safety (Tencent spot box can be reclaimed at any time; owner 2026-10-13)
- Every iteration ends with a FULL resumable checkpoint `checkpoints/ckpt_XXXXX.pt` (policy,
  encoder, critic, both optimisers, Lagrange multiplier, return scaler, budget counters,
  episode/seed cursor, numpy/torch/python RNG, `log.jsonl` byte offset, cumulative wall time),
  written atomically (tmp + fsync + rename), then the pointer `checkpoints/LATEST` (json: file,
  sha256, iteration, bytes), then `status.json`. The newest `keep_last` (3) checkpoints are kept,
  plus every `keep_every`-th (10). One checkpoint is ~2.3 MB.
- Resume is automatic when `--out` has checkpoints (`--resume` additionally requires one): the
  LATEST target if its sha256 matches, else the newest checkpoint that loads (a truncated file is
  skipped and listed in the `resume` row). Log rows written after that checkpoint (a lost
  iteration's episode/failure rows) are moved to `log.orphan_<it>_<time>.jsonl` and `log.jsonl` is
  truncated to the recorded offset, so the log always matches the checkpoint. Resume continues
  exactly (tests: straight run == truncated-checkpoint resume == stop + resume, same weights).
- Stop protocol: a `STOP_REQUESTED` file in `--out` is polled between episodes (serial), every 1 s
  while waiting for workers, and before every PPO / gradient-noise minibatch. The in-flight
  iteration is abandoned (never half-saved), the pool's own workers are killed, a `stopped` row is
  logged, `status.json` gets `stopped: true`, exit code 3. A `STOP_REQUESTED` present at start-up
  is from the previous session and is renamed `STOP_REQUESTED.consumed_<time>`.
- `run_mappo.ps1`: the detached wrapper polls the Tencent notice
  `http://metadata.tencentyun.com/latest/meta-data/spot/termination-time` every 5 s (404 = none;
  `SPOT_NOTICE_URL` overrides, `SPOT_WATCH=0` disables, `SPOT_NOTICE_FILE=<path>` simulates a notice
  for drills) and writes `<run>\out\STOP_REQUESTED` on a notice (`<run>\spot_watch.log`); STATE
  becomes `stopped_spot`. Loss = at most the in-flight iteration; the stop itself takes seconds
  (well inside the ~2 min notice).
- `status.json` (every iteration, atomic): status, iteration, episodes, failures, charged and
  simulated SUMO-seconds, env-steps, cumulative wall seconds, last checkpoint (path, sha256,
  bytes), sizes + mtimes of `log.jsonl` / `policy_latest.pt` / `RUN_SUMMARY.json`, mean return. A
  laptop puller reads `status.json`, then fetches only the new checkpoint (verify sha256), the
  actor and the log tail (byte offset).
- Laptop drill (fake env, `-Launcher "cim,startprocess"`): notice file created -> watcher wrote
  STOP_REQUESTED 4 s later -> trainer stopped at iteration 36 (ckpt_00036 + LATEST), STATE
  `stopped_spot`; `-Resume` continued from 36 (stale stop file consumed); `-Mode stop` stopped it
  gracefully at 45.

## Outputs and evaluation
`log.jsonl` (`config`/`resume`, one `episode` row per episode, `failure`, one `iter` row per
iteration: budget, learning curve vs SUMO-seconds, reward components, collisions, lane commands,
PPO stats, grad norms, gradient noise, wall times, SUMO-s per wall-s per worker), `RUN_SUMMARY.json`,
`status.json`, `checkpoints/ckpt_XXXXX.pt` + `checkpoints/LATEST`, `policy_latest.pt` (+
`policy_XXXXX.pt`): the deploy actor in the `c4_dreamer/3` payload schema (`trainer = "mappo"`,
policy, group_encoder, lane_rules) - `wms.control.policy.load_policy_fn` loads it unchanged, so
`python -m wms.control.deploy --controller policy --policy <run>/policy_latest.pt` and the eval
campaign (`--controllers policy --policy-ckpt <run>/policy_latest.pt`, own `--out`) evaluate MAPPO
exactly like the C4 policy. `mappo.eval_bridge.load_policy_fn` also accepts trainer checkpoints
(`ckpt_XXXXX.pt`).

## Commands
- Linux: `bash scripts/c4/run_mappo.sh check|smoke|sumo-smoke|run` (`RESUME=1 RUN_ID=<id>` continues).
- Windows box: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\c4\run_mappo.ps1 -Mode
  check|smoke|sumo-smoke|run|status|stop [-Workers 46] [-RunId <id> -Resume] [-Launcher ...]`.
  `run` is detached (Windows OpenSSH kills the session's process tree on disconnect) by the first
  working launcher of `cim,task,startprocess` (working = the wrapper writes STATE within 30 s):
  CIM `Win32_Process.Create` (on the laptop a Defender ASR WMI-child rule answers "access denied"
  for powershell), a one-shot scheduled task as the current user (S4U, no stored password, no time
  limit, unregistered by the wrapper at the end; NOT exercised on the laptop because it would
  create a task there), or Start-Process (warning: may die with SSH). Run `-Mode run` once on the
  box with `EXTRA_ARGS="--smoke --iterations 50 --max-sumo-seconds 1e5"`, disconnect, reconnect and
  check `-Mode status` before the real run. The wrapper re-applies PYTHONPATH/threads from
  `<run>\job.json`, logs stdout/stderr per session, runs the spot watcher and writes STATE
  (`running|done|stopped_spot|stopped|failed:<rc>`). `stop` = STOP_REQUESTED, then after 120 s
  only the run's own pid tree.
- Direct: `python -m wms.control.mappo.train --config configs/c4_mappo.toml --workers N --out DIR
  [--max-sumo-seconds X] [--iterations N] [--episodes-per-iteration E] [--horizon-s H]
  [--warmup-s W] [--update-threads T] [--set key=value ...] [--resume] [--smoke]`; exit 3 = stopped by request.

## Measured (laptop CPU, 2026-10-13; real SUMO 1.27.1, seeds 4107-4148 only)
SUMO smoke `runs/active/C4-MAPPO2-SMOKE-r2/` (local): `--workers 2`, 2 iterations x 2 episodes,
warmup 300 s + horizon 60 s: 0 failures, 0 collisions, 0 teleports. Per episode 10.6-11.2 s wall =
32-34 simulated s per wall-s per worker (native warmup ~45-60 sim-s/wall-s; controlled window
10.8-14.4 sim-s/wall-s, of which policy + reward ~40-50 ms per decision). Iteration 16.5-17.7 s
(collect 11-13 s, update 4-5 s for 120 frames / ~2750 CAV samples, 4 threads, gradient noise on).
Lane commands 354 issued / 234 completed (SUMO refuses the rest, as for the C4 policy).
Deploy check (`wms.control.deploy --controller policy`, the smoke `policy_latest.pt`, warmup 300,
horizon 60): seed 4107/2700/p0.2: 0 collisions, 6 lane commands issued (0 completed);
seed 4108/2700/p0.4: 0 collisions, 11 issued / 4 completed.
Final code (present-slot compaction in the update + per-iteration checkpoints),
`runs/active/C4-MAPPO2-SMOKE-r3/`, same settings: 0 failures / collisions / teleports; episodes
8.8-11.4 s wall = 31.6-41.0 sim-s per wall-s per worker (controlled window 13.5-15.4); iterations
17.0 / 12.7 s (collect 14.8 / 10.4 s, update 2.1 / 2.2 s incl. the gradient-noise pass, 4 threads);
lane commands 393 issued / 238 completed. Deploy of the r3 actor (seed 4108/2700/p0.4, warmup 300,
horizon 60): 0 collisions, 0 lane commands (its deterministic lane head chose stay throughout).
Update cost (fake env, K = 192, ~126 present, ~31 CAVs per frame): ~8 ms per frame per epoch at
8-16 threads (4 threads ~12 ms, 1 thread ~23 ms); no gain above 16 threads. On real SUMO frames
(~100 present, ~23 CAVs) the whole update was ~18 ms per frame (4 epochs + values + noise, 4 threads).

## 48-core Windows box sizing (estimate from the numbers above)
`-Workers 46`, `episodes_per_iteration = 46`, warmup 300 s, horizon 600 s, `minibatch_frames =
1024`, 4 epochs, `update_threads = 16`, `grad_noise_every = 5`: 27,600 frames per iteration;
collection ~1-1.5 min (if per-worker speed holds under 46 concurrent SUMOs), update ~5-8 min; ~36
iterations and ~4-6 h wall per 1e6 charged SUMO-seconds. The update dominates (single process);
fewer epochs or a data-parallel update are the levers if wall time matters. Measure 2 iterations
(`EXTRA_ARGS="--iterations 2"`) on the box first.

## Limitations / open
- The rollout ActorStep runs on all K = 192 slots (identical to PolicyExecutor); ~35-45 ms per
  decision on one thread could be cut ~2x by present-slot compaction (exact, not yet done).
- The scheduled-task launcher and the real Tencent metadata endpoint are untested (laptop has
  neither a usable WMI launch nor spot metadata); the watcher logs "metadata unreachable" once and
  keeps polling.
- `wasted_sim_seconds` counts the full planned episode for every failure (upper bound).
- No cost critic (zero-baseline cost-to-go); `cost_limit_per_step` shares the C4 budget value.
- Smoke runs are mechanics only (2 iterations), not efficacy evidence.
