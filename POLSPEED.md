# S-POLSPEED — candidate A policy-training speed (engineering, zero SUMO)

Status: IN PROGRESS (incremental commits; gpuA rental may end 2026-10-28 23:09 UTC+8).

## 1. Non-invasive profile of the live run (worker 428853)

py-spy is installed (`~/.local/bin/py-spy`) but `py-spy dump --pid 428853` fails with
`Permission denied (os error 13)` on `process_vm_readv`: the container has no
CAP_SYS_PTRACE and `kernel.yama.ptrace_scope=1`, so non-descendant processes cannot be
read. The live run was NOT signalled or attached. Instead the run's own owned trace
`POLICY/A/memory_execution.jsonl` (LONGMEM `_trace_memory`, wall-clock stamped) gives an
exact per-phase attribution. Update 183 (90 s horizon, 1 start, K=8 futures, 180 physics
steps per future, entrants ON):

| phase | seconds | share |
|---|---:|---:|
| full no-grad forward, all 8 futures (`full_forward_start->done`, 1548 tape draws) | 284.4 | 28.6 % |
| 8 x replay forward of ONE active future with grad, checkpoint segment 8, saved-tensor spill | 8 x ~48 = 384 | 38.6 % |
| 8 x VJP of that future (= checkpoint recompute of all 23 segments + backward) | 8 x ~40.5 = 324 | 32.6 % |
| between updates (optimizer, critic, metrics, independent_J, start sampling) | ~18 | 1.8 % |
| `wall_time_s` logged by CLI | 994.6 | |

What one update does: max_starts=1, so 1 start (plus its counterfactual rows) x K=8
sampled futures x 180 WM steps = 1440 future-steps, and every future-step is simulated
THREE times (no-grad full forward, grad replay forward, checkpoint recompute in backward).
~0.2 s per future-step with GPU utilisation ~0-13 %: the rollout is host/launch-bound
(Python + per-op dispatch + host syncs), not GPU-bound. `imagination_microbatch=1` is a
no-op here (1 start = full batch); the "one future at a time" replay (offload_futures)
plus segment checkpointing is what triples the simulation.

## 2. Where the per-step time goes (py-spy on OWN benchmark processes)

Own benchmark workers call `prctl(PR_SET_PTRACER_ANY)` on themselves, so py-spy works
on them (not on any other process). Profiles: `tools/polspeed/prof/*.raw`
(`python tools/polspeed/agg.py <raw>`). Startup imagination (no-grad, same `imagine`):
`rollout/step` 64 %, post-rollout reward 30 % of which `_soft_cvar` 23 % SELF. The reward
is evaluated ROW BY ROW on purpose (`nativeinit.paired_reward`: row-local reduction
geometry so identical rows give identical J) and each call runs a 32-step detached
bisection (~360 tiny CUDA launches). Under LONGMEM every aten op additionally goes
through the Python `_RandomTape.__torch_dispatch__` (39 % inclusive in the full-forward
sample). Remaining time: MOBIL lane logits (9-16 %), merge-pivot geometry, srcgrad queue
surrogate (per-step, per-entrance small-op chains). Everything is launch/Python-bound.

## 3. Memory facts measured tonight

* `imagination_checkpoint_segment = 0` + spill OFF (one future graph live, no recompute):
  **OOM at cap 0.30 (14.21 GiB allowed, 13.79 GiB allocated)** during the FIRST future's
  replay forward (`failed_attempt1/FM0_oom_cap030`). One 90 s future graph alone needs
  > 14 GiB, so an all-8-futures single graph (no checkpoint) needs > ~110 GiB: the
  "batched K8 with grad, no recompute" variant cannot fit in 0.5 x 48 GB, nor in 48 GB.
* Segment checkpointing therefore stays; tested fast variants keep segment 8.

## 4. Exact variants measured on the SAME update (181, seed 965140, resume c01.a3)

Determinism check: the live run and the earlier dry run produce bitwise-identical update
181/182 metrics, so every variant is compared against `DRY_RUN/A/updates.jsonl` with
`tools/polspeed/parity.py` (all 158 numeric leaves: losses, grad norms, per-module grads,
credit, residual stats, J; timing/memory keys excluded).

| variant (all segment 8, microbatch 1, offload_futures ON) | cap | s/update | peak alloc | parity |
|---|---:|---:|---:|---|
| live / dry reference (spill ON) | 0.45 | 1009-1018 | 11.9 GB | reference |
| F: + existing bit-identical env flags (PIVOT_DIAG_OFF, FAST_MERGE_LANE, FAST_CAV_DRIVE, FAST_LONG_HISTORY) | 0.30 | 1022.7 (3 procs sharing GPU) | 11.89 GB | **bitwise identical (0 diffs)** |
| FS8: F + saved-tensor spill OFF | 0.25 | OOM in first VJP (needs > 11.84 GiB) | - | - |

F flags give no measurable gain (the hot spots are elsewhere). Spill OFF shortened the
first future's replay forward 51.1 -> 42.8 s (-16 %) but needs more than 0.25 of the GPU.

## 5. Restart procedure (main decides; nothing was restarted or stopped by S-POLSPEED)

* There is NO resumable state of candidate A: the CLI writes `policy_latest.pt` only on an
  own-run `STOP` file (checked at the start of each update) or at phase end, and
  `policy_native_best.pt` at native-panel selections (every 10 updates). The run was
  SIGTERMed after update 185, so the only files are `POLICY/A/policy_native_best.pt`
  (= initial c01.a3 actor) and `updates.jsonl`. `policy14_recovery.load_recovery` only
  accepts an operational recovery whose cause is a recorded CUDA OOM, so a speed restart
  is a FRESH phase, not a resume. For future stops use `touch <output>/STOP` (graceful:
  saves `policy_latest.pt` after the current update) rather than a signal.
* Fresh restart from C1 64f70af4 on a different WM: `tools/c4prep/launch_a.py prepare`
  builds `resolved_A_MAIN1500_<commit>` from `tools/c4prep/common.py` constants
  `WM`, `WM_SHA`, `WM_CONFIG` (now c03.a2 / 41e18f09 / c4_wm_loop14_r16_ft.toml) and
  `ACTORS[3]` (source actor). Change those constants (and the BOUNDARY `wm` entry they
  feed; the CLI refuses a WM whose sha differs from `BOUNDARY.json`) for the adopted WM,
  then the CLI's `--wm-ckpt/--wm-config/--resume` follow the same argv as `MAIN_LAUNCH`.
* To use the speed flags: (a) env only, no config change: `WMS_SPEED_GRAPH_CVAR=1
  WMS_WMSPEED_DEDUPE_MOBIL=1 WMS_WMSPEED_MOBIL_LAZY_SAFE=1` (+ the four F flags) with a
  runtime built from branch `c4/polspeed`; (b) spill OFF additionally needs
  `[trainer] imagination_offload_saved_tensors = false` and a cap > 0.25, which today
  fails the launcher's hard-coded checks: `tools/c4prep/launch.py` `flags_ok`
  (requires microbatch 1 / segment 8 / both offloads True / audit 25 and peak <= 0.45)
  and `launch_a.py` asserts (segment 8 from `MEMORY_EXECUTION.json`). Amend
  `MEMORY_EXECUTION.json` (objective_change false) and those two checks together.
