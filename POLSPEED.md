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
