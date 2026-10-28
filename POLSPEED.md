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
