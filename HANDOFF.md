# S-WALLSW handoff

2026-11-01 · `E:/agent-work/c4-wallsw` · branch `c4/wallsw`

**Build delivered:** a switchable `wall` controller using the existing C4 v2.2
wall baseline, a SUMO library adapter, and a batched WM rollout callback.
**New SUMO usage: 0 sim-s.** CPU tests only; no remote jobs or training.

Consumers: **S-C1SWITCH2, S-SELMAPPO, S-SELMAPPO2**.

## Commits / files

- `6032e969`: recovered-law/interface plan, `WALLSW_PLAN.md`.
- `715bdbd5`: lifecycle controller and SUMO/WM adapters.
- `1b0581d0`: CPU contracts, native-safe execution correction, additive frozen-runtime
  extension loader, and cross-worktree probes.

Implementation files:

- `traffic_shepherd/c4/switchable_wall.py`: `SwitchableWallController`, the shared
  lifecycle/request object. Wraps the existing `c4.baselines.WallLaw`.
- `tools/c1switch/library.py`: `SWITCHABLE_CONTROLLERS`, `make_handover`,
  `make_wall_rollout`, `install_wall_extension`.
- `tools/c1switch/core.py`: optional expanded library and lifecycle hooks in `Handover`.
- `tools/c1switch/sumo.py`: `Controller('wall', inputs)` delegates to the shared object.
- `src/wms/control/wall.py`: `WallRolloutController`, one shared wall object per batch row.
- `src/wms/sim_bridge.py`: lazy wall factory for source-tree WM consumers.
- `tools/c1switch/wall_probe.py`: repeatable CPU integration probe against an existing tree.
- `tests/test_switchable_wall.py`: 35 new CPU tests including parametrized cases.

## Frozen law and formation policy

The recovered law is **v2.1 closure plus v2.2 merge release**. The wall holds
UPSTREAM, meters inflow INTO the merge, and releases at the **START** of the merge.

Source identity verified using Git blobs:

- `traffic_shepherd/wall_group.py` equals 7a12e126:
  `0a43814695537ee9846a9e34f6a54990db21d285`.
- `traffic_shepherd/wall_group_v22.py` equals bda1a5dc:
  `1181eaafab3dc3c897ad6870dfe7ec51ea0fa97f`.
- `src/wms/physics/wall_exec.py` is identical in this tree, `c4-wmfixmerge`,
  and `c4-rb-int`: `011b1065d02ae86828eca8c64dc64139fdb393c4`.
- ab555c24 records the original synthetic golden parity harness. bf3f330b binds
  historical wall-data eligibility to the frozen v2.1 law.

**Formation policy is the existing C4 `WallLaw` baseline**, exercised by an exact
old-baseline/new-wrapper regression test: closest eligible pair in mainline lanes
0/1, merge-relative `s` in `[-2000, -300]` m, span <=75 m; wall speed
`clip(min(member speed) - 3, 6, 25)` m/s; 30 s cooldown after handback. Nonmembers
use the frozen CIRCLES fallback. These are reused existing constants, with no new
behavioural thresholds. Historical wallregen/campaign arms have their own
registered request policies; the library arm's identity is specifically C4 `WallLaw`.

Assembly retains closure `v_wall + min(2, 0.1 * gap_to_front)` for rear members,
front target `v_wall`, pacing clip `[-3, +1.5]` m/s², alignment within 1 m/s and
6 m span, 60 s assembly timeout, and the existing safety aborts. HOLD lasts until
the existing v2.2 release/abort. Road coordinates are translated by the existing
`+2644.2` convention; `s=0` is merge release. There is no lane-command head.

## SUMO interface / S-C1SWITCH2

Use the existing one-second decision grid at ANY episode time, with the current
CAV roster; the wall owns its own formation, assembly, hold, cooldown and release.

```python
from tools.c1switch.library import make_handover, SWITCHABLE_CONTROLLERS

handover = make_handover(inputs)  # existing MAPPO/ctrl0 checkpoint bindings
handover.select("wall", env)
controller = handover.controller
accel, kwargs = controller.command(env)
state, info = env.step(accel, **kwargs)

# At the next decision, hand over in the usual way:
handover.select("pi", env)
```

Expanded action order is `native, ctrl0, mappo, pi, circles, wall` (wall index 5).
Historical `design.CONTROLLERS` remains the registered five-arm experiment.
New selectors and six-arm branch loops should use `SWITCHABLE_CONTROLLERS` and
the consuming task's six-arm scoring/accounting contract.

`command(env)` returns `[1,K]` NumPy requests (NaN = fallback) and
`{"fallback": "damping"}`. Every write runs through the existing
`CavSumoEnv.step -> CommandOwnership -> direct_target -> slowDown` path.
Switching calls the previous object's optional `stop(now_s=...)`, releases the
sole owner's live CAV commands, restores the identical CIRCLES instance, clears
PI state, and starts a fresh wall when selected. Re-selecting the same name keeps
the same object and assembly/hold state. An unavailable pair runs damping and
continues looking on subsequent decision ticks.

An external `stop` is a **controller lifecycle interruption**, recorded as
`controller_stop` with the interrupted phase. It clears wall memory even during
assembly; it does not counterfeit a frozen-law safety abort or modify the
session's HOLD-only planner-release rule. The Handover performs the actual native
handback. A stopped raw SUMO wall object returns native fallback if called again.

### Additive staging beside frozen e3final / cpuQ wrappers

Use the existing `setup()` and source/checkpoint validation, then call
`make_handover`. The extension loader appends staged package search paths and adds
the lazy wall factory to the already-loaded bridge **in process**. The frozen
runtime's files and source manifest stay on their existing hashes, including
the S-C1SWITCH `b3c031ad...` wrapper binding.

The consuming task stages the changed `tools/c1switch/{core,sumo,library}.py` plus
these **additive** files at their repository-relative paths under the tools root:

```text
traffic_shepherd/c4/switchable_wall.py
src/wms/control/wall.py
```

Bind these extension sources in the new task's source receipt. Existing frozen
`WallLaw`, `wms_glue.wall_executor`, `NumpyWallExecutor`, CIRCLES/PI, environment,
and ownership code are used from the runtime. The in-tree `sim_bridge.py` update
is useful for direct source-tree imports; the loader supplies its added factory
to an older bridge without rewriting that file. This staging path was CPU-probed
against two external worktrees, with their actual model and bridge paths reported.

## WM2 / WM-PHYS / selector interface

```python
from tools.c1switch.library import make_wall_rollout

wall = make_wall_rollout(
    model,
    fallback=frozen_circles_callback,  # fresh existing CIRCLES callback for this future
    start_s=decision_time_s,
)
roll = model.rollout(state, n_steps=120, cav_controller=wall, **rollout_kwargs)
```

The callback returns `[B,K,4]` motion, consumes merge-relative `s_m`, `v_mps`,
`lane`, `present`, `is_cav`, optional `junction`, and optional integer
`vid_index [B,K]`. Present identities must be nonnegative and unique per row.
For fixed-roster WM2 rollouts the default identity is slot index. Supply persistent
identities when admitting/recycling slots. A missing/replaced/de-CAV member aborts
the whole wall rather than transferring membership to its replacement.

Physics clock is 0.5 s, wall decisions are 1 s, and `start_s` is an arbitrary
integer decision time. Every callback invocation advances that clock once;
reuse the object across rollout chunks. Build a new object per independent
sample/episode; `deepcopy` preserves an isolated continuation including timers,
cooldown, roster, event history and held commands. Batch shape/slot layout is fixed.

Default execution takes the minimum of the model's native acceleration and the
wall request, respects the existing `cav_exec_cap_mps2` through `apply_exec_cap`,
and retains fallback lane motion and nonmember motion. Absolute target bounds
use **1 s remaining on the first physics tick, 0.5 s on the second**; applying
a full-second braking target over 0.5 s was caught and corrected by the parity test.

For the frozen **WM2 b8797242+mix** consumer, supply
`execute(state, request, target, active_mask, base_motion) -> [B,K,4]` when using
its specialised inferred-native/parity-motion execution path. That hook receives
the held one-second requests on every half-step, and should route active slots
through the consumer's existing native cap/safety/motion code. Preserve its
existing lane/merge processing around the callback. The loader and callback do
not change the model, checkpoint, mix settings, HDV dynamics, or learned heads.

At selector handover, call `wall.stop(now_s=...)` and replace the callback with
the selected peer. `stop` removes all held wall requests; if a stopped WM wall
callback is invoked, it delegates entirely to its supplied fallback. Per-CAV
selectors must treat the selected wall roster as one group action, with one
writer and atomic membership, rather than assembling independently owned walls
for each member.

## CPU verification

Existing environment: `C:/Users/jpeng/miniconda3/python.exe`, PyTorch `2.14.1+cpu`.
Run in this worktree, with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`:

```powershell
& "C:/Users/jpeng/miniconda3/python.exe" -B -m pytest -q -p no:cacheprovider tests/test_switchable_wall.py tests/test_c1switch.py tests/test_wall_group_closure.py tests/test_command_ownership.py tests/test_c1switch_feedback.py tests/test_c1switch_merge_fast.py
```

**84 passed, 1 skipped** (15.34 s on the final concurrent verification run).
The skipped pre-existing test requires the frozen host checkpoint/runtime gate.
Ruff passed for all changed Python files; `git diff --check` passed.

Coverage includes arbitrary start/stop/restart times in empty/assembly/hold/
cooldown states; both-direction handovers with every library peer; same-selection
continuation; actual ownership, direct-target, PI and CIRCLES execution; single
writer/no HDV commands; stale/missing/mode/allowed-speed safety behavior; member
identity and role changes; independent v2.2 session versus SUMO wrapper versus WM
request/motion parity; NumPy/Torch parity for both frozen laws; once-only merge
release; half-step targets; emergency braking and execution caps; batch isolation;
deepcopy; and the real `WmSModel.rollout` callback interface. Learned MAPPO/ctrl0
outputs are synthetic stand-ins in the handover test, with real wrappers and
the real environment control tick beneath them. The simulator clock is a CPU fake.

Cross-worktree commands (fresh process, no target-tree modifications):

```powershell
& "C:/Users/jpeng/miniconda3/python.exe" -B -m tools.c1switch.wall_probe --runtime-root E:/agent-work/c4-wmfixmerge
& "C:/Users/jpeng/miniconda3/python.exe" -B -m tools.c1switch.wall_probe --runtime-root E:/agent-work/c4-rb-int --wm-phys
```

Both passed: 4 synthetic physics ticks, real target-tree model and bridge,
1 wall formed, finite rollout, CPU, no checkpoint loaded, **0 SUMO sim-s**.
At verification, `c4-rb-int` exposes SUMO merge + LC2013 and lacks
`PhysicsConfig.cf_model`; its receipt is explicitly `wm_phys_complete=false`.
The probe uses that available interface and will include `cf_model='sumo_idm'`
when the CF interface arrives. Full frozen-WM2 checkpoint/mix and completed
WM-PHYS validation remain consuming-host integration gates.

## Simulation budget handoff

No canary was launched. The optional future canary requires the owner-approved
S-C1SWITCH2 allocation: at most **2 worlds / 3600 total sim-s**, charged there
exactly once, including any warmup/prefix/branches. The consuming task owns
allocation and receipts. This build's new SUMO debit is **zero**.
