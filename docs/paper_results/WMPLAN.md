# S-WMPLAN — world-model-guided cooperative planner

## Plain-language status
**The approved engineering restart is running its one-episode fail-fast smoke
on the dual-GPU services. There is no complete live efficacy result yet.** The original test stopped on an executor-wiring error. Three primary-arm
attempts reached the 300-second warmup; none ran a post-warmup physics step or
completed an episode. A durable STOP was written immediately when the abnormal
close was detected. All owned v1 local and remote services exited before the restart.

The error was a missing `initial_rows` field between the online planner and the
inherited executor. The observable roster is now attached exactly as in
`coopplan.runner`; the repair is committed as `e492210c` and 30 CPU tests pass,
including the real plant adapter. The stopped immutable run was preserved, and
the repair was dispatched only to the new immutable v2 root. MAIN's 15:55 amendment authorizes one
new attempt per episode; any new fault stops the run and is reported.

Stopped v1 accounting is **900 actual sim-s; 3,600 charged sim-s, with no refund**.
V1 had no strict-arm or group-twin simulation. Both GPUs produced forecasts;
all three completed K8 states (24 futures) are preserved and reused in v2.

FULL-v4 was selected before live outcomes: helpful-state agreement 4/8 vs FULL-v5
2/8, all-12 exact chosen saving +680.308 vs +172.704 veh-s/state. These are
teacher-quality diagnostics, not live efficacy.

## Original registered live design
The exact 199,200 sim-s plan is registered. Complete full-prefix C_G twins require
fixed 2,950 sim-s credit per world-arm: at most one early group can execute, and
none after t600 fits. This is explicitly a budget-limited planner comparison;
raw proposals, executions and resource-forced O0 decisions will all be reported.

## Requested outcomes — not estimable from this failed start
- **Paired J600/J900 and bootstrap CIs:** unavailable; zero complete 1,200-second
  episodes and zero post-warmup scoring exposure. Failed warmups are not assigned
  zero savings and are not substituted with reused MAPPO endpoints.
- **Executed moves:** zero post-warmup executed moves and zero group moves. One
  O0 decision was recorded at t300, but the adapter failed before executing its tick.
- **C_G member-alone twins:** no group execution, reservation or twin simulation;
  no C_G estimate is defined.
- **Safety:** no post-warmup collision/TTC/hard-braking comparison is available;
  the startup failure is not evidence of full-panel safety.
- **SUMO-oracle / S-DISTILL / WM-student comparisons:** no WMPLAN endpoint exists
  to compare. All 24 S-DISTILL MAPPO baseline bytes remain bound and were not rerun.
  The existing SUMO-oracle panel differs, so it cannot supply missing paired outcomes.

## Fault, repair and retained evidence
The first complete decision reached `coopplan.plant.tick`, which requires
`event['initial_rows']` to anchor inherited executor targets. The WMPLAN wrapper
omitted that field; the inherited SUMO-oracle runner supplies it. The resulting
`KeyError: 'initial_rows'` occurred before `frozen_tick` or a controlled physics step.
The episode supervisor's pool shutdown waited for queued work before exposing the
exception. Recovery detected a closed job at 300 without RESULT and wrote STOP;
admission then rejected further worlds. Exactly three jobs consumed warmup time.

The local repair also writes STOP inside a failed episode before its Future
returns to the outer supervisor. The earlier mocked-tick CPU test did not exercise
the missing field; its replacement now calls the real `coopplan.plant.tick`
adapter and mocks only the lower SUMO boundary. A second regression verifies
warmup faults stop before backend closure and before creating a worker pool.
`wmplan/EXECUTOR_ABI_FIX_CPU.json` records 30 PASS, zero new SUMO, no deployment.

`wmplan/FAULT_RECEIPTS.json` binds the original sources, all charges/closes,
recorded decision, all three observable requests/K8 predictions and future hashes,
STOP, traceback and exited ownership identities. `FAULT_REGISTRATION.json` is the
shared registration receipt. Original claims, startup-gate failures, mirror
manifests, transferred files and every forecast remain preserved.

## Engineering restart — APPROVED 15:55 / v2 dispatched
The approved restart keeps the same model, panel, decision rules, credit quotas,
executors and expected-first reporting order under the owner's original cap.
The tighter mathematical bound is:
- New 48 episodes: 57,600 sim-s.
- New group twins: at most 48 x 2,880 = 138,240 sim-s. The unchanged 2,950/world-arm
  quota admits at most one group, and the latest affordable clock is t600.
- New run upper: 195,840 sim-s; plus all 3,600 existing charges = **199,440 charged
  sim-s**, below 200,000. Worst-case combined actual is 195,840 + 900 = **196,740 sim-s**.

The owner delegated approval to MAIN, whose 15:55 amendment is in the shared
registration. It refunds no prior charge and changes no behavioural rule. The
first expected-arm world (984000) is the smoke and counts toward the 24 primary
worlds and 48 total episodes; it is not an extra episode. After its full-episode
integrity check, run the other 23 primary worlds on both GPUs, commit the verified
primary report, then release strict. Completed forecasts are reused. The stopped
v1 root and original proposal remain preserved; v2 uses a new immutable root.
See `wmplan/RESTART_PROPOSAL.json` and `wmplan/restart_v2/APPROVAL.json`.

## Recovered authority and evidence
- Branch `c4/wmplan`, clean initial HEAD `df52de61`, inherited from S-WMTEACH.
- Shared registration: `S-WMPLAN BUDGET APPROVED`, 2026-11-01 13:45;
  maximum 200,000 actual simulation seconds.
- Same fresh DEV panel as S-DISTILL: 984000–984023, 24 worlds, two per cell;
  canonical panel SHA256
  `72776ff122956f021d1caf2db9622674b0cde2a74af8bd9f20c3de5ba0a383a6`.
- Reuse completed S-DISTILL MAPPO episodes from gpuA
  `/root/autodl-tmp/distill20261101_v1`; do not dispatch MAPPO again.
- Predictor/executor contract: `tools/coopplan`, S-WMTEACH's
  `tools/wmteach/runtime.py` and `marginal.py`; K8 complete paired futures,
  H300, decisions every 60 seconds at t300..960, inherited O0..O4 and safety.
- S-WMTEACH FULL-v4 quality is complete on 12 prospectively chosen exact
  states: H300 helpful agreement 4/8 and helpful exact chosen saving
  +1020.461280948498 veh-s/state; all-state chosen saving +680.307521.
- At intake, matching FULL-v5 quality was already running in the peer-owned immutable root
  `/root/autodl-tmp/wmteach_v5quality20261101_v2`. Read completed evidence only;
  never duplicate its forecasts or control its process.
- Before any live outcome, select FULL-v5 source `b2df7458` only if its same-12
  H300 helpful-state agreement AND all-12 chosen saving are at least FULL-v4;
  otherwise select frozen FULL-v4 `22cc32f6`. Preserve both comparison results.
- Exact live budget, immutable sources, baseline hashes, model choice, CPU and
  prefix-integrity gates passed and were registered before the one-use dispatch;
  see `wmplan/FINAL_FREEZE.json`, `CPU.json`, and `DISPATCH.json`.

## Required final evidence
Paired J600/J900 savings (MAPPO minus planner, source-inclusive team delay),
whole-world bootstrap intervals, executed option counts and budget refusals,
complete exact C_G quartets for executed groups, collisions/TTC/hard braking,
and matched-panel comparison to S-DISTILL and S-WMTEACH when available.
S-COOPPLAN's existing oracle panels differ; label those comparisons unpaired.

The inherited exact-quality adapter is roster-only with frozen FIT population,
sampled FIT source priors, and cold motion/controller memory. Its quality is not
qualification of a richer history-conditioned observation adapter or calibration
of empirical K8 win probabilities. Preserve that limitation in live reporting.

## Owner14:48 speed amendment
The expected arm now runs all24 worlds first. Its complete paired efficacy,
safety and group-twin report is independently recomputed and committed before
the strict arm is released, regardless of the primary result. One WM service
uses gpuA and one uses gpuR, with the identical serial teacher-quality function
and byte-frozen model/FIT32 artifacts. Exclusive forecast claims and all SUMO
charges remain on gpuA. The gpuR service uses `/root/wmplan20261101_v1` on the
root filesystem; no inference output goes to its nearly-full autodl-tmp mount.

Local contract gate:29 tests PASS, including primary-only full-panel accounting,
rejection of incomplete/secondary jobs at primary reporting, exact choices,
full-prefix credit quotas and generated remote-script compilation. Host29 tests/
24dual guards and gpuR32FIT bootstrap also PASS; final model/source/budget freeze
was registered before live dispatch. The asset-path/package startup repairs used
identical bytes and the unchanged teacher runtime; original manifests and failed
gate evidence are retained in `wmplan/`.

<!-- WMPLAN-PROGRESS:BEGIN -->
## Current continuation status
- Approved v2 active: 0/48 complete; closed combined actual2580/charged6480sim-s (1 open jobs excluded from closed actual).
- Combined upper199440charged/196740actual; no refund. FULL-v4, expected first on both GPUs; strict waits for committed primary report. No full-panel efficacy conclusion yet.
<!-- WMPLAN-PROGRESS:END -->

## V2 continuation and receipts
- `wmplan/restart_v2/FINAL_FREEZE.json` records the exact pre-dispatch shared
  registration line and source/budget binding; `DISPATCH.json` and
  `GPU_R_READY.json` identify owned services. Original `wmplan/FAULT_RECEIPTS.json`
  and the entire stopped v1 root remain preserved.
- Frozen-host gate:32 CPU tests /24 seed guards PASS. Additional independent
  smoke/report/continuation checks:34 CPU tests PASS, zero new SUMO
  (`wmplan/restart_v2/CONTINUATION_CPU.json`).
- The receipt-only continuation never launches or retries an episode. It checks
  recorded service PID/start-tick/boot identities, verifies every smoke decision,
  executor roster and exact twin, commits the smoke receipt, then releases the
  remaining23 primary worlds. The verified full primary report is committed before
  strict is released, irrespective of efficacy. Any new fault writes durable STOP
  and preserves partial accounting; no automatic retry is available.
- Read-only resume checks: `py -3.13 -B -m tools.wmplan.continuation --gate` and
  `py -3.13 -B -m tools.wmplan.restart status`. Inspect
  `wmplan/restart_v2/CONTINUATION_LAUNCH.json` / `CONTINUATION_ONCE.json` and
  `_out/wmplan_v2_continuation.log` before doing anything else. Do not duplicate the
  dispatch, relay, or continuation.
