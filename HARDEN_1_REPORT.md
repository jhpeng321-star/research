# HARDEN-1 report: S-COOPMWM incidents of 2026-11-02, regression tests and fixes

Branch `c4/harden-1`, forked from `c4/coopnative` at `aadee535`. Worktree `E:/agent-work/c4-harden-1`.
The remote hosts were read only (ls/cat/tail/grep). No SUMO, GPU, signals or remote writes were used. Times are cpuQ local time (+08).

Sources read on cpuQ `/home/ubuntu/coopmwm20261102_warm1`:
- all `*FAILED*.json`, `WARM_GUIDED_STOP*.json` and `STOP.json`;
- `capacity_declines/`: 298 receipts, all ordinary capacity or drain declines;
- tails of every `*.log` that contains a traceback;
- `quota_requests/u004`.

Sources read on gpuA and gpuR: `/root/coopmwm20261102_*/SPOOL_FAILED.json`, `*FAILED*.json`, `SPOOL_STOP.json`, `SPOOL.log` and `RESOURCE_EXTENSION_*.log`.

Line numbers refer to `aadee535`.

## Commits

| commit | content |
|---|---|
| `5d51ed80` | B u4 fan-out stall: `parallel_groups` returns at the first accepted group; the broker serves group 0 of every batch first. MAIN integrates harden-3 `07dd3cd7` plus a breadth-first broker key instead. |
| `347d3d60` | Per-query GPU faults become declines: `plans.family_seeds` skips groups that are not 2-3 members; `query_errors` covers per-query ValueError/AssertionError and OOM; `plan_service` validates requests inside `execute`. |
| `fcb4f1ea` | `warm_rpc` exposes the COMMITTED gates; `WorkerNotReady` leads to per-slot quarantine; the STOP receipt is written once; the GPU side reads the additive resource extension; `extend_register` refuses; duplicate-keyword fix in `warm_runner`. |
| `42233748` | `tools/coopmwm/test_harden1_cpu.py` (HARDEN-1 regression tests). |
| `a0f000d7` | Faults in R's centre helper become declines (no worker death, no R episode hang); centre claims of a dead slot are settled. |

## Incident table

| # | incident | root cause (file:line @aadee535) | still present at aadee535? | regression test | fix commit |
|---|---|---|---|---|---|
| 1 | **09:41 labelling crash** on 4-6 CAV groups. gpuA `concurrent1` SPOOL_FAILED, then `WARM_GUIDED_STOP_V2` 'GPU failed; no inference retry', which stopped B u3 and R u2. | `plans.py:14` (`validate`) raises inside `family_seeds` (plans.py:43-66), called from `mppi_runtime.plan` for post-hoc labels. The broker turned the worker death into a global STOP. | **Partly.** `mppi_runtime.posthoc` (mppi_runtime.py:10) and the `check_worker`/`WorkerUnavailable` path existed. However, `family_seeds` itself still raised for 4-6 members, and it is called by startup qualification (`plan_service.py:72`), `speed2.py:69`, `plan_probe`, `plan_qualify` and `plan_scale`. Also, `query_errors.execute` (query_errors.py:9) re-raised every ValueError or AssertionError, so any per-query validation error still killed the worker. | `test_harden1_cpu.FamilySeedsTests` (2), `QueryErrorTests` (3), `PlanServiceTests`. `test_incident_cpu` fatal-path test updated to keyworded invariants. | `347d3d60` |
| 2 | **gpuA hot-add stall** (14:24): 'missing/nonexact new-worker gate STARTUP_COMMITTED_K4_H300.json'. | `hotadd.py:8-11,20-23` requires 8 gates. The restricted spool RPC `warm_rpc.status` (warm_rpc.py:8-10,15) returned only 6 receipts, so `qualified()` raised although the gate files existed on the worker. It was worked around live with the `hotadd_rpc` wrapper. | **Yes,** for any route authorised on `tools.coopmwm.warm_rpc`. | `HotAddGateTests` | `fcb4f1ea` |
| 3 | **R matching AssertionError** 'R count/regime/group-size/duration gate' (`WARM_R_FAILED_I1` / `WARM_GUIDED_STOP_V3`, 14:43). | The old `warm_runner.py:135` had `assert actual==expected`. | **No.** `warm_runner.py:131-135` now calls `r_matching.report` (best effort, shortfall recorded per stratum). | existing `test_r_matching_cpu.test_missing_stratum_recorded_without_assertion` and 5 others | none needed |
| 4 | **Disk below 2 GiB** (B u4 reserved, zero claims, 14:47). | `tools/coopnative/runner.py:41-47`: the admission guard waits while free space is below 2 GiB. It behaved as designed: it waited and did not STOP. The cause was disk growth on cpuQ; MAIN cleaned caches and gzipped idle directories. | Not a code defect. Residual gap: the wait is silent (no receipt). | none (no code change) | none |
| 5 | **R trust beta collapsed to 0** after u1. R u2-u3 executed 0 plans. | The trust rule plus the pre-verification beta draw: R would never propose again. | **No,** provided the binding carries `sumo_verify.guide_beta=1`. `sumo_verify.py:395-398` enforces beta=1 for B/R under verification (MAIN16:07) and keeps tracked beta for reporting. Without that key the tracked beta is used again. | existing `test_sumo_verify_cpu`, `test_warm_health_cpu`, `test_verify_continuation_cpu` (guide_beta) | none needed |
| 6 | **16:21 beta-transition AssertionError** (`VERIFICATION_BETA_TRANSITION_FAILED`). | `verification_beta_ops.identity` asserts the exact argv of the owner process before it may signal it. The live process argv differed. | This is ops tooling that fails closed before sending a signal, which is correct by design. It is not a training-pipeline defect. | none | not changed |
| 7 | **KeyError 'gpu_deadline'** (16:26, `VERIFICATION_BETA_TRANSITION_FAILED_V2`). | `serve` read `b['gpu_deadline']` from a binding without that key. | **cpuQ side: no.** `resource_deadline()` (verification_beta_ops.py:400-408) and the additive `resource_config.clocks` are applied by incident_recovery, capacity_broker, exact_runtime and final_*. **GPU side: yes.** `warm_spool.py:98` and `plan_service.py:99` read the original immutable `GPU_BINDING.deadline_unix`. Old GPU roots (gpuR `second1`, `concurrent3`) still carry 1793614022 (11-02 18:07), so a normal `python -m tools.coopmwm.warm_spool` restart there would fail on its first poll. | `ExtensionRecordTests.test_gpu_spool_reads_additive_extension_clock` | `fcb4f1ea` |
| 8 | **FileExistsError on immutable WARM_BINDING.json** (16:34 `extend_register`). | `verification_beta_ops.extend_register` (line 106) rewrote immutable bindings in place. It is superseded by the additive `RESOURCE_EXTENSION_1651.json` (resource_config.py), but the function was still callable. | **Yes** (callable). | `ExtensionRecordTests.test_superseded_in_place_binding_rewrite_refuses_before_any_io` | `fcb4f1ea` (refuses) |
| 9 | **16:47 DNS 'Temporary failure in name resolution' escalated to global STOP** by capacity_broker (`WARM_GUIDED_STOP_V4`). | At the time, `Remote.raw` raised a plain RuntimeError that was not retried. | **Fixed for the transport itself:** `warm_transport.py:30` matches the DNS line, exit 255 maps to `SSHConnectionError`, and there are 6 bounded attempts within 300 s. **Residual (present):** after a retry, `Remote.verify` (warm_broker.py:42) raised a plain ValueError when it found the worker dead or unqualified, and that became a global STOP instead of a per-slot quarantine. | existing `test_warm_transport_cpu.test_dns_and_ssh255_are_transport_but_code_faults_still_stop`. New: `BrokerWorkerTests` | `fcb4f1ea` |
| 10 | **gpuA second1 worker died** at 16:29 with `ValueError('complete observable group of two or three required')` in `mppi_runtime.plan -> plans.family_seeds`. | second1 ran an older source in which `mppi_runtime.py:54` called `family_seeds` directly. The root cause is the same as #1. | See #1 (`plans.family_seeds` still raised). | as #1 | `347d3d60` |
| 11 | **B u4 wave 17:09-18:35:** 10 SUMO processes at 0 CPU; 42 searches queued, 6 answered, no sumo_verify launch. | `parallel_groups.py:45-57`: the episode waited for every group of its fan-out (about 4 per episode, 42 searches on 3-4 slots, 600-1100 s each), although the selection is 'first accepted in inherited order'. `capacity_broker.py:85` sorted by state_id hash, so the answers were scattered and no episode was complete. Live evidence from `quota_requests/u004` (A accepted, D declined, a assigned, . unassigned): 986061 `aA.AAA.aD`, 986055 `.A.a..D`. Slow searches: 4-6 member groups (N=max(32,2d)), and only 3 slots because second1 had been dead since 16:29 (#10). | **Yes.** | `test_parallel_groups_cpu` (3 new), `test_capacity_broker_cpu` (1 new) | `5d51ed80` (superseded live by `07dd3cd7` plus a breadth-first key) |
| 12 | 05:18 `WARM_GUIDED_STOP` JSONDecodeError (capacity broker control), then a cascade of 'conflicting immutable spool receipt WARM_GUIDED_STOP.json' in every broker thread. | The broker read a partially written (non-atomic) request. The other threads then tried to write their own STOP. | Read race: **fixed** (30 s grace at capacity_broker.py:81 and atomic `immutable`). Cascade: mostly fixed by an exists check, but a TOCTOU race remained. | (race only) | `fcb4f1ea` (`record_stop` under the mutex) |
| 13 | 05:26 `CAPACITY_REPAIR_B` TypeError 'dict() got multiple values for keyword argument workers'. | `capacity_recovery.verify` combined `**operational` with explicit keyword arguments. | **No** in capacity_recovery (it uses `update`). The sibling at `warm_runner.py:50` (`**operational, awr=`) was still present. | `RunnerBindingTests` | `fcb4f1ea` |
| 14 | 06:13 `MAIN0548_TRANSITION_FAILED` 'command identity changed'. | `quota_transition.py:21`: an identity assert before signalling. | Ops tool, fails closed by design. | none | not changed |
| 15 | 18:06 `RESOURCE_MONITOR_gpu*_1651` ModuleNotFoundError `tools.coopmwm.hotadd`. | The monitor source root `coopmwm20261102_resourcehotadd1` was staged without `hotadd.py`. | Ops staging, not code. | none | not changed |
| 16 | gpuR `concurrent1` 03:07 and `warmmppiqual2/3` CUDA OOM during startup qualification. `concurrent2` 03:36 'immutable GPU source mismatch INPUTS.json'. | The GPU was shared with an idle predecessor (14.78 GiB); a staging hash mismatch. | Operational. A startup failure correctly fails the worker. A per-query OOM now declines (#1). | `QueryErrorTests.test_large_group_cuda_oom_is_released_and_declined` | `347d3d60` (per-query only) |
| 17 | `ACCEL_EXACT_PLAN_*_FAILED` KeyError('mainline.158' / 'ramp.36'). | Diagnostic replay: a plan member had left the network. | It is already recorded per diagnostic (`physics_retry:false`) and is non-fatal, which matches the principle. | none | none |
| 18 | 01:19 `STOP.json` 'Connection closed by 1.15.42.207 port 22' (owned warm bridge). | An SSH drop. | **Fixed:** the pattern is covered in `warm_transport` and `WARM_TRANSPORT_RELEASE`. | existing `test_warm_transport_cpu` | none |
| 19 | (latent, same class as #1/#11) Faults in R's centre helper. | `plan_service` centre branch (line ~111) was not wrapped by `execute`, so a fault killed the worker. `capacity_broker.settle_unavailable` (line 56) skipped centre claims, so R episodes waited forever in `centre_client.trajectory`. | **Yes.** | `CentreDeclineTests` | `a0f000d7` |

## Test counts

All runs were local on Windows in the `rl_driving` env. Commands: `python -X utf8 -m unittest tools.coopmwm.<module>` over every `tools/coopmwm/test_*cpu*.py`, with a local, uncommitted `fcntl` no-op shim on `PYTHONPATH`. A Linux run on cpuQ (the repo's convention) was not possible because the remote hosts are read-only for this task.

| | aadee535 (50 modules) | c4/harden-1 `a0f000d7` (51 modules) |
|---|---|---|
| tests run | 356 | 372 (16 new) |
| failing tests | 37 | 33 |
| new failures vs aadee535 | - | **0** |

- All 33 remaining failures also fail on aadee535 locally, and all are environment-only: the `tools.leverprobe` and `tools.wmteach` packages are absent from this worktree, plus Linux `/proc` and Windows file locking.
- 4 errors seen on aadee535 did not recur. They were flaky under parallel load.
- `test_plan_proposer_cpu` hit the 300 s per-module timeout on both trees. That module is unchanged.
- **A full Linux run on cpuQ is still required before deployment.**
- Every suite I touched passes:
  - test_harden1_cpu 12/12
  - test_parallel_groups_cpu 10/10
  - test_capacity_broker_cpu 9/9
  - test_resource_config_cpu 5/5
  - test_warm_spool_cpu 9/9
  - test_hotadd_cpu 4/4
  - test_warm_transport_cpu 9/9
  - test_r_matching_cpu 6/6
  - test_incident_recovery_cpu 8/8
  - test_incident_cpu 3/6: the 3 errors are the 4/5/6-member subtests that already failed on aadee535 because `tools.leverprobe` is missing.
- **Fail before, pass after:** I ran the new tests against an export of aadee535. 13 of the 16 fail there:
  - u4 stall: 4
  - family_seeds for 4-6 member groups
  - per-query declines: 2
  - plan_service
  - hot-add gate
  - broker quarantine
  - GPU clock
  - extend_register
  - warm_runner TypeError
  - centre decline

  The 3 that pass on both are intentional guards: 2-3 member prototypes are unchanged, invariants stay fatal, and the wait for earlier groups is kept.

## Could not fix / left open (by design or by instruction)

1. `capacity_broker.pending` still raises (global STOP) on a stale malformed request older than 30 s. Not edited, per MAIN's instruction not to change `pending` further. Suggested change: quarantine it into an additive record and skip it.
2. `exact_runtime.supervise` (lines ~95-103): one failed or stopped exact grade still writes a global guided STOP. MAIN's 09:44 rule keeps physics failures global. Under sumo_verify, grades come from the verification branches, so this path matters less, but it needs a MAIN decision.
3. Transport retries are still bounded: after 6 attempts in about 300 s the broker escalates to a global STOP, as MAIN16:51 specified. A per-slot quarantine would be gentler, but that is a rule change.
4. There is no circuit breaker for systematic per-query faults. A code bug that fails every query now produces recorded declines (`broker_events/*query_code_declined*`) instead of a STOP. Arm B would then silently behave like arm A, so monitor those events. No threshold was added (no hand-set thresholds).
5. `resource_config.extension` hard-codes `gpu_deadline==1793894340.0` (resource_config.py:13). A further extension needs a code change or a new additive record type.
6. Ops tools (`verification_beta_ops.identity`, `quota_transition.state`) assert exact argv before signalling. They fail closed by design; not changed.
7. `sumo_verify.verify`: an interrupted claim for the same snapshot raises (episode death) instead of declining. This is safe for new job IDs, but a resumed identical episode would hit it.
8. The disk-guard wait (`tools/coopnative/runner.py:41-47`) writes no receipt, so a stall from low disk is invisible without a monitor.

## Live checks needed (not run: no SUMO or GPU in this task)

- **GPU deployment:** `plan_service.py` is in `warm_service.NUMERICAL`. Its hash changes in `347d3d60`, `fcb4f1ea` and `a0f000d7`, so `adopted_startup_gates` will not match: a fresh worker must run all 8 startup gates. `plans.py`, `query_errors.py`, `warm_spool.py` and `warm_rpc.py` are not in NUMERICAL. To avoid re-gating tonight, deploy `plans.py` and `query_errors.py` alone first. That already fixes #1/#10 for the plan path, because `mppi_runtime.posthoc` and `execute` live there.
- Hot-add through `warm_rpc` with real gate files on a GPU root: check that `status` now returns 8 gate receipts.
- One live centre query and one plan query through the updated `plan_service` on a GPU worker (parity gates PASS, `READY`).
- On cpuQ: a broker restart with the new `capacity_broker.py`, and the WorkerNotReady quarantine path when a GPU slot is down.
