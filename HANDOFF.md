# S-RB-RUNB handoff

Updated 2026-11-01 04:06 +08:00. Branch `c4/rb-runb`, base `f0db3c82`.

## 04:00 interim — supervision alive, B2 awaiting resource admission
- Ten-minute checks continued in-session through04:05:46 (individual immutable
  OBSERVE/POLL receipts in `runs/rbrunb`). Watcher last poll04:01:35, no error;
  INT/EVAL READY order remains A2/B2/C2. New INT changes are not yet READY.
- B2 supervisor762343 alive with11:30 deadline. tasks0, no DISPATCH/workers or
  terminal receipt. Last admission sample5541MiB/95%/load25.20; independent
 04:05:46 sample5557MiB/87%/load23.42. Supervisor rechecks once per minute.
- Shared interim: `E:/agent-work/rb-shared/evalclaims/RUNB_INTERIM_0400.json`.
  No complete candidate evidence/key efficacy numbers exist yet; no `.done`.
- Noon continuation and existing claims/failed-job records remain intact.

## Fresh-session 02:08 owner noon continuation intake
- Preserved clean HEAD `f398d8da`, watcher33076 and B2 supervisor755633.
- Read both current handoffs, STATUS and `INT_NOON_SCHEDULE_REQUEST.md` first.
- 02:09 gpuR direct nvidia-smi: RTX4090,15555/24564MiB,99%,load11.59.
- B2 audit: exact boot/start-tick/cmdline matches supervisor receipt; tasks0,
  no DISPATCH, LAUNCHED, shard or terminal receipt; still WAITING_RESOURCES.
- Owner explicitly extends supervision through11:30, interim04/08 and freeze09:30.
  Existing modules capture07:00; preparing a fail-closed owner-side continuation
  of the unlaunched B2 waiter and owned Windows watcher. Claims, sources, failed
  original B and all completed peer work remain preserved.

## Current 02:12 — owner noon continuation LIVE
- Exact owned Windows watcher33076 replaced by watcher**35616** with explicit
  `--deadline 2026-11-01T11:30:00+08:00 --freeze-at 2026-11-01T09:30:00+08:00`.
- Exact owned gpuR waiter755633 continued as **762343**, deadline11:30. Identical
  immutable code/source hashes, original B2 claim/MANIFEST/LAUNCH preserved.
  Admission + launch locks held; pidfd and boot/start-tick/cmdline checks passed;
  no DISPATCH/shard/tasks existed before or after the continuation. No failed-job
  replay. Supervisor762343's receipt independently confirms deadline1793503800.
- Durable shared receipt: `rb-shared/evalclaims/RUNB_NOON_CONTINUATION.json`;
  remote `WM-PHYS-B2/OWNER_NOON_CONTINUATION{,_REQUEST}.json` and
  `SUPERVISOR_762343_*.json` (exact nanosecond suffix retained on gpuR).
- Local first noon poll02:11:35: B2 LIVE/tasks0. Resource sample16296MiB/100%,
  load16.89, still WAITING_RESOURCES; <=20000MiB admission retained.
- `py -3.13 -B tools/rbrunb/test_runner.py`:9PASS. First Windows continuation
  precheck rejected CIM's microsecond rounding (6x100ns), before any signal;
  exact handle creation tick639290592919104196 then verified and operation passed.
- Session remains alive with ten-minute sleeps/checks through11:30;04/08 interim
  status and09:30 intake freeze supersede the historical07:00 notes below.

### 02:24 live in-session check
- Watcher last poll02:21:35, both handoffs present, same A2/B2/C2 READY ordering.
- Independent read-only audit: supervisor762343 alive/deadline11:30, tasks0,
  no dispatch/workers/terminal receipt. GPU8454MiB/98%,load28.33: still busy.
- `tools/rbrunb/observe.py` records independent handoff + exact supervisor +
  nvidia-smi observations without launching work. In-session `--sleep 600`
  supports ten-minute checks; writes operational04/08 interim receipts once.
- Audit: `runs/rbrunb/OBSERVE_1793471033172223500.json`.
- 02:34 check: same READY order, watcher02:31:35 healthy, B2 tasks0/no dispatch;
  GPU15772MiB/100%,load32.72. New INT recipe explicitly NOT READY pending smoke.
  Continued in-session sleeping; no additional dispatch or replay.

## Current — corrected B2 claimed / WAITING_RESOURCES
- Existing watcher33076 automatically claimed **WM-PHYS-B2** at01:31:32 and
  launched ONE supervisor **gpuR755633**; original B remains withdrawn/tasks0.
- Immutable source4015b6412cdfa2a94168b7095a59287d0fddcd3b; INT reports8x60s
  smoke PASS and five-arm600s canary PASS (FULL;206.03s), repaired MAPPO ABI.
- Root `/root/rbeval20261031/runb/WM-PHYS-B2`; exact manifest in shared
  WM-PHYS-B2.claim. INT complete evaluate_B.py entrypoint preserves future-builder.
- Read-only01:41 audit: GPU15,483/24,564MiB/100%, load9.20; supervisor last
  admission sample15,691MiB/100%/load11.24, WAITING_RESOURCES. No DISPATCH or FAILED.
- A2/C2 assigned RUNA/gpuA; FULL-v1 already INT-owned on gpuA. No duplicates.
- Continues polling ten-minute handoffs while waiting for free gpuR compute.

## Historical 01:30 — original B withdrawn before inference
- INT01:24 reports A's three constructor failures: frozen MAPPO callback requires
  unsupported `policy_native_lanes`. Independently AST-checked B's immutable
  bridge: argument absent, no **kwargs. No DISPATCH.json, LAUNCHED or shard exists.
- Closed ONLY owned waiting supervisor728893 after boot/start-tick/cmdline checks,
  under shared GPU admission lock, using a pidfd signal. Verified exit and zero
  inference again. No peer PID touched; existing watcher33076 stays alive.
- Preserved original claim/runtime/launch/logs. Remote WITHDRAWAL_REQUEST.json
  and FAILED.json explicitly say WITHDRAWN_BEFORE_INFERENCE/tasks0; shared
  WM-PHYS-B.failed.json prevents automatic retries. No .done or efficacy claim.
- INT should remove original B from READY and publish a corrected NEW variant
  with a five-arm constructor gate; waiting for that publication.
- Owner00:44 later schedule (registration5736) noted: FULL freeze09:30, evaluate
 09:30–11:30; this runner's current in-session monitoring remains through~07:00.
- One-time operation: tools/rbrunb/withdraw_waiting_b.py. First attempt failed
  before any signal/write because old remote Python lacks os.pidfd_open; Linux
  syscall wrapper then succeeded with the same fail-closed ownership checks.

## Historical 00:35 — WM-PHYS-B claimed / waiting for resources
- INT explicitly published **WM-PHYS-B / RUNB gpuR** at ~00:34. Atomically claimed
  `E:/agent-work/rb-shared/evalclaims/WM-PHYS-B.claim`; owner S-RB-RUNB.
- Source `b0f66943ec300594fc220d3779fb482c26534b1c`, immutable published B
  entrypoint/runtime and smoke SHA are bound in that claim. Using the complete
  INT entrypoint preserves its future-builder hook; no legacy loader substitution.
- Existing watcher PID33076 retained. Single gpuR supervisor **PID728893** at
  `/root/rbeval20261031/runb/WM-PHYS-B`, launched once via existing recovery-safe
  protocol; `LAUNCH.json` and `SUPERVISOR_*.json` own the dispatch identity.
- Supervisor reports WAITING_RESOURCES: used14,818MiB, utilization100%, load51.93.
  No DISPATCH.json or candidate inference yet. Latest GPU check14,846MiB/100%.
  Admission ceiling20,000MiB, capped5GiB allocators, shared lock preserved.
- B is the only published variant, explicitly assigned to RUNB. Atomic manual
  intake handles this assignment despite the watcher's default first-item RUNA
  reservation; all later monitoring resumes the existing owned claim normally.
- Session stays alive through ~07:00; no completion or efficacy claimed.

## Fresh-session resume 00:13
- Preserved clean HEAD `df8bac33` and existing watcher PID33076 (creation
  00:01:31, exact RUNB watch.py command verified). No replacement watcher or
  evaluation dispatch launched. Its 00:11:32 poll completed successfully.
- Read current INT/EVAL handoffs: CLI published, no READY FOR EVAL candidate.
  Shared evalclaims contains only protocol notes; no candidate claims or done files.
- gpuR read-only audit 00:12:19: RTX4090, 13,898/24,564 MiB, 99% utilization,
  load15.61. Existing owners' processes preserved.
- MAIN 00:10 addendum acknowledged: this session stays alive, sleeping between
  ten-minute checks until ~07:00 while the existing bounded watcher handles
  exclusive claims and dispatch. No duplicate running work.

## Status
- Read `rb_common.md`, owner registration and four WHY reports; evaluation only.
- Polling INT and EVAL handoffs every 600 seconds through 2026-11-01 07:00 +08:00.
- Durable watcher **LIVE PID33076**, started00:01:32; first poll succeeded,
  next00:11:32. Deadline receipt:
  `runs/rbrunb/WATCH_33076_1793462492127996000.json`.
  Launch receipt: `runs/rbrunb/LAUNCH_1793462491906412900.json`.
  Implementation commit `18db346d`; initial scope receipt `64f09924`.
- EVAL harness published; INT has no `READY FOR EVAL` candidate yet. No claim or
  candidate inference launched. RUNA gets the first listed variant; RUNB the next.
- gpuR checked directly at 23:51: RTX4090, 15690/24564 MiB, 100% utilization;
  load 20.69. Existing EVAL WM2 workers preserved. Local NVIDIA driver unavailable.
- RUNB ceiling: aggregate GPU usage <=20,000 MiB; use explicit allocator caps,
  resource admission and a shared dispatch lock before launching candidate work.

## Needed from INT for unattended dispatch
Publish ordered variants under `READY FOR EVAL`, with the exact gpuR entrypoint,
runtime/checkpoint/config paths and smoke-gate receipt. Prefer a JSON manifest
linked in that section (contract in `tools/rbrunb/README.md` forthcoming).
Candidate code must be staged into a NEW immutable directory, preserving live
WM2 sources. The evaluator supports INT's checkpoint-preserving loader callback.
Population/default entrant sampling must be explicitly bound by the candidate.

## Runner implementation / verification
- `tools/rbrunb/watch.py`: immutable handoff snapshots and 600s polling; heading-
  scoped readiness; ordered list OR RUNA-compatible per-variant loader records;
  exclusive claims, peer-claim preservation, continuation of own existing claims.
- `tools/rbrunb/supervise.py`: existing published EVAL CLI only, NEW gpuR candidate
  roots, shared admission lock, up to 3x5GiB allocators, one numerical CPU thread,
  nice19; waits when GPU utilization>=90% or CPUload>=24. No automatic job retries.
- `tools/rbrunb/start.py` detaches the bounded watcher; byte lock prevents duplicates.
- CPU regression suite: `py -3.13 -B tools/rbrunb/test_runner.py`, 9 PASS. Covers
  atomic claims/order, no non-ready or peer dispatch, terminal-failure preservation,
  resource ceiling, prohibited paths, changed candidate and 160-task tamper rejection.
- gpuR code deployed and SHA/AST verified without inference:
  `/root/rbeval20261031/runb/code_e497711348b1d25c`; remote CLI `--help` passed
  before the final loader-search-order refinement, which was AST/SHA reverified.
- Latest gpuR audit 23:59: 14657/24564 MiB,100% utilization; no new inference.
- Protocol: `E:/agent-work/rb-shared/evalclaims/RUNB_PROTOCOL.md`. INT can reuse
  RUNA's loader JSON schema with host gpuR and explicit model_name; no extra
  candidate entrypoint staging needed for that schema.
- Admission closes 07:00; existing owned results are collected until07:30.
  No `.done` for partial/failed work. Live operational status:
  `runs/rbrunb/STATUS.json`; full immutable poll receipts are beside it.

## Outputs / ownership
- Atomic exclusive claims: `E:/agent-work/rb-shared/evalclaims/<variant>.claim`.
- Completion only after all 160 tasks and the frozen scorer succeed:
  `<variant>.done`, with output/report paths and measured key numbers.
- Never signal another process; no SUMO, training, truth assignments or prohibited
  raw data access. No candidate scientific result exists yet.
