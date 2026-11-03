# S-RB-RUNB handoff

Updated 2026-11-03 17:27 +08:00. Branch `c4/rb-runb`, base `f0db3c82`.

## Fresh-session terminal reconciliation — 2026-11-03 17:27
- Preserved clean HEAD `33bf902d`; read log/status, common rules, noon schedule,
  cutoff receipt and current INT/EVAL handoffs before the read-only audit.
  The extended deadline was 2026-11-01 11:30+08; supervision and original-round
  publication are complete. INT explicitly assigns round two to `c4/rb2-int`.
- Existing observer confirms B2 unchanged157/160 (53/56/48), no live owned
  supervisor and all three owned workers absent; original FAILED retained.
  gpuR nvidia-smi4406MiB/94%, CPUload16.85. Receipt:
  `runs/rbrunb/OBSERVE_1793698048943922000.json`.
- Existing claims, completed tasks and partial/failed receipts preserved. No
  dispatch, failed-job replay, scoring or process action; no `.done` for the
  incomplete panel. INT's final comparison remains authoritative.

## Fresh-session terminal reconciliation — 2026-11-03 13:28
- Preserved clean HEAD `dfaf3fc4`; read log/status, common rules, noon schedule,
  cutoff receipt and current INT/EVAL handoffs before the read-only audit.
  The extended deadline was 2026-11-01 11:30+08 and supervision is complete;
  INT confirms original-round closure and final publication.
- Existing observer confirms B2 unchanged157/160 (53/56/48), no live owned
  supervisor, all three owned workers absent and original FAILED retained.
  gpuR nvidia-smi4406MiB/96%, CPUload24.97. Receipt:
  `runs/rbrunb/OBSERVE_1793683707467188100.json`.
- Existing claims, completed tasks and partial/failed receipts preserved.
  No additional dispatch, failed-job replay, scoring or process action occurred;
  no `.done` for the incomplete panel. INT's final comparison remains authoritative.

## Fresh-session terminal reconciliation — 2026-11-03 08:20
- Preserved clean HEAD `0ed56b61`; read log/status, common rules, noon schedule
  request, cutoff receipt and current INT/EVAL handoffs before observing.
  The extended deadline was 2026-11-01 11:30+08, already completed; both peers
  confirm original-round closure and final publication.
- Existing read-only observer confirms B2 unchanged157/160 (53/56/48), no live
  owned supervisor and all three owned workers absent; original FAILED retained.
  gpuR nvidia-smi6990MiB/93%, CPUload14.88. Receipt:
  `runs/rbrunb/OBSERVE_1793665193083302800.json`.
- Existing claims, completed tasks and partial/failed receipts preserved. INT's
  final comparison remains authoritative. No dispatch, failed-job replay,
  duplicate scoring or process action; no `.done` for the incomplete panel.

## Fresh-session terminal reconciliation — 23:46
- Preserved clean HEAD `1e98bafd`; read log/status, common rules, noon request,
  cutoff receipt and both current handoffs before the read-only observation.
  The completed11:30 supervision deadline has passed; INT/EVAL confirm closure.
- Existing observer confirms B2 unchanged157/160 (53/56/48), no live owned
  supervisor, all three owned workers absent, original FAILED retained.
  gpuR nvidia-smi1MiB/0%, CPUload10.29. Receipt:
  `runs/rbrunb/OBSERVE_1793547986352514200.json`.
- Existing claims, completed tasks and partial/failed receipts preserved. Final
  comparison remains INT-owned and published; original-round closure remains in
  effect. No new dispatch, retry, scoring or process action was performed.

## Fresh-session terminal reconciliation — 21:30
- Preserved clean HEAD `84997ea0`; read log/status, common rules, existing cutoff
  receipts, noon continuation request and both current handoffs before observing.
  Current time21:30+08 is beyond the completed11:30 supervision deadline; INT/EVAL
  confirm final publication and no remaining original-round work.
- Existing read-only observer confirms B2 unchanged157/160 (53/56/48), no live
  owned supervisor, workers801887/801888/801889 absent, original FAILED retained.
  gpuR nvidia-smi4542MiB/98%, CPUload25.92. Receipt:
  `runs/rbrunb/OBSERVE_1793539804008081000.json`.
- Claims, completed tasks and partial/failed receipts preserved. Original-round
  closure remains in effect; no dispatch, replay, scoring or process action.

## Fresh-session terminal reconciliation — 19:01
- Preserved clean HEAD `4ef1b487`; read log/status, common rules, noon request,
  existing receipts and both current handoffs first. INT and EVAL confirm final
  closure; the requested11:30 supervision deadline has passed.
- Existing read-only observer confirms B2 unchanged157/160 (53/56/48), no live
  owned supervisor, workers801887/801888/801889 absent, original FAILED retained.
  gpuR nvidia-smi3007MiB/0%, CPUload11.48. Receipt:
  `runs/rbrunb/OBSERVE_1793530876215939500.json`.
- Original claims, completed tasks and final publication preserved. No new work
  is pending for this round; shared `RUNB_RESUME_CLOSED_1901.md` records closure.

## Fresh-session post-cutoff verification — 15:55
- Resumed clean HEAD `baeab991`; read log/status, common rules, prior closure,
  noon continuation request and current INT/EVAL handoffs before any operation.
  The extended11:30 evaluation cutoff has already passed.
- Existing read-only `observe.py` confirmed B2 unchanged at157/160tasks
  (53/56/48); no live owned supervisor and all three owned workers absent.
  gpuR nvidia-smi:10901MiB/100%, CPUload11.06. No dispatch or process action.
  Receipt: `runs/rbrunb/OBSERVE_1793519717435297300.json`.
- Windows CIM confirms watcher35616 absent; its last successful STATUS poll was
  11:59:08, consistent with the existing12:00 collection bound. No replacement.
- INT final publication `docs/rbint/noon/20261101T113555+0800/PUBLICATION.json`
  incorporated B2's19complete K8worlds, including all8reserved. Its final report
  `docs/paper_results/WMPHYS_MORNING.md` records reserved B2 ΔCRPS120/300/600
  −75/+58/−2643veh-s and Δ300s breakdown-rate distance +4.06pp versus WM2.
  These are INT's existing diagnostics; full-world CIs remain NA, not qualification.
- Original claims, cutoff/partial/failed receipts and all completed tasks retained.
  No `.done` for the incomplete panel; no replay, new claim or duplicate scoring.
  Shared resume note: `rb-shared/evalclaims/RUNB_RESUME_POST_CUTOFF_1555.md`.

## FINAL — owner cutoff verified; B2 157/160, NO RETRY
- Stayed alive beyond11:30 and independently verified terminal state. Remote
  STOPs11:30:11; final in-flight task preserved; owned GPU supervisor762343 and
  workers801887/801888/801889 all exited by11:33:56. No peer process action.
- Final157tasks (53/56/48),19complete K8 worlds. Missing only world969429 k5–7;
  its k0–4 retained. Verified every157 task hash/identity, completed-shard manifests
  and unchanged source/checkpoint/config hashes. No full-panel score or `.done`.
- Root `/root/rbeval20261031/runb/WM-PHYS-B2`; closure
  `CUTOFF_VERIFIED_1130.json`, original FAILED.json/logs/STOPs retained.
- Shared `evalclaims/WM-PHYS-B2.partial.json` (all157hashes) +failed.json (no retry);
  short `evalclaims/RUNB_FINAL_1130.md`. Local
  `runs/rbrunb/B2_CUTOFF_VERIFIED_1130.json`. INT's existing corrected collector
  owns complete-K8 diagnostic scoring; no duplicate analysis/inference started.
- `tools/rbrunb/close_cutoff.py` performed the real closure verification without
  process actions/dispatch/scoring.09:30 intake freeze and04/08interims retained.
- Late observed polling gap: watcher STATUS remained11:11, while the in-session
 600s observation returned11:33; cause unverified. Independent remote cutoff fired
  on time. Watcher35616 remains bounded to12:00 collection, no replacement launched.

### 10:32 live final-hour checkpoint
- B2 tasks144/160 (44/54/46), all original three workers +supervisor762343 live,
  no terminal receipt. GPU4259MiB/94%,load62.56; boundary resource waits persist.
- Slowest shard0 needs12tasks, completion by11:30 remains uncertain. Existing
  deadline/task-boundary stop retained, no re-sharding/source edit/replay.
- Shared `RUNB_B2_PROGRESS_1032.md` alerts INT's existing final collector; audit
  `runs/rbrunb/OBSERVE_1793500355153587700.json`. No `.done` for partial evidence.
- In-session10min monitoring continues through11:30; no new READY RUNB variant.

## 09:30 freeze — B2 123/160; existing evaluation continues
- New-claim intake closed by explicit09:30 watcher policy; existing B2 retained,
  no new unclaimed RUNB variant published. Evaluation/supervision through11:30.
- 09:30:40 B2 tasks123/160 (37/46/40), original three workers +supervisor alive,
  GPU3853MiB/37%,load35.69, no terminal receipt. No source/claim/dispatch changes.
- `rb-shared/evalclaims/RUNB_FREEZE_0930.md`; independent receipt
  `runs/rbrunb/OBSERVE_1793496640094194000.json`. Continuous in-session10min polls.
- EVAL's corrected baseline/uncertainty notice read. Frozen original B2 scoring
  remains source-bound; do not cite original bootstrap CIs as corrected evidence.
  INT's existing corrected collector owns source-specific matched diagnostics;
  no duplicate CPU analysis or inference started. `.done` still pending160tasks.

## 08:00 interim — B2 87/160, boundary resource wait
- Three original owned workers801887/801888/801889 +supervisor762343 live,
  tasks87 (28/31/28), deadline11:30, no terminal receipt. Ten-minute monitoring
  uninterrupted; watcher07:51:35 healthy, same A2/C2/B2 claimed manifests.
- Independent08:00:23 GPU2963MiB/0%,load52.35; no task advancement since07:50.
  Read-only ps: all three workers hrtimer_nanosleep; published run_wm.py waits
  between tasks while CPUload>24. No source edit, restart or failed-job replay.
- Shared `RUNB_INTERIM_0800.json` +short `RUNB_B2_INTERIM_0800.md`; local
  `runs/rbrunb/OBSERVE_1793491223644376700.json`. No full-panel `.done` or score.
- Session remains alive until11:30, with09:30 new-intake freeze retained.

### 07:09 continuation checkpoint
- B2 tasks75/160 (24/27/24), same three owned workers/supervisor alive beyond
  historical07:00. Verified11:30 deadline1793503800, no terminal receipt or replay.
- Ten-minute watcher last07:01:35, no error; independent audit4169MiB/95%,load26.71.
  Noon extension is active, not just a published request.04interim retained;
  session still alive for08interim,09:30freeze and11:30evaluation cutoff.
- INT now owns C2 on gpuR and FULL-v4 on gpuA; recognized existing claims and
  left those running jobs under their owners. No additional RUNB candidate published.

### 05:07 live progress
- B2 tasks16/160 (shards5/7/4), original3workers +supervisor live. GPU12263MiB,
 100%,load17.13, no terminal receipts; watcher05:01:35 healthy. Every10min
  in-session audits continued. Full-panel completion by11:30 is not assured at
  observed early throughput; notified INT via RUNB_B2_PROGRESS_0507.md.
- Recognized INT-owned FULL-v3-gpu under READY; its shared claim preserved and
  no peer redispatch. RUNB intake list remains A2/B2/C2 with B2 exclusively owned.

## Current 04:16 — B2 inference LIVE (one dispatch)
- Existing noon supervisor762343 admitted B2 at04:15:41 with GPU1064MiB/24%,
  load20.57;3shards x5GiB, immutable source4015b6412cdf, original claim preserved.
- Owned workers**801887/801888/801889**; root
  `/root/rbeval20261031/runb/WM-PHYS-B2`; DISPATCH.json + LAUNCHED_0/1/2.json.
- Independent04:16:04 audit:6716MiB/100%,load16.31; all3workers live; tasks0
  during startup, no terminal receipt. Supervisor retains11:30 deadline.
- Shared coordination: `rb-shared/evalclaims/RUNB_B2_DISPATCHED.md`.
  Existing supervisor owns full160-task verification + frozen scoring; `.done`
  follows only complete verified reports. Ten-minute in-session monitoring continues.

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
