# S-RB-RUNA handoff

Updated 2026-11-03; closure reconciled02:56 +08. Branch `c4/rb-runa`, base `f0db3c82`.

## Fresh-session reconciliation — November3 02:56 +08
- Clean HEAD `efa6ca69` preserved. Current INT/EVAL handoffs confirm the original
  November1 noon program is closed; existing READY entries are terminal or owned.
- Documented owner-side continuation already covered interims04/08, freeze09:30
  and supervision11:30 on November1. No pending RUNA evaluation remains.
- A2 remains160/160 complete; B2 terminal157 and C2 terminal102/INT-owned.
  Existing claims, completion notes, outputs and verification receipts preserved.
- Fresh read-only gpuA: RTX4090D,44457MiB used/4054free/41%; recorded RUNA
  workers/supervisor/monitor and local watcher PIDs absent. No process action.
- Receipt: `docs/rbruna/RESUME_CLOSED_20261103_0256.md`. Prior polling-gap caveat
  retained; no dispatch, failed-job replay, replacement watcher or scoring.

## Fresh-session reconciliation — November2 22:53 +08
- Clean HEAD `1992fc41` preserved. Current INT/EVAL handoffs and owner continuation
  confirm the November1 noon program is closed; no pending RUNA evaluation.
- A2 remains160/160 complete; B2 terminal157 and C2 terminal102/INT-owned.
  Existing claims, completion notes, reports and prior verification are preserved.
- Read-only gpuA: RTX4090D,45721MiB used/2790free/99%; recorded RUNA worker,
  supervisor, monitor and local watcher PIDs absent. No process action taken.
- Receipt: `docs/rbruna/RESUME_CLOSED_20261102_2253.md`. Documented November1
  continuation already covers interims04/08, freeze09:30 and supervision11:30.

## Fresh-session reconciliation — November2 20:28 +08
- Clean HEAD `f73cf92f` preserved. Current INT/EVAL handoffs confirm the original
  November1 noon program remains closed; existing READY entries are terminal or owned.
- Owner-side continuation already documented through November1 11:30. A2 remains
  160/160 complete; B2 terminal157 and C2 terminal102/INT-owned remain preserved.
- Read-only gpuA: RTX4090D,1MiB used/48510free/0%; registered RUNA worker,
  supervisor, monitor and local watcher PIDs absent. No pending RUNA evaluation.
- Receipt: `docs/rbruna/RESUME_CLOSED_20261102_2028.md`. No dispatch, scoring,
  replacement watcher, replay or process signal; existing claims/reports unchanged.

## Fresh-session reconciliation — November2 16:00 +08
- Clean HEAD `db121dbf` preserved; current INT/EVAL handoffs and owner schedule
  confirm the November1 noon program is closed. No new unclaimed work remains.
- Read-only gpuA:45083MiB used/3428free/98%; registered RUNA worker/monitor and
  local watcher PIDs absent. No dispatch, scoring or process action taken.
- A2 remains160/160 complete; B2 terminal157; C2 terminal102 and INT-owned.
  Existing claims, `.done`, reports and prior verification receipts preserved.
- Receipt: `docs/rbruna/RESUME_CLOSED_20261102_1600.md`. Original closure and
  corrected final-publication references follow below.

## Current: A2 COMPLETE; November1 noon program CLOSED
- Fresh resume preserved HEAD61a4d155 and all existing jobs/claims/artifacts.
  Audit6fd6b4e4 and verificationd2b1743e committed sequentially. November1's
  09:30freeze/11:30supervision deadline has passed; INT/EVAL final handoffs close
  this program. No new dispatch, scoring, replacement watcher or process signal.
- **WM-PHYS-A2:160/160 tasks,20complete K8 worlds,five arms**. Own peak GPU
  **3272MiB**, below18500guard. Existing shared `WM-PHYS-A2.done` preserved.
  Outputs: `gpuA:/root/rbeval20261031/runa/WM-PHYS-A2/reports/complete`.
- Read-only verification:160task hashes,3identities,15final+30interim report
  hashes,26preserved artifacts PASS; local/remote RESULT match, no FAILED marker.
  Receipt `docs/rbruna/RESUME_CLOSURE_VERIFIED_20261102.json`.
- Interims04:00:07/08:00:24 November1 contain4/10complete worlds. All registered
  A2 worker/supervisor/monitor incarnations are no longer live; local watcher
  PIDs15824/44796/33696 absent. Saved polls extend through11:34:16; max gap
  ~29m33s, so uninterrupted ten-minute polling is not established at the end.
- Original gate **NOT_QUALIFIED**. Original `.done` bootstrap intervals are
  historical; corrected final comparison/provenance belongs to INT:
  `E:/agent-work/c4-rb-int/docs/rbint/noon/20261101T113555+0800/PUBLICATION.json`
  and `docs/rbint/FINAL_DELIVERY.md`. Full-world CIs unavailable for singleton
  cells. INT reports A2/B2/C2/FULL-v4 fail the registered point switching rule.
- C2 is **INT-owned**, terminal102tasks/12complete worlds; B2 peer-owned terminal
  157tasks/19worlds. All8reserved worlds complete for each; never redispatch.
- Current shared closure notice:
  `E:/agent-work/rb-shared/evalclaims/RUNA_CLOSED_20261102.json`.
  Earlier `RUNA_NOON_CONTINUATION.json` is an immutable historical02:43 snapshot.
- gpuA current observation44959MiBused/3552free/78%; no new allocation.
  Full audit `docs/rbruna/RESUME_NOON_20261101.md` (November2 closure sections).

## Historical02:43: owner-authorized noon continuation VERIFIED
- Preserved implementation HEAD393bf36a and recovered already-executed handover;
  no deployment replay, duplicate watcher, inference launch or peer signal.
- Live local watcher **15824** replaces44796; ten-minute INT/EVAL polls through
  02:34:43 succeed. Admission freeze **09:30**, supervision through **11:30 +08**.
- Active **WM-PHYS-A2**, original workers335002/335003/335004; noon monitor
  **343598** accepted original supervisor334996's lock at02:14:25. Original
  supervisor is absent. All26 preserved artifact hashes reverified at02:43.
- A2 **22/160 tasks** at02:41, then two complete K8 worlds observed. No efficacy
  result yet. Monitor owns **04/08 interim** and frozen full-panel scoring once.
- gpuA total5160MiB at02:41; own A2 allocation2164MiB at02:43 (guard18500MiB).
- **C2 next**, unclaimed; B2/FULL-v1 peer claims and failed A/C retained.
- Full receipt: `docs/rbruna/RESUME_NOON_20261101.md`; local ignored dispatch:
  `runs/rbruna/WM-PHYS-A2/NOON_DISPATCH.json`. Shared current coordination:
  `E:/agent-work/rb-shared/evalclaims/RUNA_NOON_CONTINUATION.json`.

## Earlier 00:29 resume / launch prerequisite (historical)
- Preserved previous commit `d3d8af25` and all ignored run receipts. Worktree was
  clean. Existing watcher PID **44796**, created 23:58:14, remains live with exact
  command `python.exe -B -u tools/rbruna/watch.py`; no second watcher launched.
- Its original ten-minute cadence is verified through **00:28:14**, no poll errors.
  Admission/deadline remains **2026-11-01 07:00 +08:00**. No claims, dispatches or
  candidate predictions exist; current INT/EVAL handoffs have no READY section.
- gpuA 00:26 read-only check: RTX4090D, **1 MiB used /48,510 MiB free, 0%**, no
  compute PIDs, load **5.68/5.61/7.11**. No processes signalled.
- Found an actual contract gap: INT `tools/rbint/evaluate.py` passes
  `future_builder=candidate_future` and recipe reads `WMPHYS_VARIANT`; EVAL's
  published `run_wm.py` currently has neither the callback argument nor its wiring.
  **INT/EVAL: reconcile that runner BEFORE publishing READY / a launch spec.**
- Commit **a7205655** adds RUNA forwarding of the published future builder and
  A/B/C recipe selector, binds builder source, and refuses a legacy harness before
  calling it. WM-PHYS preflight requires both spec fields. Existing watcher reads
  the updated remote helper at dispatch, so no restart or duplicate job is needed.
- Updated shared `evalclaims/RUNA_PROTOCOL.md`; request at
  `E:/agent-work/rb-shared/RUNA_HARNESS_REQUEST.md`. Publish ordered `.runa.json`
  links after the engineering smoke and callback-capable EVAL runner are ready.
- Verification: `py -3.13 -B -m unittest tools.rbruna.test_runner -v`: **6 PASS**,
  0.328s; new contract test verifies B selector/custom futures and rejects old
  harness execution. No scientific evaluation, SUMO or training run at resume.

## Status
- Read `rb_common.md`, owner registration (5683/5708), all four WHY reports,
  EVAL design and published CLI. No READY FOR EVAL variant at first poll.
- gpuA verified 23:51:48: RTX4090D, 1/49140 MiB, 0%; load6.94.
  Initial SSH closed; nonmultiplexed SSH succeeded. Local NVIDIA unavailable.
- Durable ten-minute watcher implemented, deadline 2026-11-01 07:00 +08:00.
- RUNA claims first unclaimed variant in published order using exclusive-create
  `<variant>.claim`. Claims and failures are retained; never retry scientific jobs.

## INT / EVAL: dispatch contract requested
Please link a JSON launch specification in your `READY FOR EVAL` section, e.g.
`- WM-PHYS-A: E:/agent-work/rb-shared/evalclaims/WM-PHYS-A.runa.json`.
This lets the detached runner execute newly published variants without another
chat turn. It requires a staged gpuA runtime and the exact approved INT loader.
See `E:/agent-work/rb-shared/evalclaims/RUNA_PROTOCOL.md` for fields.
RUNA will stage a NEW immutable copy of the published EVAL harness, run three
5 GiB inference shards (aggregate own-process guard 18,500 MiB, under20GB),
then invoke its frozen scorer. No duplicate WM2 run.

## Results
- `py -3.13 -B -m unittest tools.rbruna.test_runner -v`: 5 PASS /0.55s,
  covering concurrent claims, READY ordering, plain-text peer claims, all160
  completion hashes/corruption, and refusal to signal a reused PID.
- `py -3.13 -B tools/rbruna/watch.py --once`: initial poll23:56:35, no READY specs.
- gpuA fixed-input audit23:56: existing wmcalib starts/expected hashes, policy190,
  MAPPO binding, baseline-controller runtime and python dependencies all present.
  RUNA output root is NEW; no copying/extraction of existing recordings needed.
- No candidate inference or efficacy result yet. No SUMO/training/peer signals.
