# S-RB-RUNA handoff

Updated 2026-11-01 02:43 +08:00. Branch `c4/rb-runa`, base `f0db3c82`.

## Current: owner-authorized noon continuation VERIFIED
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
