# S-RB-RUNB handoff

Updated 2026-11-01 00:00 +08:00. Branch `c4/rb-runb`, base `f0db3c82`.

## Status
- Read `rb_common.md`, owner registration and four WHY reports; evaluation only.
- Polling INT and EVAL handoffs every 600 seconds through 2026-11-01 07:00 +08:00.
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
