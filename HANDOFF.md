# S-RB-RUNB handoff

Updated 2026-10-31 23:54 +08:00. Branch `c4/rb-runb`, base `f0db3c82`.

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

## Outputs / ownership
- Atomic exclusive claims: `E:/agent-work/rb-shared/evalclaims/<variant>.claim`.
- Completion only after all 160 tasks and the frozen scorer succeed:
  `<variant>.done`, with output/report paths and measured key numbers.
- Never signal another process; no SUMO, training, truth assignments or prohibited
  raw data access. No candidate scientific result exists yet.
