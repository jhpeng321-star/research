# S-RB-EVAL resume reconciliation — 2026-11-03 01:35+08

## Preserved state and task completion

Resumed clean branch `c4/rb-eval` at `f402d36f`. Git status/history and task
receipts were read first. Reconciled original program/registration5683,
pre-scoring freeze5697, future-builder addendum5753, frozen evaluation design,
the three requested rationale documents, WMBREAKDOWN's proposed gate, shared
helper README, RUNA harness request and INT's final-delivery receipt.

The harness, future-builder CLI/API, preregistration and complete WM2 baseline
already exist. The requested November1 support cutoff and final publication
have passed. Later central registration6072 assigns both GPUs to S-COOPMWM;
separately registered follow-up evaluations retain their own ownership.

## Current read-only checks

- `python -B -m tools.rbeval.ops status` succeeded: four40/40 WM2 shards,
  original `FINALIZED.json` with160 tasks/20worlds/800arm-futures. Finalizer log
  ends FINALIZED. Support's last11:30 entry is REPORT_READY, with frozen sources
  matching and report_verified=true. No matching EVAL process was listed.
  gpuR snapshot: load49.41/33.38/31.67 on224CPUs; GPU24048/24564MiB,83%.
- `python -B -m tools.rbeval.ops candidate-status` succeeded on both hosts:
  A2 and FULL-v4=160 complete; B2=157/FAILED; C2=102. Earlier failed and
  superseded runs remain retained. These counts match INT's delivered accounting.
  gpuA snapshot: load17.03/17.18/17.65 on208CPUs; GPU41439/49140MiB,82%.
  gpuR utilization was97% at this later snapshot.
- Read corrected local `gate.json`: NOT_QUALIFIED, pilot_only=true,
  complete_worlds20, bootstrap_revision=scored-only-singleton-roundoff-v2.
- Confirmed `run_wm.main(model_loader=None, identity_sources=(),
  future_builder=None)` and callback source-identity binding in current source.
  Exact invocation remains in `HANDOFF.md`; the old RUNA blocker is resolved.

## Canonical delivery and verification scope

- Harness: `tools/rbeval/`; invocation and candidate API: `HANDOFF.md`.
- Corrected baseline: `tools/rbeval/evidence/reports/wm2_corrected_63cf5d83/`.
- Evidence: `RBEVAL_WM2_CORRECTED_VERIFIED.json` and
  `RBEVAL_WM2_RESUME_20261101T1710_VERIFIED.json` in this directory.
- CI correction: `RBEVAL_UNCERTAINTY_CORRECTION.md`; original CIs remain retracted.
- Integrated comparison:
  `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md`.
- Final candidate accounting:
  `E:/agent-work/c4-rb-int/docs/rbint/FINAL_DELIVERY.md`.

Existing independent verification and21-test PASS receipts remain applicable.
This resume checked status, receipts and source interfaces only; no forecasts,
scores or completed tests were regenerated, and no processes were signalled.
