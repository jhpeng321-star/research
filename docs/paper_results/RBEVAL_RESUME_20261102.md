# S-RB-EVAL resume reconciliation — 2026-11-02 15:48+08

## Preserved state and authority

Resumed clean branch `c4/rb-eval` at `a663469d`. Git status/history and task
receipts were read first. Reconciled owner program/registration5683, pre-scoring
freeze5697, `RBEVAL_DESIGN.md`, the three requested rationale documents,
WMBREAKDOWN's proposed gate, shared helper README, RUNA/INT runner requests,
and INT's `docs/rbint/FINAL_DELIVERY.md`.

The requested harness, future-builder API/CLI, preregistration and WM2 baseline
already exist. `HANDOFF.md` retains their exact invocation and source bindings.
The requested November1 support deadline and final publication have passed.
During this reconciliation another documentation closure commit, `e4eb8a4a`,
appeared on the same branch. Its handoff additions were inspected and preserved;
this receipt supplements that commit with the observed status snapshots.

## Current read-only checks

- `python -B -m tools.rbeval.ops status` succeeded: all four WM2 shards40/40,
  `FINALIZED.json` records160 tasks/20worlds/800arm-futures. Finalizer log ends
  FINALIZED; support log ends at its11:30 cutoff with REPORT_READY,
  frozen_sources_match=true and report_verified=true. No matching EVAL process
  was listed. gpuR snapshot: load31.72/30.98/27.41 on224CPUs, GPU23878/24564MiB,
  utilization82%.
- `python -B -m tools.rbeval.ops candidate-status` succeeded on both hosts:
  A2=160 complete; FULL-v4=160 complete; B2=157 with retained worker-exit failure;
  C2=102. These match INT's delivered accounting. Earlier failed/superseded
  candidates remain retained. gpuA snapshot: load19.48/19.10/16.06 on208CPUs,
  GPU45083/49140MiB/utilization99%; gpuR95% at this later snapshot.
- Read corrected local `gate.json`: NOT_QUALIFIED, pilot_only=true,
  complete_worlds20, bootstrap_revision=scored-only-singleton-roundoff-v2.
- Shared `RB2_INT_REGISTERED_INFRA_READY.md` describes a separately registered
  INT-owned extension. It is not an unresolved original EVAL harness request.

## Canonical deliverables and verification scope

- Harness: `tools/rbeval/`; exact candidate CLI/API: `HANDOFF.md`.
- Corrected baseline: `tools/rbeval/evidence/reports/wm2_corrected_63cf5d83/`.
- Evidence manifest: `RBEVAL_WM2_CORRECTED_VERIFIED.json`; later independent
  verification: `RBEVAL_WM2_RESUME_20261101T1710_VERIFIED.json` (160task hashes,
  15original report hashes,7200BIC decisions,70488point-data rows).
- CI correction: `RBEVAL_UNCERTAINTY_CORRECTION.md`; original CIs remain retracted.
- Integrated comparison: `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md`;
  manifest: `E:/agent-work/c4-rb-int/docs/rbint/noon/20261101T113555+0800/PUBLICATION.json`.

This resume performed status/receipt checks only; existing independent
verification and21-test PASS receipt remain the evidence. No forecast, score or
test was regenerated. Completed jobs, immutable sources, peer ownership and
separately registered follow-up work were preserved.
