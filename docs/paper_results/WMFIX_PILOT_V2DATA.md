# PILOT v2-DATA — MAIN authorization 2026-10-31 15:25

## Resume / binding
- Resumed clean `c4/wmfixmerge` at f373b47e; prior audit/design/implementation and
  failed v1/smoothed-v2 receipts retained. No prior training was relaunched.
- TRAIN_PREPARED_V2 SHA256:
  `84751dc7b0c9c27795810575f45a716b1af587f5dd6c886b93b453f4697419cc`.
- HARNESS_CONTRACT_V2 SHA256:
  `e67850f3eaac8595f35eea90e478c3a265dc5e0a250a11a0531127a3a7948155`.
- WM2 `b8797242...`; existing config/controllers, sources and CRN unchanged.
- Old smoke PID 188758 is absent; its log has only the pack_history warning,
  and `v2/smoke/SHARD_fit973002_v1step0.json` does not exist. Parity is therefore
  UNVERIFIED, not a pass. Preserve that attempt; recover into a fresh directory.
- gpuA first resume probe: 10,575 MiB free, below 14 GiB launch minimum;
  S-WMCALIB processes untouched. Other live wmfix es_probe jobs belong to Track ES.

## Explicit amended route (default behavior preserved)
`pilot_v2 --registered-data-pilot --harness-contract <registered contract>`
dispatches to `data_pilot.py`. It requires the exact prepared/contract/base hashes
above and rejects `--smooth-events`. Without this explicit flag, pilot_v2 still
requires smoothing ON and a passing, bound FD feasibility receipt.

v2-DATA uses the v1 objective's default gradient columns (upstream/crossing/road),
reports source separately, and keeps finer bases OFF. Adam lr 3e-4, seed and
curriculum `[60,30],[120,30],[300,60]` remain the v1 schedule. Only the response
head is optimized; export checks frozen tensors bitwise at EVERY update.
The MAIN 15:00 exclusion is retained: 969432-969439 never enter optimizer cycles
(44 fitting worlds); the registered gate is measured on all 52 TRAIN worlds.

Full panels: steps 0/10/20/30, all 52 groups x two fixed arms x 120/300 s.
Decision uses the UNCHANGED `pilot.decide()` with measured deterministic SD=0,
finite gradients and frozen-bitwise required. No mini-panel substitution, no
early stop, no step-5 checkpoint can satisfy this registered route. Same-gpuA
shards avoid the measured gpuR/gpuA bias from v1. Bounded inference concurrency
increases only with free GPU memory; no peer process is controlled.

PASS continues the v1 curriculum from step 30/its optimizer state in place;
FAIL returns 2 before curriculum. Every update is an immutable checkpoint.
Six-hour cap includes initialization, pilot panels and continuation.

## CPU verification
`python -m pytest tests/unit/test_wmfixmerge_pilot_v2.py tests/unit/test_wmfixmerge_pilot.py -q`
**15 passed**. Covers immutable binding, explicit dispatch/default rejection,
unchanged decide(), missing panels, fitting exclusion, and shard startup bounds.
Focused Ruff check passed. GPU parity/launch/decision remain pending.
