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
**16 passed**. Covers immutable binding, explicit dispatch/default rejection,
unchanged decide(), missing panels, fitting exclusion, and shard startup bounds.
The recovered parity check requires EXACT original per-group values, not rounded
agreement (449.62744140625 / 302.890625). Focused Ruff check passed.

## Staging / one-shot dispatcher
The 30-file local/staged tree comparison initially differs in raw hashes solely
because the existing staged tree mixes LF/CRLF. All normalized source hashes
match: aggregate `597fcc80b61c992a2fe9f353604cfd572cbcf8741a9e90eea4a333ffb6fb26fa`.
Preserve staged bytes, which the registered contract already binds. The staged
raw aggregate is `303830de3670bdef8585863a8b7b8c32bdfff588f29a92dae4e9263231bfac7f`.
The launcher rechecks BOTH aggregates before dispatch and again before training.
New driver files are separately hash-bound; full-panel receipts must bind the
exact checkpoint/prepared/base and every group/horizon/arm exactly once.

`tools.wmfixmerge.launch_data_pilot --execute` creates an exclusive DISPATCH.json,
waits for >=14 GiB free before the recovered smoke and before pilot launch,
runs smoke into NEW `v2data/smoke`, and requires exact original-v1 parity.
Only then does it create NEW `v2data/pilot` and launch with timeout 21570 s,
trainer allocator 0.16, same-gpuA inference allocators 0.08 (max four processes,
free-memory-limited). EVENTS.jsonl/STATUS.json persist waits, PIDs, commands and
decision. GPU parity/launch/decision remain pending.

## Dispatch receipt (2026-10-31 15:25:41 +08, gpuA clock)
- Dispatcher PID **197030**, `/proc` start ticks **881175520**, PPID1/SID197030.
- State **WAIT_GPU_MEMORY**, 10,843 MiB free <14,336 required at dispatch.
  No smoke or pilot is claimed launched. Exclusive `v2data/DISPATCH.json` exists;
  NEVER launch a second dispatcher while this receipt/job is unresolved.
- Git-archive payload from 6a53549c SHA256
  `85477f783c99ccea68107f5fa53f0a846da4f43d1df0275c751c95c2feac5359`;
  local/remote archive hashes match. Previous driver bytes saved in
  `v2data/pre_data_driver.tar`; no existing implementation byte was replaced.
- Staged drivers: pilot_v2 `5ff51662...`, data_pilot `315e0f61...`,
  launch_data_pilot `2e26b007...`; staged pure v2 tests **11 passed**.
  (First combined remote test invocation found the old v1 test file unstaged;
  reran only the staged v2 suite. Local combined suite remains 16/16.)
- Dry-run verified exact input/tree/driver bindings. Append-only registration
  records this dispatch. `STATUS.json` is the current state; `EVENTS.jsonl`
  preserves every transition. Read those before any recovery action.
