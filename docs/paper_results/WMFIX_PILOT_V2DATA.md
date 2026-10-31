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

## Fresh-session reconciliation, 2026-10-31 17:06 +08
- Clean worktree at df70e103. PID197030 is absent; no launch_data_pilot,
  pilot_v2, data_pilot or panel_eval_v2 process remains on gpuA.
- Authoritative STATUS is FINISHED/exit1, not an unexplained dispatcher death.
  Parity PASSED exactly (PARITY.json smoke SHA f9ea73ef...):
  449.62744140625 / 302.890625, same original-v1 values and bound prepared data.
- EVENTS: parity started epoch1793434364; passed1793436331; pilot timeout
  PID214854/start881654564 launched at1793436331 with16599MiB free;
  finished1793437373. Four finite-gradient, frozen-bitwise updates were saved.
- BLOCKED before a gate decision: SHARD_step0_full_s0.log raises
  `entrant_control needs an explicit feedback controller for direct/mixed arms`
  in grouped_feedback/inferred_timeline. No full step0 panel or step10 exists.
  This is an infrastructure failure, NOT measured pilot-gate FAIL.
- Original v2data outputs/checkpoints remain immutable. No blind redispatch:
  diagnose WMDATA controller binding, then recover in a new output directory.
  gpuA free9428MiB at17:06 (<14336); all peer jobs untouched.

## Infrastructure recovery (MAIN17:05 re-arm; registered before dispatch)
- Root cause reproduced: WMDATA `ctrl0` is a direct arm, but Runtime retained
  only ours_c1/coopoff_c1 specs. Its alias was absent. Recorded CTRL0 hash equals
  ours_c1 `64f70af4...`; bind that exact algorithm, never extrapolate direct actions.
- Future curriculum also needs WMDATA speed-MAPPO `d9042cc5...`, not the old
  DESIGN's `aecd7d70...`. The already-staged SPEEDHEAD EXPORT_BOUND checkpoint
  has the exact TRAIN recording hash. Bind it, retaining the original FIT specs.
- Runtime now fails early on any missing/mismatched direct-policy identity.
  New entrant_binding.py is included in the implementation hash list. Other
  model/objective/clock/label/scorer files are unchanged. A new harness contract
  changes only Runtime's hash; its parent e67850f3 is preserved.
- NEW output: `v2data/recovery_1705/`; old v2data receipts remain immutable.
  `recover_data_pilot --prepare` creates a separately hash-bound RECOVERY.json;
  --execute waits >=14336MiB and holds an exclusive recovery lock/dispatch receipt.
  No CUDA allocation while waiting. No peer process is signalled.
- Resume exact saved step4 + Adam; verify/copy all completed-update bytes;
  do not repeat updates1-4 or the already-passing parity smoke. Gate panels have
  no completed outputs to reuse. Preserve the original frozen tensors exactly.
- Spent budget = original FINISHED minus PILOT_RUNNING epochs (~1042s);
  remaining timeout =21570-spent, including new initialization/panels/continuation.
  No fresh cap. Same52 TRAIN panel,44 optimizer worlds, lr/curriculum/CRN,
  smoothing/finer bases OFF, unchanged decide() and PASS continuation.
- Each full step panel is now published immediately when its four shards finish,
  enabling step10 monitoring without waiting for step20/30. No partial-panel gate.
- Local CPU:29 PASS; torch-dependent actual direct-row reproduction and exact
  Adam-resume/frozen-rejection tests require the staged CPU suite before dispatch.

## Verified recovery dispatch, 2026-10-31 17:17 +08
- Staged CPU suite **31 PASS**, including the actual grouped-feedback direct-row
  failure/repaired binding and exact Adam resume + frozen-tensor rejection.
- Archive local/remote sha `da7623b877a009d33dd5945aecd39c21e52d948a54f6b03b5ec00d6e036534f3`.
  Existing staged files backed up to v2data/pre_recovery_55733a60.tar.
- RECOVERY.json SHA256 `056c295c441bff291ec3548a47be2b1c0dffd00c085e53a51e0dd5f59bca0d10`;
  contract `3cc03fed...`, controllers `cd47e8ad...`. Inputs/parents/checkpoints,
  all31 implementation files and6 drivers verified before dispatch and at launch.
- **Live dispatcher PID222535/start881844640**, PPID1/SID222535, separately
  verified after SSH exit. State WAIT_GPU_MEMORY11485MiB<14336 at17:17.
  NEW authoritative logs: R/v2data/recovery_1705/{STATUS.json,EVENTS.jsonl,
  dispatcher.log,pilot.log}; pilot.log/child appear only when memory permits.
- New pilot path R/v2data/recovery_1705/pilot. Remaining timeout20528s after
1041.774795s spent. No peer processes signalled, no SUMO, no push.

## Recovery launch, 2026-10-31 17:20:12 +08
- Memory gate opened at15337MiB free. Dispatcher222535 launched timeout223221 /
  python223222, revalidated RECOVERY056c295c before launch; remaining20528s cap.
- Initializing the registered corpus on gpuA; no completed step10/full-panel claim
  yet. CPU-only actual frozen CTRL0 and speed-MAPPO policy_feedback loads also PASS.
  Monitor `recovery_1705/pilot/{RESUME.json,update_000010.json,FULL_PANEL_STEP10.json}`.

## Step10 checkpoint milestone, 2026-10-31 17:45:12 +08
- Actual RESUME.json confirms step4/Adam restoration; all14 preserved completed
  files rehashed byte-identical to RECOVERY.json. Updates5-10 completed once.
- step_000010.pt SHA256
  `8bbd3a6d518e752d7d30d06352abbf413e91ac795a82f890b8ef241443b951e2`.
  Update10 finite_gradients=true/frozen_bitwise=true; 249.79s this update,
  elapsed2539.00s including original spent budget. Paired/anchor shares .6975/.3025
  (this sampled world only; not a panel metric or an acceptance result).
- At17:45, three baseline shards run (own PIDs224101/224982/226871); fourth
  baseline and step10 full-panel shards are pending measured-memory admission.
  No full panel has completed. Continue monitoring FULL_PANEL_STEP10.json.
