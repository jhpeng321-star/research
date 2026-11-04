# u12 diagnostic inputs restored — 2026-11-04 15:22 UTC+8

## Completed engineering correction

- Exact cause confirmed: MAIN14:35's SHA-gated `movejobs_1441` archive moved
  LS episode RESULT markers/payloads still needed transitively through
  `pending/<search>.json.gz -> episode_path -> RESULT.json`. Top-level job
  reference checks did not discover this dependency.
- Registered restoration plan
  `ddf68738c438b6ad37d416eab726ee8afc9fb5aabc5e4d2d2650f2a537cd8bf6`
  in canonical registration and warm1 authority, committed before copying.
- Restored **26 files / 51,960,259 bytes** for 13 frozen LS episode inputs from
  cpuS to their original cpuQ paths. Remote archive SHA verified before copy;
  local size/SHA verified afterward against the immutable original manifest.
  Archive originals retained; no overwrites, deletions, simulation or inference.
- Added `FROZEN_EPISODE_DEPENDENCY_BINDING_U12_1518.json` before restoration.
  The **actual live mover's `protected()` function** reports every one of the
  26 restored paths protected, with no missing member. Both movers stayed live.
- Functional check against deployed src8: **17/17 distinct snapshots load**,
  covering all 12 frozen lessons and all seven isolated-synergy states (two
  overlaps), with expected prefixes and samples.
- Post-restoration cpuQ free 6,769,057,792 bytes (~6.30 GiB), above the
  unchanged 6 GiB dispatch floor. CPU-S landing guard remains 9 GiB.

## Remaining checkpoint-safe recovery

The r3 B/R supervisor is stopped. A continues. B's completed u12 PPO checkpoint,
24 closed ordinary episode outputs and passing composite safety receipt remain
intact, but `B_UPDATE_012.json` was never written because diagnostics failed
after the checkpoint save. Original zero-graded lesson report and interrupted
isolated claim remain unchanged. They are not scientific evidence of loss.

The existing launch operator was notified with the exact input correction and
must handle checkpoint adoption, additive lesson completion and the interrupted
diagnostic claim. **Never rerun the completed u12 PPO update/episodes**, delete
the checkpoint, relax health guards or treat old r3 owners as live. Effective-
command imitation deployment remains operator-owned.

## Receipts

- `U12_RESTORE_PLAN_1793776747405205700.json`
- `U12_RESTORE_REGISTER_1793776817574515800.json`
- `U12_RESTORE_RESTORE_1793776857639598100.json`
- `U12_INPUT_VERIFIED_1793776957235172100.json`
- Operator/MAIN notice: `_sol/COOPMWM_U12_ARCHIVE_STOP_1512_MAIN.md`
