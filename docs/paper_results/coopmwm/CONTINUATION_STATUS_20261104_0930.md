# Continuation status — Nov 4 ~09:30 +08

## Observed current run
**A7 / B3 / R3.** A's durable runner is alive. B/R remain at the registered
08:52 mechanics pause; no release is observed yet. MAIN 09:00 assigns the
statistical pass-1 rule, stronger-imitation continuation and release to the
launch operator. MAIN 09:10 practice-starts integration is also delegated.

The proposed recovery commit 9ddb8d48 is staged as `launchcoop_src5`, but its
immutable Linux result is **not PASS** (`pass_=false`, 262 tests executed,
0 skips, 54.31 seconds, zero simulation). The operator is still working.
The last qualified deployed source remains `launchcoop_src4` / 69b1e264
(233 Linux tests PASS), runtime SHA prefix 4a8890cd. Do not adopt the failed
candidate as qualified or start a competing recovery.

## Last completed B/R update evidence
All numbers below are from immutable u3 receipts, not new u4 or u6 outcomes.
- **B3:** 5/5 full imitation passes; effective first pass, AWR loss drop 0.01877;
  KL 0.00163246 against 0.045 limit; eight cooperative examples imitated.
  Action MAE 1.331360 -> 1.327158 m/s². Bank intent loss 2.265207 -> 2.377996
  increased (report-only). First PPO minibatch KL 0; recorded PPO max 0.0089955.
- **R3:** 5/5 full imitation passes; first-pass AWR loss drop -0.00173569,
  which triggered the original pause. KL 0.00015508; three cooperative examples.
  Action MAE 0.453078 -> 0.452906 m/s². Bank intent loss 0.487287 -> 0.488759.
  First PPO minibatch KL 0; recorded PPO max 0.0085129.
- **Composite safety passes for both**, 24 paired worlds each. Collisions and
  teleports: zero in both arms and A. B critical-event delta vs A: -29.7708
  [-54.2089, -9.8531]; hard-brake delta -1.1458 [-3.7917, 1.3750]. R critical
  delta -5.8750 [-22.4375, 10.3542]; hard-brake delta +0.4167 [-1.5208, 2.1667].
- B3 reused closed/precharged episodes during recovery; its 309-second wall
  time is not a fresh-update throughput estimate. No B-u6 lesson or isolated
  synergy result exists yet; cooperation viability remains pending.

## Storage and collection
The approved V2V archive extension has certified 1,035,251,575 bytes moved.
Both archive owners and old runtime-bound learning monitor3 remain live.
At 09:29 cpuQ free was 6,822,576,128 bytes (~6.35 GiB). Backfill is disk-paused;
its 9 GiB resume gate is preserved. Pending A safety-reference RESULT payloads
remain protected while A is ahead of B/R. Zero new simulation was dispatched.

Current-policy collection is still blocked by V2V snapshot-capture and
imagined-actor/intent integration, previously reported to MAIN. Existing
collection graders remain alive. The saved queue is exhausted, so a restart
alone is not a collection fix. The operator has received this blocker and the
monitor3 handover reminder in `_sol/COOPMWM_RECOVERY_0925_OWNER_NOTE.md`.

## Completed requested stop
`warm1/STOPPED_PERMANENT_MAIN0910.json` is sealed, SHA256
`92da212d725d606f2272850f17acc0c6ea3c5061df53cf2483194f65ebe554fa`.
Legacy B21/R20 checkpoint hashes were independently rechecked; all five old
owned PIDs are absent. Checkpoints, bank and receipts stay intact. Canonical
REGISTRATION.md records the confirmation. No new signals were needed.

Evidence: `V2V_LEARNING_WATCH_20261104_V3_STATUS_1793755385276835100.json`,
`RESUME_ADOPTION_0915_1793755780598871400.json`,
`LEGACY_STOPPED_PERMANENT_MAIN0910.json`, and
`STORAGE_AND_LEGACY_STATUS_20261104_0915.md`.

Next: adopt the operator's final qualified release/receipt when published;
verify B/R u4 progress, monitor safety/imitation/intent, and publish actual
u6 lesson/isolated-synergy evidence. Keep the shared resource gates and movers.
