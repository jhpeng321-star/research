# Fleet PPO SSH-recovery reconciliation

Verified 2026-11-01 11:39 +08:00, after reading git status/log and existing receipts.

- MAIN11:37 correction confirmed: cpuQ uptime 518,620 seconds (about six days),
  so the outage was SSH-only. Training continued unattended.
- Original owned supervisor645235 is absent. Immutable dispatch/source hashes pass;
  no processes reference the owned final build. This is completed training, not a failure.
- Exactly169 genuine updates, schedule23+23+167x46, all7,728 new production worlds
  complete, zero outstanding and zero new failures. Final trainer status is
  `TRAIN_COMPLETE_AWAIT_REGISTERED_PANEL`.
- Charged6,998,400sim-s, including43,200 historical charges. New actual6,955,200;
  total actual bounds6,985,800..6,993,000 retain historical uncertainty.
  Evaluation14,400 remains unstarted here; exact total cap7,012,800 unchanged.
- Completed training control-window receipts report zero collisions/teleports.
  These are training safety receipts, not panel efficacy or whole-episode safety results.
- cpuQ free disk3,057,577,984 bytes (2.85GiB); own `selmappo20261031` tree
  occupies5,153,712KiB (4.92GiB). No new training outputs are growing.
  All prior builds/checkpoints/ledgers/locks remain retained.
- Local observer3668 is absent after publishing the frozen final169 receipt.
  Existing noon supervisor44716 remains live, exact Python command and creation time
  04:33:12+08:00 verified. Its exclusive lock remains; no duplicate watcher launched.
- No dispatch, repair, simulator episode, process signal, replay or relocation occurred.

Fresh evidence:
`selmappo/PROGRESS_build_9bb7869d8f590818_1793504353339657273.json`.
Frozen-final validation was rerun idempotently against the durable checkpoint, all
169 update rows, exact START/END sets, charge ledger and immutable staged sources.

## Noon supervision completed
- At12:00:19+08:00 both original local PIDs3668/44716 were absent; publisher44716
  emitted its scheduled12:00 heartbeat then exited normally. No replacement launched.
- At12:00:32+08:00 fresh remote progress/source checks reconfirmed169updates,
  7,728complete/0outstanding/0newfailures,6,998,400charged and no own-build processes.
  cpuQ free4,958,339,072bytes (4.62GiB); training accounting and final169 unchanged.
- Final receipt: `selmappo/PROGRESS_build_9bb7869d8f590818_1793505632831980263.json`.
- All169 update rows contain231,840 fleet decision frames. Sum of recorded batch
  simulation/admission/PPO wall times30,252.551130s gives827,656.48sim-s/hour for
  6,955,200newproduction sim-s. This excludes inter-update checkpoint/setup overhead
  and prior failed builds; it is measured throughput, not efficacy or total CPU-hours.
- Final-build training tree5,088,964KiB; whole owned root5,153,712KiB, all retained.
- C1SWITCH2 PANEL_v2 SHA256 independently matches the registered
  `7c780c25237dde73a657937ab577b8e48f71766817f4363f3298982e90144827`.
