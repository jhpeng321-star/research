# S-MINING duplicate-session recovery — 2026-11-06

This recovery session yields operational ownership to the already active mining
operators. It dispatched no model, SUMO branch, collector, verifier, or transfer,
and sent no process signals.

## Evidence checked before changes

- Own worktree `E:/agent-work/c4-s-mining`, branch `c4/s-mining`, recovered at
  `1823bb70`; all existing untracked executor/collection/recovery files preserved.
- `mining_receipts/PARITY_PASS.json`: exact gpuA/cpuV P3 and P4 PASS, all nine
  comparisons true for each branch; historical mining parity charge 960 sim-s.
  No parity replay is required.
- `MINING_RESUME.md`: accepted G5 `b903c6c2` already incorporated. Existing GPU
  owner and successor worker dispatches must be recovered, not duplicated.
- `RECOVERY_LIVE_GPU_02.json` appeared during this recovery (timestamp
  `2026-11-05T20:28:57.015486+00:00`) and reports live `wm_lane04` owners 6512/6514.
- Local process ancestry identifies this session as OpenCode PID 19276. A separate
  live OpenCode PID 37212 has the exact same `S-MINING-W2-sol` resume prompt and
  creation time `2026-11-06 04:27:08 +08`. Neither process was signalled.
- `CLEAN1_STATUS.md` also records a Claude operator2 takeover and ongoing
  collection/verification/runtime-overlay work. Its published mining charge is
  approximately 291k/500k sim-s, not the stale zero-charge central report. That
  approximation is not a reconciled ledger total.
- One additional read-only gpuA recovery capture is retained as
  `mining_receipts/RECOVERY_DUPLICATE_SESSION_GPU_0429.json`.

## Ownership and next action

The existing operators retain the running GPU workers and verification work.
Do not restart parity, launch a parallel verifier, replace `MINING_REPORT.md`,
publish/admit records, or alter shared ledgers from this duplicate session.
The active operator must reconcile authoritative host ledgers, qualify the G5
runtime closure and per-member solo execution, then publish only fully certified
records and the final report/transfer receipts. No verified move is claimed here.

The latest registered cap is 7,300,000 total with mining still capped at 500,000;
the latest owner requests a 2–6-member WM search, while clean3's current admission
retains its registered <=4-member ABI pending a later registered extension.
