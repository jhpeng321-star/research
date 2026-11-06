# CLEAN3 r10 continuation — STOP retained, no launch

## Adopted work and ownership

Recovered git status/history, all tracks_spec.md, the last 300 registration lines
and task receipts. Preserved the pre-existing untracked
`ops/clean3_inventory_admission.py`. Setup/parity/copies, accepted integrations,
completed tests and previous stopped roots were adopted.

The existing Claude launch operator committed ac9e4691 during recovery and
retains production fix/qualification/dispatch ownership. That exact source's
Linux qualification passed (returncode 0, 209.345 s; reported 1880 passed).
Its r10 chain ran as PID540487/start12092639, root
`/home/ubuntu/coopmwm20261106_clean3_r4`. No competing research job was started.

## Independently verified fault

Read-only F6 audit at cpuV 2026-11-06T02:44:13Z:

- Chain exited at SMOKE, 2026-11-06T02:40:21.901507Z; STOP retained.
- Actual exception: `(b) mined seed situations missing from the practice pool: B`.
- Eight r4 charge rows, 9,600 sim-s. GATE_STARTED/GATE_PASS/LAUNCHED absent.
- Line-by-line log and public receipt hashes retained in
  `CLEAN3_R10_FAULT_READONLY_LINES.json`; first raw-log capture also retained.
- This is a component-receipt failure. No new collision/teleport finding is
  established by this fault; no gate or training has run.

Source-level likely cause: bank_breadth.practice_pool deduplicates by
`digest(snapshot SHA + family)`, keeping the first row. Different mined verified
wall plans at the same seed/snapshot/family can collapse, whereas check (b)
requires every `(verification_id, example_id)` to remain in the practice pool.
Confirm exact missing identities without physics, then preserve own-plan binding
and fix with regression tests; do not weaken check (b).

The separate zero-physics `ops/clean3_pool_audit_remote.py` attempt did NOT run:
accepted F6 admission lock timed out before remote execution at 10:48 +08. No
bypass, detached audit, new simulation or transport modification was used.

## Fresh-root budget blocker

With every failed charge retained:

`3,098,640 prior + 22,800 failed clean3 + 500,000 mining reserved + 3,680,400 full quote = 7,301,840`.

This exceeds the current clean3-scope cap 7,300,000 by **1,840 sim-s**. The overall
7,550,000 approval includes a separate 250,000 coop_probe phase. This recovery
does not reallocate probe/mining slack, lower phase margins, omit a phase, change
windows, clear STOP or retry/adopt attempted smoke physics. Before a fresh-root
retry, MAIN/owner must resolve the complete budget assertion. A cached-only
recovery would require an explicitly registered, tested exact-source procedure;
the current one-use chain forbids automatic retry/adoption.

## Read-only local follower

`ops/clean3_follow.py` uses one Windows lock plus accepted F6, validates launch /
config / smoke / gate / update / monitor seals and phase/cumulative bounds, reads
scheduled diagnostics, and appends five-line CLEAN3 status summaries. It has no
remote process-control or dispatch operations. Five isolated no-SUMO tests pass
(tampering, owner loss, absolute safety finding, exact SHA-bound historical pause
release, BOTH F5 quantities). Actual prelaunch read-only probe also passed.

Follower PID37040 observed the real STOP, appended CLEAN1_STATUS.md and exited
naturally; it is not currently running. Its local state/logs remain under
`E:/agent-work/_sol/CLEAN3_FOLLOW*`. Initial local notices for the copied clean1
B9/R3 pauses were false classifications, corrected in CLEAN1_STATUS.md after
their exact sealed releases/AUTHORITY hashes independently verified. No remote
safety rule or release was altered.

## Next actions

1. Existing launch operator confirms/fixes practice-pool identity, with tests and
   exact-source Linux qualification. Preserve all existing roots/charges.
2. MAIN/owner resolves fresh-root quote deficit or explicitly registers qualified
   cached completion. Do not dispatch chargeable work before that gate.
3. Keep every smoke check, single zero-event safety gate, full 24-update schedule,
   >=15% per-phase margin, frozen primary base and u24 synergy criterion.
4. Only after an authorized chain starts, adopt its exact root/control; preserve
   r4 follower state before starting one local follower for a new root.
