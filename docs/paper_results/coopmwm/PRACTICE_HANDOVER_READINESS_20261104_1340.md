# Practice handover readiness — Nov4 13:39 UTC+8

Practice deployment is owned by the existing launch operator. No second
launcher, monitor, gate, checking job, or archive owner was started here.

## Accounting checked independently

`ACCOUNTING_COHERENCE_1793770428742047400.json` reads both live authorities
against the same ledger SHA. Charged13,847,120 of17,471,840;811,640retained
superseded counted once and fully spent. Runtime1d67a685 contains all three
registered deltas: bank+840,000 from sumo verification; bank+100,000 from
legacy train; v2v_train+345,600 from paired_branch. Its phase sum is exactly
the total cap. Bank phase82,180remaining; v2v_train1,468,800remaining;
v2v_lessons13,080remaining. The previously reported13,200future lesson gap
is still open; currentu12fits. No authority or allocation modified here.

## Boundary and monitor

`HANDOVER_READINESS_1335_1793770795336638100.json` confirms runtime
practice_starts=true, fraction0.25, repeat cap2, cooperative_only,
kl_targeted. No **v2v1** u10 B/R charge rows at the preceding corrected
probe1793770711. The first exploratory charge-name selector also included
historical legacy u10 rows; that saved raw receipt is retained, but only
the corrected v2v1-qualified selector is valid for the new handover.

B r2 PID2636963 is now absent, consistent with the operator's registered
u9 boundary stop. R2636968 remains alive at u8; A16 is visible in monitor4.
Monitor4 still covers current B9/R8 but binds06e88ecd, not the new runtime;
operator notified in `_sol/COOPMWM_PRACTICE_HANDOVER_MONITOR_NOTE_1335.md`.
The handover and u8 test dispatch remain pending, not claimed complete.

## Storage transport diagnosis

cpuQ free6,380,920,832bytes at1793770791 (~5.94GiB), below6GiB preflight.
Both movers1237127/2213098 remain alive. Regular mover's fresh scan is
healthy but has zero eligible files. Extension batch37 rsync/SSH children
were observed active, then exited naturally with connection closed/code255
at1793770743. The durable owner's registered connection-only retry retains
sources and selects a fresh batch namespace. No SHA failure, unsafe source
removal, dead owner, or authority for a competing mover is established.
Certified extension total remains3,368,783,400bytes pending the next proof;
last inventory443,890,949eligible bytes, destination16.46GBfree (>8GiBfloor).
Current forwarded socket exists and was refreshed at1793770261. No mover
was paused/signalled; no failed namespace replayed. Keep all floors and
bank/lesson/synergy protections unchanged. CPU-S landing9GiB guard and
cpuQ backfill/collector stops remain in force.

Evidence: `RESOURCE_READINESS_1130_1793770442949676900.json`,
`V2V_ARCHIVE_SCOPE_0712_V2_STATUS_1793770508010606600.json`, and
the two corrected handover probes1793770714419240400/1793770795336638100.
