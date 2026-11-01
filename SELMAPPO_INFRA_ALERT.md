# Fleet picker: cpuQ connectivity prevents current health verification

## RESOLVED — 2026-11-01 11:40 +08:00
SSH-only outage confirmed by six-day uptime. Fleet training finished169/169updates,
all7,728 continuation worlds complete,0newfailures/outstanding. No restart required.
Final169/hash/source/ledger/finite-policy validationPASS; frozen handoff already matches
`E:/agent-work/c4-c1switch2/SELMAPPO_FINAL.json`. Exact6,998,400charged plus14,400
UNSTARTED evaluation reservation stays within7,012,800. See HANDOFF.md and
docs/paper_results/SELMAPPO_RECOVERY.md. Historical incident details follow.

Observed2026-11-01T05:34:10+08:00. cpuQ SSH closes before executing a command:
`Connection closed by 1.15.42.207 port 22`. Reproduced with `ssh cpuQ true`.
The existing observer has no fresh receipt after05:15:32+08:00. Last verified:
37/169genuineupdates,1,687complete/6outstanding production,0newfailures,
1,566,900charged sim-s including43,200historical. Exact cap7,012,800unchanged.

Remote owntrainer645235 current health UNKNOWN, not established stopped.
No relaunch/signals/replay/deletions/extra mechanics/evaluation. Existing local
observer3668 and noon publisher44716 stay armed; read-only probes continue.
MAIN: please inspect host/SSH availability if this persists; no peer PID action
has been taken. Details and committed20-minute alert: HANDOFF.md and
docs/paper_results/SELMAPPO_SUPERVISION.md (05:33commit69b21694).

Follow-up: gpuR's direct SSH works; a gpuR jump to cpuQ times out during banner
exchange. Shared REGISTRATION.md now carries `S-SELMAPPO 05:48 CONNECTIVITY ALERT /`
`MONITORING CONTINUES`, with unchanged cap/schedule and no relaunch claim. MAIN's
05:28 COOPFREE and05:13 C1SWITCH2 records also report SSH transport failures.

07:14 update: MAIN07:07 says no relocation before11:30; priority WM-PHYS on
gpuA/gpuR first. Observed and followed. Both local observers remain alive; latest
20-minute progress commit1734fec5 still records SSH failure, no fresh remote state.
Recovery dependency for MAIN's proposed later continuation: the existing observer
records checkpoint hashes/remote paths, not weight bytes. No intermediate model/
optimizer transfer is recorded. Need an actual verified checkpoint and exact consumed
world/charge reconciliation before resuming on any host; cannot resume from hashes
or rerun completed worlds. Original protocol/update count/cap remain unchanged.

10:17: registered this recovery-input dependency explicitly in REGISTRATION.md.
Targeted safe globs found no `checkpoint_*.pt` in the task evidence tree or
worktree root; the observer's source confirms metadata-only intermediate records.
Both local processes verified alive10:15; latest heartbeat commit9ed3beeb.
No CPU relocation, reinitialization or new simulation has been attempted.
