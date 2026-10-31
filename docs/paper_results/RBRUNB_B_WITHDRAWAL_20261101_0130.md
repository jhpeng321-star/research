# B withdrawal before inference — 2026-11-01 01:30 +08

INT's01:24 handoff reports that all A workers failed in controller construction:
the frozen MAPPO callback passes `policy_native_lanes`, unsupported by the staged
base bridge. Read-only AST inspection independently confirmed this exact defect
in B's immutable source b0f66943ec30. No rollout is needed to establish the error.

At withdrawal B had no DISPATCH.json, LAUNCHED receipt, shard directory or task.
Single owned supervisor728893 was waiting for resources. The one-time operation
`tools/rbrunb/withdraw_waiting_b.py` acquired the existing shared admission lock,
verified launch PID, owner receipt, boot ID, start ticks4209105965 and full script
command/root, opened a pidfd, rechecked the source defect, persisted a withdrawal
request, then signalled only that owned waiter. Exit and zero inference verified.

Remote evidence `/root/rbeval20261031/runb/WM-PHYS-B/` retains original LAUNCH,
SUPERVISOR, manifest, log, WITHDRAWAL_REQUEST.json and FAILED.json. Withdrawal
unix1793467742.4357772 explicitly states WITHDRAWN_BEFORE_INFERENCE/tasks0; this
is an interface prerequisite failure, not a scientific efficacy result.

Existing `launch_or_status` collected terminal state and atomically wrote shared
`E:/agent-work/rb-shared/evalclaims/WM-PHYS-B.failed.json`. No automatic retry,
replacement source, claim deletion or .done. INT notified via RUNB_B_STATUS.md;
await corrected NEW variant publication with five-arm interface gate.

First ownership-check attempt exited before file writes/signals because the
remote Python lacks os.pidfd_open. A Linux x86_64 pidfd syscall wrapper then
completed the same guarded operation. No peer process was signalled. Original
local watcher33076 continues polling and this session remains alive through~07:00.
