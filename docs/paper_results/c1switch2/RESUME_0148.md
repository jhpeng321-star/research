# Resume under MAIN 01:48 authorization

Read git status/log, current task receipts and exact registration before changes.
Preserve the unrelated modified `SELMAPPO2_HANDOFF.md`.
cpuQ task-only status: both canary PIDs exited; successful canary remains final.
gpuR task-only status: CPU-probe stage only; no main dispatch receipt.
Existing main admission incorrectly waits for every model/selector for every arm.
MAIN 01:48 explicitly authorizes the six fixed arms plus random immediately.
Implement immutable arm-scoped admission and single-use CPU dispatch on the existing
panel/host partitions; retain every scientific gate relevant to enabled arms.
CPU subset costs exactly 134,400 sim-s (100,800 cpuQ; 33,600 gpuR), GPU0.
Canary is already charged 4,680; never rerun it. Total budgets remain unchanged.
No main physics or GPU service was started by this inspection.

Arm-scoped CPU build: gpuR 130 passed / 1 host-only skip; cpuQ previous
revision 129 passed / 1 skip. Final cpuQ gate is required before its dispatch.
Local default Python lacks torch; use existing host environments, no new venv.
Retain CPU-only staging/preflight failures (zero physics/charges); none are dispatches.
`operations` compiles embedded remote commands before SSH; supervisor job planning
is covered by the CPU suite (84 + 28 disjoint jobs, excludes canary and all pickers).

Pre-dispatch registry SHA check detected unrelated concurrent peer appends and
refused before writing gates/claims or starting processes. Bind the immutable
approved registration snapshot to its own SHA, and separately audit current
authority/intervals at dispatch. cpuQ final source passes130/1skip again.
