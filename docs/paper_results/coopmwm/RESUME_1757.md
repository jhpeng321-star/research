# Fresh-session recovery — 2026-11-01 17:57

- Read git status/log, original COOPNATIVE design/quote/handoff, COOPMWM results,
  amendments, quiesce receipt, bridge failure and active deployment code first.
- Preserved uncommitted `ops.py`, `service.py`, `wm.py`, and `BRIDGE.claim`.
  They contain an unfinished explicit frozen-cost-source fix; not CPU qualified.
- cpuQ recovery: no `tools.coopmwm` processes. Original v2 runtime, completed jobs,
  checkpoints, charge ledger, smoke and calibration-window receipts remain.
- cpuQ prior quiesce receipt: 62 closed jobs / 74,400 sim-s; no update completed.
- gpuA and gpuR recovery: no COOPMWM processes. Other GPU PIDs untouched.
  gpuA 2153MiB/49140MiB,93% utilization; gpuR3663MiB/24564MiB,47% utilization.
- Local process inventory: no live COOPMWM bridge/SSH service. Old dispatch/error
  receipts remain; bridge failed before readiness on cost-adapter ABI assertion.
- Owner17:55 admits both GPUs, draining peer jobs retained. Next step is CPU
  qualification and prospective dual-service/sampling registration before physics.
- No new SUMO, dispatch, signal, reset, cleanup, deletion or push during recovery.
