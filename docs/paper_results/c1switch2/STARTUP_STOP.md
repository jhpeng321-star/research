# S-C1SWITCH2 zero-physics startup stop

First mechanics dispatch: cpuQ PID446935, `/home/ubuntu/c1switch2_run_2d2400c953ef2545`.
The PID exited; accounting0 / actual0, no constructed job, no SUMO started, GPU0.
Receipt: `tools/c1switch2/evidence/CANARY_PROGRESS_1793465955.json`.
Failure: `runner.backend` called the two frozen guards before `runtime.setup`
installed the frozen package paths. The pristine process could not import
`traffic_shepherd.seeds`. CPU tests had run after setup and masked the ordering bug.

Repair: pure exact panel/cell/type/host membership first; hash-bound CPU-only
runtime setup second; BOTH genuine frozen guards third; backend/scenario/SUMO last.
A regression rejects invalid worlds before setup and verifies that order.
Host preflight now starts with a pristine-process runner-construction probe,
using the genuine setup/guards but intercepting backend creation before physics.
It requires the guard modules initially absent, so an earlier setup cannot mask it.

All original dispatch locks/logs and zero-debit receipts remain intact.
A corrected immutable stage requires an explicit registered predecessor, verified
exited own PID, empty ledger and empty jobs; no automatic retry/refund or PID signals.
The unused original4680 sim-s mechanics allocation remains the entire canary cap.
The design/panel/arm budgets and all scientific/controller sources are unchanged.

Repair commit1065f89d registered with the exact old-root predecessor in REGISTRATION.
Corrected CPU-only stage: `/home/ubuntu/c1switch2_run_4ad4ece01d5a9ac1`.
Pristine runner construction PASS; genuine frozen17/17guards; CPU125passed/1skip.
No physics at staging/preflight. Any dispatch must explicitly name the old zero-debit predecessor.
