# S-MINING recovery — 2026-11-06 01:15 +08

- Preserved prior blocked receipts and zero-charge cpuV staging. No existing miner or parity process on gpuA; no duplicate dispatch.
- Laptop gpuA SSH succeeds. RTX 4090 D is idle (1 MiB, 0%); quota 20 CPU cores; data disk 10,479,456,256 bytes free.
- Base gpuA environment is Python 3.12.3 / Torch 2.8.0+cu128 / NumPy 2.3.2, without SUMO. It is not qualified for parity. Prepare a separate byte-verified cpuV environment/source/input clone; keep GPU WM environment separate.
- cpuV original P3/P4 snapshots/specs/reference payloads remain available, role=train, seeds986047/986038; historical parity960 sim-s are not rerun on cpuV or recharged.
- Fresh cpuV public receipts show clean2 STOP/FAULT: whole-wave budget blocked (2026-11-06 01:06:45+08); resident chain FAILED/MONITOR. Existing monitor has already published this. No training control action by mining.
- New simulation/charges remain0. Before parity: preserve exact source/input/environment hashes, admit isolated output root and own ledger, reconcile protected allocations, precharge one branch at a time; stop on first parity mismatch.
- Capture receipts: E:/Temp/opencode/MINING_GPUA_RESUME_ENV_0112.json and MINING_CPUV_INPUTS_0115.json (complete transport stdout/stderr).
