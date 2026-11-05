# S-MINING recovery — 2026-11-06 01:15 +08

- Preserved prior blocked receipts and zero-charge cpuV staging. No existing miner or parity process on gpuA; no duplicate dispatch.
- Laptop gpuA SSH succeeds. RTX 4090 D is idle (1 MiB, 0%); quota 20 CPU cores; data disk 10,479,456,256 bytes free.
- Base gpuA environment is Python 3.12.3 / Torch 2.8.0+cu128 / NumPy 2.3.2, without SUMO. It is not qualified for parity. Prepare a separate byte-verified cpuV environment/source/input clone; keep GPU WM environment separate.
- cpuV original P3/P4 snapshots/specs/reference payloads remain available, role=train, seeds986047/986038; historical parity960 sim-s are not rerun on cpuV or recharged.
- Fresh cpuV public receipts show clean2 STOP/FAULT: whole-wave budget blocked (2026-11-06 01:06:45+08); resident chain FAILED/MONITOR. Existing monitor has already published this. No training control action by mining.
- New simulation/charges remain0. Before parity: preserve exact source/input/environment hashes, admit isolated output root and own ledger, reconcile protected allocations, precharge one branch at a time; stop on first parity mismatch.
- Capture receipts: E:/Temp/opencode/MINING_GPUA_RESUME_ENV_0112.json and MINING_CPUV_INPUTS_0115.json (complete transport stdout/stderr).

## 01:49 +08 — exact gpuA parity PASS

33,618 environment/source/input files SHA-verified. Isolated physics Python3.10.16 and all cpuV SUMO/physics pins match; frozen backend setup PASS. Both P3/P4 ran once and matched complete uncompressed payload bytes, branch receipt, result seal, steps, prefixes, totals and final. P3 payload90d82a9b..., P4ca85ea6d...; no tolerance, window change or reference replay. Mining ledger charged960/actual960; remaining499040. Existing allocations protected at5.5M bound within6.5M. Canonical small receipts fetched to mining_receipts; remote root /root/mining20261106.
