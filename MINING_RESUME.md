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

## 02:17 +08 — fresh-session running-owner recovery

Read own git status/history and receipts before action. Original GPU coordinator
3230/start928158286/boota81555d7-176e-4ed8-bc7d-896fc89cc914 is alive with exact
argv; adopted without dispatch. Five original startup receipts now exist; four
completed comparisons observed PASS, next gate computing. Real RTX4090D utilization
94%, memory19778MiB; no input query/result yet. Frozen u0 already staged. gpuA
ledger remains960 sim-s; no new physics. Preserved uncommitted lane adapter,
schedule/tests, u0 staging and status modules.

Actual cpuV STOPPED_BY_OWNER.json now exists, original owners exited and sealed
A/B/R18; prior budget fault and every protected hash remain intact. clean3 root
CONFIG/LAUNCHED absent. cpuV second-host admission can now be prepared with G1
certification priority, initially one nice19/single-thread worker, own ceiling14/48.
No cpuV verification dispatch yet. Capture receipts in
E:/Temp/opencode/MINING_RESUME_INVENTORY_GPU_0218.json and
MINING_RESUME_INVENTORY_CPU_0219.json. Earlier cpuV inventory failed read-only on
an unprivileged /root probe; fixed host detection, no remote write or simulation.

## 02:24 +08 — second recovery, accepted G5 source bound

Rebased the five mining-only commits onto accepted G5 b903c6c2 as requested;
preserved all six uncommitted adapter/inventory files and prior receipts. The
inherited sparse checkout was expanded for mining sources after rebase.
Read-only gpuA inventory confirms original GPU owner3230/start928158286 and
boot a81555d7 alive, seven startup receipt filenames, no query/result/fault.
No second model was dispatched. The first status command used an absent generic
python3 and exited127 before executing code; corrected to the existing absolute
/root/miniconda3/bin/python. Both transport outcomes are retained.
cpuV again confirms STOPPED_BY_OWNER with original owners exited, sealed A/B/R18,
mining ledger empty and clean3 CONFIG/LAUNCHED absent. gpuA mining charge remains
960sim-s; no new physics. Receipts RECOVERY_GPU_0222.json and RECOVERY_CPU_0223.json.

## 04:31 +08 — fresh-session recovery of later on-disk work

Read own history/status, required registration tail and receipts before dispatch.
Exact parity already PASS; no replay. Adopt existing gpuA WM owners6512/start928642562
and6514/start928642563, same boota81555d7. Two completed K8/H300 rank receipts;
current GPU75%, no replacement worker. Prior SUMO operator batches are stopped
with zero current verification owners. Preserve all uncommitted operator sources.
gpuA ledger123rows/305551sim-s (SHA96c2a440c3df47836737091cd4572e3f92905cd621ec9fd98b8a90d93d51cb95),
remaining194449/500000. Both prior MINED files empty. Bank-v2 lane examples report
data_missing and require diagnosis before further spend; no survivor implied by
generic first-stage 'admitted'. cpuV own ledger still empty; owner-stop confirmed,
clean3 chain434007 waiting, LAUNCHED absent. Failed guessed interpreter invocation
retained; successful system-python F6 read follows. No new physics or signal.
Registration now includes owner7.3M overall cap/released plan_fix reservation and
04:20 wider2..6-CAV WM search; mining allowance remains500000.
