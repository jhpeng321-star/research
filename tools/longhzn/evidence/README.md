# S-LONGHZN-1 evidence

Generated numerical products are `rows.json` (all model/group paired differences),
`results.json` (registered statistics, paired changes and residual decomposition),
and `receipts.json` (inputs, source bindings, task-owned process completion).

Reproduce with Python + NumPy: `python tools/longhzn/summarize.py` after both CPU
coordinators finish. This reads only task-owned `~/longhzn` products on cpuP/cpuQ.
The original registered gate source and source/panel/checkpoint copies remain there.
All rollout calls are CPU-only and use unchanged registered panel selection,
global group indices, seeds, recorded CAV commands, t0-HDV proxy and bootstrap.

## Protocol

- Authoritative prospective registration and completion receipt are append-only in
  `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md`.
- WM0 `0044eddb…` and WM1 `5257a64c…` use the same candidate configuration
  (`f95f7c4c…`). Base r16 `5b256941…` uses its original resolved configuration
  (`d2181901…`). Full hashes are in `receipts.json`.
- Gate/rollout/panel functions are imported from the unchanged registered source;
  sharding preserves **global** group index, arm order and seed `140014+7919*g`.
  Each arm is rolled out once for 600 half-second steps. All five prefixes use the
  same rollout. Panel identities match the original 96-group gate exactly.
- Both CPU boxes verify all 4,032 selected episode files against the existing full
  pairs300 source. Selected data are independent task-local copies. CPUQ uses
  immutable hard links only for nonselected manifest-census entries after a full
  byte-copy exhausted disk; the incomplete task-owned copy was retired. No peer
  file or process was modified. Runtime staging and transport failures preceded
  inference. Thirty-two inference workers all exited successfully; task-owned PID
  incarnations were checked after completion. No SUMO, training or GPU execution.
- `source.json` verifies 331 source files against gpuA and across CPU snapshots;
  only two unrelated policy/recovery source differences are excluded and listed.
  Deployed and committed runners have identical Python ASTs (one formatting edit).
- At every horizon there are 192 differences. The two exactly zero recorded
  differences are excluded **only** from sign agreement. Pearson and residual
  statistics use all 192. All models have identical recorded-reference values.
- `registered_gate.py` is the exact frozen registered source, SHA256
  `e9aac75ffd3a96fad9bedf0dcd6f38995bcd5eb9e7fa11dd6a353091f7a69970`.
  The summarizer directly calls its 2,000-draw paired-**group** percentile bootstrap,
  using seed h for signs/residuals and h+1 for Pearson. Model changes share each
  resampled group. These are pointwise registered CIs, not simultaneous intervals
  or world-cluster intervals (96 branch groups belong to 54 worlds).
- The registered threshold is **0.868421052631579**. PASS compares the sign point
  estimate to it, including the 60 s prefix; there is no new threshold or CI gate.
  Original r16 and WM1 60 s signs/CIs reproduce exactly. WM1's largest numerical
  prefix discrepancy from its earlier run is 0.000366211 veh-s.
- Residual `e = dJ_WM - dJ_recorded`. Spread is the centered population SD
  (`ddof=0`), so `RMSE² = bias² + SD²` exactly. Bias share is `bias² / MSE`.
  Between-case dispersion cannot separate stochastic rollout noise from
  state-dependent systematic misspecification; there is one registered future.

## Execution and verification

Initial setup used `stage_cpu.py`, `transfer.py`, `launch.py` sequentially for each
box; CPUQ's retained copy recovery is `repair_partial_cpu.py`. `verify_panel_cpu.py`
checks byte identity. `status_cpu.py` and `wait_cpu.py` inspect only task products
and exact owned PID incarnations. Workers check live collector metadata and load
before every rollout and yield; all run with nice19 and single-thread CPU libraries.

`python tools/longhzn/verify_source.py` reproduces source identity checks.
`python tools/longhzn/verify_products.py` verifies report rounding, evidence hashes,
all primary statistics/decompositions and task completion without new inference.
`python -m ruff check tools/longhzn` checks the analysis source.
