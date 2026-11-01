# Prospective final execution allocation — no final forecast started yet

The scientific panel/metrics remain the registered design. Final execution uses
the immutable release V2 and fixed outcome-blind world-job plan:

- `FINAL_RUN_PLAN.json` SHA185c565df34e4150c90e4b12d55391ab29772c58ab84eff7c0dc3e50ff1caa80:
  216new world jobs ×K8 =1,728new five-arm/H600 future tasks. Reuse128 completed
  baseline reserved8 tasks; no original forecast is repeated for final scoring.
- Baseline kernels now independently replayed ONLY on dev969420/future0:
  WM2 max error0,150.70s; FULL-v4 max error0,236.71s. All3clocks,15region fields,
  support and45BIC decisions agree with the original stored forecasts. These
  protocol checks are not final-panel cases or parameter-selection objectives.
- WM2 stays on gpuR/original model runtime; FULL-v4 stays on gpuA/source22cc32f6.
  FULL-v5 and LP use identical frozen sourceb2df7458 on both hosts. For **every
  world the v5/LP pair is pinned to the same host**, preventing a hardware change
  from masquerading as the LP departure-parameter effect. Host assignment is a
  deterministic4-of7 split of metadata-ordered worlds, not outcome selection.
- Maximum own slots: gpuA4×5GiB, gpuR3×5GiB, oneCPUthread/nice19 each. Leave shared
  capacity for S-WMTEACH; immediate resource admission includes CPU quota, clean
  inactive-cache-aware RAM, disk and nvidia-smi. No peer PID/config action.
- Primary replicated cells with n2/n3 are scheduled first, then larger cells;
  four descriptive singletons last. Membership, K8 and model list do not change.
  Each child owns one complete world, allowing outcome-blind dynamic load balance
  without importing two incompatible frozen physics codebases into one process.
- No new world starts after Nov2 09:00+08; already-running own worlds finish.
  Every unstarted/partial/failed task is retained as attrition. No scientific retry,
  efficacy-based ordering/stopping, replacement worlds or partial-K8 relabelling.

`FINAL_RELEASE.json` V1 is preserved. **FINAL_RELEASE_V2.json** adds explicit
hash binding of the shared original model-construction config and fixed external
controller checkpoints. This is a pre-dispatch metadata-integrity refinement;
model code/parameters, hypotheses, worlds and prior smoke results are unchanged.
V2 SHA6579ece632f1368aa8ea80c25c612297da3d5e57a44a91793e4ea4643d0e01df
is the only release used for deployment.

The one-use supervisors verify immutable metadata, claim worlds exclusively and
read only task completion/hashes. They do not inspect any forecast efficacy value.
A failed child stops new scheduling while other own children finish; no automatic
inference replay. Completed-source/model/input guards remain active in each child.

Final source/release/plan/launcher gate19CPU PASS/6.16s. Final execution/source
release is distinct from statistical qualification, which remains unestablished.
