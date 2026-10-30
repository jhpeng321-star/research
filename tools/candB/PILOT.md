# S-PILOTB — EXPLORATORY (owner 2026-10-30 19:45)

- Authority: shared `REGISTRATION.md:4999`; distinct exploratory receipt, never CCT-2/R2 PASS.
- Coordination: only new `tools/candB/pilot_*` files and this report; S-RANKCHECK files untouched.
- Preset: `configs/c4_policy14_candB.toml`, unchanged; fresh zero residual = CTRL0,
  antithetic M2 / paired K8 / 300s, headroom starts, actual mix ON, frozen scoring phase.
- Default OFF: both authority and launcher require `--exploratory-owner-20261030`.
- Independent source/config/input/WM-config/parity identity remains mandatory.
- `pilot_launch.sh` is the isolated launch.sh equivalent for explicit exploratory runs;
  `dry` runs two updates, then a separate fresh pilot starts from the original zero phase.
- R4: at phase update10, write own STOP unless raw dense dJ vs CTRL0 improves from
  frozen phase zero and dense/300s values are finite. MAIN owns subsequent monitoring.
- Initial GPU observation 19:38 +08: gpuA free22635MiB, peer938487 uses25866MiB;
  gpuR free15178MiB, peer433299 uses8906MiB. Serial gpuA is first feasible placement.
- Implementation `0d7a0979`: 75 CPU tests PASS (pilot authority, default-off,
  drift, candidate-B integration, cap-relative execution); bash syntax PASS.
- Initial broader staging check: 87 PASS / 2 fixture-import failures; gatemeas
  collection also needs `tools.gated300`, omitted by the existing source bundler.
- Initial staged runtime: gpuA `/root/autodl-tmp/c4/candB_pilot_0d7a0979_EXPLORATORY_20261030`
  (2,358 committed LF source files); no training dispatched from this first bundle.
- 19:41 gpuA: new rankcheck jobs965450/965476, free15135MiB; launch waits for >=20500MiB.
- Serial ES uses the identical ordered `eslong.forward_task` calls in-process,
  avoiding a second CUDA context; coordinator allocator remains preset .40.
- Final runtime/authority, PID/hashes/timing/peak memory and observed updates: pending.
