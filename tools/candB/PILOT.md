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
- Final staged runtime: gpuA `/root/autodl-tmp/c4/candB_pilot_ready_EXPLORATORY_20261030`,
  source **9ea62c59bb03e2c8f783f611b5ef7b35d52b0207**, 2,371 committed LF files verified.
- Final CPU suite **77 PASS** (including row/summary stamping, R4 STOP/pass and patch
  restoration); Torch2.8.0+cu128 with CUDA hidden. bash syntax/diff checks PASS.
- Exploratory authority built + independently verified; local `pilot_authority_receipt.json`.
  Authority SHA256 `5694ca0e4f1e091f5cdeb3ebff40cabb017b7f2548b852ad2e3c957dd65aab5c`;
  effective config `78a3caec6813c9b8b87996162908a09d7bb3d747f8932d2484d2cd239bbcedd3`.
- WM2 `b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436`;
  inference config `f95f7c4cbb501ccb85bb3968e29f3d6ca53db4ecef5f586225f96df1f48193e4`;
  init checkpoint `560e037440f58c665b316a3008d47f77e2ca1fbb9d10173ef3e40f565b5b276c`.
- Planned owned chain root: `/root/autodl-tmp/c4/candB_pilot_run_EXPLORATORY_20261030`;
  own `OWNERSHIP.json`, `dry_OWNERSHIP.json`, `pilot_OWNERSHIP.json` retain PID/start ticks.
- PID/timing/peak memory and observed updates: pending admission/launch.
