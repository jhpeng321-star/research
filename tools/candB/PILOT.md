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

## Admission and telemetry refinement (before any GPU training)
- Waiting supervisor968085/start874131451 launched19:51:40, observed free10901MiB;
  no dry/pilot child launched. Own STOP requested while waiting; that root is retained.
- gpuR19:53: free8837MiB; peers433299/442305 untouched. Neither card can admit serial yet.
- Before dispatch, refine executed telemetry to observe FINAL callback accelerations
  on ACTING CAV ticks (including the same Euler rounding on CTRL0), rather than
  pre-mask proposed execution. Training/objective algebra is unchanged.
- New supervisor polls memory every600s during training as well as admission.
- `pilot_stage.sh --exploratory-owner-20261030 ARCHIVE gpuA RUNTIME` invokes the
  existing `stage.sh`; a narrow bundler adapter accepts only named S-RANKCHECK dirty
  paths and still archives exclusively committed LF Git blobs. Dirty pilot files refuse.
  It also includes the omitted `tools/gated300/cct1.py` CPU fixture dependency.

## Active final runtime (supersedes all preparation bundles above)
- Source **da0bc3663609024930d799af059c13f9dcf174d3**, gpuA
  `/root/autodl-tmp/c4/candB_pilot_final_EXPLORATORY_20261030`, **2,380** committed LF files.
- Expanded CPU verification: **112 PASS / 2 CUDA-only skipped**, 194.38s; no GPU.
  Both CCT1/CCT2 fixture dependencies are now included. Earlier missing-fixture
  collection failures and one SSH test timeout are retained as preparation attempts.
- Final authority **a089c5fbe439a67cb64e8fce68de4d8ab80cd440909a1a052835ee7dcc7f641a**,
  local `pilot_final_authority_receipt.json`; source/config/input/parity checks PASS.
  Scientific effective config/input hashes above are unchanged.
- Active chain target `/root/autodl-tmp/c4/candB_pilot_final_run_EXPLORATORY_20261030`;
  detached bootstrap waits500s (started20:03) to respect the previous20:01:40 GPU poll.
  Supervisor then polls every600s, admits only >=20500MiB, runs dry2 then fresh pilot150.
- Latest capacity: gpuA20:01 free6161MiB; gpuR20:03 free8441MiB. No GPU child yet.
- Local observer `pilot_observe.py --root REMOTE_CHAIN --ledger SHARED_REGISTRATION`
  retains exact own PID/start ticks + update metrics and appends launch/every5 updates.
- Final supervisor **970521 / start ticks874204998**, detached and same incarnation
  verified at20:13; start20:12:15+08. Admission sample free7571MiB, so **WAITING**,
  no dry/pilot child. Only this supervisor will admit its own dry/pilot processes.

## Dry launch ACK / baseline in progress
- **21:22:15 +08** admitted on gpuA at **21411MiB free**. Own dry trainer
  **PID985766 / start ticks874675005**, same incarnation verified after disconnect.
- Output: `/root/autodl-tmp/c4/candB_pilot_final_run_EXPLORATORY_20261030/dry_EXPLORATORY`.
- `pilot_dry_start.log`: authority independently verified, explicit new-phase
  entrant-mix migration, residual output weight/bias exactly **0.0** after reset.
- 22:06 snapshot: alive, `RUN_SUMMARY.status=initializing`, EXPLORATORY label present;
  paired frozen baseline still computing, **0 optimizer rows**. No update speed,
  CUDA peak, ES acceptance or efficacy number is claimed before a completed update.
- gpuA/gpuR checks continue at >=600s spacing; snapshots `pilot_observation_*.json`.
  Local observer appended the dry launch ACK to shared REGISTRATION.md.

## Fresh-session recovery / owner 22:30 amendment
- 22:29:41 +08 observation: supervisor970521 and dry985766 are the same live
  incarnations; no pilot dispatched and no optimizer row yet. Preserve all prior work.
- Owner22:30 supersedes dry2: first finite, durably receipted dry update triggers own
  STOP and an immediate fresh-zero pilot. The existing supervisor requires dry2;
  replace only its orchestration, retaining the running trainer and immutable runtime.
- Existing source/authority/CPU checks above are complete and are not repeated.
   New orchestration must avoid duplicate dispatch and preserve >=600s GPU sampling.

## Fresh-session recovery / owner 22:40 amendment (current)
- 22:38:36 +08: supervisor970521 and dry985766 alive, zero optimizer rows,
  frozen baseline still initializing. No handoff controller or fresh pilot exists.
- The 22:40 amendment supersedes the fresh-zero dispatch in the preceding section.
  Preserve the running phase and its expensive baseline; ordinary full checkpoint
  continuation must retain Adam, head, original phase clock and R4 at phase10.
- 22:40 recovery: own chain-root STOP disables the old supervisor's automatic
  fresh-pilot dispatch; it does not interrupt its dry trainer or baseline work.
- Existing default-off authority, scientific runtime, staging and 112-test suite
  are complete. Only new pilot operational overlays are being prepared.
