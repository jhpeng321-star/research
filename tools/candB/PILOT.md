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
- New owner2240 overlay CPU contract suite: **10 PASS**, CUDA hidden, 1.21s;
  `pilot_cpu_2240_final.json`. Covers ordinary resume/no new-phase flags, strict
  source specialization, saved phase-clock checks, bootstrap baseline reuse,
  exclusive dispatch, finite receipt checks, memory fallback and boundary latching.
- Continuation reuses the existing independently bound scientific runtime. New
  `pilot_continue*` operational files are separately hashed; original source/input
  manifest and authority are independently reverified before dispatch. Checkpoint
  config (including any entropy annealing) is verified after full restoration.
- Two-worker recipe: coordinator .40 + workers .24 each, aggregate .90 bound;
  admit only without peers and >=43000MiB free. Actual route and seconds/update
  are written per update. Original R4 phase10 comparison is unchanged.
- 22:47:28 +08: committed overlay1803793e staged as NEW pilot files only;
  original da0bc366 source/authority independently PASS again. First stage verify
  refused missing allowed-root environment before any dispatch (retained receipt);
  corrected verification succeeded without altering any scientific source.
- Exclusive same-phase controller **999762/start875186271** now waits on the
  existing trainer. It will request own dry boundary STOP after the baseline is
  durable, validate the finite first row/checkpoint, wait for the original
  supervisor to exit, admit two workers with a >=600s GPU-query interval, and
  ordinary-resume into `pilot_samephase_EXPLORATORY`. No fresh-zero pilot dispatch.
- Staged overlay hashes and ACK: `pilot_deploy_2240_verified.json`; exact retained
  run-root controller receipt `continue_OWNERSHIP.json`; stop future continuation
  using OWN root `continue_STOP` (does not signal any peer PID).
- Pre-dispatch overlay strengthened at **2814934d**: **12 CPU PASS**; requires
  byte-identical retained baseline and exact restored policy/group/critic/Adam
  states, and refuses any saved config drift beyond receipted entropy annealing.
  Narrow operational overlay revision receipts retain before/after hashes;
  original scientific runtime/source/config remain unchanged.
- 22:54 +08 own trainer985766 still active (CPU time1h36m47s), baseline not yet
  durable/zero optimizer rows; no numerical training evidence claimed.
- `python -B -m tools.candB.pilot_watch_2240 --root REMOTE_CHAIN --ledger SHARED_REGISTRATION --spawn`
  starts an exclusive durable local observer. It samples artifacts every600s,
  appends each update here and every5 to the ledger, saves full-state/baseline
   proof, and emits `pilot_update3_ACK.json` only after phase3 is actually observed.

## Owner23:05 fresh-session recovery
- 23:04:16 +08 exact-incarnation recovery: original dry985766/start874675005
  remains runnable, CPU ticks641792; original supervisor970521 and exclusive
  continuation controller999762 remain live. No continuation trainer dispatched.
- Frozen baseline is still not durable, zero optimizer rows. Preserve this work;
  no fresh launch, no repeated authority/staging suite, no extra GPU query.
- Existing local observer43860 (creation22:56:57, exact module/root/ledger command)
  independently verified alive. It remains the sole telemetry/registration bridge.
- Last supervisor GPU sample23:02:15 free41371MiB. Sampling remains600s.
- Recovery receipt `pilot_recovery_2305_1793372657.json`; read-only collector
  `pilot_recover_2305.py` touches only named own-run receipts and owned process metadata.
- Owner23:05 parallelism amendment is being prepared before the pending clean
  boundary. SPEEDPROF currently publishes diagnosis only, no exact application diff.

### Owner23:05 boundary overlay CPU verification
- `pilot_cpu_2305.json`: **14 CPU PASS**, CUDA hidden,1.25s. Covers original
  same-phase/full-Adam/baseline proofs, boundary latching, four-worker refusal,
  three-worker memory bound and exact synthetic group-credit decomposition.
- Request up to4 ES workers; select largest count with `.24` allocator each plus
  768MiB per context, capped at **min(90% device,45GiB)**. Four cannot fit on48GiB;
  three fit only when the measured coordinator reservation allows; two/serial fallback.
- Dense pool uses3 workers and the largest1..3 VJP slots satisfying the same
  exact bound (`.24` active, `.04` idle). No training/objective arithmetic changes.
- Per-update worker `memory.json` allocator peaks and PIDs are preserved and
  attached to `updates.jsonl`; total device usage remains sampled every600s.
- Added per-group D_G/C_G/sum_D_i/D_all (dense native reference and raw300s CTRL0
  reference separately labelled), confident group-gain rows and final acting-CAV
  executed residual/braking minima. Existing raw dJ, ES acceptance, gradients and
  seconds/update stay intact. The live original dry adapter is preserved.
- **23:08:17 +08 deployed/verified** overlay5118d220, receipt
  `pilot_deploy_2305_receipt.json`. No continuation trainer exists; existing controller
  will import it only after its clean-boundary handoff. Previous operational bytes
  retained remotely. Scientific source da0bc366 independently reverified unchanged.
- Authority file digest remains a089c5fb; verification CLI187cd8d4 is the distinct
  canonical-document digest, not a changed authority file.
- Follow-up CPU verification **16 PASS** (`pilot_cpu_2305_budget.json`): adds
  acting-mask/final-acceleration minima preservation and original phase<=150 budget
  on ordinary continuation (149 remaining after phase1), never150 additional.
- Budget revision33876924 independently staged before dispatch; exact before/after
  hashes in `pilot_continue_BUDGET_2305.json`. Shared registration OWNER2305 ACK appended.
- 23:12:48 +08: original dry runnable, CPU ticks696882 (progress+55090 ticks since
  recovery); baseline still pending/zero rows. Supervisor's23:12 GPU poll free40909MiB.
- New observer successor `pilot_watch_2305.py` cooperatively stops only the verified
  prior local observer, waits on its exact Windows process handle, then assumes sole
  telemetry/registration ownership. It saves all requested group/braking/worker peaks
  into PILOT and per-update receipts, commits each actual update, and writes phase3
  ACK only after phase3 is actually observed. It cannot dispatch a trainer.
- Native-budget audit before dispatch: the original `load_continuation` already
  converts phase limit150 to149 remaining after phase1. The intermediate33876924
  adapter subtraction would double-subtract; restored original adapter argument150
  before any continuation dispatch. `pilot_cpu_2305_nativebudget.json`: **16 PASS**,
  including execution of the REAL continuation loader on a bound synthetic receipt,
  proving phase_start190/limit150/remaining149. Intermediate evidence retained.
- Corrected native-budget overlay07899bcf staged/hashed before dispatch, receipt
  `pilot_continue_NATIVEBUDGET_2305.json`; scientific source unchanged, no trainer
  had imported the intermediate overlay.
- **23:17:00 +08 observer handoff ACK**: successor16760/creation23:13:43 owns
  `pilot_watch_2305_OWNERSHIP.json`; predecessor43860 independently absent after
  cooperative STOP and exact process-handle wait. First successor observation23:17:01:
  dry0/pilot0, no continuation trainer, no exit. Monitoring through phase3 remains
  **PENDING**, not a completed-update claim. It will continue after phase3 for MAIN.
- Latest SPEEDPROF read still contains diagnosis only, no published exact speedup
   diff to apply. Requested parallelism/telemetry overlay is already staged for the
   pending boundary; no baseline discarded or recomputed.

## MAIN23:50 superseding EXPLORATORY-LEAN relaunch
- Recovered actual live receipts before action; no successor trainer had dispatched.
- Original dry985766/start874675005 and supervisor970521/start874204998 retired
  with verified-incarnation pidfd SIGTERM after own STOP files; controller999762
  exited cooperatively via `continue_STOP`. All three independently confirmed exited.
- Sunk wall **8530.66s (2.37h)**; trainer CPU899965 ticks. **Zero optimizer rows,
  no durable frozen baseline**; retained original runtime/output/logs in place.
- `pilot_lean_retirement_1793375070.json` contains exact retirement accounting.
  Two initial retirement attempts found Python lacks `os.pidfd_open`; no signal
  was issued by those attempts. Stable pidfd syscalls then succeeded.
- Existing local observer receives its own cooperative STOP; new lean monitoring
  will use a distinct root. Latest GPU sample epoch1793374935; no tighter polling.
- Next: merge published SPEEDPROF into a new immutable runtime; fresh zero,
  first4 registered panel starts, first4 CRN futures, two forward workers .28,
  rescore every5 and exact panel reuse/diagnostic skip/profile flags; phase10 R4.

### Lean implementation and CPU proof
- Merged published `c4/speedprof`43942f75 (merge `1919c0d5`).
- New isolated `pilot_lean*` adapter/authority binds the original full16 panel hash
  and its first4 ordered starts, original dense/scoring K8 and first4 cached ES CRNs.
  No resampling, dense estimator/config changes or rankcheck edits.
- CPU real-update proof PASS: K4 ES16 perturbed forwards (M2 antithetic), unchanged
  dense K8 gradient norm and RNG, exact sign reversal. Default refusal/recipe drift,
  lean row/initial-summary labels and phase10 finite/improvement STOP also PASS.
- First CPU attempt24 PASS/3 fixture-field failures; fixture corrected. Retry
  **26 PASS/1 deliberately deselected already-passed real-update proof**,115.40s.
  Disconnected test callers recovered named receipts without duplicate test jobs.
- `pilot_lean_CPU_recovered.json` retains actual retry proof. Independent rescoring
  every5 stays distinct from per-update PRE-optimizer training-start raw dJ telemetry;
  no stale score is represented as a newly measured independent score.

### Capacity-aware lean admission (before any lean training)
- First immutable runtime0ac91e2d:2562 LF files verified, authority687a9953.
  Its waiting supervisor14957/start875612011 observed gpuA9364MiB peer usage,
  free39147MiB at epoch1793375906; no trainer dispatched. Peer9000 is WMREQUAL3,
  native-lane scorer with10000MB budget, started23:37. No peer signalled.
- Waiting supervisor retired cooperatively after proving no trainer. Both its
  runtime and root retained. New v2 runtime admits>=38000MiB and checks every
  pool's measured coordinator reservation + both.28 workers + CUDA contexts +
  sampled peer memory +2GiB peer growth reserve <=min90% device/45GiB.
  It refuses an unsafe pool rather than overcommitting. New bound CPU test1PASS.
- Ordered original panel prefix (index/world/tick):39780/w963552m1900p05t300/600;
  45082/w963625m1900p10t300/2240;35565/w963495m2400p20t300/900;
  64945/w963904m2400p40t300/1700. Original panel SHA3eaa69a1 retained; only
  evaluation prefix reduced. Original K8/scoring/dense preserved, ES uses0..3.
- v2 source41bd856f/authority09ca8154 verified and waiting supervisor16415
  cooperatively retired BEFORE dispatch. During its admission wait, transport audit
  found executed-tick telemetry must be flushed from isolated forward workers,
  since the coordinator's replay cache has no physics callbacks. v3 adds only
  read-only worker aggregate receipts, captured at graceful own-pool exit, excluding
  perturbed ES and dense replays from the unperturbed executed-stat denominator.
  Worker transport/final acting-mask + lean/R4 contracts **7 CPU PASS**,1 previously
  passed heavy real-update proof deselected,1.30s. No baseline or updates discarded.

## ACTIVE EXPLORATORY-LEAN runtime / fresh launch
- Source **38c7e22f3f9da3d81026833e3ca3429b6267f330**,2573 committed LF files,
  gpuA `/root/autodl-tmp/c4/candB_pilot_EXPLORATORY-LEAN_v3_20261030`.
- Authority **f3fc51795a94927009dc2d865c2a89c47f0d49b982fd890a40545e96868f84ec**;
  original WM/config/input/mix/parity/source identity independently PASS, plus
  bound lean recipe. `pilot_lean_v3_prepare_receipt.json` retains panel identities.
- **00:09:38 +08 Oct31** owned supervisor17854/start875679254 and trainer
  **17859/start875679269**; same incarnations independently verified after SSH exit.
  Admission38277MiB free, peer16680 uses10224MiB, untouched. Resource samples>=600s.
- Run root `/root/autodl-tmp/c4/candB_pilot_run_EXPLORATORY-LEAN_v3_20261030`,
  trainer output `pilot_EXPLORATORY-LEAN`; own root STOP prevents future work and
  requests its trainer boundary STOP. Fresh zero output weight/bias exactly0.0,
  default-OFF explicit authority and lean run-summary stamp verified in live log.
- Startup and updates1-3 **PENDING** until actual receipts. Durable observer records
  launch/every update, appends startup/update1-3/every5 to registration, commits
  actual observations, and emits `pilot_lean_update3_ACK.json` only after phase3.
- Dense/scoringK8 retained; only antithetic perturbed ES first4 CRNs. No shortened
  long horizon (300s), new SUMO, rankcheck edit, peer signal, deletion or push.
- Durable local observer **42388/creation00:10:44 +08**, exact module independently
  verified live; previous16760 absent after its own STOP. Observer cannot dispatch
  trainers; it samples artifacts60s and leaves GPU queries to the600s supervisor.
  Launch ACK already appended/committedb5e096c8; design/retirement/panel/observer
  registration receipt `pilot_lean_DESIGN_REGISTERED.json` appended00:13 +08.

### EXPLORATORY-LEAN launch ACK
- gpuA owned trainer `{'label': 'EXPLORATORY-LEAN', 'pid': 17859, 'start_ticks': '875679269', 'command': ['bash', 'tools/candB/pilot_lean_launch.sh', '/root/autodl-tmp/c4/candB_pilot_run_EXPLORATORY-LEAN_v3_20261030/pilot_EXPLORATORY-LEAN', '--exploratory-owner-20261030'], 'output': '/root/autodl-tmp/c4/candB_pilot_run_EXPLORATORY-LEAN_v3_20261030/pilot_EXPLORATORY-LEAN', 'epoch': 1793376578.5352132, 'admission': {'epoch': 1793376578.5335195, 'total_MiB': 49140, 'free_MiB': 38277, 'used_MiB': 10234, 'apps': '16680, 10224'}, 'same_incarnation': True, 'state': 'R', 'cpu_ticks': 6674, 'argv': '/root/autodl-tmp/c4/candB_rankcheck_0264b967_20261030/venv/bin/python -B -u -m tools.candB.pilot_lean --exploratory-owner-20261030 --authority authority_gen/PILOT_EXPLORATORY_LEAN_AUTHORITY.json /root/autodl-tmp/c4/candB_pilot_run_EXPLORATORY-LEAN_v3_20261030/pilot_EXPLORATORY-LEAN '}`; original dry phase retired, sunk8530.66s.

### First actual scoring component timing
- Initial independent K8 score: **660.889s wall / 696.490s CPU** (update191, fresh phase). This is NOT startup or seconds/update.
- Exact trainer incarnation remains live at epoch1793377628.8688636; completed phase rows=[]; baseline durable=False. Startup/update1-3 and phase3 verification remain pending actual receipts; sole durable observer42388 remains responsible. `pilot_lean_FIRST_SCORE_ACK.json`.

## Owner00:52 fresh-session recovery / WMFAST-B handoff
- Recovered Git history/status and existing run receipts before any dispatch. At
  epoch1793379086, original lean trainer17859/start875679269 and supervisor17854/
  start875679254 remain the same live incarnations; **zero completed updates**,
  frozen baseline not yet durable. Snapshot `pilot_lean_snapshot_1793379088.json`.
- Sole local receipt observer42388/creation2026-10-31 00:10:44 independently verified
  live with exact `tools.candB.pilot_lean_observe` command. No replacement dispatched.
- Last existing supervisor GPU sample1793378979: free36904MiB; trainer6028MiB and
  one5564MiB process. No additional nvidia query was issued during recovery.
- Owner00:52 requests WMFAST-B9d7e7d8e at the first clean boundary AFTER durable
  baseline and >=1 completed update. Preserve current startup; prepare a NEW runtime,
  regenerated authority and ordinary same-phase adapter while it continues.
- WMFAST-B production S4 memory is explicitly UNMEASURED; its report estimates
  ~31.7GiB per worker vs current .28 (~13.4GiB) caps. Continuation must retain aggregate
  <=45GiB/allocator admission guards and choose a measured safe stack/pool recipe.

### Recovered lean startup defect and repair preparation
- At epoch1793379795 the original lean trainer exited naturally, return1:
  `ValueError: native gate requires 16 finite paired costs`. **Zero optimizer rows;
  no durable frozen baseline.** Sunk launch-to-exit3216.88s (53.61min).
- Its saved initial independent K8/300s score survives in RUN_SUMMARY; four native
  panel results were only in process memory and were not persisted by the prior
  adapter. No completed baseline exists to continue or recompute. All output stays.
- A three-case CPU gate test passes: exact four-cost shape/finite checks, all-start
  noninferiority (one harmful start cannot average away), distinct unqualified schema.
- Attempted safe-point receipt rescue was tested ONLY on an owned CPU subprocess;
  the host refused ptrace. No pending call or debugger attachment reached the trainer.
  gdb plus dependencies were installed without package upgrade/removal; rescue
  helpers remain as failed-attempt evidence and are not dispatched again.
- WMFAST-B merged at49b81f38. New operational files prepare: corrected four-start
  startup with each score durably saved; reuse of the SAVED independent score only
  after matching authority/config/seed/input/policy proof; original ES stack OFF
  through update1, then own STOP and ordinary same-phase S2 continuation.
- S2 uses one .50 forward worker because S4 (~31.7GiB projected) has no production
  fit proof alongside coordinator and sampled peer (~5.8GiB). Aggregate admission
  retains measured coordinator+worker caps+contexts+peers+growth <=45GiB/.90.
- Continuation preserves full actor/group/critic/Adam/RNG/TRAIN sampler, original
  phase origin191, frozen baseline bytes, phase10 R4 and the total150-update budget.
  A failed R4, missing durable baseline, nonfinite receipt or zero updates refuses.

### New runtime and CPU verification
- New immutable runtime `/root/autodl-tmp/c4/candB_pilot_WMFAST_EXPLORATORY-LEAN_20261031`,
  sourcee113760fce3b9e10a0e604611675a3f46526c314,2609 LF source files independently hashed.
  Initial transport reset left a directory without the manifest; existing archive
  transport/extraction resumed and full manifest verification then PASS.
- Regenerated/verified lean authoritycd0c9801 and WMFAST continuation authority1edba7a9;
  original source/config/input/WM/mix/parity checks retained. Full receipts
  `pilot_wmfast_prepare_receipt_1793380425.json` and subsequent status receipt.
- **13 CPU PASS**,1.51s,CUDA hidden; actual source-specialized continuation compiles,
  original phase/budget and R4 refusal, memory guard, four-start gate, and a real
  saved-score reuse call proving NO forward occurs for the durable initial score.
  Remote CPU_RESULT.json has return0; one SSH status timeout retained, CPU job not repeated.
- Previous local observer42388 independently absent after recording the original exit;
  prepare one successor for the repaired chain and its same-phase continuation.

### Repaired startup launch / saved score actually reused
- 01:17:34 +08: supervisor35923/start876086871 and repair trainer35928/start876086884,
  exact incarnations independently live after disconnect. Admission42671MiB free;
  peer25602 uses5830MiB, untouched. Sourcee113760f / leanauthoritycd0c9801.
- New chain `/root/autodl-tmp/c4/candB_pilot_run_WMFAST_EXPLORATORY-LEAN_20261031`;
  startup reuse proof is already durable: exact policy/config/input/seed bindings,
  original phase191, previous summary SHA7145f192, only three published WMFAST
  scientific-module diffs and stack flag still0. Saved score reused WITHOUT forward;
  `pilot_startup_score_0.json` durable. Native panel4 remains in progress.
- Sole successor local observer1456 records launch/every phase, startup/update1-3
  and every5 to registration, and phase3 ACK only on an actually observed phase3.
- Pre-boundary audit caught the old operational continuation assuming every row
  has `independent_J`, incompatible with approved rescore-every5. Prepare v2 NEW
  runtime/controller while repair continues unchanged: preserve the last ACTUALLY
  measured score as CLI `before`, with skipped rows still explicitly unscored.
  It waits for existing repair/supervisor exit; no concurrent or duplicate trainer.

### Corrected boundary waiter staged and verified
- v2 immutable source61bfa61236bbce4054994294cbe17350f1dfdccf,2625 LF files;
  `/root/autodl-tmp/c4/candB_pilot_WMFAST_v2_EXPLORATORY-LEAN_20261031`.
  Regenerated WMFAST authoritycc7dfb5b independently verified. **14 CPU PASS**,1.56s,
  including actual CLI source specialization plus skipped-row last-measurement reuse.
- Exclusive controller37982/start876126751 waits on existing trainer35928 and
  supervisor35923. It does not poll nvidia while the existing supervisor owns it.
  After their clean exit it requires finite phase1 and the durable original baseline,
  then assumes >=600s resource sampling and launches one ordinary continuation.
- The first staged supervisor's skipped-score assumption will refuse at its clean
  update1 boundary. It cannot launch a second trainer; corrected controller waits
  for that supervisor to exit before dispatch. All original scientific/native gates
  remain enforced. `WMFAST_CONTROL_STOP` stops only this owned future continuation.
- Startup/update1-3 and before/after WMFAST timing remain PENDING actual receipts.

### Fresh-session recovery at 02:00 +08 Oct31
- Read Git status/log and launch/authority/registration receipts before action.
  Existing exploratory mode, CPU proofs, SPEEDPROF/WMFAST merges and both immutable
  runtime stages are complete and retained; no duplicate dispatch or repeated baseline.
- `pilot_recovery_0155_1793383248.json`: repair trainer35928/start876086884,
  supervisor35923/start876086871 and exclusive WMFAST controller37982/start876126751
  independently verified SAME live incarnations. No continuation trainer, STOP,
  controller failure or run exit exists. Trainer CPU272800 ticks shows progress.
- Sole local observer1456/creation01:17:44 independently verified exact live command.
  Latest observer receipt has four durable startup score files, zero optimizer rows,
  baseline not yet durable; update1-3 timing/efficacy are still pending actual receipts.
- Existing supervisor sample1793383055: gpuA34082MiB free; own trainer8584MiB,
  peer25602 uses5830MiB. Recovery issued no nvidia query;600s cadence remains owned
  by the existing supervisor, then by the exclusive clean-boundary controller.

- S-PILOTB EXPLORATORY-LEAN REPAIR LAUNCH 2026-10-30T17:17:46.733750+00:00: {"label": "EXPLORATORY-LEAN", "pid": 35928, "start_ticks": "876086884", "command": ["/root/autodl-tmp/c4/candB_rankcheck_0264b967_20261030/venv/bin/python", "-B", "-u", "-m", "tools.candB.pilot_wmfast_launch", "repair", "--parent", "/root/autodl-tmp/c4/candB_pilot_run_EXPLORATORY-LEAN_v3_20261030/pilot_EXPLORATORY-LEAN", "/root/autodl-tmp/c4/candB_pilot_run_WMFAST_EXPLORATORY-LEAN_20261031/pilot_EXPLORATORY-LEAN"], "output": "/root/autodl-tmp/c4/candB_pilot_run_WMFAST_EXPLORATORY-LEAN_20261031/pilot_EXPLORATORY-LEAN", "epoch": 1793380654.687814, "admission": {"epoch": 1793380654.6868193, "total_MiB": 49140, "free_MiB": 42671, "used_MiB": 5840, "apps": "25602, 5830"}, "alive": true, "state": "R", "cpu_ticks": 841, "argv": "/root/autodl-tmp/c4/candB_rankcheck_0264b967_20261030/venv/bin/python -B -u -m tools.candB.pilot_lean_repaired --failed-parent /root/autodl-tmp/c4/candB_pilot_run_EXPLORATORY-LEAN_v3_20261030/pilot_EXPLORATORY-LEAN --exploratory-owner-20261030 --authority authority_gen/PILOT_EXPLORATORY_LEAN_AUTHORITY.json /root/autodl-tmp/c4/candB_pilot_run_WMFAST_EXPLORATORY-LEAN_20261031/pilot_EXPLORATORY-LEAN "}

- S-PILOTB EXPLORATORY-LEAN BASELINE DURABLE epoch1793383809.894007: repaired startup 2949.823761s (49.16min), original phase191; four fixed native/gate starts, saved independent K8 score reused; zero raw dense/300s dJ vs CTRL0. Frozen baseline SHA256 744b30b809af60fb11acc54734cbc29bf92f6163793cd306b040ee26e430d680; all five startup scores durable. Update1 has begun under original two-worker/.28 ES-stack-OFF recipe; WMFAST waits for >=1 finite completed update and clean checkpoint/old-process exit. No baseline recomputation or extra GPU query. Full receipt `E:/agent-work/c4-candB/tools/candB/pilot_repaired_STARTUP_ACK.json`.

### Peer-free gpuA / S4 boundary preparation
- Existing supervisor sample1793385456 (02:37:36 +08) contains only owned
  trainer35928 and workers53220/53221; peer25602 has exited independently.
  Free31339MiB WHILE our update is running. No extra nvidia query.
- Prepare NEW `pilot_wmfast4_*` files/runtime: same verified ordinary continuation,
  original baseline/phase191/head/Adam/RNG/sampler/R4phase10 retained. Execution-only
  S4 with ONE .75 forward worker; TWO .24 dense workers/VJP slots, same ordered tasks
  and dense algebra. No live trainer/runtime edit; S4 still waits for finite phase1.
- New waiter requires >=46000MiB free after old trainer/supervisor exit. Every pool
  retains the measured coordinator+worker allocator caps+contexts+sampled peers+2GiB
  <=min(90% device,45GiB) guard. At coordinator2GiB and no peers, S4 fits; a6GiB peer
  or7GiB coordinator REFUSES. S4 production peak/timing remains unmeasured until run.
- CPU proof and staged source/authority verification precede cooperative retirement
  of the old S2 waiter37982 only; own STOP/old-incarnation exit/exclusive launch proof
  required. Current repair trainer/supervisor and frozen baseline remain untouched.
- S4 NEW immutable runtime staged via `stage.sh`:
  `/root/autodl-tmp/c4/candB_pilot_WMFAST_S4_v3_EXPLORATORY-LEAN_20261031`,
  sourceaf78fda7bc1aa1a6d3b1c6745c1a1cc14c83ad7e,2644 LF files individually verified.
  Regenerated exploratory authority07f5e05bebb054e8d1b256c083b54eddda9154f51a4291df16451bb366c42cc2
  independently verified; original config/input/WM/mix/parity checks retained.
- **10 CPU PASS**,1.707s wall,CUDA hidden, includes real launcher/controller source
  specialization, S4 environment binding/restoration, high-coordinator/peer budget
  refusals, original phase/R4 validation and saved-score no-forward proof. Existing
  numerical S2/S4 parity and original full-resume proofs remain applicable.
  Receipts `pilot_wmfast4_{prepare_1793385923,cpu_1793385931,status_1793385944}.json`.

- S-PILOTB EXPLORATORY-LEAN UPDATE1 FAILED epoch1793385850: TypeError: 'NoneType' object is not subscriptable in dense autograd backward; ZERO completed optimizer rows. Frozen baseline744b30b8 remains durable; all original outputs/startup scores retained. Trainer35928 and supervisor35923 exited naturally; old S2 waiter37982 and prepared S4 waiter60032 both refused the non-clean parent, no continuation trainer dispatched. S4 runtimeaf78fda7 authority07f5e05b and10 CPU contracts passed but S4 production not launched. Investigating a baseline-preserving recovery; NO baseline recomputation or completed-update claim. Full traceback/receipt `E:/agent-work/c4-candB/tools/candB/pilot_repaired_UPDATE1_FAILURE.json`.

## Owner03:00 recovery — exact B-L replay fix / baseline-preserving retry
- Recovery epoch1793386792: original trainer35928, supervisor35923, S2 waiter37982
  and S4 waiter60032 are all independently absent. Neither continuation dispatched.
- Baseline SHA256 **744b30b809af60fb11acc54734cbc29bf92f6163793cd306b040ee26e430d680**
  independently rechecked; zero completed optimizer rows. Original zero191 checkpoint
  `policy_native_best.pt` SHA256 **abae6464e2ef0fbb08174b7d4b079b4bed634bbae7207ec21560f5d3f7873c02**.
- Disk-full tracked-file audit: `git diff --numstat` has only2 additions/1 deletion
  in the retained pilot_wmfast4.py edit; no truncated tracked files or mass deletions.
  Existing untracked recovery drafts are retained, including empty untracked
  pilot_wmfast_detail.py; they are not assumed verified or deployed.
- B-L c73498cd exact observational-wrapper passthrough repair is the recovery source;
  its later independently-read authority/checkpoint repair2ac8ff65 is also required.
  Owner03:00 now permits batching ON for repaired update1. New runtime/authority,
  explicit zero-progress full-state recovery and original phase10 R4 are required.
- Full named-receipt/process audit: `pilot_recovery_0300_1793386792.json`.

## Owner03:00 fresh-session recovery — exact B-L fix port committed
- Working-tree src diffs verified BYTE-IDENTICAL to B-L c73498cd hunks
  (blob hashes `5852f522..bed58b9e` imagination_memory passthrough,
  `c7c0f397..1e285f78` policy14_recovery initial_checkpoint; only commit-header
  text differed in the compared patches). No truncated tracked files (numstat
  audit above holds; 6 small diffs, zero mass deletions).
- Telemetry adapters (pilot_cli, pilot_lean, pilot_telemetry_2305) mark their
  observational wrappers `measured._wms_checkpoint_passthrough = callback`;
  `imagination_memory._controller_cells` then snapshots/releases ONLY the
  underlying physical controller, fixing the update1 dense-VJP
  `'NoneType' object is not subscriptable` at its single clearing site
  (`release_controllers`, line 245).
- New `pilot_replay_guard.py` (+ regression test): dense replay (serial _vjp and
  parallel VJP worker snapshots) uses the bare core controller factory so
  persistent telemetry closures never enter disposable replay graphs;
  `pilot_wmfast4.run` now wraps its continuation in the guard.
- Frozen-session untracked retry drafts (pilot_retry.py 03:02, retry authority,
  replay guard, baseline_retry 02:54) audited: retry requires S4 stack, binds
  PARENT/baseline744b30b8/checkpoint abae6464/seed965140, zero-progress
  `initial_checkpoint=True` recovery, byte-identical baseline copy, exact
  two-file source-drift identity, R4 phase10. Being completed/tested next;
  empty pilot_wmfast_detail.py retained untracked, imported by nothing.

## Owner03:00 retry tooling committed (CPU proof pending on gpuA runtime)
- Committed retry chain: `pilot_retry.py` (+authority), `pilot_retry_launch.py`
  (lean launcher specialized to S4 env: ES_STACK_SIDES=4, 1x.75 forward worker,
  2x.24 VJP slots, PILOT_RETRY_AUTHORITY.json, admission>=46000MiB, no
  --new-phase), `pilot_retry_supervise.py` (single exclusive dispatch, 600s
  GPU sampling, own STOP/EXIT receipts, old WMFAST-root incarnation absence
  asserted before dispatch), `pilot_retry_test.py` + `pilot_retry_cpu.py`
  (CUDA-hidden harness; asserts return0 before dispatch).
- CPU contracts include: real zero-progress `load_continuation(
  initial_checkpoint=True)` (updates==[], saved==start==191 -> remaining150;
  initial_checkpoint=False refuses), exact-two-file B-L source identity with
  tamper refusals, launcher/authority S4+owner03:00 binding, zero-progress
  `last_measured_score` fallback to `independent_score_before`, prepare_base/
  adapter replacement-target exactness, and a REAL marker-wrapper multi-update
  regression proving telemetry survives replay release WITHOUT any guard
  (bitwise losses/grads/Adam/RNG vs bare core).
- src drift vs the e113760f baseline runtime verified EXACTLY the two repaired
  files (retry identity precondition).

## Owner03:00 retry verification complete (v4 runtime)
- Runtime `/root/autodl-tmp/c4/candB_pilot_RETRY0300_v4_EXPLORATORY-LEAN_20261031`,
  source **71c802e4d9a1f989539fc1205016a792adbaa6bd** (2,691 committed LF files,
  every manifest hash verified), retry authority **3af8be9e42d60f31c79068a9afdf8c0d27444142f613249109f27d4d82f55419**;
  panel3eaa69a1 first-4 starts, S4 env, owner03:00 amendment, continuation_only
  false. Earlier v1-v3 attempts retained (fixture/scope/test-logic repairs only;
  no scientific difference between v1-v4 sources).
- **CPU PASS 17/17** (`pilot_RETRY_CPU_RESULT_1793390445.json`, 4.12s, CUDA
  hidden): retry_test(8) + replay_guard_test(2) + continue_test(7). Includes the
  real marker-wrapper multi-update regression (bitwise losses/grads/Adam/RNG vs
  bare core WITHOUT any guard — the exact production failure topology), real
  zero-progress `load_continuation(initial_checkpoint=True)`, exact-two-file
  B-L identity with manifest/tamper refusals, launcher/authority contracts.
- **B-L's own fix tests rerun in THEIR gpuR runtime** (read-only, CUDA hidden):
  `tests/unit/test_candbl_continue.py` + `test_candbl_lean.py` at their
  a19ffda9 (includes c73498cd+2ac8ff65): **14 passed** in 8.25s.
  Receipt `pilot_BLTEST_FOR_RETRY.json`.
- Parent re-verified: RUN_SUMMARY failed/TypeError/seed965140/rows0/zero_start/
  resumed191; `independent_score_before` present (4694.833); native_curve [191];
  gate_panel J_before4 passed-null; policy_native_best.pt abae6464...; frozen
  baseline 744b30b8... — all retry bindings intact.
