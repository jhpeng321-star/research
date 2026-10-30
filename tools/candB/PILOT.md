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

### EXPLORATORY-LEAN launch ACK
- gpuA owned trainer `{'label': 'EXPLORATORY-LEAN', 'pid': 17859, 'start_ticks': '875679269', 'command': ['bash', 'tools/candB/pilot_lean_launch.sh', '/root/autodl-tmp/c4/candB_pilot_run_EXPLORATORY-LEAN_v3_20261030/pilot_EXPLORATORY-LEAN', '--exploratory-owner-20261030'], 'output': '/root/autodl-tmp/c4/candB_pilot_run_EXPLORATORY-LEAN_v3_20261030/pilot_EXPLORATORY-LEAN', 'epoch': 1793376578.5352132, 'admission': {'epoch': 1793376578.5335195, 'total_MiB': 49140, 'free_MiB': 38277, 'used_MiB': 10234, 'apps': '16680, 10224'}, 'same_incarnation': True, 'state': 'R', 'cpu_ticks': 6674, 'argv': '/root/autodl-tmp/c4/candB_rankcheck_0264b967_20261030/venv/bin/python -B -u -m tools.candB.pilot_lean --exploratory-owner-20261030 --authority authority_gen/PILOT_EXPLORATORY_LEAN_AUTHORITY.json /root/autodl-tmp/c4/candB_pilot_run_EXPLORATORY-LEAN_v3_20261030/pilot_EXPLORATORY-LEAN '}`; original dry phase retired, sunk8530.66s.
