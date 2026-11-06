<!-- S-GROUPDIAG-closeout -->
**2026-11-06, S-GROUPDIAG:** [Clean2 group-cooperation research](runs/active/C4-V7FT-20261017/CLEAN2_GROUP_COOP_RESEARCH.md): B's +726.400 veh-s is reproduced by state-specific best solos (+726.434); synergy -0.034. Joint opportunities, executable teacher fidelity and second-helper credit are the key gaps. Existing receipts/literature only; zero new SUMO or charges.
<!-- /S-GROUPDIAG-closeout -->

<!-- S-DIAG-closeout -->
**2026-11-06, S-DIAG:** [Clean1 diagnosis through final u9](runs/active/C4-V7FT-20261017/CLEAN1_DIAGNOSIS_20261105.md): B is not stuck at none; sampled named braking offsets have weak, reversed instantaneous actuation. Bank/practice concentration and credit mismatch constrain cooperation. Clean1 owner STOP preserved; zero new SUMO, training or charges.
<!-- /S-DIAG-closeout -->

<!-- S-WMCHECK-closeout -->
**2026-11-05, S-WMCHECK:** [Existing-plan WM error audit](runs/active/C4-V7FT-20261017/WM_ERROR_CHECK_20261105.md): positive WM gain wins 90/221 on 542 recovered training pairs. The >=80%-confidence question remains unresolved because paired probabilities were not recorded; mixed horizons and selection bias limit interpretation. Zero new SUMO or charges.
<!-- /S-WMCHECK-closeout -->

> **Current direction (2026-10-09):** the project is **WM-S**, a physics-informed
> per-step traffic world model (IDM/MOBIL core + gated learned residual) used to pick
> controllers per situation. Contribution **C1** (zero-shot prediction of unseen
> controllers) is verified on the PI arm and, on 2026-10-09, on the unseen **WALL**;
> C2 (per-driver identification), C3 (fragility, deferred) and C4 (learned cooperative
> multi-CAV control, later) follow. This week: brake-fix verdict, the pack_v3
> zero-window fix, the entries-ON eval, then one GPU run with loss rebalancing; walls
> are one action and **regret** is the primary planner score. See
> [docs/ROUTE.md](docs/ROUTE.md), [docs/WM_STATE_PLAN.md](docs/WM_STATE_PLAN.md) and
> [docs/WM_S2_ROADMAP.md](docs/WM_S2_ROADMAP.md). The WM-N1 recovery blocks below are
> historical and preserved in [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md) and
> [docs/INDEX.md](docs/INDEX.md).
>
> Change history of every world-model change tried: [docs/WM_CHANGE_LOG.md](docs/WM_CHANGE_LOG.md).
> Latest overnight status: [docs/records/morning_report_20261010.md](docs/records/morning_report_20261010.md).

Label fix + timing audit (2026-10-03): [FIX1](runs/active/WM-N1-LABEL-TIME-FIX1-20261003/REPORT.md). The switching-run label lookup is fixed (196 chunk2/3 campaign switching items rebound to their own swnative/swpi labels; 0 excluded), and the corrected TRAIN labels pass every identity gate. The training mapper chose a later decision time than t0 in all 5,480 labelled cells. Owner approved (a) the t0 mapper, (b) excluding 263 base/native_base items from outcome supervision, and (c) the 10 s eval correction window. M1 retrain package: [RETRAIN_READY](runs/active/WM-N1-LABEL-TIME-FIX1-20261003/RETRAIN_READY.md); GPU50 smoke is the only training-path test. M1 unchanged.

Full TRAIN timing audit (2026-10-03): [stopped at corrected policy gate](runs/active/WM-N1-LABEL-TIME-FULL-20261003/REPORT.md). With source-cited policy correspondence, 900 labelled items map, but 196 chunk2/3 campaign switching items (13,524 rows) carry true-fleet labels (native_fleet/pi_fleet): suspected label-binding defect. Last-row-per-tau timing not measured; exclude/rebind decision and any further read need owner approval. M1 unchanged.

TRAIN identity diagnostic (2026-10-03): [exact row and policy-name resolution](runs/active/WM-N1-LABEL-TIME-DIAG-20261003/REPORT.md). Global row69 uses label class damping_fleet versus receipt controller damping; frozen source explains this naming difference. The prior exact-string audit check was overstrict for this row. Full TRAIN timing remains unresolved; another read needs owner approval. M1 unchanged.

Current TRAIN audit (2026-10-03): [identity-gate stop](runs/active/WM-N1-LABEL-TIME-20261003/STOP_REPORT.md). Member1 completed training, but prediction accuracy and label-timing validity remain unestablished. The TRAIN metadata projection stopped on identifier disagreement; no retraining has started. Real scoring remains disabled; the evaluation source foundation was already owner-approved and generated-test reviewed.

<!-- WM-N1-RECOVERY-01-resume -->

<!-- coordinator-status-20261002 -->
Member 1 completed all 30,000 CUDA training steps. Checkpoint b37a25a7 is independently hash-verified and preserved under runs/active/WM-N1-M1-20261003; all 30,000 training records pass the owner-approved loss checks. The completed run remains attributed to 45c9032 plus its documented box-side CUDA patch.

Source branch zcode/wd01 is now d96b57d: the four CUDA host copies and pandas 2.3.3 pin are upstreamed, and the J4 campaign/frozen wall-factor selector and G6 metadata report are corrected. The host-copy regression passed on CPU (CUDA unavailable), and all 28 guarded synthetic J4 checks passed.

TTC15/A7 remains unmerged because of integration defects. The real-checkpoint evaluation bridge is incomplete; its owner-approved source foundation passed generated checks and real scoring remains disabled. The frozen J4-only held-out protocol also conflicts with the completed full-TRAIN M1 and needs owner resolution before such a claim. CAL/J4B access, INC1, PL-WM1 and checkpoint cloud backup remain gated. No model accuracy or held-out outcome result is claimed.
<!-- /coordinator-status-20261002 -->

**WM-N1-RECOVERY-01 resumed by owner (2026-10-02):** Owner-selected Sol repairs and guarded acceptance continue; real TRAIN stages remain separately registered. Prior split-metadata exposure remains disclosed; real40/GPU and main restructuring are still held.
<!-- /WM-N1-RECOVERY-01-resume -->

<!-- WM-N1-RECOVERY-01-stop -->
**WM-N1-RECOVERY-01 stopped (2026-10-02):** a completed executor test read the frozen CAL split world-name list despite the prohibition. Four workers stopped; candidates and logs preserved. Zero real training/SUMO; GPU readiness held pending owner direction and guarded review. [Stop receipt](tmp/zcode_review/WMN1_CAL_SPLIT_EXPOSURE_20261002.md).
<!-- /WM-N1-RECOVERY-01-stop -->

# Traffic Shepherd — WM-S World Model for Per-Situation Controller Selection

> **Current route (2026-10-09):** [docs/ROUTE.md](docs/ROUTE.md). The WM-S plan is
> [docs/WM_STATE_PLAN.md](docs/WM_STATE_PLAN.md) and the contributions/weekly plan are
> [docs/WM_S2_ROADMAP.md](docs/WM_S2_ROADMAP.md). Historical WM-N1 recovery status is
> preserved in [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md) and
> [docs/HANDOFF_20261002.md](docs/HANDOFF_20261002.md).

A SUMO research pipeline for a highway merge bottleneck. A small fraction of
connected automated vehicles (CAVs) report what they see to a road-side unit (RSU).
The goal is to **reconstruct the traffic state from those sparse reports and decide
which literature-supported controller should regulate traffic right now** — native
IDM driving, the published CIRCLES RL damping controller, a PI controller with an
external target speed, or a moving-bottleneck "wall" — without ever using hidden
simulator ground truth as an online input. The current approach is **WM-S**: a
physics-informed per-step world model whose HDV mechanism is learned while CAV
controller code is executed exactly, so unseen controllers can be evaluated
zero-shot (contribution C1).

> **Branches.** `main` (this page) contains only the current work.
> The complete earlier history — the pre-CIRCLES learners, the damping/moving-wall
> escalation studies (OE-series), PPO pilots and every original test — is preserved
> on the [`previous-work`](https://github.com/jhpeng321-star/traffic-shepherd/tree/previous-work) branch.

## Research questions

1. **Controller verification (CV-series).** Do literature controllers (FollowerStopper,
   PI with saturation, CIRCLES damping, native IDM) behave as published inside this
   SUMO plant (FollowerStopper retired 2026-09-25: removed from the code and no
   longer an action; its CV-03..CV-06 evidence is historical; the per-lane
   mix layer retired 2026-09-25 WM-39 - actions are per-CAV, walls are a
   gated group action), and is there *state-dependent* control opportunity — i.e. does the best
   controller change with the traffic state?
2. **World model (WM-series).** Can the RSU reconstruct a spatial traffic state from
   timestamped CAV self-reports and neighbor detections, and roll it forward
   conditioned on the controller choice?

## Current status (2026-10-09)

- **The model is WM-S**, a physics-informed per-step next-state world model (IDM/MOBIL
  physics core + gated learned residual). CAVs are moved by the real controller code,
  so unseen controllers can be evaluated zero-shot without retraining. Plan:
  [docs/WM_STATE_PLAN.md](docs/WM_STATE_PLAN.md); contributions:
  [docs/WM_S2_ROADMAP.md](docs/WM_S2_ROADMAP.md).
- **C1 (zero-shot unseen-controller prediction) is SUPPORTED with caveats.** Verified
  on the PI arm and, on 2026-10-09, on the unseen **WALL** (structured HDV median |ds|
  8.71 m vs bb_traj 25.41 / bb_id 40.07 m at 60 s on `wall_same`;
  `E:/rd-wt/ws-wallc1/_out/WALL_C1_RESULT.md`).
- **Long-horizon error is the open modelling problem.** DIAG120 traced the 60–120 s
  blow-ups to under-braking and missed upstream jams, not phantom jams
  ([docs/DESIGN_DISCUSSION_20261008.md](docs/DESIGN_DISCUSSION_20261008.md)).
- **Braking fix in progress.** Braking-fix and no-braking models were trained
  2026-10-09 ([runs/active/WM-S-MAIN2-20261009/REGISTRATION.md](runs/active/WM-S-MAIN2-20261009/REGISTRATION.md));
  scoring is in progress (`E:/rd-wt/ws-main2score`) and is not yet a result.
- **Data findings in progress.** pack_v3 contributed **0** training windows (fix in
  progress, `E:/rd-wt/ws-packv3fix`); entries-ON rescoring is in progress
  (`E:/rd-wt/ws-entrieseval`). Direct output / per-CAV switching data is the main
  training data.
- **Planner view.** Traffic-level delay differences are more robust than car
  positions: at 300 s the model picks the better of native/PI in ~80% of cases
  (regret 19 vs 128/194 for fixed choices), but closed-loop use is untested. Regret is
  the primary planner score and walls are one action.
- **Retired (do not resume):** FollowerStopper, the per-lane mix layer, wall-only
  planning, the WM-N1 delay head, D: staging (E: only), and self-recovery/pushforward
  training.
- **Raw evidence loss (2026-09-24, MIG-04), partial restore (MIG-05):** the raw
  simulation outputs (`runs/`), pre-CIRCLES raw results and several world-model
  checkpoints were accidentally deleted. The 88 world-model dataset files were
  regenerated as 602-second prefixes and match the original hashes. Full-length
  runs were not rebuilt. See
  [`protocols/deleted_evidence.json`](protocols/deleted_evidence.json) and
  [`output/evidence_regeneration_v1/CLOSEOUT.md`](output/evidence_regeneration_v1/CLOSEOUT.md).
- No trained, closed-loop-validated world model or deployed selector/controller
  exists yet. See [`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md)
  and [`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md).

### Integration and readiness records (clean line, 2026-10-06 to 2026-10-09)

Engineering records carried over from the clean line; the bullets above are the newer research status.

- **WM-S second integration (2026-10-09):** braking response, guarded mixed
  training and unseen-wall preparation merged on `wms/main2`. CPU verification:
  **612 passed, 6 skipped**, plus **136 wall tests** and **28 chain checks**.
  [Readiness, archives and four launch commands](archive/wms_era/records/_out/MAIN2_READY.md).
  Engineering-only verification; the inherited missing-evidence ledger remains
  unchanged in error count.

- **WM-S main integration (2026-10-09):** all seven requested implementation
  branches are merged on `wms/main`, retaining hybrid/history lane heads,
  live action timelines/controllers, receipt-aware and shifted evaluation,
  ranking and optional CTM/wall executors. CPU suites: **438 passed, 6 skipped**;
  skipped real-state/source fixtures are documented. [Merge record](archive/wms_era/records/_out/MAIN_MERGE.md)
  and [READY commands](archive/wms_era/records/_out/MAIN_READY.md). This is engineering verification.
- **WM-S training setup (2026-10-09) prepared and CPU-checked.** Mixed
  slot-bound switching/direct timelines and original packs, leave-PI-and-wall-out
  filtering, zero-shot guards and a four-stage GPU launch chain are implemented.
  The real-data smoke and external split lists remain launch prerequisites.
  [Readiness and commands](archive/wms_era/records/_out/TRAINSETUP_READY.md).

- **WM-S switch model (2026-10-07) is implementation-ready.** Closed-loop
  `wms_eval` executes native IDM, CIRCLES damping and PIExternalU from predicted
  CAV states, including post-t0 entrants assigned the plan's
  `default_entrant_controller` (default `native`). Indexed switch schedules and
  suffix commands bind to stable pack columns through t0 identity sidecars and
  tick-local dense truth; synthetic tests fail closed on unmatched IDs. See
  [`_out/SWITCH_MODEL_READY.md`](archive/wms_era/records/_out/SWITCH_MODEL_READY.md).
- **WALL-C1 preparation (2026-10-09), partial:** schedule-verified leave-PI-and-wall-out
  builder, streamed unseen-wall sets/scoring and fair-split comparator chains are
  CPU-fixture verified. Real sources and canonical guard must be staged inside
  this workspace; the interrupted LPWO lists are verification-pending. No new
  training or simulation. [READY and coordinator commands](archive/wms_era/records/_out/WALL_C1_READY.md).

- **WM-S2 integrated and CPU-verified (2026-10-06).** `wms/s2int` merges
  `wms/s2` (per-driver parameter encoder + `driver_params` rollout),
  `wms/entry2` (entries v2 + trainer entry path), `wms/trafloss`
  (traffic-level losses + 300 s curriculum) and `wms/stab` (feasibility
  projection, bounded residual, conservation) flag-gated default OFF. Added an
  optional driver-parameter ground-truth sidecar loader + recovery evaluation
  (never in the loss) and a gpuA `s2_chain.sh` (smoke400 gate incl. encoder
  bounds, then 10/30/60/120/300 s at ~5 h, hard cap 8 h). The required suites
  pass (298 passed, 2 skipped) and a real-data CPU dry run with
  driver_params + entries v2 + traffic loss + stability passes finite under the
  6 GB memory rule. Status: [`_out/S2_READY.md`](archive/wms_era/records/_out/S2_READY.md).

## Architecture

1 s control step over 0.5 s physics; SUMO speedMode 31; HDVs always drive native IDM.

```text
SUMO merge (frozen network, IDM HDVs + CAVs)
        │  CAV self state + locally detected neighbors
        ▼
RSU report stream        timestamped; receipt age, missingness, coverage
        ▼
Spatial reconstructor    10 m × lane-group grid, observed vs prior provenance
        ▼
World model / planner    controller-conditioned rollout; ties default to native
        ▼
Fleet switch             native IDM │ CIRCLES damping │ PI external-U │ wall-v2
        ▼
CommandOwnership         sole longitudinal writer for CAVs
```

## Repository layout

Target structure after the clean WM-S rewrite (design: [`docs/ARCHITECTURE_V4.md`](docs/ARCHITECTURE_V4.md)).
Nothing is ever deleted: relocated material keeps its history (`git mv`) and every old path
maps to its new location in a `PATH_MAP.json`.

| path | contents |
|---|---|
| `src/wms/` | the WM-S package (physics-informed next-state world model). Sub-packages: `core` (constants, types, registry), `config` (typed dataclasses + TOML loading), `physics` (IDM/MOBIL, batched engine, wall executor), `model` (WmSModel, residual net, checkpoint IO), `data` (datasets, windows, packs, sidecars), `train` (trainer, losses, gates, curriculum), `eval` (states, scoring, verdicts, walls, `decide`), `cli` (`python -m wms ...`), `legacy` (black-box and frozen-A11 baselines kept for scoring old checkpoints). Landed by merging `wms/clean-a..e`. |
| `configs/` | one TOML preset per run (`r3.toml`, `c1fast.toml`, smoke presets) |
| `scripts/box/` | thin launch wrappers for the GPU box |
| `tools/` | repository tools: `experiment_history.py` (ledger), `fetch_data.py` + `data_registry.json` (downloads data into `data_cache/`), `data_guard.py` (WP-C window-list guard), `holdout_guard.py` (planner/reconstructor HOLDOUT guard), research-dependency tools |
| `tests/` | `unit/`, `equivalence/` and `golden/` for `src/wms`; `planner/` and the sim-layer root tests; `fixtures/` |
| `traffic_shepherd/`, `baselines/`, `scenarios/`, `assets/`, `protocols/` | frozen simulation layer, controllers, scenarios, network, registrations and ledger. Do not edit (hash-pinned evidence). |
| `docs/` | start at [`docs/INDEX.md`](docs/INDEX.md); design in `ARCHITECTURE_V4.md`; legacy import notes in `ARCHITECTURE_V4_LEGACY_PATHS.md` |
| `runs/` | new run outputs (`runs/active/`), git-ignored |
| `data_cache/` | packs, sidecars and checkpoints fetched by `tools/fetch_data.py` (never committed; git-ignored) |
| `archive/` | superseded material, all `git mv`-ed with a `PATH_MAP.json`: `circles_era/`, `pre_circles/`, `wms_era/` (legacy WM-S `wms`/`wms_eval` trees, `_out` diagnostics and records, one-off tools), `root_strays_20261010/`, `tests_dead/` (tests whose imports no longer exist) |
| `output/wd01/wm_n1/` | historical WD-01 / WM-N1 tree and its sibling packages; `wms/` and `wms_eval/` there are thin import shims onto `archive/wms_era/wm_n1/` |
| `_out/` | only the live R3 chain inputs (`train_r3_chain.sh`, window lists, sidecars, `r3_box/`); records and diagnostics moved to `archive/wms_era/` |

Most raw simulation trajectories were deleted on 2026-09-24 (MIG-04, see `protocols/deleted_evidence.json`);
superseded code lives under `archive/`, and on GitHub the same material is on the `previous-work` branch.

### Running the WM-S stack (integrated 2026-10-10, [`docs/INTEGRATION_REPORT.md`](docs/INTEGRATION_REPORT.md))

```bash
export PYTHONPATH=src                 # or `pip install -e .`
python -m wms preflight --config configs/r3.toml          # config + every data name resolves/exists
python -m wms train     --config configs/smoke_local.toml --mode run --out runs/active/my_smoke   # CPU smoke
python -m wms score     --config configs/score_smoke_local.toml                                    # vs previous best
python -m wms eval      --config configs/eval_unseen.toml --checkpoint <name-or-path>
```

Data are addressed by logical name (`pack_v3`, `lpwo_v3/windows_train_noPI_noWall.json`, ...) and resolved by
`wms.data.registry`: environment variable `WMS_DATA_<NAME>`, then `configs/data_registry.toml` (the local
development locations are registered there), then `tools/data_registry.json`.

Tests run in two groups because the legacy `wms` package (archived) and `src/wms` cannot share a process:

```bash
python -m pytest                          # new stack (unit, equivalence, golden) + sim + planner
WMS_EQ_FULL=1 python -m pytest tests/equivalence/test_eval_equivalence.py   # 20-state C1 check (minutes)
WMS_TEST_GROUP=legacy python -m pytest --basetemp=_out/pytest_tmp_legacy    # root test_wms_eval_*/test_wms_decide
```

## Setup

Python 3.10+, SUMO **1.27.1** (TraCI is pinned to this version).

```powershell
conda activate rl_driving            # or any env with requirements-dev.txt installed
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
python -m pip install -r requirements-dev.txt

python -m traffic_shepherd inspect   # identity/import checks, no SUMO
python -m pytest -q                  # tests, no SUMO
python -m ruff check traffic_shepherd tests
```

One development evaluation (2,400 s horizon; output directories are never overwritten):

```powershell
python -m traffic_shepherd evaluate --seed 911001 --loading 2000 --penetration 0.2 `
    --role development --controller native --output runs/active/native_s911001_p20
```

## Experiment records

Every simulation is registered before it runs and its outcome appended to
[`protocols/experiment_history.json`](protocols/experiment_history.json).
Browse it with `python tools/experiment_history.py search|card|open-questions|summary`.
The full chronological status log that used to live in this README is in
[`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md).

## Attribution

CIRCLES baseline: see `baselines/circles/` (MIT license and provenance).
FollowerStopper / PI with saturation: Stern et al., 2018. Moving-bottleneck literature:
`output/wall_literature_review_v1/`.
