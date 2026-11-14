# ARCHITECTURE_V4 — clean structure for the WM-S research codebase

Status: DESIGN (2026-10-10, branch `wms/clean`, base = integrated `wms/r3-int`, HEAD `bf7d4cb`).
No code was rewritten to produce this document. It is the contract for the parallel
implementation packages in section 5. Read AGENTS.md first; every rule there still holds
(frozen evidence, sealed-data tokens, no `git reset/clean/stash/checkout --`, nothing deleted).

Companion inputs: `E:/rd-wt/reorg3/layout/CLEAN_END_STATE.md` and
`E:/rd-wt/reorg3/layout/tools/fetch_data.py` (data layout; integrated in section 2.6),
`E:/research-dense/docs/WM_CHANGE_LOG.md` (what not to repeat; integrated in section 3.4).

---------------------------------------------------------------------------------------------

## 0. Decisions in one page

1. **One new package `src/wms/`** replaces `output/wd01/wm_n1/wms` + `wms_eval` (+ the WM-S-relevant parts
   of `lpwo/`, `lpo/`). Sub-packages: `core config physics model data train eval cli legacy`.
2. **`traffic_shepherd/` (the "sim" layer) is NOT moved or edited.** Controller behaviour is frozen evidence;
   a move would invalidate hash-pinned paths. "sim/" in the target layout means `traffic_shepherd/` +
   `baselines/` + `scenarios/` + the three frozen PI controller files under `output/controller_verification_v{2,4,7}/`.
   `src/wms/sim_bridge.py` is the *only* module in `src/wms` that imports them (lazily).
3. **Configs are typed dataclasses + one TOML preset per run** (`configs/r3.toml`, `configs/c1fast.toml`).
   TOML (stdlib `tomllib`, Python >= 3.11) instead of YAML because PyYAML is not installed in `rl_driving`
   and a new dependency buys nothing. The ~90 CLI flags and the 518-line env-var shell chain become
   `python -m wms train --config configs/r3.toml [--set a.b=c ...]`.
4. **Checkpoints stay format `wm_s/1`** (`{"format","model","config","history","val_by_stage",...}`, plain
   state_dict, no pickled classes). New code writes the same keys; `config` stays the *flat legacy dict*
   (plus an extra `config_v2` nested dict). Parameter names inside the state_dict must not change
   (section 4.2 lists the module attribute names that are frozen).
5. **Behaviour preservation is proven by golden tests captured from the CURRENT code first** (section 4).
   No work package may start refactoring physics/loss/model code until `WP-0` has committed
   `tests/golden/ref/`.
6. **Nothing is deleted.** Dead/one-off code is `git mv`-ed to `archive/wms_era/…` with a `PATH_MAP.json`
   entry and, where tests import it, a re-export shim (section 6).
7. **Vendor-then-trim for the legacy data stack.** The pack reader (`train/idea_a.py`, 4250 lines) and the
   helpers WM-S needs from `hdv/`, `features/`, `closedloop/`, `rollout2/` are *copied unchanged* into
   `src/wms/legacy/vendor/` behind narrow adapters now; trimming them is a later, evidence-driven step
   (coverage from the golden runs). This keeps tonight's work bit-exact and parallel.

---------------------------------------------------------------------------------------------

## 1. Inventory

Codes: **A** = on the ACTIVE path (WM-S training / eval / planner), **L** = legacy/historical-replay only,
**X** = not used. Verdict: **KR** keep-rewrite (into the module named), **KA** keep-as-is,
**AR** archive (git mv + PATH_MAP; never delete).
"Callers" were derived by import scan of the worktree (src imports + tests + `_out` scripts).

### 1.1 `output/wd01/wm_n1/wms/` (41 .py, 38 test files, 9 shell/json) — 12.3k lines

| file (lines) | purpose | use | verdict -> target |
|---|---|---|---|
| `__init__` | exports WmSModel, PhysicsConfig, BlackBoxTrajModel | A | KR -> `src/wms/__init__.py` (no BlackBox export) |
| `model.py` (1168) | WM-S: IDM/MOBIL physics core + gated residual + lane head + driver/history encoders, `forward/step/rollout`; also owns `PhysicsConfig`, `CONTROLLER_VOCAB`, `controller_index`, `HISTORY_*` | A | KR -> split: `physics/{idm,mobil,neighbours,projection}.py`, `model/{wms,gate,config}.py`, `core/constants.py` |
| `physics_torch.py` (600) | batched device-agnostic numpy-parity physics (a *second* `PhysicsConfig`, `step`, `rollout`) | A (entries, CTM, parity) | KR -> `physics/batched.py` (class renamed `BatchedPhysicsConfig`, alias kept) |
| `lane_head.py` (58) | `HistoryLaneHead`, `balanced_lane_loss` | A | KA -> `model/lane_head.py`; loss -> `train/losses/lane.py` |
| `driver_encoder.py` (84) | causal per-driver parameter encoder | A (opt) | KA -> `model/driver_encoder.py` |
| `entries.py` (875) | observed-counter vehicle entries (default OFF, REJECTED for R3 per change-log #14) | A (off) | KR -> `model/entries.py` (keep: needed to score MAIN2/old ckpts and `entries_v2` flag) |
| `stability.py` (574) | Run-2 long-horizon stability features | A (flag-off, imported by model) | KR -> `model/stability.py` |
| `presence.py` (67) | junction-aware slot presence | A | KR -> `data/presence.py` |
| `history.py` (86) | `history_spec`, `long_history_features` (30-60 s per-driver history) | A | KR -> `data/history.py` |
| `long_horizon.py` (91) | `soft_bound`, `residual_mag_loss`, `lane_rate_loss`, `LongHorizonSelection` | A (R3 guards) | KR -> `train/losses/{residual,lane_rate}.py`, `train/selection.py` |
| `brakefix.py` (315) | asym speed / decel / propagation braking losses, `BrakefixConfig`, hard asym cap helpers (REJECTED in training, #3) | A | KR -> `train/losses/braking.py`; cap helpers -> `model/residual_bound.py` (legacy, scoring old ckpts) |
| `traffloss.py` (296) | traffic-level losses (OFF in R3, change-log #7) | L (flag-off) | KA -> `train/losses/traffic.py` |
| `train.py` (2394) | `WmSTrainConfig` (90 fields), `TrajectoryDataset` (~850 lines), window/identity helpers, `collate`, `WmSTrainer` (~1000), `dry_run`, memory probes | A | KR -> `config/train.py`, `data/{dataset,collate,windows,sidecars}.py`, `train/{trainer,unroll,logging,receipt}.py` |
| `cli.py` (530) | `python -m wms.cli train` (88 argparse flags) | A | KR -> `cli/train.py` + `config/legacy_flags.py` (flag -> field map, kept for old argv compatibility) |
| `mixdata.py` (788) | original C1 packs + switching/direct-output packs mixed dataset | A | KR -> `data/mixdata.py` |
| `jamdata.py` (217) | jam-window oversampling (flag default off, DEFERRED) | A (off) | KA -> `data/jam.py` |
| `action_timeline.py` (189) | validated per-CAV per-step action timelines | A | KR -> `data/action_timeline.py` |
| `train_controllers.py` (135) | episode-persistent CAV callbacks (native/damping/PI) for training | A | KR -> `train/cav_drive.py` + `sim_bridge.py` |
| `train_guards.py` (39) | per-window controller exclusion | A | KR -> `data/windows.py::ControllerExclusion` |
| `val_split.py` (90) | TRAIN-scoped validation holdout | A | KR -> `data/val_split.py` |
| `lpo.py` (147) | leave-one-controller-out window lists | A | KR -> `data/windows.py` |
| `plan_steps.py` (162) | per-stage steps for a wall-clock budget | A | KR -> `train/curriculum.py` |
| `driver_sidecar.py` (479), `dense_truth.py` (493) | per-vehicle driver-parameter ground truth + recovery (WM-S2 C2) | A (opt: `--driver-sidecar`) | KR -> `data/driver_truth.py` (merge; shared helpers dedup) |
| `wall_exec.py` (276) | batched rollout-local executor of wall-group laws (mirrors `traffic_shepherd.wall_group`) | A (wall eval) | KA -> `physics/wall_exec.py` (equivalence to `wall_group` is a golden test) |
| `check400.py` (96), `gate_run2.py` (109), `gate_s2.py` (141), `gate_stages.py` (257) | four generations of smoke gates (400-step, run2, S2, multi-stage) | A (gate_stages, check400) / L | KR -> one parametric `train/gates.py`; the three older ones AR |
| `blackbox_traj.py` (227), `blackbox_id.py` (29) | black-box baselines (rebuilds HDV residual net) | L (only to score old C1 baselines; owner rule 10-09: no black-box retraining) | KA -> `legacy/blackbox/{traj,id}.py` |
| `parity.py` (200) | torch vs numpy `rollout2` parity harness | L (test support) | KA -> `tests/support/rollout2_parity.py` |
| `brakefix_audit.py` (495), `brakefix_smoke.py` (93) | one-off CPU audit + smoke of braking fix | X | AR |
| `c1.py` (224) | C1 LOCO protocol/metrics (overlaps `wms_eval/c1.py`) | L | AR after WP-E merges the unique parts into `eval/protocol.py` |
| `*.sh` (build_dense_truth, hybrid_chain, lanehead_chain, old_hist_chain, s2_chain, score_hetdata_recovery, wms_bb_chain, wms_chain), `smoke400.json` | per-experiment shell chains | X | AR (superseded by presets) |
| `tests/` (38 files + `mini_switch_pack.py`) | unit/integration tests | A | KR -> `tests/unit/<area>/` (map in 5.F); `mini_switch_pack.py` -> `tests/fixtures/mini_switch_pack.py` |

### 1.2 `output/wd01/wm_n1/wms_eval/` (30 .py) — 10.3k lines

| file (lines) | purpose | use | verdict -> target |
|---|---|---|---|
| `wms_plugin.py` (1017) | `WmsCheckpointModel`: ckpt load, state->tensor batching, CAV teacher/controller driving, rollout->frames | A | KR -> `model/checkpoint.py` (load), `eval/predictor.py` (`WmsPredictor`), `sim_bridge.py` (CAV driving) |
| `models.py` (364) | frozen model names, `PluginModel`, `FrozenA11Model` (old A11), `_keep_speed`, `teacher_force_cavs` | A (PluginModel, keep_speed) / L (A11) | KR -> `eval/models.py`; `FrozenA11Model` -> `legacy/a11_model.py` |
| `materialize.py` (685) | pack -> evaluation-state materialiser (`Pack`, `Identity`, `build_state`, selectors) | A | KR -> `eval/states.py` |
| `c1sets.py` (373), `seen.py` (203), `hetdata_shift.py` (253), `indexed_switch.py` (190) | named state sets (U-same/new-world, SEEN), t0-shift for hetero packs, switch-schedule binding | A | KR -> `eval/sets/{unseen,seen,shifted,indexed_switch}.py` |
| `c1v2.py` (893), `c1.py` (651) | per-state scoring, streaming workers, result building, verdict | A (v2) / L (v1) | KR -> `eval/score.py` + `eval/verdict.py`; `c1.py` unique parts merged then AR |
| `metrics.py` (224) | identity-matched car/section metrics | A | KR -> `eval/metrics.py` |
| `data.py` (169) | input/output normalisation contract | A | KR -> `eval/normalize.py` |
| `ranking.py` (607) | planner-readiness ranking (native vs PI) | A (planner) | KR -> `eval/ranking.py` |
| `horizon_report.py` (468) | horizon table + blow-up rate | A | KR -> `eval/report.py` |
| `driver.py` (413) | per-driver recovery scoring (WM-S2) | A (opt) | KR -> `eval/driver_recovery.py` |
| `cli.py` (221), `__main__.py`, `__init__.py` | eval CLI | A | KR -> `cli/eval.py` |
| `blackbox.py` (62) | black-box eval adapters | L | KA -> `legacy/blackbox/eval_adapter.py` |
| `ctm.py` (777), `ctm_alone.py` (482), `ctm_calib.py` (480), `ctm_eval.py` (184) | cell-transmission-model baseline (CTM-alone) | X (no active caller; only `test_wms_eval_ctm`) | AR (tests move with it) |
| `run1_fair.py`, `run1_report.py`, `report.py` | RUN1-era comparison/report | X | AR |
| `traffic_eval.py` (343), `r2set.py` (267), `prepare_states.py` (60), `shift_recovery.py` (221), `shift_traffic_split.py` (168) | RUN2/OLDHIST one-off analyses (0 inbound callers except cli) | X | AR; `traffic_eval` section metrics already live in `metrics.py` |

### 1.3 Scoring harness outside the two packages (ACTIVE; counted separately)

| path | purpose | verdict |
|---|---|---|
| `output/wd01/wm_n1/lpwo/{build_lpwo,wall_timeline,materialize_wall,score_wall,guard,bounded_cpu,closeout_check,prepare_pilot_bindings}.py` (1865) and `lpo/build_lpo.py` | window-list builders; `wall_same / wall_pilot / seen_controls` materialisation and strict wall scoring; guard | KR -> `data/window_lists.py` (builders), `eval/walls.py` (materialize_wall + score_wall + wall_timeline), `tools/data_guard.py` (guard) |
| `_out/train_r3_chain.sh` (518), `train_main_chain.sh` (448), `run2_chain.sh`, `run3_chain.sh`, `run_smoke.sh`, `run_closed_loop.*` | env-var shell chains | `train_r3_chain.sh` KR -> `configs/r3.toml` + `scripts/box/launch.sh` (30-line wrapper); rest AR |
| `_out/*.py` (42 diagnostic scripts: `entry*_cpu`, `exit_diag`, `fidelity_diag`, `merge_cal_*`, `run1_diag`, `bench_*`, `profile_step`, `diag_packv3`, `trainsetup_cpu`…) | one-off diagnostics | AR (`archive/wms_era/_out/`); `trainsetup_cpu.py` KR -> `wms preflight` |
| `_out/*.md/json/xml/log/tar/txt` (~190 tracked records) | READY/FIX/RESULT notes, receipts | AR -> `archive/wms_era/records/_out/` (receipts that ledger entries cite stay resolvable via PATH_MAP) |
| `wms_decide.py` (root, 572) + `tests/test_wms_decide.py` | TRAIN-only decision scoring adapter `predict_states(state, plan, horizon_s, seed)` | A (planner-facing contract) | KR -> `src/wms/eval/decide.py`, root file becomes 3-line shim |

### 1.4 Sibling `wm_n1` packages that WM-S still imports (frozen dependency stack)

| package | WM-S uses | verdict |
|---|---|---|
| `hdv/{config,net}.py` | `KIN_F`, `HdvConfig`, `HdvResidualNet`, `_KinBlock` (residual network of WmSModel) | **vendor** -> `model/residual_net.py` (KA content; attribute names frozen) |
| `train/idea_a.py` (4250) | `IdeaACorpus` pack reader (lazy-imported in train.py 637/748) | **vendor** -> `legacy/vendor/idea_a_corpus.py` + `data/corpus.py::PackCorpus` adapter (section 2.3) |
| `features/{geometry,counters,constants,idea_a}.py` | `is_junction_lane`, `s_down_of`, `DepartureLog`, `kinematics_view` | vendor needed functions -> `data/geometry.py`, `data/counters.py` |
| `closedloop/{live_state,a11_predictor,fast_hdv}.py` | `LiveState`, `DepartureHistory`, `live_state_from_row` (entries/eval), A11 baseline | vendor `live_state` -> `data/live_state.py`; A11 -> `legacy/a11_model.py` |
| `rollout2/merge_cal.py` | merge-calibration yield used by `_merge_cal_yield` | vendor -> `physics/merge_cal.py` (`merge_cal_params.json` stays a data file in `configs/`) |
| everything else under `output/wd01/` (`part*`, `partA_*`, `partB_*`, `inc1*`, `walls_design`, `val8`, `rescore1`, `bundle`, `corpus`, `dspeed`, `evalh`, 100+ `REGISTRATION_*/REVIEW_*/BRIEF_*` md…, ~1800 tracked files) | no WM-S import | **out of scope tonight**; recorded as AR *candidates* in section 6.3 (owner approval + reverse-reference scan needed because ledger entries cite them) |

Planner-side: `traffic_shepherd/planner/*` does **not** import `wms`; the WM-S/planner seam is
`wms_decide.predict_states` and `output/wd01/wm_n1/wm_api` (14 files, 2k lines, KA, out of scope).

### 1.5 `traffic_shepherd/` (20.2k lines) — verdict: KA everywhere, bit-identical

| group | files | note |
|---|---|---|
| ACTIVE sim core | `simulation, scenarios, config, sensing, damping, ownership, controller_factory, pi_session, fleet_switch, mixed_assignment, wall_group, seeds, metrics, reporting, escalation_observations, research_assets` | controller code; hash-pinned by frozen evidence. Do not edit. |
| ACTIVE run orchestration | `workflow, budget_guard, mechanics, verification, __main__` | KA |
| ACTIVE planner | `planner/{__init__, config, interface, rules, safety, grouping, walls, dispatch, holdout, harness_state, contract_adapter, kit_world, mock, candidate_replay, paths}` | KA |
| oversized, KA now, split later | `planner/closed_loop.py` (4304), `planner/perfect_sim.py` (2420), `planner/recorded.py` (2077), `planner/loop.py` (1772), `planner/safety.py` (798) | a later phase may split these *only* with golden transcripts (section 8); tonight untouched |

### 1.6 `tools/` and `tests/`

| path | verdict |
|---|---|
| `tools/experiment_history.py` | KA (AGENTS.md ledger requirement) |
| `tools/research_dependencies.py`, `migrate_research_dependencies.py` | KA (MIG-01 record) |
| `tools/wmn1_log_check.py` (+`tools/tests`) | KA (checks frozen IdeaATrainer logs) |
| `tools/brakefix_verify.py`, `main2_archive.py`, `main2_records.py` | AR (one-off MAIN2/brakefix) |
| **new** `tools/fetch_data.py`, `tools/data_registry.json` | copy from `reorg3/layout/tools/` unchanged (WP-A) |
| **new** `tools/data_guard.py` | move of `lpwo/guard.py` + `protocols/holdout_exclusions.json` guard; hard-refuses paths with sealed tokens (WP-C) |
| root `tests/test_cv09..cv28_*.py`, `test_pi_mechanics_*_v8.py`, `test_cv1x_review.py` | import `output.controller_verification_v8..v28`, but only `v2, v4, v7` exist in this tree. WP-F runs `pytest --collect-only`; tests that cannot import are moved with their era to `archive/circles_era/tests/` (MIG-03 precedent), the rest go to `tests/sim/`. |
| root `tests/test_wms_eval_*.py` (19), `tests/test_wms_decide.py` | KR -> `tests/unit/eval/` |
| root `tests/test_*` for sim (controllers, ownership, budgets, seeds, wall_group, mixed assignment, ledger) | KA -> `tests/sim/` (content untouched; pure `git mv`) |
| `tests/test_mixed_assignment.py.wm43_r6.bak`, `conftest_add.py`, `output_shim.py` | stray -> `archive/stray_root/` after checking `conftest.py` use |

### 1.7 Stray root files (all AR, `git mv` to `archive/stray_root/`, PATH_MAP entry each)

`_build_v032.err`, `_build_v032.log`, `_run_build_v032.py`, `_train_v032_det.err`, `_train_v032_det.log`,
`_train_v032_strict.err`, `_train_v032_strict.log`, `_train_v032_pids.txt`;
records `CHANGES_WALL_CLOSURE.md`, `EVAL_CORRECTION_RECOVERY.md`, `M1_LAUNCHFIX_RECOVERY.md`,
`WMN1_LOG_CHECK_REVIEW.md` -> `docs/records/` (they are cited, so they stay readable under `docs/`);
`output/wd01/wm_n1/*_RECOVERY.md`, `M1_LAUNCHFIX_RECOVERY.md` likewise.
Root `_out/` (tracked, 262 files) -> see 1.3. `.gitignore` additions: `/data_cache/`, `*.pt` (already), `_out/`.

### 1.8 Duplicates and dead code found (concrete)

* Two classes named `PhysicsConfig` (`model.py:40` WM-S core; `physics_torch.py:189` batched engine).
* `c1` protocol exists three times (`wms/c1.py`, `wms_eval/c1.py`, `wms_eval/c1v2.py`).
* Four smoke-gate generations (`check400`, `gate_run2`, `gate_s2`, `gate_stages`).
* `state_noise_std_m/_mps` declared twice in `WmSTrainConfig`.
* Hard residual cap code (`residual_cap_mps2`, `residual_cap_neg_mps2`, `bounded_value_asym`) is REJECTED for training
  (change-log #3) but must stay loadable/scorable: flag stays, default `None`, config validator emits a warning, the
  trainer refuses it unless `allow_rejected_hard_cap = true`.
* Eight experiment shell chains + 40 `_out` diagnostics have no inbound reference from the active path.
* `tests/` contains >30 tests for modules that are absent from the tree (cv09-cv28, see 1.6).

---------------------------------------------------------------------------------------------

## 2. Target repository layout

```
repo/
  AGENTS.md  README.md  pyproject.toml  PATH_MAP.json                 (PATH_MAP: root-level, additive; see 6.2)
  src/wms/                       THE package (editable install: pip install -e .)
    __init__.py
    core/        constants.py  errors.py  io.py  seeds.py  logging.py  tensors.py
    config/      schema.py  train.py  eval.py  data.py  load.py  legacy_flags.py  presets.py
    physics/     idm.py  mobil.py  neighbours.py  projection.py  merge_cal.py  batched.py  wall_exec.py
    model/       wms.py  gate.py  residual_net.py  lane_head.py  driver_encoder.py  entries.py
                 stability.py  residual_bound.py  checkpoint.py  factory.py
    data/        registry.py  corpus.py  windows.py  window_lists.py  val_split.py  sidecars.py
                 driver_truth.py  history.py  presence.py  action_timeline.py  mixdata.py  jam.py
                 dataset.py  collate.py  geometry.py  counters.py  live_state.py
    train/       trainer.py  unroll.py  curriculum.py  selection.py  cav_drive.py  preflight.py
                 gates.py  receipt.py  logging.py
                 losses/  base.py  trajectory.py  braking.py  residual.py  lane.py  lane_rate.py  traffic.py  compose.py
    eval/        states.py  sets/{unseen,seen,shifted,indexed_switch}.py  predictor.py  models.py  normalize.py
                 metrics.py  score.py  verdict.py  fingerprints.py  compare.py  ranking.py  report.py
                 walls.py  driver_recovery.py  decide.py
    cli/         __main__.py  train.py  eval.py  data.py  gate.py  checkpoint.py  golden.py
    sim_bridge.py              (lazy imports of traffic_shepherd + frozen PI controllers)
    legacy/      blackbox/{traj,id,eval_adapter}.py  a11_model.py  vendor/{idea_a_corpus.py,...}
  traffic_shepherd/              SIM LAYER (frozen, unchanged path)       "sim/"
  baselines/ scenarios/ assets/ output/controller_verification_v{2,4,7}/  (unchanged)
  configs/       r3.toml  c1fast.toml  smoke.toml  eval_unseen.toml  data_registry.toml  merge_params.json
  scripts/box/   launch.sh (30-line wrapper: env -> `wms train`)  README.md
  tools/         fetch_data.py  data_registry.json  data_guard.py  experiment_history.py  ...
  tests/         unit/{core,config,physics,model,data,train,eval,cli}/   golden/{ref,test_*.py}
                 sim/  fixtures/  support/  conftest.py
  docs/          ARCHITECTURE_V4.md  WM_CHANGE_LOG.md  records/  INDEX.md ...
  configs/ runs/   runs/<id>/ = small receipts only (registration, command, receipt, summary)
  archive/       wms_era/  stray_root/  circles_era/  pre_circles/
  data_cache/    (gitignored; fetched only by tools/fetch_data.py)
```

`pyproject.toml` gains `[tool.setuptools.packages.find] where=["src"]`; `traffic_shepherd` stays importable from
repo root exactly as today. `tests/conftest.py` keeps the root on `sys.path` so old import paths work.
`ruff` target is raised to py311 (tomllib); the `output/` exclude stays.

### 2.1 Dependency rule (acyclic; enforced by `tests/unit/test_import_graph.py`)

```
core  <-  config  <-  physics  <-  model  <-  train  <-  cli
  ^          ^           ^          ^  ^        ^
  |          |           |          |  |        |
  +--------  data  ------+----------+  +--- eval +
                legacy (may import everything below it; nothing imports legacy except cli/eval model factory)
sim_bridge: imports traffic_shepherd lazily; imported by train.cav_drive and eval.predictor only.
```
Allowed edges: `core` imports nothing from wms; `config` -> core; `physics` -> core; `model` -> core, config, physics;
`data` -> core, config (data may NOT import torch model code); `train` -> core, config, physics, model, data, sim_bridge;
`eval` -> core, config, physics, model, data, sim_bridge (eval must NOT import train);
`cli` -> everything; `legacy` -> anything lower; no lower layer imports `cli` or `legacy`. `traffic_shepherd` never imports `wms`.

### 2.2 `core` (WP-A)

```python
# core/constants.py
KIN_F: int = 7                      # s, lane one-hot(3), v, a, length  (from hdv.config)
HISTORY_H: int = 10; HISTORY_F: int = 6
CONTROLLER_VOCAB: tuple[str, ...] = ("none","native","pi","damping","wall","other","direct"); N_CONTROLLERS: int
CHECKPOINT_FORMAT: str = "wm_s/1"
def controller_index(policy: object) -> int
# core/errors.py
class WmsError(Exception); class TrainDataRefusal(WmsError, PermissionError)
class WmsCheckpointError(WmsError, RuntimeError); class MaterializeError(WmsError, ValueError); class C1v2Error(WmsError, ValueError)
# core/io.py
def sha256_file(path) -> str;  def read_json(path) -> Any;  def write_json_atomic(path, obj, *, sort_keys=True) -> None
def run_key(meta: Mapping) -> str            # exact semantics of train.run_key
def iter_jsonl(path) -> Iterator[dict]
# core/seeds.py   def seed_everything(seed:int, *, deterministic:bool=False) -> None
```
All class names above keep their historical import aliases in the shim modules (6.2).

### 2.3 `data` (WP-C)

```python
# data/registry.py   -- the ONLY place that maps logical dataset names to local paths
@dataclass(frozen=True) class DataItem: name: str; path: Path; sha_manifest: str|None
def resolve(name: str, *, cache_dir: Path|None=None, env_override: str|None=None) -> Path
   # order: WMS_DATA_<NAME> env var -> configs/data_registry.toml -> tools/data_registry.json local_path (data_cache/…)
   # raises TrainDataRefusal if the resolved path contains a sealed/CAL/HOLDOUT/chunk-5/J4B/INC1 token
def require_present(name) -> Path   # fail-closed; message tells `python tools/fetch_data.py get <name>`

# data/corpus.py     -- adapter around the vendored IdeaACorpus (unchanged semantics)
class PackCorpus:
    @classmethod def open(cls, pack_dir: Path, *, name: str|None=None) -> "PackCorpus"
    n_runs: int; k_slots: int; def run_meta(i:int)->dict; def item(run:int, tick:int)->dict
    def resolve_pack_name(self, names: Sequence[str]) -> list[str]     # the pack_v3 name-vs-basename fix (change-log #12), fail-closed

# data/windows.py    -- window lists + selection
def load_window_list(path) -> set[tuple[str,str]]
def load_exclusion_keys(paths: Sequence[Path]) -> set[str]
class ControllerExclusion:         # was train_guards
    def __init__(self, excluded_controllers: Collection[str]); def refuses(self, window_meta: Mapping) -> bool
def select_lpo_positions(corpus: PackCorpus, names: list[str], *, window_list, exclude_keys, controllers_excluded, ...) -> list[int]
def select_windows(cfg: DataConfig, corpus_set: Mapping[str, PackCorpus]) -> WindowSelection
@dataclass(frozen=True) class WindowSelection: train: list[WindowRef]; val: list[WindowRef]; counts: dict[str, tuple[int,int]]  # {"pack_v3": (294, 47), ...}

# data/dataset.py / collate.py
class TrajectoryDataset(torch.utils.data.Dataset):   # signature = current __init__ kwargs, but taking DataConfig
    def __init__(self, cfg: DataConfig, selection: WindowSelection, corpora: Mapping[str, PackCorpus], *, unroll_steps: int, seed: int)
    def set_unroll_steps(self, n:int) -> None; def __getitem__(self, i:int) -> dict
    receipt: dict
def collate(records: list[dict]) -> dict[str, Tensor|list]        # byte-identical keys/dtypes to train.collate

# data/mixdata.py:  class MixedDataset(...)   data/jam.py: class JamOversampler
# data/history.py:  history_spec(history_s, history_stride_s, dt_s) ; long_history_features(kin, mask, target, ...)
# data/sidecars.py: read_identity_rows(sidecar, *, corpus_k) ; overlays/outcomes readers ; (outcome labels offline only)
# data/val_split.py: world_disjoint_val(...)   data/action_timeline.py: ActionTimeline, CONTROLLERS
# data/window_lists.py: build_lpo / build_lpwo (the lists that give pack_v3 294/47)
```

### 2.4 `physics` + `model` (WP-B)

```python
# physics/idm.py
@dataclass(frozen=True) class IDMParams: v0:float; T:float; s0:float; a:float; b:float; delta:int ...   # == model.PhysicsConfig.idm()
def idm_accel(v: Tensor, leader_v: Tensor, gap: Tensor, v0: Tensor, p: IDMParams) -> Tensor
# physics/neighbours.py
def lead_neighbour(kin: Tensor, present: Tensor) -> tuple[Tensor, Tensor, Tensor]
def lane_neighbour(kin, present, query_lane, *, side:int) -> NeighbourSet
# physics/mobil.py
def mobil_merge_logits(kin, present, cfg: PhysicsConfig, ...) -> Tensor
# physics/projection.py   def project_gaps(...)   physics/merge_cal.py   def merge_cal_yield(...)
# physics/batched.py       BatchedPhysicsConfig, step(...), rollout(...)   (== physics_torch)
# physics/wall_exec.py     WallExecutor (== wall_exec.py)
@dataclass class PhysicsConfig: dt_s:float; ...; @classmethod from_params_file(path, **overrides)

# model/wms.py
class WmSModel(nn.Module):
    def __init__(self, cfg: ModelConfig, *, physics: PhysicsConfig|None=None)
    def forward(self, state: dict[str,Tensor], *, k_steps: Tensor|int=0, history=None, history_mask=None,
                driver_history=None, driver_history_mask=None, driver_defaults=None, driver_params=None,
                driver_embedding=None, lane_age=None, baseline_lane_logits=None, controller_id=None,
                action_command=None) -> dict[str,Tensor]            # keys unchanged
    def step(...) -> dict[str,Tensor]                               # signature unchanged
    def rollout(self, state, *, n_steps:int, k0:int=0, cav_teacher=None, present_seq=None, junction_seq=None,
                is_cav_seq=None, history=None, ..., cav_controller=None) -> dict[str,Tensor]
    def gate(self, k_steps) -> Tensor;  def physics_accel(self, state, ...) -> Tensor;  def hold_teacher(self, state) -> Tensor
# model/checkpoint.py
@dataclass class LoadedWms: model: WmSModel; train_cfg: TrainConfig; payload_meta: dict; sha256: str
def load_checkpoint(path: str|Path, *, device: str="cpu", strict: bool=True) -> LoadedWms
   # weights_only=True first (as today); accepts legacy flat "config" and nested "config_v2"
def save_checkpoint(path, model, cfg: TrainConfig, *, history=None, val_by_stage=None, extra: Mapping|None=None) -> None  # writes format "wm_s/1"
# model/factory.py   def build_model(cfg: ModelConfig, physics: PhysicsConfig|None=None) -> WmSModel
```
`WmSModel` is decomposed into small private submodules **without renaming any registered attribute** (4.2).

### 2.5 `train` (WP-D)

```python
# train/losses/base.py
class LossTerm(Protocol):
    name: str
    def __call__(self, ctx: LossContext) -> TermValue   # TermValue(value: Tensor, valid_count: Tensor|None)
@dataclass class LossContext: pred: Mapping[str,Tensor]; target: Mapping[str,Tensor]; batch: Mapping; stage_s: float; step_idx: Tensor; ...
@dataclass class LossBreakdown: total: Tensor; terms: dict[str, Tensor]; shares: dict[str,float]
# train/losses/compose.py
def build_loss(cfg: LossConfig) -> "LossComposer"
class LossComposer:  def __call__(self, ctx: LossContext) -> LossBreakdown     # term names == current train_log "terms" keys
# per-term modules: trajectory (position/speed huber, step weights, lane BCE), braking (asym, decel, propagation),
#   residual (soft_bound, residual_mag), lane_rate, lane (balanced_lane_loss), traffic (legacy)
# train/trainer.py
class WmSTrainer:
    def __init__(self, cfg: TrainConfig, model: WmSModel, dataset, *, val_dataset=None, out_dir: Path)
    def fit(self) -> TrainResult               # was run(); stages = curriculum
    def evaluate(self, val_dataset, *, unroll_steps:int, require_full_horizon:bool) -> dict
# train/curriculum.py   @dataclass StagePlan(stage_s, unroll_steps, steps); def plan_steps(cfg, budget_s)->list[StagePlan]
# train/selection.py    class LongHorizonSelection   (identical policy: longest eligible horizon, lowest mean position)
# train/preflight.py    def preflight(cfg) -> PreflightReport   (replaces the chain's HELP guard / CRLF / path checks; required_log_terms)
# train/gates.py        def run_gate(name: Literal["400","stages","s2"], cfg, out) -> GateResult   # must run on the REAL data path
# train/receipt.py      def write_train_receipt(...)   # schema "wm_s_train_receipt/1" unchanged
```

### 2.6 `eval` (WP-E) and data layout integration

```python
# eval/states.py
class Pack: ...;  class Identity: ...
def build_state(window: Mapping, packs: Mapping[str,Pack], identities, *, horizon_s:float, ...) -> dict   # == materialize.build_state
def materialize_set(spec: EvalSetConfig, out_jsonl: Path) -> MaterializeSummary
# eval/predictor.py
class WmsPredictor:
    @classmethod def from_checkpoint(cls, path, flags: RolloutFlags, *, device="cpu") -> "WmsPredictor"
    def predict_states(self, state: dict, plan_or_controllers: Mapping, horizon_s: float, seed: int) -> list[dict]  # frames
# eval/models.py     load_model(spec, plugin=None, checkpoint=None, flags=...) -> PluginModel ; keep_speed baseline ; teacher_force_cavs
# eval/score.py      evaluate_state(record, predictor, arms, horizons) ; run_model(...) ; build_result(models, sets, ...)
# eval/metrics.py    car_metrics, traffic_metrics, paired_difference, cumulative_delay, throughput   (names unchanged)
# eval/fingerprints.py  def fingerprint(per_state_results|rollout_logs) -> dict   # NEW: logged residual size, lane-change rate per HDV-step,
#                                                       blow-up rate; the R3 "separable by fingerprint" requirement (change log, Round 3)
# eval/compare.py       def compare(new: ModelResult, previous_best: ModelResult, sets: Sequence[str], horizons=(60,120)) -> Comparison
#                       # owner rule 10-09: only vs previous best (C1FAST), on wall_same / wall_pilot / seen_controls, 60/120 s + blow-up rate
# eval/walls.py         materialize_wall / score_wall / wall_timeline (from lpwo)    eval/ranking.py  rank(...)    eval/decide.py  predict_states
```
Evaluation never imports `train`. Eval reads the checkpoint's stored config through `model.checkpoint.load_checkpoint`.

### 2.7 `config` (WP-A)

See section 3.

### 2.8 `cli` (WP-A, thin; each subcommand delegates)

```
python -m wms train       --config configs/r3.toml [--set train.lr=5e-4] [--mode preflight|stage|run] [--out DIR]
python -m wms eval        --config configs/eval_unseen.toml --checkpoint X [--previous-best Y]
python -m wms materialize --config configs/eval_unseen.toml
python -m wms windows     build|count   (window lists, prints 294/47 style counts)
python -m wms gate        400|stages|s2 --config ...
python -m wms checkpoint  info|convert-config PATH         (prints format, sha, flat+nested config)
python -m wms data        list|status|get NAME             (thin wrapper over tools/fetch_data.py)
python -m wms golden      capture|check                    (section 4)
```
Old entry points `python -m wms.cli train ...` and `python -m wms_eval ...` keep working through `config/legacy_flags.py`
(argparse flag -> dataclass field table, one test per flag proving identical resolved config).

### 2.9 Data layout integration (from CLEAN_END_STATE.md)

* Local data lives only in gitignored `data_cache/`, fetched on demand by `tools/fetch_data.py {list,status,verify,get}`
  (20 GiB budget, no delete verb). `wms.data.registry` is a read-only consumer of that registry; it never calls rclone.
* Logical names to register by the data owner (the *registry file is theirs*; WP-C only reads it): `pack_v3`, `pack_dspeed`,
  `pack_damping`, `pack_switching_train`, `sidecars_packv3_v2` (identities, outcomes), `identities_dspeed`, `identities_damping`,
  `overlay_dspeed`, `overlay_damping`, `lpwo_v3` (window lists + `excluded_runs_PI_wall_v3.json`), `merge_params`,
  `ckpt_c1fast` (`models/WM-S-C1FAST-20261006/full_struct/wm_s.pt`), `ckpt_r3_*`, `eval_states_wall_same`, `eval_states_wall_pilot`,
  `eval_states_seen_controls`.
* Presets reference logical names only (`pack = "pack_v3"`), never absolute paths. A machine-specific override is
  `WMS_DATA_PACK_V3=/root/autodl-tmp/...` (box).
* Restricted material (sealed, CAL, HOLDOUT, chunk-5, J4B, INC1) is never registered and `registry.resolve` and
  `tools/data_guard.py` refuse any path containing those tokens. Tests may not open such files (CI grep test
  `tests/unit/data/test_no_sealed_tokens.py` scans the *source tree* for such path literals).
* Frozen evidence paths: `PATH_MAP.json` at root (6.2) maps old absolute/relative paths -> new path or `data_cache` key;
  no frozen manifest is rewritten.

---------------------------------------------------------------------------------------------

## 3. Configuration design

### 3.1 Typed schema (`config/schema.py`, `config/train.py`; `@dataclass(frozen=True, slots=True)`)

```python
@dataclass(frozen=True) class DataConfig:
    packs: Mapping[str,str]             # role -> logical name  {"pack_v3": "pack_v3", "dspeed": "pack_dspeed", "damping": "pack_damping"}
    sidecars: SidecarConfig             # identities, outcomes, overlays, driver_sidecar (logical names)
    windows: WindowConfig               # train/val/exclude lists (logical), leave_pi_wall_out: bool
    switching: SwitchingConfig          # pack, identities, driver_params, windows_train/val, timeline_index, mix_ratio=0.3, event_oversample=1
    history_s: float|None; history_stride_s: float = 1.0
    start_shift_prob: float = 0.75; state_noise_std_m: float = 0.0; state_noise_std_mps: float = 0.0
    jam: JamConfig                      # oversample=1 (off), speed_mps=5, cache_dir
    entries_v2: bool = False; entry_pad: int = 8
    cav_drive: Literal["teacher","native","damping","pi"] = "teacher"
@dataclass(frozen=True) class ModelConfig:
    kind: Literal["structured","blackbox_traj","blackbox_id"] = "structured"
    rollout_mode: Literal["residual","hybrid"] = "residual"; lane_head: Literal["off","history"] = "off"
    physics_params: str|None; merge_cal: bool = True; projection: ProjectionConfig
    residual_cap_mps2: float|None = None; residual_cap_neg_mps2: float|None = None   # REJECTED (#3): validator warns, trainer refuses
    residual_soft_bound_mps2: float|None; residual_soft_bound_neg_mps2: float|None; lane_logit_cap: float|None
    n_controllers: int; controller_embed_dim: int = 16; history_features: int = 6; checkpoint_residual: bool = False
@dataclass(frozen=True) class LossConfig:
    position_huber_delta_m: float = 5.0; speed_huber_delta_mps: float = 1.0
    position_weight: float = 1.0; speed_weight: float = 1.0; lane_weight: float = 1.0; step_weight_decay: float = 0.01
    braking: BrakingConfig              # asym_speed_alpha=1.0, decel_weight=1.0, decel_threshold_mps2=3.0, propagation_weight=0.0, prop_window_s=3.0, prop_max_gap_m
    residual: ResidualLossConfig        # residual_mag_weight=0.0
    lane_rate: LaneRateConfig           # weight=0.0, target=0.0123
    traffic: TrafficLossConfig          # enabled=False (legacy)
    stability_weight: float = 0.0; stability_margin: float = 1.0; stability_probes: int = 1
    conservation_enabled: bool = False; conservation_weight: float = 0.0; loss_share_logging: bool = False
@dataclass(frozen=True) class CurriculumConfig:
    unroll_s: tuple[float,...] = (10,30,60,120); stage_steps: tuple[int,...]|None = (2500,2500,1875,1250)
    truncation_steps: int = 20; max_hours: float|None
@dataclass(frozen=True) class OptimConfig:   batch_size=8; lr=1e-3; weight_decay=0.0; grad_clip=1.0
@dataclass(frozen=True) class SelectionConfig: select_by_long_horizon: bool=False; val_every: int=200; val_batches: int=8
@dataclass(frozen=True) class RuntimeConfig: device="cpu"; threads=6; num_workers=0; seed=101; omp: int=2; out_dir
@dataclass(frozen=True) class TrainConfig: schema_version:int=2; name:str; data; model; loss; curriculum; optim; selection; runtime
    def to_legacy_flat(self) -> dict          # == WmSTrainConfig.to_dict() of the current code, key for key
    @classmethod def from_legacy_flat(cls, d: Mapping) -> "TrainConfig"   # reads old ckpt["config"] / TRAIN_RECEIPT["config"]
    def fingerprint(self) -> str             # sha256 of canonical nested JSON
```
Validation lives in `__post_init__`/`validate()` and mirrors every fail-closed check in the chain and `cli.py`:
stage/steps length agreement, `lane_head=history` requires `rollout_mode=hybrid`, `NUM_WORKERS==0`,
OMP positive int, braking flags consistent with `required_log_terms`, `select_by_long_horizon` requires val list, no sealed token in any resolved path.

### 3.2 Presets and overrides (`config/load.py`)

```toml
# configs/r3.toml
schema_version = 2
name = "WM-S-R3"
extends = "c1fast.toml"             # single inheritance, deep-merge, later keys win

[data]
history_s = 60.0
history_stride_s = 1.0
[data.packs]            pack_v3 = "pack_v3"; dspeed = "pack_dspeed"; damping = "pack_damping"
[data.windows]          train = "lpwo_v3/windows_train_noPI_noWall"; val = "lpwo_v3/windows_val_noPI_noWall"; exclude = "lpwo_v3/excluded_runs_PI_wall_v3"
[data.switching]        pack = "pack_switching_train"; mix_ratio = 0.3; event_oversample = 1
[model]                 rollout_mode = "hybrid"; lane_head = "history"; residual_soft_bound_mps2 = 1.5; merge_cal = true
[loss.braking]          asym_speed_alpha = 2.0; decel_weight = 5.0; decel_threshold_mps2 = 3.0; propagation_weight = 1.0; prop_window_s = 3.0
[loss.residual]         residual_mag_weight = 0.05
[loss.lane_rate]        weight = 0.1; target = 0.0123
[selection]             select_by_long_horizon = true
```
(Numeric values in the example above are placeholders; WP-D copies the exact defaults from the R3 chain, and the golden
test G6 proves `configs/r3.toml` resolves to the argv the chain produces.) `--set data.history_s=30` overrides;
every resolved config is dumped to `<out>/config.resolved.json` and its fingerprint is written into `TRAIN_RECEIPT.json`.
`configs/c1fast.toml` = the C1FAST receipt (`start_shift_prob 0.75`, stages 2500/2500/1875/1250, unroll 10/30/60/120,
no cap, no projection, no traffic/stability/conservation, merge_cal on) — golden-checked against
`runs/active/WM-S-C1FAST-20261006/box/full_struct/TRAIN_RECEIPT.json`.
The shell chain's env knobs map 1:1 to config keys (`R3_PRESET=R3_CTRL` -> `configs/r3_ctrl.toml` that sets braking off;
`R3_*` pass-throughs -> `--set`); `scripts/box/launch.sh` only exports box paths (`WMS_DATA_*`, `PY`, `OMP`) and calls
`python -m wms train --config "$CONFIG" --mode "$MODE"`. The CRLF/`HELP guard`/mode checks move into `train/preflight.py`.

### 3.3 Eval config

`EvalConfig`: `sets: list[str]` (wall_same, wall_pilot, seen_controls), `horizons_s=(10,30,60,120)`, `models: {name: checkpoint-logical-name}`,
`previous_best = "ckpt_c1fast"`, `flags: RolloutFlags` (entries off, projection per spec, merge_cal on, `driver_params` rollout mode **forbidden**: change-log #15),
`workers`, `out_dir`. Preset `configs/eval_unseen.toml`.

### 3.4 Guard rails derived from WM_CHANGE_LOG.md (encoded as validators/tests, not prose)

| change-log item | encoded as |
|---|---|
| #3 hard residual clamp REJECTED | `TrainConfig.validate()` rejects `residual_cap_*` unless `allow_rejected_hard_cap` (used only to re-score MAIN2) |
| #5/#7 bundles, traffic/stability/conservation off | defaults off; `r3.toml` has none; test asserts resolved R3 has them off |
| #14 entries REJECTED for R3 | `entries_v2=false` in r3; eval flag `entries` default off |
| #15 `driver_params` scoring REJECTED | eval validator refuses `rollout_mode = "driver_params"` |
| #16 self-recovery/pushforward VETOED | no such option exists in the schema (do not add) |
| #18 retrain black boxes REJECTED | `kind != "structured"` training allowed only with `--legacy`; eval-only adapters in `legacy/blackbox` |
| Owner rules 10-09 | `eval/compare.py` accepts exactly one `previous_best`; ranking on unseen controllers at 60/120 s + blow-up rate |
| Round 3: one GPU, one model per round, flags default OFF | schema defaults = OFF; no paired-control orchestration code |

---------------------------------------------------------------------------------------------

## 4. Compatibility and golden tests

### 4.1 Requirements

R1. Load `C1FAST` `wm_s.pt` (`runs/active/WM-S-C1FAST-20261006/box/full_struct/wm_s.pt`), its stage checkpoints, MAIN2
checkpoints, and round-3 checkpoints with `load_checkpoint(path, strict=True)` — no key renames, no missing/unexpected keys.
R2. Forward/step/rollout outputs identical (bit-exact on CPU float32 with `torch.use_deterministic_algorithms(True)`, threads=1;
fallback tolerance documented per test below) to the current code on fixed inputs.
R3. Per-term training losses equal on a fixed batch; the logged `terms` keys equal current `train_log.jsonl` keys.
R4. Window selection counts equal (pack_v3 train/val = **294 / 47** under `lpwo_v3`; all other packs' counts as captured).
R5. A fixed-seed 3-step CPU training on the fixture pack gives the same loss sequence and final parameter hash.
R6. Eval: `car_metrics`, `traffic_metrics`, `build_result` JSON identical for fixed frames and for the golden eval states.
R7. `TrainConfig.to_legacy_flat(from_legacy_flat(d)) == d` for the C1FAST and MAIN2 receipts; presets reproduce the chain argv.
R8. `traffic_shepherd` unchanged: `git diff --stat bf7d4cb -- traffic_shepherd output/controller_verification_v2 output/controller_verification_v4 output/controller_verification_v7 baselines scenarios` is empty at every merge; sim tests (`tests/sim`) pass; wall-exec equivalence test (G10) passes.

### 4.2 Frozen state_dict surface

Module attribute names that determine state_dict keys must not change in `WmSModel` and its parts (`WP-0` records the full key/shape
list in `tests/golden/ref/state_dict_keys_c1fast.json` and `..._r3_init.json`; `tests/golden/test_state_dict_keys.py` fails on any diff).
Decomposition into private helpers is allowed only if helpers hold no parameters/buffers or are registered under the same names.

### 4.3 Harness: capture from CURRENT code first

`tools/golden/capture_reference.py` (owner WP-0). **Run from the untouched baseline commit** (`bf7d4cb`; the script refuses to run unless
`git rev-parse HEAD` is a descendant-free baseline recorded in `tests/golden/ref/BASELINE.json`, or `--allow-dirty` is passed by WP-0 itself).
It imports the *old* modules (`output/wd01/wm_n1/wms`, `wms_eval`) and writes small deterministic artefacts:

| id | captured artefact (`tests/golden/ref/…`) | built from | compare in new code |
|---|---|---|---|
| G1 | `model_forward_{init0,c1fast}.pt` — outputs of `forward`, `step`, `rollout(n=20,120)` | fixed synthetic state batch (`tests/golden/inputs.py`, seeded, 2x64 slots, with CAVs, a junction, history) with a random-init model (seed 0, R3 model config) and with the C1FAST checkpoint | `test_g1_model_outputs.py`: `torch.equal`, fallback `rtol=1e-6, atol=1e-7` forward, `atol=1e-4 m / 1e-5 m/s` for 120-step rollout (any fallback use must be stated in the PR) |
| G2 | `loss_terms.json` — value of every term {pos_huber, speed_huber, lane, asym_speed, decel, propagation, residual_mag, lane_rate, soft_bound penalty, lane BCE, traffic, stability, conservation, total} for a fixed batch+rollout | old `WmSTrainer` loss path run for one fixed step, each term toggled on in isolation and all-on (R3) | `test_g2_loss_terms.py`: `rtol=1e-6` per term, exact key set |
| G3 | `rollout_eval_states.json` — frames for 3 fixed eval states (one `wall_same`, one `wall_pilot`, one `seen_controls`) at 10/60 s using `WmsCheckpointModel` with C1FAST | real materialised states if present in `data_cache` (fetched read-only, non-restricted), else the 3 synthetic states | `test_g3_predictor.py`: per-vehicle s/v/lane equal (`atol 1e-4`) |
| G4 | `window_counts.json` — `{pack: [n_train, n_val]}` e.g. pack_v3 `[294, 47]`, plus first/last 5 window keys and a sha of the sorted key list | old `TrajectoryDataset`/`_select_lpo_positions` over `lpwo_v3` lists (skipped with `pytest.skip("data_cache missing")` if packs not fetched; the fixture-pack variant `mini_switch_pack` always runs) | `test_g4_windows.py`: exact equality |
| G5 | `dataset_items.json` — sha256 of every tensor of `ds[i]` for 6 fixed indices + `collate` of them, and mixdata ratios | fixture pack (+ real pack if available) | `test_g5_dataset.py`: exact hashes |
| G6 | `config_roundtrip.json` — `TRAIN_RECEIPT.json["config"]` of C1FAST/MAIN2 (copied, small) and the argv produced by `train_r3_chain.sh preflight` under the documented env | old `WmSTrainConfig`, the shell chain | `test_g6_config.py`: `to_legacy_flat` equality; `configs/r3.toml` -> same argv/flat config |
| G7 | `train_smoke.json` — 3 optimizer steps x stages [10,30], loss sequence, final param sha256 | fixed seed, CPU, `threads=1`, fixture pack | `test_g7_train_smoke.py`: loss `rtol 1e-6`, param hash exact (fallback `atol 1e-6` per tensor, flagged) |
| G8 | `metrics.json` — outputs of `car_metrics`, `traffic_metrics`, `paired_difference`, `build_result`, `horizon_report` for fixed frames | `wms_eval.metrics`, `c1v2`, `horizon_report` | `test_g8_eval_metrics.py`: exact JSON equality |
| G9 | `selection.json` — `LongHorizonSelection` decisions over a scripted val sequence; `plan_steps` outputs for several budgets | old modules | exact |
| G10 | `physics_units.pt` — IDM, MOBIL logits, projection, merge-cal, batched `physics_torch.step/rollout`; plus `wall_exec` vs `traffic_shepherd.wall_group` laws on a scripted scenario | old modules | `torch.equal`; wall equivalence exact |
| G11 | `state_dict_keys_*.json` — sorted `(key, shape, dtype)` for random-init R3 config, C1FAST checkpoint | old model | exact |

`MANIFEST.json` records for each artefact: baseline commit, torch/numpy versions, device, seed, sha256 of every input file
(checkpoints, lists), and generation time. Large inputs (packs, checkpoints) are *not* committed; tests resolve them through
`wms.data.registry` and `pytest.skip` with an actionable message (`python tools/fetch_data.py get ...`) when absent, **but** the
synthetic-fixture variant of every golden test always runs in CI so the suite is never vacuous. Reference files are budgeted
to < 5 MB total.
`python -m wms golden capture` is a thin wrapper that is only valid on the baseline commit; `python -m wms golden check`
runs `pytest tests/golden -q`.

### 4.4 Rules for work packages

1. A package's PR must list which golden ids it touches and show them green; no golden file may be edited in a feature PR
   (only WP-0 or the integrator, with a justification in the commit message).
2. If a golden test cannot pass bit-exactly after a legitimate numerical re-ordering, the PR records the max abs diff and the
   fallback tolerance in `docs/records/GOLDEN_TOLERANCES.md`; the integrator approves.
3. The old code stays importable until integration (shims) so that `git stash`-free A/B comparisons remain possible.

---------------------------------------------------------------------------------------------

## 5. Implementation work packages

Branches: one per WP, `wms/clean-<id>`, PRs into `wms/clean`. Disjoint file ownership (a WP may *read* anything; it writes only its own paths).
Shared files (`pyproject.toml`, `tests/conftest.py`, `PATH_MAP.json`, `src/wms/__init__.py`, `docs/*`) are owned by WP-A / the integrator;
other WPs request edits via the integrator. Start: WP-0 first (hours, one agent); then A-F in parallel (they depend only on the
interfaces in section 2, stubbed by WP-A in the very first commit); then WP-I.

### WP-0 Golden capture (prerequisite; ~2 h)
* Owns: `tools/golden/**`, `tests/golden/**` (inputs, ref/, test_*.py skeletons that import from `wms.*` new paths and are marked `xfail(strict=False)` until the owning WP lands).
* Inputs: baseline commit `bf7d4cb`, C1FAST checkpoint, `lpwo_v3` lists, `mini_switch_pack`, fetched packs (read-only).
* Outputs: G1-G11 references + `MANIFEST.json` + `BASELINE.json`; the baseline commit is recorded in `BASELINE.json` (a lightweight tag `pre-clean-baseline` may be created; never moved or deleted).
* Done when: capture is reproducible twice (identical hashes) and documented in `tests/golden/README.md`.

### WP-A core + config + cli + registry skeleton
* Owns: `pyproject.toml`, `src/wms/{__init__,core,config,cli,sim_bridge.py}`, `configs/**`, `tools/fetch_data.py`, `tools/data_registry.json`, `scripts/box/**`, `tests/conftest.py`, `tests/unit/{core,config,cli}/`, `src/wms/data/registry.py` (only this data file), `tests/unit/test_import_graph.py`.
* First commit (within 30 min, blocks nobody): empty-but-importable skeleton of *every* sub-package with the signatures from section 2 as `Protocol`s / `NotImplementedError` stubs so other WPs import against stable names.
* Inputs: `WmSTrainConfig` (train.py:189), `cli.py` flags, `train_r3_chain.sh`, reorg3 `fetch_data.py`, `CLEAN_END_STATE.md`.
* Outputs: typed configs, `from_legacy_flat/to_legacy_flat`, `legacy_flags.py` (88 flags), TOML loader with `extends`/`--set`, presets `c1fast.toml`, `r3.toml`, `r3_ctrl.toml`, `smoke.toml`, `eval_unseen.toml`, CLI dispatch, `sim_bridge` (copies of the lazy controller setup from `train_controllers.py`/`wms_plugin.py` without behaviour change; exposes `make_cav_controller(kind, ...)`).
* Must pass: G6, G11 (config side), import-graph test, one test per legacy flag.

### WP-B physics + model + checkpoint
* Owns: `src/wms/{physics,model}/**`, `tests/unit/{physics,model}/`.
* Inputs: `model.py`, `physics_torch.py`, `lane_head.py`, `driver_encoder.py`, `entries.py`, `stability.py`, `wall_exec.py`, vendored `hdv/{config,net}.py`, `rollout2/merge_cal.py`, `brakefix.py` cap helpers, `wms_plugin.py` checkpoint-loading part.
* Outputs: the interfaces in 2.4; `WmSModel` decomposed (idm/mobil/neighbours/projection/merge_cal as pure functions); `load_checkpoint/save_checkpoint`; `legacy/blackbox/*` (KA).
* Must pass: G1, G10, G11, R1 (loads C1FAST + one MAIN2 + one R3 checkpoint strictly), parity test vs numpy `rollout2` (moved support).

### WP-C data
* Owns: `src/wms/data/**` (except `registry.py`), `src/wms/legacy/vendor/**`, `tools/data_guard.py`, `tests/unit/data/`, `tests/fixtures/mini_switch_pack.py`.
* Inputs: `train.py` dataset/collate/window code, `mixdata.py`, `jamdata.py`, `lpo.py`, `train_guards.py`, `val_split.py`, `history.py`, `presence.py`, `action_timeline.py`, `driver_sidecar.py`, `dense_truth.py`, `lpwo/build_lpwo.py`, `lpo/build_lpo.py`, `lpwo/guard.py`, `train/idea_a.py` (vendored).
* Outputs: `PackCorpus`, `TrajectoryDataset`, `collate`, `MixedDataset`, window selection with counts, list builders, guard.
* Must pass: G4, G5, selection counts (pack_v3 294/47), no-sealed-token test, `NUM_WORKERS=0` pickling note preserved.

### WP-D train
* Owns: `src/wms/train/**`, `tests/unit/train/`.
* Inputs: `train.py` trainer part, `brakefix.py`, `long_horizon.py`, `traffloss.py`, `lane_head.balanced_lane_loss`, `plan_steps.py`, `train_controllers.py`, gate modules, receipt code.
* Outputs: `LossComposer` + term modules (exact term names), `WmSTrainer.fit`, curriculum, selection, `preflight`, one parametric `gates.py`, `receipt.py` (schema unchanged), checkpoint writes through `model.checkpoint.save_checkpoint` (flat legacy config + `config_v2`).
* Depends on: WP-A schema, WP-B model, WP-C dataset *interfaces* (stubs available from WP-A's skeleton; integrates for real in WP-I).
* Must pass: G2, G7, G9; resume test (`test_main_resume_integration`); R3 chain preflight equivalence.

### WP-E eval
* Owns: `src/wms/eval/**`, `src/wms/legacy/a11_model.py`, `src/wms/cli/eval.py` body (WP-A owns dispatch), `tests/unit/eval/`.
* Inputs: all of `wms_eval/` (verdicts in 1.2), `lpwo/{materialize_wall,score_wall,wall_timeline}.py`, `wms_decide.py`, `wms_plugin.py` predictor part.
* Outputs: interfaces in 2.6, NEW `fingerprints.py` and `compare.py` (previous-best-only comparison on wall_same / wall_pilot / seen_controls at 60/120 s + blow-up rate), `WmsPredictor`, `decide.py`.
* Must pass: G3, G8, existing 19 `test_wms_eval_*` tests ported (assertions unchanged), `test_wms_decide`.

### WP-F repository hygiene, tests relocation, archive, docs
* Owns: everything not owned above: `archive/**`, `PATH_MAP.json` (additive entries only; WP-A creates the file header), root stray files, `_out/**` moves, `output/wd01/wm_n1/wms*/` moves (executed *last*, after WP-I signals), `tests/sim/**`, `tests/unit/` relocation of old tests, `docs/INDEX.md`, `README.md`, `.gitignore`, `docs/records/**`.
* Tasks: (1) `pytest --collect-only` over `tests/` to classify dead tests (cv09-cv28 etc.); (2) move stray root files (1.7); (3) move AR modules (1.1-1.3) with `git mv` and PATH_MAP entries; (4) create shim packages `output/wd01/wm_n1/wms/__init__.py` etc. (6.2) *after* WP-I; (5) port the 38+19 old tests into the new tree (assertions unchanged, imports updated; mapping table kept in `docs/records/TEST_MOVE_MAP.md`); (6) README / INDEX / `.gitignore` (`/data_cache/`, `_out/`); (7) `ruff` clean for `src/ tests/`.
* Must pass: `tests/sim` unchanged-green; PATH_MAP resolution test (`tests/unit/test_path_map.py`: every old path in the map exists at the new path or is a shim); no file deleted (`git diff --diff-filter=D` against base is empty).

### WP-I Integration (owner: integrator/architect; after A-F merge)
1. Merge in order A, B, C, D, E, F; run `python -m ruff check src tests` and `python -m pytest -q` (all golden green with real data when available).
2. End-to-end smoke on the real data path: `python -m wms gate stages --config configs/smoke.toml` (400-step + multi-stage) — gates must use the real pack path (lesson from "gates must use real data path").
3. Reproduction check: `python -m wms train --config configs/r3.toml --mode preflight` prints a plan identical to the chain's preflight (G6); one 6-step CPU `run` with R3 flags equals the old code's 6-step run (G7 extended).
4. Re-score C1FAST through the new `wms eval` on the 20-state diagnostic subset; compare to `WALL_C1_RESULT.md` numbers (12.84 m median |ds| at 60 s on `wall_same`'s 20-state subset, 8.71 m full 67 states) — must match within metric tolerance.
5. Install shims (6.2); verify old entry points (`python -m wms.cli train --help`, `python -m wms_eval --help`) still work.
6. Update `docs/INDEX.md`, `docs/WM_CHANGE_LOG.md` (add "repo restructure" row: no behaviour change), ledger note if any record cites moved paths (`python tools/experiment_history.py validate` and `verify-sources` stay at zero errors).
7. Prepare GitHub push (private): confirm `.gitignore`, no `*.pt`/data, `git ls-files | xargs du` size report < 100 MB, branch protection plan (PRs from `wms/clean-<id>`).

Parallelism / conflict map: A creates the skeleton first and owns shared files; B, C, D, E, F touch disjoint directories; D and E only
*import* B/C interfaces. F's destructive-looking moves (archive) are scheduled last, preventing merge conflicts with B-E which *copy* code from the
old locations while the old tree is still in place.

---------------------------------------------------------------------------------------------

## 6. Archive, delete, shims

### 6.1 Nothing is deleted
`git diff --diff-filter=D base..HEAD` must be empty. Moves are `git mv` (history kept); large untracked artefacts are not touched.
Data stays on Drive via `tools/fetch_data.py`; no local deletion is part of this plan.

### 6.2 PATH_MAP and shims
* `PATH_MAP.json` (root, additive, checksummed for moved text records): `{"old": "<repo-relative or absolute path>", "new": "<path>", "kind": "moved|shim|data_cache", "sha256": "..."}`.
  It extends the existing `archive/circles_era/PATH_MAP.json` convention; both are tested for resolution.
* Shims keep every *historical import path* working for replay and tests that cite them:
  `output/wd01/wm_n1/wms/__init__.py` and `.../wms_eval/__init__.py` install a `sys.modules` alias table
  (`wms.model -> wms_new.model.wms`, `wms.train.WmSTrainConfig -> wms.config.train`, ...) generated from the PATH_MAP, so
  `from wms.train import WmSTrainConfig` (old name, old path root `output/wd01/wm_n1`) still resolves. Because the new package is
  also called `wms`, the shim lives under a different import root: the new package is installed as `wms`; the shim directory is only put
  on `sys.path` by legacy scripts that `sys.path.insert(0, wm_n1)` — to avoid a name clash the shim package is renamed on disk to
  `output/wd01/wm_n1/wms_legacy_shim/` and the old directories `wms/` and `wms_eval/` are moved intact to `archive/wms_era/wms/`,
  `archive/wms_era/wms_eval/`. Legacy scripts that must run unmodified use `python -c "import wms_legacy_shim; ..."` or the helper
  `tools/legacy_env.py` (adds `archive/wms_era` to `sys.path` *instead of* the new package). Frozen text references to the old
  paths resolve through PATH_MAP, not through the filesystem. (Decision recorded because import-name collision is the main hazard.)
* Root `wms_decide.py` stays as a 3-line shim: `from wms.eval.decide import *`.

### 6.3 Archive list (all `git mv`)
* `archive/wms_era/wms/`: brakefix_audit, brakefix_smoke, gate_run2, gate_s2, c1 (after merge), 8 shell chains, smoke400.json (+ their tests).
* `archive/wms_era/wms_eval/`: ctm*, run1_*, report, traffic_eval, r2set, prepare_states, shift_recovery, shift_traffic_split, c1 (after merge) (+ their tests).
* `archive/wms_era/_out/`: the 42 diagnostic scripts, non-active chains, ~190 records; `archive/stray_root/`: items in 1.7.
* `archive/wms_era/tools/`: brakefix_verify, main2_archive, main2_records.
* *Candidates only, not tonight (need owner approval + reverse-reference scan):* the non-WM-S directories of `output/wd01/` and the
  ~100 REGISTRATION/REVIEW md at `output/wd01/wm_n1/` root -> `archive/wd01_era/`.
* Stays untouched: `traffic_shepherd/`, `baselines/`, `scenarios/`, `assets/`, `protocols/`, `archive/pre_circles`, `archive/circles_era`, `runs/`, `output/controller_verification_v{2,4,7}/`, `output/wd01/wm_n1/{hdv,features,closedloop,rollout2,train,wm_api,...}` (vendored copies are made; originals remain for frozen replay).

---------------------------------------------------------------------------------------------

## 7. Branch/PR workflow for later GitHub use

* `main` = integrated tested state. Work branches `wms/<area>-<topic>`; PRs require: `ruff` clean, `pytest -q` green (golden with fixtures), import-graph test green, `git diff --stat` for `traffic_shepherd/` empty (unless a PR is explicitly labelled "sim change" and carries evidence).
* PR template asks: golden ids touched, config fingerprint of any new preset, change-log row (KEPT/REJECTED/PENDING), data names used (logical), confirmation that no restricted path was opened.
* CODEOWNERS: `src/wms/model, physics` (model owner), `src/wms/train` (training owner), `src/wms/eval` (eval owner), `traffic_shepherd/` (frozen: owner approval only).
* Private repo push excludes `data_cache/`, `runs/` bulk, `*.pt`; large evidence stays on `gdrive:research-archive` (registry).

## 8. Out of scope tonight (explicit)

Splitting `planner/closed_loop.py`, `perfect_sim.py`, `recorded.py`, `loop.py`; trimming the vendored pack reader; changing any model
or loss behaviour; any retraining; moving/deleting data on Drive or disk; edits to `traffic_shepherd` controller code; opening any
restricted dataset. A later phase may add "behaviour-changing" improvements only as separate PRs with a change-log row.

## 9. Open risks and how the design handles them

| risk | mitigation |
|---|---|
| name clash `wms` (new package vs legacy dir) | legacy dirs moved intact to `archive/wms_era/`; shim package has a different name; `tools/legacy_env.py` (6.2) |
| float drift from splitting `WmSModel` | pure-function extraction only; G1/G10/G11; state_dict surface frozen (4.2) |
| flat-config <-> nested-config divergence | `to_legacy_flat` is the compatibility contract; checkpoints carry both; G6 on real receipts |
| golden tests vacuous without data | synthetic fixture variants always run; real-data variants skip with an actionable fetch hint |
| gates silently green on fake paths | `gates.py` refuses fixture paths in `--mode run`; integration step 2 uses the real data path |
| missing modules behind old tests (cv09-cv28) | WP-F `--collect-only` classification; dead tests archived with their era, none deleted |
| concurrent edits to shared files | single owner per shared file (A for packaging/conftest/PATH_MAP header, integrator for docs) |
| restricted-data exposure | registry/guard refuse sealed tokens; CI scan of source for such literals; WPs briefed not to open such paths |
