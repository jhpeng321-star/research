# WP-I integration report - clean WM-S rewrite (2026-10-10)

Branch `wms/clean-int` (from `wms/clean-d`, which already contained A, B*, C*, D; `*` = earlier snapshots).
CPU only, no GPU, no SUMO, no full training. Interpreter `rl_driving` (Python 3.10, torch 2.12 CPU), 2 torch
threads, at most 2 pytest processes at a time. Design: `docs/ARCHITECTURE_V4.md`; per-package interface
changes: `docs/ARCHITECTURE_V4_CHANGES.md` (WP-I section at the end).

## 1. What was merged

| step | commit | notes |
|---|---|---|
| `wms/clean-e` (eval) | `ffcb0d2` | add/add on `src/wms/eval/**` (WP-A stubs vs E): E taken; `tests/equivalence/conftest.py` and `ARCHITECTURE_V4_CHANGES.md` combined |
| `wms/clean-f` (hygiene) | `776ae52` | clean merge: legacy `wms`/`wms_eval` trees moved to `archive/wms_era/wm_n1/` with shims, stray files, `_out` diagnostics, dead tests archived, README/INDEX |
| `wms/clean-b` again | `ca25e84` | WP-D had merged an *early* B: `model/checkpoint.py`, `model/factory.py` and `legacy/blackbox/traj.py` were still WP-A stubs and four B equivalence test files were missing |
| `wms/clean-c` again | `ca2f531` | same for C: `data/window_lists.py`, `data/lpwo.py`, `tools/holdout_guard.py`, `tests/fixtures/{mini_switch_pack,packsynth}.py`, six `tests/unit/data` files and `test_data_equivalence.py` were missing. WP-A's `core/*` kept (superset of C's provisional ones) |
| `wms/golden` | `2f7d3af` | `tests/golden/{README,capture_golden,golden_compare}.py` + 19 MB of reference data (incl. `c1/wm_s.pt`, 5 MB) |
| WP-I fixes | `3f8660d` .. `75d50d8` | section 2 |

No file was deleted: `git diff --diff-filter=D $(git merge-base wms/clean HEAD) HEAD` is empty (moves are renames).
R8: `git diff --stat bf7d4cb -- traffic_shepherd output/controller_verification_v{2,4,7} baselines scenarios` is empty.
No sealed / CAL / HOLDOUT / chunk-5 / J4B / INC1 data was opened (registry and `core.guard` refuse such paths; the golden
test reads only TRAIN packs, the lpwo_v3 lists and the C1 wall_same states).

## 2. Fixes made so the pieces connect

1. **Legacy loaders repointed** (WP-F moved the legacy trees): `tests/equivalence/legacy.py`, `tests/support/legacy_eval_bridge.py`,
   `tests/unit/config/old_code.py` use `archive/wms_era/wm_n1/wms` (siblings `hdv`, `features`... stay in `output/wd01/wm_n1`, only ever
   appended to `sys.path`). `tests/unit/train/_compat.py` lost all stand-ins (B/C/D are real); WP-D's stub-checkpoint shim is gone and the
   trainer writes through B's `save_checkpoint`.
2. **`eval.backend.load_backend` uses the real modules**: `wms.model.checkpoint.load_checkpoint(config_factory=TrainConfig.from_legacy_flat)`,
   `wms.data.{history,presence,action_timeline}`, `wms.model.{entries,stability}`. The eval equivalence test now drives this path (the
   legacy bridge is no longer on the default path; `tests/support/legacy_eval_bridge.py` is kept, unused).
3. **Physics of a checkpoint**: C1/R3 checkpoints do not embed `merge_cal`/`sumo_fidelity`. `load_checkpoint` now also reads the sibling
   `TRAIN_RECEIPT.json["physics"]` (what the legacy plugin was given by hand) and the trainer writes `physics.json` at the start of `fit()`
   so stage checkpoints are scoreable. The B checkpoint equivalence test was adjusted to build the legacy model with the receipt physics.
4. **`cli/assemble.py` + `cli/train.py`**: the WP-A contract (`TrajectoryDataset(cfg.data, selection, corpora, ...)`) does not exist; WP-C
   kept the legacy API. `assemble` reproduces the legacy `wms.cli` assembly with registry-resolved names (composite `PackCorpus`,
   `from_real_packs`, `ControllerExclusion`/`HeldOutGuard`, switching mix, jam, `ModelSpec.from_mapping(cfg.to_legacy_flat())`,
   `build_physics` with the merge-cal params file). `preflight` resolves every declared data name and reports unresolvable/missing ones as
   failures. Legacy argv (`python -m wms train --pack ...`) assembles the identical 911/151 datasets (checked).
5. **`wms eval` / `wms score`** -> new `wms/eval/run.py::build_result` (score each model once per configured set, compare each with the single
   previous best via `compare.compare`); `EvalSetConfig.subset` (`first:N`, `first_per_world:N`) gives the 20-state quick subset.
6. **Single source for shared constants** (`wms.core.constants`) with re-exports from `physics.state`, `model.history`, `model.controllers`,
   `data.history`, `eval.rollout_inputs`; `train.cav_drive` reaches the frozen controllers through `sim_bridge` (new test).
7. **Import-graph test**: three documented, narrow exceptions (`data/corpus -> legacy.vendor`, `data/pack_history -> model.entries`,
   `model/factory -> legacy.blackbox`); a stale exception fails the test.
8. **Configs**: `r3.toml` split into `r3_packs_only.toml` (base) + `[data.switching]` (resolved config unchanged); new `smoke_local.toml`,
   `golden_r3.toml` (proved equal to the capture argv by `tests/unit/config/test_golden_preset.py`), `score_smoke_local.toml`;
   `configs/data_registry.toml` registers the local packs / sidecars / lists / C1 checkpoint / state sets (no path in code);
   `scripts/make_smoke_lists.py` writes the 6-window validation list of the CPU smoke.
9. **Test groups**: the root `tests/test_wms_eval_*.py` and `test_wms_decide.py` import the *legacy* flat `wms.*` modules and cannot share a
   process with `src/wms`; `tests/conftest.py` runs them as `WMS_TEST_GROUP=legacy` and says so in the header of the default group.
   `tests/golden/test_golden_current.py` (legacy recapture) became `recapture_legacy_check.py` (importing the capture script prepends the legacy
   tree to `sys.path`); `pyproject.toml` ignores F841 in one frozen sim test so `ruff check src tests tools` is clean.

## 3. Test results (final runs on the committed tree)

| suite | command | result |
|---|---|---|
| unit | `pytest tests/unit` | **508 passed, 1 skipped** (`tomllib` absent on 3.10: the vendored TOML reader is used) |
| equivalence | `WMS_EQ_FULL=1 pytest tests/equivalence` | **76 passed** (485 s) incl. the 20-state C1 check (`WMS_EQ_FULL=1`) |
| golden | `pytest tests/golden` | **10 passed** (343 s) |
| legacy-`wms` root group | `WMS_TEST_GROUP=legacy pytest tests --basetemp=_out/...` | **175 passed** (87 s; 17 files, legacy modules through the shim) |
| sim + planner (default group rest) | `pytest tests --ignore=tests/{unit,equivalence,golden} --continue-on-collection-errors` | 1013 passed, 159 skipped, **69 failed, 18 errors** - all inherited, see below |
| lint | `ruff check src tests tools` | All checks passed |

Sim + planner failures are not caused by this work: the same run on an LF `git archive` of `wms/clean-d` (the branch this one is based on)
gives 65 failed / 53 errors with a failing-test set that contains every failure of this branch except four
`tests/planner/test_holdout_guard_seal.py` tests (error there because the file is absent, failure here because WP-C's `tools/holdout_guard.py` is a
different tool than the "reconstructor-owned" one the planner expects, which exposes `CHUNK5_SEAL_ENV`). The failures are missing untracked material of the main checkout
(`output/planner_v1/**` for 12 planner modules; `runs/` evidence deleted by MIG-04 for the ledger and CV corpus tests) and recorded
hashes/sizes of archived files. The identical failure set appears in the CRLF worktree and in the LF archive of HEAD.

### Golden comparison (`tests/golden/test_golden_new.py`, new stack only, tolerances of `golden_compare.TOLERANCES`)

| artefact | compared | result |
|---|---|---|
| `selection.json` | kept keys of both splits in dataset order, per-pack counts (911 train / 151 val; pack_v3 294/47, dspeed 444/76, damping 173/28), receipts | exact (list paths by basename) |
| `batches_u{10,60}_hist{on,off}` (4 variants x 3 batches x 12 tensors) + json | decoded batches | exact, 0.0 |
| `forward_c1.npz/.json` | C1 checkpoint, teacher-forced 20 and 120 step rollouts | within 1e-6 (json: stored tensors; `driver_encoder.*` absent from the C1 file, see CHANGES) |
| `r3_init_state.npz` | fresh seed-101 R3 model weights | exact 0.0 (same RNG consumption order as the legacy model) |
| `forward_r3.npz/.json` | R3 model on history-on batches | within 1e-6 |
| `losses_r3.json` | every loss term, total, gate curve, per-parameter and per-group gradient norms, 3 batches | within 1e-5 |
| `train3_r3.json`, `train3_final_params.npz` | 3 Adam steps | within 1e-5, except the 3 null-direction attention-bias parameters (<= 1.4e-5 observed, checked at 1e-4; reason in CHANGES) |
| `eval_rollouts.json`, `eval_pred_frames.npz` | C1, first 3 wall_same states, 60 s, all arms: car metrics, section delays, predicted frames | within 1e-5 |

Eval equivalence vs the legacy `wms_eval` (cached legacy output from WP-E's run of the same unchanged code, checkpoint sha `87ed02e0...`):
first 3 states within 1e-6; **20-state quick subset within 1e-3 and median 60 s HDV |ds| = 11.9576 m** (reference 11.9576, tolerance 0.001).

## 4. Real-data end-to-end (local packs, CPU)

Commands (all with `PYTHONPATH=src`; outputs in `docs/records/integration_20261010/`):

```
python -m wms preflight --config configs/r3.toml --out runs/tmp_int/e2e_r3_preflight   # exit 2
python -m wms preflight --config configs/smoke_local.toml --out runs/tmp_int/e2e_smoke_pf   # passed
python -m wms train     --config configs/smoke_local.toml --mode run --out runs/active/int_smoke_local2
python -m wms score     --config configs/score_smoke_local.toml
```

* **`preflight r3.toml`**: every dataset of the recipe resolves and exists except the six switching/direct-output names
  (`pack_switching_train`, `identities_switching_train`, `driver_params_switching_train`, `windows_train_hetdata`,
  `action_timeline_index_train`, `zero_shot_heldout`), which are not on this machine and are deliberately not registered - the run is
  refused (exit 2), as designed. Plan: stages 10/30/60/120 s, steps 2500/2500/1875/1250, terms position,speed,lane,asym_speed,decel_weight,propagation.
  `r3_packs_only.toml` / `smoke_local.toml` (the same recipe without the switching mix) pass preflight.
* **Smoke training** (`smoke_local.toml`: 2 optimizer steps per stage x 4 stages, batch 2, CPU, 2 threads, merge_cal on): the dataset log line
  reads `train kept windows total 911/3060 | pack_damping 173/173, pack_dspeed 444/444, pack_v3 294/294` and
  `val kept windows total 151/3060 | pack_damping 28/28, pack_dspeed 76/76, pack_v3 47/47` (the validation list of the smoke run itself is
  the 6-window `smoke_lists/windows_val_smoke.json`, because the trainer validates on the whole list after each stage: a first attempt with the
  full 151-window list was stopped after the first two stages, log kept). All eight steps finite; **every enabled loss term nonzero at every step**
  (position 14.7..420.9, speed 0.76..3.95, lane 0.205..0.554, asym_speed 8.97..219.9, decel_weight 0.30..20.9, propagation 0.014..0.150);
  `wm_s.pt`, four stage checkpoints, `TRAIN_RECEIPT.json` (schema `wm_s_train_receipt/1`, 911 records, `config_fingerprint`), `physics.json`
  written; `load_checkpoint(strict)` + `TrainConfig.from_legacy_flat` accept the result. Terms not enabled by the R3 recipe (residual_mag,
  lane_rate, soft bound, history) are covered by the golden losses / 3-step tests of `golden_r3.toml` (all nonzero there).
* **`wms score`** (smoke checkpoint vs C1FAST, `wall_same`, `first_per_world:20`, teacher-forced 60 s rollouts, 0 arm errors):
  C1FAST median HDV |ds| at 60 s = **11.9576 m** (reproduces the reference), smoke checkpoint 36.7743 m; paired difference +26.59 m
  [9.30, 48.03], verdict `regressed` (as expected after 8 steps). This is a mechanics check, not training evidence; no ledger entry was created.

## 5. Remaining gaps and risks

* **`tools/holdout_guard.py` name collision.** WP-C's tool (window-list source guard) has the same path as the planner's "reconstructor-owned"
  `tools/holdout_guard.py` of the main checkout (not tracked in this branch's base). Merging into the main working tree will collide; owner decision needed
  (rename WP-C's, or reconcile).
* **Not ported** (2-line stubs from WP-A remain): `eval/{walls,decide,verdict,driver_recovery}.py`, `eval/sets/*`, `legacy/a11_model.py`,
  `legacy/blackbox/{id,eval_adapter}.py`, `data/{live_state,geometry,counters}.py`. `wms.eval.decide.predict_states` still raises
  `NotYetImplemented` (root `wms_decide.py` keeps using the archived legacy through the shim); closed-loop CAV mode and black-box kinds in the
  predictor; the trainer ignores `data.sidecars.driver_sidecar` (as WP-D documented).
* **Switching pack path untested on real data** (not available locally): `assemble` is call-for-call the legacy sequence and `mixdata` has WP-C's
  unit/equivalence tests, but `build_switch_datasets` -> `mix_datasets` -> `fit` has not run end to end here.
* **Validation cost**: the trainer validates on the whole validation list after every stage (legacy behaviour, `val_batches` is recorded but unused);
  CPU smokes need a small list (`scripts/make_smoke_lists.py`).
* `tests/planner` (12 modules) cannot be collected without `output/planner_v1`; ledger/CV-evidence tests fail without the deleted `runs/` evidence.
* Root legacy tests (`WMS_TEST_GROUP=legacy`) need `--basetemp` under `_out/` (the legacy checkpoint loader insists on a workspace-confined path).
* Scratch: `runs/tmp_int/` (git-ignored, ~700 MB incl. two `git archive` extractions used for the baseline comparison) can be removed by the owner.
