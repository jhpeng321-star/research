# REVIEW — WM-N1 pieces TRAIN / BUNDLE / ADAPTER (first pass)

Reviewer: execution agent, 2026-09-30.  Scope: first-pass review of the three
piece branches against the frozen records `INTERFACES.md`, `WM_N1_DESIGN.md`,
`REGISTRATION_WM_N1.md` and `REGISTRATION_WM_N1_AMENDMENT_1.md`
(`git show zcode/wd01:output/wd01/wm_n1/REGISTRATION_WM_N1_AMENDMENT_1.md`).
No SUMO; no training beyond CPU smokes; no HOLDOUT / CAL / chunk-5 data opened
(the CAL pack used by tests is a synthetic fixture).

- TRAIN: `zcode/wmn1train` @ `520bd37`
- BUNDLE: `zcode/wmn1bundle` @ `42798c0`
- ADAPTER: `zcode/wmn1adapter` @ `a0caf6b`
- Integration target: `zcode/revwm3` @ `zcode/wmn1int` `c895083`

## 1. Merge

The three branches were merged `--no-ff` into `zcode/revwm3` on top of
`c895083` (`zcode/wmn1int`); no conflicts (disjoint piece directories):

```
90d6df2 Merge zcode/wmn1train (TRAIN piece) into zcode/revwm3 for WM-N1 first-pass review
7b7ce15 Merge zcode/wmn1bundle (BUNDLE piece) into zcode/revwm3 for WM-N1 first-pass review
4e4d68f Merge zcode/wmn1adapter (ADAPTER piece) into zcode/revwm3 for WM-N1 first-pass review
```

## 2. Verdict per acceptance item

Legend: **PASS** / **PASS\*** (passes with a recorded caveat) / **FAIL → fixed**
(path fixed in this review).

### TRAIN

| # | Item | Verdict | Evidence |
| --- | --- | --- | --- |
| T1 | Pair sampler only same `world`+`t0`, treatment/reference order matching the `contrast_loss` sign | **PASS** | `dataset.py:164-187,307-329` group by `pair_key="<world>@t0=<t0>"`; `sibling_pairs`/`contrast_batch_indices` return `(a=treatment, b=native)`; `model/losses.py:116-136` computes `dq=q_a-q_b, dy=y_a-y_b`, i.e. `V(P)-V(all-native)`. `test_dataset.py::test_sibling_pairs_*`, `test_contrast_batch_indices`. |
| T2 | Outcome mapping correct vs `features/outcomes.py` columns | **PASS** | `OUTCOME_COLUMN_MAP` maps `delay_s→total_delay_ref_s`, `ttc_lt3_s→ttc_lt3_veh_s`, `hard_brake_s→hard_brake_s`; `collision_any→collision`, `tau_available→outcome_mask` (`dataset.py:71-77`); matches `features/outcomes.py:23-28`. `test_dataset.py::test_outcome_column_map_is_exact`. |
| T3 | TRAIN-only guard + chunk-5 refusal | **PASS** | `_normalize_split` accepts only TRAIN/CALIBRATION; `_refuse_item` refuses HOLDOUT/TEST/SEALED and any `chunk5*` item at pack open (`dataset.py:198-253`). `test_dataset.py::test_cal_requires_explicit_split`, `test_chunk5_pack_is_refused`. |
| T4 | Checkpoint `weights_only` explicit | **PASS** | `trainer.py:193` `torch.load(path, map_location="cpu", weights_only=True)`. `test_trainer.py::test_resume_loads_with_weights_only_true` spies the kwarg and greps the source. |
| T5 | Member seeds recorded before training | **PASS** | `cli.py:60-62` calls `record_seed` (writes `SEEDS.json`) before constructing `Trainer`/`fit`; `_write_receipt` also records `seed`/`member_order`. Amendment A1 order `(1,2,3,4,5)` in `config.MEMBER_ORDER`. |
| T6 | Deviations D1–D5 listed and judged | **PASS** | Listed in `train/README.md:67-86`. Judgement below (§3). |

### BUNDLE

| # | Item | Verdict | Evidence |
| --- | --- | --- | --- |
| B1 | Step order and HARD STOP after timing gate | **PASS** | `STAGE_ORDER` (`queue.py:36-37`); `bundle_queue` defaults `stop_after="timing-gate"` and runs only `STAGE_ORDER[:idx+1]` (`queue.py:333-357`); `run_stage` refuses out-of-order calls. `test_queue.py::test_stop_after_timing_gate_is_default` asserts no approval/train call. |
| B2 | APPROVAL binds TIMING_GATE sha + LF-shas `666233aa…` and `decb9656…`, CRLF-safe | **PASS** | `approval.py:41-45` `LF_SHA_AMENDMENT_1=decb9656…16c57`; `LF_SHA_WM_N1=666233aa…93246` from `evalh.seals`; `lf_sha256` normalises `\r\n`/`\r`→`\n` (`seals.py:55-62`); `require_approval` re-hashes the actual records and binds the raw-bytes `TIMING_GATE.json` sha written by `timing.write_timing_gate`. Independently recomputed: `REGISTRATION_WM_N1.md` LF-sha256 = `666233aa…93246`; `REGISTRATION_WM_N1_AMENDMENT_1.md` LF-sha256 = `decb9656…16c57`. `test_approval.py` covers CRLF copies, wrong declared sha, wrong live record, missing gate, missing fields. |
| B3 | Member-1-first | **PASS** | `_stage_member_1` trains only member 1; `_stage_members_2_5` refuses unless `members/member_1.pt` exists (`queue.py:259-277`). `test_queue.py::test_member_1_first_then_members_2_5`, `test_members_2_5_refused_without_member_1_checkpoint`. |
| B4 | Sealed-shard refusal | **PASS** | `_refuse_sealed` refuses manifest entries whose path matches `holdout/chunk-5/chunk5/chunk_5` and any declared `refused` list (`queue.py:153-164`). `test_queue.py::test_sealed_shard_refusal_{chunk5,holdout,declared}`. |
| B5 | Timing gate uses the REAL `WmN1Net` | **PASS** | `timing.py:95-97,158-206` builds `WmN1Net(cfg.wmn1_config())`; records `measured_model="model.net.WmN1Net"`, `proxy=False`; `make_config(tiny=True)` is a small real config, not the evalh `Toy`. `test_timing.py` (real forward+backward, real rollout). |
| B6 | Box commands in README executable as written (dry-run where possible) | **FAIL → fixed** | `python -m bundle` from the repository root fails (`No module named bundle`): the packages are top-level only under `output/wd01/wm_n1`. README §0 was corrected to run from `output/wd01/wm_n1` and the approval heredoc paths made consistent. Dry-run from that dir: `python -m bundle --help`, `python -m train --help`, `python -m corpus --help` all succeed. |
| B7 | No hardcoded `E:/` paths | **PASS** | `rg "E:/|D:/"` over `bundle/` is empty (also empty for `train/` and `adapter/`). |

### ADAPTER

| # | Item | Verdict | Evidence |
| --- | --- | --- | --- |
| A1 | `Predictor` protocol conformance | **PASS** | `WmN1Predictor(PredictorBase)` implements `name`, `quantile_levels=QUANTILE_LEVELS`, `predict_quantiles(rows)->[N,19]` (validated by `PredictorBase._validate`), `sample_rollouts(rows, horizons_s)->RolloutBatch`. `adapter/tests/test_predictor.py`. |
| A2 | 19→7 quantile mapping exact at shared levels | **PASS** | `quantiles.map_quantiles` linear-interpolates the quantile curve with `numpy.interp` (exact at nodes) and accumulates monotonicity; `POOL_LEVELS` equals `combine.stacking.DEFAULT_QUANTILES`; all 7 pool levels lie on the 19 neural levels. `test_quantiles.py`, `test_export.py::test_export_cal_accepted_by_stacking`. |
| A3 | OOF export member-eligibility logic sound | **PASS** | `eligible_members` returns members whose training worlds do not intersect the fold; folds scored with that subset are tagged `model_kind="oof"`; without `member_train_worlds` bookkeeping rows are explicitly tagged `"full-in-sample"` + `limit_note` (never silently mislabelled). `test_export.py::test_export_oof_{uses_eligible_members,documents_member_limit}`. |
| A4 | CAL export flag-gated | **PASS** | `export_cal` requires every row's split to be CALIBRATION via `combine.guards.guard_cal_only`; TRAIN/TEST/HOLDOUT refused. `test_adapter_guards.py::test_export_cal_refuses_non_cal`, `test_export_cal_requires_explicit_split`. |
| A5 | TEST/HOLDOUT/chunk-5 refused | **PASS** | guards name TEST/HOLDOUT explicitly; `_assert_not_chunk5` refuses `chunk5`/`chunk-5` world names in both entry points. `test_adapter_guards.py::test_export_oof_refuses_sealed_splits`, `test_export_*_refuses_chunk5`. |

### Cross-piece

| # | Item | Verdict | Evidence |
| --- | --- | --- | --- |
| X1 | Train checkpoint format == adapter loader == bundle expectations (+ small cross-piece test) | **FAIL → fixed** | The adapter directory loader required `config.json` + plain `state_dict`s, but `Trainer.save_checkpoint` writes `member_<k>.pt` payloads (weights nested under `"state_dict"`, plus optimizer/epoch/seed/config) and a `member_<k>.json` sidecar — no `config.json`. Fixed in `adapter/checkpoint.py` (accepts the trainer layout and normalises the flat `TrainConfig`). New test `train/tests/test_cross_piece.py`: train 5 tiny members into the bundle `members/` layout → `adapter.checkpoint.load_ensemble` → `WmN1Predictor.predict_quantiles` (shape `(1,19)`, finite, monotone). Passed in 32.96 s. |
| X2 | torch 2.5 API check | **PASS** | Grep over `train/`, `bundle/`, `adapter/`: only `torch.{allclose,as_tensor,bool,cat,cuda,device,equal,float,Generator,is_tensor,load,long,manual_seed,nn,no_grad,ones,optim,rand,randn,randperm,save,set_num_threads,stack,tensor,utils,zeros}` — all available in 2.5; no `compile/amp/autocast/SDPA/flex_attention/export`. `torch.load`/`torch.save` call sites all pass explicit `weights_only` where loading. Local runtime is `torch 2.12.1+cpu`. |
| X3 | `synth` test-module name collision — single combined pytest collects | **FAIL → fixed** | A combined pytest failed collection with 11 errors (`synth.py` ×3 across features/model/adapter; `test_cli`/`test_interface` combine↔train; `test_guards` features↔adapter). Renamed to unique basenames and updated imports. Combined collection is now 236 tests, 0 errors. |

## 3. TRAIN deviations D1–D5 (judged)

- **D1 — `caliber` → `calibrate_on_cal`.** Correct. `INTERFACES.md` Config-keys
  names `calibrate_on_cal`; the stub's `caliber` was a typo. Implementation
  matches the frozen key. Accept.
- **D2 — additive config helpers (`from_dict`, `model_config`,
  `seed_for_member`, `loss_weights`).** Accept. The frozen `load`/`save`/
  `to_dict` are unchanged; helpers are needed by the trainer/CLI and are
  consistent with `interfaces` (seeds recorded before training).
- **D3 — packed layout is train-owned.** Accept with note. `INTERFACES.md`
  leaves the CORPUS byte format to CORPUS, so TRAIN fixes the read side and
  `packsynth.py` builds a matching fixture. **Risk:** the real CORPUS writer
  must emit exactly this layout (`manifest.json` + `.npy` arrays +
  `outcomes.parquet`) or TRAIN will refuse/open wrongly. This is an integration
  contract to re-check when CORPUS lands; flagged as an open item (§6).
- **D4 — additive dataset helpers.** Accept. Frozen `MemmapCorpus` signatures
  (`__len__`, `__getitem__`, `sibling_pairs`, `worlds`) unchanged.
- **D5 — private Trainer internals.** Accept. Public signatures
  (`fit`, `save_checkpoint`, `resume`, `evaluate_cal`) unchanged; `evaluate_cal`
  opens CAL only through the explicit guard and never tunes hyperparameters
  (consistent with registration: CAL is diagnostics/selection only).

## 4. Overclaims / inaccuracies found (no code defect)

- **ADAPTER README checkpoint section** claimed the directory layout was
  `config.json` + plain `state_dict` "as `train.Trainer` writes it", which was
  false before this review. Corrected in the same fix as X1.
- **BUNDLE README §0** claimed the commands run "from the repository root".
  Corrected to `output/wd01/wm_n1` (B6).
- **BUNDLE `requirements-wmn1.txt`** comments describe the GPU box as
  `torch==2.5.1+cu121` while the local runtime is `2.12.1+cpu`; this is
  intentional and documented, not an overclaim.
- **`--stop-after` help text** prints `None` as a choice (from
  `(None,) + STAGE_ORDER`). Cosmetic only; the default is `timing-gate`.

## 5. Fixes applied in this review ("review fix" commits)

1. `54d684d` — adapter loads `train.Trainer` member checkpoints (flat
   `TrainConfig` + nested `state_dict`); adds `train/tests/test_cross_piece.py`;
   updates `adapter/README.md`; renames the three `synth.py` helpers and the
   colliding test modules (pure file moves).
2. `9eccd00` — import updates for the disambiguated test-module basenames.
3. `6cdcc7d` — bundle README run directory (`output/wd01/wm_n1`) so its
   commands execute as written.

## 6. Reported, NOT fixed (larger / out of first-pass scope)

1. **Early 1-member model cannot be loaded through the adapter.**
   `model.ensemble.WmN1Ensemble.__init__` requires exactly `N_MEMBERS=5` seeds
   (`ensemble.py:26-30`), so `adapter.load_ensemble` cannot represent the
   amendment A2 "train member 1 alone" checkpoint (a directory with only
   `member_1.pt`). A2 allows that early model for TRAIN/CAL diagnostics and a
   planner-integration smoke. Fixing it needs either a variable-size ensemble
   in `model/ensemble.py` (frozen e1) or a partial-ensemble container in the
   adapter — a design decision beyond a review fix. The loader currently fails
   loudly rather than silently, so no wrong result is produced.
2. **`require_approval` cannot succeed on this branch as-is.**
   `approval.AMENDMENT_PATH = output/wd01/wm_n1/REGISTRATION_WM_N1_AMENDMENT_1.md`
   must be present next to the bundle, but that file is committed only on
   `zcode/wd01` (commit `79e686e`), not on `zcode/wmn1int`/`zcode/revwm3`. The
   gate fails closed (`ApprovalMissing`), which is safe; the GPU box bundle
   must ship the amendment file or merge it before the approval stage can pass.
3. **CORPUS is still a stub** (`corpus.pack.convert`, `verify.verify_pack`,
   `manifest.build_upload_manifest`, CLI all raise `NotImplementedError`), so
   bundle stages 1–2 cannot run end-to-end yet; the queue's test suite fakes
   those collaborators. Expected (CORPUS is a separate piece), but the bundle
   cannot be exercised beyond `--help` and the timing gate until CORPUS lands.
4. **Outcome mask NaN edge.** `dataset.map_outcome_targets` sets
   `mask[i] = bool(available)`; a `tau_available` of `NaN` (rather than a
   bool/0/1) would be treated as `True`. Low risk with the frozen bool label
   column; noted, not changed.
5. **`bundle.approval.require_approval` does not validate the record's
   `stage` field** against the requested stage, so a single approval file
   authorises both `member-1` and `members-2-5`. Consistent with A1's single
   approval gate, but worth an explicit stage check if stricter binding is
   wanted.

## 7. Evidence — combined pytest (all wm_n1 test dirs)

Command (verbatim; `output/wd01/wm_n1/adapter/tests output/wd01/wm_n1/bundle/tests
output/wd01/wm_n1/combine/tests output/wd01/wm_n1/evalh/tests
output/wd01/wm_n1/features/tests output/wd01/wm_n1/inventory/tests
output/wd01/wm_n1/model/tests output/wd01/wm_n1/train/tests`, `WD_FORBID_D=1`,
`python -m pytest … -q`, `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`):

```
........................................................................ [ 30%]
........................................................................ [ 61%]
........................................................................ [ 91%]
....................                                                     [100%]
236 passed in 233.82s (0:03:53)
```

Pre-rename individual piece runs: `train/tests` 30 passed, `bundle/tests`
26 passed, `adapter/tests` 25 passed. Cross-piece test:
`train/tests/test_cross_piece.py` 1 passed in 32.96 s. `ruff check` over
`adapter/ bundle/ train/`: `All checks passed!`.

## 8. Evidence — `git log -8`

```
6cdcc7d review fix: bundle README runs from output/wd01/wm_n1 so its commands execute as written
9eccd00 review fix: update imports for disambiguated wm_n1 test module basenames
54d684d review fix: adapter loads train.Trainer member checkpoints (flat config + nested state_dict)
4e4d68f Merge zcode/wmn1adapter (ADAPTER piece) into zcode/revwm3 for WM-N1 first-pass review
7b7ce15 Merge zcode/wmn1bundle (BUNDLE piece) into zcode/revwm3 for WM-N1 first-pass review
90d6df2 Merge zcode/wmn1train (TRAIN piece) into zcode/revwm3 for WM-N1 first-pass review
a0caf6b wd01 WM-N1 adapter: WmN1Predictor (evalh) + CAL/OOF export (combine), CPU tests
520bd37 wd01 WM-N1 train: MemmapCorpus (packed memmap, sibling-pair sampler, outcome mapping), Trainer (checkpoint/resume, torch.load weights_only explicit), train CLI (member-1-first seeds); 30 CPU tests green under WD_FORBID_D=1, zero SUMO
```
