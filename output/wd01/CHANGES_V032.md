# WD-01 Part B v0.3.2 —review-fix changelog (data / features / training)

Scope: every confirmed Part B review finding (incl. the B3b follow-ups),
a NEW derived dataset root `E:/research-dense/output/wd01_partB_v0data_v032`
(parent v0.3 root untouched, MANIFEST sha256
`5cf9fa479f1ba19a209c1ed2c486663ed033866a5545eea3f468de6c457f37a1`), and
deployable value-model refits with `--gbm-variant auto` for BOTH sensing
variants.  ZERO SUMO/TraCI.  v0.2/v0.3/v0.3.1 roots, results and backups
untouched; no D: paths; HOLDOUT never built/opened; CAL read once, for
the final report tables.

Owner decisions applied (output/wd01/OWNER_DECISIONS_WD01.md, 2026-09-29):
CAV-STRICT (`deployable_strict`) is the HEADLINE; the all-vehicle
entrance-detector variant (`deployable_detector`) is a labelled
SENSITIVITY; `pre_fix` is a label/stratifier only for deployable claims
—it stays a training feature but every predict path fixes it to the
post-fix value and this file says so.

## Fix table (finding -> file/function -> test)

| # | Finding / fix | File :: function | Test |
|---|---|---|---|
| 1 | Deployable ctx[1]=(t-t0)/600 together with ctx[0]=t/1000 made the campaign branch time t0 exactly recoverable -> ctx[1] identically 0 in BOTH deployable variants; oracle col 1 dropped from the build verification | `partB_v0data/deployable_features.py :: SourceObs.row/blocks`; `partB_v0data/build_v031.py :: _CTX_COLS_CHECKED` (0,3,6,7); `partB_v0data/build_v032.py :: build_sidecars` (asserts zero col) | `partB_v0data/tests/test_v032_fixes.py :: test_fix1_t0_not_recoverable`, `test_fix1_row_level_t0_invariance` (two t0 values -> byte-identical blocks); updated `test_build_v031.py` |
| 2 | `build_v0data.t_ins_pre_from_insertions_file` / v0lib insertions-log fallback returned an EMPTY pre-t0 history when the base trace was not cached (chunk1_fix27 bases live in campaign_stage1_verify27), inflating w8473 labels ~8.8x in tau30 mean wait -> fallback REMOVED; pre-t0 history always from the base run's `physics_steps` departed_ids (bases_root aware); w8473m2700p40t540__native rebuilt | `partB_v0data/build_v0data.py :: t_ins_pre_from_base_physics` (replaces `t_ins_pre_from_insertions_file`); `partB_v0data/v0lib.py :: build_run_trace` (designation from physics cav_ids only); `partB_v0data/build_v032.py :: rebuild_w8473` (+ tau30 wait-ratio hard check in [0.3, 1.0]) | `test_v032_fixes_data.py :: test_fix2_no_insertions_jsonl_in_label_feature_code`, `test_fix2_labels_build_without_insertions_file`, `test_fix2_missing_base_insertions_yields_physics_history`; builder hard check recorded in MANIFEST `build.rebuild` |
| 3 | Campaign `__native`/`__pi` arms in chunks 1-3 + fix27 are random per-CAV switching schedules, NOT fleet native/PI -> canonical policy `switching_mix` everywhere; label parquet `policy` column + MANIFEST prose relabelled; per-CAV actions only from switch_schedule.json / suffix_commands.jsonl (never the dispatcher action log) | `partB_v0/data.py :: POLICY_OF_SOURCE`; `partB_v0data/v0lib.py :: SOURCE_POLICY`; `partB_v0data/build_v032.py :: POLICY_OF_SOURCE + relabel_policy` (MANIFEST `policies`, `dispatcher_rows_read=false`); `partB_v0/model.py :: POLICIES` | `test_v032_fixes_data.py :: test_fix3_policy_map_no_native_claim`, `test_fix3_no_dispatcher_rows_read`; `test_v032_fixes_partb.py :: test_fix13_policies_canonical`; trainer maps assert equality |
| 5 | BOTH deployable variants: `deployable_strict` (HEADLINE: CAV self-reports only —no all-vehicle entrance counts, entrance queues zeroed, no penetration-revealing ratio) and `deployable_detector` (labelled SENSITIVITY: all-vehicle entrance counts + queues); `--features {oracle,deployable_detector,deployable_strict}` | `partB_v0data/deployable_features.py :: SourceObs.row/blocks (variant=)`; `partB_v0data/build_v032.py :: build_sidecars` (`_feat_dep_det.npz` / `_feat_dep_strict.npz`); `partB_v0/data.py :: FEATURE_SETS/FEAT_SUFFIX`; `partB_v0/gbm.py :: FEATURE_SETS, STRICT_DEMAND_DEP_NAMES, summary_features`; `partB_v0/__main__.py` CLI | `test_v032_fixes_data.py :: test_fix5_strict_has_no_all_vehicle_counts_and_no_ratio` (strict invariant to every non-CAV insertion), `test_mini_root_v032_build`; `test_v032_fixes_partb.py :: test_fix11_summary_names_strict`, `test_fix11_feature_set_validation_and_names` |
| 6 | Every eval_report states the ideal sensing premise | `partB_v0/evaluate.py :: render_report` + `render_net_report` | premise line asserted present in every report produced by the v0.3.2 runs (eval_report.md headers); checked in `test_v032_fixes_partb.py :: test_fix13_no_damping_minus_native_claim` scope and pasted reports below |
| 9 | NN bootstrap sampled base and switching arms independently -> world-grouped bootstrap (world id = `<chunk>/<world>`, both arms together) in the v0.2 trainer, the v0.3 net trainer and the B6 member pools | `partB_v0/data.py :: world_group_ids, bootstrap_world_ids, V03TensorData.bootstrap_worlds/set_members`; `partB_v0/train.py :: MemberTrainer.__init__` | `test_v032_fixes_partb.py :: test_fix9_bootstrap_world_grouped`, `test_fix9_member_trainer_pools_world_grouped` |
| 10 | HGB early stopping used an internal random-row validation split (a world straddled fit/validation) -> world-grouped holdout via `fit(X_val=...)` (sklearn >= 1.7), else early stopping OFF at fixed max_iter; same for the collision head and the permanent point-GBM baseline | `partB_v0/gbm.py :: _es_val_split, _fit_hgb, _hgb, fit (coll head), fit_baselines` | `test_v032_fixes_partb.py :: test_fix10_es_split_world_disjoint`, `test_fix10_fit_passes_grouped_validation`, `test_fix10_baselines_have_no_random_row_es` |
| 11 | `pre_fix` as a training feature would leak the fleet fix into deployment -> kept as a training feature, FIXED to the post-fix value on every predict path for deployable feature sets (oracle unchanged); stated in sidecar + report | `partB_v0/gbm.py :: PRE_FIX_PREDICT_VALUE/NOTE, row_feature_matrix(pre_fix_override=), GbmValueModel._pre_fix_override/_matrix_global/_row_matrix, save/load`; `baselines_section` queries CAL with the same override (apples-to-apples) | `test_v032_fixes_partb.py :: test_fix11_pre_fix_fixed_at_predict_for_deployable`, `test_fix11_override_in_row_feature_matrix`, `test_fix11_sidecar_records_the_fix` |
| 12 | Old `train.py` selected members on CALIBRATION native CRPS@300 -> REMOVED with a hard guard; selection = TRAIN world-grouped OOB CRPS@300 (`oob_crps300` over the member's OOB rows); CAL is read once, for the final report | `partB_v0/train.py :: CalSelectionRemoved, cal_native_crps300 (raises), oob_crps300, run_training eval_fn, save_best/ckpt meta` | `test_v032_fixes_partb.py :: test_fix12_cal_selection_raises`, `test_fix12_oob_selector_guard`, `test_fix12_guard_rejects_non_train`, `test_fix12_run_training_selects_on_oob` |
| 13 | B3b Delta head described as a fleet-native contrast but trains on base vs `__native` pairs = damping fleet vs switching mix -> head/metrics/docs relabelled `damping_fleet - switching_mix` everywhere; true-native pairs (chunk-4 `__native_base`) out of scope | `partB_v0/train_net.py` (docstrings, delta-head comment), `partB_v0/model.py :: V03ValueNet` docstring, `partB_v0/evaluate.py :: render_net_report` section 5 + `_policy_contrast` (section 3 renamed "Damping fleet vs switching mix"), `CHANGES_V03_NET.md` | `test_v032_fixes_partb.py :: test_fix13_no_damping_minus_native_claim` (repo-wide grep), `test_fix13_policies_canonical` |

## Dataset root v0.3.2

- Parent: `E:/research-dense/output/wd01_partB_v0data_v03` (READ-ONLY),
  MANIFEST sha256 `5cf9fa47...f37a1`, verified at link time; 1429 files
  hard-linked/copied with sha256 verified before AND after; 3 files of
  `chunk1_fix27/w8473m2700p40t540__native` excluded and rebuilt (fix #2).
- HOLDOUT: 42 worlds listed in the parent MANIFEST, never opened
  (in-code guards; `holdout_accessed=false` in MANIFEST + build summary).
- Per-source additions: `<src>_feat_dep_strict.npz` (HEADLINE) and
  `<src>_feat_dep_det.npz` (SENSITIVITY), both context-verified against
  the stored oracle context (detector: cols 0,3,6,7; strict: cols 0,3;
  col 1 dropped by fix #1) and internally consistent (strict demand ==
  detector CAV columns).
- Policy relabels (#3): 358 label parquets rewritten to the v0.3.2
  provenance prose; MANIFEST records canonical `policy_v032` per source.
- Integrity: `MANIFEST.json` + `SHA256SUMS` (MANIFEST + every worlds/
  file); backup at `E:/research-dense-backups/wd01/partB_v0data_v032/`
  (hash-verified).

Build summary (from `results/build_v032_last.json`; generated
2026-09-29T20:28:33Z, detached run of the previous session completed
after the spawn-guard fix below):

```
root: E:\research-dense\output\wd01_partB_v0data_v032
parent: E:\research-dense\output\wd01_partB_v0data_v03 (MANIFEST sha256 5cf9fa479f1b...)
link: 1429 files verified (1429 hard links, 0 copies; 3 skipped for the rebuild), 0 mismatches
relabel: 357 label parquets -> v0.3.2 provenance prose (switching_mix)
rebuild: chunk1_fix27/w8473m2700p40t540__native tau30 mean wait 1451 s vs base 1670 s -> ratio 0.87
sidecars: 358 detector (SENSITIVITY) + 358 strict (HEADLINE) written and context-verified
SHA256SUMS: 2149 entries
holdout_accessed=false
```

Independent close-out verification (this session, full-file re-hash):

```
v0.3.2 root : SHA256SUMS check: 2149 entries, 0 mismatches, 0 missing
backup copy : BACKUP SHA256SUMS check: 2149 entries, 0 mismatches, 0 missing
parent v0.3 : PARENT v0.3 root check: 1433 entries, 0 mismatches, 0 missing
```

Build-run note: the detached runner `_run_build_v032.py` had no
`if __name__ == "__main__"` guard, so on Windows spawn every pool child
re-executed the whole build and the four racing pipelines killed each
other with WinError 32 sharing violations (1209 aborted link-stage
passes in `_build_v032.log`).  Adding the guard made every subsequently
spawned child a plain pool worker; the original pass then ran to
completion on its own (no process was killed).  All writes are
.tmp+rename and sha-verified, and the final root re-verified clean, so
the racing left no corruption; it only cost wall-clock time.

## Deployable refits (gbm-variant auto)

Two NEW result dirs (v0.3.1_r2 untouched):

- HEADLINE: `E:/research-dense/output/partB_results/v0.3.2_strict`
  (`--features deployable_strict`)
- SENSITIVITY: `E:/research-dense/output/partB_results/v0.3.2_detector`
  (`--features deployable_detector`)

Selection/calibration on TRAIN world-grouped OOB only (CAL guard stays;
CAL read once for the report).  Per variant the eval_report carries:
section 0 (chosen variant per tau with OOB CRPS), section 6/7 (CAL
acceptance vs ridge on the SAME inputs, tie = TIE), 7b (TRAIN-internal
OOB acceptance), section 1 (overall CRPS vs climatology, coverage, PIT),
the #6 premise line and the #11 pre_fix statement.

Acceptance summary per variant and the comparison vs
`v0.3.1_selected_deployable_r2` (same CAL split; indicative across
dataset versions —v0.3.1_r2 was trained on the v0.3.1 root):

CAL overall (item 1, 10959 CAL rows, both policies and chunks; model
beats climatology everywhere, lower CRPS is better):

| run | CRPS model | CRPS clim | 10-90 coverage | PIT mean | data root |
|---|---|---|---|---|---|
| v0.3.2_strict (HEADLINE) | 977.0 | 1911.4 | 0.800 | 0.547 | v0.3.2 |
| v0.3.2_detector (SENSITIVITY) | 702.2 | 1911.4 | 0.806 | 0.512 | v0.3.2 |
| v0.3.1_selected_deployable_r2 (reference) | 770.9 | 1912.2 | 0.819 | 0.501 | v0.3.1 |

HARD acceptance vs ridge (section 7 = CAL, section 7b = TRAIN
world-grouped OOB; a tie is a TIE, not a pass):

| run | s7 CAL verdict | s7b OOB verdict | shipped tau variants |
|---|---|---|---|
| v0.3.2_strict | FAIL (3/5 pass) | FAIL (4/5 pass) | hybrid @30,60,120,300,570 |
| v0.3.2_detector | FAIL (3/5 pass) | FAIL (3/5 pass) | hybrid @30,60,120; ridge @570 |
| v0.3.1_r2 (reference) | FAIL (3/5 pass) | FAIL (3/5 pass) | hybrid @30,60,120; ridge @300,570 |

Per-tau detail (OOB delay CRPS, section 0 / 7b rows; hybrid unless
noted):

| tau | strict OOB | detector OOB | r2 OOB |
|---|---|---|---|
| 30 | 220.9 | 53.0 | 114.9 |
| 60 | 453.6 | 150.5 | 269.5 |
| 120 | 942.6 | 434.3 | 679.6 |
| 300 | 2570.2 | 1767.4 | 2355.1 (ridge) |
| 570 | 5272.0 | 4640.6 | 5085.3 (ridge) |

Reading (indicative across dataset versions): the detector sensitivity
lands within ~10% of the r2 reference at every tau; the strict HEADLINE
variant pays a real information price for dropping all-vehicle entrance
counts (roughly +40% OOB CRPS vs detector at 30-120 s) but still passes
the ridge bar at 30/60/120 on both CAL and OOB and beats climatology
everywhere.  The long-tau (300/570) FAILs vs ridge persist in all three
runs, matching the v0.3.1 finding that long-horizon delay sits close to
the deterministic demand-growth curve the ridge point already captures.

## Verification

```
WD_FORBID_D=1 python -m pytest output/wd01/partB_v0data/tests -q
  -> 47 passed, 4 skipped in 334.79s (skips: built-V0 fixtures, worlds/ git-ignored by design)
WD_FORBID_D=1 python -m pytest output/wd01/partB_v0/tests -q
  -> 135 passed, 367 warnings in 1016.91s
WD_FORBID_D=1 python -m pytest output/wd01/tests -q
  -> 18 passed in 0.30s
v0.3.2 fix tests (both new files together):
  -> 25 passed, 1 warning in 38.40s
ruff: no new findings vs the 6d7f7b0 baseline in any task-modified file
git log at close-out: f6dfdc1 "WD-01 Part B v0.3.2: build verified+backed
  up (2149/2149), strict+detector refits complete, r2 comparison +
  close-out evidence in CHANGES_V032" (train-time commits recorded in
  the runs' env_report.json files)
```
