# First-pass review (RE-RUN): WM-N1 preparation merges

Reviewed 2026-10-01 on branch `zcode/revwm1b` @ `cadc9548` against
`WM_N1_DESIGN.md` and `REGISTRATION_WM_N1.md`. CPU only, zero SUMO, no
training beyond the existing <=2-run TRAIN micro-smoke (re-run into a tmp
dir). The four preparation merges under review:

| merge | scope |
| --- | --- |
| `1cb02e3` | inventory + size estimate (metadata only) |
| `74cc9c8` | featurizer + memmap writer |
| `b4201e7` | combination layer |
| `cadc954` | literature note |

Review fix: `d888be2`. No simulation was run; no HOLDOUT/CAL/chunk-5 file
was opened.

## Verdict

**PASS, with three small defects fixed and three observations recorded.**
The inventory opened-path audit, the featurizer's deployment-honesty guards
and the combination layer's leakage guards are real and independently
reproduced. Two featurizer contract defects were fixed (wrong registration
bound by the chunk-5 gate; tensors not matching the documented/estimated
dtypes) plus pre-existing lint.

## PASS/FAIL per acceptance item

### Inventory (`output/wd01/wm_n1/inventory`)

| # | item | verdict | evidence |
| --- | --- | --- | --- |
| I1 | tests green under `WD_FORBID_D=1` | PASS | 21 passed |
| I2 | `RUN_RECEIPT` read only for status/rows fields | PASS (see O1) | `read_run`: `status = receipt.get("exit_status")`; `rows_written = _receipt_rows(receipt)` -> `receipt["physics_rows_written"]` or `receipt["suffix_summary"]["rows_written"]`; also `receipt.get("saved_state_id")`. `evaluate_summary` is **never** referenced by key. |
| I3 | no `evaluate_summary` load/store for CAL runs | PASS | grep for `evaluate_summary` over `inventory_lib.py`/`build_inventory.py`: 0 hits. See O2 (whole-receipt `json.loads`, values never consumed). |
| I4 | `opened_paths.json` free of HOLDOUT/excluded tokens | PASS | 4403 paths; 0 hits for 13 excluded + 60 HOLDOUT worlds + `kept_disclosed`; 0 literal `HOLDOUT`; 0 `D:` paths; only `runs/`(4395) + `output/`(8) under the data root. Method: independent script over `opened_paths.json` vs `holdout_names.json` (`excluded`, `kept_disclosed`, all HOLDOUT run worlds). |
| I5 | no `dispatcher_rows` / `insertions` opened | PASS | 0 hits; open-audit test `test_opened_paths_clean`. |
| I6 | pairs keyed `world`+`t0` | PASS | 1629 pairs, 184 keys, 0 rows where `pair_key != f"{world}@{t0_s}"`; `build_pairs` groups by `["world","t0_s"]`. |
| I7 | chunks 1-3 `__native`/`__pi` labelled switching | PASS | chunk1: 280/280 `switching`; chunk2/3: 168/280 each `switching` (WM-72 relabel) with the remaining 112 being genuine J3 truearms `native`/`pi`. `wm72_relabel=True` on exactly the switching-relabelled rows. |
| I8 | micro-sample bounded to <=3 TRAIN runs | PASS | `opened_paths.json` has exactly 3 `physics_steps.jsonl` (chunk1 w8411, chunk2 w8481, chunk3 w8551), all TRAIN-filtered. |

### Featurizer (`output/wd01/wm_n1/features`)

| # | item | verdict | evidence |
| --- | --- | --- | --- |
| F1 | tests green | PASS | 32 pre-fix -> 34 post-fix passed |
| F2 | no forbidden inputs reach tensors | PASS | `featurizer.parse_run` reads only `lane_id`, `lane_position_m`, `speed_mps`, `acceleration_mps2`, `road_id`, `departed_ids`, `cav_ids`; `test_no_forbidden_inputs_change_features` proves changing `type_id`/`speed_mode` leaves `veh_step`/`veh_mask` byte-identical; `test_feature_names_are_deployment_honest` rejects `type_id/vtype/pre_fix/seed/t0/world/penetration/speed_mode`. `is_cav` is the design-approved exception. |
| F3 | counter uses counts <= t only | PASS | `counters.DepartureLog.counter_at`: `keep = (age >= 0.0) & (age < COUNTER_BINS*COUNTER_BIN_S)`; `test_counter_bins_only_past`. |
| F4 | PI token from the planned schedule | PASS | `actions_adapter.PlannedActionTimeline.action_at` returns `fleet_policy`, else the planned `switch_schedule.json` token, falling back to the realised-owner timeline only when planned is `unknown`; `test_pi_token_is_planned_not_damping` asserts a PI CAV written by the damping owner is `action_pi`. |
| F5 | chunk-5 gate binds the chunk-5 registration LF-sha256 `f8c7d760...4b31e` | **FAIL -> FIXED** | Pre-fix `registration_sha()` raw-hashed `REGISTRATION_WM_N1.md` (wrong file, and raw != the frozen LF value). Fix `d888be2`: gate is `CHUNK5_REGISTRATION_LF_SHA256 = f8c7d760...4b31e`, LF-normalised hashing, file-drift check; `test_chunk5_gate_binds_frozen_chunk5_registration`. |
| F6 | FORMAT.md shapes match the writer | **FAIL -> FIXED** (dtype, not rank) | Ranks all matched, but `counter` was written float32 (FORMAT S5 + `expected_bytes` say float16) and `slot_index` int64 (FORMAT S2 + `expected_bytes` say int32). Fix `d888be2`: cast both; `test_tensor_dtypes_match_format`. |
| F7 | no `dispatcher_rows`/`insertions` opened | PASS | static grep + `sys.addaudithook` functional guard `test_no_forbidden_reads_functional` / `test_static_no_forbidden_filenames`. |
| F8 | writer resumable, refuses overwrite | PASS | `writer.write_shard` + `test_resumable_skip_and_index`, `test_refuses_overwrite`. |
| F9 | micro-smoke reproduces | PASS | re-run into tmp; see below. |

### Combination layer (`output/wd01/wm_n1/combine`)

| # | item | verdict | evidence |
| --- | --- | --- | --- |
| C1 | tests green | PASS | 27 passed |
| C2 | OOF world-leakage test is real | PASS | `test_oof_no_world_leakage_and_partition` checks each TRAIN row's fold model was fitted on `all TRAIN worlds - row's world`; `test_build_raises_when_builder_leaks_non_train` injects a leaky builder and expects `SplitLeakError`; `test_non_train_rows_use_only_full_train_model` pins CAL to the full-TRAIN model. |
| C3 | stacking fit only on CAL with disjoint-world cross-fit | PASS | `cross_fit_pool` calls `guard_cal_only(splits)` before fitting and folds by world (`world_folds`); `test_cross_fit_uses_disjoint_worlds` asserts `test_worlds ∩ train_worlds == ∅`; `test_guard_raises_on_test_rows` rejects TEST/HOLDOUT/TRAIN. |
| C4 | `select_variant` matches registration wording | PASS | `VARIANTS=("T","C","N")`, `COMPLEXITY_RANK T<C<N`, `tie_rel=0.02`, `taus=(60,120,300)`, `cov_bounds=(0.75,0.85)`, gating contrasts `damping-native/pi-native/wall-native`. See O3. |
| C5 | no real dataset / sumo / training | PASS | synthetic-only CLI; README states read-only reuse. |

### Literature note

| # | item | verdict | evidence |
| --- | --- | --- | --- |
| L1 | unverified citations marked `(to verify)` | PASS | 89 `(to verify)` markings; only 3 `verified-in-repo` entries, each confirmed in-repo: IDM arXiv:cond-mat/0002177 (`archive/pre_circles/docs/TRAINED_DAMPING_METHODS_DRAFT_2026-09-05.md:83`), Stern arXiv:1705.01693 (`.../2026-08-06-reference-list.md:27`), Vinitsky 2018 (`...:75-76`). The note explicitly refuses to assert DOI/arXiv/page for the rest. |

## Defects fixed (`d888be2`)

1. **F5 chunk-5 gate bound the wrong registration.** `splits.registration_sha()`
   raw-hashed `REGISTRATION_WM_N1.md`; the chunk-5 launcher gate must bind the
   chunk-5 registration's LF-sha256 `f8c7d760…4b31e` (recorded in
   `C5TEST_FREEZE.json`). Now `CHUNK5_REGISTRATION_LF_SHA256` is the gate
   constant, hashing is LF-normalised (CRLF-safe), and a non-matching file is
   refused. A same-commit fix would be impossible to hide drift: the frozen
   value is also recomputed from the file.
2. **F6 tensor dtypes disagreed with FORMAT/`expected_bytes`.** `counter` was
   float32 (doc/estimate: float16) and `slot_index` int64 (doc/estimate:
   int32). Cast in `featurizer.parse_run`; added a dtype-contract test; refreshed
   the FORMAT S10 measured shard sizes (13.300 MB / 7.388 MB).
3. **Pre-existing lint** in the reviewed modules: unused `n` in
   `_nearest_present_tick`, unused `MERGE_ACCEL`/`numpy` imports in tests.
   `ruff check` on the three packages is now clean.

## Observations (not fixed; not acceptance failures)

- **O1 (inventory).** `read_run` also reads `saved_state_id` from
  `RUN_RECEIPT.json`, used only for the `same_base_state` pair flag. It is run
  identity, not an outcome, and no value is used for labelling/fitting; but it
  is strictly a third field beyond the stated status/rows permission.
- **O2 (inventory).** `_read_receipt` `json.loads` the entire receipt, so a
  receipt's `evaluate_summary` bytes are parsed into memory. No
  `evaluate_summary` key/value is read, stored, returned or compared, so no
  outcome leaks; only `suffix_summary.rows_written` is consumed.
- **O3 (combine).** `select_variant.DEFAULT_CONTRASTS` includes `wall-native`
  unconditionally and raises if absent. Registration says "wall-native (if
  available)". The function takes `contrasts=` so a caller can drop it, but the
  default is stricter than the registration for worlds where walls are
  `not_feasible`.

## Executor-summary overclaims

1. `features/FORMAT.md` S8 documented the chunk-5 gate as the sha256 of
   `REGISTRATION_WM_N1.md` (F5) — both doc and code named the wrong
   registration. Corrected.
2. `features/FORMAT.md` S10 quoted shard byte counts produced by the
   float32/int64 code (F6); stale after the dtype correction. Refreshed.
3. No inventory/combine/lit overclaim found: the inventory's opened-path
   "NO HOLDOUT / NO dispatcher documents" claim was independently reproduced
   (I4/I5), the combine README's leakage claims match the tests (C2/C3), and
   the lit note is deliberately conservative (L1).

## External-file note

The acceptance check names
`E:/research-dense/protocols/holdout_exclusions.json` and the chunk plans.
Per the workspace-only execution policy no file outside the workspace was
read. `inventory/holdout_names.json` is a byte-faithful dump of the guard's
constructed token set (`excluded` from the exclusions file, plus every
HOLDOUT world name read from the plans), and `opened_paths.json` was checked
against it; the same guard ran `assert_clean` at build time. If the owner
wants an out-of-tree re-check, it is one read of that JSON.

## Tests run (real stdout)

```
$env:WD_FORBID_D="1"; C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest <dir> -q
output/wd01/wm_n1/inventory/tests   21 passed in 2.36s
output/wd01/wm_n1/features/tests    34 passed in 12.29s   (32 before the review fix)
output/wd01/wm_n1/combine/tests     27 passed in 77.62s
output/wd01/tests                   26 passed in 7.05s
output/wd01/partA_v2g/tests         79 passed in 44.93s
output/wd01/partB_v0/tests          142 passed in 1322.86s (0:22:02)
```

Pre-existing failures: none. All six suites green; no test weakened or
skipped.

`ruff check output/wd01/wm_n1/{inventory,features,combine}` -> `All checks
passed!` (after fix).

### TRAIN micro-smoke (<=2 runs, tmp data root, zero SUMO)

```
python -m features featurize --chunk chunk4 --worlds w8623m2600p02t540 \
    --arms base,native --data-root <tmp> --workers 1
featurize: runs=2 written=2 skipped=0 wall=24.1s
  [written] chunk4/w8623m2600p02t540__base n_ticks=2281 bytes=13300267 (13.300 MB) pair_key=w8623m2600p02t540@t0=540
  [written] chunk4/w8623m2600p02t540__native n_ticks=1201 bytes=7388218 (7.388 MB) pair_key=w8623m2600p02t540@t0=540

python -m features verify --data-root <tmp>
{"ok": true, "n_runs": 2, "errors": []}
```

## git log -6 (after fix)

```
d888be2 review fix (features): chunk-5 gate binds REGISTRATION_CHUNK5_TEST.md LF-sha (CRLF-safe); align counter/slot_index dtypes to FORMAT; drop unused lint
cadc954 Merge zcode/wmn1lit: WM-N1 literature note
b4201e7 Merge zcode/wmn1oof: WM-N1 combination layer
74cc9c8 Merge zcode/wmn1feat: WM-N1 featurizer
1cb02e3 Merge zcode/wmn1inv: WM-N1 corpus inventory tool
d20120f wd01 WM-N1 features: deployment-honest featurizer + memmap writer, split/open guards, tests (chunk-4 TRAIN micro-smoke, zero SUMO)
```

## Artifacts

- `output/wd01/wm_n1/REVIEW_WMN1_PREP.md` (this file)
- review fix commit `d888be2`
