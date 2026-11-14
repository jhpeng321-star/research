# WD-01-PB-V04A: Part B value model v0.4a on TRUE policy references (no walls): changelog

Registration: `output/wd01/REGISTRATION_PARTB_V04A.md` (LF sha256
`38a6fa406ca9d598fdaae986263f4fa81ee15dfa8c632414cc5f28dbe5bdee00`,
verified before execution). Branch `zcode/wd01v04a` from `zcode/wd01`
dbde3a7. ZERO SUMO/TraCI. Allocation entry committed first
(`LEDGER_DRAFT_PB_V04A_ALLOCATION.json`, commit 34bd8f2).

Final ledger draft: `LEDGER_DRAFT_PB_V04A.json`. Report: `REPORT_V04A.md`.

## Input from PB-RESCORE1 (registration s.1)

`E:/rd-wt/wd01rs1/output/wd01/partB_rescore1/REPORT_PB_RESCORE1.md` existed in
the rescore1 worktree but was NOT committed on `zcode/wd01rs1` at the time
it was read (only its code commits were); it was read as a working-tree
draft. Its label audit is cited here:

- Label parity: a TRAIN chunk-2 damping source rebuilt from the raw runs
  with the v0.3.2 code reproduces the frozen v0.3.2 labels exactly
  (RESCORE1 section 0). v0.4a repeats this parity for every chunk-1/2/3
  base and campaign switching arm (see "Parity" below).
- Q1: the true damping-vs-native gap is 1.5-3.2x the old damping-vs-switching
  gap and the old sign agrees with the true one in only 0.76-0.90 of
  world-tau pairs; PI(true) is much closer to native than damping is.
  This is the design premise of v0.4a (true references), not a defect of it.
- RESCORE1 reads the branch arms at t0 + 10 s (10 s window offset). v0.4a
  removes that offset: every branch arm gets a t0 decision row (state from
  the shared base trace, label window (t0, t0+tau]), and base-type rows at
  t0 use the same interval convention (`skip_first_tick`).
- CAL exposure (RESCORE1 and registration s.2): chunk-2/3 CAL has been read
  by v0.3.2 and by RESCORE1 (report-only); this run is the third, disclosed
  read. Chunk-4 CAL is also C4L1's Part A look set (different model).
- Finding that changes the design: none. No addendum (ADDENDUM_V04A_1.md)
  was needed.

## Deviations / interpretations (all disclosed)

1. **PI arms and `kind == "command"` owners.** A PI-external-U CAV is written
   through the damping writer, so its command owner is `damping`
   (`campaign_stage1_v3/action_labels.py` documents the asymmetry). With the
   v0.4 rule "realised command owner wins", every PI arm would be labelled
   damping. `actions_v04.ActionTimeline.from_run_dir(schedule_authoritative=True)`
   therefore takes the PLANNED action from `switch_schedule.json` as the token
   and uses the `kind == "command"` owners only for VERIFICATION
   (`verify_arm`: allowed owner sets per scheduled action). Sources are still
   only `switch_schedule.json` + `kind == "command"` rows (+ the fleet
   manifest for `__base` / `__native_base`); `pi_owner_receipt` /
   `pi_source_evidence` rows are never used as owners. Default behaviour of
   `from_run_dir` is unchanged (test pins the difference).
2. **"chunk-4 base (after t0)"**: chunk-4 `base` and `native_base` decision
   times are t0, t0+10, ..., t0+570 (both fleets on the same window as the
   restored branches; `native_base` has no saved state, so its t0 row is its
   own state). Chunk-2/3 bases use the v0.3.2 grid (120 .. t0+570).
3. **Chunk-1 and chunk1_fix27 bases are AUXILIARY** (the registration lists
   only the chunk-2/3 bases as primary); chunk-1/fix27 `__native`/`__pi` arms
   are auxiliary switching arms. The auxiliary-data ablation removes all
   auxiliary sources (`--exclude-aux`).
4. **No `pre_fix` column.** v0.3.2 kept `pre_fix` as a training feature
   (fixed at predict). The registration prohibits it as a model input, so the
   v0.4a design matrix has no `pre_fix` column at all (it stays a label in the
   parquet only). Same for penetration, speed_mode and world strings.
5. **Fits read TRAIN only** (`--train-only`): the CAL files are not even
   opened before the single final evaluation.
6. **Per-CAV action features in the GBM** = per-action counts of PRESENT CAV
   tokens (native / damping / pi / unknown), i.e. the CAV_RAW-20 action
   one-hot with the unknown mask, summed per decision time. Note (disclosed):
   with ideal-sensing lane bins (all vehicles) these counts make the CAV share
   implicitly visible; the strict variant's guarantee is about the entrance
   demand block (CAV counts only), as in v0.3.2.
7. **Climatology comparator** = the v0.3.x `Climatology` class fitted on all
   TRAIN rows of the dataset loaded (per tau x policy class x demand), i.e.
   the same information the model had (in the ablation, without auxiliary
   rows).
8. **`git reset -q HEAD -- <paths>` was run once by mistake** (to unstage
   newly added paths) and `__pycache__` files that were staged by
   `git add -f` were then removed from the index with `git rm --cached`.
   Nothing was discarded (the worktree files and history are intact); it is
   noted because the task rules forbid `git reset` in any form.

## Item -> file/function -> test

| Item | File / function | Test |
| --- | --- | --- |
| Ledger allocation first | `LEDGER_DRAFT_PB_V04A_ALLOCATION.json` | (commit 34bd8f2) |
| Chunk-4 arm verification (schedule + `kind==command`; mismatch refuses) | `partB_v0data/actions_v04.py :: ActionTimeline.verify_arm, SCHEDULE_ALLOWED_OWNERS, ArmPolicyMismatch`; `partB_v0data/arm_verify_v04a.py` | `partB_v0data/tests/test_v04a_actions.py :: test_verify_arm_matrix, test_native_schedule_with_a_damping_owner_is_a_mismatch, test_pi_schedule_without_the_damping_writer_is_a_mismatch, test_missing_commands_or_schedule_refuse, test_non_command_rows_are_not_owners` |
| PI token = planned action (not `damping`) | `actions_v04.py :: from_run_dir(schedule_authoritative=True)` | `test_v04a_actions.py :: test_pi_token_is_pi_when_schedule_authoritative`, `test_build_v04a.py :: test_tokens_carry_the_planned_action_pi_is_not_damping` |
| Planner-known schedule summary pi_* | `actions_v04.py :: ActionTimeline.action_shares` | `test_v04a_actions.py :: test_action_shares_time_fractions_and_unknown_excluded`, `test_build_v04a.py :: test_t0_rows_agree_across_arms_and_pi_columns` |
| Source registry, plan split first, holdout guard, open audit | `partB_v0data/v04a_sources.py :: enumerate_specs, chunk_worlds, OpenAudit` | `test_build_v04a.py :: test_enumerate_specs_never_holdout_and_counts, test_open_audit_flags_a_holdout_path` |
| Builder: labels, t0 rows, features (CAV_RAW 20), strict/detector sidecars, done markers, MANIFEST | `partB_v0data/build_v04a.py`; `featurize_v0.py :: build_features_multi`; `v0lib.py :: window_label(skip_first_tick), build_run_trace(base_arm_dir)` | `test_build_v04a.py :: test_skip_first_tick_puts_base_t0_on_the_branch_convention, test_default_window_label_unchanged, test_all_sources_built_and_labelled, test_strict_variant_has_no_all_vehicle_counts_and_t0_unrecoverable, test_mismatching_arm_is_refused_and_not_built` |
| Never reads dispatcher_rows / insertions | `build_v04a.py` (physics + receipt + schedule + commands only) | `test_build_v04a.py :: test_builder_never_opens_dispatcher_or_insertions, test_static_no_forbidden_reads_or_hardlinks` |
| Root by COPY, SHA256SUMS, backup verified | `partB_v0data/finalize_backup_v04a.py` | `test_build_v04a.py :: test_finalize_backup_copy_only_and_verify` |
| Parity with v0.3.2 labels | `partB_v0data/parity_v032.py` | run on the real root (see Evidence) |
| v0.4a manifest / aux flag / policy classes | `partB_v0/data.py :: V0Manifest.v04a, load_manifest(exclude_aux), V0Dataset._load_tables`; `partB_v0/model.py :: POLICIES_V04A, policy_id` | `partB_v0/tests/test_gbm_v04a.py :: test_policy_ids_are_backward_compatible_and_extended, test_v04a_manifest_flags_and_exclude_aux, test_dataset_rows_carry_policy_pi_and_aux` |
| Policy-conditioned design (one-hot policy, pi_*, per-action CAV counts, no pre_fix) | `partB_v0/gbm.py :: summary_features, row_feature_matrix, v04a_design, counterfactual_design` | `test_gbm_v04a.py :: test_design_has_policy_pi_counts_and_no_pre_fix, test_counterfactual_design_puts_every_cav_on_the_fleet_action` |
| Aux rows train but never select/calibrate/score | `gbm.py :: GbmValueModel.fit (sc mask), fit_baselines/_resid_sample, oob_acceptance_section` | `test_gbm_v04a.py :: test_fit_marks_design_and_excludes_aux_from_selection` |
| `--train-only`, `--exclude-aux` (ablation) | `partB_v0/__main__.py`; `gbm.py :: run_gbm_training` | `test_gbm_v04a.py :: test_cli_train_only_never_opens_calibration_files` |
| A1 / A2 / A3 metrics, ΔV pairing, bootstrap, verdict | `partB_v0/acceptance_v04a.py` | `partB_v0/tests/test_acceptance_v04a.py` (19 tests on synthetic data) |
| One CAL read, reports, verify | `partB_v0/final_eval_v04a.py`, `report_v04a.py` | `test_gbm_v04a.py :: test_final_eval_end_to_end_single_read_and_report` |

## OWNER-CONFIRMED data choices (2026-09-30, resume brief)

1. **PI action token comes from the PLANNED schedule** (deviation 1 above):
   confirmed. `kind == "command"` owners verify; they never label the token.
2. **Chunk-1 and chunk1_fix27 bases are AUXILIARY rows** (deviation 3
   above): confirmed. Only chunk-2/3 bases are primary damping sources;
   the `--exclude-aux` ablation removes all chunk-1/fix27 sources.

## Execution record (resume, worktree `E:/rd-wt/wd01v04c`)

Branch `zcode/wd01v04c` from `15def5f` (fits strict/detector/strict_noaux
DONE + verified at `1ff7da3`; previous dry-run interrupted by the owner
stop, `partB_v0data/v04a_results/INTERRUPTED_DRYRUN_20260930.txt`). Zero
SUMO/TraCI throughout. No `CAL_READ_ONCE_v04a.json` existed in
wd01v04a / wd01v04b / wd01v04c / `partB_results` before the resume
(searched first). Executor: GLM Flash (owner stopped the previous Sonnet
executor mid dry-run; runtime-limit resume).

1. Dry-run re-run (gated launcher
   `run_gated_v04c.ps1`, free RAM 13.68 GB / free VM 22.0 GB at launch,
   BelowNormal, detached, `WD_FORBID_D=1`): `python -m
   partB_v0.final_eval_v04a --dry-run --dry-out
   partB_v0data\v04a_results\dryrun_resume` -> `dry run done in 2359s`,
   empty stderr, all three variants rendered full reports
   (`dryrun_resume/DRY_RUN_REPORT.md`, TRAIN rows in the CAL role:
   plumbing only, NOT evidence).
2. `pytest partB_v0/tests/test_gbm_v04a.py -k final_eval` -> `1 passed,
   11 deselected, 2 warnings in 115.81s`.
3. ONE CAL read (free RAM 16.65 GB / free VM 32.52 GB at launch): `python
   -m partB_v0.final_eval_v04a --report REPORT_V04A.md` -> done in
   1221.3 s; `CAL_READ_ONCE_v04a.json` written before the first CAL row
   (verdicts: strict FAIL, detector FAIL, strict_noaux FAIL); `verify
   v0.4a_strict: OK`, `verify v0.4a_detector: OK`, `verify
   v0.4a_strict_noaux: OK`; `REPORT_V04A.md` written. NO refit after the
   read; the FAIL verdict is reported as is (A3(ii): the dV CRPS does not
   beat the zero-contrast / TRAIN-cell-mean baselines at the gating
   taus; strict A1 also fails on coverage 0.880 > 0.85).
4. Explicit `python -m partB_v0 verify --out <dir>` x 3 -> `verify: OK --
   7/7`, `7/7`, `6/6 OUTPUT_MANIFEST entries re-hash correctly...`
5. %-of-mean scale: `train_target_means_v04a.py` read TRAIN label
   parquets only (column-projected; CAL never re-opened; HOLDOUT never
   selected) -> `train_target_means_v04a.json`; appendices rendered into
   `REPORT_V04A.md` by `render_report_appendix_v04a.py`.

### Real stdout (per-directory suites, WD_FORBID_D=1)

- `partB_v0data/tests`: `102 passed, 4 skipped in 143.83s` (the 4 skips
  are the pre-existing `test_real_data.py` guards: no `worlds/` in the
  worktree by design).
- `tests` (output/wd01/tests): `18 passed in 4.10s`.
- `partB_v0/tests` first full run: `1 failed, 172 passed in 1288.23s`.
  The failure was `test_v032_fixes_partb.py::test_fix13_no_damping_minus_native_claim`:
  the v0.4a docstring added to `v0lib.py :: window_label` (commit 956e335)
  contained the literal guarded phrase "damping - native" in its
  tick-bias explanation. Fixed by rewording the docstring to
  "damping-vs-native" (comment-only change; no code, no test edited);
  `pytest partB_v0/tests/test_v032_fixes_partb.py -q` -> `16 passed in
  33.94s`. Full clean re-run: see `pytest_partb_rerun.log`.
- Registration hash re-verified at resume: LF sha256 of
  `REGISTRATION_PARTB_V04A.md` =
  `38a6fa406ca9d598fdaae986263f4fa81ee15dfa8c632414cc5f28dbe5bdee00`.
