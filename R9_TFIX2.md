# Round-9 integration test fixes

## Result

**PARTIAL — the ALL-tests-pass launch condition is not met.**

Baseline: `FULL_SUITE_BASELINE.txt:1883`: **17 failed, 13 errors, 3446 passed,
213 skipped**, 731.40 s. Baseline revision: `e473357b`; branch: `c4/v10r9tfix2`.

All **14 non-ledger baseline failing cases pass** in the final focused run.
That run also checks five additional regressions: **19 passed, 0 failed, 0 skipped**.
The three baseline ledger failures remain environmental blockers; they have not
been waived or marked skipped. Thirteen real-data fixture cases use their existing
missing-data skips in this worktree. Their real-data assertions and golden hashes
are unchanged and have not been claimed verified.

The full-suite execution totals are **unavailable**, rather than extrapolated
from subsets. A completed access-guarded full-suite attempt stopped in collection:
**45 collection errors, 2 existing skips, no test bodies executed**, exit 2,
59.61 s. Continued full-suite attempts exceeded the command time limits, as
described below. This is not a full-suite pass or an inherited-failure waiver.

## Per-baseline failure: cause, change, before/after

1. `tests/equivalence/test_components_equivalence.py::test_entry_v2_and_history_builders_equal`
   - Cause: `src/wms/model/entries.py:1178` adds the explicit default
     `learned_entry=False` summary field; the legacy summary predates it.
   - Fix: `tests/equivalence/test_components_equivalence.py:83` checks that field
     explicitly and compares every legacy summary field, mask and state tensor.
   - Before: equal masks, unequal key sets. After: **passed**.

2. `tests/equivalence/test_train_equivalence.py::test_fit_matches_legacy_fit`
   - Cause: `src/wms/train/trainer.py:292-294` adds `grad_norm`, `grad_clipped`
     and `lr`; legacy logs lack those fields. Model checksums already matched.
   - Fix: `tests/equivalence/test_train_equivalence.py:134-175` independently
     captures actual legacy pre-clipping norms/LR and compares the complete new
     logs against that reference, retaining parameter and loss-share checks.
   - Before: key-set mismatch. After: **passed**, including exact model checksum.

3. `tests/test_archive_boundary.py::test_path_map_entries_exist_at_new_location`
   - Cause: `archive/circles_era/PATH_MAP.json:12` records migration byte
     identities with a mixture of LF, CRLF and mixed-newline files. With
     `core.autocrlf=true`, the first file was 22157 bytes instead of 21816.
     Simply forcing the entire archive to LF also breaks CRLF-recorded entries.
   - Fix: `.gitattributes:39` pins archive LF by default and enumerates the
     38 SHA-confirmed CRLF exceptions. Existing Git blobs were re-materialized
     using those attributes; source text was not changed. Three mixed-newline
     entries use the map's existing `current_bytes/current_sha256` convention
     to pin their existing canonical LF Git blobs. Original migration identities
     are retained. The assertion now additionally verifies every file's SHA256.
   - Mixed-file details: `test_formation.py` reproduces its original hash with
     only the last newline CRLF; `test_target_adjust.py` reproduces its original
     hash with the first 103 newlines CRLF. `formation.py` is pinned to its
     unchanged Git source from `766a6cb7`; its older mixed-byte identity was
     **not** independently reconstructed or represented as re-verified.
   - Before: byte-size failure. After: **passed**, including the added hashes.

4. `tests/test_circles_controller.py::test_model_and_license_identity`
   - Cause: `baselines/circles/LICENSE` was checked out CRLF. Its raw SHA was
     `fe9b328474bde26b1a697cbd0dd306088c8ad1aa442926e8d0ece4b992e30795`;
     LF bytes reproduce the original provenance hash
     `964d95563c9b1dd57fcaf45c11c45705889199bfd780fd24053d90b699a24fd1`.
   - Fix: pin that license to LF in `.gitattributes`; retain the original
     provenance/model hash assertions. Before: license mismatch. After: **passed**.

5. `tests/test_experiment_history_ledger.py::test_all_referenced_paths_exist`
6. `tests/test_experiment_history_ledger.py::test_validator_reports_no_hard_errors`
   - Cause: `tools/experiment_history.py:465-512` previously compared absolute
     canonical-checkout offload receipts with this worktree's absolute paths.
     Additionally, many referenced artifacts are genuinely unavailable here.
   - Fix: normalize metadata identities across repo-relative paths, the current
     checkout, historical D: and canonical E: roots, including Windows receipts
     inspected on other platforms. This does not redirect reads or invent
     deletion/offload records. Added negative sibling/unregistered-path checks.
   - Before: **56** missing-path hard errors. After: **47**; the nine correctly
     recognized offloads are OE-32's three referenced records, OE-57's review
     directory and WM-12's five directories. Both original tests remain **failed**
     until the remaining genuine evidence gaps are supplied (listed below).

7. `tests/test_experiment_history_ledger.py::test_source_verification_clean_on_real_ledger`
   - Cause: `tools/experiment_history.py:793-806` correctly rejects missing
     registered sources. The three identity-only recovery receipts registered at
     `protocols/experiment_history.json:46385-46408` are absent, and no deletion
     record covers them. These are not MIG-04 deletions.
   - Fix/outcome: no fabricated receipts or relaxed checks. The canonical
     environment must supply the three exact registered sources below. This
     test is **blocked/failed**, not waived. Under the stricter workspace-only
     access guard it also cannot read the first junction-backed source,
     `output/escalation_execution/natural_final_decision.json`.

8. `tests/unit/data/test_drive_registry.py::test_committed_registry_matches_the_generator_output`
   - Cause: `tools/data_registry.json:3` retained the description count
     `dataops:personal_document=2`, although those receipts are absent from the
     committed receipt set. Offline regeneration differed only in that text.
   - Fix: synchronize the generated description with committed receipts;
     all 688 dataset entries and manifests are unchanged.
   - Before: exact-render mismatch. After: **passed**; entire module **9 passed**.

9. `tests/unit/eval/test_run.py::test_a_model_compared_with_itself_is_inconclusive`
   - Cause: fixture recordings keep vehicles beyond the road end, but the
     predictor correctly supplies the exit gate (`src/wms/eval/predictor.py:186-190`).
     Default-physics predictions then have no paired HDV values at 60 s.
   - Fix: `tests/unit/eval/test_run.py:37-48` stores explicit replay-fixture
     physics (`exit_at_road_end=False`) next to the synthetic checkpoints.
     The original zero assertion remains; pair/world counts and the point/upper
     bounds are also checked. Production exit behavior remains covered.
   - Before: lower bound `None`. After: **passed**, zero from four paired worlds.

10. `tests/unit/test_c4_v6imag.py::test_override_checkpoint_exact_resume`
11. `tests/unit/test_c4_v6imag.py::test_cli_v6_smoke_settings_and_resume`
   - Cause: the tests used ordinary external pytest temp paths, contrary to the
     deliberate guard at `src/wms/control/start_states.py:30`.
   - Fix: `tests/unit/test_c4_v6imag.py:22` provides an auto-cleaned,
     workspace-local temporary checkpoint directory. All resume, RNG, thread,
     credit-profile and checksum checks remain in place.
   - Before: workspace-path refusal / CLI exit 1. After: **both passed**.

12. `tests/unit/test_junctionfix.py::test_v4_config_differs_from_v3_only_by_the_switch_and_out_dir`
   - Cause: `configs/c4_wm_v4.toml:22-39` is now the expanded production recipe
     (GENRAND sources, paired loss, lane/entry/scheduling changes), not the
     original v3-plus-junction-switch recipe checked by this historical test.
   - Fix: `tests/fixtures/c4_historical.py:9` reconstructs the original one-switch
     fixture from v3; both junction and hygiene golden-window fixtures use it.
     Exact difference/equality assertions are retained. Production v4 is unchanged.
   - Before: many additional config differences. After: **passed**.

13-15. `tests/unit/test_v6longpair.py::test_fit_propagates_stage_rebuilds_loader_and_bypasses_short_draw`
   (`plain-loader`, `weighted-loader`, `persistent-worker`)
   - Cause: wrapper at `tests/unit/test_v6longpair.py:403` predated the
     keyword-only `batch_size` supplied by `src/wms/train/trainer.py:616-621`.
   - Fix: forward the optional keyword and assert the effective batch size.
     Stage eligibility, shapes, short-draw refusal, worker shutdown and loader
     rebuilding checks remain intact.
   - Before: **3 TypeErrors**. After: **all 3 passed**, including Windows spawn.

16. `tests/unit/test_wm_fidelity_suite.py::test_road_end_code_path_reports_no_removal`
   - Cause: the historical no-exit-bound test inspected today's corrected
     predictor instead of a missing-bound source fixture
     (`scripts/c4/wm_fidelity_suite.py:429-460`).
   - Fix: test the missing-bound branch on a synthetic source tree, retaining
     every assertion, and separately assert today's removal-capable path.
   - Before: expected False, got True. After: **both branch tests passed**.

17. `tests/test_merge_pivot_runtime.py::test_draft_launcher_uses_round8_stage300_and_preserves_frozen_recipe`
   - Cause: `tools/launch_round9.py:19-20` intentionally refuses initialization
     outside this worktree; pytest's default temp path was external.
   - Fix: `tests/test_merge_pivot_runtime.py:275-312` uses an auto-cleaned local
     **synthetic** stage300 checkpoint. Wrong-stage checks now also assert the
     exact stage-validation error. All first-passage and frozen-normalizer
     assertions remain. No real checkpoint was copied or generated by training.
   - Before: launcher exit 2. After: **passed**.

## Thirteen baseline setup errors: recorded datasets

Root causes: `tests/unit/test_datahyg.py:23-25,257-270` and
`tests/unit/test_junctionfix.py:24-25,184-197`. They implicitly probed another
checkout's cache, then attempted to assemble today's expanded v4, eventually
failing on `WMS_DATA_GENRAND_PLAIN` (`src/wms/cli/assemble.py:359`,
`src/wms/data/registry.py:113`). That is the wrong recipe for their fixed goldens.

Fix: use the historical fixture and default discovery to this workspace, with
explicit `WMS_DATA_PACK_V3`, `WMS_DATAHYG_DATA_ROOT` and
`WMS_JUNCTIONFIX_DATA_ROOT` support. The **existing** `if not HAS_DATA:
pytest.skip("C4 TRAIN data cache not present")` convention remains. No new
skip/xfail markers or missing-file exception suppression were added.

The absent-data cases are exactly:

- DATAHYG: `test_off_is_bit_identical_to_a2d3716[c4_wm_v3]` and `[c4_wm_v4]`;
  `test_strict_disables_the_damping_saturated_direct_mix_source`;
  `test_strict_dspeed_windows_avoid_designed_segments_and_relabel` — **4**.
- JUNCTIONFIX: `test_off_is_bit_identical_to_the_pre_fix_loader` at indices
  `0,7,300,582,583,1200,2040`; `test_ffill_batch_collates_with_the_same_keys_and_shapes`;
  `test_ffill_removes_the_artifact_and_keeps_recorded_changes` — **9**.

Canonical execution needs the original recorded v3 TRAIN/validation cache,
including logical registry roots exposed by `src/wms/cli/assemble.py:65-96`:
`WMS_DATA_PACK_V3`, `WMS_DATA_PACK_DSPEED`, `WMS_DATA_SIDECARS_PACKV3_V2`
(outcomes and identities), `WMS_DATA_IDENTITIES_DSPEED`, `WMS_DATA_OVERLAY_DSPEED`,
`WMS_DATA_C4_WM_LISTS`, `WMS_DATA_LPWO_V3`, `WMS_DATA_PACK_SWITCHING_TRAIN`,
`WMS_DATA_IDENTITIES_SWITCHING_TRAIN`, `WMS_DATA_DRIVER_PARAMS_SWITCHING_TRAIN`
and `WMS_DATA_ACTION_TIMELINE_INDEX_TRAIN`. The historical released-family recipe
does not need the restricted family-guard file or GENRAND. Recorded parameter
tables were not opened in this session. Real-data verification requires an
environment/policy authorized to use all of its recorded inputs.

Before: **13 setup errors**. After here: **13 existing missing-data skips**,
not 13 proven real-data passes. Synthetic checks still execute.

## Remaining ledger evidence requirements (metadata-only diagnosis)

The **47** remaining missing references are not covered by the available
deletion/offload protocols. The canonical environment must restore the actual
artifacts or provide an authoritative exposed relocation record. They cannot be
declared deleted merely because the test would otherwise fail.

- CV-02, under `output/controller_verification_v2/` — 11 references:
  `REGISTRATION.md`, `ADAPTER_PLAN.md`, `PI_SOURCE.md`, `REVIEW.md`,
  `sources/manifest.json`, `FROZEN_BASELINE_HASHES.json`, `REPORT.md`,
  `VERIFICATION.md`, `IMPLEMENTATION_HASHES.json`, `ADAPTER_IMPLEMENTATION.md`,
  `test_pi_saturation_v1.py`.
- CV-04, under `output/controller_verification_v4/` — 14:
  `NOVELTY.md`, `INTEGRATION_INVENTORY.md`, `SOURCE_PRESERVATION.json`,
  `ASTRA_PLAN.md`, `REGISTRATION.json`, `independent_review.py`,
  `REFERENCE_METADATA.json`, `REPORT.md`, `VERIFICATION.md`, `REVIEW.md`,
  `ROOT_PHYSICAL_REVIEW.json`, `IMPLEMENTATION_HASHES.json`,
  `ADAPTER_IMPLEMENTATION.md`, `test_pi_external_u.py`.
- WM-03, `output/worldmodel_dataset_v1/` and its `MANIFEST.json`, `MANIFEST.md`,
  `README.md`, `build_all.py`, `build_dataset.py`, `verification_report.json`,
  `verify_dataset.py` — 8.
- WM-32: `D:/rd-wt/wm32/output/recon_v5/REPORT.md`;
  WM-36: `D:/rd-wt/wm36/output/learned_recon_design_v1/results_20260925T1430Z_wm36_inventory/`;
  WM-37: `D:/rd-wt/wm37/output/wm12_repro_v1/REPORT.md` — 3.
- WM-38/39/40/43 — 11 references, including repeated citations to
  `output/action_model_v1/REPORT.md`, plus `fixture/apply_patch_r7.py`,
  `fixture/wm43_dispatcher_patch_r6.diff`, `fixture/wm43_scenarios_patch_r7.diff`,
  `fixture/FIXTURE_REPORT.md`, `fixture/REGISTRATION.md` below that directory.

Missing registered source receipts, under `tmp/zcode_review/`:

- `WMN1_RECOVERY_STOP_RECEIPT_20261002.json`: SHA256
  `b77088ad390e82242a2643a1d69025aedf273f1c4909f7d59402d4c1eb08a86e`.
- `WMN1_ZERO_ROW_CORRECTION_20261002.json`: SHA256
  `e90335a220727197f26c99690e11be6e8d523ea0819d1e86dacdfaf463a2fb88`.
- `WMN1_NUMPY_WRITER_FAILURE_20261002.json`: SHA256
  `b02e64dba0601f6901136c26bab3912f0e81723f788bc75c7359c980ba04a521`.

`protocols/deleted_evidence.json` does cover the historical missing runs and
selected output directories. Its existing treatment is retained. It does **not**
authorize fabricating any of the above. Restricted filenames were reasoned about
only from ledger/protocol text; restricted files were not opened or recreated.

## Verification commands and full-suite access blockers

Interpreter throughout: `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`.
The local diagnostic runner `_out/tfix2_verify.py` installs a file-open audit
boundary, uses this worktree's `_out` for TMP/TEMP, disables bytecode writing,
and calls pytest without changing assertions or outcomes. Python's installed
runtime dependencies are allowed; research files outside the worktree and all
restricted paths are refused. Local runner/log files remain ignored diagnostics.

Exact completed full-suite invocation (same pytest arguments as the requested
command, with workspace-local temp storage):

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -B _out/tfix2_verify.py pytest -q -p no:cacheprovider tests -rf --basetemp=_out/tfix2-full --tfix2-log=_out/tfix2-full-collection.txt
```

Result: **45 collection errors, 2 skipped**, exit 2, no bodies run. The precise
test-to-refused-source mapping is in `_out/tfix2-full-collection.txt`.

- Three collection paths themselves contain a forbidden substring:
  `tests/planner/test_holdout_guard.py`,
  `tests/planner/test_holdout_guard_seal.py`, `tests/test_holdout_exclusions.py`.
  They were refused before opening. Running them needs an explicit policy
  exception for these test-code filenames, not permission to read sealed data.
- The other 42 errors are code imports resolving through external junctions
  or an absolute external source. They require workspace-local copies of the
  exposed historical modules: `output/planner_v1/{box_ingest,noise_study,resmoke,closed_loop_smoke}`,
  `output/pl02a_offline_v1`, controller-verification v8/v9/v10/v12/v13/v15,
  and registration modules for v17-v25/v27/v28. `test_fusion_warning.py` also
  imports an absolute external `wall_harm_speed_v1/fusion_warning_v5.py`.
  No junction was removed/replaced and no `output/` file was written.

Continuation command (not a pass claim): add `--continue-on-collection-errors`,
`--basetemp=_out/tfix2-full-continued`,
`--tfix2-log=_out/tfix2-full-continued.txt` to the same runner invocation.
It timed out after **1200 s**. A second continuation with
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`--basetemp=_out/tfix2-full-final`, `--tfix2-log=_out/tfix2-full-final.txt`
timed out after **1800 s**. Neither produced terminal suite counts.
The literal requested command
`C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q -p no:cacheprovider tests -rf`
was also attempted with the guard loaded through local `sitecustomize`,
workspace-local `PYTEST_ADDOPTS=--tb=line --basetemp=_out/tfix2-canonical-attempt`;
that attempt timed out at **120 s**, without final counts.

Completed verification (counts overlap; they must not be added):

- Final focused baseline cases plus additional regressions: **19 passed**,
  277.81 s, `_out/tfix2-final-targets.txt`. Exact pytest node list follows.
- Runtime/eval/exit/fidelity modules:
  `tests/unit/eval/test_run.py tests/unit/test_wm_fidelity_suite.py tests/test_merge_pivot_runtime.py tests/unit/test_exit_road_end.py tests/unit/eval/test_eval_units.py`
  — **51 passed**.
- Archive + controller modules: **14 passed**. Drive-registry module: **9 passed**.
- DATAHYG + JUNCTIONFIX + V6 imagination modules: **46 passed, 13 skipped**.
- Final receipt/metadata/doc follow-up:
  `tests/test_archive_boundary.py tests/unit/test_datahyg.py tests/unit/test_junctionfix.py`
  plus the four ledger metadata nodes below — **26 passed, 13 skipped**.
- Full ledger module under the access boundary: **22 passed, 5 existing skips,
  4 failed**. The three original failures remain; `test_checksum_change_is_detected`
  additionally cannot read the first external junction-backed registered source.
- `git diff --check`: passed.

Final focused command (set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`):

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -B _out/tfix2_verify.py pytest -q -p no:cacheprovider --basetemp=_out/tfix2-final-targets tests/equivalence/test_components_equivalence.py::test_entry_v2_and_history_builders_equal tests/equivalence/test_train_equivalence.py::test_fit_matches_legacy_fit tests/test_archive_boundary.py::test_path_map_entries_exist_at_new_location tests/test_circles_controller.py::test_model_and_license_identity tests/unit/data/test_drive_registry.py::test_committed_registry_matches_the_generator_output tests/unit/eval/test_run.py::test_a_model_compared_with_itself_is_inconclusive tests/unit/test_c4_v6imag.py::test_override_checkpoint_exact_resume tests/unit/test_c4_v6imag.py::test_cli_v6_smoke_settings_and_resume tests/unit/test_junctionfix.py::test_v4_config_differs_from_v3_only_by_the_switch_and_out_dir tests/unit/test_v6longpair.py::test_fit_propagates_stage_rebuilds_loader_and_bypasses_short_draw tests/unit/test_wm_fidelity_suite.py::test_road_end_code_path_reports_no_removal tests/unit/test_wm_fidelity_suite.py::test_road_end_code_path_reports_current_removal tests/test_merge_pivot_runtime.py::test_draft_launcher_uses_round8_stage300_and_preserves_frozen_recipe tests/test_experiment_history_ledger.py::test_offloaded_repository_paths_are_worktree_independent tests/test_experiment_history_ledger.py::test_offloaded_paths_resolve_to_their_drive_copy tests/test_experiment_history_ledger.py::test_offload_map_is_well_formed tests/test_experiment_history_ledger.py::test_relocated_maps_paths_inside_a_moved_directory -rf --tfix2-log=_out/tfix2-final-targets.txt
```

One broader regression selection also timed out at 600 s; its partial progress
is `_out/tfix2-regressions.txt`, not a claimed passing run. The persistent-worker
case was then verified independently and in the completed final focused run.

## Commits and next action

- `94ab3c0a4553851e9a7fcfcd48d7998bea790da0`: entry/trainer equivalence and loader instrumentation.
- `f2bc03b38c5d7d9c55e3c999d8bcbbb25cacdca6`: synthetic eval/fidelity/round9 fixtures.
- `bfb2da31`: historical recipes, explicit local data discovery, V6 resume fixtures.
- `8e41cafc`: deterministic registry description.
- `cc38684d`: archive/license byte identities and archive SHA checks.
- `76c0ff65`: worktree-independent offload metadata identities.
- `be09153a`: portable receipt path normalization and fixture documentation.

Next action: provide an authorized canonical test environment with the exposed
historical code and registered evidence physically available within the allowed
workspace, the original recorded data cache for the 13 golden checks, and the
narrow test-filename policy exception above; then rerun the exact full-suite
command to completion. No default model/training behavior or round-9
`first_passage` implementation was changed. `FULL_SUITE_BASELINE.txt` remains
the user's untracked baseline input.
