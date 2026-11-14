# CHANGES — Part B v0.4 action and wall inputs (code only, zero SUMO)

Task: implement `BRIEF_PARTB_WALL_ACTION_INPUTS_DRAFT.md` items 1–5 (per-CAV
action tokens + wall tokens) on `zcode/wd01act` (from zcode/wd01 6b1038b).
No simulation, no dataset rebuild, no training; `dispatcher_rows` is never
read by any reconstruction path; HOLDOUT/CAL untouched (all real-data tests
verify `split == "TRAIN"` from the chunk WORLD_PLAN **before** opening any
run file, and check `protocols/holdout_exclusions.json`).

## Item → file/function → test

| Brief item | File / function | Test(s) |
| --- | --- | --- |
| 1. Per-CAV action at t from switch_schedule.json + suffix_commands.jsonl owners (never dispatcher rows) | `partB_v0data/actions_v04.py` — `ActionTimeline.from_run_dir` / `action_at` / `action_ids` / `command_owner_counts` / `canonical_action` | `partB_v0data/tests/test_actions_v04.py::test_schedule_holds_between_events_and_commands_win_ties`, `::test_every_logged_command_is_reproduced_exactly`, `::test_unknown_owner_and_action_spellings_fail_loudly` |
| 1. Real chunk2 TRAIN source `w8481m2700p02t480__native`: damping 556 / native 487 command owners, every command + schedule entry reproduced | `actions_v04.ActionTimeline` on the real run dir (split checked first) | `test_actions_v04.py::test_chunk2_native_owner_counts` (556/487), `::test_chunk2_native_every_command_reproduced` (1043 commands), `::test_chunk2_native_schedule_events_reproduced`, `::test_chunk2_native_decision_grid_actions` |
| 1. chunk-4 TRAIN `__native_base` all native (fleet policy from run manifest, command-log verified) | `actions_v04` fleet-policy resolution + `verify_fleet_policy` | `test_actions_v04.py::test_chunk4_native_base_is_all_native`, `::test_chunk4_native_base_fleet_policy_on_fresh_timeline`, `::test_fleet_run_from_manifest_is_uniform_and_verified`, `::test_fleet_damping_base_tolerates_native_fallback_but_nothing_else` |
| 1. Unknown-action mask (never silently "native") | `actions_v04.ACTION_UNKNOWN`; featurizer default stamps the `unknown` one-hot slot | `test_actions_v04.py::test_no_evidence_at_all_is_unknown_not_native`, `test_featurize_v0.py::test_cav_token_layout_default_unknown`, `::test_cav_token_layout_with_actions_timeline` |
| 1. No dispatcher_rows / insertions reads anywhere | static + functional open guards extended to the new modules | `test_v032_fixes_data.py::test_fix2_no_insertions_jsonl_in_label_feature_code` + `::test_fix3_no_dispatcher_rows_read` (mod lists extended), `test_actions_v04.py::test_reconstruction_never_opens_dispatcher_or_insertions` |
| 2. Wall tokens from wall_events.jsonl + branch_record.json (wall_driver schema) | `partB_v0data/wall_tokens_v04.py` — `WallEpisodeSet.from_branch_dir` / `walls_at` / `member_flags` | `partB_v0data/tests/test_wall_tokens_v04.py::test_lifecycle_assemble_hold_release`, `::test_lifecycle_abort_in_assembly`, `::test_lifecycle_not_feasible_never_yields_a_wall`, `::test_wall_before_request_is_invisible` |
| 2. Token layout v0.4 (CAV_RAW 17→20, WALL_RAW 6→11; explicit wall fields replace the skeleton's member-s hack; bump in ONE place) | `part4/partB/featurize.py` (ACTIONS/PHASES/CAV_RAW/WALL_RAW + `featurize()` packing); `partB_v0data/featurize_v0.py` (`featurize_at(..., actions=None, walls=None)`, `_wall_token_row`) | `test_featurize_v0.py::test_feature_shapes_and_dtypes` (320/20/11/21/8), `test_wall_tokens_v04.py::test_featurize_at_wall_member_tokens`, `::test_featurize_at_wall_tokens_absent_without_pilot`, `part4/tests/test_partB.py::test_feature_shapes_and_dtypes` |
| 2. No future/outcome leakage (shuffle events after t ⇒ features at t unchanged; branch_record outcome fields never features) | `wall_tokens_v04._state_at` uses only events `timestamp_s <= t`; only `ALLOWED_RECORD_FIELDS` bound | `test_wall_tokens_v04.py::test_no_future_or_outcome_leakage_shuffle_after_t`, `::test_member_geometry_uses_member_cav_rows_only` (no HDV-derived wall fields; non-CAV member fails loudly) |
| 3. Deployability: strict variant carries the new planner/RSU-known fields unchanged; widths data-driven | `partB_v0/data.py::V03TensorData.model_dims` (+`wall_raw`); `partB_v0/train.py::MemberTrainer` sizes token widths from the dataset | `partB_v0/tests/test_action_wall_inputs_v04.py::test_strict_variant_carries_action_and_wall_blocks_unchanged`, `::test_model_dims_carry_both_token_widths` |
| 3./acceptance: config-driven token widths, loud failure on old-checkpoint width mismatch | `partB_v0/model.py` — `check_token_widths` + `_check_input_widths` (forward guard); wired into `evaluate.load_members` / `evaluate.load_net_v03` | `test_action_wall_inputs_v04.py::test_check_token_widths_fails_loud_on_old_checkpoint`, `::test_load_members_loud_on_sidecarless_old_weights`, `::test_load_members_honours_recorded_cfg_widths`, `::test_config_width_roundtrip_and_forward_guard` |
| 4. Fixtures: 3 synthetic wall lifecycles from the campaign_walls_v1 schema (assemble→hold→release; abort in assembly; not_feasible) | synthetic wall_events.jsonl / branch_record.json writers in `test_wall_tokens_v04.py` (exact wall_driver row shapes) | the three lifecycle tests above + `::test_absent_events_file_yields_empty_set` |
| 5. Builder flag `--with-actions` (default OFF; OFF outputs byte/array-identical; ON writes `<world>__<source>_feat_act.npz` + MANIFEST/SHA256SUMS entries; no-evidence arms fail loudly) | `partB_v0data/build_v032.py` — `--with-actions`, `build_action_sidecar`, `build_world(..., with_actions)` | `partB_v0data/tests/test_build_v032_actions.py::test_with_actions_off_is_identical_and_on_adds_sidecars`, `::test_action_sidecar_values_fleet_and_switching`, `::test_action_stage_fails_loud_without_evidence` |

## Token layouts (frozen slot order)

- CAV token, `CAV_RAW = 20` (`part4/partB/featurize.py`, the single bump
  point; `featurize_v0` and `partB_v0/model.py` import the constants):
  `s/1000`, lane one-hot(3), `v/40`, action one-hot(5:
  native/damping/pi/wall/**unknown**), phase one-hot(6:
  none/requested/assembling/holding/released/aborted), wall speed `/40`,
  time-in-phase `/600`, member flag, planned hold remaining `/600`.
- Wall token, `WALL_RAW = 11`: phase one-hot(5: requested/assembling/
  holding/released/aborted), speed `/40`, time-in-phase `/600`, planned
  hold remaining `/600` (**planner-owned state** — flagged; the planner
  knows its own decision), `n_members/4`, members' mean `s_down /1000`
  and lane span `/2` (member-CAV self-report rows at t only).
- Action vocabulary mirrors `actions_v04.ACTIONS_V04` (assert-equal at
  import so the two declarations cannot drift).

## Real-data evidence (read-only, WD_FORBID_D=1)

- chunk2 `w8481m2700p02t480__native` (TRAIN per
  `results_plan_chunk2/WORLD_PLAN.json`, not in holdout exclusions):
  command owners damping **556** / native **487**; all 1043 commands
  reproduced; decision-grid CAV tokens cover every present CAV from t0
  (unknown = 0); `cav_tokens (64, 20)`, `wall_tokens (4, 11)`.
- chunk4 `w8623m2600p02t540__native_base` (TRAIN per
  `results_plan_chunk4/WORLD_PLAN.json`): fleet policy `native` resolved
  from the run manifest, command owners all `native` (1569), verification
  ok, every sampled CAV token native.

## Out of scope (per brief item 5)

The v0.4 dataset rebuild (actions + chunk-4 sources) and any training
remain a separate registered task; nothing under the dataset roots was
written. `partB_v0` and `partB_v0data` suites, `output/wd01/tests` and
`part4/tests` are green with `WD_FORBID_D=1` (per-dir invocations; mixed
invocations of the new files also pass).
