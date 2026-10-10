# WALL2_AUDIT.md — campaign_walls_v1 vs PLAN_DRAFT (WALL-2 row)

- **Audited:** 2026-09-29, ZCode (WM-59 WALL-2, code only, ZERO SUMO).
- **Source:** untracked wall-pilot code copied from
  `E:/rd-wt/wm64/output/campaign_walls_v1/` (never modified there) into this
  worktree and tracked in git; provenance + per-file sha256 in
  [COPY_MANIFEST.json](COPY_MANIFEST.json).
- **Spec:** `E:/research-dense/output/campaign_walls_v1/PLAN_DRAFT.md`
  (row `WALL-2: plan + registration + smoke`, sections 1.2/1.3/1.4, splits,
  owner rules). Status legend: **ok** = implemented + tested; **gap-fixed** =
  was missing, added here with tests; **deviation** = differs from the draft
  text with a recorded rationale (no silent relaxation).

## 1. Spec-item table

| Spec item (PLAN_DRAFT) | File / function | Status | Fix / test |
|---|---|---|---|
| Plan builder: 60 worlds, 20 per p10/p20/p40 from chunk-2+chunk-3 mode-27 bases | `common.py` constants; `plan_walls_pilot.select_worlds` (PLAN_SEED 20261101) | ok | `test_plan_shape_60_worlds_240_branches_144k`; `test_selection_draws_from_both_chunks_and_is_seed_sensitive` |
| 49:7:14 splits → 42/6/12 (per-pen 14/2/4) | `common.SPLIT_COUNTS`, `SPLIT_COUNTS_PER_PEN`; `validate_plan` | ok | shape test asserts exact split counts; `validate_plan` refuses others |
| World selection by penetration **and feasibility pre-screen** | `select_worlds` filters p10/p20/p40 only | deviation | The draft's per-world base-row pre-screen is needed only for the optional p05 arm (1.4), which the owner DEFERRED; the pilot restricts to p10/p20/p40 where the measured zero-SUMO scan shows feasible-within-120 s = 14/14 (p10+), and the ONLINE request rule + `not_feasible` logging re-checks from reports (1.4 caveat). Documented, not a silent gap |
| Arms: 4 single-wall arms (0.60/0.85 × member mean speed × early/late) | `common.ARMS/ARM_SPEED_FACTORS/TIMING_OFFSET_S`; `wall_request.arm_speed_factor/arm_timing` | ok | `test_early_and_late_target_and_deadline`; every-world arms test |
| W-2 double-wall arm | plan constants: `multi_wall_W2: "deferred pending owner confirmation"` | deviation | PLAN_DRAFT lists W-2 as "small arm, full campaign only". No recorded owner decision was found (main discussion, 2026-09-29), so it is marked "deferred pending owner confirmation", not silently dropped. WALL-3 stays future work |
| Seeds + H draws {60,120,240} from branch seed, logged | `plan_walls_pilot.branch_seed/hold_draw`; `validate_plan` hold check | ok | `test_hold_draw_is_deterministic_and_in_the_registered_choices`; replay bit-exact test |
| Speed rule: factor × members' mean REPORTED speed, floor 2.5 m/s (> SAFE_SPEED_MIN_MPS 2.0) | `wall_request.wall_speed_for` | ok | `test_speed_factor_and_floor` (incl. floor case) |
| Timing early/late; 120 s request window; else `not_feasible`, continue native | `wall_request.target_and_deadline`; `wall_driver._WallRequest` | ok | `test_not_feasible_only_after_the_full_window`; `test_not_feasible_after_120s_continues_native` |
| Feasibility: ≥2 live CAVs, same 2-lane mainline edge (1191554636/1487606178), different lanes, span ≤ 75 m, from report frames only | `wall_request.feasible_pairs/choose_pair/edge_of_lane` | ok | same-edge/distinct-lane/span-cap/other-edge/no-coordinate tests; report-frames-only by construction (SensorFrame) |
| Member choice: feasible pair nearest upstream of the merge, one per lane | `choose_pair` (sort by mean distance_to_merge, ties by span then ids) | ok | `test_member_choice_nearest_the_merge`; lanes `[0,1]` asserted in driver tests |
| Reuse existing native sibling as reference (never re-run, outcomes never read) | `plan_walls_pilot.source_world_entries` (records run + runs_root only); registration pins ids; `check` requires COMPLETE receipts | ok | `test_check_requires_complete_native_sibling_receipts`; metadata-only reads asserted by construction |
| Pin base-state digests (state.xml + STATE_MANIFEST.json + saved_state_id) | `plan_walls_pilot.base_state_entry`; `walls_pilot_registration.worlds_record` | ok | digest-vs-fake-root test; `test_check_refuses_changed_or_missing_base_state` (tamper) |
| Registration draft/finalize/check, write-once final | `walls_pilot_registration.draft/finalize/check` | ok | round-trip, write-once refusal, changed-plan refusal (`test_check_refuses_a_changed_plan`), gate-file probe test |
| Registration pins native-sibling ids; check re-verifies live | `WORLD_PIN_FIELDS`; live re-hash of 60 states + receipt checks | ok | `test_base_state_digests_cover_all_sixty_worlds` + check tests |
| Smoke ≤ 80 sim-s: assemble one wall from a restored state; check event log (assembly-complete OR logged abort; members locked; non-members free) | `smoke_walls.run_smoke` via the REAL child; `evaluate_smoke_checks` (5 checks); write-once `SMOKE_VERDICT.json` bound to registration sha + package code digest | ok | green/abort/failure-mode/command-to-nonmember/two-loadState tests; verdict binding + launch-gate refusal tests |
| Smoke `--dry-run` (validate inputs; print the exact SUMO command it WOULD run) | `smoke_walls.dry_run_smoke` + `render_sumo_command` + `--dry-run` flag | **gap-fixed** | Was absent. Added: read-only pre-flight (plan schema, smoke selection, branch wall fields, registration, base saved state + scenario cfg, horizon ≤ 75 ticks, write-once refusals) printing the exact child argv and the exact `start_sumo` command (incl. the WM-64 rewritten-copy cfg path). 4 tests: happy path (asserts commands + creates nothing), missing base state, existing runs-root/verdict, missing registration / over-budget horizon |
| WALL-1 driver: online request → dispatcher wall lifecycle (assemble → hold H → release/handback; aborts any phase with reason; `wall_events.jsonl` + branch_record outcome enum) | `wall_driver.drive_wall_branch`, `WallEpisodeTracker`, `_WallRequest` | ok | REAL `MixedAssignmentSession` end-to-end: assemble/release with locked rows + handback timing (`assembled + hold`), abort-in-assembly (`member_frame_missing`), abort-in-hold, `not_feasible`, active-at-horizon fails loudly |
| Members locked from assembly to handback (group action; exclusive ownership) | dispatcher rows `wall-v2` `locked=true` with wall id/phase; `validate_wall_dispatcher_row` | ok | per-row member lock assertions; unlocked-member rows refuse |
| Non-members free/native, never commanded | `build_wall_plan` (strict roster, all non-members `native`); smoke check 4 | ok | `test_non_members_untouched`; `test_wall_command_to_a_non_member_fails` |
| CommandOwnership sole longitudinal writer | driver loop only LOGS `session.step` records; issues no speed command itself (`wall_driver.py` command-issuance contract) | ok | wall-owned command rows target exactly the member set |
| HDVs receive no commands | plans name ONLY roster CAVs (`strict_roster=True`); dispatcher iterates CAV roster | ok | `test_request_row_carries_the_declared_fields` (assignment set == CAV roster); HDVs are outside the roster by construction |
| Walls optional (no forced escalation) | separate opt-in campaign package; nothing in `traffic_shepherd` runtime changes | ok | scope note; active controller paths untouched |
| Wall assembly needs feasibility AND incremental-benefit approval | gate chain: registration → owner `AUTHORIZATION.json` (exact registration sha) → passed digest-current smoke verdict → launch | ok | authorization/verdict/gate/preflight refusal tests in `test_launch_walls.py` |
| Planner-side wall bar (median p_assembly ≥ 0.9, p_abort ≤ 0.1) | n/a (planner-side acceptance over RECORDED outcomes) | out of scope | WALL-2 only produces the data (`wall_events.jsonl`, `branch_record.json`); the bar is evaluated post-campaign by the planner tasks |
| `run_wall_branch` argument/plan wiring | `resolve_run` (refuses non-wall/unknown runs), child refusals (existing dir, partial injection), receipt schema, CLI `main` | ok + **gap-fixed** | Existing: resolve/refuse/no-injection (monkeypatched `branch_sumo_session`, asserts the WM-64 rewritten-copy target) tests. Added: `test_cli_wiring_forwards_run_plan_and_optional_flags` + argparse-required-args test |
| Import-time SUMO safety (tests never start SUMO) | whole package | ok + **gap-fixed** | Added `tests/test_import_safety.py`: fresh-interpreter import of all 8 modules leaves `traci`/`sumolib` OUT of `sys.modules` |

## 2. SUMO/TraCI launch-point inventory (exhaustive)

Every place that can start SUMO/TraCI, and why it is unreachable at import
time and from tests:

1. `run_wall_branch.py::run_wall_branch_child` — the ONLY production launch
   path: a lazy, in-function `from campaign_pilot_v1.run_pilot import
   branch_sumo_session` + `start_branch_sumo_with_lock_retry(...)` guarded by
   `if api is None`. Reached only via the `_exec_wall_branch` CLI (launcher /
   smoke subprocess). Tests always inject `api`; the no-injection test
   monkeypatches `branch_sumo_session` and asserts call args (no SUMO).
2. `smoke_walls.py::run_smoke` — `subprocess.run` of the child argv; only via
   `main()` WITHOUT `--dry-run` (or a direct call). Tests inject `launcher`.
   `--dry-run` returns before any subprocess/mkdir.
3. `launch_walls.py::_exec_child` / `launch()` — dispatches the 240 children;
   only via the `launch` CLI. Tests inject `launcher`.
4. Dependency `campaign_pilot_v1.run_pilot.branch_sumo_session` — `import
   traci` + `start_sumo(...)` live INSIDE the function (WM-64
   rewritten-scenario copy). Imported at module level by
   `campaign_stage1_v3.run_chunk`, but that import starts nothing.
5. `traffic_shepherd.simulation.start_sumo` — the single place a SUMO process
   is actually started; reached only through 1.

Proof: `tests/test_import_safety.py` imports all 8 package modules in a fresh
interpreter and asserts `traci`/`sumolib` are absent from `sys.modules`
(traci IS importable in this env, so the check is meaningful). All other
tests run on `wall_fakes.py` (FakeApi/FakeSensor/FakeOwnership/real
dispatcher session) — no test launches or connects to SUMO.

## 3. Environment notes / decisions

- **Dependencies (untracked, code-only):** this fresh worktree had no
  `output/` infrastructure. `campaign_stage1_v3` and `campaign_pilot_v1`
  (WM-64-era, `copy_dir`-capable — required by the wall child's signatures)
  were copied CODE-ONLY from `E:/rd-wt/wm64/output/` (the newest approved
  state; `E:/research-dense` copies predate WM-64 and lack `copy_dir`) into
  `E:/rd-wt/wd01z/output/{campaign_stage1_v3,campaign_pilot_v1}/` and left
  UNTRACKED (matching the repo convention that `output/` is live per-worktree
  state). They are NOT part of this commit.
- **`results_plan/` not copied:** the wm64 copy holds GENERATED artifacts
  (WORLD_PLAN.json built against the real chunk-2/3 E:/ sources + its draft
  registration), which pin the wm64 package code digest. Rebuild in the
  target with `plan_walls_pilot.py` + `walls_pilot_registration.py
  draft|finalize` when the integrator prepares the launch (registered
  commands, no simulation).
- **No `D:` paths** in any copied or new file (grep-verified; the only
  `D:`-substring hits are `WORLD:` identifiers). No wm64 paths appear in the
  package code; the pinned E:/ gate/launch paths are the registered constants
  from the source, unchanged.
- **Shared records untouched:** ledger, ROUTE/INDEX/README/AGENTS.md not
  edited (this is a code-delivery task; ledger close-out belongs to the
  integrating/planner task).

## 4. Verification (this worktree, no SUMO)

- `python -m pytest output/campaign_walls_v1/tests -q` → **63 passed**
  (56 copied + 7 added: 4 dry-run, 2 CLI wiring, 1 import-safety).
- `python -m ruff check output/campaign_walls_v1` → All checks passed!
- `smoke_walls.py --dry-run` → validated on a synthetic fixture; prints the
  exact child command + SUMO command, creates nothing (output in the task
  report).
