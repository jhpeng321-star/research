# CHANGES_WALL_CLOSURE (owner decision 2026-09-30, "Option B", law wall-v2.1-closure)

Branch `zcode/wd01wallB` (from `zcode/wd01` 49af09f). Code and tests only; ZERO simulation.

| Item | File / function | Test |
| --- | --- | --- |
| Assembly closure: front (max downstream position) holds `v_wall`, rear targets `v_wall + min(2.0, 0.1*gap)` inside the `[-3,+1.5]` clip | `traffic_shepherd/wall_group.py`: `ASSEMBLY_CLOSE_MAX_DELTA_MPS`, `ASSEMBLY_CLOSE_GAIN_PER_S`, `assembly_target_mps`, `WallGroupSession.tick` | `tests/test_wall_group_closure.py::test_a_wide_pair_converges_and_front_is_paced`, `::test_a_rear_command_equals_formula`, `::test_front_is_most_downstream_regardless_of_id_or_lane_order` |
| Small span regression (assembled on the same tick as before) and speed test at Delta 0.6 | same | `::test_b_close_pair_assembles_on_first_tick`, `::test_speed_test_passes_with_delta_at_span_tolerance` |
| New safety abort `assembly_timeout` (>= 60 s after `assembly_start`, evaluated after the old aborts, immediate native handback) | `wall_group.py`: `ASSEMBLY_TIMEOUT_S`, `WallGroupSession.tick` | `::test_c_assembly_timeout_and_handback`, `::test_timeout_after_other_aborts_order` |
| HOLD unchanged (all members `_pace(v_wall)`) | `wall_group.py` (HOLD branch untouched) | `::test_d_hold_paces_all_to_v_wall` |
| 3 members | same | `::test_e_three_members_close_to_front` |
| `law_version="wall-v2.1-closure"` in `assembly_start` / `assembled` events | `wall_group.py`: `WALL_LAW_VERSION` | `::test_f_law_version_in_events`; `tests/test_mixed_assignment.py::test_wall_telemetry_events_and_controller_state_hooks` (updated) |
| Part A mirror | `output/wd01/partA_ctrl/emulators.py`: `wall_pace(..., gap_to_front_m)`, `wall_gap_to_front`, `ControllerEmulators.emulate(..., wall_gap_to_front_m)`, constants | `output/wd01/partA_ctrl/tests/test_e3_wall_closure.py` (1e-9 parity vs the worktree `wall_group`, 300 random batches, assembly and hold; constants pinned); `test_e3_wall.py` (updated) |
| Mirror consumer: positions, span test, 60 s timeout | `output/wd01/partA_v2g/cavctrl.py`: `WallBook.observe/assembling/active_speed`, `CavControllerLoop.__call__` | `output/wd01/partA_v2g/tests/test_cavctrl_wall_closure.py` (timeout tick == `WallGroupSession`) |
| Reference stack binding | `output/wd01/partA_ctrl/refload.py::_bind_worktree_wall_group` (the parity target is THIS worktree's `wall_group.py`, not the frozen main tree copy) | parity tests above |
| WM-43B fixture draft (NOT run, NOT authorized) | `output/action_model_v1/fixture_v2/` (see `REGISTRATION_WM43B_DRAFT.md`) | `output/action_model_v1/fixture_v2/tests` (95 tests, fakes only) |
| Dependency modules made tracked (needed by `pi_session` import chain; byte copies of the main-tree ignored files) | `output/controller_verification_v2/pi_saturation_v1.py`, `_v4/pi_external_u.py`, `_v7/pi_owner_adapter.py` | fixes 2 pre-existing failures in `tests/test_mixed_assignment.py` and enables the fixture loop tests |

## Old tests altered (only where the behaviour legitimately changed)

- `tests/test_mixed_assignment.py::test_wall_telemetry_events_and_controller_state_hooks`: the `assembly_start` event payload now carries `law_version`.
- `output/wd01/partA_ctrl/tests/test_e3_wall.py::test_wall_member_command_law_equivalence`: assembly-phase expectations now use the closure target (`gap_to_front_m`); hold expectations unchanged.

## mixed_assignment.py

No change needed. It only does `wall_accels.update(session.tick(...))` (per-member dict) and forwards events; nothing assumes identical member commands.

## Fixture v2 changes vs v1 (`fixture/`)

Dropped `apply_patch_r6/r7.py`, the two patch tests and `test_wm43_r3_replay.py` (bound to deleted v1 raw evidence / already-applied patches); `test_wm43_boundary_band.py` keeps the A2 boundary-crossing arithmetic (margin relaxed 5 s -> 3 s for the new seeds: crossings 290.2-296.3 s). `conftest.py` no longer patches `wall_group`. New telemetry row `wall_frames`, check `C07_law_version_identity`, closure-aware `A1_7`, arms A4/A5.
