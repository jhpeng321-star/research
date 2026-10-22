# R9_LANE - ramp lane-end stop (physics.ramp_lane_end_stop) and residual diagnosis

Branch c4/v10r9lane (merged c4/v10r9launch 78b93f16). Status block at the bottom is refreshed as results arrive.

## Change (file:line)
- `src/wms/physics/config.py` `PhysicsConfig.ramp_lane_end_stop: bool = False` (+ bool validation, excluded from `from_params_file`).
- `src/wms/model/merge_pivot.py` `ramp_lane_end_stop(kin, next_s, next_v, present, physics, route_shift)`: a present ramp-lane car (lane one-hot of the PRE snapshot) whose longitudinal update puts its front past `physics.merge_zone_s1_m` (the network lane end, 75.44 m; cast to the tensor dtype, no tolerance) is held at the end with speed 0. Route-coordinate aware (`x = s + shift`).
- `src/wms/model/wms.py` `pivot_decision_geometry` (shared by recorded replay, fitter, runtime audit and the native step): decision geometry sees the pinned front (idempotent). `step()`: the pinned state goes into `_apply_merge_pivot`; afterwards cars that CHANGED lane that step get their free (unpinned) motion back - cars that stay on the ramp lane keep position = lane end, v = 0.
- `src/wms/train/losses/merge_pivot.py` `build_recorded_geometry`: pins `moved` before the geometry so the replay observers (runtime cache `proposals`, fitter sequences) see the same motion as the audit.
- Plumbing: `config/train.py` (`PhysicsOptions.ramp_lane_end_stop`, inference switch), `cli/assemble.py` `build_physics`, `config/legacy_flat.py`, `train/settings.py`.
- Round-9 pin (owner-approved 2026-10-22): `configs/c4_wm_v10round9.toml [physics] ramp_lane_end_stop = true`; `configs/r9_physics_round9.json` (fitter physics) regenerated = `asdict(build_physics(toml))` (only that key added). Code default stays false; round-8 preset unchanged.
- `tools/r9_support_reconcile.py --ramp-lane-end-stop on|off|preset` (default preset). Fitter honours the flag through the physics JSON; runtime cache digest already hashes `str(physics)` and the source files.
- Not a tolerance/padding: no clamp slack, no threshold; the end is the geometry value already in `physics`. Flag off = identical code path (guarded by `getattr(..., False)`).

## Tests
`tests/test_r9_lane_end_stop.py` (11): end==75.44; stopped car at the end stays (pinned, v 0); approaching car stops exactly at the end; cars not reaching it / lane-1 cars / absent slots untouched; route shift honoured; flag off = unchanged (car carried past end in legacy); step holds a staying car and leaves other slots bit-identical; a merging car is decided at the end and keeps its free motion (next_s/next_v equal the flag-off free motion); decision geometry pinned + idempotent; round-9 preset + physics JSON pin; bool validation.
