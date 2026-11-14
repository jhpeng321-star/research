# Controller-switching and direct-output generator — readiness

**Status: READY — the corrected suffix/action log was physically verified in
both switching and direct-mix mechanics runs, targeted tests pass, and the box
bundle has been rebuilt and SHA-256 checked.** This authorizes no full generation:
that remains a separate owner-approved step. The original six-run pilot and the
two-run amendment are fully consumed; no further SUMO runs are included here.

## Design and action semantics

- `switching` generates independent per-CAV sequences from
  `{native, damping, pi-external-u, direct}`. `direct-mix` uses
  `{native, damping, direct}`. Segment durations are seeded log-uniform draws,
  rounded to whole 1 s control ticks (default bounds 1–60 s); each CAV's
  contiguous schedule is written to `switch_schedule.json` before SUMO starts.
- At every control second the runner submits a strict per-CAV
  `AssignmentPlan` to `MixedAssignmentSession` with dwell horizon 0. The
  session applies switches on their scheduled ticks; `CommandOwnership` remains
  the single longitudinal writer. Native releases an existing override via
  `setSpeed(-1)`; all CAV commands retain SUMO speedMode31. HDVs receive no
  commands.
- One `CirclesController` and one owner instance persist through the episode.
  CIRCLES inference is stateless per decision; leaving and re-entering damping
  does not rebuild or reset the instance. PI uses one persistent session, but
  its per-CAV recurrence is invalidated on PI exit and reinitialized from the
  measured speed on re-entry (also invalidated on missing/stale/departed state).
  Direct acceleration profiles are deterministic functions of `(seed, CAV id,
  episode time)` and do not restart on segment entry.
- The direct profile is a seeded four-sinusoid mixture, clipped to
  `|a| <= 1.5 m/s^2` and jerk-limited to `0.5 m/s^3` per one-second tick. It is
  converted to a target speed through the existing `direct_target` bounds and
  ownership path. SUMO's issued target and measured response remain separate in
  telemetry. PI's adapter requires the persisted owner receipt to be identified
  as damping; the schedule-based action timeline therefore records logical
  `owner` and underlying `command_owner` separately, and selects the actual
  fallback owner when a safety fallback occurs.
- HETDATA v2's scenario generation and its three continuous class-anchored
  driver distributions are unchanged. `--hetero-drivers` continues to emit
  `hetdata_truth/2` and full 0.5 s `physics_steps.jsonl` rows using
  `wm41_logging/1`.

## Bundle integration

- `hetdata_kit/run_kit.py` exposes `--arms switching,direct-mix`; both run through
  the evaluator with heterogeneous drivers. `_out/box_bundle/box_runner.py`
  accepts both arms and filters its pilot to three worlds × two selected arms.
- `_out/box_bundle/build_box_bundle.py` now roots the worktree at this workspace
  and copies the three transitive PI sources from `_in/pi_deps/` into the root
  and live bundle paths. The source for saturation is under the supplied
  `controller_verification_v2/history/` path; the builder maps it to the
  import-time `controller_verification_v2/pi_saturation_v1.py` path. It reapplies
  those copies after whole-package copies so they cannot be overwritten.
- The bundle tar was rebuilt workspace-locally from the prior validated archive,
  overlaying the current `traffic_shepherd/`, `hetdata_kit/`, box runner and
  supplied PI dependency sources; its file manifest and archive SHA-256 were
  regenerated. No external tree was read.
- Rebuilt `_out/box_bundle/hetdata_box_bundle.tar`: 7,792,640 bytes,
  440 manifest entries, SHA-256
  `1bec31cf0b85a331bb08e8110842142d67191c6d44e4a7ae3db8583d2fb6cfd3`.

## Tests and mechanics evidence

- Specified interpreter:
  `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe` (torch 2.12.1+cpu).
- Scoped active verification:
  `PYTHONPATH=_in/pi_deps python -m pytest -q tests/test_switching_data.py tests/test_hetero_plan.py tests/test_mixed_assignment.py tests/test_active_workflow.py tests/test_command_ownership.py tests/test_hetdata_kit.py tests/test_rsu_reporting.py tests/test_pi_workflow.py`
  — **103 passed**.
- `python -m ruff check traffic_shepherd tests/test_switching_data.py tests/test_hetero_plan.py tests/test_mixed_assignment.py tests/test_hetdata_kit.py hetdata_kit/run_kit.py _out/box_bundle/build_box_bundle.py _out/box_bundle/box_runner.py`
  — passed.
- Full `python -m pytest -q` with the supplied interpreter and `PYTHONPATH` —
  **collection interrupted: 39 errors, 3 skipped**. Errors are missing archived
  `output/controller_verification_v*`, `output/planner_v1/resmoke`, and
  `wm20_common` sources/fixtures; the PI imports required by the new runtime are
  supplied and work. This is the same partial-worktree limitation that makes
  `python -m traffic_shepherd inspect` fail on
  `output.controller_verification_v7`; inspect was skipped as instructed.
- Mechanics budget `WM-SW-DATA-20261007-MECHANICS-PILOT`: three admitted
  development worlds (920002–920004, 2700 veh/h, p=0.05), each selected for both
  families, 180 s maximum. Five runs completed in SUMO 1.27.1: 920002
  direct-mix; 920003 switching and direct-mix; 920004 switching and direct-mix.
  The 920002 switching attempt failed before tick 0 with
  `CAVReport: unknown command ownership` on the newly selectable direct owner;
  `traffic_shepherd/reporting.py` now accepts the direct and PI owner labels.
  No retry was made; all six allocations are consumed.
- Pilot audit: all five completed runs had 0 collision involvements, 0 starting
  teleports, speedMode31, 360 valid HETDATA physics rows, and separate measured
  responses. Across them, 3938 live-CAV schedule/action rows exactly matched the
  prewritten schedules; 3938 command rows and 7856 response rows were present;
  1800 total physics rows passed the `wm41_logging/1` row validator. The two
  successful `switching` runs contained 504 PI-assigned ticks. The pilot is
  mechanics-only and supports no efficacy claim.
- In the five original outputs, initial `suffix_commands.jsonl` PI rows used the
  adapter's damping writer identity, although `mixed_assignment.action` matched
  the PI schedule. This was corrected in source and is now resolved by the
  amendment's physical suffix-log verification below.
- Amendment verification `WM-SW-DATA-20261007-MECHANICS-PILOT` (same-world
  switching + direct-mix): seed 920003, 2700 veh/h, p=0.05, `--role development`,
  180 s each, SUMO 1.27.1; outputs are `_out/swgen_suffix_verify/920003/`.
  Both had 360 HETDATA physics rows, speedMode31, zero collision involvements
  and zero teleports. Switching produced 837 schedule-matched assignments and
  suffix rows, 1669 measured responses and 112 PI logical-owner rows; direct-mix
  produced 835 schedule-matched assignment/suffix rows and 1665 measured
  responses. For every suffix row, `owner` equals scheduled mode without a
  fallback and equals actual `command_owner` with a fallback. The earlier
  pre-SUMO preflight failures (unregistered seed / missing development role)
  consumed zero simulated seconds and are recorded in the registration.
- Targeted tests rerun after the correction: **103 passed**.
- World-model identity join for `switch_schedule.json` / `suffix_commands.jsonl`:
  use the literal `vehicle_id` string as the stable key. In each run's HETPACK
  identity sidecar, look up that string in `vehicle_map.json`; its mapped
  identity row is the corresponding entry in `identities.npy` (use the pack's
  CAV indicator to retain only CAV identities). At time index `t`, resolve that
  stable identity through the per-tick dense slot map to the current packed
  column `k`; gather state as `states[t, k, :]`. The schedule's integer
  `timestamp_s` is the 1 s control tick; the suffix `vehicle_id` and timestamp
  join to the same identity and tick, while suffix command fields remain
  controller labels/issued inputs and are not WM state features. The slot column
  is tick-local and must not be treated as a stable vehicle identity. Missing
  or departed IDs have no dense slot at that tick; do not infer one from column
  order. The workspace contains no actual `vehicle_map.json` / `identities.npy`
  pack sidecars, so this describes the exact join path, not a pack-specific row
  number audit.
- Pilot evidence: `_out/swgen_pilot/<seed>/<arm>/`; suffix amendment evidence:
  `_out/swgen_suffix_verify/920003/{switching,direct-mix}/`; allocation/closeout
  note: `_in/WM-SW-DATA_REGISTRATION.md`.

## Proposed full generation (not authorized or started)

Use 150 HETDATA v2 worlds × two families = **300 runs**, each 1200 s, with
matching seed/population across `switching` and `direct-mix`. Keep the existing
split by world (TRAIN 90 / VAL 30 / TEST 30), never split a world across splits,
and do not change the continuous three-anchor driver draws. Use resumable
per-attempt receipts and workers only after the bundle is rebuilt and the
post-correction pilot check is registered and completed. Process one run/state
at a time; open `.npy` packs with `numpy.load(..., mmap_mode="r")`; keep every
Python process below 6 GB.

At HETDATA v2's estimate of 0.70 CPU-h and 80 MiB raw per 1200 s run, the plan
is approximately **210 CPU-h**, **23.4 GiB raw**, and **4.2 GiB compressed**
(0.18 estimated compression ratio). These are planning estimates only; no gpuA
or remote system was accessed.

## Required next action

Full generation remains separate and requires owner authorization. When
authorized, use the rebuilt bundle and registered 150-world × two-family plan;
keep every Python process below 6 GB, mmap `.npy` packs, and process one state/run
at a time. No generation was started by this verification task.
