# JUNCTIONFIX_NOTES: junction-transit rows in the C4 WM TRAIN windows

Branch `c4/junctionfix` (from c4/int a8f552e). Switch: `data.junction_fix = "off" | "ffill"`
(default `"off"`, bit-identical). Config `configs/c4_wm_v4.toml` = `c4_wm_v3.toml` + the switch
on (plus its own `runtime.out_dir`). No SUMO, no GPU. Pack bytes are not modified (the fix is in
the loader). Only TRAIN windows were read (no sealed/, held-out, CAL or eval files).

## What the data actually contains (corrects the GAPRULE description)
* Raw packs (`veh_kin`/`veh_step`): a junction-internal-lane row has an all-zero lane one-hot,
  `veh_step` JUNCTION flag = 1, `lane_next = -1`, the vehicle's REAL speed and its LAST on-lane
  `s` (frozen while it crosses; the internal-lane length is not part of the route coordinate,
  so the next on-lane `s` jumps by ~2 steps of travel). The canonical window decodes lane -1.
* `_PackWindowLoader` sets `loss_valid = present & lane >= 0`, so **junction rows are NOT
  loss-valid** (0 of 290,563 in the census). GAPRULE's "loss_valid=True" was wrong; its 70 %
  artefact count comes from reading `states` with `present`.
* But `_states_from_teacher` clips lane -1 to 0, so in `states` every junction row is a lane-0
  car with a frozen `s`. Consumers that see this:
  1. **t0 initial state**: a vehicle on a junction at t0 starts the rollout on lane 0 (fake
     lane, fake lane-0 neighbour) while its later loss targets are on lane 1/2. 880 slots in the
     2041 TRAIN windows start on a lane different from their first supervised lane; their
     83,289 supervised HDV targets (0.28 % of all) are trained from that wrong start (lane CE
     asks for a change that never happened, gaps to the real lane-0 cars are wrong).
  2. **CAV teacher**: 29,366 CAV rows are teacher-forced with lane -1 (invisible to followers)
     and a frozen `s` that then jumps.
  3. **Aggregate loss** (loss.aggregate, entries-v2 road presence): 350,764 recorded road rows
     (0.63 %) have lane -1 and are counted in no cell, while the model keeps the car on a lane.
  4. Any `present`-masked reader (e.g. tools/gaprule_fit.py): fake 1->0->1 changes.
  The per-car losses (trajectory, lane, lane_event, flip_rate, braking, reveal prefix) already
  mask with `loss_valid` (lane_event needs both ends valid), so their change targets were clean.
* Not touched (noted): pre-t0 driver history (`long_history_features`) argmaxes the zero
  one-hot to lane 0 when searching leaders - evaluation uses the same featurisation, so changing
  it on the train side only would create an inconsistency. 78,922 supervised rows with a frozen
  `s` beyond |s| > 2048 m are float16 quantisation (2 m spacing), not junctions (GAPRULE's
  frozen-s filter also dropped those).

## Fix (`ffill`, src/wms/data/junction_fix.py, applied in `_PackWindowLoader`)
Evaluation (`wms.eval.rollout_inputs.recorded_teacher`) carries a transiting vehicle at its
previous lane and skips the loss; deployment (`LaneDwell`) keeps the last lane. `ffill`:
* lane = last on-lane lane of the same vehicle (same-vehicle continuity from `veh_mask` +
  `target_mask`); for a t0 transit the last on-lane row BEFORE t0 is read from the pack
  (<= 40 ticks, never across the run start) - the information exists, so it is used;
* `s` = linear interpolation in time between the bracketing on-lane rows (removes the frozen
  `s` + jump; the coordinate is continuous across the junction); held at the last on-lane `s`
  if no later on-lane row is inside the window (evaluation convention, 3,779 rows);
* `v`, `a` unchanged (real measurements); `loss_valid` unchanged (transit step still not
  supervised - its `s` is an interpolation, not a measurement);
* a t0 transit vehicle with no on-lane pre-t0 row (100 of 1,443) is dropped from the window
  (evaluation drops every t0 junction vehicle) and removed from the entries-v2 road presence
  while it stays in its slot, so the aggregate targets count exactly the simulated population;
* the filled lane goes into `cav_teacher` too, so the aggregate recorded lane and the CAV
  teacher use it (recorded cell fields count a transiting car at its filled lane and `s`).
Why not just `loss_valid=False`: those rows already are; the corruption is in the states that
are used as inputs (t0, CAV teacher, aggregate), so the faithful fix is to fill them.
Applies to every pack window: pack_v3, pack_dspeed and pack_switching_train (all through
`TrajectoryDataset.from_real_packs`), TRAIN and validation alike. `c4_episodes` (off in C4) and
`from_arrays` are not changed.

## Census, all 2041 TRAIN windows of configs/c4_wm_v3.toml (docs/junctionfix/train_census.json)
`tools/junctionfix_census.py` (same start shift per window in both modes). HDV rows unless noted.

| | off | ffill |
|---|---|---|
| raw changes, present-masked (GAPRULE view) | 207,205 | 54,418 |
| - of which an end is a junction row | 153,645 (74 %) | 884 (real changes across a transit) |
| - overtake 1->0 / return 0->1 / merge 2->1 / other | 103,609 / 90,142 / 8,591 / 4,863 | 29,761 / 15,944 / 8,708 / 5 |
| changes seen by the losses (both ends loss_valid) | 53,560 | 53,534 (-26 = dropped slots) |
| - overtake / return / merge | 29,303 / 15,661 / 8,591 | 29,294 / 15,660 / 8,575 |
| junction rows (HDV + CAV) | 261,197 + 29,366 | same rows, filled |
| junction rows with frozen `s` | 289,120 | 3,779 (held at window end) |
| junction rows with lane -1 in the teacher (CAV teacher, aggregate lane) | 290,563 | 0 |
| junction rows that are loss_valid | 0 | 0 |
| t0 junction vehicles (HDV / CAV), windows with one | 1,293 / 150, 1,077 | 1,205 / 138 kept, 100 dropped |
| t0 slots starting on a lane != first supervised lane | 880 | 14 (real change across the transit) |
| supervised targets of t0-junction slots (of 29.79 M) | 83,289 (0.28 %) | 76,932 (start lane now real) |
| loss_valid rows removed / added | - | 7,410 (0.025 %, dropped slots) / 0 |
| road_present rows with recorded lane -1 (aggregate) | 350,764 of 55.58 M | 66 |
| windows whose speeds changed | - | 0 |

By source: original (583 windows: pack_v3 139 + pack_dspeed 444) raw junction-end changes
50,351 -> 150, t0 lane-mismatch 280 -> 2; switching direct-mix (1458) 103,294 -> 734, 600 -> 12.
(The GAPRULE 121k / 36.8k numbers used its own legal-target filter; the ratio agrees.)

## Tests / smoke
* `tests/unit/test_junctionfix.py` (18): synthetic fill (interpolation, real change kept,
  change across a transit, window-end hold, t0 lookback / unfilled, slot reuse, absent rows,
  run prefix, pre-t0 lookback continuity); v4 = v3 + switch + out_dir only (flat and nested,
  flat roundtrip, settings); real TRAIN data: `off` record hashes equal golden hashes captured
  with c4/int a8f552e for 7 windows (pack_v3, pack_dspeed, switching), `ffill` collates with the
  same keys/shapes, removes fake 1->0->1 and frozen lane-0 rows, no lane -1 teacher rows left,
  loss-valid changes / loss_valid / on-lane states / speeds unchanged except dropped slots.
* OMP_NUM_THREADS=1: test_junctionfix + test_gaprule + test_wmlane + test_wmloss +
  test_lane_rule + test_c4_core + test_import_graph -> **302 passed**. Also config/, data/,
  test_c4_loop, test_r5_switches, train_layer, equivalence: 300 passed, 14 skipped, 1 failed
  (`test_drive_registry::test_committed_registry_matches_the_generator_output`, compares
  tools/data_registry.json with the local data_ops tree; unrelated, no file of it touched).
  ruff clean on touched files.
* CPU smoke `tools/c4_smoke.py E:/agent-work/tmp/junctionfix_smoke_v4 configs/c4_wm_v4.toml`
  (data via `wmlane_smoke.point_data(E:/research-dense)`): **passed**, 2 optimizer steps,
  finite, loss 11.16 -> 9.69, 2041 TRAIN / 261 val windows, TRAIN_RECEIPT records
  `junction_fix = "ffill"`.
