# DATAHYG_NOTES: no designed controllers in the C4 WM-S training windows

Branch `c4/datahyg` (from c4/int a2d3716). Switch: `data.controller_hygiene = "off" | "strict"`
(default `"off"`, bit-identical). Config `configs/c4_wm_v4_hygiene.toml` = `c4_wm_v3.toml` +
`data.controller_hygiene = "strict"` + `data.junction_fix = "ffill"` + its own
`runtime.out_dir`. No SUMO, no GPU, pack bytes untouched (the fix is in the loader). Only TRAIN
packs / sidecars / C4 lists were read; no guard file (ZERO_SHOT_HELDOUT), CAL, holdout or
sealed path was opened.

Owner rule: the world model learns action effects only from native driving, undirected random
direct-output commands and the learned policy's own rollouts. Audit:
E:/agent-work/tmp/actiondata (summary_full2.txt, table.txt).

## What the WM actually consumes (why the drop matters and the relabel mostly does not)
* Structured WM-S (all C4 configs: `cav_drive = "teacher"`, not black box) TEACHER-FORCES every
  CAV from `cav_teacher` = the recorded realized `(s, lane, v, a)` (`WmSModel._apply_cav_overrides`);
  `WmSTrainer` sets `model.apply_direct_commands = False`; the controller-id / command embedding
  exists only in the black box (`_controller_embedding` returns None otherwise). The losses
  supervise HDVs only. So what the model learns from a window is **how HDVs react to the
  recorded CAV motion** - a damping CAV in the window teaches the reaction to a designed
  controller regardless of its label. Removing those windows is what changes the teaching.
* The `action_timeline` labels/commands are NOT read by the structured teacher-forced model.
  They are read by: `cav_drive = "closed_loop"` (TimelineControllers runs native / damping /
  direct per label), the black box (controller + command embedding), `switch_window_fraction`
  and the switching-source event oversampling (both inert in C4: event_oversample 1.0).
  The dspeed relabel therefore corrects the record (and protects those modes) but does not
  change a C4 v3/v4 gradient. To make sure a wrong command is never taught, `command_known`
  marks recorded commands and `prepare_inputs` refuses `direct` steps with an unknown command
  under closed-loop or the black box.

## Findings confirmed
1. **direct-mix is switching, not random direct-output.** The generator
   (`traffic_shepherd/switching_data.py` + `workflow.py`, branch wms/swpack) draws per-vehicle
   schedules over (native, damping, direct), holds log-uniform 1-60 s, seed = world seed. The
   pack `veh_step` ownership channel says damping for every CAV (useless); the per-item timeline
   labels only the t0 slots. Re-implemented bit-identically (golden test); on the real TRAIN
   items every one of the 306,801 recorded damping steps is a scheduled damping second
   (0 mismatches; 128,858 scheduled-damping steps were recorded native/direct = safety
   fallbacks). Post-t0 entrants (1.17 M CAV rows of 3.84 M in the 120 s windows) are named by
   following their slot's same-vehicle chain to the run's next snapshot item.
2. **dspeed.** `trajectory_arm.py` (branch impl/dspeed-arm): 30 % / 50 % of the t0 CAVs follow
   10-30 s segments of five uniformly drawn types for 120 s, then everyone is native for 480 s.
   `merge_aimed` / `gap_opening` are designed. Re-implemented bit-identically (type/length
   sequence independent of the lane limit; golden test); the reconstructed selection gives
   57.34 directed CAV-h on the 444 TRAIN items (audit: 57.41). `_fixed_timeline` labelled every
   CAV of a dspeed window `direct` (1118 CAV-h) with command = `cav_plan_ch` = realized v/40, a/3.
   The planned commands are not recoverable: `cav_plan_ch` is realized, overlay_dspeed is
   outcomes only, `plan.npy` holds decision snapshots, the shards are gone.

## Strict mode (src/wms/data/controller_hygiene.py; hooks in dataset.py, mixdata.py, assemble.py)
* **direct-mix (`mixdata.DirectMixHygiene`, after the held-family filter):** drop an item when
  its window `[t0, t0 + horizon_steps]` (= the CAV-teacher span; horizon_steps = 240, the 120 s
  stage) holds a damping step of ANY CAV on the road (t0 cohort: recorded label OR schedule;
  entrants: schedule; a CAV whose identity is unresolved counts as damping).
* **Option choice.** Drop removes **99.66 %** of the TRAIN items (5 of 1458 kept; 1451 damping,
  2 unresolved) and 99.4 % of VAL (1 of 162) - far above the 60 % threshold. The mask
  alternative (truncate supervision at the first damping step) keeps only 2.8 % of the
  supervised window ticks: median clean prefix 0 s (a CAV is already damping at t0 in > 75 % of
  items); items with a clean prefix >= 10 s / 30 s / 60 s / 120 s: 128 / 49 / 19 / 4 (TRAIN),
  13 / 5 / 2 / 0 (VAL) (census tool, `docs/datahyg/census_h240_directmix.json`; the tool
  reads run ends from INDEX.json and counts damping-or-unresolved together, hence its 4 kept
  vs the loader's 5). Neither
  option leaves usable direct-mix data, so **strict drops and, because > 60 % is removed,
  leaves the whole switching source out of the mix** (`HYGIENE_MAX_DROP_FRAC = 0.6`; receipt
  `switching_disabled_by_controller_hygiene`, stderr line). No mask mode was built.
  **Integrator decision needed** (see "Consequence").
* **dspeed (`dataset._dspeed_hygiene` + `_PackWindowLoader`):** per item a `DspeedRun`
  (selected slots, same-vehicle directed span <= 120 s, designed ticks = merge_aimed /
  gap_opening segments of any directed CAV still in its slot, +-2 tick margin). A window may
  not contain a designed tick. Items with no allowed start are dropped (0 of 444); a refused
  start (unshifted, or a start-shift draw) moves to a uniformly drawn allowed start (global
  torch RNG), or to the earliest allowed start when the dataset does not start-shift (VAL).
* **Relabel:** dspeed timeline = `direct` only for directed CAVs while directed, `native` for
  every other CAV and for all steps after the 120 s window; commands 0 with
  `command_known = False` (unknown, not the realized acceleration). In strict mode every pack
  timeline carries `command_known` (True only on recorded direct commands of a provider
  timeline); collate stacks it when every record has it; shapes of every other key unchanged.

## Census (TRAIN / VAL of configs/c4_wm_v3.toml; tools/datahyg_census.py, docs/datahyg/)
| source | items before | items after (strict) | CAV-h by controller before | after |
|---|---|---|---|---|
| pack_v3 native | 139 / 23 | 139 / 23 | native 310.2 / 74.8 (run spans) | unchanged |
| pack_dspeed | 444 / 76 | 444 / 76 (0 dropped; 411 / 72 unshifted windows refused, allowed starts 80 % / 78 %) | old label: direct 1117.7 / 270.2 (all CAVs); truly directed 57.3 / 13.8, of which outside designed ticks 7.3 / 0.9 | label direct = directed only; per window draw directed share 2.90 % -> 0.04 % (VAL 2.83 % -> 0.01 %) |
| direct-mix (120 s item spans, t0 cohort, recorded label) | 1458 / 162 | 0 / 0 (source disabled; 5 / 1 would pass) | native 156.5 / 26.7, damping 85.4 / 13.0, direct 122.5 / 21.8 (+168.7 / 26.8 CAV-h entrants) | 0 |

Exogenous random direct CAV time after strict: direct-mix 0; dspeed random-segment directed
share per window draw 0.39 % -> 0.04 % of CAV rows (TRAIN), 0.23 % -> 0.01 % (VAL). With the
switching source gone, **strict TRAIN data is native driving with ~0.04 % random directed CAV
time** (before: ~30 % of draws from direct-mix with ~34 % direct cohort time plus dspeed ~2.9 %).

## Consequence (for the integrator)
The rule as specified, applied at the 120 s window horizon, removes essentially all exogenous
action data: designed segments cover almost the whole dspeed directed phase whenever >= 2
CAVs are directed, and damping is everywhere in direct-mix. Options (not built):
(i) stage-aware windows (a window only needs to be clean over the unroll it is used for:
dspeed allowed starts / random-directed share at 10 s, 30 s, 60 s horizons are in
docs/datahyg/census_h{20,60,120}_dspeed.json - random directed share stays <= 0.4 %, so this
does not rescue dspeed; direct-mix keeps 128 / 49 / 19 items at 10 / 30 / 60 s);
(ii) generate new pure random direct-output data (direct-mix with choices (native, direct));
(iii) rely on the learned policy's own rollouts (c4_episodes) for action effects.
Caveat kept: pre-t0 driver history (60 s) of a kept window can include HDV behaviour shaped by
earlier designed segments / damping (input only, never a target).

## Tests / smoke
* `tests/unit/test_datahyg.py`: generator goldens (schedule, dspeed selection/segments),
  DspeedRun directed/designed/allowed starts, slot reuse, relabel on a synthetic timeline,
  start-shift move, direct-mix cohort/entrant/unknown damping, config v4_hygiene = v3 + 3 keys
  (flat + nested, roundtrip, settings validation), kwargs omitted when off, collate
  `command_known`, unknown-command refusal; real data: `off` record hashes equal goldens from
  a2d3716 (v3 and v4, TRAIN 8 + VAL 4 windows each), strict disables direct-mix with the
  receipt numbers above and 0 recorded-damping/schedule mismatches, strict dspeed windows
  (27 draws, TRAIN + VAL) contain no designed tick and carry the relabelled timeline,
  collate keys/shapes equal to off + `command_known`.
* OMP_NUM_THREADS=1: test_datahyg 15 passed (real data present); test_junctionfix +
  test_gaprule + test_wmlane + test_wmloss + test_c4_core + test_import_graph + test_lane_rule
  304 passed; tests/unit/config + data + train + test_r5_switches + test_c4_loop 302 passed,
  1 failed (`test_drive_registry::test_committed_registry_matches_the_generator_output`, the
  known unrelated local data_ops mismatch also reported by JUNCTIONFIX). ruff clean.
