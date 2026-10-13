# SCHED: WM-S training-schedule switches (2026-10-14)

Branch `c4/sched` (from `c4/int` 0be5b72). Owner-approved 03:20. Code, tests and a CPU smoke
only: **no GPU run, no SUMO**. Every switch defaults OFF, and with them off the training loop
makes the same decisions as before (test-checked).

## Why (E:/agent-work/tmp/v3diag, MORNING_WM_DIAG_NOTES "v3 error breakdown")
* The old selection compares validations only at the longest eligible horizon, so the final
  stage always wins (V3best == V3last). After stage 1 nothing validates at 10 s, so the 10 s
  braking transient that appears in the 30 s stage went unnoticed.
* The 60 s gap vs C4 main opened in the 30 s stage (because of the objective, not the step count).

## Switches (`[curriculum]`; flat keys in parentheses, omitted from receipts while default)
| key | default | effect |
|---|---|---|
| `val_all_horizons` (`val_all_horizons`) | false | each validation runs every horizon in `selection_horizons_s` up to the stage, plus the stage horizon (10 s stage: 10; 30 s: 10, 30; 60 s: 10, 30, 60; 120 s: 10, 30, 60, 120). All rows go to val_log. |
| `selection` (`selection_mode`) | `"long_horizon"` | `"multi_horizon"`: one decision per validation event (see below) |
| `selection_horizons_s` | [10, 30, 60, 120] | horizons that get scored |
| `selection_reference` | [7.0356, 49.4889, 128.7179, 0.0] | reference val `mean_position` per horizon; a value <= 0 means "this run's own first validation at that horizon" |
| `selection_min_stage_s` | 30 | events compete only from the first stage whose horizon is >= this (the C4 imagination horizon) |
| `selection_agg_reward_weight` | 0 | adds w x mean_h(agg_reward_h / own first agg_reward_h) |
| `mixed_horizon_frac` | 0 | in stages >= `mixed_horizon_min_stage_s` (30), this fraction of batches unroll a short horizon |
| `mixed_horizon_s` | [10, 20] | short horizons, drawn uniformly. Only horizons shorter than the stage are used |
| `mixed_horizon_min_stage_s` | 30 | |

Both validation switches require `selection.select_by_long_horizon = true`, which is the
validation schedule (`val_every` plus each stage end).

## Multi-horizon selection (`wms.train.selection.MultiHorizonSelection`)
score(event) = mean over the horizons validated in the event that are selection horizons
(finite, valid targets) of `mean_position_h / ref_h`, plus the optional agg-reward term.
wm_s_best.pt goes to the lowest-scoring eligible event (ties keep the earlier one). It carries
`selection` = that table row. Per-stage checkpoints and the final `wm_s.pt` are unchanged.
The full table (every event: step, stage, score, per-horizon ratios, eligible, selected) is
written to `TRAIN_RECEIPT.long_horizon_selection.multi_horizon` and to `RUN_SUMMARY.json`
under `selection` (that key is absent when the switch is off). val_log rows get a
`selection_score` field.

**Reference choice (documented decision):** a fixed external reference = the C4 main WM's
best val `mean_position` at 10 / 30 / 60 s (E:/research-dense/runs/active/WM-S-C4-20261013/
fetched/gpuW/c4_wm_main/val_log.jsonl: 7.0356 @ step 1500, 49.4889 @ 3000, 128.7179 @ 3600).
So each ratio reads as "relative to the old model", and every horizon gets comparable weight.
I rejected normalising every horizon by the run's own first validation: the first 10 s
validation is at step 200 (untrained, about 11), while the first 120 s validation comes
after 3000+ steps, so the 10 s term would be strongly down-weighted. C4 main never validated
120 s, so 120 s uses the run's own first 120 s validation (ratio 1 at the first 120 s event).
Caveats: C4 main validated without entries, while v3/v4 validate with entries-v2 (V3's
values are about 7 % higher at 10/30/60 s). The ratios are a scale, not a claim to beat C4 main.

**Comparability caveat:** with `val_all_horizons` a 30 s-stage event is scored on {10, 30}
and a 120 s-stage event on {10, 30, 60, 120} ("available" horizons, as specified). A 30 s-stage
checkpoint is therefore not penalised for its unvalidated 60/120 s error. Check the table
before using wm_s_best for 60/120 s decisions. To make every eligible event score all four
horizons, validate them all from the 30 s stage on: about +0.8 h on gpuW, and the 120 s
rollout of a 30 s-trained model may blow up (non-finite raises). Not implemented.

## Mixed-horizon batches (`MixedHorizonSchedule`)
The schedule has its own `random.Random(seed * 1000003 + 4242)` stream (torch and python
training RNG untouched; frac 0 draws nothing). A short batch uses the same loader batch,
loss composer, stage lr and stage BPTT chunk length. The unroll is `round(h / dt)` steps
(prepare_inputs truncates the window), and a chunk longer than the unroll is a single chunk.
The aggregate loss ramp is indexed by horizon step, and its normalisers (`w_field`,
`w_reward`) are computed over the unrolled steps, guarded by `> 0`. So a 10 s batch gets the
same 5-30 s ramp weights as the 10 s stage. The entry pad (`entry_pad_auto`) is sized from the
actual unroll. train_log rows of short batches carry `mixed_horizon_s`.

## Config `configs/c4_wm_v4_sched.toml`
extends c4_wm_v3 + val_all_horizons + multi_horizon + mixed_horizon_frac 0.3 +
stage_steps [1200, **1500**, 400, 160] + own out_dir. The flattened diff vs v3 is exactly
{stage_steps, val_all_horizons, selection_mode, mixed_horizon_frac, out_dir} (test). The
reference / horizons / mixed_horizon_s values are written out but equal the defaults.
Wall-time estimate (detailed in the config header): training about 5.35 h plus validation
about 0.5 h, so **about 5.9 h** on gpuW. That is under the 8 h target and under the
inherited max_hours 7.0. Cuts if slower: val_every 300, then stage_steps[3] 120, then
stage_steps[1] 1200.

## Tests / smoke
* `tests/unit/test_sched.py` (14 tests):
  * off and explicit-off give identical train/eval call sequences and the same old selection, on a faked fit;
  * a real-compute fit with the keys absent is bit-identical to one with them explicitly off;
  * validation horizons, and val_log holding every horizon;
  * multi-horizon score arithmetic and pick on a synthetic log (the old policy picks the final stage on the same log);
  * agg weight, and NaN / unknown-horizon rows;
  * multi selection through `fit` (receipt, RUN table, wm_s_best payload, stage checkpoints);
  * mixed rate: 20000 draws within 0.3 +- 0.015, and in fit 600 long-stage batches, only in stages >= 30 s, with the right unroll lengths;
  * a real-compute fit with mixed and multi on;
  * the v4 config diff and round trip.
* `OMP_NUM_THREADS=1`: test_sched + test_wmloss + test_lanefix + test_c4_core + test_import_graph
  **269 passed**; `tests/unit -k "train or config or r5"` **300 passed** (764 deselected).
  ruff clean on touched files.
* CPU smoke `tools/c4_smoke.py E:/agent-work/tmp/sched_smoke_v4sched configs/c4_wm_v4_sched.toml`
  (data via `wmlane_smoke.point_data(E:/research-dense)`): **passed**, 2 optimizer steps, finite,
  loss 14.48 -> 11.23, 583 TRAIN / 99 val windows. Note the dry run collapses to one 10 s stage,
  so it checks that the config assembles and trains; the SCHED loop paths (validation horizons,
  selection, mixed batches) are covered by the real-compute unit fits on synthetic data.
