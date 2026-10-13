# PAIRLOSS - GENRAND data hookup + paired-difference loss (branch c4/pairloss, 2026-10-14)

Owner-approved 2026-10-14 ~03:20. Code, tests and a CPU smoke only. The integrator launches the
GPU run. Base: c4/int 0be5b72. Defaults are OFF and bit-identical.

## 1. Multiple C4 sources (`[[data.c4_sources]]`)

- `src/wms/config/data.py` adds `C4SourceConfig`, used as `data.c4_sources = (...)`. Fields:
  `name`, `path`, `mix_ratio`, `window_stride_s`, `start_s`, `use_val`, `paired`.
  - `path` is a logical name resolved through the registry, so `genrand_plain` resolves via
    `WMS_DATA_GENRAND_PLAIN` and `genrand_branch` via `WMS_DATA_GENRAND_BRANCH`. Default local
    paths are registered in `configs/data_registry.toml`.
  - `mix_ratio` is the sampling share of the source's ordinary windows. It may be 0 only for a
    paired source.
  - Validation: names must be unique and the ratios (plus `data.c4_episodes`) must sum to < 1.
  - The older `data.c4_episodes` path is unchanged.
- `wms.cli.assemble._add_c4_sources` mixes the sources after `c4_episodes` using
  `wms.data.c4_pairs.add_c4_sources`. Each source gets exactly its share. Existing sources keep
  their relative shares, scaled by `1 - sum`. Validation windows are appended as `<name>_val`.
- **Split by world:** `world_roles` gives each world ONE side across every C4 manifest. A world
  that is validation in any source is validation everywhere; overrides are listed in
  `receipt.c4_world_split`.
  - Within one converted dir the converter already groups by world, so a base episode and its
    branches share a side (checked on the driver smoke dir: w4110 base + branches = val).
  - GENRAND plain seeds (961000-961479) and branch seeds (961480-961515) are disjoint worlds.
- **v3/v4 switches on C4 windows:**
  - `entries_v2`: entry features and recorded road presence, unchanged.
  - Aggregate targets: `road_present`.
  - `junction_fix = "ffill"`: the same fill as the pack loader (`C4WindowLoader._fix_junctions`):
    lane forward-filled, s interpolated, the transit step still loss-excluded, and a t0
    junction vehicle with no on-lane row dropped. Before this, C4 windows ignored the switch.
  - `controller_hygiene = "strict"`: designed controllers in C4 data are still refused (fail
    closed). GENRAND is native + direct only, so it is kept, and its windows carry
    `command_known = True`, because every issued command is recorded. Without this flag, mixed
    batches dropped the key.
  - `lane_model` hazard / `lane_event`: realized lane changes are in the states. CAV lane
    commands are not a WM input.
- **CAV teacher:** confirmed. The C4 records' `cav_teacher` is the REALIZED SUMO motion
  (`veh_kin` s / lane / v / a). The issued command is only in `action_timeline`; the
  test_c4_genrand test checks that the two differ. `cav_drive` stays `"teacher"`: no config sets
  it, and `prepare_inputs` feeds `batch["cav_teacher"]`.

## 2. Paired-difference loss (`[loss.paired_diff]`, default weight 0 = off)

**Batches (`wms.data.c4_pairs`).**
- `PairedBranchSource` builds one window per arm of the paired source, starting at the branch
  time (`warmup_s` of a branch episode).
- The horizon is `min(global horizon, branch length)`, which is 120 steps for 60 s branches.
  The trainer's `prepare_inputs` already caps `n_steps = min(stage unroll, T-1)`, so a paired
  batch unrolls for the stage horizon capped at 60 s.
- Groups are formed by `pair_key`. The reference is the `B` (all-native) arm, else the first
  arm. A group is dropped, and counted in the receipt, if it has fewer than 2 arms or if its
  arms disagree on the t0 `vid` / `veh_kin`. Identical t0 slots mean slot k is the same
  vehicle in every arm.
- `PairedBatchDataset` wraps the final TRAIN dataset and delegates all attributes
  (`stage_weights`, `receipt`, ...). It implements `__getitems__`, which torch's fetcher calls
  with the whole batch of sampled indices, so `trainer.py` is untouched.
- With probability `fraction` the batch is replaced by a paired batch of `groups` groups, with
  `min(max_arms, batch_size // groups)` arms each, reference always included. The draw is
  seeded from the indices, so it is reproducible and leaves the global RNG alone.
- Records carry `pair_group` / `pair_ref`, collated to `batch["pair_group"]` / `["pair_ref"]`.

**Loss (`wms.train.losses.paired_diff`).** The three chunk terms are appended after every
existing term:
- `pair_pos` / `pair_speed`: `huber((p_a - p_r) - (x_a - x_r), delta) / delta` per t0 HDV and
  step where the vehicle is supervised in BOTH arms (`hdv_mask`). Steps are weighted like the
  per-car terms and normalised over the whole unroll.
- `pair_agg`: per 5 s bin, on-road delay (whole road and RSU zone) and soft throughput past the
  merge end, using the aggregate loss's `soft_cell_fields` / `soft_crossings` and its reward
  normalisation. The t0 cohort is used: entrants are drawn per row, so their noise would not
  cancel in a difference. When the reference is predicted exactly, this equals `agg_reward` of
  the arm (unit test).
- On ordinary batches the terms are a graph-connected zero, and log keys are always present.
- The log record gets `paired_value_share`, `paired_batch` and `paired_arms`.
- The model sees CAV actions only through the realized CAV teacher. Nothing new is a model
  input.

**Plumbing.** `PairedDiffLossConfig` in `LossConfig.paired_diff`, legacy flat
`paired_diff_*` (omitted when default), `TrainSettings` validation, `required_log_terms`,
composer, `UnrollTargets.pair`, `build_targets`. `validate_for_training` refuses weight > 0
without a `paired = true` source.

## 3. Config `configs/c4_wm_v4_data.toml`

- Extends `c4_wm_v4_lane.toml`, with `controller_hygiene = "strict"` and
  `junction_fix = "ffill"`.
- `genrand_plain` has mix 0.30. `genrand_branch` is `paired = true` with mix 0.05; that 0.05
  covers only its full-horizon ordinary windows, i.e. base episodes if they were converted into
  the branch dir, and is skipped if none exist.
- `[loss.paired_diff]`: weight 0.5, fraction 0.25, groups 3, max_arms 5. With batch 16 this
  means 3 groups x 5 arms = 15 rows.
- Ordinary-window GENRAND share is 35 %. Counting the 25 % paired batches, about 51 % of the
  sampled windows are GENRAND. Lower `fraction` or `mix_ratio` if that is too much.
- `runtime.out_dir = runs/active/c4_wm_v4_data`. The integrator adds the c4/sched schedule.

## 4. Verification

- `tests/unit/test_pairloss.py`, 13 tests:
  - off = bit-identical training step (record + grads), even with pair keys in the batch;
  - on: per-car terms unchanged, pair terms finite/positive, share in (0, 1), zero on ordinary
    batches;
  - zero when the predicted differences equal the recorded ones (identical arms; a common
    offset), positive and differentiable otherwise, with gradient into the reference rows;
  - the aggregate difference equals the aggregate code's `agg_reward`;
  - world-role rule; multiple sources load, split by world, and keep exact shares and
    per-source stage-weight sums; `command_known` is set;
  - loader defaults are unchanged by the new options;
  - a paired batch has identical t0 state / hist / driver_hist / entry features across arms,
    differing realized CAV teachers after t0, and direct vs native timelines;
  - the trainer's DataLoader path reaches `__getitems__`;
  - paired draws are reproducible and their frequency matches `fraction`.
- Suites, run with `OMP_NUM_THREADS=1`: see the commit message / final report for counts.
- **CPU smoke:**
  - Command: `_scratch/run_smoke.py` = `tools/c4_smoke.py configs/c4_wm_v4_data.toml`, with
    packs via `wmlane_smoke.point_data(E:/research-dense)` and the GENRAND smoke outputs
    (`plain_conv`, `branch_conv`).
  - `tools/c4_smoke.py` now also runs 2 updates on one paired batch and reports the windows per
    source.
  - Results are in section 5.

## 5. Smoke results

(filled in below)
