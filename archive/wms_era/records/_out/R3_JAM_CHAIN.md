# R3_JAM_CHAIN — ROUND-3 jam-window oversampling + training chain + gpuB pack_v3 subset

Status: COMPLETE (code/data prep only; no GPU, no SUMO, no training).
Branch: `wms/r3-jam`. All new behaviour is behind flags that default OFF.

## (A) JAM-WINDOW OVERSAMPLING (`wms/jamdata.py`, `wms/mixdata.py`, `wms/cli.py`)

New flags (default OFF = byte-identical path):
`--jam-oversample K` (default `1`), `--jam-speed-mps V` (default `5`),
`--jam-cache-dir DIR` (default: the run `--out` directory).

* A *jam window* is a training window whose **recorded HDV speed** (present,
  non-CAV slots of `states[..., 4]`) drops below `V` somewhere in its own
  recorded horizon. One window is read at a time from the memory-mapped A11
  packs; `K == 1` never scans and never writes a sidecar.
* Per-window flags + counts are computed once and cached in a sidecar JSON
  (`wms_jam_windows/1`): `speed_mps`, `source_key`, `n_windows`, `n_jam`,
  `n_hdv_windows`, `n_hdv_scanned`, `min_speed_mps`, `flags[]`. A matching
  cache is reused without rescanning.
* Sampling composes **multiplicatively** with the existing mix:
  `w = where(event, event_oversample, 1) * where(jam, jam_oversample, 1)`,
  then normalised inside each source and scaled by its `--switch-mix-ratio`
  share. Without `--switch-pack`, `JamOversampledDataset` supplies the same
  per-stage weights to the trainer's `WeightedRandomSampler`.
* Counts are recorded in `TRAIN_RECEIPT.json` under `data["jam"]` (and per-source
  `n_jam_items` / `jam_fraction_uniform` in `data["stage_sampling"]`).
* `K == 1` produces exactly the pre-fix `stage_weights` (unit test
  `test_jam_off_weights_are_bit_identical`).

## (B) ROUND-3 TRAINING CHAIN (`_out/train_r3_chain.sh`)

Derived from `_out/train_main_chain.sh`; keeps its guards and adds the box
patches as env opt-outs:

| env | default | meaning |
|---|---|---|
| `SMOKE_CONTENTION_GUARD` | `1` | `0` disables the smoke CPU-contention enforcement |
| `RSS_GUARD_BYTES` | `6000000000` | per-process RSS cap (bytes; `0` disables) |
| `MIN_FREE_OVERRIDE` | `10737418240` | replaces the 10 GiB min-free requirement |

`STABILITY_WEIGHT` / `TRAFFIC_LOSS` / `CONSERVATION_*` / `PROJECTION_ENABLED` /
`RESIDUAL_CAP_MPS2` / `LANE_LOGIT_CAP` / `RESIDUAL_CAP_NEG_MPS2` are
env-configurable and default OFF; the gate `--require-terms "$TERMS"` follows
exactly the enabled terms (base `position,speed,lane`; `traffic` when the
traffic loss is on; `stability`/`conservation` when their weights are > 0;
`asym_speed`/`decel_weight`/`propagation` when braking is enabled).

ROUND-3 recipe (= C1FAST + additions): `--start-shift-prob 0.75`,
`--switch-mix-ratio 0.3`, `--event-oversample 1`, stage steps
`2500/2500/1875/1250`, unroll `10/30/60/120`, no residual cap / projection /
traffic / stability / conservation losses, `--merge-cal` with
`_out/merge_params.json` (`MERGE_CAL=1`, toggle with `MERGE_CAL=0`).
pack_v3 windows come from `_out/lpwo_v3/windows_{train,val}_noPI_noWall.json`
with `_out/sidecars_packv3_v2/{identities,outcomes}`; `--leave-pi-wall-out` is
OFF (so the 0.75 start-shift applies; the lpwo_v3 lists are already PI/wall-free).

Presets (`R3_PRESET`):

| preset | braking losses |
|---|---|
| `R3_FULL` (default) | `--asym-speed-loss 2 --decel-weight 5 --decel-threshold 3 --propagation-loss 1 --prop-window 3` |
| `R3_CTRL` | all braking off (`1/1/0`) |

After-each-stage hook: the chain exports `WM_S_STAGE_CKPT_COPY_DIR=$W/stage_snapshots`;
`WmSTrainer.fit` copies every saved `wm_s_stage<S>.pt` to that directory as
`wm_s_stage<S>_<timestamp>.pt` so stages can be scored while training continues
(unset env == no copy, byte-identical run).

Pass-through hooks for the parallel branches (each adds its CLI flag only when
the env var is set; a branch that lacks the flag fails the HELP guard):
`R3_RESIDUAL_SOFT_BOUND_MPS2`, `R3_RESIDUAL_MAG_WEIGHT`,
`R3_LANE_RATE_WEIGHT`, `R3_LANE_RATE_TARGET`, `R3_SELECT_BY_LONG_HORIZON`,
`R3_HISTORY_S`, `R3_HISTORY_STRIDE_S`, plus this branch's `JAM_OVERSAMPLE`,
`JAM_SPEED_MPS`, `JAM_CACHE_DIR`. `--entries-v2` is an env toggle
(`ENTRIES_V2`, default `0` = OFF).

Preset sanity (TERMS): `R3_FULL -> position,speed,lane,asym_speed,decel_weight,propagation`;
`R3_CTRL -> position,speed,lane`.

## (C) gpuB PACK_V3 SUBSET (`D:\rd-staging\pack_v3_r3_subset\`)

Built by `_out/r3_packv3_subset/build_subset.py` from the read-only full pack
`D:\rd-staging\wmn1_feat\pack_v3` (memory-mapped; no source byte modified).
Keeps only the **341 runs** referenced by the lpwo_v3 `pack_v3` train/val
windows, in the same format/manifest semantics.

* free space before: `45,921,153,024` bytes (initial read `36,714,307,584`);
* subset total: **`9,204,824,672` bytes (8.57 GiB)** — 567,341 ticks, 6,820
  decisions, 22,839 outcome rows;
* per-array shapes e.g. `veh_step[567341,192,16]`, `veh_kin[567341,192,7]`,
  `plan[6820,192,19,8]` (all dtypes unchanged);
* `manifest.json`, `INDEX.json`, `outcomes.parquet`, `SHA256SUMS`,
  `identities/` (a position-remapped, strict-loadable sidecar bound to the
  subset manifest), and `_out/r3_packv3_subset/SUBSET_RECEIPT.json`.

Verification (`_out/r3_packv3_subset/verify_subset.py`, CPU): the loader keeps
**identical pack_v3 windows** from the subset and from the full pack — 294
train / 47 val; 4 sampled windows decode identically; 22,839 outcome rows match.
For gpuB, set `PACK_V3` to the subset, `IDENTITIES_V3` to the subset
`identities/`, and use the pack_v3-only lpwo lists
`_out/r3_packv3_subset/windows_{train,val}_pack_v3.json`.

Note: `_out/sidecars_packv3_v2/{identities,outcomes}` (the ~16 MB full-pack
sidecar bundle named by `PACKV3_FIX.md`) were restored into this worktree from
the sibling `ws-packv3fix` worktree for the verification; they are git-ignored
and must be staged to the box (or overridden) by the operator.

## Tests / lint

* `python -m pytest output/wd01/wm_n1/wms/tests` — **479 passed, 6 skipped**
  (TMP/TEMP and `--basetemp` inside the worktree).
* New `test_wms_jam.py` (10 tests: HDV-speed extraction, counts + cache
  round-trip + invalidation, flag-shape refusal, OFF-path bit-identity,
  event*jam*share composition, wrapper, stage-snapshot hook, parser defaults).
* New `test_r3_chain.py` (6 tests: LF-only, guards/box opt-outs, preset braking,
  R3 data/flags, jam + pass-through hooks, required-terms-follow-enabled).
* `python -m ruff check output/wd01/wm_n1/wms output/wd01/wm_n1/wms/tests` —
  All checks passed.

## Constraints honoured

No GPU / SUMO / training; CPU dry-runs and tiny unit tests only. Packs and runs
read-only, memory-mapped. Writes only inside `E:\rd-wt\ws-r3-jam` and the new
staging dir `D:\rd-staging\pack_v3_r3_subset`. CAL / HOLDOUT / sealed / chunk-5 /
J4B / INC1 were never opened. No processes killed; RAM bounded.
