# R2 / Path 1, item A2+A3: entries v2 (observed-history vehicle entries)

Status: **DONE (code + synthetic tests + CPU diagnostic), flag-gated default OFF.**
Branch `wms/entry2` (parent `wms/integrate` 59ba48b). TRAIN-only, CPU, one heavy
process, no SUMO/GPU, no sealed data.
Commits: `b8d80b7` (entries-v2 arrivals + trainer/CLI wiring) and `57101ec`
(per-step entrant mask + section ids interface for `wms/trafloss`).
Scope is A2+A3 only; A4 (merge_cal) and B1/B2 (traffic losses, 300 s curriculum)
are separate items (B1 on `wms/trafloss`).

## Design

New entrants are generated only from the RSU-observable **history before t0**;
no recorded future arrival is ever read.

1. **Arrival count -> recency rate with trend.** `_recency_trend` reads the most
   recent `window_bins=12` ten-second bins of the t0 `[2,30]` entrance counter
   (bin 0 most recent), applies a decaying (half-life 4 bins) weighted mean and a
   decayed linear fit, and takes the mean of the weighted mean and the one-bin
   extrapolation.  `rate` is veh/s; `trend` is the bounded (0.25..4) one-bin
   ratio.  Older/future bins cannot leak (tested).
2. **Arrival process -> gamma renewal.** `TorchEntryProcessV2` samples
   inter-arrival times from a gamma renewal fit to the observed pre-t0 entry
   times (`shape = 1/cv^2`, mean `1/rate`).  Bursty/platoon history gives a high
   observed CV and quantifiably higher arrival dispersion than a pure Poisson
   (tested).  History can come from an explicit `EntranceHistory`, from observed
   `times/entrance`, or from the t0 counter alone.
3. **How they enter (empirical, shrunk to defaults).** Per arrival: entry lane
   (observed mainline lane-1 share, ramp -> lane 2), entry speed = observed
   entry/local-speed **ratio** times the *current* local traffic speed at the
   entrance, entry gap from the observed distribution, and class mix (CAV share
   and HDV vType lengths).  Every empirical quantity is shrunk to network
   defaults by `n_obs/(n_obs+shrink_k)`, `shrink_k=8`, so short history falls
   back safely.
4. **Seeded, common random numbers.** One `np.random.default_rng(seed)` per
   process; the same seed + history yields byte-identical entrants (tested), so
   compared controllers/models see the identical entrant stream.
5. **Trainer (`--entries-v2`, default off).** `TrajectoryDataset(entries_v2=...)`
   attaches a per-window `entry_feat` vector built from the window's t0 counter
   and its pre-t0 `veh_kin`/`veh_mask` ticks (`_entry_features_from_corpus`;
   first-appearance slots = observed entrants).  `WmSTrainer._entry_process`
   turns each feature row into an `EntranceHistory` and a seeded
   `TorchEntryProcessV2` over `entry_pad` free padding slots.  `_extend_for_entries`
   appends the padding columns: entrants occupy them, move by physics (+
   residual) and influence neighbours, but `loss_valid` is False there so they
   are **excluded from the per-car loss**.  `--entry-pad` (default 8) sets the
   pool size; `--entries-v2` also selects the v2 process for `wms_eval --entries`.
6. **Clean interface for `wms/trafloss` (B1).** `WmSModel.rollout` now also
   returns, for every compared call:
   - `entrant_seq` `[B, n_steps, K]` bool - **per-step entrant membership**
     (marked from the insertion step onward; all-False when `entries=None`), so a
     traffic-level loss can include entrants while the per-car loss excludes them;
   - `sections` `[B, n_steps, K]` int64 - **model-frame section id** per slot per
     step (0 upstream / 1 merge / 2 downstream, -1 absent), identical to
     `wms_eval.metrics._section`, plus the pure helper
     `wms.entries.section_ids(s_m, present=None)` (numpy and torch).
   Entrant membership is now carried across truncated-BPTT chunks via the
   `state["entrant"]` key, so a `wms/trafloss` loss over a chunked unroll sees
   the full entrant set.

The v1 process (`--entries`, `TorchEntryProcess`) is unchanged and remains the
default-OFF behaviour; `--entries-v2` implies `--entries`.

## Files

- `output/wd01/wm_n1/wms/entries.py` - v2 history builder (`build_entrance_history`,
  `EntranceHistory`, `_recency_trend`, `read_insertion_history`),
  `TorchEntryProcessV2`, and `section_ids`/`SECTION_*`.
- `output/wd01/wm_n1/wms/model.py` - `rollout` returns `entrant_seq`/`sections`,
  carries `entrant` across chunks (`_section_ids` import).
- `output/wd01/wm_n1/wms/train.py` - `entry_feat` dataset path, `_entry_process`,
  `_extend_for_entries`, `entries_v2`/`entry_pad` config.
- `output/wd01/wm_n1/wms/cli.py` - `--entries-v2`, `--entry-pad`.
- `output/wd01/wm_n1/wms_eval/models.py`, `cli.py`, `wms_plugin.py` -
  `--entries-v2` flag, `entries_v2` rollout flag, plugin v2 process and summary.
- `output/wd01/wm_n1/wms/tests/test_wms_entries.py`,
  `output/wd01/wm_n1/wms/tests/test_wms.py`,
  `tests/test_wms_eval_wms_plugin.py` - tests.

## Tests

- `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` (rl_driving, torch 2.12.1+cpu):
  - `wms/tests` + `tests/test_wms_eval_wms_plugin.py`: 102 passed, 2 skipped
    (pre-existing TRAIN chunk-4 parity skips).
  - `tests/test_wms_eval_{seen,run1_report,paths,metrics,confinement,c1}.py`,
    `tests/test_wms_decide.py`: 40 passed.
  - `ruff check` clean on all changed files.
- New synthetic coverage: recency window only / no older-bin leakage; rising vs
  falling trend; gamma burstiness dispersion; shrinkage to defaults with short
  history; empirical headway CV; determinism + common random numbers; padding-pool
  confinement; observed CAV/vType entry attributes; `read_insertion_history`
  drops future rows; **entry_feat from observed history only**; trainer padding
  excluded from the per-car loss (`valid_targets` unchanged on vs off);
  **per-step `entrant_seq` monotone and equal to cumulative on the last step, all
  False when off, carried across chunks**; `sections` -1 exactly where absent and
  matching `wms_eval.metrics._section`; CLI flag wiring default off / v2 implies
  v1.

## CPU diagnostic (5 TRAIN dev states, run-1 checkpoint, 300 s, `_out/entry2_cpu.py`)

Read-only run-1 checkpoint + ws-eval dev states; pooled merge-section means
(`_out/entry2_cpu.json`):

| stream   | queue m | stopped | delay s | mean speed m/s | flow veh/min |
|----------|---------|---------|---------|----------------|--------------|
| recorded | 8.90    | 1.25    | 1996.6  | 8.95           | 44.92        |
| v1       | 3.28    | 0.52    | 858.2   | 16.54          | 42.24        |
| v2       | 2.23    | 0.34    | 665.1   | 19.45          | 36.32        |

Per-state inserted (v1 / v2 / recorded), mainline+ramp: w8676 283/156/260,
w8626 199/158/246, w8629 190/171/200, w8678 263/169/249, w8677 256/168/239;
`no_slot=0` for both. v2 rates ~0.46-0.72 veh/s mainline with 39-43 ramp
observations. Diagnostic only (3-5 states, one checkpoint), **not efficacy
evidence**.

## Open questions

1. **v2 under-inserts vs recorded** (156 vs 260 for w8676) with `no_slot=0`.
   The empirical entry gap (item 3) is imposed *in addition to* the SUMO-like
   safe gap, and the per-step drop cap (`max_pending=6`) then discards deferred
   arrivals. Candidate fix: treat the observed gap as the *desired* gap and admit
   when either condition holds, or raise/remove `max_pending`; needs an A2
   trainer dry run and a new comparison against recorded, not just the 5-state
   diagnostic.
2. **No real-data trainer run yet.** A2 is validated on synthetic windows only;
   the `--entries-v2` trainer path has not been exercised on a real TRAIN pack
   (needs a small, explicitly-budgeted CPU mechanics run).
3. **Traffic-level loss not added** (B1, `wms/trafloss`): this branch only
   exposes `entrant_seq` + `sections`; the merge must consume them.
4. **A4 `merge_cal`** is not wired into the entries-v2 trainer path (flag only).
5. **B2 300 s curriculum** and GPU training are out of scope here.
6. Whether v2's lower merge queue than v1 (2.23 vs 3.28; recorded 8.90) is due to
   fewer entrants or different entry speeds is not separated by the current
   diagnostic; the entrant-count ablation answers it once (1) is settled.
