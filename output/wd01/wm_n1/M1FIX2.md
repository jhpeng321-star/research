# M1FIX2 - training-quality fixes before member-1

Branch `zcode/m1fix2` (base `649be00` = `zcode/boxfix`), work dir
`output/wd01/wm_n1`. Zero SUMO, zero real training; the only real-pack access
was ONE read-only CPU measurement on `pack_v3` TRAIN (no CAL rows, no writes to
D:).

## What changed and why

1. **Target normalization (item 1).** Outcome targets
   (`delay_s = total_delay_ref_s`, `ttc_lt3_s = ttc_lt3_veh_s`,
   `hard_brake_s`) are raw, so CRPS/contrast were dominated by scale
   (`correction_crps = 18065` vs `residual_nll ~ 1.1`). `model/outcome_norm.py`
   fits a robust per-(tau, target) location/scale (median / IQR, std fallback,
   `min_scale=1e-3`) from **TRAIN items only** via
   `train.dataset.MemmapCorpus.outcome_target` (reads only the packed outcome
   rows). Training applies it to the outcome targets of `correction_crps` and
   the outcome part of `paired_response_contrast`; predictions are
   de-normalized through the single helper
   `model.outcome_norm.denormalize_quantiles` at eval (`evalh.idea_a._outcome`,
   `wm_api.engine.outcome_head`). The stats ride inside the checkpoint
   (`outcome_norm`) and are written as `OUTCOME_NORM.json` next to
   `member_<k>.pt`; its sha256 is in `TRAIN_RECEIPT.json`. CAL/TEST are never
   read for fitting.

2. **State global token (item 2).** The correction head received
   `global_token = zeros`. `corr/state_token.py:StateGlobalToken` builds the
   token from STATE only: masked mean+max pooling over the t0 `veh_kin` slots
   (7-d kinematics, no CAV flag) -> MLP, concatenated with the entrance-counter
   features -> MLP -> `d_model`. Wired through
   `IdeaATrainer.global_token` for training, `wm_api.engine.outcome_head`, and
   `evalh.idea_a._outcome`. Config flag `state_global_token` (default `True`);
   `False` reproduces the old zeros path exactly (`--no-state-global-token`).
   The label-leak test is extended (`corr/tests/test_corr_interface.py`,
   `train/tests/test_m1fix2.py`).

3. **Loss balance (item 3).** The first `loss_balance_batches` (default 4)
   batches record the five term magnitudes; `loss_balance_report` (leave-one-out
   median, factor 100) is written to `TRAIN_RECEIPT.json` under `loss_balance`.
   It only WARNS; lambdas are unchanged.

4. **Throughput (item 4).** `train.dataset.PairAwareBatchSampler` +
   `epoch_batches` give a deterministic pair-aware `batch_sampler`;
   `IdeaATrainer._iter_batches_loader` uses a `torch.utils.data.DataLoader`
   (persistent workers, prefetch, `pin_memory` only on CUDA). `--num-workers`
   (default `min(8, cpu-2)`; `0` = in-process) is exposed on `train.idea_a` and
   in `bundle.timing.GateConfig`. The `IdeaACorpus.window` per-t Python loops
   were vectorized (contiguous memmap slices + a `cummax` forward-fill); a
   parity check confirms the old/new windows are bit-identical (atol 1e-5).

5. **Timing gate fail-closed (item 5).** A non-tiny `run_timing_gate` with no
   `pack_dir` now raises (`measure_idea_a_corpus` / `loss_decrease_idea_a`);
   it never silently measures a synthetic pack. `pack_source` and
   `num_workers` are recorded in the gate and the queue stage.

DataLoader/in-process batch parity is tested; in-process batching is unchanged
in coverage (every item seen once per epoch, one completed pair per batch).

## Measured evidence

### Normalized first-batch loss magnitudes - real `pack_v3` TRAIN (read-only)

CPU, 4 threads, BelowNormal, `batch_size=8`, first epoch batch 0, member-1
config (`build_idea_a_bundle`). `fit_outcome_norm` over the 2153 TRAIN items
took 7.4 s; `OUTCOME_NORM.json` sha256
`e8bc14bc5bf46636d2613489c7df418ef56737816778414aac36e4b839cd4537`.

| term | before (raw targets) | after (normalized) |
| --- | --- | --- |
| correction_crps | 18065 | 36.11 |
| response_contrast | 159 | 8.06 |
| residual_nll | ~1.1 | 1.199 |
| rollout | ~0.21 | 0.224 |
| arrival | ~0.95 | 0.896 |

`loss_balance` receipt: `balanced=true`, no term exceeds 100x the leave-one-out
median of the others (max ratio 34.5, `correction_crps`). The raw max/min
pairwise ratio is now 161x (correction_crps / rollout) versus ~16000x before;
the registered acceptance is the 100x-others-median receipt check.

### CPU s/batch before/after

| pack | old | new | speedup |
| --- | --- | --- | --- |
| real `pack_v3` TRAIN (K=192, h=20) | 33.43 | 0.113 | 296x |
| synthetic pack (K=192, 96 items) | 0.0878 | 0.0535 | 1.64x |

The real-pack gain is dominated by no longer copying the whole
`accel_next`/`target_mask` arrays into every window; the vectorized
forward-fill is the rest. Old/new windows are bit-identical.

### Test counts (this checkout)

| suite | passed |
| --- | --- |
| train/tests | 122 |
| bundle/tests | 152 |
| evalh/tests | 98 |
| corpus/tests | 80 |
| wm_api/tests | 34 |
| inc1eval/tests | 36 |
| corr/tests + model/tests | 46 |
| `python -m preflight all --data-root E:/research-dense` | 6/6 PASS |

## Exact box commands

```
python -m train.idea_a --member 1 --seed 101 --corpus <PACK> --out <RUN> \
    --device cuda --k-max 192 --num-workers 8
# reproduce the pre-M1FIX2 zero global token:
python -m train.idea_a --member 1 --seed 101 --corpus <PACK> --out <RUN> \
    --device cuda --k-max 192 --num-workers 8 --no-state-global-token
# tests
python -m pytest train/tests bundle/tests evalh/tests corpus/tests \
    wm_api/tests inc1eval/tests corr/tests model/tests -q
python -m preflight all --data-root E:/research-dense
```

## Files

New: `model/outcome_norm.py`, `corr/state_token.py`,
`train/tests/test_m1fix2.py`, `M1FIX2.md`.
Changed: `train/idea_a.py`, `train/dataset.py`, `corr/__init__.py`,
`corr/tests/test_corr_interface.py`, `train/tests/test_idea_a_checkpoint.py`,
`evalh/idea_a.py`, `wm_api/checkpoint.py`, `wm_api/engine.py`,
`bundle/timing.py`, `bundle/queue.py`.

`inc1eval` consumes trajectories only (no outcome head / `corr`), so it has no
prediction to de-normalize; the shared helper is available for any future
outcome consumer.
