# GPUFIX - GPU-path fixes from the box run on `bc6a51b`

Branch `zcode/gpufix` (base `bc6a51b`; resumes the dead executor's WIP
`47e9e04`). Work dir `output/wd01/wm_n1`. **Zero SUMO. Zero real training.**
CPU <= 4 threads, BelowNormal. The only real-pack access is read-only TRAIN on
`D:/rd-staging/wmn1_feat/pack_v3` (no CAL/HOLDOUT/chunk-5/J4B, no writes to
`D:`, `WD_FORBID_D=1` for every test command). This laptop has no GPU, so the
device fixes are correct **by construction** and the CUDA paths are pinned by
`meta`/strict-device tests.

## (A) Timing-gate crash `Expected all tensors on the same device`

### Cause

`model/rollout.py:_rollout_once` built the lane sentinel with a host
`torch.full((B, K), -1, dtype=torch.long)` (old line 239) and then
`torch.where(cond_cuda, argmax_cuda, sentinel_cpu)`; on `cuda:0` the host
sentinel raised. Reached from `bundle/timing.py:measure_throughput` ->
`model_rollout` (old line 290). The arrival/insertion stacks
(`torch.tensor(np.stack(...))`) and `bundle/timing.py:make_scene`
(`torch.zeros(B, K, C.VEH_F)`) had the same defect.

### Fix (device-correct by construction)

* `model/rollout.py`
  * `torch.full(..., device=veh.device)` (line 253);
  * `arrivals`/`insertions` `torch.tensor(..., device=s.device)` (306-307);
  * `_uniform` / `_sample_gmm` generate uniform/Gaussian noise on the tensor
    device (a CPU generator is moved with `.to(device)`);
  * `rollout()` derives the default generator device from the net parameters,
    falling back to CPU for `meta` (no generator device) - lines 320-328.
* `bundle/timing.py` `make_scene` / `make_targets` / `r()` all pass
  `device=device`.
* `arrival/sampler.py` count/speed/insertion tensors follow the head output /
  `present.device`.
* `rollout2/step.py:_kin_features` `torch.zeros(..., device=s.device)`.
* `hdv/net.py` `zero_residual` lane logits already used `kin.device`;
  `hdv/input_schema.py` perturbation mask now uses `veh.device`.
* `rollout2/engine.py` `present`/`wall_group`/`slot_id` fallbacks now use
  `kin.device`.
* `inc1eval/incident_ctrl.py` `r1/r2/upstream_select` obstacle tensors now use
  `s.device`.
* `train/idea_a.py` `global_token` zero fallback uses `self.device`;
  `_pairs` builds the contrast indices on the batch device; `_to_device` is
  now recursive over dict/list/tuple (`_move_to_device`).

### Audit table (every `full/zeros/ones/arange/tensor/as_tensor/empty`)

| file:line | status |
| --- | --- |
| `model/rollout.py:253,306,307` + samplers | fixed |
| `bundle/timing.py:170-204` | fixed |
| `arrival/sampler.py:49,61,92` | fixed |
| `rollout2/step.py:63` | fixed |
| `rollout2/step.py:345` `as_tensor(counter)` | rollout2 is the CPU/numpy reference engine; counter already carries its device. Referenced |
| `rollout2/engine.py:192,208,210` | fixed |
| `rollout2/state.py:143-155` | explicit `device=cpu` (CPU engine) |
| `rollout2/equivalence.py`, `smoke.py` | analysis/CPU tooling, no model op |
| `hdv/net.py:257` | already `device=kin.device` |
| `hdv/input_schema.py:58` | fixed |
| `arrival/head.py:55` | already `device=` parameter |
| `corr/head.py:73` | already `device=pred.device` |
| `corr/ood.py:25,54,57` | device follows tensor input; model stats moved to `s.device` |
| `model/losses.py:96,129` | already `device=quantiles/q_a.device` |
| `model/outcome_norm.py:141,142` | already `device=ref.device` |
| `model/net.py:170` | `nn.Parameter` init; moved by `.to(device)` |
| `train/idea_a.py:674-796` window/data-prep | CPU Dataset/worker data prep by design; moved by `_to_device` |
| `train/idea_a.py:1362` | fixed |
| `train/idea_a.py:1499,1617` | already `device=s/qa.device` |
| `train/idea_a.py:1563` `_pairs` | fixed |
| `wm_api/engine.py` (`build_world`, `wall_speed_tensor`, `_actions_for_tick`, `_state_kin`) | CPU float64 vectorized engine; `wm_api/checkpoint.py` loads on `map_location="cpu"` then `.to(float64)`, so device-correct by construction |
| `inc1eval/incident_ctrl.py:98,114,126` | fixed |
| `inc1eval/incident_ctrl.py:168-192`, `scripted_mover.py:158-162` | already `device=dev/inp.s_m.device` |

### Pinning tests

* `test_forward_loss_on_meta_device_has_no_host_tensors` (forward + GMM loss on
  `meta` params/inputs),
* `test_timing_gate_measurement_has_no_host_tensors` (`measure_throughput`
  train step + rollout under a strict-device factory guard; module init
  excluded),
* `test_one_fit_step_has_no_host_tensors` (full `fit`, five losses + backward
  + `train_log.jsonl` under the same guard),
* `test_rollout_lane_where_full_is_on_the_vehicle_device`.
  The strict guard exempts torch-internal caller frames (optimizer state
  scalars) and reports the offending caller file for any host tensor reached in
  our code.

## (B) `residual_nll` / `rollout` exactly `0.0` on the box

Two compounding defects produced exactly-zero HDV terms:

### B1. DataLoader corpus serialization (spawn workers)

`train/idea_a.py:IdeaACorpus` and `train/dataset.py:MemmapCorpus` cached open
`np.memmap` arrays (and `MemmapCorpus` a pandas frame). `_iter_batches_loader`
calls `_source_item_meta(source)` and `_collate_for_source(source)` - both
touch `source[0]` - **before** it constructs the `DataLoader`, so the cache is
populated when the workers are spawned. A spawned worker pickles the dataset:
`np.memmap` pickling copies the whole array, so a real `pack_v3` feed
(`veh_step.npy` = 19.6 GB, `cav_plan_ch.npy` = 9.8 GB, ...) is serialized.
Reproduced on CPU (Windows spawn) with the pre-fix code:

```
OSError: [Errno 22] Invalid argument          # reduction.dump
_pickle.UnpicklingError: pickle data was truncated
```

On a fork start method the inherited memmaps survive, so the failure is
platform-dependent - consistent with the box's 8-worker feed differing from the
laptop's in-process CPU path. Fix: `__getstate__` drops `_mm` (and
`_outcome_rows`); workers re-open lazily on first access. Pickle size after the
fix is the metadata-only corpus (1.15 MB untouched / item 149 KB) instead of
multi-GB.

### B2. Fail-open empty mask

`_masked_mean` returned `(x*m).sum() / m.sum().clamp(min=1.0)` = **exactly
`0.0`** for an empty mask. `residual_nll` and `rollout` are the only A3.2
terms that can be exactly `0.0` this way. Fix: when `_require_all_terms` is set
(a real, fail-closed corpus run), an empty `residual_nll` mask or an all-empty
`rollout` HDV mask raises `IdeaAConfigError` instead of training on `0.0`.
Tests: `test_residual_nll_empty_mask_fails_closed_on_a_real_run`,
`test_rollout_empty_mask_fails_closed_on_a_real_run`.

### In-process vs DataLoader on real `pack_v3` TRAIN (read-only)

`batch_size=8`, `seed=101`, first epoch batch 0, `build_idea_a_bundle`,
`fit_outcome_norm` over the 2153 TRAIN items. Batches are byte-identical from
end to end:

```
TRAIN items: 2153 k_max: 192 format: v3
keys equal: True
tensor diffs: []
[in-process] target_mask.sum=906 rollout_mask.sum=17474 is_cav.sum=136 present0.sum=906
[loader]    target_mask.sum=906 rollout_mask.sum=17474 is_cav.sum=136 present0.sum=906
[in-process] terms: {"arrival": 0.8956710696220398, "correction_crps": 42.05474090576172,
                     "residual_nll": 1.2104331254959106,
                     "response_contrast": 9.326354026794434,
                     "rollout": 0.23975563049316406}
[loader]     terms: {"arrival": 0.8956710696220398, "correction_crps": 42.05474090576172,
                     "residual_nll": 1.2104331254959106,
                     "response_contrast": 9.326354026794434,
                     "rollout": 0.23975563049316406}
```

Before the fix the loader path could not deliver the real pack to workers
(spawn) and any empty-mask batch silently logged `residual_nll = rollout = 0.0`
(fail-open). Suite test `test_dataloader_worker_batches_equal_in_process_tensor_by_tensor`
pins tensor-by-tensor equality **and** equality of all five A3.2 terms on a
synthetic real-format pack with `num_workers=2`; `test_corpus_is_picklable_after_array_access`
pins the `__getstate__` regression.

**Reproduction boundary (honest):** the exact box trigger (CUDA +
`--num-workers 8`) cannot be rerun on this CPU-only laptop; what is reproduced
is (a) the loader-path failure/corruption mechanism and (b) the fail-open
exactly-zero mechanism. Both are now closed by construction (picklable corpus,
fail-closed masks).

## (C) Actual device recorded + per-step logging

* `IdeaATrainer.__init__` records the trainer's real device into the config
  (`self.config.device = str(self.device)`), so `member_<k>.json`,
  the checkpoint payload and `TRAIN_RECEIPT.json` no longer claim `cpu` when
  `--device cuda` was passed.
* `IdeaATrainer._assert_on_device` runs on the first fit batch and raises
  `IdeaAConfigError` if any bundle parameter or batch tensor is not on
  `self.device`.
* `fit` writes `<out>/train_log.jsonl` every step with
  `step`, `total`, each of the five terms, `s_per_step`, and CUDA memory
  (`cuda_mem_bytes`/`cuda_peak_bytes`, `null` on CPU). `--log-every N`
  (default 10) prints the record every N steps; `--log-every 0` disables the
  print (the file is always written).
* Tests: `test_actual_device_recorded_and_asserted`,
  `test_train_log_jsonl_written_per_step`.

## (D) Verification

Commands (cwd `output/wd01/wm_n1`, `WD_FORBID_D=1`, `OMP/MKL_NUM_THREADS=4`):

```
python -m pytest train/tests bundle/tests evalh/tests corpus/tests \
    wm_api/tests inc1eval/tests model/tests rollout2/tests corr/tests -q
python -m preflight all --data-root E:/research-dense
python -m ruff check --isolated --select E4,E7,E9,F --line-length 100 <changed files>
```

Result: **628 passed, 0 failed** (`train` 133, `bundle` 152, `evalh` 98,
`corpus` 80, `wm_api` 34, `inc1eval` 42, `model` 33, `rollout2` 43, `corr` 13);
`preflight` **6/6 PASS** (zero-residual, ctrl-parity, label-leak, overfit,
e2e-smoke, heldout-dryrun); ruff clean.

## Files

Changed: `model/rollout.py`, `bundle/timing.py`, `train/idea_a.py`,
`train/dataset.py`, `arrival/sampler.py`, `rollout2/step.py`,
`rollout2/engine.py`, `hdv/input_schema.py`, `inc1eval/incident_ctrl.py`.
New: `train/tests/test_gpufix.py`, `GPUFIX.md`.

Loss math, lambdas, `K_max` and registrations are unchanged. No archived code is
imported. `train/idea_a.py` edits are localized to device recording, masks,
logging and the DataLoader path.
