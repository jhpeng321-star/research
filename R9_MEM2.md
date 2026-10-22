# R9 memory, part 2: batch 4 at 120/300 s WITH the full pathwise adjoint (state_st = true)

Branch c4/v10r9mem2 (from c4/v10r9lane 659e08c7). Round-9 config has `merge_pivot_state_st = true` again (owner decision);
the option stays available.

## Cause (recap, R9_MEM.md)
With state_st=true the pivot lane weights carry a straight-through adjoint into the next rollout step, so the 120-step rollout
graph (all frozen modules) is retained: >23 GB at batch 2/4, 11-13 GB at batch 1 (120 s). `model.checkpoint_step` (per-step
checkpoint, already true in the config chain) is DISABLED whenever a stateful CAV controller callback is passed to the rollout
(rollout(): `step_checkpoint = ... and cav_controller is None`), which is the case for the C4 packs, so nothing was recomputed.

## Change
The model already had a runtime switch `WmSModel.checkpoint_forward` (used by the dreamer): `step()` wraps the pure per-step
`forward` in `torch.utils.checkpoint` (non-reentrant, RNG preserved), recomputing encoders/physics/residual-net in backward.
The CAV controller callback runs OUTSIDE `forward`, so it is never replayed. I only wired it into training:
`model.checkpoint_forward` (config/train.py, legacy_flat.py, train/settings.py) -> `WmSTrainer._apply_model_overrides`.
Default False (bit-identical). configs/c4_wm_v10round9.toml now sets checkpoint_forward = true (and merge_pivot_state_st = true).

## Exactness
tests/test_checkpoint_forward_exact.py (state_st True and False): 8-step first_passage/hybrid rollout in train mode, perturbed
weights, loss + backward with and without checkpoint_forward: identical loss value, every parameter gradient within 1e-8 relative
Measured: loss identical and ALL 16 gradient tensors bit-identical (worst relative difference 0.0), for state_st True and False.

Real-batch check (gpuR, real TRAIN batches, round-8 init, ALL 131 tensors trainable via `--set optim.freeze_prefixes=[]`, batch 1,
60 s window, state_st=true, same RNG; probe-only masking of the non-finite pivot term; scripts/r9_mem_probe.py
`R9_PROBE_KIND=ckf R9_PROBE_COMPARE=1`): two batches, checkpoint_forward on vs off:
loss total identical (max abs diff 0.0); every gradient tensor bit-identical (max abs diff 0.0, cosine 1.000000000000, global
gradient norms 18.5 and 7.21); peak memory 517 MB (on) vs 16,240 MB (off) and 544 MB vs 16,331 MB.

## gpuR trial: batch 4, state_st=true, checkpoint_forward=true, round-8 init, real TRAIN batches
(forward+backward of the chunked unroll, no optimizer step; probe masks non-finite terms and attaches a zero anchor to
no-grad chunks, probe only; the GPU was shared with another agent's ~3 GB jobs during these runs; step time includes data/target
preparation and the 2120-step F2 pre-window replay and is inflated by the contention)

| stage (truncation) | step | max_memory_allocated | reserved | step time |
|---|---|---|---|---|
| 120 s (120) | 0 | 2,362 MB | 3,016 MB | 203 s |
| 120 s (120) | 1 | 2,390 MB | 3,016 MB | 364 s |
| 300 s (60)  | 0 | 3,292 MB | 3,846 MB | 302 s |

Before (same code, checkpoint_forward off): CUDA OOM at >23 GB (batch 2 and 4, 120 s, in residual_net during the rollout).
Both stages fit with ~20 GB headroom. The per-step recompute costs one extra forward per step (not isolated here, because the
no-checkpoint run cannot complete at batch 4). The `NOGRAD` probe abort at the end of each step comes from the once-per-unroll
terms and happens after every chunk's backward.

## Tests
pytest -q tests/test_merge_pivot.py tests/test_merge_pivot_hybrid.py tests/test_merge_pivot_pre_window.py tests/test_merge_pivot_runtime.py
tests/test_merge_pivot_training.py tests/test_merge_pivot_support.py tests/test_merge_pivot_memory.py tests/test_checkpoint_forward_exact.py:
235 passed (before the config test was added; runtime + exactness files re-run after: 21 passed).
