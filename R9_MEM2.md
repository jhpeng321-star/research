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
Default False (bit-identical). Enable for round 9 with `--set model.checkpoint_forward=true` (config file NOT changed; the
launcher must pass it, or set `checkpoint_forward = true` in [model] of configs/c4_wm_v10round9.toml).

## Exactness
tests/test_checkpoint_forward_exact.py (state_st True and False): 8-step first_passage/hybrid rollout in train mode, perturbed
weights, loss + backward with and without checkpoint_forward: identical loss value, every parameter gradient within 1e-8 relative
Measured: loss identical and ALL 16 gradient tensors bit-identical (worst relative difference 0.0), for state_st True and False.
