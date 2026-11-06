# V10R5MEM - bounded merge-negotiation gap matrix (branch c4/v10r5mem, 2026-10-20)

Round 3 crashed with a CUDA OOM (+7.18 GiB) in `MergeNegotiationLoss.prepare`, which built
the dense pairwise tensor `ds = s_i - s_j` of shape `[B, T, K, K]` (T=600, B=16) under
`no_grad`.

## Change

- `src/wms/train/losses/merge_negotiation.py`: `prepare` now calls `_nearest_behind_gap`,
  which materialises the pairwise matrix `pair_chunk_steps` (default 50) time steps at a
  time. Peak is `[B, min(T, 50), K, K]` instead of `[B, T, K, K]`, i.e. ~12x lower at
  T=600 (a 7.18 GiB allocation becomes ~0.6 GiB) and independent of T.
- The min reduction is over target columns, never over time, so `gap_min` / `idx` and every
  downstream output (`f_acc`, `yield`, `has`, `merged`, counts) are bit-identical.
- New validated config field `MergeNegotiationLossConfig.pair_chunk_steps = 50`.
- `tests/unit/test_merge_negotiation.py`: new test compares the chunked `prepare` against
  the old dense implementation on random targets (position ties, no-candidate/inf gaps,
  all-false masks, T=23 not divisible by the chunk, T=0) and across chunk sizes
  1/5/7/50/1000 with exact tensor equality.

## Pairwise-allocation scan

- `src/wms/model/merge_negotiation.py::pair_features`: builds `[B, K, K]` but it is called
  per rollout step from `WmSModel.forward`, so its peak is K^2 only, not T*K^2. Unchanged.
- `src/wms/train/losses/merge_timing.py`: no pairwise allocation; all tensors are
  `[B, T, K]`. Unchanged.
- Out of scope but noted: `src/wms/train/courtesy.py:35` also builds `[B, T, K, K]`
  (`courtesy_follower_targets`, a teacher-assisted path). Left untouched.
