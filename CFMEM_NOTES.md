# C4 imagination memory: checkpointed imagination (branch c4/cfmem, 2026-10-13)

Problem: the C4 cooperation trainer (`wms.control.dreamer` + `wms.control.counterfactual`) ran out of
memory on a 32 GB GPU (peak > 17 GB) at batch 8 / H 15 / max_starts 2 / n_group 1 / n_flip 1, while
the same settings without counterfactual rows fit.

## Where the memory goes (profiled first)
Method: tally of the bytes autograd stores for backward (`torch.autograd.graph.saved_tensors_hooks`,
unique storages, by source line) on the real R5b `wm_s_best.pt` and real TRAIN start windows
(K = 192 slots).  Without checkpointing, autograd keeps ~33-37 MB per row and physics step:
- 84 %: the WM-S residual network, mostly `_KinAttention._pair_bias` (a 6 -> 64 -> heads MLP over all
  [K,K] slot pairs: two [rows,192,192,64] fp32 tensors per layer, 3 layers, every physics step);
- ~10 %: the actor / RSU encoder at each decision (`_masked_attention` over reports, set blocks);
- ~2 %: the reward's pairwise TTC cost ([rows,K,K]); ~1 %: physics neighbours, `_merge_lane`.
So the peak is linear in (rows x physics steps) and the counterfactual rows (2-6 x the all-CAV rows)
multiply it.  The [B,K,K] tensors of reward / observations / safety are NOT the problem.

## Change (exact: same objective, same gradients)
1. `trainer.checkpoint_imagination` (default **true**, `DreamerConfig`, `configs/c4_policy.toml`):
   activation checkpointing (non-reentrant `torch.utils.checkpoint`) of every PURE per-step call in
   imagination, recomputed during backward instead of stored:
   - the WM step's `forward` (driver encoder, IDM/MOBIL physics, residual net) via the new runtime
     switch `WmSModel.checkpoint_forward` (set by the trainer only for its rollouts, restored after;
     never saved in checkpoints).  Unlike the existing `checkpoint_step` (which wraps the whole step
     and is therefore disabled with a `cav_controller`), this never replays the stateful CAV
     controller: the callback runs in `step` after `forward`, outside the checkpoint;
   - inside the bridge controller: the policy call (CRN noise is an explicit argument; without it the
     RNG state is restored for the recomputation), the RSU group encoder, the safety filter and the
     native-CAV IDM; and the per-transition reward and the target-critic bootstrap values.
   The controller's own state (held action, dwell clock, command memory, records) is never
   recomputed, so no replay-safety rework of the bridge was needed.
2. `_merge_lane` (bridge) reads its geometry detached: its output is a discrete lane index, so values
   are identical and autograd no longer keeps the K-iteration loop's temporaries.
Gradients: `tests/unit/test_c4_cfmem.py` compares actor and critic gradients, losses and the C_G /
M_i / D_* terms of one full `update()` with checkpointing on vs off on a tiny real WM-S (hybrid and
residual, lanes + counterfactual rows, and the plain all-CAV objective): equal at rtol 1e-5; it also checks that the controller is called exactly once per physics step (no replay) and that the stored activations shrink.  On GPU, recomputation can differ from the
stored forward only by kernel nondeterminism (atomics), i.e. float noise.

## Measured (CPU, 1 torch thread, real R5b `wm_s_best.pt`, real TRAIN windows, K = 192, update 1500
## curriculum weights; `scripts/c4_cfmem_profile.py`)
Peak = maximum live bytes of all CPU tensor storages created during one `trainer.update()`
(TorchDispatchMode registering every ATen output storage + `weakref.finalize` on free; the CPU
analogue of CUDA `max_memory_allocated` above the entry level, without allocator rounding;
cross-checked against `torch.profiler(profile_memory=True)` allocator events: 4.50 vs 4.43 GB and
1.17 vs 0.97 GB on the same B8/H3 update).  Rows vary per update with the sampled groups.

| setting | checkpoint_imagination | rows | peak allocated | s / update (CPU) |
|---|---|---|---|---|
| B8, H2, ms2, ng1, nf1 | off | 19 | 2.60 GB | |
| B8, H3, ms2, ng1, nf1 | off | 21 | 4.50 GB | 10.3-11.0 |
| B8, H3, ms2, ng1, nf1 | on | 21 | 1.17 GB | 14.6-14.9 |
| B8, H4, ms2, ng1, nf1 | off | 20 | 5.85 GB | |
| B8, H5, ms2, ng1, nf1 | off | 24 | 8.94 GB | 20.1-21.4 |
| **B8, H15, ms2, ng1, nf1** | **off** | 18-21 | **~20-23 GB** (extrapolated: 36.5 MB x rows x 30 steps; cannot run under the 12 GB process cap) | ~45-60 (extrapolated) |
| **B8, H15, ms2, ng1, nf1** | **on** | 18 | **1.23 GB** | **66-76** (16-18 rows) |
| B8, H15, no counterfactual | off | 8 | 9.50 GB | 23-30 |
| **B16, H20, ms6, ng2, nf2 (gpuW)** | **on** | 86 | **6.28 GB** | 436 (1 thread; 0.13 s per row-step) |
| B16, H20, ms6, ng2, nf2 (gpuW) | off | 86 | ~126 GB (extrapolated) | |

Per row and physics step: off ~36 MB stored; on ~0.5 MB stored plus a transient of ~45-50 MB per
row while ONE step's forward is recomputed and back-propagated (all three pair-bias layers of that
step at once).  The gpuW setting is therefore ~2 GB stored + ~4 GB transient = 6.3 GB.
GPU estimate (4080, 16 GB): 6.3 GB allocated + CUDA context (~0.5-1 GB) + caching-allocator slack
(~10-25 %) = ~8-9 GB reserved: fits the ~12 GB budget.  Measure on the box with
`torch.cuda.max_memory_allocated()` in the first `--max-updates 5` run before the long run.
Cost: ~+40 % CPU time per update (one extra forward of the WM step, dominated by the pair-bias MLP;
the other wrapped calls are within timing noise: keeping only the WM forward + policy checkpointed
gave the same time, but 1.24 vs 1.14 GB at H4 and the gap grows with H).  On a launch-bound GPU the
recomputation adds the forward's kernels to backward, expect a similar +30-40 %.

## Considered, not adopted (documented decisions)
- (a) Base rows under `torch.no_grad()`: base rows are 1 per start (6 of 86 rows at gpuW) and are
  batched in the same tensors as gradient rows; excluding them needs a separate sequential rollout,
  i.e. more kernel launches on a launch-bound GPU, for <= 7 % of the stored memory.  Their value
  enters only detached (B), so they cost memory but no gradient work.
- (b) Row chunks with per-chunk backward (`counterfactual.chunk_rows`): not exactly decomposable as
  implemented - the lane-head entropy bonus is a masked mean over ALL rows (`_masked_mean(lane_entropy,
  active)`), whose denominator is known only after every chunk's rollout; an exact version would need
  an extra no-grad pass for the counts (or two backward traversals per chunk).  Chunks also serialise
  the rows on a launch-bound GPU (~chunks x time).  With checkpointing the target fits with margin,
  so no chunk knob was added; it remains the lever for much larger settings (then also chunk the
  pre-pass).
- (d) Rewriting the [B,K,K] reward-TTC / observation / safety tensors: profiling shows they are a few
  percent; checkpointing the reward covers the TTC part.
- Further headroom if ever needed: a nested checkpoint of `_pair_bias` per attention layer would cut
  the ~4 GB recomputation transient at gpuW by ~2/3 for one more pair-bias forward.

## Reproduce
```
export PYTHONPATH=src C4_ALLOWED_ROOTS=E:/research-dense WMS_DATA_PACK_V3=... (see the script doc)
python scripts/c4_cfmem_profile.py --wm-ckpt <R5b wm_s_best.pt> --batch 8 --horizon 15 \
    --max-starts 2 --n-group 1 --n-flip 1 --checkpoint-imagination on --updates 1
python scripts/c4_cfmem_profile.py ... --no-profile --updates 2      # time only
```
CLI unchanged; `checkpoint_imagination = false` in `[trainer]` restores the old behaviour.
Unrelated pre-existing failures (also on c4/int HEAD): `test_c4_mappo.py::test_config_toml_defaults_and_smoke_merge`
(LaneRules defaults), `tests/unit/data/test_drive_registry.py::test_committed_registry_matches_the_generator_output`.
