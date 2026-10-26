# S-SPEED2 — opt-in training and policy speed-ups

All flags default **OFF**; checkpoints/config schemas are unchanged. Set flags for an entire run, including backward/checkpoint recomputation. Evidence and source fingerprints: `docs/speed2/*.json`; reproducible probes: `tools/speed2_{measure,real_probe,verify}.py`.

## Measured speed

gpuB, Torch 2.8/cu128, nice 19, one thread/worker, at most two concurrent cores, owned GPU usage below 4 GiB. Synchronized wall seconds; kernels/replays use 3–5 repeats, complete steps/rollouts one run. Shared-host timings are approximate. Synthetic training: B=1, K=595, 100 present, TBPTT=1; this is **not** the production ~340 s recipe. Policy: 96 slots/24 present, 10 control steps, checkpointing, four counterfactual rows.

| Work / device | Off → on (s) | Gain |
|---|---:|---:|
| 120 s training step / CUDA, compact+diagnostics off | 26.950 → 22.691 | 1.19× |
| Same training step / CPU | 364.625 → 34.393 | 10.60× |
| Policy14 `wm_law`+head / CUDA, exact flags | 7.997 → 2.788 | 2.87× |
| Policy14 legacy gap6 / CUDA, exact flags | 7.486 → 2.330 | 3.21× |
| Trained 13E-A, real TRAIN 300 s rollout / CUDA, compact+diagnostics off | 26.849 → 22.187 | 1.21× |
| Residual forward+backward / CPU, compact | 0.980 → 0.0255 | 38.5× |
| Synthetic 300 s rollout / CPU, compact+diagnostics off | 306.927 → 36.054 | 8.51× |
| Policy14 legacy gap6 / CPU, exact+compact | 16.872 → 1.882 | 8.96× |
| Four ragged prefix replays / CUDA, batched | 0.968 → 0.592 | 1.63× |
| Pivot step after 240 waiting ticks / CUDA, diagnostics off | 0.0399 → 0.00124 | 32.0× |
| Long-history features / CUDA | 0.00984 → 0.000989 | 9.94× |
| Timeline callback / CUDA | 0.0146 → 0.00187 | 7.79× |
| gap6 bridge, no candidates / CUDA | 0.1043 → 0.000371 | 281× |

CPU-only history batching and uncombined replay batching were slower (0.055→0.062 and 21.08→33.41 s); prefer those flags for CUDA work.

## SAFE flags

Names below have prefix **`WMS_SPEED_`**; boolean value is `1`.

- **Round training, exact:** `PIVOT_DIAG_OFF`, `FAST_CAV_DRIVE` (timeline callbacks), `FAST_LONG_HISTORY` (history outputs). Hybrid pivot skipping removes only rollout `logS/event_mass/cdf`, replacing them with NaNs. Built-in training NLLs integrate their own recorded/on-policy boundary histories; their quadrature, losses and gradients remain intact. Leave diagnostics enabled for external consumers that read them.
- **Round training, numerical:** `COMPACT_SLOTS`, `BATCH_REPLAY`. Present-only gather/network/scatter and batched frozen replays preserve the equations, with floating-point reassociation. Active dropout falls back; compaction also bypasses full-width cases, and incompatible replay inputs/hooks preserve serial execution.
- **Policy loop, exact:** `PIVOT_DIAG_OFF=1`, `GRAD_AUDIT_EVERY=10`; add `FAST_MERGE_LANE=1` for gap6. Audits run on updates 1,11,…; action audits and actual actor/critic updates remain unchanged. The measured exact bundle preserves weights bit-for-bit. `COMPACT_SLOTS=1` is optional numerical acceleration, especially on CPU. Replay/timeline flags do not accelerate this policy path.

## Equivalence

**604 tests pass**, two pre-existing round-9 JSON-schema failures reproduced on the base; **178 flag-off tensors match original `77d9e574` storage bytes**. Ruff passes.

Across synthetic/trained residual checks, max abs/relative differences: outputs **1.43e−6/9.28e−5**, losses **2.38e−7/7.61e−8**, gradients **3.81e−6/0.844**. Relative denominator is `max(|reference|,1e−8)`; near-zero gradients explain large ratios. Complete training-loss difference: **1.22e−4 / 7.04e−8 relative**.

Synthetic 300 s compaction drift (CPU/CUDA maxima): position **0.977 mm / 1.08e−4 relative**, speed **6.87e−5 / 1.15e−4**, acceleration **6.09e−5 / 14.0** (near zero); lanes/masks identical. Diagnostics-only skipping is bit-identical/RNG-neutral. Real TRAIN state `969824@300`, 79 present, causal 60 s history: both skips preserve the full 300 s trajectory and RNG exactly; 13E-A disables residual acceleration, so standalone trained-residual checks accompany it.
