# K_MAX overflow — cost of the three options (ANALYSIS ONLY)

**Scope.** WM-N1 Idea A vehicle-slot overflow at `K_MAX = 128`. This note costs
option (A) raise K, (B) per-lane / spatial-window cap at K=128, (C) keep 128 and
disclose. **No K, model or feature code was changed**; all benchmarks are
synthetic tensors, CPU-only, zero SUMO, 2 threads, BELOW_NORMAL (a
featurization/pack job was running). Branch `zcode/kmax` @ `9172a55`.

Base corpus reference (owner-provided): **FEAT1 v2 = 35.07 GB at K=128**. Price
is per training member (design 9–17 GPU-h at K=128) and per full-corpus
re-featurization (~5 h at 6 workers, `--idea-fast`).

Definitions used below (frozen array layout, `features/FORMAT.md`,
`features/idea_a.py`, `corpus/pack.py`):

| per-slot / per-tick | bytes @K=128 | K-scaling |
| --- | --- | --- |
| `veh_step` [T,K,16] f16 | 32/slot/tick | ×K |
| `veh_kin` [T,K,7] f16 | 14/slot/tick | ×K |
| `hdv_base` [T,K,6] f16 | 12/slot/tick | ×K |
| `cav_plan_ch` [T,K,8] f16 | 16/slot/tick | ×K |
| `veh_mask`/`target_mask`/`lane_next` | 3/slot/tick | ×K |
| `accel_next` [T,K] f16 | 2/slot/tick | ×K |
| `plan` [N_dec,K,19,8] f16 | 304/slot/decision | ×K |
| `wall`+`wall_mask`+`counter`+`arrivals`+`event_*` | 352/tick | ×1 (K-independent) |

K-dependent share of the corpus at K=128: **96.8 %** (K-dependent ≈ 33.96 GB,
K-independent ≈ 1.11 GB, anchored to 35.07 GB with the pack coefficients and
`Nd ≈ 20·R`, matching the registered ~2463-run / ~3.2 M-tick selection).

---

## 1. Corpus memory scaling

| K | K-dependent (GB) | K-independent (GB) | **corpus total (GB)** | Δ vs 128 | ratio |
| --- | --- | --- | --- | --- | --- |
| 128 | 33.96 | 1.11 | **35.07** | 0 | 1.000 |
| 192 | 50.94 | 1.11 | **52.05** | +16.98 | 1.484 |
| 256 | 67.92 | 1.11 | **69.03** | +33.96 | 1.968 |

K=128 breakdown (GB): `veh_step` 12.98, `cav_plan_ch` 6.49, `veh_kin` 5.68,
`hdv_base` 4.87, `plan` 1.92, masks+lane 1.22, `accel_next` 0.81, K-independent
1.11. (Each of the first four doubles at K=256.)

**Disk on the staging root** additionally needs shards + pack; Option A at
K=256 roughly doubles the K-dependent shard/pack footprint (~68 GB shards +
~69 GB pack), i.e. provision ≈150 GB free vs ≈40 GB today.

## 2. GPU activation memory per batch (HDV attention O(K²) + rollout)

Measured with `torch.autograd.graph.saved_tensors_hooks` on the **REAL** Idea A
member (real `HdvResidualNet` d_model 128 / 4 heads / 3 layers / ffn 512, real
`IdeaAArrivalHead` counter_channels 128, real 64-d `CavMotionSummary` +
`CorrectionHead` d_model 256) on the full A3.2 loss (residual NLL + 20-step
differentiable rollout + arrival + correction CRPS + paired response contrast),
**B=8, horizon=20** (`IdeaATrainConfig`), fp32 saved activations:

| K | saved activation (MiB) | (GiB) | peak process RSS (MiB) |
| --- | --- | --- | --- |
| 128 | 5470.2 | 5.34 | 15047.9 |
| 192 | 11782.1 | 11.51 | 15074.9 |
| 256 | 20487.0 | 20.01 | 20101.3 |

Fit: `saved_activation(MiB) ≈ 0.291·K² + 5.44·K` (K² attention term matches the
`_pair_bias` `[B,K,K,64]` MLP + `[B,H,K,K]` logits/attn saved in each of the 3
layers, replicated over the 22 graph passes = 1 residual + 20 rollout + 1
contrast). Measured growth ×2.15 at K=192 and ×3.74 at K=256.

## 3. CPU timing (real member, forward+backward, median of 5)

Command (cwd `output/wd01/wm_n1`):

```
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe kmax_bench.py --k 128 --k 192 --reps 5 --out kmax_bench_128_192.json
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe kmax_bench.py --k 256 --reps 5 --out kmax_bench_k256.json
```

stdout (real, verbatim; first command, then the K=256 command):

```
{"batch": 8, "below_normal": true, "event": "start", "horizon": 20, "ks": [128, 192], "reps": 5, "threads": 2, "torch": "2.12.1+cpu"}
{"K": 128, "base_working_set_mb": 256.71484375, "batch": 8, "event": "result", "horizon": 20, "median_s": 13.290287400013767, "n_params": 1176777, "peak_rss_mb": 15047.9140625, "saved_activation_mb": 5470.239730834961, "times_s": [13.552582399977837, 13.290287400013767, 13.194450999959372, 12.846502500004135, 14.404129200032912], "working_set_after_mem_mb": 5919.4140625}
{"K": 192, "base_working_set_mb": 11539.55078125, "batch": 8, "event": "result", "horizon": 20, "median_s": 24.403569800022524, "n_params": 1176777, "peak_rss_mb": 15074.85546875, "saved_activation_mb": 11782.063461303711, "times_s": [26.25545339996461, 24.403569800022524, 24.31096749997232, 23.47159400000237, 24.712369599961676], "working_set_after_mem_mb": 15047.9140625}
{"batch": 8, "event": "done", "horizon": 20, "reps": 5, "results": [{"K": 128, "base_working_set_mb": 256.71484375, "batch": 8, "event": "result", "horizon": 20, "median_s": 13.290287400013767, "n_params": 1176777, "peak_rss_mb": 15047.9140625, "saved_activation_mb": 5470.239730834961, "time_ratio_vs_128": 1.0, "times_s": [13.552582399977837, 13.290287400013767, 13.194450999959372, 12.846502500004135, 14.404129200032912], "working_set_after_mem_mb": 5919.4140625}, {"K": 192, "base_working_set_mb": 11539.55078125, "batch": 8, "event": "result", "horizon": 20, "median_s": 24.403569800022524, "n_params": 1176777, "peak_rss_mb": 15074.85546875, "saved_activation_mb": 11782.063461303711, "time_ratio_vs_128": 1.8361957921238967, "times_s": [26.25545339996461, 24.403569800022524, 24.31096749997232, 23.47159400000237, 24.712369599961676], "working_set_after_mem_mb": 15047.9140625}]}
{"batch": 8, "below_normal": true, "event": "start", "horizon": 20, "ks": [256], "reps": 5, "threads": 2, "torch": "2.12.1+cpu"}
{"K": 256, "base_working_set_mb": 257.00390625, "batch": 8, "event": "result", "horizon": 20, "median_s": 48.88014389999444, "n_params": 1176777, "peak_rss_mb": 20101.34375, "saved_activation_mb": 20487.00828552246, "times_s": [45.40524259995436, 47.34256020002067, 57.00786780001363, 48.88014389999444, 55.05848010000773], "working_set_after_mem_mb": 16898.05078125}
{"batch": 8, "event": "done", "horizon": 20, "reps": 5, "results": [{"K": 256, "base_working_set_mb": 257.00390625, "batch": 8, "event": "result", "horizon": 20, "median_s": 48.88014389999444, "n_params": 1176777, "peak_rss_mb": 20101.34375, "saved_activation_mb": 20487.00828552246, "times_s": [45.40524259995436, 47.34256020002067, 57.00786780001363, 48.88014389999444, 55.05848010000773], "working_set_after_mem_mb": 16898.05078125}]}
```

| K | median fwd+bwd (s) | ratio vs K=128 |
| --- | --- | --- |
| 128 | 13.290 | 1.000 |
| 192 | 24.404 | 1.836 |
| 256 | 48.880 | 3.678 |

**GPU-hour extrapolation from the design 9–17 GPU-h/member at K=128** (assuming
step time scales as measured; the attention-only FLOP ratio 4.0 at K=256
brackets it). The 3 held-out tests (1 member each) scale the same way.

| K | per member (GPU-h) | 5 Idea A members (GPU-h) |
| --- | --- | --- |
| 128 | 9.0–17.0 | 45–85 |
| 192 | 16.5–31.2 | 83–156 |
| 256 | 33.1–62.5 | 166–313 |

## 4. Does a 48 GB card fit the training batch at K=256?

At B=8 (the registered batch size) the measured activation is **20.0 GiB**;
adding 1.18 M params + grads + Adam moments (~19 MB), CUDA context / cuDNN /
cuBLAS workspaces and allocator fragmentation (~15–25 % + ~2–3 GB), resident
use is **≈26–28 GB → fits 48 GB with ≈20 GB headroom**.

Caveats: activation is linear in B, so **B=16 at K=256 does not fit** (≈40 GiB
+ overhead ≈ 46–49 GB, marginal/over); `B=16` at K=192 (≈23 GiB) fits. The
figure assumes the registered full A3.2 loss (rollout included); the current
`batch_from_corpus_items` builds no `teacher`/`summary_raw`, so a corpus-driven
run that silently drops those terms would look much smaller — this analysis
costs the designed loss.

## 5. Re-featurization cost

* **Option A (raise K):** the packed corpus has one uniform K (`pack.py` raises
  on any shard shape mismatch), so **every shard must be re-featurized** →
  full corpus ≈ **5 h at 6 workers** (~30 worker-h) plus a new ~52 GB (K=192)
  / ~69 GB (K=256) pack; only a new padded-pack reader variant could restrict
  the re-run to the overflowing runs, and that is a code change.
* **Option B (cap at K=128):** the slot *selection rule* changes (which
  vehicles keep the 128 slots), so the corpus must also be re-featurized
  (~5 h at 6 workers) **but the tensors, model, memory and timing are
  unchanged** (no K change). Requires a registered `FORMAT.md` revision.
* **Option C (keep 128):** zero re-featurization; cost is only the unquantified
  dropped-share bias.

## 6. Slot-pool census fold-in

`E:/rd-wt/feat1v2c/output/wd01/wm_n1/SLOT_POOL_CENSUS.json` (produced by another
task; read-only poll up to 60 min):

> **STATUS: not present as of this commit.** The census had not been written at
> the end of the 60-minute poll window. The dropped-share numbers therefore
> cannot be folded in yet; §7 states how they change the recommendation.

## 7. Recommendation

Costs:

| | corpus | per member | 5 members | 48 GB @K=256 | re-featurize |
| --- | --- | --- | --- | --- | --- |
| **A** K=192 | 52.05 GB (+17.0) | 16.5–31.2 | 83–156 | fits (B=8, B=16) | full, ~5 h@6 |
| **A** K=256 | 69.03 GB (+34.0) | 33.1–62.5 | 166–313 | fits B=8, **not B=16** | full, ~5 h@6 + ~150 GB disk |
| **B** cap @128 | 35.07 GB (0) | 9–17 (0) | 45–85 (0) | n/a | full, ~5 h@6 (rule change) |
| **C** keep 128 | 35.07 GB (0) | 9–17 (0) | 45–85 (0) | n/a | none |

1. **Do not raise K to 256 by default.** It is ~3.7× the member cost for
   ~2× corpus size, and it forces a full re-featurization plus ~150 GB staging
   disk. K=192 (+48 %) is the only raise with a proportionate trade-off, and it
   still fits 48 GB at B=16.
2. **Prefer Option B when the census shows overflow is a slot-selection
   artifact** (far-from-merge vehicles occupying slots that near-merge vehicles
   need). It removes the drop at *zero* memory/timing cost and the mandatory
   re-featurization is already paid. Only a `FORMAT.md` + selection-rule change
   is needed.
3. **Option A (K=192) is required when the census shows genuinely >128
   concurrent vehicles near the merge at the same tick** — B cannot fix that.
4. **Option C is acceptable only if the census shows a small dropped share away
   from the merge** (< ~1 % of vehicle-steps and off the merge influence
   window); otherwise it injects a state-dependent bias into the TRAIN labels.
5. Until the census lands, treat A/K=256 as unaffordable and C as unjustified:
   the decision is **B if selection-limited, else A/K=192**. Fold the census
   dropped-share into row 1 of §1 before committing.

### Evidence

* Benchmark script: [`kmax_bench.py`](kmax_bench.py) (synthetic tensors only).
* Raw results: [`kmax_bench_128_192.json`](kmax_bench_128_192.json),
  [`kmax_bench_k256.json`](kmax_bench_k256.json).
