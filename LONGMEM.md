# S-LONGMEM

Default OFF. Full-batch forward fixes the objective and its VJP; start microbatches
accumulate that VJP before the existing single clip/Adam step. Global quantiles,
masked means, counterfactual weights, sample disagreement, RNG, duals and EMA remain
global. Critic MSE is streamed. Checkpoint segments preserve history, driver identity,
controller/lane caches, entries and gradient damping; backward rolls back side effects.
Constant future draws and, optionally, saved autograd tensors spill to host.

## Enable

Add to the policy TOML (these are `DreamerConfig` fields, not launcher arguments):

```toml
[trainer]
imagination_microbatch = 1
imagination_checkpoint_segment = 16 # gpuA; use 8 on gpuR
imagination_offload_futures = true
imagination_offload_saved_tensors = true
credit_profile_every = 0
```

Keep effective `batch_size` unchanged. Active configurations already use one start:
microbatching alone cannot help them. Recommend these settings at all four horizons;
larger effective batches can still use one-start accumulation. Segment units are
0.5-second physics steps. Explicit memory settings survive full checkpoint resume.
Internal-action credit profiling is explicitly incompatible.

## Evidence and projection

CPU-only, cpuP, one thread/nice15; synthetic recurrent WM, B=1/K=8/E=2/width4096.
Peak **live tensor** MiB and unprofiled seconds/update below; raw RSS and commands:
`tools/longmem_cpu_bench.json`, `scripts/longmem_bench.py`. A/R denote segments16/8.

| Horizon s | Tensor MiB old/A/R | CPU seconds old/A/R | Legacy GiB | gpuA GiB | gpuR GiB |
|---:|---:|---:|---:|---:|---:|
|30|30.37/4.62/2.66|0.20/0.47/0.48|18.2|6.47|5.42|
|60|60.74/4.72/2.80|0.39/0.93/0.94|36.5|6.53|5.50|
|120|121.49/4.93/3.08|0.86/1.90/1.95|73.0|6.64|5.65|
|300|303.72/5.55/3.93|2.19/5.04/4.95|182.5|6.97|6.10|

GPU numbers are **conditional models**, not measured fits: legacy=36.5H/60;
optimized=4+(36.5−4)×CPU_tensor_peak/60.7448 GiB. Assumptions: 4-GiB resident
allowance, unchanged slot/future/variant geometry, transferable synthetic ratios.
Actual R16 padding growth, allocator reserve and PCIe cost are uncalibrated;
gpuR uses gpuA's anchor. Tiny real-WM CPU RSS showed little improvement.
No GPU jobs or simulator runs were launched.

22 CPU tests, wheel build and Ruff passed. Equivalence tests compare gradients,
parameters, Adam states, target EMA and RNG:
`atol=2e-6, rtol=2e-5`, including production actor/WM, stochastic futures, entries,
lanes and correlated exploration.

Merge with S-LOOPLONG at `DreamerTrainer.imagine`/imagined-MSE and memory-only resume dispatch;
retain its horizon/reward/bootstrap changes and auxiliary critic blocks. No objective,
discount, truncation, curriculum or rollout-model code is changed here.
