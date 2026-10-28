# S-POLSPEED2 — exact parallel candidate-A policy updates

Branch `c4/polspeed2`, based on `c4/polspeed` **f6dea57b**. Task output:
`/root/autodl-tmp/c4/polspeed2_20261028` on gpuA. Measurements: October 28–29, 2026.

## Result and recommendation

**Qualified strict-exact recipe: FGA + four replay workers + parallel no-grad
forward + checkpoint segment 2, original nine rows. Update 181 takes
350.756 s: 2.90x faster than 1,018 s, or 2.25x faster than FGA's 788.2 s.**

* All **158 logged leaves**, including the independent rescore, are bitwise
  identical to both the original dry run and S-POLSPEED FGA. Timing and CUDA
  memory keys are excluded exactly as in the original comparison.
* All **157 actor/group-encoder/Adam tensors** are byte-identical between the
  N2 and N4 nine-row executions. CPU/CUDA tests additionally compare full
  gradients and optimizer state against serial execution.
* Sampled owned GPU peak: **19,394 MiB / 49,140 MiB = 39.47%**, below the 60%
  aggregate ownership cap, including coordinator and worker contexts.
* 150 updates: **14.61 h of update time**, approximately **15.94 h including
  observed startup/rescore/panel overhead**. This is a projection, not a
  measured 150-update training run.

The six-row preset is integrated, but is **not bitwise-equivalent on the real
update**. Its measured differences and faster timings are reported separately
below. Keep the original nine-row geometry for strict-exact adoption.

## Qualified activation

All production execution flags default OFF: worker count defaults to 1,
parallel forward to false, and VJP slots default to the worker count. The
six-row preset requires explicit config selection.

Keep the original microbatch 1 and both offloads enabled. Apply the following
memory-only override to the resolved candidate-A config:

```toml
[trainer]
imagination_checkpoint_segment = 2
```

Use the existing FGA flags plus these execution settings:

```sh
WMS_SPEED_FUTURE_WORKERS=4
WMS_SPEED_PARALLEL_FORWARD=1
WMS_SPEED_FUTURE_VJP_SLOTS=4
WMS_SPEED_FUTURE_MEMORY_FRACTION=.10
WMS_SPEED_FORWARD_MEMORY_FRACTION=.04
WMS_SPEED_REPLAY_FORWARD_MEMORY_FRACTION=.022
C4_GPU_MEM_FRACTION=.10
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WMS_SPEED_GRAPH_CVAR=1
WMS_SPEED_ASYNC_SPILL=1
WMS_WMSPEED_DEDUPE_MOBIL=1
WMS_WMSPEED_MOBIL_LAZY_SAFE=1
WMS_SPEED_PIVOT_DIAG_OFF=1
WMS_SPEED_FAST_MERGE_LANE=1
WMS_SPEED_FAST_CAV_DRIVE=1
WMS_SPEED_FAST_LONG_HISTORY=1
WMS_SPEED_GRAD_AUDIT_EVERY=25
```

Install the optional dependency with `pip install '.[parallel-futures]'`
(cloudpickle 3.1.2 was used). Main's adoption receipt/launcher must record the
new segment size and aggregate multi-process memory budget; its existing
segment-8 assertions are described in `POLSPEED.md`.

## Measurements on the same update 181

Inputs: optimization seed 965140, original C1 `c01.a3` resume at update 180,
WM `c03.a2`, original resolved candidate-A POLICY/BOUNDARY/HORIZON_AUTHORITY,
1 start, K8 futures, 90 s / 180 WM steps, entrants on. FGA is enabled throughout.
Every completed run stops after update 181 using its own STOP file.

### Original nine-row geometry

* **N2, segment 8:** **454.188 s**, **24,882 MiB owned peak (50.63%)**;
  all 158 reference leaves bitwise equal. Projection: **18.92 h** update-only,
  **20.22 h** including estimated scoring overhead.
* **N4, segment 2:** **350.756 s**, **19,394 MiB (39.47%)**;
  all 158 reference leaves and all 157 N2 actor/Adam tensors bitwise equal.
  Projection: **14.61 h / 15.94 h**, respectively.
* **N8, segment 1, per-worker cap .045:** incomplete; CUDA allocator OOM
  diagnostics from several workers, and a peer exit `-9` was observed.
  Owned sampled peak **20,812 MiB**. No completed-update timing or exactness
  claim. CUDA reported substantial physical free memory: the per-process
  allocator cap was binding.
* **N8, segment 1, cap .055:** incomplete; owned sampled peak **20,836 MiB**.
  Full diagnostics are in the corresponding evidence directory.
* The tested gated N8 recipes were also incomplete: caps `(active, idle)`
  `(.09,.018)`, `(.088,.024)` and `(.084,.028)`, with four VJP slots and
  segment 2. Sampled owned peaks were **13,766**, **13,762** and **14,354 MiB**.
  They have no completed-update timing or parity claim. N4 is the largest
  **qualified strict-exact** worker count in this task.

### Equivalent six-row preset + FGA

Preset imported from auditfix commit `24928e9d` as **db05c3e0**:
`configs/c4_policy14_c4prep_sixrow.toml`. For the registered three-member group,
the real-arithmetic objective is
`D_all + 49.7 D_G - 10.3 sum_i D_(G\i)`. Individual and coop weights become zero,
group becomes 18.8, and the three zero-coefficient individual rows are removed.

* **Serial, segment 8:** **746.248 s**, **8,256 MiB owned peak**.
* **N2, segment 8:** **440.823 s**, **17,068 MiB**; 150-update time **18.37 h**.
* **N4, segment 2:** **286.974 s**, **13,718 MiB**; **11.96 h** update-only,
  approximately **13.21 h** including scoring.
* **N8, segment 1:** **231.420 s**, **21,474 MiB**; **9.64 h** update-only.

All three parallel six-row runs are bitwise equal to **serial six rows** on
all 158 logged leaves and all 157 actor/Adam tensors. Thus process parallelism
and the tested checkpoint sizes add **zero** observed numerical error within
the same row geometry.

### Six rows versus the original reference: observed bounds

There are **61 logged differences** and **101 differing tensors out of 157**.
Some logged changes are intentional coverage/weight/decomposition changes:
dead individual-row diagnostics cannot retain their original interpretation
after those rows are removed. Other changes are floating-point rounding caused
by changing CUDA row/reduction geometry.

Observed numerical differences, identical across the completed six-row modes:

* Actor loss: **7.778406143e-6 absolute**, **2.584392546e-4 relative**.
* Return mean: **9.765625e-4 absolute**; return scale: **6.103509804e-7**.
* Actor gradient norm: **9.765625e-3 absolute**, **4.210985180e-7 relative**.
* Pessimistic paired credit: **1.067352295e-2 absolute**.
* Updated policy maximum absolute difference: **2.808678801e-6**.
* Group encoder maximum absolute difference: **1.117587090e-6**.
* Adam state maximum absolute difference: **9.536743164e-7**.
* Independent rescore leaves: **bitwise equal** on update 181.

These are tight **observed update-181 bounds**, not a universal guarantee for
future updates. Near-zero attention biases can have large relative errors;
a blanket small relative tolerance is unsupported. The imported synthetic
preset test establishes algebraic/tolerance parity, not real-update bitwise
parity. Full key/tensor differences are retained in `evidence/REPORT.json`.

## Implementation and exactness

`src/wms/control/parallel_futures.py` starts independent Python processes with
owned snapshots of the actor/WM, input batch, random tape, cached futures and
outer VJP vectors. No start-state sampler/dataset or actor optimizer is sent
to a worker. Each
worker has independent RNG, mutable rollout switches, checkpoint/controller
state, saved-tensor hooks, allocator and FGA CUDA-graph cache.

The complete full-K outer objective remains in the coordinator. Workers replay
one active future while the other seven remain cached constants. Returned
parameter gradients are added in original **(start, future)** order; workers
do not pre-sum their contributions. CPU gradient transport preserves bytes.
The coordinator performs the original direct real-pair term, clipping and Adam
step. No new critic/value credit is introduced.

The no-grad pass also distributes futures, preserving each future's original
row geometry. It records a shared setup-tape prefix plus one segment per
future, then concatenates those segments in future order. Original future-0
advanced RNG state is restored. The unchanged full-K aggregation consumes the
reassembled cache. This avoids enlarging GEMMs along the row axis.

For equal per-future VJPs, the serial recurrence
`A_0 = g_0; A_k = float32(A_(k-1) + g_k)` is identical in the coordinator.
Byte copies introduce no floating-point arithmetic. Hash-randomized reward
dictionary ordering is aligned through the coordinator's TreeSpec, not worker
leaf positions. Final-future spill counters are returned to preserve telemetry.
Claims are qualified on gpuA/update 181; cross-device bitwise identity is not
asserted.

Measured phase times, including process/snapshot/transport overhead:

* Nine rows N2: **110.535 s no-grad + 340.346 s replay/VJP + 3.307 s other**.
* Nine rows N4: **83.352 s + 262.449 s + 4.955 s**.
* Six rows N4: **71.341 s + 212.387 s + 3.245 s**.

## Memory and VJP-slot experiments

Initial `nvidia-smi`, 2026-10-28 23:09:03: RTX 4090 D, **49,140 MiB total**, 1 MiB
used, no compute processes. Later, other workloads came and went. Timings are
single-update measurements under that shared load, not isolated repeat means.

The coordinator releases unused cached blocks before spawning and rejects a
budget above .60, accounting for current coordinator reservation, worker
allocator limits and **768 MiB per worker and coordinator** for contexts and
libraries. The benchmark samples NVML/nvidia-smi every two seconds and
attributes allocations through the owned PID receipts. Owned and whole-device
peaks are distinct: e.g. N2 owned 50.63% while whole-device usage reached 61.69%.
Per-worker exact allocator peaks are also retained in `memory.json`.

Optional `WMS_SPEED_FUTURE_VJP_SLOTS=M` bounds simultaneous large backwards;
default M=N adds no gating. Gated workers use
`WMS_SPEED_REPLAY_FORWARD_MEMORY_FRACTION` before the VJP, acquire an owned
directory token, raise their allocator cap, compute, free the graph/cache,
lower the cap, then release the token. CPU/CUDA unit tests cover this path.

The observed **end-of-forward** reservation of 855,638,016 bytes is **not** a
bound on transient forward peak. Trials using idle fraction .018 failed a
324 MiB WM allocation before reaching the VJP; N8 idle-.024 and idle-.028 trials
also did not complete. These incomplete gated recipes have no update-time or parity
claim and are not the qualified N4 recommendation. Their receipts/diagnostics
are preserved rather than counted as speed results.

An initial monitor conservatively counted another workload against the owned
.60 cap and stopped an attempt. A later N4 attempt was stopped when other
workloads drove physical headroom low. Incomplete attempts and their STOPs
are preserved under `failed_shared_monitor/` and `failed_physical_headroom/`.
The corrected runner budgets the specific recipe against current use and
requests owned STOPs above .60 owned usage or below 512 MiB physical headroom.
Other processes were not signalled.

## Projection, verification and artifacts

Update-only projections use `150 * wall_time_s`, consistent with POLSPEED.
The end-to-end estimate additionally includes 150 observed post-update
rescores, 15 native panels, one phase-update10 gate panel, and uncached startup.
Panel cost is estimated from the measured initial native+gate pair after
subtracting a matching cached startup. It assumes all 150 updates are allowed
to complete; changing scenes, load and future actor behavior are not modeled.

Verification already completed:

* Original LONGMEM suite: **25 CPU passes, 2 CUDA skips** across the full run
  and two checkpoint-test reruns with the temporary-output allowlist corrected.
* Parallel/preset CUDA qualification: initial **7 passed in 37.92 s**.
* Expanded qualification: 9 passed and one allocator-device-index API failure;
  explicit CUDA index fixed it; the gated CUDA retest passed in **15.11 s**.
* Local gated replay and Linux orphan-STOP checks passed; last-row selection,
  missing-None/signed-zero parity and peer-OOM diagnostics: **3 passed**.
* Final expanded CUDA/CPU/report qualification: **13 passed in 55.87 s** on
  gpuA (`qualification5.log`), including both gated CPU/CUDA cases and the
  report/lifecycle checks.

The comparator selects the **last** row for update 181: the trainer writes
first, then the CLI appends independent rescore fields. The initial 151-leaf
trainer-only comparator was corrected to all 158 leaves and all completed
runs were re-compared against both reference logs.

Worker PIDs/traces/STOPs live under `out/<recipe>/future_workers/<uuid>/`.
`one_update.py` uses the CLI's registered two-update dry run and writes an
owned STOP immediately after update 181; exit 1 with `own-run STOP requested;
checkpoint saved` is the expected completed-one-update outcome. Linux orphan
workers create their pool STOP if their recorded parent disappears. Windows
venv launchers can insert an intermediate parent, so that detection is Linux-only.

Repeatable tools: `benchmark.py`, `finalize_evidence.py`, `summarize_report.py`
and `stop_owned.py` under `tools/polspeed2/`. `stop_owned.py` writes only task-owned
STOPs and verifies recorded PIDs/GPU allocations, without signals/name matching.
Committed compact evidence is under `tools/polspeed2/evidence/`; large transport
snapshots are removed after use, while checkpoints and per-future gradient
tensors remain in the owned gpuA output.

Final STOP verification: **58 owned pools stopped, zero live recorded owned
PIDs, zero owned GPU allocations**. `STOP_RECEIPT.json` records the exact PID
set and checks. Only task-owned STOP files were written; no signals or
process-name matching were used.
