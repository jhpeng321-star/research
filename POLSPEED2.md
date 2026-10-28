# S-POLSPEED2 — ordered parallel future execution

Branch: `c4/polspeed2`, based on `c4/polspeed` `f6dea57b`.
Owned remote output: `/root/autodl-tmp/c4/polspeed2_20261028`.

Initial gpuA inspection, 2026-10-28 23:09:03 server time: RTX 4090 D,
49,140 MiB total, 1 MiB used, no compute processes. Aggregate cap: 0.60
(29,484 MiB including the coordinator and worker CUDA contexts).

Work in progress. Timings and exactness claims will be recorded only after
measurement. The six-row algebra preserves the mathematical objective; its
floating-point and telemetry parity must be measured separately from replay
parallelism with unchanged row geometry.

## Execution design

`WMS_SPEED_FUTURE_WORKERS=N` (integer 1–8, default 1) opts into independent
Python processes for per-future replay plus VJP. Each worker has a frozen
snapshot of the same actor/WM and its own RNG, tape, mutable rollout switches,
saved-tensor hooks, controller closures, allocator and CUDA-graph cache.
No actor optimizer or dataset is sent to a worker. The coordinator retains
the complete full-K outer graph and passes its VJP vectors into each replay.
Returned gradients are added in the original `(start, future)` order, using
the same device/dtype elementwise additions as the serial implementation.
CPU gradient transport is lossless. Hash-randomized reward-dictionary ordering
is handled by matching the coordinator's TreeSpec, not worker leaf positions.
The final future's spill counters are returned to preserve serial telemetry.

`WMS_SPEED_PARALLEL_FORWARD=1` additionally distributes the no-grad futures.
It preserves the original row geometry instead of enlarging GEMMs. Workers
record a shared setup tape prefix plus one future segment each; the coordinator
reassembles these in future order and runs the unchanged aggregation with
cached futures. The original future-0 advanced RNG state is restored. This
keeps global RNG consumption and the full-K risk/uncertainty reductions.

All flags are default off. Existing segment checkpointing and all eight FGA
flags remain available. `pip install '.[parallel-futures]'` installs the optional
cloudpickle dependency; only trusted owned snapshots are deserialized.

## Memory / lifecycle

`WMS_SPEED_FUTURE_MEMORY_FRACTION` bounds each replay worker's CUDA allocator;
`WMS_SPEED_FORWARD_MEMORY_FRACTION` bounds each no-grad worker (default .04).
Before spawning, the coordinator releases unused cached blocks and rejects a
budget above .60 of device memory, counting its current reserved pool, worker
allocator limits and an additional 768 MiB per worker for contexts/libraries.
The benchmark samples whole-device nvidia-smi memory every two seconds and
requests owned STOPs if the aggregate cap is exceeded. It refuses a new variant
when another workload uses more than 5% of the GPU, pending re-budgeting.

Worker PIDs, execution traces, logs, peak allocator measurements and STOP files
are under each run's `future_workers/<uuid>/`. Pool shutdown writes its STOP
file and waits for its own workers. Large transport scratch files are removed
after use; per-future gradient results and execution receipts are retained.
`touch /root/autodl-tmp/c4/polspeed2_20261028/STOP` stops the owned benchmark
cooperatively; individual CLI runs also accept their own output-directory STOP.

## Local qualification

Python 3.13, torch 2.15.0+cpu, cloudpickle 3.1.2, OMP/MKL=2:

* Parallel replay-only and parallel-forward+replay tests: bitwise equality of
  loss, returns, actor gradients, updated parameters, Adam state and RNG.
* New parallel / imported six-row / two resume tests: 7 passed, 2 CUDA skips.
* Remaining LONGMEM suite: 23 passed, 2 CUDA skips. Its two checkpoint tests
  initially failed the temporary-output allowlist and passed after setting
  `C4_ALLOWED_ROOTS=E:/Temp` (included in the seven-pass command above).
* The first combined local test command exceeded the 120-second harness
  timeout; the subsequent LONGMEM command completed in 118 seconds.
