# S-POLSPEED2 — ordered parallel future execution

Branch: `c4/polspeed2`, based on `c4/polspeed` `f6dea57b`.
Owned remote output: `/root/autodl-tmp/c4/polspeed2_20261028`.

Initial gpuA inspection, 2026-10-28 23:09:03 server time: RTX 4090 D,
49,140 MiB total, 1 MiB used, no compute processes. Aggregate cap: 0.60
(29,484 MiB including the coordinator and worker CUDA contexts).

Measurements in progress. **N=2, original nine-row geometry: 454.188 s/update,
24,882 MiB owned GPU peak (50.63%), 2.24x versus 1,018 s; projected 150 updates
18.92 h.** The initial comparator saw 151 trainer leaves; final qualification
also compares the later CLI row with independent rescore quantities.
The six-row algebra preserves the mathematical objective; floating-point and
telemetry parity are measured separately from unchanged-row parallelism.

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

`WMS_SPEED_FUTURE_VJP_SLOTS=M` (default N, no additional gating) optionally
keeps N replay forwards concurrent but permits only M live recompute/backwards.
Measured on the actual nine-row update: a replay forward reserves 855,638,016
bytes, versus 12,132,024,320 bytes peak for backward. This permits a four-worker
pipeline without changing segment 8 or row geometry: two peak-sized backwards
plus two small forward graphs. Workers use the lower
`WMS_SPEED_REPLAY_FORWARD_MEMORY_FRACTION` before their VJP, acquire an atomic
owned-directory token, raise their allocator limit, compute the VJP, free the
graph/cache, lower the limit, then release the token. Unexpected retained GPU
storage fails closed without releasing a high-memory token.

All flags are default off. Existing segment checkpointing and all eight FGA
flags remain available. `pip install '.[parallel-futures]'` installs the optional
cloudpickle dependency; only trusted owned snapshots are deserialized.

## Memory / lifecycle

`WMS_SPEED_FUTURE_MEMORY_FRACTION` bounds each replay worker's CUDA allocator;
`WMS_SPEED_FORWARD_MEMORY_FRACTION` bounds each no-grad worker (default .04).
Before spawning, the coordinator releases unused cached blocks and rejects a
budget above .60 of device memory, counting its current reserved pool, the
phase-specific allocator limits, and 768 MiB per worker **and coordinator** for
contexts/libraries. The benchmark samples nvidia-smi every two seconds,
attributes allocations using the owned PID receipts, and records owned and
whole-device peaks separately. A recipe-specific launch budget accounts for
other workloads; owned usage above .60 or less than 512 MiB physical headroom
requests owned STOPs.

Worker PIDs, execution traces, logs, peak allocator measurements and STOP files
are under each run's `future_workers/<uuid>/`. Pool shutdown writes its STOP
file and waits for its own workers. Large transport scratch files are removed
after use; per-future gradient results and execution receipts are retained.
`touch /root/autodl-tmp/c4/polspeed2_20261028/STOP` stops the owned benchmark
cooperatively; individual CLI runs also accept their own output-directory STOP.
Linux workers also create their pool STOP if their recorded parent disappears.
Windows venv launchers can insert an intermediate parent; automatic reparenting
detection is therefore Linux-only. Normal STOP shutdown is supported on both.

The benchmark invokes the registered two-update CLI dry run and writes an owned
STOP immediately after update 181. Thus exit status 1 / `own-run STOP requested;
checkpoint saved` is the expected completed-one-update outcome. No second
update runs. Startup native/gate scorers can reuse previously **measured**
results keyed by actor tensors, WM identity, panel, full scoring configuration
and row preset; those scorers restore all RNG and do not affect training.

An initial conservative monitor stopped an owned attempt because it counted
an unrelated workload against the .60 limit. Its incomplete output and STOP
are preserved under `failed_shared_monitor/`; measured worker allocator peaks
were 11,837,503,488 bytes each. Completed N2 used the corrected owned-PID
monitor and stayed below .60 even while whole-device usage reached 61.69%.

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
* Gated four-worker CPU qualification plus imported preset: 6 passed, 3 CUDA
  skips. Additional gated replay + Linux orphan-STOP tests: 2 passed locally.
* gpuA torch 2.8.0+cu128: initial replay/parallel-forward/preset suite 7 passed
  in 37.92 s. Updated gated-worker qualification is logged separately.
