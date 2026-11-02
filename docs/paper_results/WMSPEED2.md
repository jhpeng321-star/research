# WMSPEED2 — accelerate the frozen S-COOPMWM planner

## In plain language

The planner spends most of its time considering empty vehicle slots. The first
qualified observation has 54 vehicles, but each imagined world carries a
1,600-slot roster. Lane-change and gap calculations build pairs of these slots,
so padding is especially expensive: 1,600 squared is 2.56 million pairs per row.

The fix skips never-used trailing slots during model and controller geometry
calculations, then restores the full roster before memory, insertion, and cost
accounting need it. A host-side insertion ledger tells us when the working prefix
must grow. This preserves each vehicle's slot identity and avoids reading a GPU
mask back to the CPU on every step.

On the small CPU-only H60/K2/width-4 comparison, wall time fell from **229.25 s to
15.10 s: 15.18 times faster**. Throughput rose from **4.19 to 63.56 imagined
future-steps/s**. Team costs, candidate ordering, presence masks, lanes, positions,
speeds, and accelerations all matched exactly on this comparison.

The initial capacity check (2026-11-01 21:36:30) found both GPUs occupied, so CPU
work came first. On recovery at 22:35 both GPUs were idle with no compute processes.
The requested H120/K4/width34 GPU comparison then passed with **zero cost and
trajectory error and identical ranking** on both GPUs:

- gpuA: **95.679 → 31.810 s, 3.008x**, 341.14 → 1,026.08 future-steps/s.
- gpuR: **97.743 → 32.877 s, 2.973x**, 333.94 → 992.78 future-steps/s.

These timings include copying qualification trajectories to the CPU; they compare
the same fixture, roster, threads, source sampling, and padded candidate width.
They are evaluator batch measurements, not complete CEM-search plans/hour claims.
H300/K8/width10 also passed with zero cost/trajectory error and identical ranking:

- gpuA: **136.321 → 58.456 s, 2.332x**, 352.11 → 821.14 future-steps/s.
- gpuR: **140.592 → 60.470 s, 2.325x**, 341.41 → 793.78 future-steps/s.

All four comparisons are retained as `coopmwm/WMSPEED2_gpu{A,R}_gpu{120,300}cold1.json`,
with source manifests, owned PID identities, environments, and per-future costs.
Warm/extended-plan qualification is tracked separately below.

The implementation is opt-in through `GPU_BINDING.json: "speed2": true`.
Activation requires a new worker's startup cost/ranking/trajectory qualification.
The later transfer add-on (`"speed2_transfers": true`) also batches acceleration
table writes and newborn parameter transfers. Its full-roster qualification gives
**4.529x warm GPU speed-up at H120** (149.839 →33.085s) and **2.496x cold at H300**
(137.542 →55.111s), with exact0 costs/trajectories and identical ranks/draw IDs.
These are paired evaluator measurements; the add-on's separate incremental
measurements and deployment recipe are below.

## Measured CPU hotspots

`cProfile` measured one complete fixed H60/K2/width-4 batch, with CUDA hidden,
one Torch/BLAS thread, and process niceness 19. Bootstrap is outside the timed
region. The baseline took 226.2208 s. Self-time percentages below are exclusive;
cumulative times include callees and must not be added to self-time shares.

- `physics/lc2013_geometry.py:53 neighbour_displacements`: **25.98% self**,
  58.76 s self / 78.63 s cumulative, 480 calls.
- `physics/lc2013.py:110 _neighbours`: **10.14% self**,
  22.93 s self / 111.34 s cumulative, 480 calls.
- `physics/junction_merge.py:61 target_gaps`: **6.56% self**,
  14.84 s self / 28.46 s cumulative, 837 calls.
- `physics/neighbours.py:41 lane_neighbour`: **6.07% self**,
  13.72 s self / 23.88 s cumulative, 1,263 calls.
- `control/lanes.py:124 _target_gap_ok`: **4.59% self**,
  10.39 s self / 18.20 s cumulative, 540 calls.
- Tensor `where` calls account for **16.80% self** and tensor `min` for **7.56%**.
  They are also included in their Python callers' cumulative time.
- LC2013 `lane_change` totals **120.27 s cumulative, 53.16%** of the batch.

The companion statement-timing run took 193.6453 s; ordinary run-to-run CPU
variation means it is diagnostic, not the denominator for a paired speed-up.
Its inclusive statement attribution identifies:

- `lc2013.py:231/232`, the two neighbor calls: **46.95% / 45.33%** of
  `lane_change` time.
- `lc2013_geometry.py:69`, pair-distance/link arithmetic: **40.30%** of
  `neighbour_displacements`; line 77's preceding-edge calculation: **18.34%**.
- `control/lanes.py:131/132`, pair-distance and leader reduction:
  **27.64% / 33.79%** of `_target_gap_ok`.

This is primarily padded pairwise geometry. The inherited bridge already calls
the actor once over the complete candidate/future batch at each decision tick.

## Implementation and numerical boundary

- `tools/coopmwm/speed2_runtime.py` observes the source process's existing
  `_commit(state, work, new_mask)` host insertion pairs. A monotonically increasing
  high-water mark includes every initial slot and every inserted slot in every
  batch row, including slots belonging to vehicles that later exit. The hook is
  restored when its rollout finishes.
- `tools/coopmwm/fused.py` uses that bound for its existing compact-forward proxy.
  Full virgin-slot defaults are restored before source insertion. Full state is
  also restored before the recurrent bridge callback. There is no new per-step
  `nonzero`/`item` device read in this optimized bound calculation. The untrimmed
  path also avoids computing an unused live-slot bound.
- Pure observation, lane-feasibility, lane-choice, safety, and execution-cap
  geometry uses the same bound. Results are scattered/padded back before the
  inherited recurrent controller consumes them. Actor and group-encoder batch
  dimensions, controller command memory, dwell clocks, and RSU reductions retain
  the full roster. All CAVs and all candidate/future rows remain batched.
- `tools/coopmwm/plan_wm.py` compiles acceleration tables on the CPU and transfers
  each completed tensor once. This replaces many scalar device assignments.
- `tools/coopmwm/plan_service.py` exposes the opt-in binding and runs old full-roster
  versus optimized startup qualification at K4/H120 and K8/H300. It writes receipts
  before READY and refuses admission on a cost, ranking, draw-ID, or trajectory
  mismatch. Qualification snapshots are held on the CPU to limit GPU memory.

Sampling and source planning run at full capacity **before** compaction. Slot
allocation, future keys, CRN draws, source queues, movement/departure order,
acceleration-profile interpretation, safety formulas, and the qualified team-cost
reducer are inherited. The physical FULL-v4 source at `22cc32f6` is read-only.
The source process still performs its inherited host admission work; this patch
does not claim to eliminate those existing transfers. No compiler or CUDA graph
mode is enabled.

## Verification

The reference evaluator and `FusedModel` are loaded from the hash-checked CPU36
snapshot, separately from the candidate overlay. CPU placement is translated in
the frozen bootstrap/device literals only; both sides use the same CPU runtime.
The qualified cost-source bytecode binding remains active.

CPU regression suites: **79 existing tests pass, zero skips**, plus **6 focused
tests pass, zero skips**. Focused tests cover host insertion high-water across
rows, a real frozen MAPPO bridge through new slots and exits, and trajectory
comparison including an exit movement while excluding virgin placeholders,
legacy/300s four-member interpolation, and empty-memory/observable-field guards.
Latest full-suite snapshot: `coopmwm/CPU_coopmwm20261101_wmspeed2_cpu4.json`.

The extended parity matrix is recorded in the companion evidence receipts.
Every comparison checks per-action/per-future team costs (limit 0.001 veh-s),
candidate ordering, and vehicle trajectories. Presence masks are exact; real
cohort trajectories include the exit movement. Virgin placeholders have no
vehicle trajectory and are excluded from kinematic comparisons.

First-fixture probes use the two baselines plus the first two registered family
profiles, with the registered synthetic own continuation of -0.25 m/s². Other
qualified rosters that lack such family seeds use fixed absolute -1/+0.5 m/s²
group profiles, shared by old and new. Wider batches pad these four real actions
to their declared width, as in the qualification probes. Thus width-34 checks
exercise 34 batch columns, with four distinct candidate costs/ranks.

## Warm memory and longer/larger plans

`evaluate(..., memory=WarmQuery(...), speed2=True)` transports an owner-prepared
observable history and native finite-support per-driver posteriors through the
same frozen WMTEACH sampler. The warm adapter does not fit a new posterior law:
native FULL-v4 `fitted_state_at_start` / `condition_population` constructs the
engineering fixture's causal motion posterior. The inherited learned history
encoder still runs with the full roster; its embedding and shrinkage survive
compaction. Native population fallback, lane continuation, arrivals, CRN row keys,
and sampling order are unchanged between full and accelerated warm comparisons.

The fixed warm engineering fixture is **FIT world969420 at t300**, explicitly
inside the bootstrap's969400..969431 allowlist. It contains94 observed vehicles,
61 one-second history samples per slot over240..300s,4,371 valid samples,
70 motion-conditioned HDV posteriors and9 HDV population fallbacks. Only genuine
observable start/history exports and the frozen model's inferred population are
used. This is separate from the TRAIN-roster cold fixture; it exercises the native
warm model API without fabricating histories for unrelated vehicles.

- Warm GPU R, H120/K4/width34,3 CAVs/120s varying plans:
  **168.265 → 46.079 s,3.652x**,193.98 →708.35 future-steps/s;
  **zero cost and trajectory error, identical ranking and draw IDs**.
- Warm GPU A, H300/K8/width10,4 CAVs/300s varying plans:
  **279.503 →81.154 s,3.444x**,171.73 →591.47 future-steps/s;
  zero cost/trajectory error and identical ranking/draw IDs.
- Warm GPU R, the same H300/K8/width10 four-CAV comparison:
  **289.916 →89.773 s,3.229x**,165.57 →534.68 future-steps/s;
  zero cost/trajectory error and identical ranking/draw IDs.
- Warm CPU, H60/K2/width4,2 CAVs:592.790 →25.626s,23.132x observed wall ratio,
  zero cost/trajectory error and identical ranking/draw IDs. Existing CPU-only
  qualifications were concurrent; this ratio is not an isolated throughput guarantee.
- Cold extended plans: gpuR3 CAVs/120s gives2.933x; gpuA4 CAVs/300s gives2.281x,
  both with zero cost/trajectory error and identical ranking/draw IDs.
- Warm CPU cProfile + statement timing:33.633s for optimized H60/K2/width4.
  Linear layers15.03% self, attention9.06%, `where`5.61%, contiguous5.59%,
  lane-neighbor4.42%, deepcopy4.36%; LC2013 now12.39% cumulative.
  The single learned history-encoder call costs **0.619s/query (1.84%)**;
  native posterior preparation is separately0.618s outside the rollout timer.
  GPU R encoder calls in the paired H120 run were0.370s first-use and0.062s
  subsequent-use, one encoder call per evaluator batch, not per future step.

The long-plan decoder supports arbitrary supplied member count and supplied knot
count/spacing (`knot_s`, default10); tested profiles include2–4 members and300s.
The registered CEM sampling/search limits in `SEARCH` are preserved. The warm
restart owner supplies its separately registered group/duration/search amendment.
Reference and optimized extended-plan tests share that decoder, while preserving
the frozen reference rollout/model/controller implementation.

`WarmQuery.from_record(rt, record)` accepts an optional `warm_memory` JSON payload:
`t0_s`, `driver_history [1,N,H,9]`, `driver_history_mask [1,N,H]`, optional short
`history`/`history_mask`, and `posteriors_by_vid` (native finite-support records).
History slots align exactly with `record.rows`; no driver-default/hidden-setting
field is accepted. The history producer owns causal60s sensing/identity validation
and native posterior construction. Empty payload or fully masked history with no
posterior follows the exact cold path. A warm worker requires an explicit payload,
including `{}` for a legitimate empty history. These interfaces let S-COOPMWM use
its genuine rolling history rather than this task's fixed FIT fixture.

The first two wide warm attempts hit the inherited8GiB bootstrap allocator cap.
Their failures are retained. Subsequent qualification applies the registered
0.65 memory fraction **after bootstrap**, as persistent workers already do. No
other GPU process was present at admission; free device memory was not the issue.

## Evidence and reproduction

- Read-only fixture source:
  `E:/agent-work/c4-coopnative/docs/paper_results/coopmwm/VOLUME_INPUTS_1854.json`.
  All four records are observable TRAIN probe rosters.
- Frozen staging authority: `CPU_coopmwm20261101_cpu36.json` in that same directory.
- Raw CPU profiles and parity receipts:
  `gpuA:/root/coopmwm20261101_wmspeed2/<tag>/`.
- Profile tags: `baseline1` (full), `compact1` (model-only compaction),
  `speed2b` (model and controller geometry), `baseline_lines` (statement timing).
- Paired small comparison: `parity60final`; release check: `parity60release`.
- CPU-suite receipts in this checkout:
  `coopmwm/CPU_coopmwm20261101_wmspeed2_cpu2.json`.
- Every recent run has immutable source overlays, `MANIFEST.json`,
  `ENVIRONMENT.json`, and an owned-PID receipt. Earlier bootstrap/test-harness
  failures are retained at their original tags; failed launches were not reused.

From this worktree, with the existing SSH aliases:

```powershell
# Only first use of a new remote WMSPEED2 root needs --stage.
python -B -m tools.coopmwm.speed2_ops launch profile-new --stage --lines
python -B -m tools.coopmwm.speed2_ops launch parity-new --job-mode parity --H 300 --K 8 --width 4
python -B -m tools.coopmwm.speed2_ops launch ramp-new --job-mode parity --H 60 --K 4 --width 4 --fixture-index 1
python -B -m tools.coopmwm.speed2_ops wait parity-new,ramp-new --wait-seconds 600
python -B -m tools.coopmwm.speed2_ops status parity-new
```

Use a fresh tag per attempt. The commands above hide CUDA before starting Python,
run inference/tests only, and write solely to the WMSPEED2 root. GPU probes need
an explicit `--gpu` and pass the idle-device admission check; the helper never
starts training.

## Registered hot-swap recipe for S-COOPMWM

1. **Use a separate immutable worker source root.** Stage the existing qualified
   CPU36 snapshot plus release `9d06c6e9`'s `fused.py`, `plan_wm.py`,
   `speed2_runtime.py`, `speed2_memory.py`, `plans.py`, `accel_diagnostic.py`,
   `speed2.py`, `test_speed2_cpu.py`, and `plan_service.py`.
   Keep the same FULL-v4 model, MAPPO export, frozen cost
   source, and WMTEACH dependencies. Hash the complete staged worker snapshot.
2. **Admit GPU qualification only when there is genuine spare compute capacity.**
   Check `nvidia-smi` utilization and active owner receipts; free memory alone is
   insufficient. Existing training PIDs remain under the training owner's control.
3. **Create a new binding, not an edit to a live binding.** Copy the current
   registered numerical search, width, memory fraction, and deadline settings into
   the new worker's `GPU_BINDING.json`; set `speed2: true` and
   `startup_qualification: true`, and bind the staged source hashes. Preserve the
   already-qualified baseline's qualification receipt. The new worker performs
   full-versus-optimized K4/H120 and K8/H300 cost/ranking/trajectory/draw-ID checks
   before publishing READY. Archive both startup receipts and measure GPU
   future-steps/s on the same fixture before admission.
4. **Swap the planner at a drained query boundary.** The training owner drains
   and persists the existing broker's in-flight responses, then closes only its
   identity-verified owned broker/services. Start a broker from the amendment's
   immutable source, with `plan_bridge.ROOT` pointing at the *existing* CPU queue
   and `plan_bridge.GPU_ROOT` at the new GPU worker root. The broker's service
   command loads `<GPU_ROOT>/pytools`, so staging a new worker directory alone
   does not redirect an already-running broker. Retain trainer/optimizer/RNG state,
   request identities, completed responses, and all accounting receipts.
5. **Admit the qualified replacement slots through the existing worker-admission
   mechanism.** Keep the registered search parameters, query sampling law, budgets,
   source prior, and execution rules. Check the first response's request digest and
   finite/conservation receipts. Reversion uses a fresh hash-bound baseline worker
   root at another drained boundary; completed queries are not replayed.

### Exact staged release (source commit `9d06c6e9`)

Both hosts already have the identical, immutable source root:
`/root/coopmwm20261101_wmspeed2/release_warm9d06c6e9`.
`SOURCE_MANIFEST.json` SHA256 is
`3d14b177d9f1de7a42978c801972ab298193dc9fa0ed18ae09fc92eb34cc2c99`.
Local receipts are `coopmwm/WMSPEED2_gpu{A,R}_release_warm9d06c6e9.json`.
Staging creates no binding or service. The training owner performs these steps:

1. At its drained boundary, select a **fresh operational root** per worker and
   copy the manifest-bound source files there byte-for-byte; retain the release
   manifest. Create its `forecasts/` directory. Copy its existing registered binding
   into a new `GPU_BINDING.json`, replacing `hashes` with the staged manifest's
   `hashes`, retaining `qualification` and the original `deadline_unix`. Set:

   ```json
   {"speed2": true, "startup_qualification": true,
    "memory_mode": "warm", "memory_fraction": 0.65,
    "batch_width": 34, "batch_width_by_K": {"4": 34, "8": 10}}
   ```

   Widths/fraction above match this qualification; another concurrent allocation
   needs its own admission measurement. Every `hashes` entry is relative to the
   operational root. Do not transplant stale READY/claim/forecast files.
2. Launch the broker from the owner's newly registered immutable broker source,
   setting `plan_bridge.ROOT` to its existing CPU queue and `plan_bridge.GPU_ROOT`
   to that fresh operational root. For slots `gpuA@i`/`gpuR@i`, the service root
   is `<GPU_ROOT>_wi`, while the Python source still loads `<GPU_ROOT>/pytools`.
   Existing `ACCEL_GPU_*` logs are exclusive-create: give this broker a fresh
   local evidence directory via `ops.EVIDENCE` while preserving old receipts.
   The actual service command generated by the broker is:

   ```sh
   nice -n 19 /root/miniconda3/bin/python -B -u -c "import sys;sys.path.insert(0,'<GPU_ROOT>/pytools');from pathlib import Path;from tools.coopmwm.plan_service import main;main(Path('<SERVICE_ROOT>'))"
   ```

3. Before admission require `STARTUP_PARITY_K4_H120.json`,
   `STARTUP_PARITY_K8_H300.json`, and both `STARTUP_WARM_PARITY_*.json` to report
   **0.0** cost and trajectory errors, identical ranks/draw IDs, then READY.
   The service's inherited numeric gate allows 0.001; this amendment requires
   the stronger exact observed result. Preserve `SERVICE_OWNER.json` identity.
4. The genuine rolling-history producer adds `warm_memory` with the ABI above
   before hashing/queuing the request. An explicit `{}` means cold fallback.
   New MPPI/search/group policies remain the search owner's separate amendment;
   port these evaluator/transport changes into its immutable snapshot and repeat
   startup qualification there rather than replacing its new search with CPU36.

This task supplies a cold/warm GPU-qualified amendment and the worker startup
gate. Activation belongs to S-COOPMWM. SUMO simulation charge is **zero**.

## Further speed work after OWNER23:04

The staged `9d06c6e9` release above remains the qualified handoff. Additional
experiments run only in separate immutable WMSPEED2 directories.

### GPU fast-path profile

Torch profiler sampled four consecutive one-step calls after12 wait and2 warmup
steps, with full H120/K4/w34 and H300/K8/w10 warm rollouts. GPU R/H120 used3 CAVs;
GPU A/H300 used4. Each launch first required no GPU compute process and <5% GPU
utilization. Receipt names are `WMSPEED2_gpuR_fastprof120.json` and
`WMSPEED2_gpuA_fastprof300.json`; hash-bound kernel-only summaries end `_kernels.json`.

- H120:32,980 kernel launches in four sampled steps; launch dispatch **13.23%**
  of CPU self-time,5,777 stream synchronizations **8.34%**,6,879 async copies
  **4.02%**. There were2,756 scalar reads and548 `nonzero` operations.
- H300:30,949 launches; dispatch **17.61%**,3,361 stream synchronizations
  **8.45%**,4,338 async copies **3.87%**. There were1,516 scalar reads and324
  `nonzero` operations.
- Actual kernel durations total167.445ms/H120 and118.050ms/H300 for the four-step
  samples. Efficient-attention kernels account for **27.97% /25.56%** of these
  kernel totals; the largest SGEMM kernel accounts for **8.00% /6.61%**.
  Remaining GPU work is spread across small geometry/copy/reduction kernels.

These are sampled, instrumented attribution numbers, not throughput measurements.
Use the kernel-only summaries for GPU percentages: the original Torch key-average
device totals also include overlapping CPU-op attribution and user annotations,
so adding those device columns would double-count work.

The small GPU cProfile companion (`fastcprof60`, warm H60/K2/w4) takes10.58s:
LC2013 is22.88% cumulative, the bridge13.87%, and Python acceleration-table
construction4.00% self. This motivates graphing the fixed-roster neural calls and
bulk time-axis table construction while preserving the exact interpolation law.
Initial graph harness failures (Observation tree registration, then CUDA capture
rejecting host-to-device construction of normalization constants) are retained;
neither attempt changes the staged worker.

### Measured additional trials (relative to eager speed2)

- **Bulk time-axis acceleration tables:** warm GPU A H120/K4/w34/3 CAVs,
  **44.987 →35.929s,1.252x** (725.54 →908.47 future-steps/s); warm H300/K8/w10/
  4 CAVs, **81.175 →75.788s,1.071x** (591.31 →633.35 future-steps/s).
  Both have exact0 costs/trajectories, identical ordering and draw IDs.
  Cold CPU H60/K2/w4/4 CAVs also matches exactly (18.039 →17.279s,1.044x).
  CPU raw/mask/residual tensors match bitwise at4/34/80/136 rows, including empty
  members, fractional residual windows, unknown IDs, background overwrite, and
  5-second knots. Table-only construction is37.53–66.89x faster; this is **not**
  the end-to-end speed-up. Receipts: `tables120`, `tables300`, `tables_cpu60`,
  and `tablecpu1` on GPU A.
- **CUDA graphs of the actor and RSU encoder:** after literal-device-constant
  caching, warm GPU R H60/K2/w4 passes exact0 at **10.273 →8.883s,1.156x**;
  H120/K4/w34 passes exact0 at **46.030 →45.652s,1.008x**. The latter is too
  small to establish a material wide-batch gain. Each graph is captured once,
  then replayed60/120 times; input and output ownership are preserved by copies.
  Physics/source/controller memory and draws remain eager. Capture overhead is
  included. Receipts: `actorgraph60v4`, `actorgraph120v4_snapshot1` on GPU R.
- **Batched newborn parameter writes:** warm GPU A H60/K2/w4/4 CAVs passes exact0
  at **9.392 →8.578s,1.095x**. Original native `BirthMix.candidate` calls and
  ordering remain; newborn lane/CAV reads and parameter scatters are batched.
  A further trial keeps the read-only birth population on the CPU, where its
  native identity-keyed discrete sampler already runs, avoiding repeated scalar
  GPU reads. H120/K4/w34 is exact at **44.577 →41.749s,1.068x** (732.22 →781.82
  future-steps/s), independently of the table optimization.
- **Combined table + newborn transfers:** warm GPU R H300/K8/w10/4 CAVs is exact
  at **80.707 →71.211s,1.133x** (594.74 →674.05 future-steps/s). Warm CPU H60 is
  exact at26.963 →26.708s,1.010x; cold CPU H120/K4/w4 on qualified roster1 is
  exact at102.548 →103.034s,0.995x. This add-on targets GPU transfer overhead;
  the paired CPU measurements do not establish a CPU throughput improvement.
- **Long graph run:** warm GPU R H300/K8/w10 is exact at81.729 →80.865s,1.011x.
  Together with H120's1.008x result this does not justify enabling graphs in the
  worker release. No `torch.compile` or graph mode is enabled by its binding.

### Opt-in transfer add-on API

The measured bulk-table and newborn-transfer functions now live in
`tools/coopmwm/speed2_transfers.py`. The evaluator accepts
`evaluate(..., speed2=True, transfers=True, memory=...)`; the persistent binding
uses `"speed2_transfers": true` alongside `speed2` and startup qualification.
It propagates to both search and diagnostic evaluation. Newborn slot identities,
native candidate call order, admission ordinals, counts and assigned totals stay
unchanged. Only the read-only birth support's device placement and tensor writes
change; no sampler is rewritten or approximated. Contexts restore their patches
after each evaluation, including failures.

The add-on's startup gate compares full-roster baseline against both optimizations
and requires **0.0** errors, ranks and draw IDs, rather than the inherited0.001
tolerance. The base release's existing tolerance is preserved. Latest CPU suites
are **79+7 passing, zero skips** (`CPU_coopmwm20261101_wmspeed2_cpu5.json`,
`WMSPEED2_gpuA_transfers_tests.json`); the extra native-sampler test checks untouched
slots, CAV/HDV and mainline/ramp births, empty births, counters and patch restoration.

The actual integrated startup API is qualified on full-roster comparisons:

- GPU A warm H120/K4/w34/3 CAVs/120s: **149.839 →33.085s,4.529x**,
  217.83 →986.54 future-steps/s. Receipt `transfers_gate_warm120_snapshot1`.
- GPU R cold H300/K8/w10/4 CAVs/300s: **137.542 →55.111s,2.496x**,
  348.98 →870.97 future-steps/s. Receipt `transfers_gate_cold300`.

Both report `speed2_transfers: true`, `tolerance_veh_s: 0.0`, zero cost and all
trajectory errors, identical rankings and draw IDs. The combined warm H300 and
CPU cold/warm comparisons above independently exercise the same transfer functions.

The first68-action wide-batch attempt completed its rollouts but its trajectory
checker rejected two-chunk versus one-chunk shapes. It is retained as `wide68` and
does not qualify a width change. The harness now aligns future-major action rows
before comparing different chunk widths, and writes costs/timings before that
comparison so a diagnostic failure cannot lose completed numerical evidence.

### Updated immutable release and hot-swap recipe

**Recommended GPU source: `fcc50296`.** Both hosts have
`/root/coopmwm20261101_wmspeed2/release_transfersfcc50296`, with identical
`SOURCE_MANIFEST.json` SHA256
`b7a279ad20aef6df874b4700dadfdd692ff857fbd491135399f39cc5f3bfc5a2`.
Receipts are `WMSPEED2_gpu{A,R}_release_transfersfcc50296.json`.

Use the exact drained-boundary operational-root/broker/admission steps of the
earlier recipe, substituting this release and its manifest hashes. Add
`"speed2_transfers": true` to its binding. Preserve the registered34/10 phase
widths, memory fraction, original deadline, full model/WMTEACH/cost hashes, and
the owner's new search module. Stage **all** files in the manifest, including
`speed2_transfers.py`. A port into the owner's MPPI source needs the evaluator,
`speed2_transfers.py`, `speed2_runtime.py`, and service/diagnostic flag propagation;
do not replace the owner's separately amended search with this CPU36 base search.

Require all four cold/warm startup receipts to have0.0 errors and identical
ranks/draws before READY/admission. They now enforce the exact gate automatically
when the transfer flag is true. Rollback is the earlier hash-bound release at a
new drained boundary, with its own binding; completed requests are not replayed.
The GPU binding does not enable the exploratory graphs or wider batches.

The matched H120 profiler confirms the intended transfer reduction over four
sampled steps: stream synchronizations **5,777 →3,552 (38.5% fewer)** and async
copies **6,879 →4,654 (32.3% fewer)**. Kernel launch counts remain similar
(32,980 →32,877); scalar reads fall2,756 →2,484 and `nonzero` stays548. Thus this
add-on removes substantial newborn transfer overhead, not all inherited source
admission synchronization. Receipt: `WMSPEED2_gpuR_transfersprof120.json`.

### Wider candidate batch and final recovery state

The repaired `wide68v2` trial passes exact0 costs, all trajectories and identical
rankings/draw IDs for **68 distinct smooth three-CAV plans** on warm GPU A at
H120/K4. Two width34 chunks take **87.988s**, one width68 chunk **73.460s**:
**1.198x** additional batching gain,741.92 →888.65 future-steps/s. This trial
isolates width on the eager base fast path; it is not a measurement of combining
width68 with transfers. It batches candidates within one query, not independent
query histories. Receipt: `WMSPEED2_gpuA_wide68v2.json`.

Keep production34/10 until the owner separately admits its new search/batch size.
For a width68 GPU-A amendment, use `batch_width_by_K["4"] = 68` in a fresh binding,
retain the registered K8 width, and require full-roster cold/warm startup exact
qualification and a memory admission measurement on that source. GPU R width68
and cross-query fusion are not qualified by this trial.

At23:53 both GPU inventories show0% utilization,1MiB used, and no compute process.
All GPU experiments from this session have ended. Inventory receipts are
`WMSPEED2_gpu{A,R}_final_recovery_inventory.json`. Two inherited CUDA-hidden CPU
comparisons remain alive on GPU A and were preserved without duplication:
`parity300` PID552087/start892109765 and `parity60wide`
PID552073/start892109667. Their immutable roots remain under the WMSPEED2 root;
they are not dependencies of the completed CPU/GPU qualifications above.

### Fresh-session recovery — 2026-11-02

At10:00 both hosts' WMSPEED2 owner inventories contain **no live task processes**.
GPU A is99% utilized with two owner compute processes; GPU R has two owner compute
processes and23,204MiB allocated despite a0% utilization sample. Neither device
passes this task's idle-device admission rule. No additional GPU probe was launched.
Inventory receipts: `WMSPEED2_gpu{A,R}_resume20261102_inventory.json`.

The two inherited CPU runs completed and their original receipts are now collected:
`WMSPEED2_gpuA_parity300.json` and `WMSPEED2_gpuA_parity60wide.json`. Both report
zero cost error, identical rankings, exact presence masks and zero reported
position/speed/acceleration error, but **fail their original lane gate**. They are
not counted as successful trajectory qualifications or added speed-up claims.
Inspection of the immutable `parity300` harness confirms that it compares lanes
over all1,600 slots, including virgin placeholders; its kinematic comparison uses
only current presence and does not include the exit movement. It does not persist
trajectory tapes, so these receipts alone cannot localize the lane mismatch to
real vehicles versus placeholders. Their failed results remain unchanged.

The later cohort-aware gate above includes exit movements and excludes virgin
placeholders consistently for all trajectory fields. The already-published exact
CPU and cold/warm GPU qualifications use that corrected gate. The recommended
`fcc50296` release and its startup exact-parity requirement remain the handoff;
these older failed probes do not qualify any additional deployment configuration.

Recovery receipt commit: `e1ccd4f4`. A source diff against `fcc50296` confirms no
changes to the numerical evaluator, fused model, runtime, memory/transfer adapters,
plan decoder, service/diagnostic propagation, or focused CPU tests. Existing79+7
passing tests therefore remain the release's test evidence; no repeated inference
or test dispatch was needed for receipt collection and documentation.

The canonical `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md`
already contains the single `S-WMSPEED2 ENGINEERING AMENDMENT` entry, including
the recommended manifest and exact startup gate. It was preserved without a
duplicate entry. Recovery status was appended to
`E:/agent-work/c4-coopnative/HANDOFF_COOPMWM.md`; that was the only file changed
in the training owner's worktree. The existing source-bound hot-swap recipe above
remains the deliverable for its separately registered activation.

### Follow-up recovery — 2026-11-02 16:22 host time

Fresh owner inventories again show **no live WMSPEED2 jobs** on either host.
GPU A is 83% utilized with 45,083 MiB allocated to two other compute processes;
GPU R is 91% utilized with 23,878 MiB allocated to two other compute processes.
Neither device qualifies for another GPU experiment. Receipts are
`coopmwm/WMSPEED2_gpu{A,R}_resume20261102b_inventory.json`, committed as `bb840a69`.

The source diff against `fcc50296` still contains only the inventory helper in
`speed2_ops.py`; numerical evaluator, runtime, memory, transfers and startup gate
are unchanged. The integrated GPU qualification receipts were rechecked: warm
H120/K4/w34 is 4.529x and cold H300/K8/w10 is 2.496x, both with 0.0 cost/trajectory
errors and identical ranks/draw IDs. Existing CPU 79+7 passing evidence applies.
The single registration entry and appended owner hot-swap handoff are present.
No completed probe was rerun, no new inference job was dispatched, and no process
was changed. The recommended release and exact activation recipe remain above.

### Follow-up recovery — 2026-11-02 19:07 host time

Inventories committed as `08ad3330` again find **no live WMSPEED2 jobs**.
GPU A is 97% utilized with 46,811 MiB allocated and two compute processes;
GPU R is 96% utilized with 23,524 MiB allocated and two compute processes.
Receipts are `coopmwm/WMSPEED2_gpu{A,R}_resume20261102c_inventory.json`.
The first GPU-A inventory connection closed before returning a receipt; a single
read-only retry succeeded. Neither host has capacity for another GPU probe.

The worktree was clean at entry on `9b76cdc8`. Comparing the numerical source to
`fcc50296` again finds no changes; only the inventory helper differs under
`tools/coopmwm`. The single canonical registration and the owner's appended
release/hot-swap handoff were verified. Existing exact cold/warm qualification,
79+7 passing CPU tests, and the source-bound recipe above remain the evidence.
No completed comparison was rerun and no new inference or training was dispatched.

### Follow-up recovery — 2026-11-02 22:39 host time

Fresh inventories committed as `304c4852` find **no live WMSPEED2 jobs**.
GPU A is 73% utilized with 44,329 MiB allocated and two compute processes;
GPU R is 16% utilized with 17,121 MiB allocated and one compute process.
Neither device passes idle-device admission, so further GPU speed experiments
remain capacity-blocked. Receipts are
`coopmwm/WMSPEED2_gpu{A,R}_resume20261102d_inventory.json`.

The integrated full-roster qualification receipts were re-read: warm H120/K4/w34
remains 4.529x and cold H300/K8/w10 remains 2.496x, with zero cost/trajectory
errors and identical ranks/draw IDs. Numerical source still matches `fcc50296`;
only the inventory helper differs under `tools/coopmwm`. Existing CPU 79+7 passing
evidence applies. The single canonical engineering-amendment entry and owner's
release/hot-swap handoff were verified and preserved. No completed test or probe
was repeated, no inference/training was launched, and no process was changed.
