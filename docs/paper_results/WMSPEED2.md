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

CPU regression suites: **79 existing tests pass, zero skips**, plus **3 focused
tests pass, zero skips**. Focused tests cover host insertion high-water across
rows, a real frozen MAPPO bridge through new slots and exits, and trajectory
comparison including an exit movement while excluding virgin placeholders.

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

Use a fresh tag per attempt. All commands above hide CUDA before starting Python,
run inference/tests only, and write solely to the WMSPEED2 root. The helper does
not start GPU qualification or training.

## Registered hot-swap recipe for S-COOPMWM

1. **Use a separate immutable worker source root.** Stage the existing qualified
   CPU36 snapshot plus this commit's `fused.py`, `plan_wm.py`, `speed2_runtime.py`,
   and `plan_service.py`. Keep the same FULL-v4 model, MAPPO export, frozen cost
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

This task supplies an opt-in CPU-qualified amendment and the worker startup gate.
It does not activate a replacement on either occupied GPU. SUMO simulation charge
for the work is **zero**.
