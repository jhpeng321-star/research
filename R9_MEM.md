# R9 hybrid-quadrature memory (S-MEM), 2026-10-22

## Cause
`hybrid_log_masses` (src/wms/model/merge_pivot.py) built z/lw over [cars, M knots x order] with M ~ 5T (T = path length,
up to ~2120 for F2 pre-window prefixes), then looped over all T steps keeping ~12 [cars, M*order] float64 tensors per step
for backward: O(cars x T x 5T x order) memory; the unique_consecutive only removed knot columns identical in ALL rows.

## Change (commit 9a6619ed + 567846ff follow-up "always checkpoint")
`hybrid_log_masses(..., node_budget=None)` now delegates to `_hybrid_scan` / `_hybrid_block` (same file, next to it):
1. per-row compaction of live (non -inf) steps (vetoed steps have zero hazard; events -inf, survival carried forward, 0 before
   the first live step) -- exact;
2. per-row removal of zero-prior-mass integration intervals (coincident knots; the knot set incl. the 0-knot for vetoed steps
   is unchanged, same Gauss order, same ends/atoms/-inf handling) -- exact;
3. persistent-z nodes are independent: processed in groups sized by `_HYBRID_NODE_BUDGET` (1.5e7 elements of one
   [rows, live steps, nodes] block, ~1.5 GB), joined by a log-sum-exp, each group wrapped in `torch.utils.checkpoint`
   (use_reentrant=False) whenever grad is on, so a call retains only its inputs; the time loop became a cumsum.
Quadrature order, refinement certificate (`hybrid_quadrature_report` untouched), atoms: unchanged. No floors/truncation.

## Equivalence (tests/test_merge_pivot_memory.py, old implementation kept as private `_reference`)
8 tests, all pass (5 min): values rel. diff <= 1e-10 (observed ~1e-15 class), gradients w.r.t. c, tau, w <= 1e-8 relative;
cases: sparse vetoes, all -inf rows, +inf atoms (first/mid/last), repeated values, 1-D and 3-D leading dims, T=2100 prefixes
(shared live columns with grad; ragged rows values-only), order 64, tiny node budgets (1, 777, 3000) forcing many groups.
Existing suites: `pytest -q tests/test_merge_pivot.py tests/test_merge_pivot_hybrid.py tests/test_merge_pivot_pre_window.py
tests/test_merge_pivot_runtime.py tests/test_merge_pivot_training.py tests/test_r9_exact.py tests/test_r9_fit_exact.py
tests/test_r9_runtime.py tests/test_r9_support_audit.py` (run on the first version, before the always-checkpoint tweak):
238 passed, 69 errors, 1 failed. All 69 errors = missing local fixture `data/r9_train/wm_s_stage300.pt` in this worktree
(test_r9_exact / test_r9_fit_exact / test_r9_runtime); the 1 failure (`test_runtime_risk_cohort_does_not_drop_first_step_zone_entry`)
is a workspace-path refusal ("path outside workspace refused: E:\Temp\pytest-...") -- neither touches hybrid_log_masses.
The hybrid/pivot/training suites (test_merge_pivot*.py) pass in full. The always-checkpoint tweak was re-covered by
tests/test_merge_pivot_memory.py only if re-run; see "Open" below.

## Memory
CPU (real 30 s batch-1 local smoke): previous agent measured >12 GB (order 8, killed) / 16 GB OOM (order 32) uncompacted,
8.5 GB with column compaction only. New code: local run reported working set 0.4 GB at 512 s CPU time (process then stopped
by me to apply the checkpoint tweak); final-code CPU peak not captured (result file empty at deadline).
GPU (gpuR RTX 4090 24 GB, real TRAIN batches from v5_native/genrand_plain, round-8 stage300 init, round-9 config,
120 s stage, truncation 120; probe = forward+backward, no optimizer step; scripts/r9_mem_probe.py, r9_mem_trial.sh;
dir /root/autodl-tmp/c4/v5/memtrial_r9_bb2beb77):

| batch | stage | result (torch.cuda.max_memory_allocated) |
|---|---|---|
| 4 | 120 s | CUDA OOM at 23.2 GB, inside `residual_net._pair_bias` (rollout forward), BEFORE the first loss/hybrid call (composer_calls=0); identical with old-vs-new hybrid code (3 runs) |
| 2 | 120 s | CUDA OOM at 23.5 GB, same place (a concurrent b1 job held some GPU memory in the first attempt; the solo rerun out8 also OOM'd) |
| 1 | 120 s | 11.5 GB step 0, 13.3 GB step 1 (reserved up to 23 GB: fragmentation); batches without scored merge cars have no grad_fn (probe records memory anyway) |
| 4 / 300 s | not run (120 s already OOMs in the rollout) |

For comparison round 8 (`runs/c4_wm_v10_round8/train_log.jsonl`) peaked at 2.2 GB (120 s) and 4.2 GB (300 s).

## Verdict
The hybrid quadrature is no longer the memory driver (O(T^2) removed; per call graph = inputs only; one 1.5 GB block at a time),
but batch 4 at 120 s / 300 s DOES NOT FIT on the 24 GB card, and the OOM occurs in the WM rollout forward before any hybrid call.
Round 9 trains only merge_pivot params (everything else frozen) yet retains >10 GB per sample for 120 steps: hypothesis (not
verified) -- the first_passage decision output feeds the on-policy rollout state, so the whole frozen rollout graph is kept
for backward (round 8's merge path did not). Needs a separate fix (e.g. detach the pivot output from the rollout state / score
pivot on the no-grad recorded-geometry rollout, or gradient checkpointing of the rollout) or batch 1 with short truncation.
Launching the gate as written (batch 4, 60/120 s) will OOM.
Open: re-run tests/test_merge_pivot_memory.py on the final commit (the checkpoint-always edit came after the 5-min run).

## Status at owner pause (2026-10-22 ~17:10)
* Merged c4/v10r9fix (clean). Hypothesis CONFIRMED: the first_passage lane straight-through weights (`merge_pivot_lane_weights`) were fed into the next rollout step's residual_kin (wms.py `forward`, ST adjoint), so the whole frozen rollout graph was retained (OOM in residual_net). New `model.merge_pivot_state_st` (default true = legacy; round-9 config sets false) keeps the hard forward value exactly (adds `w - w.detach()` = 0) and cuts only that pathwise adjoint; unit test + config test added; 241 merge-pivot/route/memory tests pass on the merged branch.
* GPU b4 (probe, real TRAIN batches, state_st=false): 120 s peak 1.66-1.70 GB, 300 s peak 3.29 GB (was OOM >23 GB at b2/b4). Caveat: probe steps hit NOGRAD chunks and zero-mass inf (masked in probe only), so backward through every chunk was not exercised; rerun out12/out13 (anchored probe) was in progress and killed at pause (partial rows: 120 s 1.66 GB, 300 s 3.29 GB).
* NOT done: magnitude of the removed pathwise gradient on a real batch (compare mode r9_mem_probe.py R9_PROBE_COMPARE=1 written; local finite-mask run inconclusive/killed). memtrial dir left in place on gpuR; no processes of mine remain.
