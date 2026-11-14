# RUN3_READY — WM-S Run 3 launch preparation

Status: **code + chain + tests complete; NOT launched (GPU launch needs owner
approval).** Branch `wms/run3` = `wms/run2b` (run-2 code, merge `d7383f1` of
`wms/trafloss-fix` `9e130ec`) + this run-3 commit **4b3fbefbc0d45cd09cdd90d716f25acb509f3cec**.
TRAIN-only, CPU-only; no SUMO, no GPU, no SSH, no CAL/HOLDOUT/chunk-5/sealed
bytes. No reset/rebase/clean/checkout/stash.

## 1. Traffic-loss review (task 1)

`git merge wms/trafloss-fix` is already in (merge commit `d7383f1`; fix commit
`9e130ec`). Reviewed critically; kept with these decisions:

| question | decision | why |
|---|---|---|
| are the new `metric_scales` sensible? | yes, as per-*bin* typical magnitudes: `mean_speed 10.0`, `density 20.0`, `stopped_vehicles 10.0`, `merge_queue_length 100.0`, `normalized_delay 10.0` | each is the O(1) divisor in `((pred-target)/scale)^2`; run-2 shipped 50-100x smaller values that acted as gains. `residual_cap=5.0` bounds any remaining outlier. |
| `normalized_delay` averaged per bin instead of summed? | **averaged** (not summed) | the other metrics are per-step means; the sum scaled with bin length. With `truncation-steps 20` (10 s) < `traffic-bin-s 30` (60 step) each chunk covers a single partial bin, so mean-vs-sum is only a constant factor; the mean is the horizon-invariant, comparable choice. Scale adjusted 100.0 → 10.0 to match the per-step quantity. |
| logged `traffic` averaged per chunk instead of summed? | **averaged** (`traffic_total / n_chunks`) | the term is added once per 20-step chunk, so the raw sum scaled with the chunk count (run 2's stage-300 was ~30x its stage-10 value from the chunk count alone). The logged `loss` is reconstructed per-chunk too. |

The off-path record is unchanged: `traffic`/`chunks` are written **only** when
`--traffic-loss` is on. `normalized_delay` uses the free-speed-normalised
per-step accumulation divided by the metric scale.

## 2. Multi-stage smoke gate (task 2)

`wms/gate_stages.py` replaces the single stage-10 smoke gate. The chain runs a
short multi-stage smoke (`--unroll-s 10 30 60 120 300 --stage-steps 6,6,6,6,6`,
~30 optimizer steps) with **every** loss term active, then gate-checks it:

- every numeric field finite, gate monotone, `g(0)=0`;
- all six terms (`position,speed,lane,traffic,stability,conservation`) present
  in **every** stage;
- dominance: no term may exceed `100x` the position term;
- growth: no term may grow `>10x` between consecutive stages. The bounded
  per-car terms (position/speed/lane) use the position term's own horizon growth
  as their baseline, so the expected growth of a Huber/CE loss with the unroll
  length is **not** a false positive (confirmed by the CPU dry run: stage-10
  position 8.4 → stage-30 118.3, a 14x jump that is horizon growth); traffic is
  exempt while its 30-120 s ramp opens.

It prints the per-term magnitude table and the per-stage gradient shares, and
fails closed (`GATE FAIL ...`, non-zero exit). Unit tests (`tests/test_run3_gate.py`)
feed run-2's legacy per-stage medians and assert the gate rejects the blow-up,
plus non-finite/missing-term/missing-stage/growth/ramp-exemption cases and the
horizon-growth exemption.

## 3. Per-term gradient-norm shares (task 3)

`--loss-share-logging` (default off; the chain passes it to the smoke) writes
`loss_share.jsonl` per curriculum stage: each term's share of the total gradient
norm, measured on one cheap probe chunk at the traffic-ramp saturation point
(every term live, per-car targets still present) — so no single term can hide.
The gate prints these shares. A dedicated file is used so the shares are logged
without a validation set. Unit-tested.

## 4. CPU dry run (task 4)

Command (real packs, same argv path as the chain; one process, mmap packs,
`--num-workers 0`, threads 4, batch 1):

```
python _out/run3_dry.py --tag stages_cpu_1 --stage-steps 1,1,1,1,1 --batch-size 1
```

`_out/run3_dry.py` shells out to `wms.cli train` with the run-3 flag set
(entries-v2, merge-cal, traffic-loss, projection, residual caps, stability,
conservation metric **and** loss, loss-share logging) over all five stages, then
applies `wms.gate_stages`. Result (`_out/run3_dry_stages_cpu_1.json`):

- rc=0, 5 optimizer steps, all five stages present, every term finite;
- per-stage magnitudes: position 8.4 → 118.3 → 101.5 → 205.3 → 258.5; traffic
  0 → 0 → 0.0097 → 0.0227 → 0.128 (bounded, not 1e7-1e9);
- **peak RSS 3797.1 MB < 6 GB** (`over_6gb: false`);
- per-step seconds 7.4 / 18.2 / 40.2 / 225.5 / 314.9 (CPU, 600-step 300 s unroll
  dominates); wall 1112.8 s.

The gate checked against this exact log with the final gate code passes
(`_out/run3_dry_stages_cpu_1_gate.json` and a re-run of `python -m wms.gate_stages`
on the same log: `status: passed`, no failures, per-stage magnitude table
printed). The share-probe refinement added after that run (probe chunk at ramp
saturation instead of the last chunk) is covered by unit tests; a full final-code
re-run was attempted but starved by concurrent `wms_eval.c1v2` CPU workers
(a stage-10 step that takes 7.4 s alone took 270-651 s), so it was not repeated.

## 5. Tests

`python -m pytest output/wd01/wm_n1/wms/tests -q` → **153 passed, 2 skipped**
(pre-existing TRAIN chunk-4 parity skips) in 296 s. `ruff check` (E4/E7/E9/F) on
the changed modules is clean (the repo config excludes `output/`, so ruff was run
with an explicit `--isolated` selection).

## 6. Code tar and launch

- code tar `_out/code_run3_da567b39e23d.tar` (9.83 MB, `git archive` of the run-3
  speed-fix commit `da567b39e23d`: `output/wd01/wm_n1` + `_out/merge_params.json`
  + `_out/run3_chain.sh` + `_out/run3_dry.py`), sha256
  `9c006ac909c01bd684ba3b0cac2885bf38ee2008d157503abac1dac31dccfd55`
  (`_out/code_run3_da567b39e23d.tar.sha256`). It supersedes the pre-fix tar
  `_out/code_run3_4b3fbefbc0d4.tar` (sha `4da0a0a3…`, kept for provenance).
- GPU chain `_out/run3_chain.sh` (gpuA): flag-presence guard → multi-stage smoke
  + `wms.gate_stages` (all six terms) + `loss_share.jsonl` ≥ 5 stages → an
  **explicit** `--stage-steps 2500,2500,1875,1250,400` projected from the smoke's
  measured per-stage step time (`wms.plan_steps … --stage-steps … --max-hours 10`)
  → full run. The step budget no longer depends on a smoke timed under CPU
  contention: the 10/30/60/120 s stages MATCH C1 exactly and the chain REFUSES to
  start when the projection exceeds `MAX_HOURS=10` (it never silently shrinks the
  steps). Launch: `W=/root/autodl-tmp/wms_run3_20261007/work bash _out/run3_chain.sh`
  (stage the tar as `$W/code_run3.tar`, the LPO window lists, sidecars and
  `merge_params.json` as in run 2).
- GPU time estimate (see §7): with the run-2 CPU hot spots fixed, the per-step
  time is projected back to the C1 measured times (0.74/2.15/4.21/7.81 s at
  10/30/60/120 s; ~19.5 s extrapolated at 300 s). The explicit 8525-step budget
  projects to **≈ 9.1 h**, just under the 10 h refusal cap; run 3 was 15x
  under-trained at 539 steps, so this is the required minimum to match C1. The
  multi-stage smoke is ~30 optimizer steps and expected to take tens of minutes.


## 7. Run-3 CPU speed fix (why run 2 was 6.5x slower than C1)

**Root cause (CPU-bound, GPU util 2-19%).** The dominant cost was not FLOPs but
host/device synchronisation in the run-2-only code paths.  The old
FeasibilityProjection._project_gaps walked vehicles in a Python loop with a
ool(projectable[row, cur]) (and a second ool(...) on the clamp test) per
vehicle per lane per step.  Instrumented on the real batch this is
**~242 device syncs per sim step at batch 2 and ~970 at batch 8** (≈ 19.4k per
10 s optimiser step, ≈ 58k per 30 s step).  Because there is no async on CPU, it
costs CPU time; on the GPU it serialises the stream, so each sync stalls behind
the queue and the GPU idles - exactly the observed low utilisation.  The
entries-v2 insert additionally issued **6 device->host copies every sim step**
(_observe + s/v/length/lane/present/is_cav), each a sync on GPU.  Both scale
linearly with the unroll, which is why the C1->run-2 ratio stayed ≈ 6.5x at every
stage.  (data-loader workers and metrics were secondary: the seconds timer
excludes data loading, and traffic/stability/conservation run once per chunk.)

**Fixes (vectorised, no new host syncs; numerics unchanged).**
- wms/stability.py: _project_gaps rewritten as a closed-form **segmented
  prefix minimum** over each (row, lane) group (sort once, prefix-sum
  L+min_gap, cumulative-max anchor position, masked min).  The per-vehicle
  loop and all ool(...) syncs are gone; the original loop is kept only as
  _project_gaps_reference for the parity tests.
- wms/entries.py: TorchEntryProcessV2.insert advances the renewal clock
  first and returns early when no arrival is due and none is pending, skipping
  the six per-step host copies on the common step.  A test-only
  _FORCE_MATERIALISE restores the old path for parity.
- wms/plan_steps.py: new project() + --stage-steps/--max-hours CLI; the
  chain now uses an explicit budget and fails closed above the cap.

**Per-component profile** (_out/profile_before_after.txt, cProfile, CPU,
batch 2, 30 s stage, all flags; the two halves ran under different host load, so
compare shares/ratios, not absolute totals):

| component | reference loop | vectorised |
|---|---|---|
| _project_gaps tottime | 2.189 s (9.4% of 23.36 s) | 0.169-0.239 s (~1%) |
| device ool() syncs / sim step (batch 8) | ~970 | 0 |
| entries host copies / sim step | 6 | 0 when idle |

_out/bench_step_after.json (min-of-3 fresh-model steps, batch 2) and
_out/profile_step_b2.json are the raw per-flag wall-clock runs; host
contention makes those absolute numbers noisy, so they are evidence of ordering,
not of precise effect sizes.

**Parity tests** (wms/tests/test_run3_vectorization_parity.py, +1
plan_steps.project test in 	est_wms.py):
- vectorised vs reference _project_gaps is exact in float64 (max abs <= 1e-6;
  measured ~1e-14) with identical gap/clamp counts over 300 random cases;
- on float32 it agrees to a few ulps (<= 2e-4 at |s| ~ 200-2600 m) with
  **identical** gap/clamp counts;
- the entries fast-path is bit-identical to forced materialisation over a 60-step
  rollout (masks, states, summary);
- a full trainer step with every run-2/run-3 flag gives the same loss (<=
  relative float32 ulp) and identical projection counters for the vectorised and
  reference projection.

**Test results** (C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest
output/wd01/wm_n1/wms/tests -q): **160 passed, 2 skipped** (pre-existing TRAIN
chunk-4 parity skips); 
uff check --isolated --select E4,E7,E9,F clean on all
changed modules.

**New projected GPU hours.** C1 per-step 0.74/2.15/4.21/7.81 s at 10/30/60/120 s
(300 s extrapolated ~19.5 s) x the explicit 2500/2500/1875/1250/400 step budget
= **~9.1 h**, under the 10 h refusal cap.  The chain still measures the real
smoke and refuses if the projection exceeds MAX_HOURS (the smoke's project
report is in plan_steps.json/projected_hours.txt).
