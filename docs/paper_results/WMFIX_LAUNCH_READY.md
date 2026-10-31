# WMFIX launch readiness — 2026-10-31 ~09:30 (resume session; MAIN reviews and launches)

Status: **Gates (a),(c),(d) CLOSED PASS; gate (b) feasibility RUNNING on gpuA.**
MAIN/Claude replaces the unavailable Astra REVIEW.json after FEASIBILITY lands.

## Closed since the draft (receipts in WMFIX_RESUME2/SHARD_STAGING/RETARGET.md)
1. (c) Zero-head panel COMPLETE 36/36 + DONE.json. gpuA PID 100400 (0-27, 32-35)
   + gpuR shard PID 484326 (28-31; OOM at 0.55 fraction -> relaunched at 0.85 with
   expandable_segments; batch 31 needs a contiguous 3.92 GiB). Cross-device check
   on double-computed groups 33/34: max single-cell |dJ| 225/272 veh-s but
   group draw-medians IDENTICAL (rel diff 0.0) and every comparator metric uses
   medians -> mixed provenance immaterial; flags identical on ALL36 and NATIVE32.
2. Retarget decision (ZERO_HEAD_COMPARE.json c5210190...): upstream paired MAE
   120-300 s WM2 796.3 vs entry 771.3, SE 90.7 -> gone=false changed=false,
   residual REMAINS -> head training GO. Attribution: entry path removes source
   error (300 s 584->82.5 veh-s) while upstream survives (1457->1393) — the
   bundle confound is isolated. Paired J worsens under entry alone (unmasked
   cancellation): head must beat the step_000000 ENTRY baseline; verdict panel A
   keeps the three-way WM2/entry/head attribution.
3. runtime.py import-shim fix (7b32125e): grouped_feedback needs tools/wmrequal
   on sys.path; feasibility AND train would crash at launch without it. Local
   tests re-imported OK; gpuA tree synced (runtime sha 54900efd...). No verdict
   producer binds runtime.py (checked B/C1 closures). HARNESS_CONTRACT.json
   rebuilt against the fixed tree: f5548777... (supersedes b93bd87c...), schema
   wmfix_harness_contract/1, source_clock_match true, residual_remains true.
4. (a) TRAIN_PREPARED.json regenerated earlier (7bbd41f3...); (d) B/C1 producers
   bound (verdict_plan.json f47a8f4c..., C1 fail-closed blockers by design).

## Running gate
- (b) real TRAIN feasibility, gpuA PID 130682/130684 (launched 09:2x, cuda;
  box had zero compute apps). Checks: 12x2x{120,300}s panel teacher ceiling,
  frozen-rerun noise, finite-difference gradient, one Adam step improving total
  objective AND paired J beyond noise, admission ablation, frozen bitwise parity,
  clock contract match. Expected duration ~40-100 min; writes FEASIBILITY.json
  with launch_ready true/false + blocker. If launch_ready=false: STOP, report,
  no training.

## Exact launch command (MAIN, after FEASIBILITY launch_ready:true + REVIEW)
Unchanged from the draft: gpuA, /root/autodl-tmp/c4/wmfixmerge20261031, timeout
21570s, `tools.wmfixmerge.train --enable-merge-response --conserve-queue
--learned-admission --device cuda --prepared TRAIN_PREPARED.json --config
/root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml
--checkpoint /root/autodl-tmp/c4/restart20261026/arm/loop/wm/c02/wm_s.pt
--controllers /root/autodl-tmp/c4/overnight20261028/W5_COMBINED/DESIGN.json
--corrected-report WMREQUAL3.md --feasibility FEASIBILITY.json --review REVIEW.json
--out train`. Feasibility/train now import-safe end-to-end after the shim fix.
GPU budget <= 6 h, every-update immutable checkpoints; a single-update CUDA
microbenchmark remains a MAIN pre-launch requirement (CPU cannot certify it).

## Post-training verdict (one command; unchanged)
quartet_candidate on gpuR with FINAL checkpoint + candidate_config.json -> copy
to gpuA `quartet_candidate/` after COMPLETE.json; then
`tools.wmfixmerge.verdict --plan tools/wmfixmerge/verdict_plan.json --train-dir
.../train --out .../verdict --execute --device cuda`. C1 stays BLOCKED until the
unseen-PI/unseen-wall cohort + lineage runner are staged (fail-closed by design).

## Schedule
Panel+retarget finished ~4 h earlier than the single-track draft projection
(gpuR shard + faster late batches). Feasibility lands ~10:10-11:10; with a
REVIEW by ~11:30, training 12:00-18:00, verdict ~19:30-21:00 — inside the
owner's evening target with margin for one restart.
