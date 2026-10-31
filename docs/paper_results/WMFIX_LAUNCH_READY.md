# WMFIX launch readiness — 2026-10-31 ~11:15 FINAL (resume session; MAIN decides)

Status: **NOT LAUNCH READY. Gate (b) real TRAIN feasibility BLOCKED (fail-closed)
as pre-registered** — see WMFIX_FEAS_BLOCKED.md for the full diagnosis and the
three options (no-change / amended k-step panel-descent pilot / training-only
smoothed admission). MAIN/Claude replaces the unavailable Astra REVIEW.json and
decides; no training was launched by this session.

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

## Feasibility outcome (gate b) — BLOCKED
FEASIBILITY.json (sha 65771d02...) ran to completion on gpuA cuda (~31 min).
PASS: clock contract, frozen-bitwise, finite J/crossing gradients, exact source
conservation, admission ablation, teacher ceiling BOTH horizons (120 s 283->170;
300 s 2185->526). FAIL: FD agreement (analytic 0.032 vs FD 1.18; eps sweep shows
dense hard-event breakpoints — unsatisfiable as written) and one-step panel
improvement (300 s -497 veh-s, 120 s +25). Diagnosis + options in
WMFIX_FEAS_BLOCKED.md. Do NOT run the training command until MAIN resolves the
gate decision per those options.

## Exact launch command (only after a passing gate decision per above)
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
Panel+retarget closed ~4 h earlier than the single-track projection (gpuR shard,
faster late batches): retarget ~09:10, feasibility verdict ~10:45, diagnosis
~11:10. If MAIN picks option (b) now: pilot ~30-60 min, review, training could
still run ~12:30-18:30 with verdict ~20:00-21:30. Option (a) ends the task with
no adoptable fix; option (c) slips past today.
