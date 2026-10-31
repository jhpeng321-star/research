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

## MAIN ADDENDUM 12:45 sharded dispatch (prepared 12:48-13:00, git b7742a11)
Panel A is now shardable WITHOUT touching the frozen measurement: verdict_panels
.validation gained (backward-compatible, verdict.py call site unchanged) optional
`groups=` subset + immutable per-group A_{name}_{i:03d}.json files carrying an
exact candidate binding (fail-closed reuse on mismatch); assembly ALWAYS re-reads
the per-group files, so the one-command verdict later reuses exactly the sharded
values (same score.score_batch path, per-call CRN; determinism proven r1==r0 and
gate-(c) cross-device group checks). New CLI tools/wmfixmerge/verdict_shard.py
(NOT in the producer closure; verdict_panels.py digest changes are recorded in
receipts; B/C1 producer files untouched). Local tests 41 passed (test_wmfix_shard
5 new; 1 pre-existing torch-import box test); box: shard tests 5/5 on gpuA,
smoke parse OK; files sha-verified relayed gpuA+gpuR (verdict_panels 081d2a95,
verdict_shard dc40e995). Panel B stays single-process BY DESIGN (frozen producer
closure: quartet_candidate.py flock + hardcoded OUT). gpuR plan inputs verified
(reference/base_config/controllers/validation_real/e3data OK; ACCEPT_PANEL.json
is created deterministically by accept.py on first use; WMREQUAL3.md /
HARNESS_CONTRACT.json / step_000000.pt are verdict.py-preflight inputs that stay
on gpuA). Immediately after FINAL.json on gpuA:
1. gpuR: reconcile the canary leftovers (MOVE, never delete:
   `mkdir -p ../wmfix_candidate_quartet_canary_20261031 && mv IDENTITY.json
   canary_7_0.json canary.log ../wmfix_candidate_quartet_canary_20261031/` — the
   canary IDENTITY binds an older candidate and would abort the real run), then
   `nohup env PYTHONPATH=src:. /root/miniconda3/bin/python -m
   tools.wmfixmerge.quartet_candidate --candidate <FINAL abs path> --config
   /root/autodl-tmp/c4/wmfixmerge20261031/train/candidate_config.json
   > quartet_candidate.log 2>&1 &` in /root/autodl-tmp/c4/speedhead20261030
   (self-caps 8 GiB; 64 immutable w{b}_k{f}.json; COMPLETE.json at the end).
   Relay FINAL + candidate_config.json hash-verified BEFORE launch.
2. gpuA (parallel): `verdict_shard --groups 18-35 --train-dir .../train --out
   .../verdict --device cuda --memory-fraction 0.35` (out dir pre-created empty).
3. gpuR (parallel to the quartet, 24 GB fits 8 GiB cap + shard):
   same command with `--groups 0-17 --memory-fraction 0.3` after relaying the
   FINAL checkpoint + candidate_config.json (hash-verified).
4. After COMPLETE.json: relay gpuR wmfix_candidate_quartet -> gpuA
   `quartet_candidate/` (producer_b's --candidate-dir), then the one-command
   verdict above on gpuA. C1 producer stays fail-closed (c1_cohort.json/
   c1_runner.json not staged by design); SAFETY creates ACCEPT_PANEL.json.

## Schedule
Panel+retarget closed ~4 h earlier than the single-track projection (gpuR shard,
faster late batches): retarget ~09:10, feasibility verdict ~10:45, diagnosis
~11:10. If MAIN picks option (b) now: pilot ~30-60 min, review, training could
still run ~12:30-18:30 with verdict ~20:00-21:30. Option (a) ends the task with
no adoptable fix; option (c) slips past today.
