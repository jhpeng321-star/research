# WMFIX operation (MAIN launch; CPU preparation first)

The audit/design are in `docs/paper_results/WMFIX_PRIOR.md` and `WMFIX_DESIGN.md`.
All commands run from this worktree root. Use a CPU PyTorch environment for preparation/tests.
No command here invokes SUMO. No training has been launched by the implementation session.

## Inputs and fail-closed gates

- TRAIN: `/root/autodl-tmp/c4/e3data`, only the 12 `SEEDS` in `data.py`, role **fit**,
  native/ours_c1/coopoff_c1 pairs at t300. Old lane-head MAPPO tapes are excluded after WMREQUAL3.
  No validation/pairs300/SPEEDHEAD fitting. Final speed-only MAPPO remains required on SPEEDHEAD verdict B.
- Base checkpoint: `/root/autodl-tmp/c4/restart20261026/arm/loop/wm/c02/wm_s.pt`,
  SHA256 `b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436`.
- Config: `/root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml`.
- Entrant controllers: `/root/autodl-tmp/c4/overnight20261028/W5_COMBINED/DESIGN.json`.
- Corrected report: task-named `E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md`;
  MAIN must copy its exact bytes to the GPU-side `WMREQUAL3.md` below when available.
- MAIN creates existing empty directories `/root/autodl-tmp/c4/wmfixmerge20261031` and its `train` child.
  Persistent exclusive `RUN.json` and immutable update files block accidental duplicate invocation.

Preparation hashes every array opened by the production loader and verifies equal observed histories,
Q0 and recorded due schedules across pairs. Missing request ledgers refuse preparation.
Only exogenous TRAIN request increments are teacher-forced; admissions and new-head motion remain free-running.

```bash
python -m tools.wmfixmerge.prepare --root /root/autodl-tmp/c4/e3data --out /root/autodl-tmp/c4/wmfixmerge20261031/TRAIN_PREPARED.json
python -m tools.wmfixmerge.feasibility --prepared /root/autodl-tmp/c4/wmfixmerge20261031/TRAIN_PREPARED.json --config /root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml --checkpoint /root/autodl-tmp/c4/restart20261026/arm/loop/wm/c02/wm_s.pt --controllers /root/autodl-tmp/c4/overnight20261028/W5_COMBINED/DESIGN.json --corrected-report /root/autodl-tmp/c4/wmfixmerge20261031/WMREQUAL3.md --harness-contract /root/autodl-tmp/c4/wmfixmerge20261031/HARNESS_CONTRACT.json --out /root/autodl-tmp/c4/wmfixmerge20261031/FEASIBILITY.json
```

`HARNESS_CONTRACT.json` is an independently checked `wmfix_harness_contract/1` receipt with
`corrected_report_sha256`, `prepared_sha256`, `native_lanes_sha256`, `scorer_sha256`, `runtime_sha256`,
`source_clock_match: true`, `residual_remains: true`, plus the actual clock-comparison evidence.
This is needed because the inherited recorded labels use left-boundary source rectangles while the
inherited WM scorer samples after insertion. A report's existence alone cannot certify their parity.
If corrected WM2's residual disappears, retarget the design before preparing this receipt.

`FEASIBILITY.json` must be real TRAIN, with teacher upstream ceiling improvement, finite-difference
agreement, a free-running optimizer step improving both total objective **and paired J**, and frozen parity.
The strengthened gate (`panel_gate`) additionally evaluates all 12 groups x both arms at 120/300 s,
measures frozen-rerun noise, and requires the post-step paired-J MAE to beat the pre-step panel
beyond that noise at both horizons. The 48-rollout panel is slow on CPU; MAIN may pass `--device cuda`.
`TRAIN_PREPARED.json` must be regenerated after the anchor-scale rebalance (old files lack
`anchor_scales` and fail closed).
Astra's `REVIEW.json` must identify reviewer `Astra`, status `APPROVED`, and bind
`feasibility_sha256` to the passing file. The implementation session does not generate approvals.

## Exact eventual training command (MAIN only, after gates)

```bash
timeout --signal=INT --kill-after=30s 21570s python -m tools.wmfixmerge.train --enable-merge-response --conserve-queue --learned-admission --device cuda --prepared /root/autodl-tmp/c4/wmfixmerge20261031/TRAIN_PREPARED.json --config /root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml --checkpoint /root/autodl-tmp/c4/restart20261026/arm/loop/wm/c02/wm_s.pt --controllers /root/autodl-tmp/c4/overnight20261028/W5_COMBINED/DESIGN.json --corrected-report /root/autodl-tmp/c4/wmfixmerge20261031/WMREQUAL3.md --feasibility /root/autodl-tmp/c4/wmfixmerge20261031/FEASIBILITY.json --review /root/autodl-tmp/c4/wmfixmerge20261031/REVIEW.json --out /root/autodl-tmp/c4/wmfixmerge20261031/train
```

The checked-in schedule and all three field/ledger/admission flags default OFF. Explicit CLI flags enable the run.
Admission uses the SAME entrance/ordinal-keyed inferred pool draw as later entrant motion (no class-setting lookup).
Physical stop-speed formula and geometric lane limits/nominal vehicle length remain; no new behavioural threshold.
Real feasibility records ledger-only versus inferred-admission/zero-head ablations before the head's own optimizer step.
The passing receipt binds implementation hashes; changing these files invalidates launch approval.
Only `merge_response.*` (1,473 parameters) enter Adam, lr3e-4, FP32. Frozen buffers/parameters are
checked on every export; canonical `load_checkpoint` must reload every tensor exactly.
Each microbatch contains native plus one arm; cycle worlds then arm index, shared random futures,
mix ON, native-lane timeline repair, physical exits, full-fleet/source J. Absolute anchors are crossing/road only.
Curriculum: up to 30x60s, 30x120s, 60x300s. Final completed checkpoint is primary; no validation selection.
Step0 and every completed update are immutable, with optimizer state and source-error telemetry.
`candidate_config.json` enables the head for canonical scoring. `step_000000.pt` is the exact frozen-WM2
zero-response reference for same-config accept.py; its legacy tensors must match the base checkpoint.

## Budget (estimate, not a measurement)

gpuR: 24 GB, <=6 hours total. Native+arm two-row full-300s graph with forward activation checkpointing:
estimated 8–20 GB, 120–300 s/update. Hence approximately 60–120 completed updates in six hours;
the time cap governs. Every-update checkpoints are more frequent than the requested 1–2 hour cadence
provided the GPU microbenchmark confirms the estimate. A single CUDA update memory/time benchmark
is a MAIN pre-launch requirement; CPU tests cannot certify the estimate. The external timeout preserves
the last completed checkpoint if a single pathological update exceeds the internal reserve estimate.
The timeout's 21,570 s signal plus at most 30 s grace keeps the hard process budget within 21,600 s.

## Verification completed locally

`WMFIX_CPU.md` records tests and `WMFIX_SYNTHETIC*.json` retain the original plumbing result,
the strengthened gate's failure, and a conditional synthetic teacher's pass. The latter is still synthetic.
`WMFIX_REAL_FEASIBILITY.json` is explicitly BLOCKED by the missing local prepared remote corpus.
The corrected report was delivered and read before handoff (hash/retarget in WMFIX_DESIGN.md);
it confirms that the upstream/source intervention error remains. Remote preparation, source-clock parity,
real TRAIN gradient/optimizer feasibility and independent review are pending. **Current status: NOT LAUNCH READY.**

## One-command post-training verdict

MAIN must bind `verdict_plan.json` to exact frozen B/C1 producers and the existing canonical accept TRAIN panel.
The driver refuses all scoring when either adapter or an input is missing; a zero exit/missing result never passes.
It will not restart a directory with a persistent `RUN.json` (running or interrupted); reconcile it first.

```bash
python -m tools.wmfixmerge.verdict --plan tools/wmfixmerge/verdict_plan.json --train-dir /root/autodl-tmp/c4/wmfixmerge20261031/train --out /root/autodl-tmp/c4/wmfixmerge20261031/verdict --execute --device cuda
```

Default is dry-run; MAIN supplies an existing verdict output directory. The final immutable update from
`FINAL.json` is primary, verified by hash; candidate_config.json must actually instantiate the head.
A reuses this worktree's inherited WMREQUAL3 native-lane scorer, K8/mix ON with physical exits;
it separately reports unmatched MAPPO diagnostics and all source/upstream/downstream/crossing contrasts.
The source contract must bind its scorer/ownership hashes to the delivered corrected-harness contract.
SAFETY calls the unmodified `tools/loop14_checks/accept.py` with its existing decision thresholds.
Step0 is checked bitwise against original WM2 before being used as the same-config zero-head reference.

### Required B/C1 producer contract (currently unbound — not verdict-ready)

`stages.B` and `stages.C1` each need `producer = {script, files, args}`. `files` maps every frozen
producer/dependency/cohort artifact path to SHA256, including the entry script. `args` is an argument
list, with `{candidate}`, `{reference}`, `{config}`, `{out}`, `{binding}`, `{device}` expanded literally.
The driver runs that Python script sequentially under the same restricted-input audit guard, retaining logs.
Alternatively bind an existing receipt and its exact `sha256`; source provenance is still mandatory.
The JSON result must carry `binding` equal to BINDING.json and `producer` equal to the registered producer object.
No shell command string, automatic retry, unbound artifact reuse, or placeholder PASS is accepted.

- B schema `wmfix_speedhead/1`: K8, mix ON, native_lanes, matched_t300_history_queue,
  closed_loop_all_cavs=true. `rows` contains all 8 worlds × MAPPO/PI/CIRCLES × 300/600 s contrasts,
  each with world/arm/horizon/real/base_draws[8]/candidate_draws[8]. `absolute` contains all four arms
  (including CTRL0) over the same grid, with matching J and road/source component draws/real values.
  Use the real final MAPPO export d9042cc5 and frozen CTRL0/PI/CIRCLES recipe from SPEEDHEAD;
  update190 or recorded-command replay is not a substitute for the four closed-loop controllers.
- C1 schema `wmfix_c1_regression/1`: reference_label=C1, canonical_cohort_sha256,
  canonical_worlds={PI:[...],unseen_wall:[...]}, and exactly one aggregated error/blowup row per
  original cohort world (base_error/candidate_error/base_blowups/candidate_blowups).
  Preserve the original C1 estimand, history, walls and full deployed admission path; restricted source paths refuse.

The exact B/C1 producers/cohort paths are not present in this worktree. Their bindings remain null intentionally.
Thus the one-command orchestration and metric gates are implemented, but the full verdict cannot yet execute.

### B producer generalization spec (MAIN / SPEEDHEAD-owner side)

The SPEEDHEAD GPU quartet worker (`tools/speedhead/wm_gpu.py` in E:/agent-work/c4-speedhead,
deployed at /root/autodl-tmp/c4/speedhead20261030) pins the base side by SHA (WM2 b8797242,
MAPPO export d9042cc5, seed 965140, update 190, K8, native lanes, <=8 GiB). The candidate
side needs the same worker run with: the trained candidate checkpoint + candidate_config.json
(merge_response=true), conserve_queue + learned_admission ON in the entry process, identity
recorded with the candidate sha256 instead of the WM2/export pins, identical design.json
worlds, CRN futures and 8x8 world/future grid. Reduce both sides into `wmfix_speedhead/1`
rows (arm-minus-CTRL0 draws) and the 64-row absolute grid with road/source components,
embed `binding` and the registered producer object, then bind it in verdict_plan.json.
The real-side numbers and frozen CTRL0/PI/CIRCLES recipe come from SPEEDHEAD.md; no SUMO
run is licensed by this task. Until bound, stage B blocks the verdict (fail-closed, by design).
