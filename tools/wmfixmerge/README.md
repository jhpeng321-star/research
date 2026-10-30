# WMFIX operation (MAIN launch; CPU preparation first)

The audit/design are in `docs/paper_results/WMFIX_PRIOR.md` and `WMFIX_DESIGN.md`.
All commands run from this worktree root. Use a CPU PyTorch environment for preparation/tests.
No command here invokes SUMO. No training has been launched by the implementation session.

## Inputs and fail-closed gates

- TRAIN: `/root/autodl-tmp/c4/e3data`, only the 12 `SEEDS` in `data.py`, role **fit**,
  native/ours_c1/coopoff_c1/mappo pairs at t300. No validation/pairs300/SPEEDHEAD fitting.
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
Synthetic-only receipts cannot pass `train.check_gates`.
Astra's `REVIEW.json` must identify reviewer `Astra`, status `APPROVED`, and bind
`feasibility_sha256` to the passing file. The implementation session does not generate approvals.

## Exact eventual training command (MAIN only, after gates)

```bash
timeout --signal=INT --kill-after=30s 21600s python -m tools.wmfixmerge.train --enable-merge-response --conserve-queue --learned-admission --device cuda --prepared /root/autodl-tmp/c4/wmfixmerge20261031/TRAIN_PREPARED.json --config /root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml --checkpoint /root/autodl-tmp/c4/restart20261026/arm/loop/wm/c02/wm_s.pt --controllers /root/autodl-tmp/c4/overnight20261028/W5_COMBINED/DESIGN.json --corrected-report /root/autodl-tmp/c4/wmfixmerge20261031/WMREQUAL3.md --feasibility /root/autodl-tmp/c4/wmfixmerge20261031/FEASIBILITY.json --review /root/autodl-tmp/c4/wmfixmerge20261031/REVIEW.json --out /root/autodl-tmp/c4/wmfixmerge20261031/train
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

## Verification completed locally

`WMFIX_CPU.md` records tests and `WMFIX_SYNTHETIC*.json` retain the original plumbing result,
the strengthened gate's failure, and a conditional synthetic teacher's pass. The latter is still synthetic.
`WMFIX_REAL_FEASIBILITY.json` is explicitly BLOCKED by the missing local prepared remote corpus.
The corrected report was delivered and read before handoff (hash/retarget in WMFIX_DESIGN.md);
it confirms that the upstream/source intervention error remains. Remote preparation, source-clock parity,
real TRAIN gradient/optimizer feasibility and independent review are pending. **Current status: NOT LAUNCH READY.**
