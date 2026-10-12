# WM-S-R5-PREP (2026-10-12) — round-5 preparation on branch `wms/r5-prep`

Status: COMPLETE (code + configs + CPU smoke; no GPU, no real training run, no SUMO).
Continuation of the cancelled morning session: all review findings in
`CODE_REVIEW_SONNET.md` are applied (see "review follow-ups").

Scope: code + configs + CPU smoke only. Data: read-only absolute paths under
`E:/research-dense/data_cache/...` via `WMS_DATA_*` environment overrides (the worktree
has no `data_cache/` and no registry edits — housekeeping item 5 of the review).
Artifacts under `runs/active/WM-S-R5-PREP-20261012/` (smoke runs, memory probes,
`DIAGNOSIS_PLAN.md`); scratch under `E:/agent-work/tmp/`.

Owner rule honoured: driver settings are LEARNED from prediction error only. No
driver-type list, no type labels, no true driver parameters in training/model code or
configs (regression-tested in `tests/unit/test_r5_switches.py`). The R4 checkpoint init
transfers only learned network weights (`INIT_FROM.json` in every smoke: 103 tensors
loaded, 2 new = lane-gate parameters, 0 shape mismatches).

## What changed (implementation summary)

Three independent training switches, each default OFF (existing configs/checkpoints
bit-identical; regression tests + golden records in `tests/unit/test_r5_switches.py`):

* FIX 1 `rollout_mode="hybrid"` (existing) + `curriculum.driver_grad_across_chunks`
  (REDESIGNED per review finding 2): the drivers are encoded ONCE at t0 from the
  observed pre-t0 history — exactly where inference encodes them (`WmSModel.encode_
  drivers`) — and every truncated-BPTT chunk consumes detached LEAF tensors of that
  encoding; after the last chunk the accumulated leaf gradients are pushed through the
  single encoder graph (`torch.autograd.backward([values, embedding], [leaf grads])`).
  Exact gradient for everything truncation retains (one encoder pass, no per-chunk
  settings drift, train/inference consistent); the detached carry of the old path is
  kept bit-identically when the flag is off.
* FIX 2 `curriculum.truncation_steps_per_stage` (e.g. `[20, 60, 120, 240]`) +
  `model.checkpoint_step` (per-step activation checkpointing inside each chunk).
  Step-level checkpointing is skipped for stateful closed-loop CAV controllers
  (recomputation would replay controller state).
* FIX 3 lanes: (a) `model.lane_logit_cap = 1.5` (existing flag); (b)
  `model.lane_gate.kind = "saturating"` with own `g_max`/`tau` (init `g_max 0.16`,
  `tau 30` steps) separate from the acceleration gate; (c) `loss.flip_rate.
  {weight,allowance}` hinge loss on the per-step leave-lane probability above the
  allowance (default allowance = TRUE training rate 0.00124327163/HDV-step, junction
  lane -1 excluded by the shared supervision mask, as LANECAL). Replaces the
  ineffective lane-rate loss in F3 configs (lane-rate stays available, default off).

Review follow-ups (all five):
1. EXIT HANG: `src/wms/cli/train.py` now writes the full summary to
   `<out>/RUN_SUMMARY.json` and prints ONE short line; the smoke wrapper
   (`smoke/runs/run_smoke.py`) streams child stdout/stderr to a file (no PIPE).
   Verified: all four smokes exit 0 within 0.0-0.1 s of the summary write.
2. MEMORY: the hybrid magnitude accumulators are accumulated DETACHED in
   `UnrollRunner._collect_rollout` (the graph-carrying roll tensors still feed the
   chunk-local `ResidualPenalty`); the forward-only loss-share pass runs every
   non-probe chunk under `torch.no_grad` (only the probe chunk needs its graph).
   Review finding 3 (moving `lead_neighbour`/`push_history`/`_record_step` inside the
   checkpoint) was NOT needed: measured after-fix memory is flat and small (table
   below) — documented decision, not applied.
3. FIX 1 redesign: as above.
4. Measured memory (see table): batch 2 x 240-step truncation peaks ~1.1 GB process
   RSS on CPU — far below any 24 GB concern; extrapolated batch 8 fits easily.
5. Housekeeping: `configs/data_registry.toml` restored to git state (env overrides
   only; `WMS_DATA_CKPT_R4` resolves `init_from = "ckpt_r4"`); smoke lists moved to
   `runs/active/WM-S-R5-PREP-20261012/smoke/lists/`.

Post-run attribution switches (inference-time, default OFF, usable from the scorer;
`wms.eval.predictor.RolloutFlags` / config `flags.diag_*` / `python -m wms eval --set
flags.diag_*=true`): `diag_lane_gate_shared` (lane correction falls back to the shared
acceleration gate), `diag_lane_cap_off`, `diag_encoder_population` (encoder settings ->
population defaults, zero shrinkage), `diag_recorded_lanes` (recorded-lane teacher
forcing via the new `model.rollout(lane_teacher=...)` argument). Pre-registered
analysis: `DIAGNOSIS_PLAN.md` (error split wrong-lane vs right-lane, survivor speed at
120 s, lane rates vs recorded, switch-offs, encoder settings vs true driver types
EVALUATION-ONLY, comparators C1/R4).

New configs: `configs/r5_base.toml` (R4 recipe + FIX 1), `r5_f2.toml` (+FIX 2),
`r5_f3.toml` (+FIX 3), `r5_f23.toml` (+FIX 2+3; the ONE round-5 GPU run per owner
decision — its 120 s stage back-propagates through the whole 120 s, truncation 240,
grad clipping on via the inherited `optim.grad_clip = 1.0`). All four init matching
weights from the R4 checkpoint via `model.init_from` (shape-matched, strict=False).

## Test + smoke results

Unit tests (`python -m pytest -q tests/unit`): 626 passed, 1 skipped, 4 failed — all
four failures (`cli/test_launch_script.py` x3, `data/test_drive_registry.py` x1)
reproduced on a clean HEAD worktree (pre-existing Windows/registry issues, unrelated to
this branch). `tests/unit/test_r5_switches.py` (29 tests) covers: bit-identity of the
R4 flat form, default flat form, default unroll record and composer/state-dict; FIX 1
(encoder grad from a loss confined to chunks >= 2: 15.27 vs exactly 0.0 flag-off; leaf
gradient == full-BPTT gradient with one chunk; feedback-free toy equality; flag on/off
forward bit-identity); FIX 2 (per-stage truncation plan, runner override, checkpoint
transparency + differentiability, closed-loop controllers never checkpointed); FIX 3
(separate lane gate, shared default bit-identical, flip-rate value/gradient/composition);
r5 configs reach settings/spec; the four diag switches (default-off bit-identity, each
switch changes exactly its component, predictor/config flag mapping).

Real-data CPU smokes (batch 2 for r5_f23, batch 1 for the brief trio;
`curriculum.stage_steps=[1,1,1,1]`, real pack_v3/dspeed/damping windows via the smoke
list; outputs under `smoke/runs/`):

| config | exit | wall | peak RSS | losses by stage (finite) |
|---|---|---|---|---|
| r5_f23 | 0 | 859 s | 1.49 GB | 12.8 / 116.1 / 183.7 / 418.0 |
| r5_base | 0 | 767 s | 2.35 GB | 6.1 / 62.0 / 93.0 / 104.1 |
| r5_f2 | 0 | 563 s | 1.29 GB | 6.1 / 62.0 / 102.5 / 103.4 |
| r5_f3 | 0 | 329 s | 2.39 GB | 6.1 / 73.5 / 63.5 / 115.4 |

- All losses finite at every stage; validation losses finite; stage checkpoints written.
- Every run exits cleanly 0.0-0.1 s after `RUN_SUMMARY.json` (exit-hang fixed).
- Loss-share table (capture mode, as logged by the trainer): position 0.64-0.95,
  residual_mag up to 0.17, flip_rate 0.0003 (10 s) -> 0.0078 (30 s) -> 0.0054 (60 s)
  -> 0.1505 (120 s). The 120 s value is above the 5-10% target band because the
  untrained-transfer model over-flips ~20x the allowance (predicted rate 2.7% vs
  allowance 0.12%); the hinge is designed to fade as the rate falls. Monitor this
  share in the real run: if it stays >15% after stage-2 rates drop, lower
  `loss.flip_rate.weight`.
- Encoder grad from chunks >= 2 on REAL data (probe, loss confined to chunks 2-3 of 4,
  truncation 40): flag ON 15.27, flag OFF exactly 0.0 (`memory/probe_full.json`).
- Known pre-existing quirk (not from this branch): probe-mode loss shares are all-zero
  on these windows (the last chunk has no valid targets when traffic loss is off);
  capture-mode shares (what the trainer logs) are correct.

## Memory / time vs truncation length (CPU, 2 threads, real smoke windows)

Peak process RSS (torch+model+dataset base ~0.6-0.9 GB included), after the fix
(`memory/probe_full.json`, `memory/probe_full.log`):

| mode | batch | 40 | 80 | 120 | 240 steps |
|---|---|---|---|---|---|
| share pass (after fix) | 1 | 735 | 712 | 701 | 707 MB |
| backward (after fix) | 1 | 713 | 739 | 799 | 923 MB |
| share pass (pre-fix emu.) | 1 | 707 | 728 | 760 | 859 MB |
| share pass (after fix) | 2 | 991 | 965 | 961 | 936 MB |
| backward (after fix) | 2 | 994 | 1009 | 1044 | 1141 MB |
| share pass (pre-fix emu.) | 2 | 982 | 956 | 987 | 1011 MB |

The pre-fix emulation (graph-carrying hybrid accumulators + graphs on every chunk of
the forward-only pass) grows LINEARLY with unroll steps (per-chunk RSS series in the
probe log: e.g. batch 2 x 240 rises 915->1010 MB over 12 chunks, ~0.4 MB/step on these
few-slot smoke windows; the mechanism scales with slots per window, which is what
produced the observed ~20 GB on the real training windows at batch 2). After the fix
the per-chunk series is FLAT — no cross-chunk retention at all. Backward-mode deltas
stay 40-282 MB for one chunk up to 240 steps; the loss-share probe pass costs 26-80 s
CPU per stage end (acceptable; GPU ~30-50x faster).

## GPU feasibility: r5_f23 on one 24 GB GPU

Owner decision: ONE combined run (`r5_f23`), stage 4 = 120 s unroll with truncation
240 and grad clipping on. Measured CPU-RSS activation cost of a full 240-step chunk is
~200-280 MB at batch 1-2 with per-step checkpointing on; scaling ~linearly to batch 8
gives ~0.8-1.2 GB of live activations, plus parameters/optimizer (~3 MB params x (1+
2 Adam) + encoder graph) and CUDA context — a comfortable estimate well under 4 GB,
i.e. r5_f23 at batch 8 x 240 fits a 24 GB GPU with >6x headroom (the share-pass
diagnostic is flat-memory after the fix and adds nothing scaling with the unroll).
CPU time 150 s per 240-step chunked step at batch 2 on 2 threads suggests ~3-8 s/step
on a modern GPU -> stage 4 (1250 steps) ~1-3 h; the full curriculum fits the usual
overnight budget. r5_base/r5_f2/r5_f3 stay as diagnostic configs only.

## Open risks / next actions

1. flip_rate share at 120 s starts ~15% (model over-flips 20x allowance). Watch the
   stage-2/3 share; if it does not fall below ~10% as the lane rate drops, halve
   `loss.flip_rate.weight` (1.0 -> 0.5) — config-only change.
2. Probe-mode gradient shares are broken on windows whose last chunk has no valid
   targets (pre-existing); the trainer logs capture-mode shares, which are correct —
   no action for round 5, ticket for later.
3. FIX 1 truncation caveat (by design): the encoder gradient across chunk boundaries
   excludes INDIRECT paths through detached state (true of any TBPTT scheme); with
   stage-4 truncation 240 = whole 120 s unroll, stage 4 is exact full BPTT.
4. Before the GPU run: preflight `python -m wms train --config configs/r5_f23.toml
   --mode preflight` on the GPU box with the `WMS_DATA_*` overrides set, then run
   `--mode run`. Post-run attribution per `DIAGNOSIS_PLAN.md`.
