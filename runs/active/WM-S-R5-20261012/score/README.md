# WM-S-R5-20261012/score — one-command round-5 scoring + diagnosis

CPU-only analysis tooling for the round-5 checkpoint. No training, no SUMO, no GPU.
Round-5 code tree: `E:/agent-work/r5-score/src` (commit 46b10d5; `rollout_mode hybrid`,
`flags.diag_*` in `wms.eval.predictor.RolloutFlags`) — both scripts insert it on
`sys.path` themselves. Python: `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`;
TEMP/TMP=E:/agent-work/tmp. Writes only inside this directory.

## Commands (when the checkpoint arrives, ~16:00)

```
cd E:/research-dense/runs/active/WM-S-R5-20261012/score
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe score_round5.py --ckpt <wm_s.pt> --ckpt-best <wm_s_best.pt> --workers 8
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe diagnose_round5.py --ckpt <wm_s.pt> --workers 8
```

Outputs: `ROUND5_SCORE.md` + `results.json` (< 5 MB) and `DIAGNOSIS.md` +
`diagnosis.json`. Every run first re-verifies the scorer (step0: C1 wall_same quick-20
60 s = 11.9576 m) and the R4 cache baseline gate (C1 8.71/99.84, R4 9.96/110.69); it
exits non-zero if either fails. Per-state rows are cached under `cache/` keyed by
checkpoint sha8 — interrupted or repeated runs are incremental and cheap.

## What they compute

`score_round5.py` (A): wall_same 67 + wall_pilot 53 + real-history 120 (FULL, from
`data_cache/wallc1_states_h60_real`); clean scorer `wms.eval.score.score_states`,
pairing world+key+source_arm+t0, 10/30/60/120 s state-median HDV |ds|, blow-ups, paired
world-bootstrap 95% CIs vs C1 and vs R4 (R4-cache rows for the wall sets; CHK-FADE
confirmation rows for real-120), and ROUND4_SCORE-style fingerprints on wall_same
(accel + separate lane gate curves, learned g_max/tau, late |g*da| 0-60/60-120,
signed speed error, TRUE lane-change rates vs recorded, wrong-lane share, hard-brake).
C1/R4 rollout fingerprints are copied from `WM-S-R4-20261011/score/results.json`
(identical models/sets — no recomputation).

`diagnose_round5.py` (B): pre-registered `WM-S-R5-PREP-20261012/DIAGNOSIS_PLAN.md`
measurements. M1 error split wrong-lane vs right-lane (10/30/60/120 s, median/p90);
M2 survivor (t0-present, alive at 120 s) signed speed error at 60/90/120 s with
jam(<8 m/s)/free split + section total delay at 120 s; M3 TRUE lane rates 0-10/10-60/
60-120 s vs recorded (allowance 0.00124327163/HDV-step); M4 the four switch-offs
`diag_lane_gate_shared` / `diag_lane_cap_off` / `diag_encoder_population` /
`diag_recorded_lanes` (R5, wall_same 67; C1/R4 comparators run the same code); M5'
the encoder's learned per-car settings (9 params) vs TRUE driver types — per-type
distributions, nearest-centroid 3-way separability (fit/test split by md5(id) parity),
correlations with true tau/minGap/accel/speedFactor. True types are evaluation-only
(parent-process joins; never fed to the model). Real-history quick-20 adds
base(C1/R4/R5) + encoder_population + recorded_lanes. Decision rules R1-R3 are
auto-evaluated at the end of DIAGNOSIS.md.

## Verified (2026-10-12, before the checkpoint)

- Plumbing smoke with `E:/agent-work/tmp/smoke_r5_f23/wm_s.pt` (barely trained; numbers
  meaningless): full score pipeline (3 sets x last+best, fingerprints incl. the separate
  lane gate) and full diagnosis pipeline (M1-M5', all four switch-offs, real quick-20)
  completed without errors; incremental reruns fully cached.
- Exact reproduction with the R4 checkpoint through `score_round5.py`:
  wall_same 60/120 s = 9.9628/110.6869 m and wall_pilot 9.4069/107.3172 m (R4 cache:
  9.96/110.69, 9.41/107.3); paired R4-C1 CIs identical to ROUND4_SCORE
  (e.g. wall_same 60 s -0.07 [-1.57, 0.34], 120 s +6.6 [-11.1, 47.5]); real-120
  60 s median 13.8115 m matches the CHK-FADE confirmation rows (13.81); blow-ups
  0/66, 0/52 (real 6/114, consistent with R4's 3/19 on quick-20).
- Peak RAM parent+8 workers: 1.18 GB (cap 12 GB).

## Runtimes (measured -> projected for the real run, 8 workers)

- Measured: R4-ckpt scoring of all 240 states 607 s (2.5 s/state); smoke-R5 hybrid
  ~14 s/state single-thread (~1.8-3 s/state parallel); diagnosis smoke 392 s for 12
  variant runs x 5 states; fingerprints ~67 states in a few minutes.
- Projected full run: score (last+best, 3 sets, 480 rollouts + fingerprints) ~ 30 min;
  diagnosis (7 wall variants x 67 + 5 real x 20 = 569 rollouts + M5') ~ 25-30 min.
  Total ~ 60 min, inside the 90-min budget. If the trained R5 is slower than the smoke
  model, cut M4 to quick-20 first (drop the four switch-off wall_same runs to 20 states
  via `--limit`, or run them selectively).
