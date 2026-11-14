# DIAG 3 — Why does R4 predict MORE early (0-60 s) lane changes than C1?

Run `WM-S-R4-DIAG-LANE-20261011` (analysis only; CPU, 5 workers; no training/SUMO/GPU).
Data: `wall_same` (h60 states; quick-20 = first state per world, 20 worlds, same rule as
`WM-S-R4-20261011/score/score_round4.py`), entries OFF, CAVs teacher-forced from the
recorded wall arm, 120 s rollouts, clean package (`PYTHONPATH=E:/research-dense/src`).
TRUE lane-change definition = lane forward-filled over junction `-1`; change = lane differs
with both >= 0; HDVs present in both frames (`WM-S-LANEDWELL-20261010/scripts/lane_lib.py`,
`score_round4.true_lane_rate`). Scripts: `scripts/diag_lane.py`, `scripts/aggregate.py`;
raw per-state traces: `out/tasks/*.pkl`; numbers: `results.json`.

Models: C1 = `data_cache/ckpt_c1fast/wm_s.pt`; R4 = `runs/active/WM-S-R4-20261011/box/gpuA/
r4_main_a1/wm_s.pt` (saturating gate g_max 0.473 / tau 48.6; lane_rate_target 0.00124327163,
weight 1.0; lane_head off; lane_logit_cap **null**; training list `windows_train_noPI_noWall`
— 911 windows, 155 worlds, **zero wall worlds**). C1/R4 physics receipts are IDENTICAL
(merge_cal=False, same thresholds, merge zone 0..75.44 m) => the MOBIL/merge baseline in
`lane_logits = MOBIL + bounded(g(k)*lane_delta)` is the same function for both models; only
the learned correction, the gate and the losses differ. C1 had **no lane-rate loss**
(`ckpt_c1fast` train log has no lane_rate term; receipt has no lane_rate weight/target).

Reference (full 67-state wall_same, R4 score): lane changes/HDV/10 s — 0-60 s: rec 0.024,
C1 0.159, R3 0.352, R4 0.260; 60-120 s: rec 0.037, C1 4.50, R3 1.22, R4 1.56.
Quick-20 replication (this run): rec 0.028/0.038, C1 0.194/4.85, R4 0.365/2.38.

## Q1 — What the lane-rate loss actually saw (training logs, measured)

Implementation (`src/wms/train/losses/lane_rate.py`, `train/unroll.py`, `model/wms.py::_record_step`):
the term pulls the **mean softmax mass off the current lane** — `1 - P(cur lane)` of the full
`lane_logits` (MOBIL + g*delta) over all aux-masked HDV steps of the training unroll — toward
`0.00124327163` with weight 1.0 (whole-batch squared deviation, chunk-differentiated through a
gradient-free reference replay). Junction steps are masked (aux_mask reuses `loss_valid`,
same mask LANECAL verified for the CE), so it does NOT count junction toggles — but it is a
**probability mass, not a realized argmax-flip count**, and it is measured on the model's own
free-running training rollout of noWall/noPI windows (never on wall states).

| stage (unroll steps) | lane_rate p50 (first/last) | predicted rate p50 (min–max) | vs target 0.00124 | grad-norm share |
|---|---|---|---|---|
| 10 s (20)   | 2.3e-4 (1.4e-4 / 1.8e-4)  | 0.0163 (0.008–0.033)  | 13x | 2.5e-5 |
| 30 s (60)   | 1.2e-3 (5.1e-3 / 1.2e-3)  | 0.0356 (0.011–0.084) | 29x | 3.0e-3 |
| 60 s (120)  | 2.8e-3 (9.4e-3 / 8.1e-3)  | 0.0542 (0.016–0.128) | 44x | 7.1e-3 |
| 120 s (240) | 4.0e-3 (1.7e-2 / 1.3e-3)  | 0.0642 (0.021–0.167) | 52x | 2.8e-3 |

Val (`val_log.jsonl`): predicted rate 0.022 (stage 10) → 0.137 (240-step, step 7000) → 0.108
(final); never below 0.022. Position Huber owns 82–95% of gradient norm; lane CE p50 0.15–0.18
nats (P(correct lane) ≈ 0.85 — logits gaps ~2 nats). Gate learned g(k): 0.088/0.160/0.335/
0.433/0.470 at k=10/20/60/120/240.

**Verdict — the loss does not work as intended.** (a) It never came near its target: the mean
mass stayed 13–100x above 0.00124 for the whole run. (b) The target is effectively unreachable
for this proxy: mean mass 0.00124 needs a ~7.4-nat stay-vs-other logit gap at nearly every step
(MOBIL neutral = 0 gap), which the co-trained per-step CE (itself at 0.15 nats) and the shared
lane-delta head do not want. (c) With weight 1.0 on a squared batch mean the term's gradient
share was ≤0.7% — it could not move anything. (d) Even if satisfied, mean mass is not the
scored quantity (argmax flips at inference on wall states); the mass over-counts realized flips
by ~6x (final val mass 0.108 vs realized 0.018/step on quick-20). And C1 trained with no such
term at all, so the loss cannot explain the C1<R4 ordering — it simply failed to constrain R4.

## Q2 — MOBIL alone vs the learned delta at R4's early (0-60 s) change steps (measured)

Instrumented decomposition at every 0.5 s step (quick-20): decision = argmax(MOBIL + corr),
`corr = g(k)*lane_delta` (unbounded; lane_logit_cap null). "Delta-driven" = MOBIL alone would
stay (margin = base[cur] − max_other base > 0); "corr_adv" = corr[new] − corr[cur].

| quick-20, 0-60 s | C1 | R4 |
|---|---|---|
| lane-change decisions (HDV-steps) | 1,432 | 2,673 (1.87x) |
| MOBIL alone would also change | 275 (19%) | 694 (26%) |
| delta-driven (MOBIL margin > 0) | 1,157 (81%) | 1,979 (74%) |
| 2-lane jumps | 0 | 1 |
| MOBIL stay margin @ delta-driven, p50 (p90) | 3.30 (8.0) | 3.60 (8.0) |
| corr_adv @ delta-driven, p50 (p90) | 4.59 (8.9) | **6.01 (10.4)** |
| corr_adv @ MOBIL-would steps, p50 | 0.60 | **2.13** |
| max |corr| @ decisions, p50 | 4.22 | **5.59** |
| gate g(k) @ decisions, p50 | 0.388 | 0.400 |
| k @ decisions, p50 | 95 | 91 |
| MOBIL-only flip rate on own trajectory | 0.068/10 s | 0.121/10 s |

- The learned correction is the source in both models (74–81% strictly delta-driven), and R4's
  correction is bigger and more pro-change: corr_adv p50 6.0 vs 4.6 at delta-driven steps, and
  +2.13 vs +0.60 even where MOBIL already wants to change; with nearly identical margins and
  gates, R4's raw lane delta is ~30% larger and biased toward changing.
- Counterfactual trajectories (Q4 `delta0`) confirm: MOBIL-only R4 flips at **0.015/10 s**
  0-60 s (recorded 0.028) — the baseline UNDER-changes; 96% of R4's early changes (0.365 →
  0.015) come from `g*delta`.

## Q3 — Where/when the extra early changes happen (measured, quick-20)

- **Time**: R4 changes LESS than C1 in the first 10 s (44 vs 63), then 1.5–3.6x more per 5-s
  bin from 10–15 s through 55–60 s (bin ratios 0.8, 0.7, 2.1, 1.6, 2.0, 2.2, 3.6, 2.2, 1.8,
  2.0, 1.9, 1.5); the excess accumulates with k as g(k) grows (g: 0.09@10 → 0.34@60 steps).
- **Space**: spread across the whole road, not localized: s<−100 m 45.7%, −100..0 m 18.1%,
  merge zone 0..75.44 m 18.1%, downstream 18.1% (C1: 40.7/16.3/19.0/24.1). Lanes (current):
  lane0 44%, lane1 50%, lane2 6% (C1 43/50/8).
- **CAV proximity**: 34% of R4's changes within 25 m of a CAV vs 40% for C1; 23% farther than
  100 m vs 19% — i.e. C1's flips concentrate near the wall, R4 adds generic over-changing on
  top (the wall is a trigger but not the main driver).
- **Junction/entries/exits**: 29% of R4's changes occur within ±5 s of the same vehicle's
  recorded junction transit (C1 33%); 7% exit within 10 s (C1 11%); entries are OFF in this
  eval. Only 16% are within ±5 s of a recorded TRUE change of that vehicle (C1 11%) — i.e.
  ≥84% are spurious relative to the recording.

## Q4 — Inference ablations on wall_same quick-20 (indicative, 20 states)

| variant | lane/HDV/10 s 0-60 | 60-120 | wrong-lane 60/120 | ds_med 60/120 m (p90@120) | blow-ups | paired ds120 vs R4 base |
|---|---|---|---|---|---|---|
| recorded | 0.028 | 0.038 | — | — | — | — |
| C1 base | 0.194 | 4.85 | 0.159 / 0.389 | 11.96 / 111.8 (154) | 0 | − |
| R4 base | 0.365 | 2.38 | 0.107 / 0.180 | 11.96 / 114.7 (293) | 0 | − |
| R4 delta0 (MOBIL only) | **0.015** | **0.041** | 0.122 / 0.248 | 11.63 / 102.3 (319) | 0 | mean −7.6 m, median −13.0 (10 better/9 worse) |
| R4 gcap20 (lane gate ≤ g(20)=0.1595) | **0.051** | **0.117** | 0.105 / 0.260 | 11.37 / 104.5 (303) | 0 | mean +3.3 m, median +0.03 (9/10) |

- Zeroing the delta (or capping its gate at g(20)) removes the early excess (0.365 → 0.015 /
  0.051 vs recorded 0.028) and the late flood (2.38 → 0.041 / 0.117 vs recorded 0.038), with
  60 s error unchanged-to-better (11.96 → 11.63 / 11.37) and 120 s not worse (delta0 slightly
  better). So the early lane excess is NOT load-bearing for position accuracy.
- Cost of full removal: wrong-lane at 120 s rises 0.180 → 0.248/0.260 (fewer late changes →
  more vehicles parked in the wrong lane); R4 base's delta is what gives it the much better
  wrong-lane share vs C1 (0.18 vs 0.39). A capped delta (gcap20) keeps some of that
  (0.26) — a bounded/separate lane gate is the better trade than removal.

## Mechanism (answer)

R4's extra early lane changes are **not** MOBIL (identical physics; MOBIL-only under-changes:
0.015 vs recorded 0.028/10 s) and not the lane-rate loss being absent in C1 — the loss was
present in R4 but **ineffective by construction**: it constrains the mean softmax mass off the
current lane, sat 13–100x above its 0.00124 target all training with ≤0.7% gradient share
(position Huber dominates), and even a satisfied mass target would not pin the argmax flip
rate. The excess comes from the **learned lane delta being larger and pro-change-biased in R4**
(corr_adv p50 6.0 vs 4.6; +2.1 vs +0.6 at MOBIL-would steps; |corr| p50 5.6 vs 4.2 at similar
margins ~3.3–3.6 and similar gates), amplified as g(k) grows through the first minute
(0.09→0.34 by 60 steps), on wall-family states R4 never trained on (its window list has zero
wall worlds). Result: 74% of early changes are strictly delta-driven, spurious (≤16% near a
recorded change), spread over the whole road and all lanes.

## Recommended training-side fixes for round 5 (measured basis → proposal)

1. **Fix the lane-rate loss** (primary): as implemented it is not meaningful — replace the
   mean-mass target with a per-step flip-rate surrogate that penalizes change mass above a
   small threshold (e.g. mean(max(0, P(other) − τ)) or expected-flip weighting), calibrated to
   the TRUE per-list rate (~0.0013/step), and raise its weight so its gradient share is ~5–10%
   (currently ≤0.7%). Measured justification: Q1 table.
2. **Bound the lane delta** (direct lever): set `lane_logit_cap` (currently null) to ~1–2
   logits, or cap |corr_adv|; at delta-driven decisions corr_adv p50 6.0 overruns MOBIL stay
   margins of p50 3.6. Inference check gcap20: 7x fewer early changes, 60 s error −0.6 m,
   120 s median −13.0 m (delta0: −13.0 m median) — the excess is not buying accuracy.
3. **Separate, saturating lane gate**: give the lane delta its own gate ceiling (tau ~20–40
   steps, g_max_lane ≲ 0.16) instead of riding the accel gate (LANECAL #82 already suggested
   separate gates). Measured: decision-time g p50 ≈ 0.40 while the excess accumulates with k.
4. **MOBIL calibration: not needed** — the baseline is shared and under-changes alone; leave it.
5. Optionally add wall-family windows to the training list (R4's list excludes them; C1-vs-R4
   wall behaviour differences may partly stem from this).

Measured vs inference: Q1 is measured from training logs; Q2/Q3 are measured from instrumented
quick-20 rollouts; Q4 ablations are inference-time indicative checks, not training-side proofs.
