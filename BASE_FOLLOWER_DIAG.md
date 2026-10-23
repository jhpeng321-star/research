# BASE_FOLLOWER_DIAG - why the base WM over-predicts the lane-1 target follower's acceleration before a merge

S-BASEDIAG, 2026-10-24, 03:00-07:00. This is diagnosis only: no training of any model, no SUMO run, no change to model or
training code.
- **Compute and data.** Local CPU (at most 8 workers). TRAIN windows only, from the filtered roots `c4-v10r10b/tmp_r10b/roots`
  (pointing to `c4-v10r9fix/data/r9_train`: 192 episodes, v5_native + genrand_plain).
- **What was not opened.** No sealed, holdout, cal-split or eval-seed path; no `hetdata_truth`; no vType file; no LCSTATE log
  header. Only the LCSTATE.md aggregates and the population medians already in `src/wms/train/losses/driver_prior.py` were used.
- **Round-8 checkpoint.** `c4-v10r9fix/data/r9_train/wm_s_stage300.pt` (sha256 ae094e59...). This is the base that rounds
  9/10/10b/11 share.
- **Round-10 checkpoint.** `c4-v10r10/tmp_diag/r10/wm_s.pt`: the copy S-FIXBDIAG pulled from gpuR `runs/c4_wm_v10_round10`.
  I used it read-only and copied nothing from gpuR myself.
- **Scripts.** `tools/diag_base_*.py` on branch `c4/v10r11`.
- **Raw outputs.** `tmp_basediag/` (not committed).

Numbers marked *measured* come from these scripts. Anything about rollouts, delays or ranking is *proposed* (reasoned, not run).

## 0. Answer first

**Root cause (measured).** The over-prediction (last 3 s before the merge: model - recorded = +1.34 m/s^2 for round 8, +1.35 for
round 10) comes from the **IDM core**. It does not come from the learned residual (net -0.10), the gate (0.23-0.24), a
courtesy/yield remnant (the lane-1 core is pure IDM on 100.00 % of rows) or the plant clamp (binding on 10-11 % of rows). It has
three additive parts:

| part | size in the last 3 s | what it is | merge-specific? |
|---|---|---|---|
| A | **~+0.6** | **The IDM settings cannot reach the data's population.** The v10 line inherits the v5 `legacy` encoder: legacy default (a 2.49, b 4.2) + shrink x tanh x bound (0.8 / 1.5). That gives a floor of a >= 1.78 and b >= 2.87, and the encoder is pinned on it (encoded a: p1 = p10 = 1.78, median 1.87; b: p1 = p10 = 2.87; the p90 values 2.49 / 4.2 are drivers without history). SUMO HDVs are textbook IDM (a 1.2, b 2.0, T 1.2, s0 2.0). Far from the merge, start-ups fit a = 1.1 (MAE 0.23, bias -0.01; n = 48k). The WM therefore starts slow cars about 2x too fast everywhere: slow-start rows +1.52 m/s^2. | no (generic) |
| B | ~+0.1 | The residual cannot correct A. The attention residual head is **dead**: its pre-gate output is -1.5 (tanh bound) on 100 % of rows at every k. With g_max ~0.39 it is a constant offset. The temporal MLP has no merge or neighbour context and adds one-step noise: all-row MAE is 0.98 for the model vs 0.43 for the calibrated core alone. | no |
| C | **~+0.6..0.8** | **The SUMO target follower "holds" (does not accelerate) while a ramp car waits beside or just ahead of it.** With SUMO's own textbook IDM it should accelerate +0.84 in the last 3 s; recorded is 0.00. "Do not accelerate", i.e. min(own IDM, 0), fits: bias -0.12, MAE 0.48. The ramp car as a full IDM leader at its current position does **not** fit: bias -5.0 (it is nearly alongside: median bumper gap 1.5 m, 34 % overlapping). | yes |

Plus a teacher-forced-replay artefact, D. `_recorded` drops junction-transit vehicles (loss_valid = 0) from the replay STATE. On
**20 % of last-3-s partner rows** a lane-1 vehicle hidden that way sits between the follower and its replay leader (median 10.8 m
vs 68 m). This affects fix-B controls, the R11 contrast controls and every offline number (including R11's +1.46), but not
rollouts.

**Answers to the specific hypotheses.**
- "IDM sees the ramp car as absent / not a leader": true in the WM. But SUMO does not treat it as a leader either; what SUMO adds
  is a hold.
- "Lane-1 leader assignment ignores the ramp vehicle": same answer.
- "Free-road term near the merge": only via artefact D.
- "IDM settings mis-set": **yes** (part A, accel/decel floors).
- "Residual saturation": **yes** (part B, dead attention head).
- "Trapezoid integration": not involved; the error is in the one-step acceleration itself.

**Recommended fixes, ranked (proposed).**
1. **Reachable IDM settings.** Set `model.driver_param_mode = "physical"` (V6ENC: implemented and tested; a 0.3-4.0, b 0.8-6.0;
   learned centre; prediction error only). Keep the textbook `driver_prior` that the config already carries; it is ineffective today
   because of the floor. Retrain together with a re-initialised residual: zero the dead attention head's output layer. Do not
   reuse its constant offset: with a = 1.2 and the old residual, post-merge rows go from +0.07 to -0.48. V5_DIAGNOSIS Q3 found
   this on 10-15 and v6 fixed it, but the v10 line descends from v5m and never got the fix.
2. **The merge-specific hold as a learned term on the calibrated base.** The R11 contrast courtesy head (already built) is the
   vehicle: with an unbiased base, matched controls carry ~0 generic error, so the contrast target equals the absolute target. Its
   inputs already carry what gates the hold (d_end, wait / zone time, ramp acceleration, gap). On the measured rows, "calibrated
   core + hold" brings the last 3 s to -0.08 (MAE 0.46) vs +1.34 (MAE 1.84) now. A ramp-car virtual leader is **not**
   recommended: -0.52 / MAE 1.24 even on the calibrated core.
3. **Fix the replay visibility mask** (part D). Make the replay's `present` the filled window presence (junction_fix = ffill),
   and keep loss_valid only for supervision. Do this before any further teacher-forced loss.

**Expected effect (proposed).**
- Queues in and before the merge zone discharge more slowly, so merge-zone delay should rise from 0.48x toward SUMO.
- Target followers stop accelerating into the ramp car's gap, so ramp delay should fall from 2.1x toward SUMO and stuck ramp cars
  (R9_STUCK bistability) should become rarer.
- Ranking: today every controller that opens or closes gaps is scored against HDVs that react about 2x too eagerly. The direction
  of the ranking change cannot be predicted offline.

**Offline test (section 7 has the result of a first version).**
- Fit the physical-mode encoder (head + centre) teacher-forced on TRAIN windows, residual off; then fit the R11 head on that base.
- Pass if: recovered accel median 0.9-1.5 and not piled at a bound; far-upstream start-up bias |.| < 0.2; braking rows not worse;
  last-3-s target-follower bias <= +0.9 before the head and within +-0.3 after it.
- Then a TRAIN-state rollout check of merge-zone delay, ramp delay, stuck cars and post-cut-in braking against round 10b.

## 1. Method

* **Model replay.** `tools/diag_base_decomp.py` runs the same causal teacher-forced replay the cut-in / courtesy losses use
  (`one_step_replay`). It covers 96 TRAIN windows x 120 s (1 per episode, seeded) for round 8 (round10b config + courtesy v2,
  head = exact 0) and for round 10 (round10 config). Hooks record, for every lane-1 HDV transition row:
  - the IDM core (`physics_accel_mps2`, encoded driver parameters);
  - gate x attention-head emission and gate x temporal emission;
  - the bounded correction actually added, the pre-clamp and final acceleration;
  - the encoded (v0, T, s0, a, b);
  - the model's own leader gap and speed;
  - the R10B/R11 geometric ramp partner, its time to merge, and the hidden-vehicle flag.

  432,374 rows per model; 1,850 target-follower rows in the last 3 s before a recorded merge (merge-zone partner).
* **Recorded SUMO rows.** `tools/diag_base_vleader.py` covers all 192 TRAIN episodes, t >= 300 s, 1 s sampling, 5.24 M lane-1
  HDV rows; 26,603 target-follower rows in the last 3 s. For each row it gives IDM with three parameter sets:
  - WM default (a 2.49, b 4.2, T 1.135, s0 2.6);
  - "SUMO defaults" (a 2.6 assumed, plus LCSTATE section 9 medians);
  - textbook (a 1.2, b 2.0, T 1.2, s0 2.0).

  Each set is evaluated against the own leader and against the partner as a virtual leader at its current position.
* Counterfactuals on the model rows: `tools/diag_base_params_cf.py`, `tools/diag_base_cf_forms.py`.
* Per-driver and encoder fits: `tools/diag_base_perdriver_fit.py`, `tools/diag_base_encfit.py`.
* **Sign convention.** Bias = model - recorded (positive = model accelerates too much). Groups:
  - "partners last 3 s": merge-zone partner rows with 0 < t_merge - t <= 3 s;
  - "slow start": unpartnered rows with v < 2 m/s and leader gap 6-30 m.

## 2. Q1 - which component (measured)

### 2.1 Target-follower rows by time to the merge (round 8; round 10 within 0.05 everywhere)

| t - t_merge | n | recorded | final | IDM core | gate x attn | gate x temp | bounded corr | gate | final - rec | IDM - rec | pre >= +3 (clamp) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [-10,-6) | 1802 | -0.05 | +0.41 | +0.59 | -0.36 | +0.01 | -0.15 | 0.24 | +0.47 | +0.65 | 0.05 |
| [-6,-4) | 1080 | -0.07 | +0.63 | +0.78 | -0.37 | +0.08 | -0.09 | 0.24 | +0.70 | +0.85 | 0.07 |
| [-4,-3) | 614 | -0.03 | +0.90 | +1.05 | -0.36 | +0.15 | -0.07 | 0.24 | +0.93 | +1.08 | 0.10 |
| [-3,-2) | 634 | -0.05 | +1.08 | +1.32 | -0.36 | -0.08 | -0.19 | 0.24 | +1.13 | +1.37 | 0.08 |
| [-2,-1) | 603 | -0.12 | +1.38 | +1.50 | -0.36 | +0.27 | -0.03 | 0.24 | +1.50 | +1.62 | 0.10 |
| [-1,0) | 613 | +0.08 | +1.50 | +1.66 | -0.37 | +0.21 | -0.07 | 0.25 | +1.42 | +1.58 | 0.14 |
| **last 3 s** | 1850 | -0.03 | +1.32 | **+1.49** | -0.36 | +0.13 | **-0.10** | 0.24 | **+1.34** | **+1.52** | 0.11 |
| round 10, last 3 s | 1850 | -0.03 | +1.32 | +1.49 | -0.34 | +0.10 | -0.11 | 0.23 | +1.35 | +1.52 | 0.10 |

* The IDM core alone over-predicts by +1.52. The residual removes only 0.1-0.2.
* Recomputing IDM from the cached state and encoded parameters reproduces the core on 100.00 % of rows. So for lane-1 rows the core
  has no lane-end obstacle, merge_cal or mainline-yield term: no courtesy / yield remnant.
* The attention head's pre-gate emission is -1.5 on every row (p1 = p50 = p99 = -1.5 at k = 2...239). It is tanh-saturated and
  carries no state information.
* Gate: saturating, g_max ~0.39, tau ~70 steps (k = 20: 0.13; k = 80: 0.31; k = 239: 0.39).
* Round 10 (fix B, temporal and gate trained) changes nothing material here (+1.35 vs +1.34). Fix B did not touch the cause.

### 2.2 Where the over-prediction lives (last-3-s rows)

* By follower speed: v < 0.5 m/s +1.90; 0.5-1 +1.79; 1-2 +1.55; 2-4 +0.83; > 8 +0.69.
* By own-leader gap: 8-12 m +1.90, 12-20 m +1.40.
* The followers are near standstill (median v 1.1 m/s) with a lane-1 leader pulling away (median v_l 4.2 m/s, gap 12 m) and the ramp
  car almost alongside (median bumper gap 1.5 m). IDM then outputs close to its maximum acceleration (encoded median a 1.87; the
  clamp binds on 11 %).
* **State-matched unpartnered controls (5-NN, R11 coordinates; 89 % of them inside the merge zone):** final - rec +1.08,
  IDM - rec +1.49, residual -0.20. This is the "+0.9 carried by controls" of R11/R10B. It is the generic part A plus the zone, not
  adjacency.

## 3. Generic vs merge-specific (measured)

### 3.1 SUMO start-ups follow textbook IDM away from the merge; the WM's IDM is ~2x too eager

Recorded rows with s < -400 m, v < 2 m/s, lane-1 leader pulling away (v_l > v + 0.5), by bumper gap:

| gap (m) | n | recorded | IDM WM default | IDM textbook (a 1.2, b 2.0, T 1.2, s0 2.0) |
|---|---|---|---|---|
| 3-5 | 15,740 | +0.90 | +1.25 | +0.88 |
| 5-8 | 22,942 | +1.10 | +1.91 | +1.07 |
| 8-12 | 18,526 | +1.03 | +2.20 | +1.14 |
| 12-20 | 6,963 | +0.84 | +2.36 | +1.18 |

Fitting a constant a on these start-up rows (gap 5-20 m) gives a = 1.1 (MAE 0.23, bias -0.01). Free-road acceleration far upstream
(5 < v < 20, gap > 60 m) has mean +0.60, p99 +1.53. The same fit gives an effective a of 0.8 inside the merge zone (unpartnered)
and 0.5 on the approach (-400..0 m). Part of that is hidden junction leaders (73 % of start-up rows at s in [-25, 0) have one).
Plausibly part is selection: in queues, rows with "leader pulling away" over-represent sluggish drivers. Per-driver a would carry
that, if the encoder could reach it.

### 3.2 The encoder is pinned on its floor

Encoded parameters over all 432k rows, p1 / p10 / p50 / p90 / p99: a 1.78 / 1.78 / 1.87 / 2.49 / 2.49; b 2.87 / 2.88 / 3.04 /
4.20 / 4.20. Code: `PARAMETER_BOUNDS` accel 0.8, decel 1.5 around the legacy defaults, shrink = n / (n + 8). The gradient
pushes every observed driver to the lower bound. The config's `driver_prior` (target textbook, weight 0.05) asks for a = 1.2 but
cannot get it.

### 3.3 Parameter counterfactual on the model rows

Only a and b are replaced; encoded T, s0, v0 and the bounded residual are kept unless stated. Values are mean bias [MAE]:

| group | n | rec | model | a = 1.2 | a = 1.2, b = 2.0 | a = 1.2, b = 2.0, **no residual** |
|---|---|---|---|---|---|---|
| partners last 3 s | 1850 | -0.03 | +1.34 [1.84] | +0.70 [1.37] | +0.72 [1.35] | +0.83 [1.01] |
| partners [-6,-3) s | 1694 | -0.05 | +0.78 [1.65] | +0.40 | +0.42 | +0.53 [0.89] |
| slow start (unpartnered v < 2, gap 6-30) | 5343 | +0.16 | +1.52 [1.96] | +0.77 | +0.78 | +0.88 [0.91] |
| far upstream slow start (s < -400) | 92 | +1.01 | +0.47 | -0.52 | -0.49 | +0.09 [0.25] |
| after the merge zone s > 75 m | 94,856 | +0.50 | +0.07 [1.07] | -0.48 | -0.47 | +0.11 [0.32] |
| recorded braking (rec < -1) | 25,228 | -1.70 | +0.68 [1.13] | +0.70 | +0.72 | +0.39 [0.75] |
| v > 15 m/s | 279,324 | +0.03 | +0.18 [0.80] | -0.02 | -0.03 | +0.10 [0.36] |
| all lane-1 rows | 432,374 | +0.05 | +0.11 [0.98] | -0.06 [0.94] | -0.06 [0.93] | +0.04 [0.43] |

Readings:
* **a = 1.2 removes about half of the last-3-s bias** (+1.34 to +0.70) and of the slow-start bias.
* The residual is co-adapted to the high a. With a = 1.2 and the residual kept, post-merge and far-upstream accelerations become
  -0.5 too low. With the residual removed they are right (+0.11, +0.09).
* The calibrated core alone has less than half the one-step error of the full model on every group. This is a one-step metric; it
  says nothing about the residual's rollout role.
* A constant a in 0.8-1.2 minimises the all-row MAE (0.94 vs 0.97 at the floor 1.78).

### 3.4 What remains after calibration is adjacency (SUMO, recorded rows)

* For last-3-s target followers, SUMO's state-matched response (25-NN on v, gap, leader speed) is +0.66 among far-upstream rows
  and +0.58 among unpartnered merge-zone rows. Recorded is 0.00.
* Textbook IDM predicts +0.84 for the followers themselves. So the merge-specific deficit is ~0.6-0.8, of which ~0.1 is zone-wide
  and the rest adjacency.
* This agrees with R9_YIELD's matched excess (-0.4..-0.5 in C3 cells). The absolute deficit is larger here because the reference is
  IDM, not same-state jammed controls.

## 4. Q2 - what the SUMO follower "sees" (measured on recorded rows; SUMO internals hedged)

| description of the target follower (textbook IDM) | last 3 s bias [MAE] | [-6,-3) s | partner merges > 10 s later / never | far partner (gap > 30 m) |
|---|---|---|---|---|
| own lane-1 leader only | +0.84 [1.05] | +0.47 [0.94] | +0.31 [0.81] | +0.43 [0.57] |
| + ramp car as virtual leader at its current position | **-5.04 [5.15]** | -6.29 | -6.19 | -5.05 |
| hold: min(own IDM, 0) | **-0.12 [0.48]** | -0.25 [0.59] | -0.25 [0.54] | +0.10 [0.31] |

* **The ramp car is not a car-following leader of the SUMO follower before the change.** At the merge-imminent state the ramp car is
  almost alongside: median bumper gap 1.46 m, 34 % overlapping (gap < 0), gap_r < gap_l on 100 % of rows. IDM to it would brake at
  -6..-9 m/s^2. The least-squares share of that deceleration that SUMO followers realise is 0.09-0.30 (by gap; textbook IDM), and
  0.17-0.40 with the WM's IDM.
* **The realised behaviour is a hold.** The follower does not accelerate into the opening gap (recorded mean 0.00 at every
  t - t_merge in [-10, 0) s) and brakes in a minority of rows: 15-23 % below -1 m/s^2 (R9_YIELD). This matches LC2013's cooperative
  speed adjustment as aggregated in LCSTATE 7.3: the target-lane follower is in a cooperative state on 66.4 % of wait car-seconds
  and 18.2 % at the merge step. *Hedge: the per-step LC2013 request (informFollower / speed patch weighted by lcCooperative) is my
  reading of SUMO; this task did not verify it in source or per-vehicle logs (the logs carry vType headers, which I did not open).*
* Hold is too broad when the merge is far (> 10 s: -0.25) and with overlap (-0.35). So it must be gated by the situation (d_end,
  wait, ramp speed and acceleration, gap). This is what the R11 head's inputs carry, so it is learnable, not a rule.
* Virtual leader vs pure cooperation: the virtual leader explains none of the deficit at its natural weight (it overshoots by ~6x).
  Cooperation in the form of a situation-gated hold explains essentially all of it (residual -0.12 in the last 3 s).

## 5. Q3 - inference-only counterfactual on the model's replay rows (round 8; round 10 identical within 0.01)

| group | n | rec | model as is | model + virtual leader | calibrated core (a 1.2, b 2.0), no residual | calibrated + virtual leader | calibrated + hold |
|---|---|---|---|---|---|---|---|
| partners last 3 s | 1850 | -0.03 | +1.34 [1.84] | -0.98 [2.30] | +0.83 [1.01] | -0.52 [1.24] | **-0.08 [0.46]** |
| partners [-6,-3) s | 1694 | -0.05 | +0.78 [1.65] | -0.98 [2.28] | +0.53 [0.89] | -0.55 [1.23] | -0.14 [0.55] |
| partners, merge > 10 s away / never | 11,748 | -0.08 | +0.51 | -1.64 | +0.38 | -0.92 | -0.23 [0.50] |
| far partner (gap_r > 30 m) | 2020 | -0.44 | +0.66 | -1.32 | +0.45 | -0.95 | +0.12 [0.40] |
| partner overlapping (gap_r < 0) | 5796 | -0.02 | +0.17 | +0.17 | +0.15 | +0.15 | -0.35 [0.55] |
| unpartnered zone | 46,356 | +0.06 | -0.28 [1.44] | -0.28 | -0.13 [0.61] | -0.13 | -0.13 [0.61] |
| all rows | 432,374 | +0.05 | +0.11 [0.98] | +0.02 [1.02] | +0.04 [0.43] | -0.02 [0.45] | +0.02 [0.41] |

* Virtual leader with the model's encoded IDM: the follower over-brakes (-0.98). It would open far more gaps than SUMO, so the
  54 % yield would be overshot rather than explained.
* Calibrating alone removes ~40 %. Calibration + a (situation-gated) hold removes the rest.
* **The 54 % active-yield merges of R9_YIELD are the hold.** A held follower stays near 0 while its leader pulls away, so a gap opens
  ahead of it. That gap is the one the ramp car takes. In the WM the follower accelerates into it (+1.3..1.5).

## 6. The replay visibility artefact (measured; affects teacher-forced terms and offline numbers, not rollouts)

`wms.train.losses.cutin_reaction._recorded` sets `valid = loss_valid & onehot.sum > 0` and the replay uses it as `present`. With
`junction_fix = "ffill"` a junction-transit vehicle has a filled lane in `states`, but loss_valid = 0, so it disappears from the
replay state on its transit steps. Round 10 (round 8 identical):

| group | share of rows with a hidden lane-1 vehicle between follower and replay leader | final - rec, hidden rows | not hidden | median hidden gap / replay gap |
|---|---|---|---|---|
| partners last 3 s | 0.199 | +1.60 | +1.28 | 10.8 / 67.9 m |
| partners [-6,-3) s | 0.199 | +1.27 | +0.66 | 9.8 / 61.5 m |
| rows at s 50-75.44 m | 0.293 | +0.86 | -0.52 | 5.4 / 41.0 m |
| rows at s -25..0 m | 0.134 | +1.80 | -0.43 | 1.9 / 9.1 m |
| all lane-1 rows | 0.048 | +0.69 | +0.09 | 16.9 / 76.9 m |

Recorded data agree: in the raw rows a junction (one-hot 0) vehicle is nearer than the lane-1 leader for 26 % of rows at s 50-75 and
12.6 % of last-3-s target followers.
- **Last-3-s mean.** It contributes only ~+0.06 (the base bias without hidden rows is +1.28).
- **Teacher-forced controls.** It corrupts them where they matter: fix-B controls, and the R11 contrast controls, which sit
  86-89 % in the merge zone.

## 7. Offline test of fix 1 (physical encoder, measured, teacher-forced, residual off)

`tools/diag_base_encfit.py`:
- **Capture.** Every driver-encoder call's inputs, from the same replays with the model forward stubbed.
- **Fit.** The physical-mode encoder (head initialised from round 8, centre from the physics defaults), trained by Huber loss of
  the pure IDM core on the recorded lane-1 HDV acceleration.
- **Data.** 72 fit windows; 24 held-out windows, held out by episode.
- **Trained parameters.** Head + centre only, 150 Adam steps x 8 windows.

RESULT_ENCFIT (filled below when the fit finishes)

Caution from a cruder per-driver grid fit (`tools/diag_base_perdriver_fit.py`: each driver's a, b fitted on its own approach rows,
pure IDM).
- The per-driver fitted a piles on the grid floor (median 0.4). One-step MSE alone rewards a sluggish IDM in stop-and-go traffic.
- Pure IDM with those values gives last-3-s bias +0.27 [0.52] and all-row MAE 0.27, but braking +0.50 and post-merge -0.28.
- So a reachable encoder must be trained with the full objective (rollout position/speed terms) and the textbook prior. Check it on
  start-up and braking rows, not only on one-step MAE.

## 8. Q4 - recommendation (proposed)

1. **Fix 1: reachable IDM settings + residual re-initialisation.** This is one base change, so it needs its own run vs the current
   best (one-fix protocol).
   - Config: `model.driver_param_mode = "physical"`, `driver_prior` textbook kept (0.05).
   - Trainable: `driver_encoder.*` (head, centre_logits), `residual_net` with the attention output layer re-zeroed, `temporal.*`,
     gate.
   - Losses: the base WM-S rollout losses as in round 8, fix-B controls (unpartnered only), plus a one-step lane-0/1 HDV anchor.
   - Do not reuse the round-8 residual's constant.
   - Expected effect: last-3-s bias about +1.34 to ~+0.8; slow-start bias +1.52 to ~+0.1..0.8 (zone-dependent); merge-zone
     discharge slower; merge-zone delay ratio up from 0.48.
2. **Fix 2: learned hold.** The R11 contrast head on the fix-1 base; same code, same rules.
   - Expected effect: last-3-s absolute bias to ~0 (the calibrated + hold diagnostic gives -0.08), the follower no longer fills the
     ramp gap, ramp delay ratio down from 2.1, fewer stuck ramp cars.
   - Keep fix B for the post-cut-in over-braking (R9_YIELD 5.2), which is a separate channel.
3. **Fix 3: replay mask.** Small code change in `_recorded` / `one_step_replay`: present from the filled window presence, loss rows
   from loss_valid. Ideally do it before fix 1's training if fix 1 uses teacher-forced anchors. Re-run R11's offline numbers after
   it.
4. **Not recommended.**
   - A hand-set or virtual-leader yield (over-brakes; the owner forbids rules anyway).
   - More rounds on temporal/gate/fix B alone: round 10 shows they leave the cause untouched.
   - Interpreting R11's contrast head as an absolute fix before fix 1.

**Offline gates before any GPU run.**
- (a) The section 7 fit plus the same fit with rollout-free textbook-prior variants. Pass: accel median 0.9-1.5, not at a bound;
  far-upstream start-up |bias| < 0.2; braking rows not worse than the model; all-row MAE <= 0.5.
- (b) R11 offline fit (`tools/r11_courtesy_offline.py`) re-run on the fix-1 base with the fix-3 mask. Pass: last-3-s ABSOLUTE bias
  within +-0.3, M1 excess -0.2..-0.45, negative controls within 0.2.
- (c) TRAIN-state rollouts (CPU or a free GPU) vs round 10b: merge-zone delay ratio, ramp delay ratio, stuck ramp cars, follower min
  acceleration after cut-ins, TTC.

## 9. Reproduction

```
PY=python; R=E:/agent-work/c4-v10r10b/tmp_r10b/roots; CK8=E:/agent-work/c4-v10r9fix/data/r9_train/wm_s_stage300.pt
python tools/diag_base_landscape.py tmp_r11/cache                         # bias landscape from the R11 cache
for i in 0..7: python tools/diag_base_decomp.py --config c4_wm_v10round10b.toml --set model.follower_courtesy_version=2 \
    --ckpt $CK8 --roots $R/v5_native $R/genrand_plain --out tmp_basediag/r8 --max-windows 12 --shard $i --n-shards 8
(same with --config c4_wm_v10round10.toml --ckpt c4-v10r10/tmp_diag/r10/wm_s.pt --out tmp_basediag/r10)
python tools/diag_base_decomp_an.py tmp_basediag/r8 tmp_basediag/r10
python tools/diag_base_params_cf.py tmp_basediag/r8; python tools/diag_base_cf_forms.py tmp_basediag/r8
python tools/diag_base_vleader.py $R tmp_basediag/vleader 1; python tools/diag_base_vleader_an.py; python tools/diag_base_vleader_forms.py
python tools/diag_base_perdriver_fit.py tmp_basediag/r8
for i in 0..7: python tools/diag_base_encfit.py capture --ckpt $CK8 --set model.follower_courtesy_version=2 --roots ... --shard $i --n-shards 8
python tools/diag_base_encfit.py fit --ckpt $CK8 --set model.follower_courtesy_version=2 --threads 8 --steps 150
```
Limits:
- One-step teacher-forced metrics only; no rollouts.
- 96 windows per model (1,850 last-3-s target-follower rows).
- Matched references are nearest-neighbour, not cells.
- The SUMO mechanism behind the hold is inferred from aggregates.
