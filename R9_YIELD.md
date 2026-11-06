# R9_YIELD - do lane-1 followers yield to waiting ramp cars, is it driver-dependent, does the WM reproduce it

S-YIELD, 2026-10-23. Analysis only: no training, no SUMO, no edits to model or training code. TRAIN data only
(`c4-v10pfit/data/r9_train`, 192 episodes = 96 `v5_native` + 96 `genrand_plain`, t >= 300 s, demand 1900-2800 vph, penetration 0.05-0.40).
No `hetdata_truth`, no vType files, no sealed/holdout paths, no eval seeds. Scripts: `tools/diag_yield_*.py` (committed on `c4/v10r9excl`, not pushed).
Raw outputs: `E:/agent-work/diag_yield_out/` (q1q2.txt, c3.txt, jam.txt, wmstats96*.txt, encounters.csv).

## 0. Answer first

1. **Followers do yield, and the effect is local in space and time.** Only the *target follower* (nearest lane-1 vehicle behind the waiting ramp car, 1-15 m behind its
   rear bumper, offset dA = s_ramp - s_follower of 6-20 m) shows it: matched excess acceleration -0.40 to -0.51 m/s^2 (mean), median -0.3 to -0.4, 14-23 % of rows below
   -1 m/s^2 (controls ~1 %). It starts ~6 s before the merge (mean excess -0.14), peaks ~1.5 s before (-0.45; 38 % of rows < -0.5) and is gone at the merge instant.
   Other lane-1 followers, and followers with the ramp car *behind* them, show ~0 (+-0.05). It grows when the ramp car is near the lane end (d_end < 10 m: -0.5 to -0.9
   vs -0.1 for d_end > 35 m), has waited long (>= 40 s: -0.4 vs -0.18 for < 5 s), creeps (0.5-4 m/s) and the bumper gap is 2-5 m (-0.54 to -0.65). Demand and CAV
   penetration do not matter.
2. **Merges enabled by active follower slowdown (C3 control): 54 % of merges, 74 % of merge-lane wait time**; 17 % (14 % of wait) enter an existing slow-jam gap with no
   extra slowdown; 29 % (11 %) enter a free gap (follower far behind or fast). The follower's median RAW acceleration at waiting moments is 0.00 and its median speed
   1.5 m/s (IQR 0.6-3.7), so the stuck-diagnosis observation holds for raw numbers: the yield shows up as a matched deficit (the adjacent follower does not discharge as a
   same-state non-adjacent follower does, plus a 15-20 % tail of real braking), not as a median brake.
3. **No detectable driver dependence.** 82 % of the 7,331 followers with an encounter have exactly one; for the 1,376 with >= 2, corr(first, second encounter yield score)
   = 0.087 [0.031, 0.147] raw and 0.038 [-0.013, 0.095] after removing the situation (d_end, wait, ramp speed, gap, own speed, demand; the situation alone explains R2 = 0.56).
   Pre-encounter car-following behaviour (headway, accel/brake style, speed level, closing response) adds R2 = 0.000 under held-out-episode CV. Yield propensity here is a
   function of the situation, not an inferable per-driver trait.
4. **The WM does not yield before the merge (rounds 8 and 9), and over-reacts after the cut-in.** With the recorded CAV arm replayed (96 TRAIN states, k = 2, 120 s) the target
   follower's excess acceleration vs same-condition non-adjacent followers is +0.15 [+0.07, +0.23] (r9) / +0.15 [+0.06, +0.24] (r8) for dA 6-15 m (SUMO TRAIN: -0.40), +0.41 / +0.50
   at 1 s before the merge (SUMO -0.27 on the same states), and -0.33 / -0.17 at +1 s after it (SUMO +0.23). A causal removal test (same seed, with vs without the waiting ramp
   car) shows the r9 follower only ~0.2 m/s slower after 3-10 s (r8 ~0.35), versus a matched SUMO deficit of ~1.4 m/s over 5 s. Round 8's hard-coded yield is visible only at long
   range (removing a ramp car >= 60 m ahead of a follower changes its speed by 1-3 m/s). The 10-21 claim ("over-reaction after the cut-in, not a missing pre-merge yield") is therefore
   half right for r8/r9: **both exist**.
5. **Proposal (section 6):** a learned pair-interaction courtesy head for lane-1 followers (nearest waiting ramp car as the pair partner, smooth geometry inputs, no thresholds,
   no yield rule), trained with matched-control contrast losses and a merge-event rollout term; a per-driver courtesy latent only as an ablation, because section 4 finds no driver signal.

## 1. Data and definitions

* Samples: every 1 s, every lane-0/1 vehicle on s in [-150, 80] m (pack frame; junction rows have no lane and are skipped as neighbours): 2,054,265 rows, 1,119,768 HDV lane-1 rows.
  Acceleration is the recorded `veh_kin[..., 5]` (backward difference: the acceleration that produced row t; verified against dv/dt, error 1e-6).
* Waiting ramp car R: lane 2, 0 <= s <= 75.44 m (acceleration lane). Merge = lane 2 -> lane 1 in consecutive rows: 18,549 events (14,785 with an HDV follower).
* Target follower F of R: nearest lane-1 vehicle with s_F <= s_R. dA = s_R - s_F (bumper gap = dA - 5 m).
* Matched expectation: mean recorded acceleration of HDV controls in the same covariate cell: own speed (9 bins) x time headway to own lane-1 leader (8) x closing speed to the
  leader (6) x position (6) x demand (2); a cell needs >= 20 control rows. Excess = a - cell mean.
  - **C1**: lane-1 HDV rows with no ramp-lane car within [s-30, s+150] (quiet-ramp control; covers 81 % of rows, biased to freer cells).
  - **C2**: lane-0 HDV rows (never adjacent to the ramp; lane 1 is generally more congested, so C2 over-states the effect by 0.1-0.3 m/s^2; sensitivity only).
  - **C3 (cleanest)**: lane-1 HDV rows in zones where a waiting ramp car exists but is not adjacent (not the target follower, no waiting ramp car within 30 m ahead or 20 m behind).
    This holds "a ramp car waits / the zone is jammed" fixed and varies only adjacency. 513,823 control rows, covers 71 % of target-follower rows.
* Placebo: the same event statistic 20 s earlier on the same followers (window -25..-20 s); checks the contrast is not generic stop-and-go.
* CAV followers are excluded from all tables (HDVs are what the WM must predict). Everything is descriptive; no thresholds enter any model.

## 2. Q1 - when and how much (SUMO TRAIN)

### 2.1 Excess acceleration of lane-1 HDV target followers vs ramp-car offset dA

| dA (m) | n (C1) | mean (C1) | median | p10 | share < -1 | mean (C3) | share < -1 (C3) |
|---|---|---|---|---|---|---|---|
| 0-3 (alongside) | 45,952 | -0.185 | -0.083 | -0.80 | 0.067 | -0.166 | 0.061 |
| 3-6 (alongside) | 38,071 | -0.031 | +0.037 | -0.73 | 0.053 | -0.003 | 0.047 |
| 6-10 | 52,845 | -0.470 | -0.392 | -1.28 | 0.227 | -0.402 | 0.202 |
| 10-15 | 13,250 | -0.510 | -0.394 | -1.44 | 0.193 | -0.405 | 0.162 |
| 15-20 | 3,724 | -0.399 | -0.341 | -1.16 | 0.144 | -0.232 | 0.093 |
| 20-30 | 3,847 | -0.352 | -0.301 | -1.17 | 0.145 | -0.078 | 0.061 |
| 30-60 | 7,115 | -0.38 | -0.31 | -1.25 | 0.185 | -0.06 | 0.04 |
| 60-100 | 5,920 | -0.278 | -0.209 | -1.02 | 0.108 | -0.026 | 0.032 |

Beyond ~20 m C1 still shows -0.3 but C3 shows ~0: C1 leaks the general "ramp car waiting => jammed zone" difference; the local yield is the C3 pattern (dA < 20 m).
Controls: lane-1 rows with a ramp car behind (dB 0-10 m) +0.045; non-target lane-1 followers -0.05 to +0.03 (C1) at every dA 6-150; C1 quiet-ramp rows 0.000.

### 2.2 Event-aligned profile of the follower at the merge (HDV F*, t_m = first lane-1 row of R)

| t - t_m (s) | n (C1) | mean excess C1 | median C1 | share < -0.5 (C1) | mean excess C3 |
|---|---|---|---|---|---|
| -30 | 5,983 | -0.036 | -0.001 | 0.084 | -0.009 |
| -10 | 5,812 | -0.076 | -0.006 | 0.120 | -0.056 |
| -8 | 6,087 | -0.088 | -0.008 | 0.130 | -0.074 |
| -6 | 6,400 | -0.165 | -0.031 | 0.180 | -0.136 |
| -5 | 6,408 | -0.186 | -0.039 | 0.199 | -0.155 |
| -4 | 6,492 | -0.242 | -0.082 | 0.244 | -0.208 |
| -3 | 6,919 | -0.368 | -0.217 | 0.323 | -0.334 |
| -2 | 6,852 | -0.426 | -0.261 | 0.366 | -0.390 |
| -1.5 | 6,947 | -0.451 | -0.271 | 0.376 | - |
| -1 | 7,133 | -0.405 | -0.252 | 0.352 | -0.375 |
| 0 (merge row) | 7,246 | -0.015 | +0.001 | 0.098 | +0.056 |
| +1 | 10,978 | -0.113 | -0.051 | 0.168 | -0.013 |
| +2 | 11,703 | -0.077 | +0.008 | 0.175 | -0.019 |
| +4 | 11,201 | -0.077 | -0.020 | 0.134 | -0.029 |

* **Onset ~6 s before the merge** (mean excess first below -0.15), duration ~6 s, mass in the last 3 s. There is no after-merge deficit in SUMO (the cut-in itself costs ~0).
* Where the ramp car is at the merge: s = 4.5 / 24.9 / 47.4 / 66.8 / 75.4 m (p5/25/50/75/95), median distance to lane end 28 m; wait in the zone p10/50/90 = 2.5 / 21 / 71 s.
* Integrated effect, excess speed shed over the last 5 s: mean 1.46 m/s (C1) / 1.37 (C3) vs 0.17 / 0.05 in the placebo window; share shedding > 1 m/s 0.56 / 0.54 vs 0.15 / 0.13;
  > 2 m/s 0.39 / 0.38 vs 0.09 / 0.06. Over 10 s: 1.95 / 1.84 vs 0.35 / 0.09.

### 2.3 Dependence (target followers, dA < 20 m, C1; C3 agrees: d_end 0-5 -0.49, 5-10 -0.59, 10-20 -0.38, 20-35 -0.17, 35-50 -0.10, 50-76 -0.05)

| variable | bins | n | mean excess | share < -1 |
|---|---|---|---|---|
| ramp car distance to lane end (m) | 0-5 / 5-10 / 10-20 / 20-35 / 35-50 / 50-76 | 21.7k/11.9k/13.6k/21.1k/22.1k/41.9k | -0.48 / -0.56 / -0.40 / -0.21 / -0.13 / -0.11 | 0.23 / 0.21 / 0.14 / 0.11 / 0.08 / 0.06 |
| ramp car wait so far (s) | <5 / 5-10 / 10-20 / 20-40 / 40-80 / >80 | 26k/20k/29k/35k/29k/16k | -0.18 / -0.25 / -0.23 / -0.26 / -0.38 / -0.41 | 0.09 / 0.13 / 0.11 / 0.12 / 0.17 / 0.20 |
| ramp car speed (m/s) | <0.5 / 0.5-1 / 1-2 / 2-4 / 4-8 / >8 | 86k/13k/16k/13k/16k/8.7k | -0.24 / -0.41 / -0.40 / -0.39 / -0.23 / -0.13 | 0.13 / 0.17 / 0.14 / 0.15 / 0.12 / 0.07 |
| bumper gap follower->ramp car (m) | <0 / 0-1 / 1-2 / 2-3 / 3-5 / 5-10 / 10-15 | 71k/13k/16k/18k/19k/13k/3.7k | -0.14 / +0.02 / -0.18 / -0.54 / -0.65 / -0.51 / -0.40 | 0.06 / 0.05 / 0.11 / 0.24 / 0.32 / 0.19 / 0.14 |
| follower speed (m/s) | <1 / 1-3 / 3-6 / 6-10 / 10-15 / >15 | 46k/64k/21k/9.9k/4.6k/7.4k | -0.23 / -0.23 / -0.40 / -0.56 / -0.45 / -0.20 | 0.11 / 0.16 / 0.12 / 0.13 / 0.13 / 0.06 |
| demand (vph) | 1800-2000 / 2000-2200 / 2200-2500 / 2500+ | 23k/43k/44k/44k | -0.31 / -0.29 / -0.26 / -0.26 | 0.14 / 0.13 / 0.13 / 0.13 |
| CAV penetration | <0.1 / 0.1-0.2 / 0.2-0.3 / 0.3+ | 23k/46k/48k/36k | -0.28 / -0.28 / -0.28 / -0.28 | 0.13 / 0.13 / 0.13 / 0.13 |

d_end x wait (mean excess, C1), wait bins <10 / 10-30 / 30-80 / >80 s: d_end 0-10: -0.91 / -0.57 / -0.49 / -0.43; 10-25: -0.47 / -0.28 / -0.35 / -0.44; 25-50: -0.21 / -0.11 / -0.09 / -0.09;
50-76: -0.13 / -0.09 / -0.06 / -0.13. The strongest yield is a new arrival at the lane end (urgency); it stays ~ -0.45 for long waits; mid-lane waits are barely yielded to.
Alongside overlap (bumper gap < 1 m) gives no yield (the follower is not behind the car it must let in).
The round-8/9 courtesy loss mask (lane-1 vehicles 0-40 m behind a ramp car within 40 m of the end, `src/wms/train/courtesy.py`) has only 52 % of its rows in this effective set (nearest-ramp-car
approximation); the other 48 % carry excess -0.001 (diluting the 3x weight).

## 3. Q2 - what ends the follower-blocked wait (merge events, HDV F*, F* in lane 1 for the last 5 s)

Active yield := excess speed shed > 1 m/s in the last 5 s (C3 expectation); "existing jam" := no such shed and F* speed < 4 m/s 5 s before; "free gap" := no shed and F* faster.

| class | share of merges | share of merge-lane wait time | median final bumper gap (m) | median wait (s) |
|---|---|---|---|---|
| active follower slowdown (C3) | 0.540 | 0.743 | 3.4 | 22 |
| existing slow-jam gap, no extra slowdown | 0.172 | 0.143 | 3.6 | 10 |
| free gap (follower far / fast) | 0.289 | 0.114 | 212 | 4 |

* Hold-speed counterfactual (F* keeps its speed from 10 s before): the bumpers would already overlap (gap < 0) in 22.9 % of all merges, be < 2 m in 29.8 %, < 3.4 m in 36.1 % (41.5 % overlap among
  the "slowed" merges): in those the merge was physically enabled by the follower giving up distance (median 3.3 m given up).
* By F* speed 5 s before the merge (excess shed in the last 5 s vs placebo, C3): v 0-1 m/s: 0.96 vs -0.22 (share > 1 m/s 0.48 vs 0.01); 1-2: 1.96 vs 0.05 (0.78 vs 0.04); 2-4: 2.92 vs 0.21
  (0.90 vs 0.15); 4-8: 5.17 vs 0.70 (0.98 vs 0.38); > 8: -0.42 vs 0.15 (0.11 vs 0.23; fast followers are those with a free gap). For near-stationary followers "shed" means not accelerating as
  same-state controls do; they cannot brake.
* **Raw-only caution (the stuck-diagnosis question).** Raw follower acceleration at waiting moments (target followers, dA < 20 m): median +0.000, p25 -0.34, p10 -0.87, speed median 1.5 m/s; the
  matched controls' median expected acceleration is +0.26, and 71 % of rows have v_F < 3 m/s. Raw speed change alone cannot separate yield from stop-and-go (share losing >= 0.5 m/s in the last 5 s:
  35.7 % vs 34.2 % in the placebo window). Raw classification: 45 % of merges (55 % of wait time) look like entering an existing slow jam, 36 % (37 %) a slowdown, 19 % (8 %) a fast gap.
  Matched classification (C1 and C3): the target follower is systematically slower/less accelerating than state-matched non-adjacent followers, strongly and only when adjacent (section 2.1; ramp car behind and
  non-target followers show 0). Both statements hold: **merges happen in a slow jam, and the jam's adjacent follower is held back or slowed by the ramp car's presence.** The matched reading rests on
  the control being valid; C1 and C3 agree, C2 (lane 0) is larger.
* The LC2013 log agrees qualitatively: the target-lane follower is in a cooperative state in 66 % of wait car-seconds but only 18 % at the merge step (LCSTATE.md section 7.3).

## 4. Q3 - driver dependence (behaviour only)

Encounter := (episode, follower, ramp car) with >= 3 s of target-follower exposure at dA < 20 m; score := mean C1 excess over the encounter. 8,889 encounters, 7,331 followers
(5,955 with 1 encounter, 1,204 with 2, 162 with 3, 10 with 4; mean 1.21).

| test | value |
|---|---|
| within-encounter odd/even-row correlation (measurement reliability) | 0.753 (n = 5,414) |
| situation-only regression R2 on the encounter score (d_end, d_end^2, log wait, ramp speed, dA, own speed, demand) | 0.556 |
| corr(first, second encounter) of the same follower, raw | 0.087 [0.031, 0.147] (n = 1,376) |
| same, situation-adjusted residual | 0.038 [-0.013, 0.095] |
| P(strong yield [score < -0.5] at 2nd encounter, given strong at 1st / not strong at 1st) | 0.223 / 0.162 (base 0.167) |
| held-out-episode (5-fold) R2: situation only / + past behaviour (7 features) | 0.543 / 0.543 |
| residual correlation of the encounter score with past features (headway median and p20, p90 accel, p10 decel, p90 speed, follow share, closing accel) | |r| <= 0.07 (decel p10 -0.070, closing accel -0.059; others < 0.05) |

Past behaviour = rows with s < -60 m before the first encounter (>= 30 s of data; 2,505 complete encounters). A driver trait explaining >= 10 % of the adjusted variance would give an encounter correlation
>= 0.10; the observed upper bound is 0.095. With 1.2 encounters per driver and these behaviour summaries, yield propensity in SUMO TRAIN is explained by the situation; a per-driver courtesy latent has nothing
measurable to capture (caveat: only 1,376 repeat drivers; a trait explaining < ~10 % of the adjusted variance would be missed).

## 5. Q4 - does the WM yield? (round 8 = `c4_wm_v10_round8/wm_s_stage300.pt`, round 9 = `c4_wm_v10_round9/wm_s_stage300.pt`)

Setup: 96 TRAIN states (every 4th episode of each source, t0 = 450 and 750 s; records built from the converted episode arrays, entries OFF, t0 roster only, 120 s), recorded native arm replayed,
k = 2 lane seeds, own dir on gpuR (`/root/autodl-tmp/c4/v5/diag_yield`, outputs `/root/yield_out`). **Contract note:** the round-9 `first_passage` step ignores `cav_teacher`; the harness feeds the
recorded CAV arm through the `cav_controller` hook (as `tf=1` in `diag_r9stuck_trace.py`). A first run without replay (48 states) showed the same qualitative picture and was discarded.
SUMO columns are the recorded future of the same t0 rosters (small n: 347 events). The same code (`rows_from_arrays`, `events_from_arrays`) defines all quantities on SUMO and WM trajectories; each
side is compared with its OWN C3-type controls (lane-1 rows with a waiting ramp car in the zone but not adjacent), because the WM baseline acceleration distribution differs (control p10 -2.7 vs -1.3).

### 5.1 Near-field excess (target followers, C3-own, mean [95 % state-bootstrap CI] where available)

| | SUMO recorded (same states) | SUMO full TRAIN | WM r9 | WM r8 |
|---|---|---|---|---|
| dA 6-15 m | -0.034 [-0.093, +0.014] (n 866) | -0.40 (C3, n 75k) | **+0.147 [+0.069, +0.226]** (n 2,658) | **+0.154 [+0.063, +0.240]** (n 2,568) |
| dA < 20, d_end 0-10 m | -0.515 (n 121) | -0.49 / -0.59 | +0.015 (n 1,630) | -0.006 (n 1,413) |
| dA < 20, d_end 10-20 m | -0.156 (n 66) | -0.38 | -0.006 (n 837) | -0.096 (n 753) |
| dA < 20, d_end 40-80 m | +0.022 | -0.05 | +0.165 | +0.169 |
| p10 of raw acceleration, quiet-ramp lane-1 rows | -1.29 | - | -2.67 | -2.65 |

### 5.2 Event-aligned follower response at the merge (HDV F*, C3-own mean excess [95 % CI])

| t - t_m | SUMO recorded (n 47-80) | WM r9 (n 87-129) | WM r8 (n 103-160) |
|---|---|---|---|
| -6 s | -0.139 [-0.28, +0.01] | +0.156 [-0.10, +0.42] | +0.368 [+0.10, +0.63] |
| -3 s | -0.055 [-0.28, +0.09] | +0.233 [-0.02, +0.53] | +0.172 [-0.14, +0.51] |
| -2 s | -0.137 [-0.33, +0.02] | +0.150 [-0.13, +0.44] | +0.220 [-0.10, +0.49] |
| -1 s | **-0.270 [-0.41, -0.14]** | **+0.411 [+0.18, +0.65]** | **+0.496 [+0.30, +0.65]** |
| +1 s | **+0.230 [+0.10, +0.37]** | **-0.332 [-0.70, -0.04]** | -0.168 [-0.43, +0.08] |
| +2 s | +0.079 [-0.04, +0.18] | +0.362 [+0.13, +0.62] | +0.473 [+0.23, +0.73] |

With the SUMO-train-cell expectation the signs are the same: at -2 / -1 s SUMO -0.43 / -0.48, WM r9 -0.04 / +0.25, r8 -0.11 / +0.23; at +1 s SUMO +0.04, r9 -0.55, r8 -0.66.
Follower minimum acceleration (HDV F*, medians, m/s^2): window [-6, +6] s SUMO -0.91, r9 -3.21, r8 -3.23; pre [-6, 0) -0.59 / -2.14 / -2.17; post [0, +6] -0.35 / -3.06 / -2.97.
Both models (a) do not slow ahead of the merge and (b) over-brake after it. The 10-21 decomposition's "pre-slowdown share" counted any slowdown, not one matched to the ramp car, so it could not see (a).

### 5.3 Causal removal test (same seed; remove one waiting HDV ramp car from state, history and future; 60 steps = 30 s)

172 near pairs per model (target follower 3-25 m behind), 20 far placebos (follower >= 60 m behind), 2 seeds; mean of v_F(with R) - v_F(without R) [95 % CI]:

| model, pair kind | 3 s | 6 s | 10 s | 20 s | share with dv(10 s) < -0.5 |
|---|---|---|---|---|---|
| r9 near | -0.22 [-0.33, -0.14] | -0.11 [-0.29, +0.06] | -0.20 [-0.51, +0.09] | -0.11 [-0.56, +0.37] | 0.17 |
| r8 near | -0.37 [-0.49, -0.26] | -0.30 [-0.49, -0.10] | -0.35 [-0.62, -0.05] | -0.46 [-0.92, +0.03] | 0.27 |
| r9 far (placebo) | +0.25 [-0.15, +0.94] | +0.57 | +1.08 | -1.45 | 0.15 |
| r8 far (placebo) | -1.01 [-1.87, -0.31] | -2.54 | -3.15 [-5.97, -0.71] | -4.89 | 0.40 |

By offset (10 s): r9 dA 3-8 -0.26, 8-14 -0.14, 14-25 -0.14; r8 -0.61, -0.49, +0.36. The r9 follower reacts to a waiting ramp car by ~0.2 m/s (about 1/7 of the SUMO matched deficit of ~1.4 m/s per 5 s) and
the effect fades by 6 s; r8 ~0.35 m/s and, through the hard-coded `mainline_merge_yield` (lookahead 120 m; 60 m / 2.0 m/s^2 scope), also slows followers >= 60 m back by 1-3 m/s - a long-range effect the data
do not have (SUMO C3 beyond 20 m: ~0). Round 9's pivot step bypasses that rule (`merge_core` only runs when `merge_pivot is None`); the learned courtesy inputs (`courtesy_inputs`, `courtesy_weight` = 3 in
both rounds) have not produced a measurable pre-merge response.

Caveats: 96 states x 2 seeds, roster-limited (no entrants; SUMO exits end slots); 347 / 573 / 600 events; CAV followers excluded (the WM has 275 CAV-follower events vs 92 in SUMO because teacher-replayed CAVs are
followers more often); the removal test has no SUMO counterfactual (the SUMO reference is the matched deficit); the r8 far placebo shows the removal test is sensitive to long-range rules.

## 6. Q5 - design proposal (not implemented)

**What the data require.** A response on one pair (target follower <- waiting ramp car), local (dA ~ 6-20 m), active ~6 s before the merge with peak -0.45 m/s^2 mean (heavy tail: 15-23 % of rows < -1 m/s^2),
with amplitude depending on d_end, wait, ramp speed and bumper gap, not on demand, penetration or (measurably) the driver; plus a separate post-insertion channel that is ~0 in SUMO.

**Model.** Add a pair-interaction head to the HDV residual (it adds a term to the existing acceleration residual; it replaces nothing):
1. Partner by geometry, not by a threshold: for each lane-1 HDV slot i take the nearest lane-2 slot j with s_j >= s_i - L (or soft attention over lane-2 slots with a learned distance kernel); a "none"
   flag and zeroed features when no ramp-lane car exists. This is the target-follower relation used in this analysis.
2. Inputs, all observable in rolled-out state: bumper gap, v_j, v_i, v_j - v_i, a_j, d_end(j), carried merge wait of j (`merge_wait_s`, exists in the state), the merge head's detached intent probability of j
   (exists), i's own leader gap/closing speed/time headway (so the head only learns what car-following does not explain), and the driver embedding of i as an optional input.
3. Output: a stochastic acceleration offset (heteroscedastic or Student-t head). The observed offset is heavy-tailed, so a mean-only Huber head under-produces the tail. No sign constraint is needed by the data
   (the excess is negative on average, not always); a softplus magnitude would be an architectural prior, not a rule.
4. Per-driver courtesy latent: not recommended as a first-class component given section 4 (corr 0.04 [-0.01, 0.10]); keep the existing driver embedding as a head input and evaluate by ablation (zero it; require a
   held-out gain) before adding any dedicated latent.

**Training.**
* Teacher-forced one-step term on recorded lane-1 follower accelerations with the pair input present. Weights come from the pair relation itself (target follower of an existing waiting ramp car), not from the
  40 m / 40 m window of `courtesy_follower_mask` (52 % effective rows). Add matched-negative contrast: for each exposed row sample a state-matched non-exposed row (cells as in section 1, built offline from TRAIN)
  and penalise the difference between the predicted acceleration change (pair present minus pair removed) and the matched recorded difference. This identifies the effect against the jammed-zone confound (C1 vs C3).
* Rollout term: for each recorded merge event take [t_m - 12 s, t_m + 6 s], roll the WM from t_m - 12 s with the recorded CAV arm, apply (i) the usual follower position/speed loss and (ii) the first-passage merge
  log-mass of the ramp car, so the gradient of "the merge happened when it did" flows into the follower's acceleration through the gap. This gives the courtesy channel its functional role (opening a gap) without a
  written rule and targets finding Q2 (74 % of wait time ends via follower slowdown).
* Post-insertion: a separate response term for the new follower after t_m using the same matched-control contrast (SUMO excess ~0 at +1 s; WM -0.3 to -0.7), so the pre-merge head cannot be paid for by a harsher
  post-cut-in reaction. The post-insertion over-brake (min accel -3.0 vs -0.35 m/s^2) is the larger distortion in absolute size and should be fixed first or jointly.
* Acceptance before any Dreamer-loop training: (a) matched near-field excess (C3-own) for dA 6-15 and d_end < 10 within the SUMO TRAIN band (-0.4 to -0.6) with a CI excluding 0; (b) event profile -6..0 s with onset
  ~-6 s, peak ~-0.45 at -1.5 s, ~0 at +1..+2 s; (c) removal-test follower speed deficit of the SUMO order (~1 m/s over 5 s); (d) negative controls: no response for followers with the ramp car behind, dA > 30 m, and
  lane-0 cars; (e) simulated yield-enabled merge share (Q2: ~54 % of merges / 74 % of wait) and the merge-lane wait distribution.

**Must not be done.** No hand-set yield rule (round 8's `mainline_merge_yield` created a long-range effect the data do not have); no thresholds or window constants in the model or loss that encode "yield if ..."
(matching cells are an offline evaluation device, not model inputs); no type labels, `hetdata_truth`, vType or lcCooperative values, and no use of the LC2013 cooperative state bits as training targets (simulator-internal,
not observable at deployment); no per-driver trait fitted on Q3 data (nothing to fit); no tuning against sealed/eval seeds.

## 7. Reproduction and limits

```
python tools/diag_yield_extract.py 20            # samples.npy
python tools/diag_yield_events.py 20             # events.npz (18,549 merges)
python tools/diag_yield_drivers.py 20 && python tools/diag_yield_q3.py
python tools/diag_yield_q1q2.py; python tools/diag_yield_c3.py; python tools/diag_yield_jam.py
PYTHONPATH=src:. python tools/diag_yield_states.py states96.json.gz 4      # state records from TRAIN episodes
# gpuR (repo_r9_3e5610fe_fit): diag_yield_wm.py --states states96.json.gz --var r9=<ckpt> --var r8=<ckpt> --workers 24 --k 2 --steps 240
#                              diag_yield_remove.py --states states96.json.gz --var r9=<ckpt> --var r8=<ckpt> --workers 24 --k 2 --steps 60
python tools/diag_yield_wmstats.py wm96.pkl; python tools/diag_yield_removestats.py remove96.pkl
```
Limits: matched-control coverage is 71-81 % of rows (cells with < 20 controls are dropped, biased to freer states); one lane-1 neighbour definition; 1 s sampling; pack-frame positions (junction rows skipped);
Q4 uses 96 states, k = 2, roster-limited rollouts; section 2.3 bin counts are rounded.
