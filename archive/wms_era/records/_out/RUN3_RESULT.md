# RUN 3 vs the C1 structured model (run-2 bundle)

CPU-only, no training, no SUMO. Run 3 (checkpoints `wm_s.pt` = final /
stage-300, and `wm_s_stage120.pt`) was trained on the SAME leave-PI-out
windows and C1-matched stage steps as the C1 structured model, so this is
the clean equal-training test of the run-2 bundle (entries v2 + merge-cal
+ junction fix + projection/residual cap/contractive stability +
conservation + fixed traffic loss + 300 s stage).

## Verdict

- **run-3 final**: U-same paired Δ (run-3 − C1) @30 s -0.56 [-0.93, -0.08], @60 s 0.15 [-1.49, 2.09], @120 s 44.90 [29.54, 67.24] m (negative favours run 3).
- **run-3 stage-120**: U-same paired Δ (run-3 − C1) @30 s -0.84 [-1.39, -0.47], @60 s -0.62 [-2.48, 0.96], @120 s 23.60 [9.31, 48.61] m (negative favours run 3).

**Does the run-2 bundle beat C1 struct at equal training?** **Partly, and not uniformly.** Both run-3 checkpoints are better than C1 struct
at the near horizons (10 s and 30 s, U-same CIs below 0), are statistically
indistinguishable at 60 s, and the **final (stage-300) checkpoint is clearly
worse at 120 s** (U-same +44.9 m [29.5, 67.2], CI excludes 0) with a higher
blow-up rate. The **stage-120 checkpoint** roughly halves that long-horizon
penalty (U-same +23.6 m [9.3, 48.6], still worse than C1 but far less than the
final model) and on the PI-free `seen` set it is slightly better than C1
(−15.0 m [−43.6, 0.4]) with a lower blow-up rate. Traffic-level downstream
errors (speed, delay, queue, stopped) are smaller for run 3, and the projected
trajectories have far fewer same-lane overlaps. So: the bundle helps short
horizons, traffic aggregates and physical plausibility; it does not help (and can
hurt) long-horizon HDV position accuracy, and the 300 s stage over-extends.

## 1. HDV |ds| median (m) by horizon — run 3 vs C1 struct vs black-box

Arm under test: PI for the unseen PI sets (`u_*`), native for `seen`.

| set | horizon | run3 (final) | run3 stage120 | C1 struct | bb-traj | bb-id |
|---|---|---|---|---|---|---|
| u_same | 10 s | 1.81 | 2.00 | 2.35 | 2.92 | 2.37 |
| u_same | 30 s | 6.22 | 5.71 | 6.69 | 10.79 | 9.88 |
| u_same | 60 s | 17.32 | 15.04 | 16.94 | 23.79 | 28.17 |
| u_same | 120 s | 152.82 | 119.42 | 69.74 | 156.48 | 94.48 |
| u_new | 10 s | 1.65 | 1.84 | 2.23 | 2.39 | 2.04 |
| u_new | 30 s | 6.94 | 6.55 | 5.65 | 10.17 | 10.69 |
| u_new | 60 s | 12.76 | 12.54 | 15.43 | 22.40 | 23.59 |
| u_new | 120 s | 105.02 | 69.81 | 39.50 | 173.30 | 83.70 |
| seen | 10 s | 1.66 | 1.84 | 2.17 | 2.65 | 2.43 |
| seen | 30 s | 7.36 | 7.87 | 8.19 | 13.27 | 14.08 |
| seen | 60 s | 29.70 | 25.04 | 28.50 | 32.31 | 47.71 |
| seen | 120 s | 148.86 | 126.59 | 146.80 | 196.17 | 217.45 |

## 2. Paired world bootstrap vs C1 struct (median Δ, 95% CI)

Negative favours run 3. Whole worlds resampled; states paired by (world, t0).

| set | horizon | run3 final Δ | run3 stage120 Δ |
|---|---|---|---|
| u_same | 10 s | -0.50 [-0.62, -0.42] | -0.30 [-0.38, -0.24] |
| u_same | 30 s | -0.56 [-0.93, -0.08] | -0.84 [-1.39, -0.47] |
| u_same | 60 s | 0.15 [-1.49, 2.09] | -0.62 [-2.48, 0.96] |
| u_same | 120 s | 44.90 [29.54, 67.24] | 23.60 [9.31, 48.61] |
| u_new | 10 s | -0.36 [-0.56, -0.19] | -0.33 [-0.59, -0.11] |
| u_new | 30 s | 0.08 [-0.70, 0.35] | 0.07 [-1.21, 0.91] |
| u_new | 60 s | 0.02 [-3.06, 5.24] | -1.49 [-4.55, 4.37] |
| u_new | 120 s | 36.70 [-10.77, 198.18] | 17.09 [-3.45, 93.17] |
| seen | 10 s | -0.48 [-0.68, -0.16] | -0.26 [-0.51, -0.15] |
| seen | 30 s | -0.40 [-1.02, 0.02] | -0.30 [-1.25, 0.10] |
| seen | 60 s | -1.15 [-3.61, 4.52] | -2.17 [-4.17, 0.19] |
| seen | 120 s | 20.03 [-20.46, 70.91] | -14.95 [-43.58, 0.38] |

## 3. Blow-up rate (HDV median |ds| > 50 m @60 s AND > 100 m @120 s)

| model | u_same | u_new | seen |
|---|---|---|---|
| run3 | 0.085 (11/130) | 0.158 (3/19) | 0.327 (50/153) |
| run3_s120 | 0.046 (6/130) | 0.053 (1/19) | 0.183 (28/153) |
| wms | 0.023 (3/130) | 0.053 (1/19) | 0.261 (40/153) |
| blackbox_traj | 0.031 (4/130) | 0.000 (0/19) | 0.327 (50/153) |
| blackbox_id | 0.123 (16/130) | 0.105 (2/19) | 0.438 (67/153) |

## 4. Predicted same-lane overlaps / gaps < 2 m (projection check)

Gap = s_lead − s_follow − length_lead (the projection's convention, 5 m
default length). Counts are summed over states at each horizon.

| model | set | horizon | overlap pairs (gap<0) | states with gap<2 m | min gap (m) |
|---|---|---|---|---|---|
| run3 | u_same | 10 s | 17 | 24/140 | -4.51 |
| run3 | u_same | 30 s | 23 | 36/140 | -4.94 |
| run3 | u_same | 60 s | 21 | 22/140 | -4.92 |
| run3 | u_same | 120 s | 6 | 13/140 | -4.72 |
| run3 | u_new | 10 s | 4 | 4/21 | -3.78 |
| run3 | u_new | 30 s | 5 | 4/21 | -1.58 |
| run3 | u_new | 60 s | 1 | 2/21 | -2.28 |
| run3 | u_new | 120 s | 2 | 2/21 | -3.45 |
| run3 | seen | 10 s | 18 | 30/157 | -4.97 |
| run3 | seen | 30 s | 40 | 44/157 | -4.56 |
| run3 | seen | 60 s | 32 | 40/157 | -4.79 |
| run3 | seen | 120 s | 25 | 32/157 | -4.69 |
| run3_s120 | u_same | 10 s | 20 | 29/140 | -4.52 |
| run3_s120 | u_same | 30 s | 29 | 40/140 | -4.92 |
| run3_s120 | u_same | 60 s | 26 | 32/140 | -4.95 |
| run3_s120 | u_same | 120 s | 19 | 26/140 | -4.70 |
| run3_s120 | u_new | 10 s | 5 | 6/21 | -3.73 |
| run3_s120 | u_new | 30 s | 4 | 5/21 | -4.92 |
| run3_s120 | u_new | 60 s | 8 | 6/21 | -5.00 |
| run3_s120 | u_new | 120 s | 5 | 4/21 | -3.45 |
| run3_s120 | seen | 10 s | 20 | 40/157 | -4.97 |
| run3_s120 | seen | 30 s | 55 | 59/157 | -4.92 |
| run3_s120 | seen | 60 s | 52 | 50/157 | -4.65 |
| run3_s120 | seen | 120 s | 25 | 36/157 | -4.71 |
| c1struct | u_same | 10 s | 39 | 54/140 | -4.92 |
| c1struct | u_same | 30 s | 71 | 67/140 | -4.99 |
| c1struct | u_same | 60 s | 82 | 56/140 | -4.97 |
| c1struct | u_same | 120 s | 134 | 85/140 | -4.99 |
| c1struct | u_new | 10 s | 6 | 11/21 | -4.81 |
| c1struct | u_new | 30 s | 12 | 10/21 | -4.54 |
| c1struct | u_new | 60 s | 8 | 10/21 | -4.40 |
| c1struct | u_new | 120 s | 30 | 15/21 | -4.59 |
| c1struct | seen | 10 s | 69 | 78/157 | -4.81 |
| c1struct | seen | 30 s | 206 | 106/157 | -4.90 |
| c1struct | seen | 60 s | 313 | 124/157 | -4.94 |
| c1struct | seen | 120 s | 340 | 122/157 | -5.00 |

## 5. Traffic-state split by t0 near-merge speed (`vnear`, m/s)

From `c1_congestion.json`; congested = `vnear` < 15 m/s.

| model | set | stratum | n | 10 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|---|---|---|
| run3 | u_same | congested | 74 | 1.85 | 5.91 | 17.80 | 87.77 |
| run3 | u_same | free | 66 | 1.68 | 6.76 | 16.08 | 180.71 |
| run3 | u_new | congested | 12 | 2.21 | 7.51 | 11.73 | 73.17 |
| run3 | u_new | free | 9 | 1.07 | 6.27 | 13.44 | 108.27 |
| run3 | seen | congested | 84 | 2.16 | 8.17 | 43.84 | 155.51 |
| run3 | seen | free | 73 | 1.43 | 7.05 | 20.68 | 147.51 |
| c1struct | u_same | congested | 74 | 2.37 | 5.81 | 14.15 | 46.13 |
| c1struct | u_same | free | 66 | 2.22 | 7.18 | 18.98 | 161.98 |
| c1struct | u_new | congested | 12 | 2.29 | 5.29 | 12.06 | 33.52 |
| c1struct | u_new | free | 9 | 1.84 | 6.90 | 17.13 | 117.96 |
| c1struct | seen | congested | 84 | 2.58 | 8.13 | 29.61 | 140.71 |
| c1struct | seen | free | 73 | 1.94 | 8.83 | 24.02 | 167.76 |

## 6. Traffic-level section metrics, bins 60-90 / 90-120 s

Median |predicted − recorded| across states (arm under test). Lower is
better. `delay` = normalized delay (s); `queue` = merge queue length (m).

| model | set | section | mean_speed | stopped | queue | delay |
|---|---|---|---|---|---|---|
| run3 | u_same | upstream | 0.893 | 0.033 | 0.000 | 34.44 |
| run3 | u_same | merge | 2.230 | 0.075 | 0.000 | 41.67 |
| run3 | u_same | downstream | 2.769 | 0.000 | 0.000 | 47.70 |
| run3 | u_new | upstream | 0.915 | 0.075 | 0.000 | 48.36 |
| run3 | u_new | merge | 1.538 | 0.042 | 0.000 | 31.77 |
| run3 | u_new | downstream | 2.093 | 0.000 | 0.000 | 33.58 |
| run3 | seen | upstream | 0.998 | 0.267 | 0.000 | 48.53 |
| run3 | seen | merge | 1.731 | 0.033 | 0.000 | 31.44 |
| run3 | seen | downstream | 2.406 | 0.000 | 0.000 | 36.46 |
| c1struct | u_same | upstream | 0.722 | 0.367 | 0.000 | 26.01 |
| c1struct | u_same | merge | 1.908 | 0.200 | 0.301 | 41.08 |
| c1struct | u_same | downstream | 7.272 | 0.950 | 0.000 | 97.29 |
| c1struct | u_new | upstream | 0.425 | 0.483 | 0.000 | 22.53 |
| c1struct | u_new | merge | 1.008 | 0.200 | 0.568 | 27.07 |
| c1struct | u_new | downstream | 7.125 | 1.042 | 0.000 | 84.23 |
| c1struct | seen | upstream | 1.022 | 0.917 | 0.000 | 45.42 |
| c1struct | seen | merge | 1.716 | 0.167 | 0.484 | 28.51 |
| c1struct | seen | downstream | 5.839 | 1.167 | 0.000 | 78.20 |

## 7. Entries ON (entries v2, observed counters at t0) — bounded diagnostic

The entries-v2 rollout pads a state to ~rate×horizon slots; on these
counter-bearing sets (~73–110 vehicles) that is ~6x the entries-OFF cost, so
the full 3-set × 2-model ON pass projects to ~10 h at the mandated ≤4 workers.
The numbers below compare the first 12 windows of each set (ON vs the
SAME first 12 OFF windows). Full-set ON is not claimed.

| set | horizon | run3 ON | run3 OFF (head) | stage120 ON | stage120 OFF (head) |
|---|---|---|---|---|---|
| u_same | 10 s | 1.65 | 1.65 | 2.13 | 2.13 |
| u_same | 30 s | 6.02 | 6.02 | 5.75 | 5.75 |
| u_same | 60 s | 18.17 | 16.24 | 18.59 | 16.69 |
| u_same | 120 s | 167.64 | 193.72 | 200.27 | 478.73 |
| u_new | 10 s | 1.93 | 1.93 | 1.70 | 1.70 |
| u_new | 30 s | 6.87 | 7.18 | 6.15 | 6.36 |
| u_new | 60 s | 12.78 | 11.73 | 14.87 | 14.36 |
| u_new | 120 s | 183.32 | 92.59 | 40.60 | 42.17 |
| seen | 10 s | 1.97 | 1.97 | 1.94 | 1.94 |
| seen | 30 s | 7.55 | 7.29 | 8.50 | 8.71 |
| seen | 60 s | 51.84 | 50.16 | 29.49 | 27.28 |
| seen | 120 s | 155.77 | 245.22 | 116.30 | 143.47 |

## 8. Caveats

- Equal-training comparison: run 3 used the same LPO TRAIN windows and the
  C1-matched stage steps; the C1 structured checkpoint is the frozen
  `WM-S-C1FAST-20261006` comparator (its per-state files were reused; our
  re-score of the same checkpoint reproduced its medians exactly).
- The 120 s HDV error is heavy-tailed (medians blow up), so the paired
  world bootstrap is the primary long-horizon statement, not the raw median.
- `u_new` has only 21 worlds; its intervals are wide.
- Entries-ON is a bounded (first 12 windows) diagnostic (§7), not a full-set
  result; the clean equal-training test is entries OFF.
- Nothing under E:/research-dense/runs/ or E:/rd-wt/ws-c1eval/ was modified;
  no CAL/HOLDOUT/sealed/chunk-5/J4B/INC1 data was opened.
