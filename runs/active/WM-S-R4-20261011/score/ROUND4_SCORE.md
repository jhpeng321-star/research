# Round 4 (WM-S-R4-20261011) vs C1 — scoring summary

Written by the coordinator from `results.json` (the OpenCode scoring task, openai/gpt-6-luna, computed everything but hit
its 240-min runtime limit before writing this report). Clean scorer `src/wms` (`wms.eval.score.score_states`), pairing key
world+key+source_arm+t0, CPU, teacher-forced CAVs, entries off. Step 0: C1 60 s quick-20 = 11.9576 m (pass).
R4-best is identical to R4-last on every metric (the long-horizon selection kept the final checkpoint).

## Accuracy (median HDV |ds|, m) and paired R4 - C1 (95% CI over worlds)
| set | model | 10 s | 30 s | 60 s | 120 s | blow-ups |
|---|---|---|---|---|---|---|
| wall_same (67) | C1 | 2.04 | 5.07 | 8.71 | 99.8 | 0/66 |
| | R4 | 2.17 | 4.88 | 9.96 | 110.7 | 0/66 |
| | paired | +0.09 [-0.02, 0.19] | +0.12 [-0.86, 0.42] | -0.07 [-1.57, 0.34] | +6.6 [-11.1, 47.5] | |
| wall_pilot (53) | C1 | 2.06 | 5.06 | 8.34 | 96.8 | 0/52 |
| | R4 | 2.18 | 4.76 | 9.41 | 107.3 | 0/52 |
| | paired | +0.09 [-0.08, 0.23] | +0.12 [-0.39, 0.35] | -0.22 [-1.57, 1.22] | -2.8 [-18.2, 18.5] | |
| real 60 s history (quick 20) | C1 | 2.10 | 8.59 | 16.80 | 66.4 | 0/19 |
| | R4 | 2.50 | 8.96 | 27.73 | 78.5 | 3/19 |
| | paired | +0.15 [-0.20, 0.41] | +0.05 [-1.07, 1.12] | +2.67 [0.07, 13.1] | -5.9 [-23.2, 165.9] | |
For reference R3 (full wall_same): 60 s 13.39, 120 s 131.9, blow-ups 2/66; real-history 60 s 28.3, 3/19.

## Fingerprints (wall_same 67)
| | C1 | R4 | recorded |
|---|---|---|---|
| gate g(k) at k=10/20/60/120/240 | 0.155/0.195/0.287/0.428/0.836 | 0.088/0.160/0.335/0.433/0.470 | |
| learned g_max / tau | (cumsum gate) | 0.473 / 48.6 steps (init 0.30 / 50) | |
| mean (p95) |g*da| 0-60 s, m/s^2 | 1.68 (3.37) | 0.96 (4.97) | |
| mean (p95) |g*da| 60-120 s, m/s^2 | 17.97 (118.6) | 2.51 (10.6) | |
| mean signed speed error 0-60 s, m/s | -0.34 | -0.32 (R3 -1.08) | |
| lane changes /HDV/10 s, 0-60 s (TRUE def.) | 0.159 | 0.260 | 0.024 |
| lane changes /HDV/10 s, 60-120 s | 4.50 | 1.56 | 0.037 |
| wrong-lane share 60 / 120 s | 13.8% / 35.3% | 8.7% / 20.0% | |
| hard-brake recall / false brakes per veh-min | 84.9% / 2.71 | 76.2% / 2.61 | |

## Reading (measured vs inference)
- Measured: on both unseen-wall sets R4 is level with C1 at every horizon (all paired CIs include 0), with 0 blow-ups; it is
  clearly better than R3 (60 s 9.96 vs 13.39 m, 120 s 110.7 vs 131.9 m, blow-ups 0 vs 2).
- Measured, intended effects: the gate saturated (g_max learned 0.47, g(240) 0.47 vs C1 0.84); late correction collapsed
  (60-120 s mean |g*da| 2.5 vs 18.0, p95 10.6 vs 118.6); speed bias back to C1 level (-0.32 vs R3 -1.08); late lane flips
  down 3x and wrong-lane share down (20% vs 35% at 120 s).
- Measured, not fixed: early lane changes are HIGHER than C1 (0.26 vs 0.16 per HDV per 10 s; 11x recorded) despite the
  corrected lane-rate target; real-60 s-history states remain bad (60 s 27.7 vs 16.8 m, 3/19 blow-ups) exactly as in R3.
- Inference: removing the late-correction explosion and the speed bias did not move the 120 s error (~100 m for both), so
  another error source dominates long horizons (candidates: unmodelled vehicle entries, 70-110 entrants per state by 120 s;
  the remaining lane over-prediction; the real-history failure mode shared by R3/R4 but not C1).
