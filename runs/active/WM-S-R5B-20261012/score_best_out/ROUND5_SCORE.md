# Round 5 (WM-S-R5-20261012) vs C1 / R4 — scoring summary

Clean scorer `E:/agent-work/r5-score/src` (`wms.eval.score.score_states`), pairing world+key+source_arm+t0, CPU, teacher-forced CAVs, entries off. C1/R4 rows from the round-4 caches (real-history 120: CHK-FADE confirmation rows); baseline gate PASS (C1 8.71/99.84, R4 9.96/110.69); step0 C1 quick-20 60 s = 11.9576 m (pass).

## Accuracy (median HDV |ds|, m) and paired differences (95% CI over worlds)
| set | model | 10 s | 30 s | 60 s | 120 s | blow-ups |
|---|---|---|---|---|---|---|
| wall_same (67) | C1 | 2.04 | 5.07 | 8.71 | 99.84 | 0/66 |
|  | R4 | 2.17 | 4.88 | 9.96 | 110.69 | 0/66 |
|  | R5-last | 1.72 | 4.47 | 12.48 | 116.86 | 1/66 |
| | R5-last − C1 | -0.37 [-0.60, -0.07] | -0.45 [-1.91, 0.69] | 1.19 [-1.52, 4.11] | 4.99 [-18.36, 34.06] | |
| | R5-last − R4 | -0.51 [-0.55, -0.28] | -0.38 [-1.15, 0.25] | 0.91 [-0.57, 3.49] | 6.75 [-38.54, 21.81] | |
| wall_pilot (53) | C1 | 2.06 | 5.06 | 8.34 | 96.76 | 0/52 |
|  | R4 | 2.18 | 4.76 | 9.41 | 107.32 | 0/52 |
|  | R5-last | 1.72 | 3.47 | 9.08 | 112.63 | 1/52 |
| | R5-last − C1 | -0.29 [-0.58, -0.06] | -0.66 [-1.91, -0.16] | 0.44 [-2.00, 2.58] | 0.37 [-36.03, 28.68] | |
| | R5-last − R4 | -0.46 [-0.55, -0.26] | -0.78 [-1.23, 0.11] | 0.91 [-0.38, 2.57] | 9.98 [-21.87, 22.47] | |
| real-history (120) | C1 | 2.04 | 5.62 | 13.04 | 87.78 | 0/114 |
|  | R4 | 2.03 | 5.62 | 13.81 | 61.51 | 6/114 |
|  | R5-last | 1.63 | 4.49 | 10.86 | 83.95 | 9/114 |
| | R5-last − C1 | -0.35 [-0.54, -0.22] | -0.93 [-2.12, -0.62] | -0.57 [-2.12, 0.43] | -0.43 [-10.25, 19.10] | |
| | R5-last − R4 | -0.35 [-0.58, -0.18] | -1.16 [-1.73, -0.69] | -1.51 [-2.94, 0.42] | -0.90 [-15.48, 6.22] | |

## Fingerprints (wall_same 67)
| | C1 | R4 | R5-last | recorded |
|---|---|---|---|---|
| gate g(k) 10/20/60/120/240 | 0.155/0.195/0.287/0.428/0.836 | 0.088/0.160/0.335/0.433/0.470 | 0.094/0.170/0.357/0.461/0.499 |  |
| learned g_max / tau (accel gate) | (cumsum) | 0.473 / 48.6 | 0.503 / 48.4 |  |
| lane gate g(k) 10/20/60/120/240 | 0.155/0.195/0.287/0.428/0.836 | 0.088/0.160/0.335/0.433/0.470 | 0.055/0.094/0.166/0.187/0.191 |  |
| learned lane g_max / tau | (shared) | (shared) | 0.191 / 29.7 |  |
| mean (p95) |g*da| 0-60 s, m/s^2 | 1.68 (3.37) | 0.96 (4.97) | 0.91 (4.42) |  |
| mean (p95) |g*da| 60-120 s, m/s^2 | 17.97 (118.57) | 2.51 (10.63) | 2.66 (12.68) |  |
| mean signed speed error 0-60 s, m/s | -0.34 | -0.32 | -0.45 |  |
| lane changes /HDV/10 s, 0-60 s (TRUE def.) | 0.159 | 0.260 | 0.036 | 0.024 |
| lane changes /HDV/10 s, 60-120 s | 4.498 | 1.560 | 0.152 | 0.037 |
| wrong-lane share 60 / 120 s | 0.138 / 0.353 | 0.087 / 0.200 | 0.081 / 0.288 |  |
| hard-brake recall / false brakes per veh-min | 0.849 / 2.71 | 0.762 / 2.61 | 0.726 / 1.91 |  |

Provenance: C1/R4 rollout fingerprints copied from `WM-S-R4-20261011/score/results.json` (identical models/sets); gate curves recomputed. R5 fingerprints computed here.
