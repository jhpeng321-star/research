# Round 5 (WM-S-R5-20261012) vs C1 / R4 — scoring summary

Clean scorer `E:/agent-work/r5-score/src` (`wms.eval.score.score_states`), pairing world+key+source_arm+t0, CPU, teacher-forced CAVs, entries off. C1/R4 rows from the round-4 caches (real-history 120: CHK-FADE confirmation rows); baseline gate PASS (C1 8.71/99.84, R4 9.96/110.69); step0 C1 quick-20 60 s = 11.9576 m (pass).

## Accuracy (median HDV |ds|, m) and paired differences (95% CI over worlds)
| set | model | 10 s | 30 s | 60 s | 120 s | blow-ups |
|---|---|---|---|---|---|---|
| wall_same (67) | C1 | 2.04 | 5.07 | 8.71 | 99.84 | 0/66 |
|  | R4 | 2.17 | 4.88 | 9.96 | 110.69 | 0/66 |
|  | R5-last | 2.29 | 8.43 | 18.42 | 160.88 | 7/66 |
| | R5-last − C1 | 0.32 [-0.13, 1.04] | 2.63 [0.64, 4.71] | 5.18 [3.29, 11.94] | 71.67 [44.98, 148.47] | |
| | R5-last − R4 | 0.22 [-0.03, 0.81] | 2.40 [0.98, 4.86] | 6.27 [3.37, 12.71] | 82.73 [11.13, 125.18] | |
| wall_pilot (53) | C1 | 2.06 | 5.06 | 8.34 | 96.76 | 0/52 |
|  | R4 | 2.18 | 4.76 | 9.41 | 107.32 | 0/52 |
|  | R5-last | 2.29 | 8.15 | 15.93 | 146.68 | 3/52 |
| | R5-last − C1 | 0.23 [-0.22, 1.31] | 1.88 [0.61, 4.29] | 5.18 [3.29, 12.75] | 50.85 [11.75, 95.13] | |
| | R5-last − R4 | 0.25 [-0.07, 0.83] | 2.13 [0.72, 3.91] | 6.51 [2.60, 12.62] | 47.50 [11.13, 108.02] | |
| real-history (120) | C1 | 2.04 | 5.62 | 13.04 | 87.78 | 0/114 |
|  | R4 | 2.03 | 5.62 | 13.81 | 61.51 | 6/114 |
|  | R5-last | 2.44 | 8.50 | 24.06 | 166.97 | 16/114 |
| | R5-last − C1 | 0.33 [0.10, 0.53] | 2.50 [0.74, 3.14] | 9.98 [2.68, 14.83] | 71.71 [51.17, 102.97] | |
| | R5-last − R4 | 0.27 [0.08, 0.58] | 2.29 [1.18, 3.60] | 6.26 [3.31, 11.57] | 67.01 [54.46, 91.02] | |

## Fingerprints (wall_same 67)
| | C1 | R4 | R5-last | recorded |
|---|---|---|---|---|
| gate g(k) 10/20/60/120/240 | 0.155/0.195/0.287/0.428/0.836 | 0.088/0.160/0.335/0.433/0.470 | 0.093/0.169/0.356/0.459/0.498 |  |
| learned g_max / tau (accel gate) | (cumsum) | 0.473 / 48.6 | 0.501 / 48.5 |  |
| lane gate g(k) 10/20/60/120/240 | 0.155/0.195/0.287/0.428/0.836 | 0.088/0.160/0.335/0.433/0.470 | 0.054/0.092/0.163/0.184/0.187 |  |
| learned lane g_max / tau | (shared) | (shared) | 0.187 / 29.7 |  |
| mean (p95) |g*da| 0-60 s, m/s^2 | 1.68 (3.37) | 0.96 (4.97) | 1.51 (8.42) |  |
| mean (p95) |g*da| 60-120 s, m/s^2 | 17.97 (118.57) | 2.51 (10.63) | 4.49 (18.28) |  |
| mean signed speed error 0-60 s, m/s | -0.34 | -0.32 | -0.69 |  |
| lane changes /HDV/10 s, 0-60 s (TRUE def.) | 0.159 | 0.260 | 0.068 | 0.024 |
| lane changes /HDV/10 s, 60-120 s | 4.498 | 1.560 | 0.318 | 0.037 |
| wrong-lane share 60 / 120 s | 0.138 / 0.353 | 0.087 / 0.200 | 0.121 / 0.399 |  |
| hard-brake recall / false brakes per veh-min | 0.849 / 2.71 | 0.762 / 2.61 | 0.804 / 4.43 |  |

Provenance: C1/R4 rollout fingerprints copied from `WM-S-R4-20261011/score/results.json` (identical models/sets); gate curves recomputed. R5 fingerprints computed here.
