# Round 5 (WM-S-R5-20261012) vs C1 / R4 — scoring summary

Clean scorer `E:/agent-work/r5-score/src` (`wms.eval.score.score_states`), pairing world+key+source_arm+t0, CPU, teacher-forced CAVs, entries off. C1/R4 rows from the round-4 caches (real-history 120: CHK-FADE confirmation rows); baseline gate PASS (C1 8.71/99.84, R4 9.96/110.69); step0 C1 quick-20 60 s = 11.9576 m (pass).

## Accuracy (median HDV |ds|, m) and paired differences (95% CI over worlds)
| set | model | 10 s | 30 s | 60 s | 120 s | blow-ups |
|---|---|---|---|---|---|---|
| wall_same (67) | C1 | 2.04 | 5.07 | 8.71 | 99.84 | 0/66 |
|  | R4 | 2.17 | 4.88 | 9.96 | 110.69 | 0/66 |
|  | R5-last | 1.49 | 4.35 | 18.61 | 198.14 | 2/66 |
| | R5-last − C1 | -0.29 [-0.47, 0.01] | -0.53 [-1.05, 0.05] | 7.48 [5.25, 10.44] | 95.02 [62.62, 203.79] | |
| | R5-last − R4 | -0.41 [-0.69, -0.16] | -0.69 [-1.27, -0.34] | 8.14 [5.02, 11.32] | 110.76 [58.05, 228.45] | |
| wall_pilot (53) | C1 | 2.06 | 5.06 | 8.34 | 96.76 | 0/52 |
|  | R4 | 2.18 | 4.76 | 9.41 | 107.32 | 0/52 |
|  | R5-last | 1.49 | 4.26 | 17.00 | 204.02 | 0/52 |
| | R5-last − C1 | -0.29 [-0.46, -0.08] | -0.56 [-1.93, -0.20] | 6.82 [5.19, 8.72] | 83.64 [27.55, 201.62] | |
| | R5-last − R4 | -0.41 [-0.73, -0.01] | -0.89 [-1.41, -0.40] | 7.35 [4.46, 9.54] | 110.76 [54.41, 212.01] | |
