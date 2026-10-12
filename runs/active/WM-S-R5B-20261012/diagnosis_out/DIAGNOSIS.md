# Round 5 diagnosis (WM-S-R5-20261012) — pre-registered measurements M1-M5'

Plan: `WM-S-R5-PREP-20261012/DIAGNOSIS_PLAN.md`. Round-5 code tree `E:/agent-work/r5-score/src`; comparators C1/R4 same code paths; true driver types evaluation-only (`WM-S-R4-CHK-TYPEID-20261011/out/truth_full.json`).

## M1 — |ds| split wrong-lane vs right-lane (wall_same 67)
### wall_same
| model/variant | h | n | wrong share | wrong-lane med (p90) | right-lane med (p90) |
|---|---|---|---|---|---|
| R5/base | 10 | 4532 | 0.019 | 8.68 (77.05) | 1.63 (12.93) |
| R5/base | 30 | 3801 | 0.049 | 40.79 (378.21) | 4.34 (41.77) |
| R5/base | 60 | 2735 | 0.081 | 44.23 (418.10) | 9.81 (105.25) |
| R5/base | 120 | 858 | 0.288 | 114.17 (354.62) | 109.34 (396.40) |
| R4/base | 10 | 405 | 0.022 | 2.32 (30.82) | 2.11 (10.51) |
| R4/base | 30 | 332 | 0.069 | 14.21 (178.91) | 7.46 (44.15) |
| R4/base | 60 | 226 | 0.199 | 46.02 (177.34) | 17.45 (66.39) |
| R4/base | 120 | 68 | 0.191 | 169.18 (967.68) | 83.68 (371.73) |
| C1/base | 10 | 405 | 0.020 | 3.92 (111.35) | 2.03 (9.81) |
| C1/base | 30 | 332 | 0.075 | 6.07 (176.33) | 7.40 (59.87) |
| C1/base | 60 | 226 | 0.181 | 39.55 (193.49) | 16.96 (70.74) |
| C1/base | 120 | 68 | 0.324 | 209.73 (503.14) | 124.87 (277.08) |
| R5/diag_lane_gate_shared | 10 | 4532 | 0.019 | 8.68 (76.49) | 1.64 (12.97) |
| R5/diag_lane_gate_shared | 30 | 3801 | 0.051 | 36.45 (376.50) | 4.39 (41.77) |
| R5/diag_lane_gate_shared | 60 | 2735 | 0.095 | 37.02 (309.96) | 10.45 (112.89) |
| R5/diag_lane_gate_shared | 120 | 858 | 0.314 | 114.53 (444.63) | 125.22 (446.39) |
| R5/diag_lane_cap_off | 10 | 4532 | 0.038 | 7.89 (49.23) | 1.67 (12.86) |
| R5/diag_lane_cap_off | 30 | 3801 | 0.134 | 25.75 (178.74) | 6.41 (72.25) |
| R5/diag_lane_cap_off | 60 | 2735 | 0.258 | 72.71 (279.10) | 36.21 (234.77) |
| R5/diag_lane_cap_off | 120 | 858 | 0.316 | 186.80 (631.90) | 202.71 (699.81) |
| R5/diag_encoder_population | 10 | 4532 | 0.066 | 7.81 (36.54) | 2.17 (15.08) |
| R5/diag_encoder_population | 30 | 3801 | 0.141 | 46.36 (320.65) | 6.58 (49.47) |
| R5/diag_encoder_population | 60 | 2735 | 0.224 | 79.38 (658.55) | 16.76 (148.39) |
| R5/diag_encoder_population | 120 | 858 | 0.498 | 195.19 (805.05) | 198.17 (868.96) |
| R5/diag_recorded_lanes | 10 | 4532 | 0.000 | n/a (n/a) | 1.62 (12.55) |
| R5/diag_recorded_lanes | 30 | 3801 | 0.000 | n/a (n/a) | 3.94 (38.53) |
| R5/diag_recorded_lanes | 60 | 2735 | 0.000 | n/a (n/a) | 9.30 (80.35) |
| R5/diag_recorded_lanes | 120 | 858 | 0.000 | n/a (n/a) | 60.50 (257.84) |
### wall_same_real
| model/variant | h | n | wrong share | wrong-lane med (p90) | right-lane med (p90) |
|---|---|---|---|---|---|
| R5/base | 10 | 1582 | 0.035 | 7.47 (38.96) | 1.96 (17.28) |
| R5/base | 30 | 1343 | 0.094 | 54.09 (218.60) | 6.03 (59.31) |
| R5/base | 60 | 991 | 0.090 | 77.56 (255.87) | 16.62 (145.24) |
| R5/base | 120 | 397 | 0.146 | 168.62 (591.06) | 47.15 (607.99) |
| R4/base | 10 | 430 | 0.047 | 7.50 (34.49) | 2.59 (17.08) |
| R4/base | 30 | 365 | 0.112 | 70.83 (344.91) | 8.11 (66.18) |
| R4/base | 60 | 277 | 0.108 | 103.60 (231.83) | 26.85 (121.53) |
| R4/base | 120 | 100 | 0.090 | 400.76 (514.14) | 55.71 (677.08) |
| C1/base | 10 | 430 | 0.051 | 4.25 (30.40) | 2.90 (17.55) |
| C1/base | 30 | 365 | 0.101 | 65.37 (274.07) | 7.19 (55.31) |
| C1/base | 60 | 277 | 0.224 | 32.63 (142.03) | 15.94 (83.09) |
| C1/base | 120 | 100 | 0.450 | 166.65 (457.25) | 152.42 (462.67) |
| R5/diag_encoder_population | 10 | 1582 | 0.085 | 6.69 (40.76) | 2.34 (18.41) |
| R5/diag_encoder_population | 30 | 1343 | 0.163 | 43.03 (212.85) | 8.74 (83.81) |
| R5/diag_encoder_population | 60 | 991 | 0.209 | 61.12 (496.56) | 29.76 (197.59) |
| R5/diag_encoder_population | 120 | 397 | 0.380 | 227.60 (707.87) | 74.32 (615.90) |
| R5/diag_recorded_lanes | 10 | 1582 | 0.000 | n/a (n/a) | 1.91 (15.73) |
| R5/diag_recorded_lanes | 30 | 1343 | 0.000 | n/a (n/a) | 5.84 (57.51) |
| R5/diag_recorded_lanes | 60 | 991 | 0.000 | n/a (n/a) | 14.96 (129.63) |
| R5/diag_recorded_lanes | 120 | 397 | 0.000 | n/a (n/a) | 32.56 (634.56) |

## M2 — survivor speed drift (t0-present HDVs alive at 120 s) + delay at 120 s
| model/variant | set | n surv | dv60 signed (jam) | dv90 | dv120 | |dv|120 med | delay120 pred/rec (veh-s) |
|---|---|---|---|---|---|---|---|
| R5/base | wall_same | 873 | 0.50 (0.24) | 1.52 | -4.12 | 5.23 | 154156 / 144683 |
| R4/base | wall_same | 69 | -0.38 (n/a) | 3.53 | -5.87 | 2.98 | 10794 / 9754 |
| C1/base | wall_same | 69 | -0.52 (n/a) | -0.66 | -11.34 | 15.64 | 11803 / 9754 |
| R5/diag_lane_gate_shared | wall_same | 873 | 0.43 (0.34) | 2.05 | -3.81 | 4.85 | 153279 / 144683 |
| R5/diag_lane_cap_off | wall_same | 873 | -0.74 (0.82) | 1.28 | -7.65 | 8.48 | 170515 / 144683 |
| R5/diag_encoder_population | wall_same | 873 | 2.10 (5.09) | 5.17 | -2.31 | 7.15 | 142927 / 144683 |
| R5/diag_recorded_lanes | wall_same | 873 | -0.14 (0.25) | -0.30 | -4.61 | 2.42 | 155630 / 144683 |
| R5/base | wall_same_real | 405 | -1.04 (-1.23) | -0.38 | -3.16 | 2.47 | 61108 / 55599 |
| R4/base | wall_same_real | 104 | -1.13 (-0.47) | 0.40 | -2.05 | 3.85 | 14911 / 13719 |
| C1/base | wall_same_real | 104 | 0.46 (-0.34) | 1.56 | -3.78 | 5.99 | 14546 / 13719 |
| R5/diag_encoder_population | wall_same_real | 405 | -0.74 (-0.59) | 0.56 | -2.87 | 5.12 | 59599 / 55599 |
| R5/diag_recorded_lanes | wall_same_real | 405 | -1.23 (-1.34) | -0.66 | -3.19 | 1.97 | 60671 / 55599 |

## M3 — TRUE lane-change rates (changes per HDV-step; per HDV per 10 s in parens)
| model/variant | set | 0-10 s | 10-60 s | 60-120 s | recorded 0-120 |
|---|---|---|---|---|---|
| R5/base | wall_same | 0.00063 (0.013) | 0.00212 (0.042) | 0.00760 (0.152) | 0.00184 |
| R4/base | wall_same | 0.00202 (0.040) | 0.02499 (0.500) | 0.10745 (2.149) | 0.00158 |
| C1/base | wall_same | 0.00071 (0.014) | 0.00876 (0.175) | 0.13366 (2.673) | 0.00158 |
| R5/diag_lane_gate_shared | wall_same | 0.00071 (0.014) | 0.00299 (0.060) | 0.01351 (0.270) | 0.00184 |
| R5/diag_lane_cap_off | wall_same | 0.03554 (0.711) | 0.26060 (5.212) | 0.46650 (9.330) | 0.00184 |
| R5/diag_encoder_population | wall_same | 0.00396 (0.079) | 0.01342 (0.268) | 0.03500 (0.700) | 0.00184 |
| R5/diag_recorded_lanes | wall_same | 0.00105 (0.021) | 0.00124 (0.025) | 0.00184 (0.037) | 0.00184 |
| R5/base | wall_same_real | 0.00093 (0.019) | 0.00339 (0.068) | 0.00418 (0.084) | 0.00086 |
| R4/base | wall_same_real | 0.00400 (0.080) | 0.03255 (0.651) | 0.03630 (0.726) | 0.00148 |
| C1/base | wall_same_real | 0.00133 (0.027) | 0.01657 (0.331) | 0.32659 (6.532) | 0.00148 |
| R5/diag_encoder_population | wall_same_real | 0.00530 (0.106) | 0.01721 (0.344) | 0.02930 (0.586) | 0.00086 |
| R5/diag_recorded_lanes | wall_same_real | 0.00187 (0.037) | 0.00178 (0.036) | 0.00086 (0.017) | 0.00086 |

Training flip-rate allowance (TRUE rate) = 0.00124327163 per HDV-step.

## M4 — inference-time switch-offs (R5, wall_same 67)
| variant | 60 s ds med | 120 s ds med | wrong-lane share 120 | lane rate 60-120/step | dv120 signed |
|---|---|---|---|---|---|
| R5/base | 12.48 | 116.86 | 0.288 | 0.00760 | -4.12 |
| R5/diag_lane_gate_shared | 13.12 | 126.49 | 0.314 | 0.01351 | -3.81 |
| R5/diag_lane_cap_off | 42.67 | 158.50 | 0.316 | 0.46650 | -7.65 |
| R5/diag_encoder_population | 27.11 | 156.30 | 0.498 | 0.03500 | -2.31 |
| R5/diag_recorded_lanes | 9.58 | 72.43 | 0.000 | 0.00184 | -4.61 |
- R5/diag_lane_gate_shared: 60 s delta +0.63 m, 120 s delta +9.63 m (+8.2% vs R5 base)
- R5/diag_lane_cap_off: 60 s delta +30.18 m, 120 s delta +41.64 m (+35.6% vs R5 base)
- R5/diag_encoder_population: 60 s delta +14.62 m, 120 s delta +39.43 m (+33.7% vs R5 base)
- R5/diag_recorded_lanes: 60 s delta -2.91 m, 120 s delta -44.43 m (-38.0% vs R5 base)

## M5' — encoder's learned per-car settings vs TRUE driver types (evaluation-only)
| parameter | aggressive_car (n=156) | cautious_car (n=367) | normal_car (n=1206) |
|---|---|---|---|
| v0 | 30.73 [30.47, 30.88] | 30.82 [30.62, 30.96] | 30.74 [30.43, 30.91] |
| time_headway | 1.09 [1.07, 1.12] | 1.14 [1.12, 1.16] | 1.11 [1.09, 1.13] |
| min_gap | 2.18 [2.05, 2.35] | 2.45 [2.25, 2.76] | 2.25 [2.10, 2.42] |
| accel | 2.61 [2.46, 2.65] | 2.46 [2.31, 2.60] | 2.57 [2.46, 2.63] |
| decel | 3.40 [3.05, 3.86] | 3.30 [3.14, 3.64] | 3.35 [3.09, 3.83] |
| politeness | 0.50 [0.47, 0.51] | 0.50 [0.47, 0.50] | 0.50 [0.47, 0.51] |
| mobil_threshold | 0.26 [0.24, 0.27] | 0.27 [0.25, 0.27] | 0.27 [0.25, 0.27] |
| merge_gap | 6.47 [6.09, 6.75] | 6.83 [6.61, 7.19] | 6.52 [6.31, 6.81] |
| merge_urgency | 0.77 [0.73, 0.83] | 0.73 [0.70, 0.78] | 0.75 [0.72, 0.81] |

Separability (nearest centroid, half/half split): test accuracy **0.436** vs majority 0.698 (fit 808/test 921).

Correlations with true parameters: time_headway_vs_tau: r=0.30 (rho=0.33, n=1729); min_gap_vs_minGap: r=0.31 (rho=0.24, n=1729); accel_vs_accel: r=0.18 (rho=0.14, n=1729); v0_vs_speedFactor: r=-0.09 (rho=-0.10, n=1729)

## Pre-registered decision rules (auto-evaluated)
```
{
 "R1_lane_vs_encoder": {
  "recorded_lanes_cut_120s_pct": -38.01815613590067,
  "encoder_population_change_120s_pct": 33.74445380063753,
  "verdict": "mixed / inconclusive"
 },
 "R3_lane_rate": {
  "pred_over_recorded": 1.7092511013215859,
  "verdict": "middle ground"
 },
 "R2_jam_discharge": {
  "jam_signed_dv_60_90_120": [
   0.2383092533458363,
   0.5208408445506901,
   -5.924906569641906
  ],
  "free_signed_dv_60_90_120": [
   0.5203904197855511,
   1.6190755137396817,
   -3.949724590883183
  ],
  "monotone_growth_in_abs": true,
  "verdict": "keep jam-discharge hypothesis"
 }
}
```
