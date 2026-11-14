# REPORT_V04A: Part B value model v0.4a on TRUE policy references (no walls)

Registration: `output/wd01/REGISTRATION_PARTB_V04A.md` (LF sha256 `38a6fa406ca9d598fdaae986263f4fa81ee15dfa8c632414cc5f28dbe5bdee00`). Acceptance rule fixed BEFORE running; ONE CAL read; no refit afterwards.

Ideal-sensing premise: lane bins come from the TRUE state; the reconstruction gap is measured separately (Part A). Inputs carry no penetration, pre_fix, speed_mode or world strings; t0 is unrecoverable (ctx col 1 = 0). Wall tokens are empty (v0.4b adds walls).

## Verdict table

| variant | VERDICT | A1 | A2 gating (30/60/120) | A3 damping-native | A3 pi-native |
|---|---|---|---|---|---|
| deployable_strict (HEADLINE) | FAIL | fail | PASS | fail | fail |
| deployable_detector (SENSITIVITY) | FAIL | PASS | PASS | fail | fail |
| deployable_strict WITHOUT auxiliary chunks 1-3 (ABLATION) | FAIL | fail | PASS | fail | fail |

## deployable_strict (HEADLINE)

VERDICT **FAIL**

### A1

| tau | n | CRPS model | CRPS climatology | beats clim | MAE(median) | 10-90 coverage |
|---|---|---|---|---|---|---|
| 30 | 7542 | 169.9 | 450.4 | yes | 244.7 | 0.892 |
| 60 | 7173 | 351.2 | 879.5 | yes | 501.3 | 0.885 |
| 120 | 6435 | 734.7 | 1678.2 | yes | 1039.2 | 0.881 |
| 300 | 4221 | 2109.2 | 3620.9 | yes | 3073.5 | 0.837 |
| 570 | 900 | 4195.5 | 5738.4 | yes | 6206.3 | 0.930 |

Overall 10-90 coverage 0.880 (band 0.75-0.85: OUT); A1 = fail

### A2

| tau | n | MAE model | MAE ridge | CRPS model | CRPS ridge | verdict | gating |
|---|---|---|---|---|---|---|---|
| 30 | 7542 | 244.7 | 286.9 | 169.9 | 210.7 | PASS | gating |
| 60 | 7173 | 501.3 | 584.2 | 351.2 | 428.2 | PASS | gating |
| 120 | 6435 | 1039.2 | 1192.5 | 734.7 | 867.3 | PASS | gating |
| 300 | 4221 | 3073.5 | 3285.0 | 2109.2 | 2334.8 | PASS | reported |
| 570 | 900 | 6206.3 | 6044.4 | 4195.5 | 4235.3 | FAIL | reported |

A2-gating (tau 30, 60, 120 all PASS): PASS

### A3 (all contrasts)

| contrast | tau | pairs (worlds) | eligible |dY|>med | sign acc [90% CI] | (i) | CRPS dV | CRPS zero | CRPS cell-mean | (ii) | CRPS dV comonotone | tau pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| damping-native | 30 | 23 (23) | 11 (thr 27.2) | 0.909 [0.727, 1.000] | PASS | 102.6 | 41.2 | 21.9 | fail | 70.3 | fail |
| damping-native | 60 | 23 (23) | 14 (thr 75.2) | 0.929 [0.786, 1.000] | PASS | 210.3 | 127.9 | 74.3 | fail | 124.7 | fail |
| damping-native | 120 | 23 (23) | 15 (thr 205.3) | 0.867 [0.733, 1.000] | PASS | 467.2 | 439.9 | 278.1 | fail | 247.5 | fail |
| damping-native | 300 | 23 (23) | 14 (thr 874.6) | 1.000 [1.000, 1.000] | PASS | 1378.6 | 1885.7 | 1013.8 | fail | 840.3 | fail |
| damping-native | 570 | 23 (23) | 14 (thr 3245.3) | 0.929 [0.786, 1.000] | PASS | 3626.5 | 5584.2 | 3303.9 | fail | 2830.5 | fail |
| fixed-native | 30 | 18 (9) | 6 (thr 10.8) | 1.000 [1.000, 1.000] | PASS | 106.7 | 14.0 | 9.8 | fail | 60.2 | fail |
| fixed-native | 60 | 18 (9) | 9 (thr 41.2) | 1.000 [1.000, 1.000] | PASS | 217.8 | 55.0 | 45.5 | fail | 98.2 | fail |
| fixed-native | 120 | 18 (9) | 11 (thr 124.5) | 1.000 [1.000, 1.000] | PASS | 475.3 | 186.5 | 148.1 | fail | 223.0 | fail |
| fixed-native | 300 | 18 (9) | 9 (thr 648.0) | 0.889 [0.667, 1.000] | PASS | 1411.6 | 789.2 | 703.2 | fail | 703.0 | fail |
| fixed-native | 570 | 18 (9) | 9 (thr 2568.4) | 0.778 [0.556, 1.000] | PASS | 3209.0 | 2308.1 | 2149.2 | fail | 2062.4 | fail |
| pi-native | 30 | 23 (23) | 7 (thr 7.5) | 1.000 [1.000, 1.000] | PASS | 106.2 | 9.2 | 13.3 | fail | 79.6 | fail |
| pi-native | 60 | 23 (23) | 10 (thr 30.6) | 1.000 [1.000, 1.000] | PASS | 213.9 | 47.7 | 50.8 | fail | 137.3 | fail |
| pi-native | 120 | 23 (23) | 11 (thr 123.8) | 1.000 [1.000, 1.000] | PASS | 464.9 | 205.1 | 229.3 | fail | 237.5 | fail |
| pi-native | 300 | 23 (23) | 11 (thr 731.3) | 0.909 [0.727, 1.000] | PASS | 1379.6 | 995.6 | 920.2 | fail | 773.3 | fail |
| pi-native | 570 | 23 (23) | 10 (thr 2162.1) | 0.900 [0.700, 1.000] | PASS | 3367.7 | 2576.8 | 2656.1 | fail | 2231.3 | fail |
| switching-native | 30 | 27 (9) | 10 (thr 8.8) | 1.000 [1.000, 1.000] | PASS | 104.5 | 12.0 | 8.1 | fail | 48.1 | fail |
| switching-native | 60 | 27 (9) | 15 (thr 27.3) | 0.933 [0.812, 1.000] | PASS | 215.5 | 43.1 | 30.3 | fail | 85.7 | fail |
| switching-native | 120 | 27 (9) | 13 (thr 89.4) | 1.000 [1.000, 1.000] | PASS | 471.9 | 161.0 | 113.1 | fail | 180.6 | fail |
| switching-native | 300 | 27 (9) | 12 (thr 747.0) | 1.000 [1.000, 1.000] | PASS | 1422.5 | 937.6 | 680.5 | fail | 626.2 | fail |
| switching-native | 570 | 27 (9) | 14 (thr 2217.8) | 0.857 [0.667, 1.000] | PASS | 3309.0 | 3114.1 | 2266.0 | fail | 2323.6 | fail |

- damping-native: A3 (i AND ii at >= 2 of 3 taus in 60/120/300) = fail (0/3 taus)
- pi-native: A3 (i AND ii at >= 2 of 3 taus in 60/120/300) = fail (0/3 taus)

Per-policy coverage:

| policy | n | coverage |
|---|---|---|
| damping_fleet | 6571 | 0.878 |
| fixed_mix | 3546 | 0.857 |
| native_fleet | 6304 | 0.910 |
| pi_fleet | 4531 | 0.878 |
| switching_mix | 5319 | 0.863 |

TRAIN-OOB verdict (labelled, not out-of-sample): FAIL

## deployable_detector (SENSITIVITY)

VERDICT **FAIL**

### A1

| tau | n | CRPS model | CRPS climatology | beats clim | MAE(median) | 10-90 coverage |
|---|---|---|---|---|---|---|
| 30 | 7542 | 51.6 | 450.4 | yes | 73.6 | 0.805 |
| 60 | 7173 | 148.4 | 879.5 | yes | 212.5 | 0.793 |
| 120 | 6435 | 414.6 | 1678.2 | yes | 596.2 | 0.815 |
| 300 | 4221 | 1612.1 | 3620.9 | yes | 2314.3 | 0.809 |
| 570 | 900 | 4319.6 | 5738.4 | yes | 6234.5 | 0.807 |

Overall 10-90 coverage 0.805 (band 0.75-0.85: in); A1 = PASS

### A2

| tau | n | MAE model | MAE ridge | CRPS model | CRPS ridge | verdict | gating |
|---|---|---|---|---|---|---|---|
| 30 | 7542 | 73.6 | 85.9 | 51.6 | 61.0 | PASS | gating |
| 60 | 7173 | 212.5 | 231.6 | 148.4 | 163.9 | PASS | gating |
| 120 | 6435 | 596.2 | 628.7 | 414.6 | 444.6 | PASS | gating |
| 300 | 4221 | 2314.3 | 2481.9 | 1612.1 | 1756.3 | PASS | reported |
| 570 | 900 | 6234.5 | 6264.7 | 4319.6 | 4340.9 | PASS | reported |

A2-gating (tau 30, 60, 120 all PASS): PASS

### A3 (all contrasts)

| contrast | tau | pairs (worlds) | eligible |dY|>med | sign acc [90% CI] | (i) | CRPS dV | CRPS zero | CRPS cell-mean | (ii) | CRPS dV comonotone | tau pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| damping-native | 30 | 23 (23) | 11 (thr 27.2) | 0.909 [0.727, 1.000] | PASS | 34.4 | 41.2 | 21.9 | fail | 30.5 | fail |
| damping-native | 60 | 23 (23) | 14 (thr 75.2) | 0.929 [0.786, 1.000] | PASS | 94.2 | 127.9 | 74.3 | fail | 80.0 | fail |
| damping-native | 120 | 23 (23) | 15 (thr 205.3) | 0.867 [0.733, 1.000] | PASS | 295.1 | 439.9 | 278.1 | fail | 300.4 | fail |
| damping-native | 300 | 23 (23) | 14 (thr 874.6) | 1.000 [1.000, 1.000] | PASS | 1129.6 | 1885.7 | 1013.8 | fail | 1067.8 | fail |
| damping-native | 570 | 23 (23) | 14 (thr 3245.3) | 0.929 [0.786, 1.000] | PASS | 3056.9 | 5584.2 | 3303.9 | PASS | 2831.4 | PASS |
| fixed-native | 30 | 18 (9) | 6 (thr 10.8) | 0.833 [0.500, 1.000] | fail | 29.5 | 14.0 | 9.8 | fail | 11.9 | fail |
| fixed-native | 60 | 18 (9) | 9 (thr 41.2) | 0.778 [0.500, 1.000] | fail | 85.7 | 55.0 | 45.5 | fail | 42.9 | fail |
| fixed-native | 120 | 18 (9) | 11 (thr 124.5) | 0.818 [0.500, 1.000] | fail | 254.7 | 186.5 | 148.1 | fail | 150.7 | fail |
| fixed-native | 300 | 18 (9) | 9 (thr 648.0) | 0.889 [0.667, 1.000] | PASS | 1022.2 | 789.2 | 703.2 | fail | 716.4 | fail |
| fixed-native | 570 | 18 (9) | 9 (thr 2568.4) | 0.667 [0.375, 0.900] | fail | 2752.6 | 2308.1 | 2149.2 | fail | 2255.6 | fail |
| pi-native | 30 | 23 (23) | 7 (thr 7.5) | 0.429 [0.143, 0.714] | fail | 29.7 | 9.2 | 13.3 | fail | 10.2 | fail |
| pi-native | 60 | 23 (23) | 10 (thr 30.6) | 1.000 [1.000, 1.000] | PASS | 86.7 | 47.7 | 50.8 | fail | 42.1 | fail |
| pi-native | 120 | 23 (23) | 11 (thr 123.8) | 1.000 [1.000, 1.000] | PASS | 273.7 | 205.1 | 229.3 | fail | 193.5 | fail |
| pi-native | 300 | 23 (23) | 11 (thr 731.3) | 0.909 [0.727, 1.000] | PASS | 1091.1 | 995.6 | 920.2 | fail | 862.6 | fail |
| pi-native | 570 | 23 (23) | 10 (thr 2162.1) | 0.900 [0.700, 1.000] | PASS | 2831.1 | 2576.8 | 2656.1 | fail | 2322.2 | fail |
| switching-native | 30 | 27 (9) | 10 (thr 8.8) | 0.600 [0.333, 1.000] | fail | 29.4 | 12.0 | 8.1 | fail | 10.8 | fail |
| switching-native | 60 | 27 (9) | 15 (thr 27.3) | 0.800 [0.667, 0.933] | PASS | 84.1 | 43.1 | 30.3 | fail | 37.3 | fail |
| switching-native | 120 | 27 (9) | 13 (thr 89.4) | 0.923 [0.812, 1.000] | PASS | 251.4 | 161.0 | 113.1 | fail | 144.0 | fail |
| switching-native | 300 | 27 (9) | 12 (thr 747.0) | 1.000 [1.000, 1.000] | PASS | 1065.4 | 937.6 | 680.5 | fail | 836.0 | fail |
| switching-native | 570 | 27 (9) | 14 (thr 2217.8) | 0.857 [0.667, 1.000] | PASS | 2886.1 | 3114.1 | 2266.0 | fail | 2655.3 | fail |

- damping-native: A3 (i AND ii at >= 2 of 3 taus in 60/120/300) = fail (0/3 taus)
- pi-native: A3 (i AND ii at >= 2 of 3 taus in 60/120/300) = fail (0/3 taus)

Per-policy coverage:

| policy | n | coverage |
|---|---|---|
| damping_fleet | 6571 | 0.807 |
| fixed_mix | 3546 | 0.802 |
| native_fleet | 6304 | 0.812 |
| pi_fleet | 4531 | 0.798 |
| switching_mix | 5319 | 0.802 |

TRAIN-OOB verdict (labelled, not out-of-sample): FAIL

## deployable_strict WITHOUT auxiliary chunks 1-3 (ABLATION)

VERDICT **FAIL**

### A1

| tau | n | CRPS model | CRPS climatology | beats clim | MAE(median) | 10-90 coverage |
|---|---|---|---|---|---|---|
| 30 | 7542 | 172.9 | 449.4 | yes | 248.0 | 0.895 |
| 60 | 7173 | 357.1 | 877.7 | yes | 510.0 | 0.872 |
| 120 | 6435 | 746.0 | 1675.3 | yes | 1067.8 | 0.858 |
| 300 | 4221 | 2155.8 | 3621.2 | yes | 3217.9 | 0.894 |
| 570 | 900 | 4014.0 | 5790.4 | yes | 5911.8 | 0.877 |

Overall 10-90 coverage 0.879 (band 0.75-0.85: OUT); A1 = fail

### A2

| tau | n | MAE model | MAE ridge | CRPS model | CRPS ridge | verdict | gating |
|---|---|---|---|---|---|---|---|
| 30 | 7542 | 248.0 | 291.1 | 172.9 | 220.2 | PASS | gating |
| 60 | 7173 | 510.0 | 592.5 | 357.1 | 437.1 | PASS | gating |
| 120 | 6435 | 1067.8 | 1207.3 | 746.0 | 875.1 | PASS | gating |
| 300 | 4221 | 3217.9 | 3284.8 | 2155.8 | 2302.8 | PASS | reported |
| 570 | 900 | 5911.8 | 6019.3 | 4014.0 | 4195.8 | PASS | reported |

A2-gating (tau 30, 60, 120 all PASS): PASS

### A3 (all contrasts)

| contrast | tau | pairs (worlds) | eligible |dY|>med | sign acc [90% CI] | (i) | CRPS dV | CRPS zero | CRPS cell-mean | (ii) | CRPS dV comonotone | tau pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| damping-native | 30 | 23 (23) | 11 (thr 27.2) | 0.909 [0.727, 1.000] | PASS | 105.3 | 41.2 | 21.9 | fail | 69.3 | fail |
| damping-native | 60 | 23 (23) | 14 (thr 75.2) | 0.929 [0.786, 1.000] | PASS | 222.6 | 127.9 | 74.3 | fail | 133.9 | fail |
| damping-native | 120 | 23 (23) | 15 (thr 205.3) | 0.867 [0.733, 1.000] | PASS | 491.2 | 439.9 | 278.1 | fail | 280.3 | fail |
| damping-native | 300 | 23 (23) | 14 (thr 874.6) | 1.000 [1.000, 1.000] | PASS | 1459.4 | 1885.7 | 1013.8 | fail | 820.8 | fail |
| damping-native | 570 | 23 (23) | 14 (thr 3245.3) | 0.929 [0.786, 1.000] | PASS | 3277.2 | 5584.2 | 3303.9 | PASS | 2753.2 | PASS |
| fixed-native | 30 | 18 (9) | 6 (thr 10.8) | 1.000 [1.000, 1.000] | PASS | 105.0 | 14.0 | 9.8 | fail | 47.7 | fail |
| fixed-native | 60 | 18 (9) | 9 (thr 41.2) | 0.889 [0.625, 1.000] | PASS | 222.4 | 55.0 | 45.5 | fail | 92.4 | fail |
| fixed-native | 120 | 18 (9) | 11 (thr 124.5) | 1.000 [1.000, 1.000] | PASS | 488.6 | 186.5 | 148.1 | fail | 236.3 | fail |
| fixed-native | 300 | 18 (9) | 9 (thr 648.0) | 0.889 [0.667, 1.000] | PASS | 1511.0 | 789.2 | 703.2 | fail | 681.4 | fail |
| fixed-native | 570 | 18 (9) | 9 (thr 2568.4) | 0.778 [0.556, 1.000] | PASS | 2865.0 | 2308.1 | 2149.2 | fail | 2045.2 | fail |
| pi-native | 30 | 23 (23) | 7 (thr 7.5) | 1.000 [1.000, 1.000] | PASS | 109.0 | 9.2 | 13.3 | fail | 80.9 | fail |
| pi-native | 60 | 23 (23) | 10 (thr 30.6) | 1.000 [1.000, 1.000] | PASS | 225.7 | 47.7 | 50.8 | fail | 153.4 | fail |
| pi-native | 120 | 23 (23) | 11 (thr 123.8) | 1.000 [1.000, 1.000] | PASS | 486.2 | 205.1 | 229.3 | fail | 264.4 | fail |
| pi-native | 300 | 23 (23) | 11 (thr 731.3) | 0.909 [0.727, 1.000] | PASS | 1434.0 | 995.6 | 920.2 | fail | 624.2 | fail |
| pi-native | 570 | 23 (23) | 10 (thr 2162.1) | 0.900 [0.700, 1.000] | PASS | 2981.4 | 2576.8 | 2656.1 | fail | 2279.7 | fail |
| switching-native | 30 | 27 (9) | 10 (thr 8.8) | 1.000 [1.000, 1.000] | PASS | 104.6 | 12.0 | 8.1 | fail | 44.2 | fail |
| switching-native | 60 | 27 (9) | 15 (thr 27.3) | 0.933 [0.812, 1.000] | PASS | 220.7 | 43.1 | 30.3 | fail | 83.1 | fail |
| switching-native | 120 | 27 (9) | 13 (thr 89.4) | 1.000 [1.000, 1.000] | PASS | 481.8 | 161.0 | 113.1 | fail | 169.4 | fail |
| switching-native | 300 | 27 (9) | 12 (thr 747.0) | 1.000 [1.000, 1.000] | PASS | 1514.7 | 937.6 | 680.5 | fail | 631.1 | fail |
| switching-native | 570 | 27 (9) | 14 (thr 2217.8) | 0.857 [0.667, 1.000] | PASS | 2963.8 | 3114.1 | 2266.0 | fail | 2271.3 | fail |

- damping-native: A3 (i AND ii at >= 2 of 3 taus in 60/120/300) = fail (0/3 taus)
- pi-native: A3 (i AND ii at >= 2 of 3 taus in 60/120/300) = fail (0/3 taus)

Per-policy coverage:

| policy | n | coverage |
|---|---|---|
| damping_fleet | 6571 | 0.876 |
| fixed_mix | 3546 | 0.860 |
| native_fleet | 6304 | 0.916 |
| pi_fleet | 4531 | 0.879 |
| switching_mix | 5319 | 0.852 |

TRAIN-OOB verdict (labelled, not out-of-sample): FAIL


## Appendix A: errors as % of the mean target

Scale: TRAIN mean remaining-delay target per tau over the TRAIN rows of the loaded dataset (the climatology comparator population: all TRAIN rows for strict/detector, non-auxiliary rows for the ablation; `partB_v0data/v04a_results/train_target_means_v04a.json`). CAL files were not re-opened for this section.


### deployable_strict (HEADLINE)

| tau | MAE model (s) | % of mean target | CRPS model (s) | % of mean target | mean target (s) |
|---|---|---|---|---|---|
| 30 | 244.7 | 13.7% | 169.9 | 9.5% | 1787.2 |
| 60 | 501.3 | 14.0% | 351.2 | 9.8% | 3572.3 |
| 120 | 1039.2 | 14.6% | 734.7 | 10.3% | 7138.0 |
| 300 | 3073.5 | 17.3% | 2109.2 | 11.8% | 17801.4 |
| 570 | 6206.3 | 18.6% | 4195.5 | 12.6% | 33284.6 |

### deployable_detector (SENSITIVITY)

| tau | MAE model (s) | % of mean target | CRPS model (s) | % of mean target | mean target (s) |
|---|---|---|---|---|---|
| 30 | 73.6 | 4.1% | 51.6 | 2.9% | 1787.2 |
| 60 | 212.5 | 5.9% | 148.4 | 4.2% | 3572.3 |
| 120 | 596.2 | 8.4% | 414.6 | 5.8% | 7138.0 |
| 300 | 2314.3 | 13.0% | 1612.1 | 9.1% | 17801.4 |
| 570 | 6234.5 | 18.7% | 4319.6 | 13.0% | 33284.6 |

### deployable_strict WITHOUT auxiliary chunks 1-3 (ABLATION)

| tau | MAE model (s) | % of mean target | CRPS model (s) | % of mean target | mean target (s) |
|---|---|---|---|---|---|
| 30 | 248.0 | 14.0% | 172.9 | 9.8% | 1765.6 |
| 60 | 510.0 | 14.5% | 357.1 | 10.1% | 3528.5 |
| 120 | 1067.8 | 15.1% | 746.0 | 10.6% | 7049.2 |
| 300 | 3217.9 | 18.3% | 2155.8 | 12.3% | 17578.2 |
| 570 | 5911.8 | 17.9% | 4014.0 | 12.2% | 32934.9 |

## Appendix B: comparison to v0.3.2_strict (reference)

v0.3.2_strict numbers are from ITS OWN single CAL read on the v0.3.2 root (`partB_results/v0.3.2_strict/eval_report.md`, 10959 CAL rows); v0.4a is on the new TRUE-policy root (7542/7173/6435/4221/900 CAL rows per tau). The two CAL sets differ (chunk-4 worlds added; native arms restored), so this is a like-for-like family comparison, NOT a paired difference.


| quantity | v0.3.2_strict | v0.4a strict (HEADLINE) |
|---|---|---|
| native reference | random per-CAV SWITCHING mix (WM-72 caveat) | restored TRUE fleet-native arms |
| overall CRPS model vs climatology | 977.0 vs 1911.4 (coverage 0.800, in band) | 807.2 vs 1558.9 (coverage 0.880, OUT of band) |
| A2 vs ridge, gating 30/60/120 | PASS (FAIL at 300/570 reported) | PASS (strict: 300 PASS, 570 FAIL reported) |
| A1 overall | (no formal A1; coverage 0.800 in band) | fail (CRPS beats climatology at every tau; overall 10-90 coverage 0.880, above the 0.85 upper band edge) |
| contrast evidence | damping vs switching-mix 'native' at tau=570 only: sign 0.76, Spearman 0.030 | full A3: sign accuracy PASS at all gating taus, but dV CRPS does not beat the zero-contrast / TRAIN-cell-mean baselines at 60/120/300 -> A3 fail |

Reading: v0.4a improves per-tau sharpness (CRPS below v0.3.2 at 30-300 s on a harder, larger CAL set) and keeps A2-gating, but (i) the conformal 10-90% band is too WIDE on CAL (coverage 0.880 > 0.85: A1 fail) and (ii) the learned contrast distribution is not yet sharp enough to beat a zero contrast on |dY|>median pairs: the TRUE damping-vs-native and PI-vs-native gaps are smaller and noisier than the old switching-mix contrasts (PB-RESCORE1 premise). VERDICT per registration s.6: FAIL (strict, detector, ablation all FAIL; A3(ii) is the binding constraint).

## Appendix C: demand / penetration breakdowns (report strata only)

Coverage and CRPS by demand and penetration, strict (HEADLINE); detector and ablation strata are in their `eval_v04a.json` (`a1.by_demand`, `a1.by_penetration`).

| stratum | n | coverage | CRPS model | CRPS climatology |
|---|---|---|---|---|
| demand=2000 | 3375 | 0.908 | 455.4 | 323.0 |
| demand=2400 | 1482 | 0.961 | 746.0 | 1497.1 |
| demand=2600 | 11157 | 0.850 | 819.4 | 1782.8 |
| demand=2700 | 10257 | 0.891 | 918.6 | 1730.8 |
| penetration=0.02 | 4878 | 0.938 | 742.7 | 1676.5 |
| penetration=0.05 | 4998 | 0.964 | 584.5 | 1148.7 |
| penetration=0.1 | 4968 | 0.946 | 619.1 | 1211.5 |
| penetration=0.2 | 6630 | 0.720 | 1173.4 | 1847.2 |
| penetration=0.4 | 4797 | 0.885 | 793.6 | 1827.8 |

## Appendix D: always-reported items (registration s.6)


- per-policy coverage: per-variant section tables above.
- switching-native and fixed-native contrasts: A3 tables above (reported, not gating).
- auxiliary-data ablation: strict_noaux variant above (removing aux rows changes A1 coverage 0.880 -> 0.879 and leaves every verdict FAIL: the failure is not an auxiliary-data artefact).
- comonotone dV sensitivity: `CRPS dV comonotone` column of every A3 table (same pass/fail pattern as independent).
- demand / penetration strata: Appendix C.
- TRAIN-OOB versions of A1-A3 (selection-free of CAL, in-sample for the conformal shifts): `TRAIN-OOB verdict` lines above; full tables in each `eval_v04a.json` (`train_oob`).

## Appendix E: disclosures (executor + process)


- Executor change: the original executor (Sonnet) was stopped by owner decision 2026-09-30 mid dry-run; this resume was executed by GLM Flash (ZCode) in worktree `E:/rd-wt/wd01v04c`, branch `zcode/wd01v04c` from `15def5f`, after a runtime-limit resume of the session.
- Owner-stop interruption: the first dry-run (2026-09-30 17:03 local, PID 6816) was interrupted by the owner stop (laptop taken out) BEFORE any CAL read; 0 `CAL_READ_ONCE_v04a.json` existed then and now before this resume (searched wd01v04a, wd01v04b, wd01v04c and `partB_results`). See `partB_v0data/v04a_results/INTERRUPTED_DRYRUN_20260930.txt`.
- ONE CAL read: performed exactly once in this resume (lock `CAL_READ_ONCE_v04a.json` written before the first CAL row; started/finished per the lock; no refit after it).
- MemoryError failed attempts: the first strict/detector fit launches and the first backup launch died at start with MemoryError (receipts in `partB_v0data/v04a_results/*.err`, logs under `logs/attempt1_memoryerror/`); the half-created result dirs were refitted in place (attempt 2), all verified.
- `git reset -q HEAD -- <paths>` was run once by mistake in the previous session to unstage paths (plus `git rm --cached` for staged `__pycache__`); nothing was discarded. Disclosed because the rules forbid git reset in any form (also CHANGES_V04A.md item 8).
- CAL exposure (from the registration): chunk-2/3 CAL was read once each by v0.3.2 and RESCORE1; chunk-4 CAL is also C4L1's Part A look set. This run is the disclosed third read of chunk-2/3 CAL.
- Memory gates: before the dry-run 13.68 GB free RAM / 22.0 GB free VM; before the CAL read 16.65 GB / 32.52 GB (both >= 12 / >= 8 GB; readings logged in `dryrun_resume.ram.log` / `cal_read.ram.log`). Runs were detached, BelowNormal priority, <= 4 workers, WD_FORBID_D=1, zero SUMO/TraCI.
