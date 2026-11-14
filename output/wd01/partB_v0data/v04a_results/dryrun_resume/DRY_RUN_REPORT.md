# DRY RUN (TRAIN rows in the CAL role; plumbing only, NOT evidence)

# REPORT_V04A: Part B value model v0.4a on TRUE policy references (no walls)

Registration: `output/wd01/REGISTRATION_PARTB_V04A.md` (LF sha256 `38a6fa406ca9d598fdaae986263f4fa81ee15dfa8c632414cc5f28dbe5bdee00`). Acceptance rule fixed BEFORE running; ONE CAL read; no refit afterwards.

Ideal-sensing premise: lane bins come from the TRUE state; the reconstruction gap is measured separately (Part A). Inputs carry no penetration, pre_fix, speed_mode or world strings; t0 is unrecoverable (ctx col 1 = 0). Wall tokens are empty (v0.4b adds walls).

## Verdict table

| variant | VERDICT | A1 | A2 gating (30/60/120) | A3 damping-native | A3 pi-native |
|---|---|---|---|---|---|
| deployable_strict (HEADLINE) | FAIL | fail | PASS | fail | fail |
| deployable_detector (SENSITIVITY) | FAIL | fail | PASS | fail | fail |
| deployable_strict WITHOUT auxiliary chunks 1-3 (ABLATION) | FAIL | fail | PASS | fail | fail |

## deployable_strict (HEADLINE)

VERDICT **FAIL**

### A1

| tau | n | CRPS model | CRPS climatology | beats clim | MAE(median) | 10-90 coverage |
|---|---|---|---|---|---|---|
| 30 | 52428 | 164.1 | 450.9 | yes | 227.0 | 0.931 |
| 60 | 49845 | 328.1 | 883.7 | yes | 454.1 | 0.925 |
| 120 | 44679 | 692.4 | 1698.4 | yes | 920.8 | 0.925 |
| 300 | 29181 | 1943.1 | 3789.3 | yes | 2747.8 | 0.923 |
| 570 | 5934 | 3658.9 | 6206.3 | yes | 4574.4 | 0.921 |

Overall 10-90 coverage 0.926 (band 0.75-0.85: OUT); A1 = fail

### A2

| tau | n | MAE model | MAE ridge | CRPS model | CRPS ridge | verdict | gating |
|---|---|---|---|---|---|---|---|
| 30 | 52428 | 227.0 | 326.7 | 164.1 | 230.6 | PASS | gating |
| 60 | 49845 | 454.1 | 660.9 | 328.1 | 467.9 | PASS | gating |
| 120 | 44679 | 920.8 | 1345.1 | 692.4 | 957.0 | PASS | gating |
| 300 | 29181 | 2747.8 | 3537.6 | 1943.1 | 2521.0 | PASS | reported |
| 570 | 5934 | 4574.4 | 7003.0 | 3658.9 | 4972.9 | PASS | reported |

A2-gating (tau 30, 60, 120 all PASS): PASS

### A3 (all contrasts)

| contrast | tau | pairs (worlds) | eligible |dY|>med | sign acc [90% CI] | (i) | CRPS dV | CRPS zero | CRPS cell-mean | (ii) | CRPS dV comonotone | tau pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| damping-native | 30 | 161 (161) | 80 (thr 27.2) | 0.975 [0.938, 1.000] | PASS | 105.2 | 47.3 | 19.1 | fail | 63.9 | fail |
| damping-native | 60 | 161 (161) | 80 (thr 75.2) | 0.963 [0.925, 0.988] | PASS | 218.9 | 127.8 | 60.5 | fail | 129.9 | fail |
| damping-native | 120 | 161 (161) | 80 (thr 205.3) | 0.912 [0.863, 0.963] | PASS | 490.1 | 348.4 | 193.0 | fail | 296.0 | fail |
| damping-native | 300 | 161 (161) | 80 (thr 874.6) | 0.900 [0.850, 0.950] | PASS | 1485.3 | 1446.6 | 934.5 | fail | 966.1 | fail |
| damping-native | 570 | 161 (161) | 80 (thr 3245.3) | 0.875 [0.812, 0.938] | PASS | 3517.5 | 4262.1 | 2884.5 | fail | 2644.1 | fail |
| fixed-native | 30 | 126 (63) | 63 (thr 10.8) | 0.952 [0.902, 0.985] | PASS | 103.8 | 21.9 | 10.6 | fail | 76.5 | fail |
| fixed-native | 60 | 126 (63) | 63 (thr 41.2) | 0.825 [0.714, 0.923] | PASS | 215.1 | 65.9 | 36.1 | fail | 151.8 | fail |
| fixed-native | 120 | 126 (63) | 63 (thr 124.5) | 0.746 [0.633, 0.846] | PASS | 480.1 | 222.7 | 147.9 | fail | 292.9 | fail |
| fixed-native | 300 | 126 (63) | 63 (thr 648.0) | 0.794 [0.696, 0.881] | PASS | 1480.7 | 1097.9 | 902.4 | fail | 1066.3 | fail |
| fixed-native | 570 | 126 (63) | 63 (thr 2568.4) | 0.714 [0.600, 0.815] | PASS | 3557.5 | 3370.7 | 2863.6 | fail | 2954.0 | fail |
| pi-native | 30 | 161 (161) | 80 (thr 7.5) | 0.775 [0.700, 0.850] | PASS | 110.3 | 16.4 | 13.9 | fail | 78.9 | fail |
| pi-native | 60 | 161 (161) | 80 (thr 30.6) | 0.775 [0.700, 0.850] | PASS | 226.7 | 64.8 | 56.7 | fail | 152.4 | fail |
| pi-native | 120 | 161 (161) | 80 (thr 123.8) | 0.775 [0.700, 0.850] | PASS | 500.5 | 227.2 | 194.1 | fail | 319.0 | fail |
| pi-native | 300 | 161 (161) | 80 (thr 731.3) | 0.713 [0.625, 0.800] | PASS | 1511.0 | 996.4 | 930.8 | fail | 1050.0 | fail |
| pi-native | 570 | 161 (161) | 80 (thr 2162.1) | 0.613 [0.525, 0.700] | PASS | 3488.0 | 2860.7 | 2610.9 | fail | 2737.1 | fail |
| switching-native | 30 | 189 (63) | 94 (thr 8.8) | 0.904 [0.830, 0.968] | PASS | 102.4 | 20.4 | 11.9 | fail | 68.3 | fail |
| switching-native | 60 | 189 (63) | 94 (thr 27.3) | 0.830 [0.735, 0.914] | PASS | 213.1 | 61.2 | 39.2 | fail | 136.7 | fail |
| switching-native | 120 | 189 (63) | 94 (thr 89.4) | 0.798 [0.711, 0.873] | PASS | 477.6 | 196.6 | 139.1 | fail | 273.4 | fail |
| switching-native | 300 | 189 (63) | 94 (thr 747.0) | 0.745 [0.656, 0.821] | PASS | 1458.8 | 1003.3 | 883.7 | fail | 936.1 | fail |
| switching-native | 570 | 189 (63) | 94 (thr 2217.8) | 0.723 [0.617, 0.821] | PASS | 3499.4 | 3030.2 | 2708.4 | fail | 2677.8 | fail |

- damping-native: A3 (i AND ii at >= 2 of 3 taus in 60/120/300) = fail (0/3 taus)
- pi-native: A3 (i AND ii at >= 2 of 3 taus in 60/120/300) = fail (0/3 taus)

Per-policy coverage:

| policy | n | coverage |
|---|---|---|
| damping_fleet | 44167 | 0.920 |
| fixed_mix | 24822 | 0.930 |
| native_fleet | 44128 | 0.932 |
| pi_fleet | 31717 | 0.917 |
| switching_mix | 37233 | 0.932 |

TRAIN-OOB verdict (labelled, not out-of-sample): FAIL

## deployable_detector (SENSITIVITY)

VERDICT **FAIL**

### A1

| tau | n | CRPS model | CRPS climatology | beats clim | MAE(median) | 10-90 coverage |
|---|---|---|---|---|---|---|
| 30 | 52428 | 43.2 | 450.9 | yes | 61.4 | 0.882 |
| 60 | 49845 | 122.1 | 883.7 | yes | 175.1 | 0.871 |
| 120 | 44679 | 352.9 | 1698.4 | yes | 495.9 | 0.880 |
| 300 | 29181 | 1419.3 | 3789.3 | yes | 2054.9 | 0.908 |
| 570 | 5934 | 4103.7 | 6206.3 | yes | 5879.0 | 0.824 |

Overall 10-90 coverage 0.881 (band 0.75-0.85: OUT); A1 = fail

### A2

| tau | n | MAE model | MAE ridge | CRPS model | CRPS ridge | verdict | gating |
|---|---|---|---|---|---|---|---|
| 30 | 52428 | 61.4 | 83.8 | 43.2 | 59.5 | PASS | gating |
| 60 | 49845 | 175.1 | 221.5 | 122.1 | 157.4 | PASS | gating |
| 120 | 44679 | 495.9 | 603.6 | 352.9 | 429.5 | PASS | gating |
| 300 | 29181 | 2054.9 | 2288.0 | 1419.3 | 1629.0 | PASS | reported |
| 570 | 5934 | 5879.0 | 5881.9 | 4103.7 | 4168.5 | PASS | reported |

A2-gating (tau 30, 60, 120 all PASS): PASS

### A3 (all contrasts)

| contrast | tau | pairs (worlds) | eligible |dY|>med | sign acc [90% CI] | (i) | CRPS dV | CRPS zero | CRPS cell-mean | (ii) | CRPS dV comonotone | tau pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| damping-native | 30 | 161 (161) | 80 (thr 27.2) | 0.975 [0.938, 1.000] | PASS | 36.5 | 47.3 | 19.1 | fail | 33.6 | fail |
| damping-native | 60 | 161 (161) | 80 (thr 75.2) | 0.963 [0.925, 0.988] | PASS | 96.1 | 127.8 | 60.5 | fail | 77.4 | fail |
| damping-native | 120 | 161 (161) | 80 (thr 205.3) | 0.912 [0.863, 0.963] | PASS | 274.3 | 348.4 | 193.0 | fail | 212.4 | fail |
| damping-native | 300 | 161 (161) | 80 (thr 874.6) | 0.900 [0.850, 0.950] | PASS | 1128.0 | 1446.6 | 934.5 | fail | 934.8 | fail |
| damping-native | 570 | 161 (161) | 80 (thr 3245.3) | 0.863 [0.800, 0.925] | PASS | 3040.5 | 4262.1 | 2884.5 | fail | 2968.0 | fail |
| fixed-native | 30 | 126 (63) | 63 (thr 10.8) | 0.905 [0.836, 0.957] | PASS | 31.6 | 21.9 | 10.6 | fail | 18.1 | fail |
| fixed-native | 60 | 126 (63) | 63 (thr 41.2) | 0.825 [0.717, 0.919] | PASS | 87.7 | 65.9 | 36.1 | fail | 49.5 | fail |
| fixed-native | 120 | 126 (63) | 63 (thr 124.5) | 0.730 [0.619, 0.828] | PASS | 267.2 | 222.7 | 147.9 | fail | 185.5 | fail |
| fixed-native | 300 | 126 (63) | 63 (thr 648.0) | 0.746 [0.639, 0.845] | PASS | 1140.6 | 1097.9 | 902.4 | fail | 958.6 | fail |
| fixed-native | 570 | 126 (63) | 63 (thr 2568.4) | 0.683 [0.571, 0.783] | PASS | 3125.8 | 3370.7 | 2863.6 | fail | 3225.1 | fail |
| pi-native | 30 | 161 (161) | 80 (thr 7.5) | 0.613 [0.512, 0.700] | PASS | 31.0 | 16.4 | 13.9 | fail | 15.3 | fail |
| pi-native | 60 | 161 (161) | 80 (thr 30.6) | 0.812 [0.738, 0.875] | PASS | 91.2 | 64.8 | 56.7 | fail | 57.8 | fail |
| pi-native | 120 | 161 (161) | 80 (thr 123.8) | 0.787 [0.713, 0.863] | PASS | 274.8 | 227.2 | 194.1 | fail | 208.7 | fail |
| pi-native | 300 | 161 (161) | 80 (thr 731.3) | 0.713 [0.625, 0.800] | PASS | 1136.0 | 996.4 | 930.8 | fail | 967.6 | fail |
| pi-native | 570 | 161 (161) | 80 (thr 2162.1) | 0.613 [0.525, 0.700] | PASS | 2990.6 | 2860.7 | 2610.9 | fail | 2929.8 | fail |
| switching-native | 30 | 189 (63) | 94 (thr 8.8) | 0.872 [0.807, 0.929] | PASS | 31.2 | 20.4 | 11.9 | fail | 15.9 | fail |
| switching-native | 60 | 189 (63) | 94 (thr 27.3) | 0.809 [0.719, 0.891] | PASS | 88.3 | 61.2 | 39.2 | fail | 46.7 | fail |
| switching-native | 120 | 189 (63) | 94 (thr 89.4) | 0.798 [0.714, 0.866] | PASS | 264.2 | 196.6 | 139.1 | fail | 162.3 | fail |
| switching-native | 300 | 189 (63) | 94 (thr 747.0) | 0.723 [0.630, 0.804] | PASS | 1111.1 | 1003.3 | 883.7 | fail | 885.1 | fail |
| switching-native | 570 | 189 (63) | 94 (thr 2217.8) | 0.734 [0.628, 0.833] | PASS | 3004.1 | 3030.2 | 2708.4 | fail | 2820.3 | fail |

- damping-native: A3 (i AND ii at >= 2 of 3 taus in 60/120/300) = fail (0/3 taus)
- pi-native: A3 (i AND ii at >= 2 of 3 taus in 60/120/300) = fail (0/3 taus)

Per-policy coverage:

| policy | n | coverage |
|---|---|---|
| damping_fleet | 44167 | 0.871 |
| fixed_mix | 24822 | 0.888 |
| native_fleet | 44128 | 0.881 |
| pi_fleet | 31717 | 0.877 |
| switching_mix | 37233 | 0.891 |

TRAIN-OOB verdict (labelled, not out-of-sample): FAIL

## deployable_strict WITHOUT auxiliary chunks 1-3 (ABLATION)

VERDICT **FAIL**

### A1

| tau | n | CRPS model | CRPS climatology | beats clim | MAE(median) | 10-90 coverage |
|---|---|---|---|---|---|---|
| 30 | 52428 | 146.8 | 449.7 | yes | 214.0 | 0.956 |
| 60 | 49845 | 306.4 | 881.3 | yes | 452.5 | 0.955 |
| 120 | 44679 | 656.3 | 1693.0 | yes | 932.3 | 0.957 |
| 300 | 29181 | 1751.6 | 3772.9 | yes | 2397.0 | 0.959 |
| 570 | 5934 | 2981.1 | 6176.5 | yes | 3941.3 | 0.989 |

Overall 10-90 coverage 0.957 (band 0.75-0.85: OUT); A1 = fail

### A2

| tau | n | MAE model | MAE ridge | CRPS model | CRPS ridge | verdict | gating |
|---|---|---|---|---|---|---|---|
| 30 | 52428 | 214.0 | 326.2 | 146.8 | 237.6 | PASS | gating |
| 60 | 49845 | 452.5 | 660.4 | 306.4 | 477.7 | PASS | gating |
| 120 | 44679 | 932.3 | 1345.8 | 656.3 | 975.5 | PASS | gating |
| 300 | 29181 | 2397.0 | 3539.4 | 1751.6 | 2534.6 | PASS | reported |
| 570 | 5934 | 3941.3 | 6950.7 | 2981.1 | 4947.3 | PASS | reported |

A2-gating (tau 30, 60, 120 all PASS): PASS

### A3 (all contrasts)

| contrast | tau | pairs (worlds) | eligible |dY|>med | sign acc [90% CI] | (i) | CRPS dV | CRPS zero | CRPS cell-mean | (ii) | CRPS dV comonotone | tau pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| damping-native | 30 | 161 (161) | 80 (thr 27.2) | 0.975 [0.938, 1.000] | PASS | 105.5 | 47.3 | 19.1 | fail | 64.1 | fail |
| damping-native | 60 | 161 (161) | 80 (thr 75.2) | 0.950 [0.912, 0.988] | PASS | 222.7 | 127.8 | 60.5 | fail | 136.9 | fail |
| damping-native | 120 | 161 (161) | 80 (thr 205.3) | 0.912 [0.863, 0.963] | PASS | 497.8 | 348.4 | 193.0 | fail | 338.8 | fail |
| damping-native | 300 | 161 (161) | 80 (thr 874.6) | 0.900 [0.850, 0.950] | PASS | 1570.0 | 1446.6 | 934.5 | fail | 1020.5 | fail |
| damping-native | 570 | 161 (161) | 80 (thr 3245.3) | 0.875 [0.812, 0.938] | PASS | 3117.5 | 4262.1 | 2884.5 | fail | 2434.9 | fail |
| fixed-native | 30 | 126 (63) | 63 (thr 10.8) | 0.937 [0.879, 0.985] | PASS | 102.1 | 21.9 | 10.6 | fail | 68.6 | fail |
| fixed-native | 60 | 126 (63) | 63 (thr 41.2) | 0.825 [0.714, 0.923] | PASS | 212.6 | 65.9 | 36.1 | fail | 140.4 | fail |
| fixed-native | 120 | 126 (63) | 63 (thr 124.5) | 0.746 [0.633, 0.846] | PASS | 478.2 | 222.7 | 147.9 | fail | 305.7 | fail |
| fixed-native | 300 | 126 (63) | 63 (thr 648.0) | 0.794 [0.696, 0.881] | PASS | 1564.4 | 1097.9 | 902.4 | fail | 1079.2 | fail |
| fixed-native | 570 | 126 (63) | 63 (thr 2568.4) | 0.746 [0.641, 0.844] | PASS | 3118.0 | 3370.7 | 2863.6 | fail | 2666.2 | fail |
| pi-native | 30 | 161 (161) | 80 (thr 7.5) | 0.762 [0.688, 0.838] | PASS | 110.9 | 16.4 | 13.9 | fail | 81.9 | fail |
| pi-native | 60 | 161 (161) | 80 (thr 30.6) | 0.762 [0.688, 0.838] | PASS | 229.6 | 64.8 | 56.7 | fail | 159.5 | fail |
| pi-native | 120 | 161 (161) | 80 (thr 123.8) | 0.775 [0.700, 0.850] | PASS | 503.4 | 227.2 | 194.1 | fail | 349.5 | fail |
| pi-native | 300 | 161 (161) | 80 (thr 731.3) | 0.725 [0.649, 0.812] | PASS | 1579.0 | 996.4 | 930.8 | fail | 1007.4 | fail |
| pi-native | 570 | 161 (161) | 80 (thr 2162.1) | 0.637 [0.550, 0.725] | PASS | 3097.8 | 2860.7 | 2610.9 | fail | 2456.6 | fail |
| switching-native | 30 | 189 (63) | 94 (thr 8.8) | 0.894 [0.817, 0.959] | PASS | 102.1 | 20.4 | 11.9 | fail | 69.0 | fail |
| switching-native | 60 | 189 (63) | 94 (thr 27.3) | 0.819 [0.726, 0.902] | PASS | 213.6 | 61.2 | 39.2 | fail | 142.9 | fail |
| switching-native | 120 | 189 (63) | 94 (thr 89.4) | 0.798 [0.711, 0.873] | PASS | 478.4 | 196.6 | 139.1 | fail | 304.5 | fail |
| switching-native | 300 | 189 (63) | 94 (thr 747.0) | 0.745 [0.656, 0.821] | PASS | 1542.8 | 1003.3 | 883.7 | fail | 988.3 | fail |
| switching-native | 570 | 189 (63) | 94 (thr 2217.8) | 0.734 [0.632, 0.833] | PASS | 3050.2 | 3030.2 | 2708.4 | fail | 2366.8 | fail |

- damping-native: A3 (i AND ii at >= 2 of 3 taus in 60/120/300) = fail (0/3 taus)
- pi-native: A3 (i AND ii at >= 2 of 3 taus in 60/120/300) = fail (0/3 taus)

Per-policy coverage:

| policy | n | coverage |
|---|---|---|
| damping_fleet | 44167 | 0.951 |
| fixed_mix | 24822 | 0.971 |
| native_fleet | 44128 | 0.955 |
| pi_fleet | 31717 | 0.946 |
| switching_mix | 37233 | 0.969 |

TRAIN-OOB verdict (labelled, not out-of-sample): FAIL

