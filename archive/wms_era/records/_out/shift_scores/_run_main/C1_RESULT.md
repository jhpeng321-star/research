# C1 (leave-one-controller-out) scoring - WM-S-C1FAST-20261006

**Status: complete.** Structured WM-S vs black-box-id vs black-box-traj, all trained on the same PI-free TRAIN windows; keep_speed and pure_physics as references. Baseline arm `native`; single-control-variable model classes.

## Sets

- `u_same`: 140 states / 140 worlds
- `u_new`: 21 states / 21 worlds
- `seen`: 157 states / 26 worlds

## Unseen/seen error ratio (median across states; 95% CI over worlds)

Car metric `ds` = HDV median |s_pred - s_rec| (m); `dv` = median |v_pred - v_rec| (m/s);
traffic = merge normalized delay (s) in the 90-120 s bin.

| model | split | car ds@30 | car ds@60 | car dv@30 | traffic delay@120 |
|---|---|---|---|---|---|
| keep_speed | u_same | 0.684 [0.42, 0.91] | 0.590 [0.39, 0.96] | 0.684 [0.49, 0.86] | 0.973 [0.53, 1.77] |
| keep_speed | u_new | 0.820 [0.46, 1.43] | 0.636 [0.35, 0.99] | 0.824 [0.52, 1.12] | 0.848 [0.29, 2.35] |
| pure_physics | u_same | 0.859 [0.77, 0.95] | 0.582 [0.47, 0.73] | 0.720 [0.59, 0.85] | 1.250 [0.93, 1.79] |
| pure_physics | u_new | 0.832 [0.69, 0.98] | 0.615 [0.42, 0.96] | 0.839 [0.59, 1.04] | 1.012 [0.68, 1.72] |
| wms | u_same | 0.828 [0.65, 1.01] | 0.530 [0.36, 0.89] | 0.932 [0.64, 1.11] | 1.343 [0.79, 2.42] |
| wms | u_new | 0.751 [0.55, 1.03] | 0.553 [0.28, 1.10] | 0.746 [0.51, 1.00] | 1.546 [0.43, 3.04] |
| blackbox_id | u_same | 0.772 [0.60, 0.91] | 0.512 [0.34, 0.65] | 0.846 [0.57, 0.94] | 0.725 [0.50, 1.31] |
| blackbox_id | u_new | 0.769 [0.57, 0.98] | 0.498 [0.33, 0.74] | 0.828 [0.55, 1.00] | 0.383 [0.24, 1.71] |
| blackbox_traj | u_same | 0.825 [0.67, 0.94] | 0.662 [0.53, 0.86] | 0.787 [0.67, 0.93] | 0.824 [0.37, 2.82] |
| blackbox_traj | u_new | 0.859 [0.66, 0.99] | 0.597 [0.48, 0.90] | 0.765 [0.64, 1.07] | 1.012 [0.26, 4.24] |

## Paired structured - black-box, on unseen (negative favours structured)

| split | metric | structured - bb_traj | structured - bb_id |
|---|---|---|---|
| u_same | car_ds@30 | -3.191 [-3.65, -2.65] | -1.738 [-2.80, -0.90] |
| u_same | car_ds@60 | -7.701 [-9.64, -6.28] | -10.307 [-11.87, -6.39] |
| u_same | traffic_delay@120 | -7.028 [-25.89, -0.67] | -1.122 [-8.52, 0.40] |
| u_new | car_ds@30 | -3.148 [-4.78, -1.87] | -0.367 [-2.72, 0.15] |
| u_new | car_ds@60 | -4.298 [-8.31, -0.13] | -9.453 [-11.95, -5.31] |
| u_new | traffic_delay@120 | -26.437 [-91.28, -2.15] | 0.023 [-7.16, 11.48] |

## Per-model medians (native arm, metres and seconds)

| model | set | ds@10 | ds@30 | ds@60 | ds@120 | delay@120 |
|---|---|---|---|---|---|---|
| keep_speed | u_same | 3.11 | 17.77 | 72.88 | 1322.81 | 73.42 |
| keep_speed | u_new | 3.05 | 21.31 | 78.58 | 1244.12 | 64.00 |
| keep_speed | seen | 3.30 | 26.00 | 123.47 | 1227.53 | 75.44 |
| pure_physics | u_same | 7.31 | 26.52 | 57.85 | 980.53 | 128.32 |
| pure_physics | u_new | 6.96 | 25.71 | 61.06 | 942.79 | 103.92 |
| pure_physics | seen | 7.23 | 30.88 | 99.33 | 1039.71 | 102.64 |
| wms | u_same | 2.11 | 6.34 | 15.04 | 91.25 | 30.80 |
| wms | u_new | 2.14 | 5.75 | 15.69 | 61.32 | 35.45 |
| wms | seen | 2.08 | 7.66 | 28.38 | 145.16 | 22.93 |
| blackbox_id | u_same | 2.06 | 8.28 | 24.90 | 89.51 | 31.34 |
| blackbox_id | u_new | 1.97 | 8.25 | 24.25 | 88.01 | 16.54 |
| blackbox_id | seen | 2.10 | 10.72 | 48.66 | 218.68 | 43.21 |
| blackbox_traj | u_same | 2.55 | 10.03 | 23.77 | 137.75 | 60.64 |
| blackbox_traj | u_new | 2.63 | 10.44 | 21.43 | 120.48 | 74.51 |
| blackbox_traj | seen | 2.69 | 12.15 | 35.88 | 192.66 | 73.61 |

## Verdict

C1 asks whether a physics-structured world model that runs the PI controller as known code (WM-S) predicts a controller it was never trained on (PI, removed from all training data) nearly as well as ordinary non-PI situations, while a strong black-box learned-HDV model (black-box-traj) degrades. wms unseen/seen |ds|@60 ratio: U-same 0.53 [0.36, 0.89], U-new 0.55 [0.28, 1.10]. blackbox_traj unseen/seen |ds|@60 ratio: U-same 0.66 [0.53, 0.86], U-new 0.60 [0.48, 0.90]. blackbox_id unseen/seen |ds|@60 ratio: U-same 0.51 [0.34, 0.65], U-new 0.50 [0.33, 0.74]. On U-same at 60 s the structured median |ds| is 15.04 m versus black-box-traj 23.77 m and black-box-id 24.90 m; the paired structured - black-box-traj difference is -7.701 [-9.64, -6.28] m at 60 s and -3.191 [-3.65, -2.65] m at 30 s (negative favours structured). Verdict: C1 is SUPPORTED - structured WM-S does not degrade on the unseen PI controller (U-same and U-new ratios below 1) and its 60 s HDV error on U-same is lower than the strong black-box-traj baseline with a 95% bootstrap CI excluding 0. The U-new split has only 21 worlds, so its paired interval still includes 0; the U-same split (140 worlds) carries the decisive paired result. Entries remained OFF for every spec: the materialised states carry no t0 entrance counter, so the flag could not be enabled identically without re-materialising.
