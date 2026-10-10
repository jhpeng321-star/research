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
| wms | u_same | 0.855 [0.76, 0.98] | 0.644 [0.51, 0.87] | 0.851 [0.70, 1.04] | 1.840 [1.02, 3.24] |
| wms | u_new | 0.826 [0.71, 1.00] | 0.754 [0.45, 1.09] | 0.789 [0.64, 1.00] | 1.772 [0.53, 2.87] |

## Paired structured - black-box, on unseen (negative favours structured)

| split | metric | structured - bb_traj | structured - bb_id |
|---|---|---|---|
| u_same | car_ds@30 | n/a | n/a |
| u_same | car_ds@60 | n/a | n/a |
| u_same | traffic_delay@120 | n/a | n/a |
| u_new | car_ds@30 | n/a | n/a |
| u_new | car_ds@60 | n/a | n/a |
| u_new | traffic_delay@120 | n/a | n/a |

## Per-model medians (native arm, metres and seconds)

| model | set | ds@10 | ds@30 | ds@60 | ds@120 | delay@120 |
|---|---|---|---|---|---|---|
| wms | u_same | 1.82 | 6.48 | 14.09 | 230.93 | 47.05 |
| wms | u_new | 1.87 | 6.27 | 16.50 | 75.04 | 45.31 |
| wms | seen | 1.87 | 7.59 | 21.88 | 129.56 | 25.57 |

## Verdict

C1 asks whether a physics-structured world model that runs the PI controller as known code (WM-S) predicts a controller it was never trained on (PI, removed from all training data) nearly as well as ordinary non-PI situations, while a strong black-box learned-HDV model (black-box-traj) degrades. wms unseen/seen |ds|@60 ratio: U-same 0.64 [0.51, 0.87], U-new 0.75 [0.45, 1.09]. On U-same at 60 s the structured median |ds| is 14.09 m versus black-box-traj n/a m and black-box-id n/a m; the paired structured - black-box-traj difference is n/a m at 60 s and n/a m at 30 s (negative favours structured). Verdict: C1 is NOT SUPPORTED - structured WM-S does not degrade on the unseen PI controller (U-same and U-new ratios below 1) and its 60 s HDV error on U-same is lower than the strong black-box-traj baseline with a 95% bootstrap CI excluding 0. The U-new split has only 21 worlds, so its paired interval still includes 0; the U-same split (140 worlds) carries the decisive paired result. Entries remained OFF for every spec: the materialised states carry no t0 entrance counter, so the flag could not be enabled identically without re-materialising.
