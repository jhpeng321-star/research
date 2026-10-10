# MAIN2 world-model score result (BRAKE / NOBRAKE on the unseen wall sets)

Generated 2026-10-09T07:34:45Z. CPU only, no training, no SUMO. Scored with the exact wall_c1_score harness (CAVs teacher-forced from the recorded arm, entries OFF); each checkpoint uses its own TRAIN_RECEIPT physics.

## Contamination check (STEP 1)

Training lists: BRAKE and NOBRAKE both used `lpwo_v2/windows_train_noPI_noWall.json` (the receipt window list; 1172 windows / 184 worlds, of which 791 damping+dspeed windows / 174 worlds are effective) plus the switching pack (`windows_sw_train_part.json`). The switching list is NOT present locally, so only the lpwo_v2 overlap is verifiable.

| set | states | world in lpwo_v2 | run in lpwo_v2 | ctrl-seen flag | world-seen flag | world-disjoint states |
|---|---:|---:|---:|---:|---:|---:|
| wall_same | 67 | 67 | 0 | 0 | 67 | 0 |
| wall_pilot | 53 | 51 | 0 | 0 | 51 | 2 |
| seen_controls | 217 | 203 | 203 | 217 | 203 | 14 |

**Overlap verdict.** Every wall_same state sits in a *world* that appears in the lpwo_v2 training list, but no wall *run* does (the wall controller is left out by construction); `controller_seen_in_training` is 0 for all wall states. So the wall and PI controllers are unseen while the scenarios are in-distribution. seen_controls is genuinely seen (203/217 runs in lpwo_v2). The world-disjoint subsets are tiny: wall_same 0, wall_pilot 2, seen_controls 14. The switching-list overlap is unverifiable locally.

## Sets (wall_same / wall_pilot / seen_controls)

| set | states | worlds | world-disjoint states |
|---|---:|---:|---:|
| wall_same | 67 | 29 | 0 |
| wall_pilot | 53 | 27 | 2 |
| seen_controls | 217 | 31 | 14 |

## HDV |ds| (m) - median / IQR / p90 across states

### `wall_same` (arm `wall`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| brake | 10s | 67 | 3.88 | 0.99 | 5.20 |
| brake | 30s | 67 | 16.48 | 8.40 | 25.39 |
| brake | 60s | 67 | 49.65 | 33.76 | 85.09 |
| brake | 120s | 66 | 187.39 | 199.44 | 501.85 |
| nobrake | 10s | 67 | 2.18 | 0.75 | 2.88 |
| nobrake | 30s | 67 | 8.60 | 5.44 | 16.13 |
| nobrake | 60s | 67 | 42.82 | 38.63 | 77.75 |
| nobrake | 120s | 66 | 179.42 | 310.74 | 794.34 |
| struct | 10s | 67 | 2.04 | 0.85 | 2.89 |
| struct | 30s | 67 | 5.07 | 3.01 | 10.54 |
| struct | 60s | 67 | 8.71 | 10.40 | 27.54 |
| struct | 120s | 66 | 99.84 | 84.96 | 217.67 |
| blackbox_traj | 10s | 67 | 2.78 | 0.95 | 4.21 |
| blackbox_traj | 30s | 67 | 9.79 | 4.76 | 15.81 |
| blackbox_traj | 60s | 67 | 25.41 | 20.12 | 38.86 |
| blackbox_traj | 120s | 66 | 152.46 | 174.96 | 456.55 |
| blackbox_id | 10s | 67 | 2.78 | 1.88 | 4.67 |
| blackbox_id | 30s | 67 | 9.66 | 9.14 | 27.00 |
| blackbox_id | 60s | 67 | 40.07 | 53.11 | 114.88 |
| blackbox_id | 120s | 66 | 245.44 | 212.61 | 519.52 |
| pure_physics | 10s | 67 | 7.51 | 1.41 | 8.93 |
| pure_physics | 30s | 67 | 29.44 | 9.16 | 42.16 |
| pure_physics | 60s | 67 | 66.06 | 68.89 | 175.42 |
| pure_physics | 120s | 64 | 671.56 | 539.91 | 1252.54 |

### `wall_pilot` (arm `wall`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| brake | 10s | 53 | 3.73 | 0.83 | 4.50 |
| brake | 30s | 53 | 15.25 | 6.81 | 22.58 |
| brake | 60s | 53 | 44.54 | 34.27 | 75.65 |
| brake | 120s | 52 | 142.66 | 223.54 | 343.16 |
| nobrake | 10s | 53 | 2.13 | 0.93 | 2.83 |
| nobrake | 30s | 53 | 8.41 | 5.48 | 16.71 |
| nobrake | 60s | 53 | 38.98 | 32.67 | 69.73 |
| nobrake | 120s | 52 | 179.42 | 263.17 | 580.36 |
| struct | 10s | 53 | 2.06 | 0.81 | 2.73 |
| struct | 30s | 53 | 5.06 | 3.34 | 9.77 |
| struct | 60s | 53 | 8.34 | 10.09 | 24.80 |
| struct | 120s | 52 | 96.76 | 102.85 | 241.01 |
| blackbox_traj | 10s | 53 | 2.76 | 1.00 | 3.42 |
| blackbox_traj | 30s | 53 | 9.79 | 3.85 | 13.38 |
| blackbox_traj | 60s | 53 | 20.85 | 17.49 | 35.85 |
| blackbox_traj | 120s | 52 | 147.03 | 150.76 | 312.61 |
| blackbox_id | 10s | 53 | 2.78 | 1.61 | 4.05 |
| blackbox_id | 30s | 53 | 10.15 | 8.99 | 27.51 |
| blackbox_id | 60s | 53 | 40.07 | 53.93 | 122.45 |
| blackbox_id | 120s | 52 | 229.75 | 207.43 | 547.66 |
| pure_physics | 10s | 53 | 7.42 | 1.40 | 8.68 |
| pure_physics | 30s | 53 | 30.55 | 8.79 | 40.36 |
| pure_physics | 60s | 53 | 83.64 | 68.32 | 204.89 |
| pure_physics | 120s | 50 | 671.56 | 455.14 | 1170.69 |

### `seen_controls` (arm `native`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| brake | 10s | 217 | 3.67 | 1.39 | 5.02 |
| brake | 30s | 217 | 18.38 | 10.81 | 35.52 |
| brake | 60s | 217 | 70.09 | 54.15 | 162.24 |
| brake | 120s | 212 | 256.73 | 276.81 | 556.18 |
| nobrake | 10s | 217 | 2.42 | 1.63 | 4.68 |
| nobrake | 30s | 217 | 13.68 | 15.31 | 34.68 |
| nobrake | 60s | 217 | 64.11 | 66.29 | 143.53 |
| nobrake | 120s | 212 | 392.46 | 376.35 | 902.90 |
| struct | 10s | 217 | 2.24 | 1.34 | 4.85 |
| struct | 30s | 217 | 7.51 | 11.61 | 26.68 |
| struct | 60s | 217 | 26.27 | 65.89 | 110.92 |
| struct | 120s | 212 | 189.45 | 270.33 | 503.31 |
| blackbox_traj | 10s | 217 | 3.24 | 2.69 | 7.00 |
| blackbox_traj | 30s | 217 | 14.62 | 21.98 | 49.21 |
| blackbox_traj | 60s | 217 | 44.64 | 81.47 | 158.07 |
| blackbox_traj | 120s | 212 | 192.86 | 199.91 | 460.90 |
| blackbox_id | 10s | 217 | 2.90 | 4.00 | 12.47 |
| blackbox_id | 30s | 217 | 19.79 | 66.73 | 125.94 |
| blackbox_id | 60s | 217 | 107.09 | 202.81 | 370.63 |
| blackbox_id | 120s | 212 | 313.86 | 386.26 | 741.29 |
| pure_physics | 10s | 217 | 7.83 | 2.81 | 13.70 |
| pure_physics | 30s | 217 | 36.61 | 55.35 | 126.45 |
| pure_physics | 60s | 217 | 154.05 | 268.28 | 459.53 |
| pure_physics | 120s | 207 | 1080.28 | 434.90 | 1463.03 |

## HDV |dv| (m/s) - median / IQR / p90 across states

### `wall_same` (arm `wall`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| brake | 10s | 67 | 0.47 | 0.20 | 0.71 |
| brake | 30s | 67 | 1.14 | 0.47 | 1.52 |
| brake | 60s | 67 | 2.63 | 2.03 | 4.58 |
| brake | 120s | 66 | 10.60 | 12.03 | 22.79 |
| nobrake | 10s | 67 | 0.33 | 0.11 | 0.41 |
| nobrake | 30s | 67 | 0.70 | 0.28 | 1.09 |
| nobrake | 60s | 67 | 2.44 | 2.22 | 4.29 |
| nobrake | 120s | 66 | 4.42 | 3.80 | 12.64 |
| struct | 10s | 67 | 0.27 | 0.14 | 0.44 |
| struct | 30s | 67 | 0.40 | 0.20 | 0.59 |
| struct | 60s | 67 | 0.72 | 0.41 | 1.15 |
| struct | 120s | 66 | 6.94 | 12.61 | 22.92 |
| blackbox_traj | 10s | 67 | 0.41 | 0.15 | 0.67 |
| blackbox_traj | 30s | 67 | 0.63 | 0.36 | 0.95 |
| blackbox_traj | 60s | 67 | 1.04 | 0.67 | 1.80 |
| blackbox_traj | 120s | 66 | 13.04 | 15.37 | 25.28 |
| blackbox_id | 10s | 67 | 0.55 | 0.19 | 0.81 |
| blackbox_id | 30s | 67 | 0.85 | 0.57 | 1.65 |
| blackbox_id | 60s | 67 | 1.73 | 3.22 | 7.92 |
| blackbox_id | 120s | 66 | 15.45 | 14.56 | 25.27 |
| pure_physics | 10s | 67 | 1.13 | 0.26 | 1.64 |
| pure_physics | 30s | 67 | 1.36 | 1.02 | 2.56 |
| pure_physics | 60s | 67 | 3.44 | 4.18 | 8.65 |
| pure_physics | 120s | 64 | 6.14 | 8.01 | 19.77 |

### `wall_pilot` (arm `wall`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| brake | 10s | 53 | 0.44 | 0.15 | 0.61 |
| brake | 30s | 53 | 1.10 | 0.51 | 1.49 |
| brake | 60s | 53 | 2.23 | 1.58 | 4.22 |
| brake | 120s | 52 | 11.31 | 12.42 | 23.17 |
| nobrake | 10s | 53 | 0.33 | 0.12 | 0.41 |
| nobrake | 30s | 53 | 0.70 | 0.28 | 1.32 |
| nobrake | 60s | 53 | 2.18 | 2.31 | 4.33 |
| nobrake | 120s | 52 | 4.37 | 2.76 | 9.71 |
| struct | 10s | 53 | 0.24 | 0.12 | 0.43 |
| struct | 30s | 53 | 0.37 | 0.17 | 0.54 |
| struct | 60s | 53 | 0.69 | 0.41 | 1.31 |
| struct | 120s | 52 | 7.12 | 11.40 | 22.57 |
| blackbox_traj | 10s | 53 | 0.42 | 0.15 | 0.67 |
| blackbox_traj | 30s | 53 | 0.59 | 0.37 | 0.96 |
| blackbox_traj | 60s | 53 | 0.89 | 0.62 | 1.77 |
| blackbox_traj | 120s | 52 | 14.18 | 12.55 | 25.59 |
| blackbox_id | 10s | 53 | 0.55 | 0.16 | 0.75 |
| blackbox_id | 30s | 53 | 0.86 | 0.60 | 1.72 |
| blackbox_id | 60s | 53 | 2.33 | 3.27 | 9.02 |
| blackbox_id | 120s | 52 | 15.48 | 9.68 | 24.91 |
| pure_physics | 10s | 53 | 1.13 | 0.26 | 1.48 |
| pure_physics | 30s | 53 | 1.51 | 1.13 | 2.59 |
| pure_physics | 60s | 53 | 4.49 | 4.21 | 10.61 |
| pure_physics | 120s | 50 | 6.14 | 6.06 | 15.27 |

### `seen_controls` (arm `native`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| brake | 10s | 217 | 0.53 | 0.26 | 0.91 |
| brake | 30s | 217 | 1.51 | 1.26 | 3.61 |
| brake | 60s | 217 | 3.84 | 4.26 | 8.85 |
| brake | 120s | 212 | 7.57 | 7.64 | 16.18 |
| nobrake | 10s | 217 | 0.37 | 0.28 | 0.87 |
| nobrake | 30s | 217 | 1.29 | 1.77 | 3.83 |
| nobrake | 60s | 217 | 4.47 | 3.93 | 7.72 |
| nobrake | 120s | 212 | 7.61 | 6.83 | 16.51 |
| struct | 10s | 217 | 0.32 | 0.25 | 0.82 |
| struct | 30s | 217 | 0.63 | 1.33 | 2.88 |
| struct | 60s | 217 | 1.73 | 4.04 | 6.51 |
| struct | 120s | 212 | 7.14 | 8.31 | 17.05 |
| blackbox_traj | 10s | 217 | 0.58 | 0.52 | 1.48 |
| blackbox_traj | 30s | 217 | 1.08 | 2.35 | 4.02 |
| blackbox_traj | 60s | 217 | 2.43 | 4.73 | 7.63 |
| blackbox_traj | 120s | 212 | 7.52 | 7.36 | 18.03 |
| blackbox_id | 10s | 217 | 0.63 | 0.59 | 2.58 |
| blackbox_id | 30s | 217 | 1.34 | 4.19 | 7.60 |
| blackbox_id | 60s | 217 | 3.91 | 6.51 | 10.64 |
| blackbox_id | 120s | 212 | 7.87 | 11.69 | 20.44 |
| pure_physics | 10s | 217 | 1.28 | 0.62 | 3.03 |
| pure_physics | 30s | 217 | 2.19 | 4.03 | 8.75 |
| pure_physics | 60s | 217 | 4.83 | 6.89 | 12.25 |
| pure_physics | 120s | 207 | 13.81 | 13.37 | 23.27 |

## Blow-up rate (HDV median |ds| > 50 m @60 s and > 100 m @120 s)

| set | model | rate | blowups / evaluated | excluded |
|---|---|---:|---:|---:|
| wall_same | brake | 0.379 | 25/66 | 1 |
| wall_same | nobrake | 0.273 | 18/66 | 1 |
| wall_same | struct | 0.000 | 0/66 | 1 |
| wall_same | blackbox_traj | 0.015 | 1/66 | 1 |
| wall_same | blackbox_id | 0.379 | 25/66 | 1 |
| wall_same | pure_physics | 0.688 | 44/64 | 3 |
| wall_pilot | brake | 0.308 | 16/52 | 1 |
| wall_pilot | nobrake | 0.250 | 13/52 | 1 |
| wall_pilot | struct | 0.000 | 0/52 | 1 |
| wall_pilot | blackbox_traj | 0.019 | 1/52 | 1 |
| wall_pilot | blackbox_id | 0.404 | 21/52 | 1 |
| wall_pilot | pure_physics | 0.740 | 37/50 | 3 |
| seen_controls | brake | 0.651 | 138/212 | 5 |
| seen_controls | nobrake | 0.632 | 134/212 | 5 |
| seen_controls | struct | 0.321 | 68/212 | 5 |
| seen_controls | blackbox_traj | 0.467 | 99/212 | 5 |
| seen_controls | blackbox_id | 0.594 | 126/212 | 5 |
| seen_controls | pure_physics | 0.889 | 184/207 | 10 |

## World-disjoint subset HDV |ds| (60/120 s)

| set | model | n | 60s median | 120s median |
|---|---|---:|---:|---:|
| wall_same | brake | 0 | n/a | n/a |
| wall_same | nobrake | 0 | n/a | n/a |
| wall_same | struct | 0 | n/a | n/a |
| wall_same | blackbox_traj | 0 | n/a | n/a |
| wall_same | blackbox_id | 0 | n/a | n/a |
| wall_same | pure_physics | 0 | n/a | n/a |
| wall_pilot | brake | 2 | 51.71 | 156.49 |
| wall_pilot | nobrake | 2 | 64.05 | 501.97 |
| wall_pilot | struct | 2 | 16.31 | 184.42 |
| wall_pilot | blackbox_traj | 2 | 43.79 | 313.11 |
| wall_pilot | blackbox_id | 2 | 39.06 | 175.67 |
| wall_pilot | pure_physics | 2 | 122.51 | 1013.52 |
| seen_controls | brake | 14 | 68.81 | 155.90 |
| seen_controls | nobrake | 14 | 47.18 | 643.51 |
| seen_controls | struct | 14 | 14.63 | 53.70 |
| seen_controls | blackbox_traj | 14 | 35.03 | 264.83 |
| seen_controls | blackbox_id | 14 | 30.92 | 78.64 |
| seen_controls | pure_physics | 14 | 135.37 | 1247.25 |

## Paired world bootstrap of HDV |ds| (model - reference; negative favours model)

| comparison | set | h | difference [95% CI over worlds] | worlds | pairs |
|---|---|---|---|---|---:|
| brake-nobrake | wall_same | 10s | 1.74 [1.37, 1.99] | 29 | 67 |
| brake-nobrake | wall_same | 30s | 7.53 [4.10, 10.36] | 29 | 67 |
| brake-nobrake | wall_same | 60s | 9.39 [0.62, 17.15] | 29 | 67 |
| brake-nobrake | wall_same | 120s | -15.26 [-99.74, 69.98] | 29 | 66 |
| brake-nobrake | wall_pilot | 10s | 1.67 [1.36, 1.89] | 27 | 53 |
| brake-nobrake | wall_pilot | 30s | 5.79 [3.52, 8.25] | 27 | 53 |
| brake-nobrake | wall_pilot | 60s | 6.32 [-0.57, 16.59] | 27 | 53 |
| brake-nobrake | wall_pilot | 120s | -6.71 [-70.62, 66.13] | 27 | 52 |
| brake-nobrake | seen_controls | 10s | 1.12 [0.90, 1.41] | 31 | 217 |
| brake-nobrake | seen_controls | 30s | 4.48 [2.84, 6.46] | 31 | 217 |
| brake-nobrake | seen_controls | 60s | 10.90 [5.40, 16.84] | 31 | 217 |
| brake-nobrake | seen_controls | 120s | -87.16 [-223.31, -22.53] | 31 | 212 |
| brake-struct | wall_same | 10s | 1.60 [1.35, 2.02] | 29 | 67 |
| brake-struct | wall_same | 30s | 11.28 [8.75, 13.30] | 29 | 67 |
| brake-struct | wall_same | 60s | 35.63 [25.66, 45.89] | 29 | 67 |
| brake-struct | wall_same | 120s | 76.81 [39.54, 145.81] | 29 | 66 |
| brake-struct | wall_pilot | 10s | 1.58 [1.31, 1.70] | 27 | 53 |
| brake-struct | wall_pilot | 30s | 9.60 [8.61, 13.00] | 27 | 53 |
| brake-struct | wall_pilot | 60s | 29.84 [19.04, 45.89] | 27 | 53 |
| brake-struct | wall_pilot | 120s | 72.08 [5.84, 104.73] | 27 | 52 |
| brake-struct | seen_controls | 10s | 1.15 [0.97, 1.47] | 31 | 217 |
| brake-struct | seen_controls | 30s | 9.40 [8.41, 10.92] | 31 | 217 |
| brake-struct | seen_controls | 60s | 37.63 [31.57, 44.26] | 31 | 217 |
| brake-struct | seen_controls | 120s | 55.02 [41.40, 73.44] | 31 | 212 |
| nobrake-struct | wall_same | 10s | 0.14 [-0.14, 0.31] | 29 | 67 |
| nobrake-struct | wall_same | 30s | 4.41 [0.77, 5.50] | 29 | 67 |
| nobrake-struct | wall_same | 60s | 26.73 [13.48, 37.47] | 29 | 67 |
| nobrake-struct | wall_same | 120s | 72.49 [4.84, 182.97] | 29 | 66 |
| nobrake-struct | wall_pilot | 10s | 0.11 [-0.14, 0.31] | 27 | 53 |
| nobrake-struct | wall_pilot | 30s | 3.66 [1.11, 5.54] | 27 | 53 |
| nobrake-struct | wall_pilot | 60s | 19.13 [12.62, 36.70] | 27 | 53 |
| nobrake-struct | wall_pilot | 120s | 69.82 [5.20, 136.01] | 27 | 52 |
| nobrake-struct | seen_controls | 10s | 0.16 [0.00, 0.27] | 31 | 217 |
| nobrake-struct | seen_controls | 30s | 3.94 [2.56, 5.06] | 31 | 217 |
| nobrake-struct | seen_controls | 60s | 26.48 [19.43, 33.32] | 31 | 217 |
| nobrake-struct | seen_controls | 120s | 138.00 [67.68, 226.48] | 31 | 212 |

## STEP 4 - braking diagnostics (mean over states; arm `wall`/`native`)

`speed_bias` = mean (pred - recorded) HDV speed (m/s; >0 = model too fast). `hard share` = fraction of matched HDVs with a<-3 m/s^2 at any time in [t0,t0+h]. `recall` = share of recorded <5 m/s HDVs the model also places <5 m/s at the horizon.

| model | set | h | n | speed_bias | pred hard | rec hard | pred slow | rec slow | slow recall |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|

## STEP 5 - paired wall-vs-native ranking

| model | set | h | n | sign agree | Spearman | model regret s [CI] | always-wall | always-native |
|---|---|---|---:|---|---:|---|---:|---:|
| brake | wall_same | 60.0s | 16 | 1.00 | 0.99 | 0.00 [0.00, 0.00] | 1.51 | 0.00 |
| brake | wall_same | 120.0s | 16 | 1.00 | 0.99 | 0.00 [0.00, 0.00] | 15.38 | 39.48 |
| brake | wall_pilot | 60.0s | 7 | 1.00 | 0.89 | 0.00 [0.00, 0.00] | 12.15 | 0.00 |
| brake | wall_pilot | 120.0s | 7 | 0.86 | 0.86 | 10.73 [0.00, 32.18] | 35.15 | 56.99 |
| nobrake | wall_same | 60.0s | 16 | 1.00 | 0.99 | 0.00 [0.00, 0.00] | 1.51 | 0.00 |
| nobrake | wall_same | 120.0s | 16 | 0.88 | 0.72 | 6.13 [0.00, 15.95] | 15.38 | 39.48 |
| nobrake | wall_pilot | 60.0s | 7 | 1.00 | 0.89 | 0.00 [0.00, 0.00] | 12.15 | 0.00 |
| nobrake | wall_pilot | 120.0s | 7 | 0.57 | 0.43 | 24.73 [5.55, 49.09] | 35.15 | 56.99 |
| struct | wall_same | 60.0s | 16 | 1.00 | 0.99 | 0.00 [0.00, 0.00] | 1.51 | 0.00 |
| struct | wall_same | 120.0s | 16 | 0.94 | 0.88 | 2.43 [0.00, 7.28] | 15.38 | 39.48 |
| struct | wall_pilot | 60.0s | 7 | 1.00 | 0.89 | 0.00 [0.00, 0.00] | 12.15 | 0.00 |
| struct | wall_pilot | 120.0s | 7 | 0.86 | 1.00 | 5.55 [0.00, 16.64] | 35.15 | 56.99 |

## Verdict

**Braking fix (BRAKE vs NOBRAKE).** Blow-up rate:
- `wall_same`: NOBRAKE 0.273 (18/66) -> BRAKE 0.379 (25/66).
- `wall_pilot`: NOBRAKE 0.250 (13/52) -> BRAKE 0.308 (16/52).
- `seen_controls`: NOBRAKE 0.632 (134/212) -> BRAKE 0.651 (138/212).

Paired BRAKE-NOBRAKE HDV |ds| (negative favours BRAKE):
- `wall_same`: 10s 1.74 [1.37, 1.99] ; 30s 7.53 [4.10, 10.36] ; 60s 9.39 [0.62, 17.15] ; 120s -15.26 [-99.74, 69.98]
- `wall_pilot`: 10s 1.67 [1.36, 1.89] ; 30s 5.79 [3.52, 8.25] ; 60s 6.32 [-0.57, 16.59] ; 120s -6.71 [-70.62, 66.13]
- `seen_controls`: 10s 1.12 [0.90, 1.41] ; 30s 4.48 [2.84, 6.46] ; 60s 10.90 [5.40, 16.84] ; 120s -87.16 [-223.31, -22.53]

**New vs old C1 struct on the unseen wall cohorts** (paired BRAKE-struct / NOBRAKE-struct; negative favours the new model):
- `wall_same` 60s: BRAKE-struct 35.63 [25.66, 45.89] | NOBRAKE-struct 26.73 [13.48, 37.47] | 120s: BRAKE-struct 76.81 [39.54, 145.81] | NOBRAKE-struct 72.49 [4.84, 182.97]
- `wall_pilot` 60s: BRAKE-struct 29.84 [19.04, 45.89] | NOBRAKE-struct 19.13 [12.62, 36.70] | 120s: BRAKE-struct 72.08 [5.84, 104.73] | NOBRAKE-struct 69.82 [5.20, 136.01]

**Caveats.** (1) The new models were trained leave-PI-and-wall-out, but every wall_same/wall_pilot world is present in the lpwo_v2 list; only the wall controller is unseen. (2) The switching training list is absent locally, so switching overlap is unverified. (3) The two new models and the old C1 struct use different TRAIN_RECEIPT physics (main2 merge_cal=true vs C1 merge_cal=false), so the cross-model contrast is not a pure architecture comparison. (4) The world-disjoint subsets are tiny (0/2/14 states).

