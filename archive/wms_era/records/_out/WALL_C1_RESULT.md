# WALL-C1 RESULT - C1 world models on the unseen WALL controller

Generated 2026-10-08T21:40:27Z. CPU only, no training, no SUMO. The three C1 checkpoints were trained on pack_dspeed + pack_damping only (pack_v3 contributed 0 windows), so a wall is an unseen group controller for all of them. Each checkpoint is scored with its own TRAIN_RECEIPT physics (sumo_fidelity=true); CAVs are teacher-forced from the recorded arm so wall members replay the recorded wall motion; entries OFF.

Sets (wall_same and wall_pilot overlap in worlds; never summed):

| set | states | worlds |
|---|---:|---:|
| wall_same | 67 | 29 |
| wall_pilot | 53 | 27 |
| seen_controls | 217 | 31 |

## HDV |ds| on `wall_same` (arm `wall`)

| model | h | n | median | IQR | p90 | blow-up |
|---|---|---:|---:|---:|---:|---:|
| struct | 10s | 67 | 2.04 | 0.85 | 2.89 | - |
| struct | 30s | 67 | 5.07 | 3.01 | 10.54 | - |
| struct | 60s | 67 | 8.71 | 10.40 | 27.54 | - |
| struct | 120s | 66 | 99.84 | 84.96 | 217.67 | 0.000 |
| blackbox_traj | 10s | 67 | 2.78 | 0.95 | 4.21 | - |
| blackbox_traj | 30s | 67 | 9.79 | 4.76 | 15.81 | - |
| blackbox_traj | 60s | 67 | 25.41 | 20.12 | 38.86 | - |
| blackbox_traj | 120s | 66 | 152.46 | 174.96 | 456.55 | 0.015 |
| blackbox_id | 10s | 67 | 2.78 | 1.88 | 4.67 | - |
| blackbox_id | 30s | 67 | 9.66 | 9.14 | 27.00 | - |
| blackbox_id | 60s | 67 | 40.07 | 53.11 | 114.88 | - |
| blackbox_id | 120s | 66 | 245.44 | 212.61 | 519.52 | 0.379 |
| pure_physics | 10s | 67 | 7.51 | 1.41 | 8.93 | - |
| pure_physics | 30s | 67 | 29.44 | 9.16 | 42.16 | - |
| pure_physics | 60s | 67 | 66.06 | 68.89 | 175.42 | - |
| pure_physics | 120s | 64 | 671.56 | 539.91 | 1252.54 | 0.688 |
| keep_speed | 10s | 67 | 4.36 | 3.55 | 8.01 | - |
| keep_speed | 30s | 67 | 25.53 | 21.73 | 51.52 | - |
| keep_speed | 60s | 67 | 103.64 | 83.08 | 259.22 | - |
| keep_speed | 120s | 66 | 968.58 | 577.35 | 1495.28 | 0.939 |

## HDV |ds| on `wall_pilot` (arm `wall`)

| model | h | n | median | IQR | p90 | blow-up |
|---|---|---:|---:|---:|---:|---:|
| struct | 10s | 53 | 2.06 | 0.81 | 2.73 | - |
| struct | 30s | 53 | 5.06 | 3.34 | 9.77 | - |
| struct | 60s | 53 | 8.34 | 10.09 | 24.80 | - |
| struct | 120s | 52 | 96.76 | 102.85 | 241.01 | 0.000 |
| blackbox_traj | 10s | 53 | 2.76 | 1.00 | 3.42 | - |
| blackbox_traj | 30s | 53 | 9.79 | 3.85 | 13.38 | - |
| blackbox_traj | 60s | 53 | 20.85 | 17.49 | 35.85 | - |
| blackbox_traj | 120s | 52 | 147.03 | 150.76 | 312.61 | 0.019 |
| blackbox_id | 10s | 53 | 2.78 | 1.61 | 4.05 | - |
| blackbox_id | 30s | 53 | 10.15 | 8.99 | 27.51 | - |
| blackbox_id | 60s | 53 | 40.07 | 53.93 | 122.45 | - |
| blackbox_id | 120s | 52 | 229.75 | 207.43 | 547.66 | 0.404 |
| pure_physics | 10s | 53 | 7.42 | 1.40 | 8.68 | - |
| pure_physics | 30s | 53 | 30.55 | 8.79 | 40.36 | - |
| pure_physics | 60s | 53 | 83.64 | 68.32 | 204.89 | - |
| pure_physics | 120s | 50 | 671.56 | 455.14 | 1170.69 | 0.740 |
| keep_speed | 10s | 53 | 4.41 | 3.39 | 7.05 | - |
| keep_speed | 30s | 53 | 25.67 | 18.31 | 48.92 | - |
| keep_speed | 60s | 53 | 111.96 | 112.48 | 271.06 | - |
| keep_speed | 120s | 52 | 874.50 | 472.45 | 1317.54 | 0.923 |

## HDV |ds| on `seen_controls` (arm `native`)

| model | h | n | median | IQR | p90 | blow-up |
|---|---|---:|---:|---:|---:|---:|
| struct | 10s | 217 | 2.24 | 1.34 | 4.85 | - |
| struct | 30s | 217 | 7.51 | 11.61 | 26.68 | - |
| struct | 60s | 217 | 26.27 | 65.89 | 110.92 | - |
| struct | 120s | 212 | 189.45 | 270.33 | 503.31 | 0.321 |
| blackbox_traj | 10s | 217 | 3.24 | 2.69 | 7.00 | - |
| blackbox_traj | 30s | 217 | 14.62 | 21.98 | 49.21 | - |
| blackbox_traj | 60s | 217 | 44.64 | 81.47 | 158.07 | - |
| blackbox_traj | 120s | 212 | 192.86 | 199.91 | 460.90 | 0.467 |
| blackbox_id | 10s | 217 | 2.90 | 4.00 | 12.47 | - |
| blackbox_id | 30s | 217 | 19.79 | 66.73 | 125.94 | - |
| blackbox_id | 60s | 217 | 107.09 | 202.81 | 370.63 | - |
| blackbox_id | 120s | 212 | 313.86 | 386.26 | 741.29 | 0.594 |
| pure_physics | 10s | 217 | 7.83 | 2.81 | 13.70 | - |
| pure_physics | 30s | 217 | 36.61 | 55.35 | 126.45 | - |
| pure_physics | 60s | 217 | 154.05 | 268.28 | 459.53 | - |
| pure_physics | 120s | 207 | 1080.28 | 434.90 | 1463.03 | 0.889 |
| keep_speed | 10s | 217 | 4.88 | 7.14 | 18.64 | - |
| keep_speed | 30s | 217 | 48.61 | 107.77 | 187.19 | - |
| keep_speed | 60s | 217 | 219.31 | 310.47 | 505.28 | - |
| keep_speed | 120s | 212 | 1284.12 | 483.96 | 1749.67 | 0.920 |

## Unseen(wall)/seen HDV |ds| ratio (negative metric: <1 means wall is easier)

| model | wall set | h | ratio [95% CI over worlds] |
|---|---|---|---|
| struct | wall_same | 10s | 0.91 [0.74, 1.06] |
| struct | wall_same | 30s | 0.68 [0.51, 0.87] |
| struct | wall_same | 60s | 0.33 [0.26, 0.58] |
| struct | wall_same | 120s | 0.53 [0.41, 0.73] |
| struct | wall_pilot | 10s | 0.92 [0.72, 1.04] |
| struct | wall_pilot | 30s | 0.67 [0.48, 0.88] |
| struct | wall_pilot | 60s | 0.32 [0.23, 0.54] |
| struct | wall_pilot | 120s | 0.51 [0.35, 0.74] |
| blackbox_traj | wall_same | 10s | 0.86 [0.73, 1.00] |
| blackbox_traj | wall_same | 30s | 0.67 [0.53, 0.82] |
| blackbox_traj | wall_same | 60s | 0.57 [0.36, 0.75] |
| blackbox_traj | wall_same | 120s | 0.79 [0.54, 1.24] |
| blackbox_traj | wall_pilot | 10s | 0.85 [0.70, 0.99] |
| blackbox_traj | wall_pilot | 30s | 0.67 [0.54, 0.81] |
| blackbox_traj | wall_pilot | 60s | 0.47 [0.34, 0.70] |
| blackbox_traj | wall_pilot | 120s | 0.76 [0.53, 1.09] |
| blackbox_id | wall_same | 10s | 0.96 [0.62, 1.29] |
| blackbox_id | wall_same | 30s | 0.49 [0.29, 0.91] |
| blackbox_id | wall_same | 60s | 0.37 [0.27, 0.71] |
| blackbox_id | wall_same | 120s | 0.78 [0.60, 1.15] |
| blackbox_id | wall_pilot | 10s | 0.96 [0.64, 1.29] |
| blackbox_id | wall_pilot | 30s | 0.51 [0.30, 0.92] |
| blackbox_id | wall_pilot | 60s | 0.37 [0.28, 0.74] |
| blackbox_id | wall_pilot | 120s | 0.73 [0.58, 1.10] |
| pure_physics | wall_same | 10s | 0.96 [0.84, 1.02] |
| pure_physics | wall_same | 30s | 0.80 [0.68, 0.92] |
| pure_physics | wall_same | 60s | 0.43 [0.31, 0.57] |
| pure_physics | wall_same | 120s | 0.62 [0.49, 0.84] |
| pure_physics | wall_pilot | 10s | 0.95 [0.83, 1.01] |
| pure_physics | wall_pilot | 30s | 0.83 [0.72, 0.95] |
| pure_physics | wall_pilot | 60s | 0.54 [0.36, 0.77] |
| pure_physics | wall_pilot | 120s | 0.62 [0.52, 0.83] |
| keep_speed | wall_same | 10s | 0.89 [0.54, 1.29] |
| keep_speed | wall_same | 30s | 0.53 [0.39, 0.88] |
| keep_speed | wall_same | 60s | 0.47 [0.39, 0.77] |
| keep_speed | wall_same | 120s | 0.75 [0.65, 0.95] |
| keep_speed | wall_pilot | 10s | 0.90 [0.55, 1.30] |
| keep_speed | wall_pilot | 30s | 0.53 [0.41, 0.88] |
| keep_speed | wall_pilot | 60s | 0.51 [0.39, 0.85] |
| keep_speed | wall_pilot | 120s | 0.68 [0.60, 0.78] |

## Paired HDV |ds| difference vs C1 struct (negative favours the model)

Pairs are matched on the full identity (world, key, source_arm, t0_s); the bootstrap resamples whole worlds (2,000 resamples, seed 0), so distinct states sharing one (world, t0_s) are not collapsed.

| model | set | h | difference [95% CI over worlds] | worlds | pairs |
|---|---|---|---|---|---:|
| blackbox_traj | wall_same | 10s | 0.78 [0.28, 1.03] | 29 | 67 |
| blackbox_traj | wall_same | 30s | 4.59 [2.72, 5.81] | 29 | 67 |
| blackbox_traj | wall_same | 60s | 9.98 [5.09, 16.78] | 29 | 67 |
| blackbox_traj | wall_same | 120s | 49.21 [7.31, 110.05] | 29 | 66 |
| blackbox_traj | wall_pilot | 10s | 0.73 [0.24, 1.00] | 27 | 53 |
| blackbox_traj | wall_pilot | 30s | 4.58 [2.55, 5.26] | 27 | 53 |
| blackbox_traj | wall_pilot | 60s | 10.71 [5.37, 14.81] | 27 | 53 |
| blackbox_traj | wall_pilot | 120s | 38.92 [-0.60, 82.12] | 27 | 52 |
| blackbox_traj | seen_controls | 10s | 0.83 [0.61, 1.05] | 31 | 217 |
| blackbox_traj | seen_controls | 30s | 6.12 [5.08, 7.31] | 31 | 217 |
| blackbox_traj | seen_controls | 60s | 12.03 [9.60, 15.92] | 31 | 217 |
| blackbox_traj | seen_controls | 120s | 22.72 [-7.07, 42.05] | 31 | 212 |
| blackbox_id | wall_same | 10s | 0.72 [0.15, 1.28] | 29 | 67 |
| blackbox_id | wall_same | 30s | 5.08 [3.12, 8.05] | 29 | 67 |
| blackbox_id | wall_same | 60s | 31.78 [18.61, 45.37] | 29 | 67 |
| blackbox_id | wall_same | 120s | 165.10 [99.80, 217.17] | 29 | 66 |
| blackbox_id | wall_pilot | 10s | 0.72 [0.26, 1.25] | 27 | 53 |
| blackbox_id | wall_pilot | 30s | 5.08 [3.79, 8.05] | 27 | 53 |
| blackbox_id | wall_pilot | 60s | 35.69 [23.66, 45.51] | 27 | 53 |
| blackbox_id | wall_pilot | 120s | 141.73 [99.78, 216.80] | 27 | 52 |
| blackbox_id | seen_controls | 10s | 0.76 [0.48, 1.35] | 31 | 217 |
| blackbox_id | seen_controls | 30s | 11.97 [7.65, 26.28] | 31 | 217 |
| blackbox_id | seen_controls | 60s | 50.44 [36.60, 83.58] | 31 | 217 |
| blackbox_id | seen_controls | 120s | 99.08 [74.30, 137.59] | 31 | 212 |
| pure_physics | wall_same | 10s | 5.40 [5.14, 5.88] | 29 | 67 |
| pure_physics | wall_same | 30s | 24.08 [21.21, 26.15] | 29 | 67 |
| pure_physics | wall_same | 60s | 54.90 [41.46, 77.61] | 29 | 67 |
| pure_physics | wall_same | 120s | 561.90 [426.22, 764.42] | 28 | 64 |
| pure_physics | wall_pilot | 10s | 5.31 [4.74, 6.05] | 27 | 53 |
| pure_physics | wall_pilot | 30s | 24.78 [23.57, 29.20] | 27 | 53 |
| pure_physics | wall_pilot | 60s | 70.22 [46.28, 102.19] | 27 | 53 |
| pure_physics | wall_pilot | 120s | 561.90 [435.70, 751.06] | 26 | 50 |
| pure_physics | seen_controls | 10s | 5.79 [5.36, 6.11] | 31 | 217 |
| pure_physics | seen_controls | 30s | 30.88 [28.14, 33.45] | 31 | 217 |
| pure_physics | seen_controls | 60s | 119.15 [101.73, 143.92] | 31 | 217 |
| pure_physics | seen_controls | 120s | 837.98 [757.48, 910.47] | 31 | 207 |
| keep_speed | wall_same | 10s | 1.96 [1.26, 3.50] | 29 | 67 |
| keep_speed | wall_same | 30s | 19.29 [14.28, 31.39] | 29 | 67 |
| keep_speed | wall_same | 60s | 95.78 [69.69, 112.58] | 29 | 67 |
| keep_speed | wall_same | 120s | 843.15 [721.67, 1142.34] | 29 | 66 |
| keep_speed | wall_pilot | 10s | 1.96 [1.41, 3.26] | 27 | 53 |
| keep_speed | wall_pilot | 30s | 21.27 [16.66, 25.58] | 27 | 53 |
| keep_speed | wall_pilot | 60s | 98.97 [77.01, 132.22] | 27 | 53 |
| keep_speed | wall_pilot | 120s | 745.87 [658.98, 876.83] | 27 | 52 |
| keep_speed | seen_controls | 10s | 2.50 [1.91, 3.47] | 31 | 217 |
| keep_speed | seen_controls | 30s | 40.44 [24.51, 49.48] | 31 | 217 |
| keep_speed | seen_controls | 60s | 174.30 [130.21, 219.18] | 31 | 217 |
| keep_speed | seen_controls | 120s | 1045.31 [961.67, 1155.65] | 31 | 212 |

## Wall-status and assembly-timing splits (HDV |ds| median)

| model | set | subgroup | n | 10s | 30s | 60s | 120s |
|---|---|---|---:|---:|---:|---:|---:|
| struct | wall_same | assembled_released | 32 | 1.84 | 5.07 | 8.26 | 83.03 |
| struct | wall_same | aborted | 35 | 2.04 | 4.84 | 13.61 | 110.02 |
| struct | wall_same | engaged | 51 | 2.06 | 5.07 | 8.34 | 90.32 |
| struct | wall_same | assembled_pre_t0 | 28 | 1.83 | 5.45 | 7.62 | 85.90 |
| struct | wall_same | assembled_post_t0 | 23 | 2.17 | 4.84 | 11.08 | 96.76 |
| struct | wall_same | never_assembled | 16 | 2.04 | 4.71 | 13.61 | 128.99 |
| struct | wall_pilot | assembled_released | 33 | 1.89 | 5.07 | 8.34 | 86.06 |
| struct | wall_pilot | aborted | 20 | 2.16 | 4.78 | 11.09 | 107.47 |
| struct | wall_pilot | engaged | 53 | 2.06 | 5.06 | 8.34 | 96.76 |
| struct | wall_pilot | assembled_pre_t0 | 28 | 1.83 | 5.45 | 7.62 | 85.90 |
| struct | wall_pilot | assembled_post_t0 | 25 | 2.17 | 4.84 | 11.39 | 103.44 |
| struct | wall_pilot | never_assembled | 0 | n/a | n/a | n/a | n/a |
| blackbox_traj | wall_same | assembled_released | 32 | 2.68 | 9.32 | 20.32 | 167.87 |
| blackbox_traj | wall_same | aborted | 35 | 2.95 | 10.36 | 29.69 | 102.30 |
| blackbox_traj | wall_same | engaged | 51 | 2.76 | 9.71 | 20.85 | 142.19 |
| blackbox_traj | wall_same | assembled_pre_t0 | 28 | 2.77 | 8.66 | 17.51 | 152.44 |
| blackbox_traj | wall_same | assembled_post_t0 | 23 | 2.50 | 10.96 | 28.62 | 114.82 |
| blackbox_traj | wall_same | never_assembled | 16 | 3.07 | 10.84 | 35.89 | 269.74 |
| blackbox_traj | wall_pilot | assembled_released | 33 | 2.72 | 9.71 | 19.93 | 183.68 |
| blackbox_traj | wall_pilot | aborted | 20 | 2.87 | 10.10 | 23.39 | 81.89 |
| blackbox_traj | wall_pilot | engaged | 53 | 2.76 | 9.79 | 20.85 | 147.03 |
| blackbox_traj | wall_pilot | assembled_pre_t0 | 28 | 2.77 | 8.66 | 17.51 | 152.44 |
| blackbox_traj | wall_pilot | assembled_post_t0 | 25 | 2.50 | 10.96 | 28.62 | 132.97 |
| blackbox_traj | wall_pilot | never_assembled | 0 | n/a | n/a | n/a | n/a |
| blackbox_id | wall_same | assembled_released | 32 | 2.34 | 9.66 | 37.26 | 227.37 |
| blackbox_id | wall_same | aborted | 35 | 3.28 | 10.57 | 45.23 | 279.95 |
| blackbox_id | wall_same | engaged | 51 | 2.85 | 10.57 | 40.07 | 234.14 |
| blackbox_id | wall_same | assembled_pre_t0 | 28 | 2.85 | 12.71 | 43.35 | 216.45 |
| blackbox_id | wall_same | assembled_post_t0 | 23 | 2.70 | 9.66 | 38.23 | 326.19 |
| blackbox_id | wall_same | never_assembled | 16 | 1.90 | 8.54 | 45.23 | 295.81 |
| blackbox_id | wall_pilot | assembled_released | 33 | 2.23 | 9.66 | 36.28 | 220.52 |
| blackbox_id | wall_pilot | aborted | 20 | 3.36 | 14.03 | 52.82 | 257.16 |
| blackbox_id | wall_pilot | engaged | 53 | 2.78 | 10.15 | 40.07 | 229.75 |
| blackbox_id | wall_pilot | assembled_pre_t0 | 28 | 2.85 | 12.71 | 43.35 | 216.45 |
| blackbox_id | wall_pilot | assembled_post_t0 | 25 | 2.60 | 9.07 | 38.23 | 300.81 |
| blackbox_id | wall_pilot | never_assembled | 0 | n/a | n/a | n/a | n/a |
| pure_physics | wall_same | assembled_released | 32 | 7.42 | 29.27 | 77.38 | 777.58 |
| pure_physics | wall_same | aborted | 35 | 7.83 | 29.99 | 59.62 | 664.02 |
| pure_physics | wall_same | engaged | 51 | 7.42 | 30.08 | 82.76 | 662.62 |
| pure_physics | wall_same | assembled_pre_t0 | 28 | 7.42 | 29.02 | 83.64 | 728.48 |
| pure_physics | wall_same | assembled_post_t0 | 23 | 7.47 | 33.37 | 76.34 | 657.01 |
| pure_physics | wall_same | never_assembled | 16 | 7.92 | 27.57 | 52.72 | 838.64 |
| pure_physics | wall_pilot | assembled_released | 33 | 7.42 | 29.27 | 82.76 | 794.64 |
| pure_physics | wall_pilot | aborted | 20 | 7.32 | 33.42 | 86.82 | 662.62 |
| pure_physics | wall_pilot | engaged | 53 | 7.42 | 30.55 | 83.64 | 671.56 |
| pure_physics | wall_pilot | assembled_pre_t0 | 28 | 7.42 | 29.02 | 83.64 | 728.48 |
| pure_physics | wall_pilot | assembled_post_t0 | 25 | 7.13 | 33.37 | 82.76 | 670.16 |
| pure_physics | wall_pilot | never_assembled | 0 | n/a | n/a | n/a | n/a |
| keep_speed | wall_same | assembled_released | 32 | 3.80 | 24.28 | 109.53 | 785.62 |
| keep_speed | wall_same | aborted | 35 | 4.62 | 29.91 | 102.25 | 1145.12 |
| keep_speed | wall_same | engaged | 51 | 4.47 | 25.67 | 111.96 | 867.31 |
| keep_speed | wall_same | assembled_pre_t0 | 28 | 4.48 | 25.67 | 118.53 | 849.88 |
| keep_speed | wall_same | assembled_post_t0 | 23 | 4.41 | 27.88 | 97.56 | 893.00 |
| keep_speed | wall_same | never_assembled | 16 | 2.65 | 21.06 | 101.44 | 1415.31 |
| keep_speed | wall_pilot | assembled_released | 33 | 3.33 | 24.28 | 107.31 | 785.62 |
| keep_speed | wall_pilot | aborted | 20 | 5.46 | 33.55 | 134.20 | 968.58 |
| keep_speed | wall_pilot | engaged | 53 | 4.41 | 25.67 | 111.96 | 874.50 |
| keep_speed | wall_pilot | assembled_pre_t0 | 28 | 4.48 | 25.67 | 118.53 | 849.88 |
| keep_speed | wall_pilot | assembled_post_t0 | 25 | 3.64 | 26.08 | 97.56 | 926.77 |
| keep_speed | wall_pilot | never_assembled | 0 | n/a | n/a | n/a | n/a |

## Paired ranking: predicted vs recorded wall-minus-native delay

| model | set | h | n | sign agree [CI] | Spearman | model regret s [CI] | always-wall | always-native |
|---|---|---|---:|---|---:|---|---:|---:|
| struct | wall_same | 60.0s | 16 | 1.00 [1.00, 1.00] | 0.99 | 0.00 [0.00, 0.00] | 1.51 | 0.00 |
| struct | wall_same | 120.0s | 16 | 0.94 [0.81, 1.00] | 0.88 | 2.43 [0.00, 7.28] | 15.38 | 39.48 |
| struct | wall_pilot | 60.0s | 7 | 1.00 [1.00, 1.00] | 0.89 | 0.00 [0.00, 0.00] | 12.15 | 0.00 |
| struct | wall_pilot | 120.0s | 7 | 0.86 [0.57, 1.00] | 1.00 | 5.55 [0.00, 16.64] | 35.15 | 56.99 |
| blackbox_traj | wall_same | 60.0s | 16 | 1.00 [1.00, 1.00] | 1.00 | 0.00 [0.00, 0.00] | 1.51 | 0.00 |
| blackbox_traj | wall_same | 120.0s | 16 | 0.88 [0.69, 1.00] | 0.73 | 6.13 [0.00, 15.95] | 15.38 | 39.48 |
| blackbox_traj | wall_pilot | 60.0s | 7 | 1.00 [1.00, 1.00] | 1.00 | 0.00 [0.00, 0.00] | 12.15 | 0.00 |
| blackbox_traj | wall_pilot | 120.0s | 7 | 0.57 [0.14, 0.86] | 0.79 | 24.73 [5.55, 49.09] | 35.15 | 56.99 |
| blackbox_id | wall_same | 60.0s | 16 | 0.50 [0.25, 0.75] | 0.52 | 0.00 [0.00, 0.00] | 1.51 | 0.00 |
| blackbox_id | wall_same | 120.0s | 16 | 0.19 [0.00, 0.38] | 0.02 | 31.92 [4.29, 68.22] | 15.38 | 39.48 |
| blackbox_id | wall_pilot | 60.0s | 7 | 1.00 [1.00, 1.00] | -0.43 | 0.00 [0.00, 0.00] | 12.15 | 0.00 |
| blackbox_id | wall_pilot | 120.0s | 7 | 0.43 [0.14, 0.71] | 0.43 | 47.62 [5.55, 107.87] | 35.15 | 56.99 |
| pure_physics | wall_same | 60.0s | 16 | 1.00 [1.00, 1.00] | 0.97 | 0.00 [0.00, 0.00] | 1.51 | 0.00 |
| pure_physics | wall_same | 120.0s | 16 | 0.88 [0.69, 1.00] | 0.59 | 11.82 [0.00, 31.75] | 15.38 | 39.48 |
| pure_physics | wall_pilot | 60.0s | 7 | 0.86 [0.57, 1.00] | 0.00 | 9.06 [0.00, 27.19] | 12.15 | 0.00 |
| pure_physics | wall_pilot | 120.0s | 7 | 0.71 [0.43, 1.00] | 0.61 | 27.01 [0.00, 64.11] | 35.15 | 56.99 |
| keep_speed | wall_same | 60.0s | 16 | 1.00 [1.00, 1.00] | 0.97 | 0.00 [0.00, 0.00] | 1.51 | 0.00 |
| keep_speed | wall_same | 120.0s | 16 | 0.88 [0.69, 1.00] | 0.59 | 11.82 [0.00, 31.75] | 15.38 | 39.48 |
| keep_speed | wall_pilot | 60.0s | 7 | 0.86 [0.57, 1.00] | 0.00 | 9.06 [0.00, 27.19] | 12.15 | 0.00 |
| keep_speed | wall_pilot | 120.0s | 7 | 0.71 [0.43, 1.00] | 0.61 | 27.01 [0.00, 64.11] | 35.15 | 56.99 |

## Verdict

On the unseen WALL arm at 60 s (`wall_same`, 67 states / 29 worlds) the structured model's median HDV |ds| is 8.71 m versus black-box-traj 25.41 m and black-box-id 40.07 m. The paired world bootstrap (black-box minus structured, so positive favours the structured model) is black-box-traj - struct 9.98 [5.09, 16.78] m and black-box-id - struct 31.78 [18.61, 45.37] m; both 95% intervals exclude 0, so the structured model predicts human reactions to the unseen wall better than both black boxes. On `wall_pilot` (53 states / 27 worlds) the same 60 s contrasts are black-box-traj - struct 10.71 [5.37, 14.81] and black-box-id - struct 35.69 [23.66, 45.51].

The structured model also has the lowest blow-up rate on the wall arm (`wall_same` 60 s/120 s rule: struct 0.000, black-box-traj 0.015, black-box-id 0.379).

Verdict: **SUPPORTED** - the structured WM-S predicts the unseen wall controller better than both black boxes: the paired black-box-minus-structured HDV |ds| interval excludes 0 at 60 s on both wall cohorts, and the structured wall/seen error ratio stays below 1.

**Comparison to the PI result.** In the C1-FAST leave-PI-out evaluation (same checkpoints, PI arm) the structured model beat black-box-traj by about 7.9 m at 60 s on U-same (140 worlds) and 2.9 m on U-new (21 worlds). Here the unseen WALL advantage is of the same order and, on `wall_same`, larger (9.98 m vs black-box-traj at 60 s). The wall/seen error ratio, however, is below 1 for *every* model (struct 0.33, black-box-traj 0.57, black-box-id 0.37 at 60 s), exactly as in C1: the ratio does not separate the models because the wall states are simply easier than the seen controls, so the paired differences carry the claim, not the ratio.

**Ranking (planner view).** On the paired states (16 `wall_same` + 7 `wall_pilot`; the two cohorts overlap in worlds and are never summed), the structured model's predicted wall-minus-native network-delay delta agrees in sign with the recorded delta and its decision regret is lower than always-wall or always-native. At 60 s all models are at sign agreement 1.0 (the recorded wall is beneficial for every paired state, so the test saturates); at 120 s the structured model keeps the highest sign agreement / lowest regret while black-box-id degrades (wall_same 120 s regret: struct 2.43 s, black-box-traj 6.13 s, black-box-id 31.92 s).

## Caveats

- **Small paired n.** Only 16 (`wall_same`) and 7 (`wall_pilot`) states have a verified identical-initial native sibling; the paired-delta intervals are wide and the 120 s `wall_pilot` black-box-traj interval includes 0. The horizon contrasts use all 67/53 wall states (29/27 worlds).
- **Teacher-forced wall motion.** CAVs are replayed from the recorded arm for every model (the mode C1 was scored in), so this measures HDV response given the realized wall, not closed-loop wall control. The pure_physics and keep_speed references overlay the recorded CAVs on an arm-independent HDV rollout, so their predicted wall-minus-native delta reflects only the CAV placement; report them as references, not as planners.
- **Selection by the materialiser.** Walls whose members were born after t0 are not in the t0 CAV roster and were rejected, so walls assembling ~60-90 s after t0 are absent; `never_assembled` here means the timeline had no assembly time, not that no wall existed. 130 wall runs were removed from training entirely, but the checkpoints were trained on pack_dspeed+pack_damping only (pack_v3 = 0 windows), so the wall is unseen by construction.
- **Black-box-id contract.** black-box-id learns CAV motion and ignores the recorded teacher, an unequal-CAV contract; it is reported separately, not as a teacher-matched three-model result.
- **No retraining.** The checkpoints are the C1FAST box checkpoints (read-only); no model was retrained on the corrected LPWO split, so this is a scoring result on the existing models, not the leave-PI-and-wall-out retraining.

