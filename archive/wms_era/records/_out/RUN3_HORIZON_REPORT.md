# Horizon report (HDV |ds| by horizon, per-state result files)

Generated 2026-10-08T03:46:01Z. Metric: HDV |ds| (m): per-state median, summarised across states.
Blow-up rule: |ds|>50m @60s and |ds|>100m @120s. Reference model for the paired bootstrap: `wms` (negative difference favours the model over the reference).
The scored arm per set is the controller arm under test (PI for the unseen PI sets, native for the PI-free seen sets).

## Sets

- `run3` / `u_same`: 140 states / 140 worlds, tested arm `pi` (arms present: native, pi)
- `run3` / `u_new`: 21 states / 21 worlds, tested arm `pi` (arms present: native, pi)
- `run3` / `seen`: 157 states / 26 worlds, tested arm `native` (arms present: native)
- `run3_s120` / `u_same`: 140 states / 140 worlds, tested arm `pi` (arms present: native, pi)
- `run3_s120` / `u_new`: 21 states / 21 worlds, tested arm `pi` (arms present: native, pi)
- `run3_s120` / `seen`: 157 states / 26 worlds, tested arm `native` (arms present: native)
- `wms` / `u_same`: 140 states / 140 worlds, tested arm `pi` (arms present: native, pi)
- `wms` / `u_new`: 21 states / 21 worlds, tested arm `pi` (arms present: native, pi)
- `wms` / `seen`: 157 states / 26 worlds, tested arm `native` (arms present: native)
- `blackbox_traj` / `u_same`: 140 states / 140 worlds, tested arm `pi` (arms present: native, pi)
- `blackbox_traj` / `u_new`: 21 states / 21 worlds, tested arm `pi` (arms present: native, pi)
- `blackbox_traj` / `seen`: 157 states / 26 worlds, tested arm `native` (arms present: native)
- `blackbox_id` / `u_same`: 140 states / 140 worlds, tested arm `pi` (arms present: native, pi)
- `blackbox_id` / `u_new`: 21 states / 21 worlds, tested arm `pi` (arms present: native, pi)
- `blackbox_id` / `seen`: 157 states / 26 worlds, tested arm `native` (arms present: native)

## HDV |ds| at 10 s

| model | set | arm | n | median | IQR | p90 | sd | p90 within (median) |
|---|---|---|---|---|---|---|---|---|
| run3 | u_same | native | 140 | 1.74 | 0.89 | 2.76 | 0.66 | 13.92 |
| run3 | u_same | pi | 140 | 1.81 | 0.86 | 2.72 | 0.65 | 13.55 |
| run3 | u_new | native | 21 | 1.64 | 0.87 | 2.80 | 0.88 | 13.13 |
| run3 | u_new | pi | 21 | 1.65 | 0.82 | 2.67 | 0.83 | 13.86 |
| run3 | seen | native | 157 | 1.66 | 0.93 | 3.16 | 0.96 | 13.86 |
| run3_s120 | u_same | native | 140 | 1.99 | 0.74 | 2.78 | 0.62 | 13.06 |
| run3_s120 | u_same | pi | 140 | 2.00 | 0.76 | 2.75 | 0.59 | 13.23 |
| run3_s120 | u_new | native | 21 | 1.75 | 0.57 | 2.68 | 0.67 | 13.09 |
| run3_s120 | u_new | pi | 21 | 1.84 | 0.57 | 2.71 | 0.67 | 12.91 |
| run3_s120 | seen | native | 157 | 1.84 | 0.86 | 2.89 | 0.75 | 14.05 |
| wms | u_same | native | 140 | 2.34 | 0.84 | 3.21 | 0.74 | 14.82 |
| wms | u_same | pi | 140 | 2.35 | 0.96 | 3.28 | 0.74 | 14.73 |
| wms | u_new | native | 21 | 2.14 | 0.85 | 3.35 | 0.89 | 13.49 |
| wms | u_new | pi | 21 | 2.23 | 0.79 | 3.09 | 0.84 | 12.81 |
| wms | seen | native | 157 | 2.17 | 1.05 | 3.36 | 0.98 | 15.47 |
| blackbox_traj | u_same | native | 140 | 2.94 | 1.20 | 4.22 | 1.00 | 19.68 |
| blackbox_traj | u_same | pi | 140 | 2.92 | 1.21 | 4.04 | 1.04 | 19.20 |
| blackbox_traj | u_new | native | 21 | 2.38 | 1.83 | 5.13 | 1.29 | 19.35 |
| blackbox_traj | u_new | pi | 21 | 2.39 | 1.91 | 4.84 | 1.40 | 18.82 |
| blackbox_traj | seen | native | 157 | 2.65 | 1.71 | 5.32 | 1.51 | 20.05 |
| blackbox_id | u_same | native | 140 | 2.31 | 1.53 | 4.38 | 1.34 | 21.43 |
| blackbox_id | u_same | pi | 140 | 2.37 | 1.57 | 4.39 | 1.32 | 21.53 |
| blackbox_id | u_new | native | 21 | 2.26 | 1.85 | 4.47 | 2.11 | 18.87 |
| blackbox_id | u_new | pi | 21 | 2.04 | 1.58 | 4.16 | 1.45 | 18.87 |
| blackbox_id | seen | native | 157 | 2.43 | 1.91 | 6.15 | 2.56 | 25.62 |

## HDV |ds| at 30 s

| model | set | arm | n | median | IQR | p90 | sd | p90 within (median) |
|---|---|---|---|---|---|---|---|---|
| run3 | u_same | native | 140 | 5.72 | 5.29 | 12.19 | 4.07 | 69.45 |
| run3 | u_same | pi | 140 | 6.22 | 5.26 | 12.12 | 3.90 | 69.45 |
| run3 | u_new | native | 21 | 6.15 | 4.54 | 10.17 | 11.84 | 55.21 |
| run3 | u_new | pi | 21 | 6.94 | 5.65 | 10.32 | 11.68 | 56.06 |
| run3 | seen | native | 157 | 7.36 | 6.49 | 18.81 | 10.71 | 81.46 |
| run3_s120 | u_same | native | 140 | 5.05 | 4.14 | 10.65 | 3.30 | 60.71 |
| run3_s120 | u_same | pi | 140 | 5.71 | 4.31 | 10.89 | 3.30 | 60.73 |
| run3_s120 | u_new | native | 21 | 6.17 | 5.12 | 9.91 | 3.01 | 52.70 |
| run3_s120 | u_new | pi | 21 | 6.55 | 4.82 | 10.22 | 3.15 | 47.27 |
| run3_s120 | seen | native | 157 | 7.87 | 5.99 | 14.62 | 5.48 | 67.68 |
| wms | u_same | native | 140 | 6.32 | 4.53 | 12.33 | 3.57 | 54.63 |
| wms | u_same | pi | 140 | 6.69 | 4.77 | 12.33 | 3.41 | 54.72 |
| wms | u_new | native | 21 | 5.58 | 5.08 | 10.89 | 3.15 | 52.54 |
| wms | u_new | pi | 21 | 5.65 | 5.40 | 10.89 | 3.29 | 50.10 |
| wms | seen | native | 157 | 8.19 | 6.90 | 17.99 | 8.02 | 73.62 |
| blackbox_traj | u_same | native | 140 | 10.49 | 5.15 | 17.33 | 4.38 | 70.49 |
| blackbox_traj | u_same | pi | 140 | 10.79 | 5.29 | 17.56 | 4.17 | 72.29 |
| blackbox_traj | u_new | native | 21 | 10.39 | 5.61 | 14.07 | 3.32 | 53.57 |
| blackbox_traj | u_new | pi | 21 | 10.17 | 5.11 | 14.07 | 3.15 | 67.89 |
| blackbox_traj | seen | native | 157 | 13.27 | 8.35 | 28.77 | 11.46 | 99.95 |
| blackbox_id | u_same | native | 140 | 10.06 | 6.04 | 19.98 | 5.87 | 86.97 |
| blackbox_id | u_same | pi | 140 | 9.88 | 5.51 | 19.93 | 6.01 | 86.77 |
| blackbox_id | u_new | native | 21 | 11.54 | 6.92 | 19.34 | 5.83 | 81.67 |
| blackbox_id | u_new | pi | 21 | 10.69 | 7.71 | 16.76 | 5.02 | 80.45 |
| blackbox_id | seen | native | 157 | 14.08 | 16.68 | 72.85 | 33.14 | 125.04 |

## HDV |ds| at 60 s

| model | set | arm | n | median | IQR | p90 | sd | p90 within (median) |
|---|---|---|---|---|---|---|---|---|
| run3 | u_same | native | 140 | 16.88 | 20.06 | 45.45 | 18.67 | 128.83 |
| run3 | u_same | pi | 140 | 17.32 | 20.23 | 46.91 | 19.66 | 130.77 |
| run3 | u_new | native | 21 | 11.85 | 13.54 | 51.54 | 35.59 | 129.00 |
| run3 | u_new | pi | 21 | 12.76 | 12.28 | 51.54 | 32.65 | 130.83 |
| run3 | seen | native | 157 | 29.70 | 47.59 | 106.00 | 39.83 | 241.01 |
| run3_s120 | u_same | native | 140 | 13.55 | 14.61 | 33.74 | 13.60 | 111.09 |
| run3_s120 | u_same | pi | 140 | 15.04 | 13.90 | 36.52 | 15.57 | 108.87 |
| run3_s120 | u_new | native | 21 | 15.59 | 18.17 | 44.24 | 15.43 | 118.66 |
| run3_s120 | u_new | pi | 21 | 12.54 | 18.97 | 38.38 | 13.52 | 142.08 |
| run3_s120 | seen | native | 157 | 25.04 | 30.73 | 70.08 | 28.40 | 193.21 |
| wms | u_same | native | 140 | 16.24 | 14.85 | 36.47 | 14.44 | 101.66 |
| wms | u_same | pi | 140 | 16.94 | 14.40 | 34.53 | 13.12 | 103.05 |
| wms | u_new | native | 21 | 17.07 | 16.42 | 36.42 | 13.60 | 124.26 |
| wms | u_new | pi | 21 | 15.43 | 12.41 | 34.01 | 12.11 | 118.19 |
| wms | seen | native | 157 | 28.50 | 36.85 | 87.81 | 39.44 | 205.53 |
| blackbox_traj | u_same | native | 140 | 24.81 | 13.46 | 40.11 | 11.23 | 185.67 |
| blackbox_traj | u_same | pi | 140 | 23.79 | 11.57 | 39.96 | 11.70 | 184.84 |
| blackbox_traj | u_new | native | 21 | 20.34 | 11.75 | 38.66 | 10.85 | 175.38 |
| blackbox_traj | u_new | pi | 21 | 22.40 | 11.98 | 32.17 | 9.94 | 173.22 |
| blackbox_traj | seen | native | 157 | 32.31 | 38.43 | 101.96 | 40.90 | 280.90 |
| blackbox_id | u_same | native | 140 | 28.14 | 18.26 | 53.77 | 18.38 | 143.38 |
| blackbox_id | u_same | pi | 140 | 28.17 | 19.02 | 60.05 | 18.04 | 137.77 |
| blackbox_id | u_new | native | 21 | 30.28 | 25.77 | 60.38 | 16.57 | 143.10 |
| blackbox_id | u_new | pi | 21 | 23.59 | 26.59 | 61.02 | 17.17 | 136.57 |
| blackbox_id | seen | native | 157 | 47.71 | 104.99 | 285.97 | 113.36 | 266.08 |

## HDV |ds| at 120 s

| model | set | arm | n | median | IQR | p90 | sd | p90 within (median) |
|---|---|---|---|---|---|---|---|---|
| run3 | u_same | native | 129 | 178.35 | 364.06 | 675.74 | 342.13 | 725.37 |
| run3 | u_same | pi | 130 | 152.82 | 356.93 | 627.80 | 330.78 | 686.35 |
| run3 | u_new | native | 19 | 105.02 | 115.04 | 605.46 | 257.86 | 871.61 |
| run3 | u_new | pi | 19 | 105.02 | 289.81 | 715.85 | 272.43 | 799.87 |
| run3 | seen | native | 153 | 148.86 | 209.58 | 525.68 | 278.90 | 724.16 |
| run3_s120 | u_same | native | 129 | 132.53 | 353.27 | 653.84 | 332.65 | 640.66 |
| run3_s120 | u_same | pi | 130 | 119.42 | 329.58 | 654.67 | 326.60 | 621.51 |
| run3_s120 | u_new | native | 19 | 75.50 | 121.57 | 461.32 | 222.05 | 645.11 |
| run3_s120 | u_new | pi | 19 | 69.81 | 221.27 | 455.77 | 219.97 | 699.39 |
| run3_s120 | seen | native | 153 | 126.59 | 214.47 | 449.68 | 253.87 | 677.90 |
| wms | u_same | native | 129 | 94.49 | 136.90 | 361.62 | 200.58 | 449.49 |
| wms | u_same | pi | 130 | 69.74 | 169.05 | 395.53 | 183.67 | 443.66 |
| wms | u_new | native | 19 | 73.05 | 121.05 | 268.65 | 94.55 | 333.31 |
| wms | u_new | pi | 19 | 39.50 | 152.44 | 250.67 | 97.71 | 308.95 |
| wms | seen | native | 153 | 146.80 | 192.86 | 340.37 | 165.31 | 531.73 |
| blackbox_traj | u_same | native | 129 | 157.51 | 220.74 | 402.82 | 139.50 | 513.07 |
| blackbox_traj | u_same | pi | 130 | 156.48 | 225.27 | 417.74 | 141.89 | 512.82 |
| blackbox_traj | u_new | native | 19 | 188.68 | 235.53 | 460.51 | 159.22 | 532.36 |
| blackbox_traj | u_new | pi | 19 | 173.30 | 179.23 | 406.28 | 138.98 | 538.03 |
| blackbox_traj | seen | native | 153 | 196.17 | 184.04 | 424.17 | 143.94 | 528.03 |
| blackbox_id | u_same | native | 129 | 108.40 | 150.58 | 385.92 | 171.95 | 384.79 |
| blackbox_id | u_same | pi | 130 | 94.48 | 152.86 | 395.61 | 177.25 | 425.04 |
| blackbox_id | u_new | native | 19 | 98.09 | 149.17 | 279.96 | 104.00 | 339.28 |
| blackbox_id | u_new | pi | 19 | 83.70 | 132.41 | 434.86 | 187.50 | 476.37 |
| blackbox_id | seen | native | 153 | 217.45 | 318.98 | 581.21 | 246.48 | 600.99 |

## Blow-up rate (HDV median |ds|)

| model | set | arm | rate | blowups / evaluated | excluded |
|---|---|---|---|---|---|
| run3 | u_same | native | 0.085 | 11 / 129 | 11 |
| run3 | u_same | pi | 0.085 | 11 / 130 | 10 |
| run3 | u_new | native | 0.158 | 3 / 19 | 2 |
| run3 | u_new | pi | 0.158 | 3 / 19 | 2 |
| run3 | seen | native | 0.327 | 50 / 153 | 4 |
| run3_s120 | u_same | native | 0.039 | 5 / 129 | 11 |
| run3_s120 | u_same | pi | 0.046 | 6 / 130 | 10 |
| run3_s120 | u_new | native | 0.053 | 1 / 19 | 2 |
| run3_s120 | u_new | pi | 0.053 | 1 / 19 | 2 |
| run3_s120 | seen | native | 0.183 | 28 / 153 | 4 |
| wms | u_same | native | 0.016 | 2 / 129 | 11 |
| wms | u_same | pi | 0.023 | 3 / 130 | 10 |
| wms | u_new | native | 0.053 | 1 / 19 | 2 |
| wms | u_new | pi | 0.053 | 1 / 19 | 2 |
| wms | seen | native | 0.261 | 40 / 153 | 4 |
| blackbox_traj | u_same | native | 0.016 | 2 / 129 | 11 |
| blackbox_traj | u_same | pi | 0.031 | 4 / 130 | 10 |
| blackbox_traj | u_new | native | 0.000 | 0 / 19 | 2 |
| blackbox_traj | u_new | pi | 0.000 | 0 / 19 | 2 |
| blackbox_traj | seen | native | 0.327 | 50 / 153 | 4 |
| blackbox_id | u_same | native | 0.124 | 16 / 129 | 11 |
| blackbox_id | u_same | pi | 0.123 | 16 / 130 | 10 |
| blackbox_id | u_new | native | 0.105 | 2 / 19 | 2 |
| blackbox_id | u_new | pi | 0.105 | 2 / 19 | 2 |
| blackbox_id | seen | native | 0.438 | 67 / 153 | 4 |

## Paired model - reference (`wms`), median HDV |ds| difference, bootstrap over worlds

- run3 / u_same (arm `pi`) @ 10 s: -0.50 [-0.62, -0.42] (worlds=140, pairs=140, n_boot=1000)
- run3 / u_same (arm `pi`) @ 30 s: -0.56 [-0.93, -0.08] (worlds=140, pairs=140, n_boot=1000)
- run3 / u_same (arm `pi`) @ 60 s: 0.15 [-1.49, 2.09] (worlds=140, pairs=140, n_boot=1000)
- run3 / u_same (arm `pi`) @ 120 s: 44.90 [29.54, 67.24] (worlds=130, pairs=130, n_boot=1000)
- run3 / u_new (arm `pi`) @ 10 s: -0.36 [-0.56, -0.19] (worlds=21, pairs=21, n_boot=1000)
- run3 / u_new (arm `pi`) @ 30 s: 0.08 [-0.70, 0.35] (worlds=21, pairs=21, n_boot=1000)
- run3 / u_new (arm `pi`) @ 60 s: 0.02 [-3.06, 5.24] (worlds=21, pairs=21, n_boot=1000)
- run3 / u_new (arm `pi`) @ 120 s: 36.70 [-10.77, 198.18] (worlds=19, pairs=19, n_boot=1000)
- run3 / seen (arm `native`) @ 10 s: -0.48 [-0.68, -0.16] (worlds=26, pairs=26, n_boot=1000)
- run3 / seen (arm `native`) @ 30 s: -0.40 [-1.02, 0.02] (worlds=26, pairs=26, n_boot=1000)
- run3 / seen (arm `native`) @ 60 s: -1.15 [-3.61, 4.52] (worlds=26, pairs=26, n_boot=1000)
- run3 / seen (arm `native`) @ 120 s: 20.03 [-20.46, 70.91] (worlds=26, pairs=26, n_boot=1000)
- run3_s120 / u_same (arm `pi`) @ 10 s: -0.30 [-0.38, -0.24] (worlds=140, pairs=140, n_boot=1000)
- run3_s120 / u_same (arm `pi`) @ 30 s: -0.84 [-1.39, -0.47] (worlds=140, pairs=140, n_boot=1000)
- run3_s120 / u_same (arm `pi`) @ 60 s: -0.62 [-2.48, 0.96] (worlds=140, pairs=140, n_boot=1000)
- run3_s120 / u_same (arm `pi`) @ 120 s: 23.60 [9.31, 48.61] (worlds=130, pairs=130, n_boot=1000)
- run3_s120 / u_new (arm `pi`) @ 10 s: -0.33 [-0.59, -0.11] (worlds=21, pairs=21, n_boot=1000)
- run3_s120 / u_new (arm `pi`) @ 30 s: 0.07 [-1.21, 0.91] (worlds=21, pairs=21, n_boot=1000)
- run3_s120 / u_new (arm `pi`) @ 60 s: -1.49 [-4.55, 4.37] (worlds=21, pairs=21, n_boot=1000)
- run3_s120 / u_new (arm `pi`) @ 120 s: 17.09 [-3.45, 93.17] (worlds=19, pairs=19, n_boot=1000)
- run3_s120 / seen (arm `native`) @ 10 s: -0.26 [-0.51, -0.15] (worlds=26, pairs=26, n_boot=1000)
- run3_s120 / seen (arm `native`) @ 30 s: -0.30 [-1.25, 0.10] (worlds=26, pairs=26, n_boot=1000)
- run3_s120 / seen (arm `native`) @ 60 s: -2.17 [-4.17, 0.19] (worlds=26, pairs=26, n_boot=1000)
- run3_s120 / seen (arm `native`) @ 120 s: -14.95 [-43.58, 0.38] (worlds=26, pairs=26, n_boot=1000)
- blackbox_traj / u_same (arm `pi`) @ 10 s: 0.51 [0.42, 0.66] (worlds=140, pairs=140, n_boot=1000)
- blackbox_traj / u_same (arm `pi`) @ 30 s: 3.84 [3.16, 4.63] (worlds=140, pairs=140, n_boot=1000)
- blackbox_traj / u_same (arm `pi`) @ 60 s: 7.86 [5.09, 9.49] (worlds=140, pairs=140, n_boot=1000)
- blackbox_traj / u_same (arm `pi`) @ 120 s: 42.01 [29.43, 67.70] (worlds=130, pairs=130, n_boot=1000)
- blackbox_traj / u_new (arm `pi`) @ 10 s: 0.41 [0.10, 1.19] (worlds=21, pairs=21, n_boot=1000)
- blackbox_traj / u_new (arm `pi`) @ 30 s: 4.52 [1.16, 6.95] (worlds=21, pairs=21, n_boot=1000)
- blackbox_traj / u_new (arm `pi`) @ 60 s: 2.86 [1.61, 11.19] (worlds=21, pairs=21, n_boot=1000)
- blackbox_traj / u_new (arm `pi`) @ 120 s: 56.19 [41.37, 133.80] (worlds=19, pairs=19, n_boot=1000)
- blackbox_traj / seen (arm `native`) @ 10 s: 0.52 [0.17, 0.91] (worlds=26, pairs=26, n_boot=1000)
- blackbox_traj / seen (arm `native`) @ 30 s: 3.67 [2.28, 5.81] (worlds=26, pairs=26, n_boot=1000)
- blackbox_traj / seen (arm `native`) @ 60 s: 3.85 [-0.94, 9.95] (worlds=26, pairs=26, n_boot=1000)
- blackbox_traj / seen (arm `native`) @ 120 s: 70.08 [26.63, 131.01] (worlds=26, pairs=26, n_boot=1000)
- blackbox_id / u_same (arm `pi`) @ 10 s: 0.05 [-0.13, 0.25] (worlds=140, pairs=140, n_boot=1000)
- blackbox_id / u_same (arm `pi`) @ 30 s: 3.19 [2.13, 4.26] (worlds=140, pairs=140, n_boot=1000)
- blackbox_id / u_same (arm `pi`) @ 60 s: 12.12 [8.21, 15.79] (worlds=140, pairs=140, n_boot=1000)
- blackbox_id / u_same (arm `pi`) @ 120 s: 14.50 [-0.17, 27.39] (worlds=130, pairs=130, n_boot=1000)
- blackbox_id / u_new (arm `pi`) @ 10 s: 0.28 [-0.24, 0.46] (worlds=21, pairs=21, n_boot=1000)
- blackbox_id / u_new (arm `pi`) @ 30 s: 2.72 [1.71, 4.64] (worlds=21, pairs=21, n_boot=1000)
- blackbox_id / u_new (arm `pi`) @ 60 s: 10.57 [9.01, 26.59] (worlds=21, pairs=21, n_boot=1000)
- blackbox_id / u_new (arm `pi`) @ 120 s: 26.82 [-2.79, 61.11] (worlds=19, pairs=19, n_boot=1000)
- blackbox_id / seen (arm `native`) @ 10 s: 0.23 [-0.13, 0.61] (worlds=26, pairs=26, n_boot=1000)
- blackbox_id / seen (arm `native`) @ 30 s: 2.16 [0.15, 5.87] (worlds=26, pairs=26, n_boot=1000)
- blackbox_id / seen (arm `native`) @ 60 s: 14.20 [3.89, 19.52] (worlds=26, pairs=26, n_boot=1000)
- blackbox_id / seen (arm `native`) @ 120 s: 35.21 [6.39, 57.76] (worlds=26, pairs=26, n_boot=1000)
