# Horizon report (HDV |ds| by horizon, per-state result files)

Generated 2026-10-08T00:20:34Z. Metric: HDV |ds| (m): per-state median, summarised across states.
Blow-up rule: |ds|>50m @60s and |ds|>100m @120s. Reference model for the paired bootstrap: `c1_struct` (negative difference favours the model over the reference).
The scored arm per set is the controller arm under test (PI for the unseen PI sets, native for the PI-free seen sets).

## Sets

- `c1_struct` / `u_same`: 140 states / 140 worlds, tested arm `pi` (arms present: native, pi)
- `c1_struct` / `u_new`: 21 states / 21 worlds, tested arm `pi` (arms present: native, pi)
- `c1_struct` / `seen`: 157 states / 26 worlds, tested arm `native` (arms present: native)
- `blackbox_traj` / `u_same`: 140 states / 140 worlds, tested arm `pi` (arms present: native, pi)
- `blackbox_traj` / `u_new`: 21 states / 21 worlds, tested arm `pi` (arms present: native, pi)
- `blackbox_traj` / `seen`: 157 states / 26 worlds, tested arm `native` (arms present: native)
- `blackbox_id` / `u_same`: 140 states / 140 worlds, tested arm `pi` (arms present: native, pi)
- `blackbox_id` / `u_new`: 21 states / 21 worlds, tested arm `pi` (arms present: native, pi)
- `blackbox_id` / `seen`: 157 states / 26 worlds, tested arm `native` (arms present: native)
- `keep_speed` / `u_same`: 140 states / 140 worlds, tested arm `pi` (arms present: native, pi)
- `keep_speed` / `u_new`: 21 states / 21 worlds, tested arm `pi` (arms present: native, pi)
- `keep_speed` / `seen`: 157 states / 26 worlds, tested arm `native` (arms present: native)
- `pure_physics` / `u_same`: 140 states / 140 worlds, tested arm `pi` (arms present: native, pi)
- `pure_physics` / `u_new`: 21 states / 21 worlds, tested arm `pi` (arms present: native, pi)
- `pure_physics` / `seen`: 157 states / 26 worlds, tested arm `native` (arms present: native)
- `oldhist` / `u_same`: 140 states / 140 worlds, tested arm `pi` (arms present: native, pi)
- `oldhist` / `u_new`: 21 states / 21 worlds, tested arm `pi` (arms present: native, pi)
- `oldhist` / `seen`: 157 states / 26 worlds, tested arm `native` (arms present: native)

## HDV |ds| at 10 s

| model | set | arm | n | median | IQR | p90 | sd | p90 within (median) |
|---|---|---|---|---|---|---|---|---|
| c1_struct | u_same | native | 140 | 2.11 | 0.81 | 3.08 | 0.79 | 13.41 |
| c1_struct | u_same | pi | 140 | 2.06 | 0.81 | 2.93 | 0.63 | 12.15 |
| c1_struct | u_new | native | 21 | 2.14 | 0.89 | 2.79 | 0.71 | 13.08 |
| c1_struct | u_new | pi | 21 | 1.89 | 0.55 | 2.76 | 0.55 | 11.32 |
| c1_struct | seen | native | 157 | 2.08 | 0.99 | 3.55 | 1.11 | 13.72 |
| blackbox_traj | u_same | native | 140 | 2.55 | 1.30 | 4.19 | 1.09 | 17.97 |
| blackbox_traj | u_same | pi | 140 | 2.44 | 1.13 | 3.57 | 0.90 | 16.88 |
| blackbox_traj | u_new | native | 21 | 2.63 | 1.59 | 4.47 | 1.13 | 16.48 |
| blackbox_traj | u_new | pi | 21 | 2.49 | 1.42 | 3.58 | 1.08 | 15.42 |
| blackbox_traj | seen | native | 157 | 2.69 | 1.73 | 5.65 | 1.68 | 18.67 |
| blackbox_id | u_same | native | 140 | 2.06 | 1.22 | 3.59 | 1.15 | 16.20 |
| blackbox_id | u_same | pi | 140 | 1.90 | 0.99 | 3.22 | 0.91 | 15.48 |
| blackbox_id | u_new | native | 21 | 1.97 | 1.39 | 4.41 | 1.40 | 16.24 |
| blackbox_id | u_new | pi | 21 | 1.91 | 0.87 | 3.64 | 1.15 | 16.24 |
| blackbox_id | seen | native | 157 | 2.10 | 1.69 | 7.37 | 2.79 | 19.85 |
| keep_speed | u_same | native | 140 | 3.11 | 2.63 | 5.98 | 2.37 | 46.66 |
| keep_speed | u_same | pi | 140 | 2.95 | 2.38 | 5.72 | 1.94 | 45.32 |
| keep_speed | u_new | native | 21 | 3.05 | 4.09 | 7.22 | 3.92 | 49.08 |
| keep_speed | u_new | pi | 21 | 3.11 | 3.09 | 7.56 | 2.85 | 47.15 |
| keep_speed | seen | native | 157 | 3.30 | 4.29 | 11.46 | 5.25 | 53.19 |
| pure_physics | u_same | native | 140 | 7.31 | 1.48 | 8.91 | 1.24 | 21.14 |
| pure_physics | u_same | pi | 140 | 7.02 | 1.14 | 8.73 | 1.08 | 19.89 |
| pure_physics | u_new | native | 21 | 6.96 | 1.08 | 8.59 | 1.05 | 20.26 |
| pure_physics | u_new | pi | 21 | 6.90 | 0.90 | 7.51 | 0.76 | 17.56 |
| pure_physics | seen | native | 157 | 7.23 | 1.32 | 9.89 | 1.90 | 21.63 |
| oldhist | u_same | native | 140 | 1.82 | 0.39 | 2.46 | 0.41 | 8.83 |
| oldhist | u_same | pi | 140 | 1.78 | 0.39 | 2.31 | 0.36 | 8.45 |
| oldhist | u_new | native | 21 | 1.87 | 0.35 | 2.35 | 0.34 | 8.56 |
| oldhist | u_new | pi | 21 | 1.81 | 0.29 | 2.06 | 0.20 | 7.92 |
| oldhist | seen | native | 157 | 1.87 | 0.40 | 2.53 | 0.51 | 8.97 |

## HDV |ds| at 30 s

| model | set | arm | n | median | IQR | p90 | sd | p90 within (median) |
|---|---|---|---|---|---|---|---|---|
| c1_struct | u_same | native | 140 | 6.34 | 4.35 | 11.29 | 3.57 | 52.99 |
| c1_struct | u_same | pi | 140 | 6.63 | 3.72 | 10.92 | 3.31 | 48.41 |
| c1_struct | u_new | native | 21 | 5.75 | 3.65 | 11.28 | 3.36 | 46.43 |
| c1_struct | u_new | pi | 21 | 5.71 | 3.90 | 13.53 | 3.98 | 46.40 |
| c1_struct | seen | native | 157 | 7.66 | 9.22 | 21.12 | 7.88 | 61.94 |
| blackbox_traj | u_same | native | 140 | 10.03 | 4.62 | 15.32 | 4.49 | 66.19 |
| blackbox_traj | u_same | pi | 140 | 9.93 | 4.34 | 16.31 | 3.92 | 66.88 |
| blackbox_traj | u_new | native | 21 | 10.44 | 3.59 | 15.18 | 3.29 | 46.14 |
| blackbox_traj | u_new | pi | 21 | 10.36 | 4.36 | 13.44 | 3.32 | 55.89 |
| blackbox_traj | seen | native | 157 | 12.15 | 10.04 | 36.04 | 16.33 | 75.31 |
| blackbox_id | u_same | native | 140 | 8.28 | 4.56 | 14.98 | 4.19 | 69.30 |
| blackbox_id | u_same | pi | 140 | 8.05 | 4.48 | 15.90 | 5.40 | 66.11 |
| blackbox_id | u_new | native | 21 | 8.25 | 3.03 | 12.61 | 3.60 | 56.52 |
| blackbox_id | u_new | pi | 21 | 8.27 | 7.97 | 17.14 | 5.38 | 63.59 |
| blackbox_id | seen | native | 157 | 10.72 | 21.05 | 83.39 | 33.80 | 84.08 |
| keep_speed | u_same | native | 140 | 17.77 | 9.93 | 33.46 | 12.03 | 296.95 |
| keep_speed | u_same | pi | 140 | 17.76 | 11.38 | 33.93 | 12.25 | 297.98 |
| keep_speed | u_new | native | 21 | 21.31 | 21.48 | 47.81 | 18.21 | 297.67 |
| keep_speed | u_new | pi | 21 | 20.98 | 19.50 | 45.41 | 19.00 | 314.57 |
| keep_speed | seen | native | 157 | 26.00 | 46.71 | 127.58 | 51.04 | 340.63 |
| pure_physics | u_same | native | 140 | 26.52 | 6.55 | 35.25 | 5.87 | 108.35 |
| pure_physics | u_same | pi | 140 | 26.85 | 7.31 | 36.40 | 5.89 | 110.46 |
| pure_physics | u_new | native | 21 | 25.71 | 7.77 | 34.24 | 6.87 | 106.05 |
| pure_physics | u_new | pi | 21 | 22.91 | 8.06 | 32.25 | 6.68 | 101.54 |
| pure_physics | seen | native | 157 | 30.88 | 19.19 | 88.32 | 30.22 | 145.47 |
| oldhist | u_same | native | 140 | 6.48 | 2.44 | 9.74 | 2.42 | 43.63 |
| oldhist | u_same | pi | 140 | 6.39 | 2.50 | 9.66 | 2.45 | 41.83 |
| oldhist | u_new | native | 21 | 6.27 | 1.52 | 8.33 | 1.47 | 38.19 |
| oldhist | u_new | pi | 21 | 6.91 | 3.19 | 9.71 | 2.32 | 37.47 |
| oldhist | seen | native | 157 | 7.59 | 3.75 | 16.05 | 5.13 | 74.23 |

## HDV |ds| at 60 s

| model | set | arm | n | median | IQR | p90 | sd | p90 within (median) |
|---|---|---|---|---|---|---|---|---|
| c1_struct | u_same | native | 140 | 15.04 | 14.81 | 35.22 | 14.05 | 102.67 |
| c1_struct | u_same | pi | 140 | 16.04 | 14.59 | 34.59 | 13.08 | 98.26 |
| c1_struct | u_new | native | 21 | 15.69 | 27.12 | 49.55 | 18.71 | 105.93 |
| c1_struct | u_new | pi | 21 | 14.39 | 17.79 | 44.64 | 17.04 | 97.21 |
| c1_struct | seen | native | 157 | 28.38 | 44.58 | 97.74 | 44.53 | 229.70 |
| blackbox_traj | u_same | native | 140 | 23.77 | 15.74 | 41.79 | 12.10 | 194.60 |
| blackbox_traj | u_same | pi | 140 | 23.04 | 13.23 | 40.28 | 13.16 | 187.81 |
| blackbox_traj | u_new | native | 21 | 21.43 | 12.75 | 34.72 | 9.83 | 200.50 |
| blackbox_traj | u_new | pi | 21 | 22.18 | 13.69 | 30.88 | 8.45 | 165.51 |
| blackbox_traj | seen | native | 157 | 35.88 | 44.10 | 121.21 | 46.96 | 292.23 |
| blackbox_id | u_same | native | 140 | 24.90 | 16.43 | 47.43 | 14.73 | 117.32 |
| blackbox_id | u_same | pi | 140 | 23.38 | 16.98 | 51.80 | 17.69 | 118.52 |
| blackbox_id | u_new | native | 21 | 24.25 | 16.15 | 55.76 | 14.74 | 130.71 |
| blackbox_id | u_new | pi | 21 | 23.59 | 26.53 | 52.74 | 14.66 | 124.91 |
| blackbox_id | seen | native | 157 | 48.66 | 106.16 | 270.14 | 111.91 | 244.76 |
| keep_speed | u_same | native | 140 | 72.88 | 53.27 | 151.19 | 52.28 | 771.94 |
| keep_speed | u_same | pi | 140 | 73.78 | 54.52 | 143.30 | 53.31 | 775.63 |
| keep_speed | u_new | native | 21 | 78.58 | 69.66 | 177.84 | 67.04 | 800.99 |
| keep_speed | u_new | pi | 21 | 78.69 | 68.28 | 133.52 | 57.71 | 793.41 |
| keep_speed | seen | native | 157 | 123.47 | 190.94 | 423.93 | 146.61 | 859.72 |
| pure_physics | u_same | native | 140 | 57.85 | 29.56 | 97.54 | 26.62 | 416.65 |
| pure_physics | u_same | pi | 140 | 59.11 | 33.58 | 98.71 | 26.69 | 406.91 |
| pure_physics | u_new | native | 21 | 61.06 | 48.98 | 101.25 | 30.96 | 443.63 |
| pure_physics | u_new | pi | 21 | 47.20 | 31.63 | 107.47 | 37.85 | 353.45 |
| pure_physics | seen | native | 157 | 99.33 | 155.93 | 371.55 | 136.06 | 524.11 |
| oldhist | u_same | native | 140 | 14.09 | 11.93 | 30.41 | 15.71 | 129.41 |
| oldhist | u_same | pi | 140 | 14.32 | 11.73 | 31.94 | 15.18 | 151.65 |
| oldhist | u_new | native | 21 | 16.50 | 15.45 | 39.22 | 16.03 | 239.39 |
| oldhist | u_new | pi | 21 | 13.64 | 12.13 | 23.21 | 10.49 | 116.42 |
| oldhist | seen | native | 157 | 21.88 | 31.38 | 88.35 | 43.73 | 298.73 |

## HDV |ds| at 120 s

| model | set | arm | n | median | IQR | p90 | sd | p90 within (median) |
|---|---|---|---|---|---|---|---|---|
| c1_struct | u_same | native | 125 | 91.25 | 160.12 | 364.95 | 153.24 | 421.80 |
| c1_struct | u_same | pi | 122 | 71.83 | 142.30 | 308.51 | 141.54 | 402.55 |
| c1_struct | u_new | native | 19 | 61.32 | 138.68 | 322.70 | 144.98 | 335.45 |
| c1_struct | u_new | pi | 18 | 31.45 | 109.93 | 306.59 | 151.16 | 321.89 |
| c1_struct | seen | native | 151 | 145.16 | 210.41 | 353.05 | 151.84 | 493.81 |
| blackbox_traj | u_same | native | 125 | 137.75 | 252.88 | 443.56 | 145.63 | 526.67 |
| blackbox_traj | u_same | pi | 122 | 157.31 | 282.33 | 439.12 | 155.08 | 533.84 |
| blackbox_traj | u_new | native | 19 | 120.48 | 246.48 | 446.50 | 159.95 | 555.60 |
| blackbox_traj | u_new | pi | 18 | 136.81 | 191.89 | 366.69 | 121.63 | 580.93 |
| blackbox_traj | seen | native | 151 | 192.66 | 233.97 | 430.55 | 152.14 | 545.82 |
| blackbox_id | u_same | native | 125 | 89.51 | 151.43 | 296.49 | 201.45 | 337.03 |
| blackbox_id | u_same | pi | 122 | 96.75 | 133.10 | 275.54 | 115.40 | 317.81 |
| blackbox_id | u_new | native | 19 | 88.01 | 134.22 | 310.41 | 186.78 | 314.88 |
| blackbox_id | u_new | pi | 18 | 74.54 | 136.74 | 344.29 | 153.99 | 359.24 |
| blackbox_id | seen | native | 151 | 218.68 | 329.57 | 634.33 | 225.41 | 554.57 |
| keep_speed | u_same | native | 125 | 1322.81 | 712.81 | 1765.35 | 491.82 | 1813.76 |
| keep_speed | u_same | pi | 122 | 1376.23 | 711.51 | 1736.86 | 496.54 | 1986.21 |
| keep_speed | u_new | native | 19 | 1244.12 | 591.38 | 1774.10 | 380.73 | 2242.09 |
| keep_speed | u_new | pi | 18 | 1421.41 | 671.97 | 1852.10 | 438.05 | 2243.42 |
| keep_speed | seen | native | 151 | 1227.53 | 571.37 | 1774.50 | 377.67 | 2049.25 |
| pure_physics | u_same | native | 121 | 980.53 | 434.63 | 1343.05 | 334.04 | 1314.92 |
| pure_physics | u_same | pi | 119 | 1022.36 | 492.48 | 1382.85 | 339.90 | 1456.62 |
| pure_physics | u_new | native | 19 | 942.79 | 466.86 | 1174.86 | 304.59 | 1438.97 |
| pure_physics | u_new | pi | 18 | 1012.45 | 517.71 | 1374.32 | 342.48 | 1552.57 |
| pure_physics | seen | native | 151 | 1039.71 | 429.30 | 1367.90 | 352.79 | 1661.07 |
| oldhist | u_same | native | 125 | 230.93 | 621.05 | 1135.17 | 524.70 | 742.07 |
| oldhist | u_same | pi | 122 | 233.21 | 694.59 | 1150.87 | 489.97 | 743.90 |
| oldhist | u_new | native | 19 | 75.04 | 335.11 | 1112.76 | 409.34 | 517.97 |
| oldhist | u_new | pi | 18 | 23.47 | 325.11 | 1113.57 | 429.23 | 439.09 |
| oldhist | seen | native | 151 | 129.56 | 317.64 | 830.87 | 365.76 | 700.35 |

## Blow-up rate (HDV median |ds|)

| model | set | arm | rate | blowups / evaluated | excluded |
|---|---|---|---|---|---|
| c1_struct | u_same | native | 0.016 | 2 / 125 | 15 |
| c1_struct | u_same | pi | 0.025 | 3 / 122 | 18 |
| c1_struct | u_new | native | 0.053 | 1 / 19 | 2 |
| c1_struct | u_new | pi | 0.111 | 2 / 18 | 3 |
| c1_struct | seen | native | 0.272 | 41 / 151 | 6 |
| blackbox_traj | u_same | native | 0.024 | 3 / 125 | 15 |
| blackbox_traj | u_same | pi | 0.033 | 4 / 122 | 18 |
| blackbox_traj | u_new | native | 0.000 | 0 / 19 | 2 |
| blackbox_traj | u_new | pi | 0.000 | 0 / 18 | 3 |
| blackbox_traj | seen | native | 0.305 | 46 / 151 | 6 |
| blackbox_id | u_same | native | 0.048 | 6 / 125 | 15 |
| blackbox_id | u_same | pi | 0.066 | 8 / 122 | 18 |
| blackbox_id | u_new | native | 0.053 | 1 / 19 | 2 |
| blackbox_id | u_new | pi | 0.111 | 2 / 18 | 3 |
| blackbox_id | seen | native | 0.444 | 67 / 151 | 6 |
| keep_speed | u_same | native | 0.816 | 102 / 125 | 15 |
| keep_speed | u_same | pi | 0.828 | 101 / 122 | 18 |
| keep_speed | u_new | native | 0.737 | 14 / 19 | 2 |
| keep_speed | u_new | pi | 0.778 | 14 / 18 | 3 |
| keep_speed | seen | native | 0.848 | 128 / 151 | 6 |
| pure_physics | u_same | native | 0.719 | 87 / 121 | 19 |
| pure_physics | u_same | pi | 0.714 | 85 / 119 | 21 |
| pure_physics | u_new | native | 0.632 | 12 / 19 | 2 |
| pure_physics | u_new | pi | 0.556 | 10 / 18 | 3 |
| pure_physics | seen | native | 0.854 | 129 / 151 | 6 |
| oldhist | u_same | native | 0.024 | 3 / 125 | 15 |
| oldhist | u_same | pi | 0.041 | 5 / 122 | 18 |
| oldhist | u_new | native | 0.053 | 1 / 19 | 2 |
| oldhist | u_new | pi | 0.056 | 1 / 18 | 3 |
| oldhist | seen | native | 0.179 | 27 / 151 | 6 |

## Paired model - reference (`c1_struct`), median HDV |ds| difference, bootstrap over worlds

- blackbox_traj / u_same (arm `pi`) @ 10 s: 0.38 [0.24, 0.49] (worlds=140, pairs=140, n_boot=1000)
- blackbox_traj / u_same (arm `pi`) @ 30 s: 3.11 [2.64, 3.75] (worlds=140, pairs=140, n_boot=1000)
- blackbox_traj / u_same (arm `pi`) @ 60 s: 7.33 [5.19, 9.55] (worlds=140, pairs=140, n_boot=1000)
- blackbox_traj / u_same (arm `pi`) @ 120 s: 66.83 [48.54, 93.05] (worlds=122, pairs=122, n_boot=1000)
- blackbox_traj / u_new (arm `pi`) @ 10 s: 0.67 [0.21, 1.17] (worlds=21, pairs=21, n_boot=1000)
- blackbox_traj / u_new (arm `pi`) @ 30 s: 3.11 [1.47, 5.35] (worlds=21, pairs=21, n_boot=1000)
- blackbox_traj / u_new (arm `pi`) @ 60 s: 3.17 [-2.31, 7.99] (worlds=21, pairs=21, n_boot=1000)
- blackbox_traj / u_new (arm `pi`) @ 120 s: 110.40 [65.30, 219.89] (worlds=18, pairs=18, n_boot=1000)
- blackbox_traj / seen (arm `native`) @ 10 s: 0.65 [0.11, 1.00] (worlds=26, pairs=26, n_boot=1000)
- blackbox_traj / seen (arm `native`) @ 30 s: 3.33 [1.35, 5.65] (worlds=26, pairs=26, n_boot=1000)
- blackbox_traj / seen (arm `native`) @ 60 s: 6.84 [4.37, 11.64] (worlds=26, pairs=26, n_boot=1000)
- blackbox_traj / seen (arm `native`) @ 120 s: 92.64 [39.48, 202.32] (worlds=26, pairs=26, n_boot=1000)
- blackbox_id / u_same (arm `pi`) @ 10 s: -0.03 [-0.20, 0.08] (worlds=140, pairs=140, n_boot=1000)
- blackbox_id / u_same (arm `pi`) @ 30 s: 1.74 [0.92, 2.39] (worlds=140, pairs=140, n_boot=1000)
- blackbox_id / u_same (arm `pi`) @ 60 s: 8.71 [6.44, 11.26] (worlds=140, pairs=140, n_boot=1000)
- blackbox_id / u_same (arm `pi`) @ 120 s: 16.82 [8.94, 33.27] (worlds=122, pairs=122, n_boot=1000)
- blackbox_id / u_new (arm `pi`) @ 10 s: 0.18 [-0.16, 0.37] (worlds=21, pairs=21, n_boot=1000)
- blackbox_id / u_new (arm `pi`) @ 30 s: 1.73 [-0.19, 3.74] (worlds=21, pairs=21, n_boot=1000)
- blackbox_id / u_new (arm `pi`) @ 60 s: 7.20 [-0.52, 13.89] (worlds=21, pairs=21, n_boot=1000)
- blackbox_id / u_new (arm `pi`) @ 120 s: 33.73 [14.22, 83.21] (worlds=18, pairs=18, n_boot=1000)
- blackbox_id / seen (arm `native`) @ 10 s: 0.27 [-0.27, 0.47] (worlds=26, pairs=26, n_boot=1000)
- blackbox_id / seen (arm `native`) @ 30 s: 0.91 [-0.70, 3.52] (worlds=26, pairs=26, n_boot=1000)
- blackbox_id / seen (arm `native`) @ 60 s: 10.06 [8.33, 15.00] (worlds=26, pairs=26, n_boot=1000)
- blackbox_id / seen (arm `native`) @ 120 s: 42.13 [19.48, 57.01] (worlds=26, pairs=26, n_boot=1000)
- keep_speed / u_same (arm `pi`) @ 10 s: 0.98 [0.53, 1.26] (worlds=140, pairs=140, n_boot=1000)
- keep_speed / u_same (arm `pi`) @ 30 s: 10.92 [9.53, 13.05] (worlds=140, pairs=140, n_boot=1000)
- keep_speed / u_same (arm `pi`) @ 60 s: 60.11 [52.61, 66.37] (worlds=140, pairs=140, n_boot=1000)
- keep_speed / u_same (arm `pi`) @ 120 s: 1265.08 [1145.36, 1381.78] (worlds=122, pairs=122, n_boot=1000)
- keep_speed / u_new (arm `pi`) @ 10 s: 1.34 [0.79, 2.33] (worlds=21, pairs=21, n_boot=1000)
- keep_speed / u_new (arm `pi`) @ 30 s: 13.94 [6.51, 28.98] (worlds=21, pairs=21, n_boot=1000)
- keep_speed / u_new (arm `pi`) @ 60 s: 53.15 [28.83, 86.30] (worlds=21, pairs=21, n_boot=1000)
- keep_speed / u_new (arm `pi`) @ 120 s: 1289.08 [1018.77, 1646.06] (worlds=18, pairs=18, n_boot=1000)
- keep_speed / seen (arm `native`) @ 10 s: 0.80 [-0.11, 1.54] (worlds=26, pairs=26, n_boot=1000)
- keep_speed / seen (arm `native`) @ 30 s: 12.56 [6.25, 23.56] (worlds=26, pairs=26, n_boot=1000)
- keep_speed / seen (arm `native`) @ 60 s: 62.89 [30.43, 70.30] (worlds=26, pairs=26, n_boot=1000)
- keep_speed / seen (arm `native`) @ 120 s: 1203.35 [1033.97, 1411.95] (worlds=26, pairs=26, n_boot=1000)
- pure_physics / u_same (arm `pi`) @ 10 s: 5.09 [4.76, 5.33] (worlds=140, pairs=140, n_boot=1000)
- pure_physics / u_same (arm `pi`) @ 30 s: 20.07 [18.88, 21.40] (worlds=140, pairs=140, n_boot=1000)
- pure_physics / u_same (arm `pi`) @ 60 s: 41.67 [34.78, 48.40] (worlds=140, pairs=140, n_boot=1000)
- pure_physics / u_same (arm `pi`) @ 120 s: 927.78 [803.77, 1028.97] (worlds=119, pairs=119, n_boot=1000)
- pure_physics / u_new (arm `pi`) @ 10 s: 4.84 [4.48, 5.13] (worlds=21, pairs=21, n_boot=1000)
- pure_physics / u_new (arm `pi`) @ 30 s: 17.50 [14.47, 19.42] (worlds=21, pairs=21, n_boot=1000)
- pure_physics / u_new (arm `pi`) @ 60 s: 32.25 [21.19, 47.89] (worlds=21, pairs=21, n_boot=1000)
- pure_physics / u_new (arm `pi`) @ 120 s: 826.04 [627.03, 1114.68] (worlds=18, pairs=18, n_boot=1000)
- pure_physics / seen (arm `native`) @ 10 s: 4.93 [4.61, 5.49] (worlds=26, pairs=26, n_boot=1000)
- pure_physics / seen (arm `native`) @ 30 s: 18.79 [16.36, 23.50] (worlds=26, pairs=26, n_boot=1000)
- pure_physics / seen (arm `native`) @ 60 s: 41.35 [23.84, 66.01] (worlds=26, pairs=26, n_boot=1000)
- pure_physics / seen (arm `native`) @ 120 s: 881.09 [775.76, 986.36] (worlds=26, pairs=26, n_boot=1000)
- oldhist / u_same (arm `pi`) @ 10 s: -0.24 [-0.32, -0.15] (worlds=140, pairs=140, n_boot=1000)
- oldhist / u_same (arm `pi`) @ 30 s: -0.14 [-0.73, 0.27] (worlds=140, pairs=140, n_boot=1000)
- oldhist / u_same (arm `pi`) @ 60 s: -1.31 [-3.11, 0.32] (worlds=140, pairs=140, n_boot=1000)
- oldhist / u_same (arm `pi`) @ 120 s: 67.33 [19.73, 238.01] (worlds=122, pairs=122, n_boot=1000)
- oldhist / u_new (arm `pi`) @ 10 s: -0.12 [-0.30, 0.23] (worlds=21, pairs=21, n_boot=1000)
- oldhist / u_new (arm `pi`) @ 30 s: 0.62 [-0.77, 1.42] (worlds=21, pairs=21, n_boot=1000)
- oldhist / u_new (arm `pi`) @ 60 s: 0.82 [-8.17, 5.34] (worlds=21, pairs=21, n_boot=1000)
- oldhist / u_new (arm `pi`) @ 120 s: 13.22 [-2.93, 63.27] (worlds=18, pairs=18, n_boot=1000)
- oldhist / seen (arm `native`) @ 10 s: -0.02 [-0.39, 0.19] (worlds=26, pairs=26, n_boot=1000)
- oldhist / seen (arm `native`) @ 30 s: 0.38 [-0.39, 1.54] (worlds=26, pairs=26, n_boot=1000)
- oldhist / seen (arm `native`) @ 60 s: 0.02 [-1.58, 2.84] (worlds=26, pairs=26, n_boot=1000)
- oldhist / seen (arm `native`) @ 120 s: 26.29 [-2.95, 154.73] (worlds=26, pairs=26, n_boot=1000)

## Traffic-state split (HDV |ds| median at 60 s)

t0 state class from `mean t0 speed of non-CAV vehicles with s_down in [-300, 300] m (fallback: all HDVs); free >20, congested 10..20, jam <10 m/s`. The paired column is the median of `model - c1_struct` over the same states (negative favours the model).

### `u_same` (arm `pi`)

| class | n | c1_struct | oldhist | blackbox_traj | blackbox_id | keep_speed | pure_physics | oldhist - c1_struct |
|---|---|---|---|---|---|---|---|---|
| free | 29 | 18.36 | 17.91 | 22.32 | 22.85 | 37.34 | 40.23 | -3.70 |
| congested | 85 | 16.84 | 14.14 | 24.12 | 24.41 | 81.00 | 63.51 | -1.87 |
| jam | 26 | 7.95 | 10.05 | 19.85 | 21.65 | 92.16 | 67.07 | 0.28 |

### `u_new` (arm `pi`)

| class | n | c1_struct | oldhist | blackbox_traj | blackbox_id | keep_speed | pure_physics | oldhist - c1_struct |
|---|---|---|---|---|---|---|---|---|
| free | 6 | 18.43 | 15.09 | 24.74 | 22.41 | 31.95 | 33.77 | 3.47 |
| congested | 11 | 11.41 | 14.35 | 22.18 | 31.96 | 95.16 | 70.11 | 0.82 |
| jam | 4 | 9.90 | 8.09 | 21.05 | 19.01 | 68.65 | 55.41 | -4.47 |

### `seen` (arm `native`)

| class | n | c1_struct | oldhist | blackbox_traj | blackbox_id | keep_speed | pure_physics | oldhist - c1_struct |
|---|---|---|---|---|---|---|---|---|
| free | 7 | 17.97 | 15.90 | 28.63 | 26.47 | 30.00 | 39.80 | -1.77 |
| congested | 14 | 11.45 | 15.64 | 25.55 | 25.28 | 79.85 | 84.06 | 2.68 |
| jam | 5 | 8.43 | 7.04 | 20.54 | 32.27 | 74.21 | 58.65 | 0.12 |

## OLDHIST recovered-parameter check (original vs shifted history)

Full detail: `_out/shift_scores/oldhist/RECOVERY_ORIG_VS_SHIFT.{md,json}`.

- original (~1 valid pre-t0 step): nearest-vType accuracy 0.6937 (n=32750); every vehicle maps to `normal_car` (reproduces the published 0.6937).
- shifted (real 10-step history): nearest-vType accuracy 0.4242 (n=31429); recovered parameters spread across all three vTypes but are not calibrated to the labels (paired accuracy 0.4353 vs 0.6921 on 16381 shared vehicles).
- Conclusion: real history changes the recovered parameters substantially (the encoder no longer collapses), but the oldhist encoder (trajectory-loss only, no driver sidecar) still does not identify the SUMO vType; the hybrid run needs WM-S2 driver-parameter supervision for calibrated recovery.
