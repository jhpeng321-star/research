# Round-3 score: E:\research-dense\runs\active\WM-S-R3-20261010\box\gpuA\extracted\wms_train_r3\r3_main_a2\full vs E:\research-dense\runs\active\WM-S-C1FAST-20261006\box\full_struct

Generated 2026-10-10T01:31:05Z; mode `quick`; sets `wall_same`; entries `off`; workers 2. CPU only, no training, no SUMO. CAVs teacher-forced from the recorded arm; each checkpoint scored with its own TRAIN_RECEIPT physics. Metric = across-states median of the per-state HDV median |pred-recorded| (arm `wall`, `native` for seen_controls). Negative paired differences favour the candidate.

## Checkpoints

| role | source | sha256 | physics merge_cal |
|---|---|---|---|
| cand_last | E:\research-dense\runs\active\WM-S-R3-20261010\box\gpuA\extracted\wms_train_r3\r3_main_a2\full\wm_s.pt | 09602e395e58ca6c... | True |
| ref_last | E:\research-dense\runs\active\WM-S-C1FAST-20261006\box\full_struct\wm_s.pt | 87ed02e0707e5e4e... | False |
| cand_best | E:\research-dense\runs\active\WM-S-R3-20261010\box\gpuA\extracted\wms_train_r3\r3_main_a2\full\wm_s_best.pt | 9ee1bba1f4573911... | True |
| cand_stage10 | E:\research-dense\runs\active\WM-S-R3-20261010\box\gpuA\extracted\wms_train_r3\r3_main_a2\full\wm_s_stage10.pt | 2b9e85789879d6c3... | True |
| cand_stage30 | E:\research-dense\runs\active\WM-S-R3-20261010\box\gpuA\extracted\wms_train_r3\r3_main_a2\full\wm_s_stage30.pt | 39388947b5f711fa... | True |
| cand_stage60 | E:\research-dense\runs\active\WM-S-R3-20261010\box\gpuA\extracted\wms_train_r3\r3_main_a2\full\wm_s_stage60.pt | 564cb17edc418538... | True |
| cand_stage120 | E:\research-dense\runs\active\WM-S-R3-20261010\box\gpuA\extracted\wms_train_r3\r3_main_a2\full\wm_s_stage120.pt | acd6476512b419b9... | True |

## HDV |ds| (m) - median / IQR / p90 across states

### `wall_same` (arm `wall`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| cand_last | 10s | 20 | 1.89 | 0.81 | 2.74 |
| cand_last | 30s | 20 | 5.51 | 2.51 | 8.25 |
| cand_last | 60s | 20 | 14.80 | 8.80 | 28.19 |
| cand_last | 120s | 20 | 110.77 | 108.13 | 282.00 |
| ref_last | 10s | 20 | 2.29 | 0.61 | 3.16 |
| ref_last | 30s | 20 | 5.08 | 4.47 | 9.82 |
| ref_last | 60s | 20 | 11.96 | 12.56 | 25.34 |
| ref_last | 120s | 20 | 111.82 | 58.60 | 154.30 |
| cand_best | 10s | 20 | 1.96 | 0.57 | 2.58 |
| cand_best | 30s | 20 | 5.46 | 2.25 | 7.64 |
| cand_best | 60s | 20 | 15.50 | 9.40 | 29.05 |
| cand_best | 120s | 20 | 148.01 | 131.51 | 291.95 |
| cand_stage10 | 10s | 20 | 3.71 | 0.82 | 4.16 |
| cand_stage10 | 30s | 20 | 33.24 | 17.85 | 50.52 |
| cand_stage10 | 60s | 20 | 206.57 | 111.47 | 259.55 |
| cand_stage10 | 120s | 20 | 467.68 | 650.73 | 1263.10 |
| cand_stage30 | 10s | 20 | 2.51 | 0.62 | 3.16 |
| cand_stage30 | 30s | 20 | 5.93 | 2.34 | 10.14 |
| cand_stage30 | 60s | 20 | 41.19 | 24.65 | 58.08 |
| cand_stage30 | 120s | 20 | 258.03 | 271.95 | 452.66 |
| cand_stage60 | 10s | 20 | 2.50 | 0.64 | 3.32 |
| cand_stage60 | 30s | 20 | 5.29 | 2.22 | 7.80 |
| cand_stage60 | 60s | 20 | 16.14 | 8.04 | 25.08 |
| cand_stage60 | 120s | 20 | 118.51 | 100.46 | 270.14 |
| cand_stage120 | 10s | 20 | 1.89 | 0.81 | 2.74 |
| cand_stage120 | 30s | 20 | 5.51 | 2.51 | 8.25 |
| cand_stage120 | 60s | 20 | 14.80 | 8.80 | 28.19 |
| cand_stage120 | 120s | 20 | 110.77 | 108.13 | 282.00 |

## HDV |dv| (m/s) - median / IQR / p90 across states

### `wall_same` (arm `wall`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| cand_last | 10s | 20 | 0.32 | 0.06 | 0.43 |
| cand_last | 30s | 20 | 0.39 | 0.20 | 0.57 |
| cand_last | 60s | 20 | 1.16 | 0.52 | 1.48 |
| cand_last | 120s | 20 | 4.84 | 10.98 | 17.29 |
| ref_last | 10s | 20 | 0.32 | 0.13 | 0.41 |
| ref_last | 30s | 20 | 0.40 | 0.22 | 0.59 |
| ref_last | 60s | 20 | 0.75 | 0.57 | 1.43 |
| ref_last | 120s | 20 | 6.64 | 8.53 | 19.16 |
| cand_best | 10s | 20 | 0.28 | 0.07 | 0.32 |
| cand_best | 30s | 20 | 0.43 | 0.22 | 0.83 |
| cand_best | 60s | 20 | 1.14 | 0.74 | 1.81 |
| cand_best | 120s | 20 | 7.32 | 11.25 | 16.99 |
| cand_stage10 | 10s | 20 | 0.64 | 0.14 | 0.77 |
| cand_stage10 | 30s | 20 | 4.02 | 2.11 | 5.88 |
| cand_stage10 | 60s | 20 | 8.88 | 3.27 | 15.64 |
| cand_stage10 | 120s | 20 | 16.65 | 11.44 | 22.72 |
| cand_stage30 | 10s | 20 | 0.37 | 0.08 | 0.41 |
| cand_stage30 | 30s | 20 | 0.50 | 0.22 | 0.71 |
| cand_stage30 | 60s | 20 | 3.72 | 2.44 | 6.25 |
| cand_stage30 | 120s | 20 | 8.95 | 4.75 | 12.39 |
| cand_stage60 | 10s | 20 | 0.28 | 0.11 | 0.40 |
| cand_stage60 | 30s | 20 | 0.52 | 0.26 | 0.75 |
| cand_stage60 | 60s | 20 | 1.28 | 0.71 | 2.10 |
| cand_stage60 | 120s | 20 | 4.71 | 6.66 | 12.06 |
| cand_stage120 | 10s | 20 | 0.32 | 0.06 | 0.43 |
| cand_stage120 | 30s | 20 | 0.39 | 0.20 | 0.57 |
| cand_stage120 | 60s | 20 | 1.16 | 0.52 | 1.48 |
| cand_stage120 | 120s | 20 | 4.84 | 10.98 | 17.29 |

## Blow-up rate (|ds|>50 m @60 s and >100 m @120 s)

| set | model | rate | blowups / evaluated | excluded |
|---|---|---:|---:|---:|
| wall_same | cand_last | 0.050 | 1/20 | 0 |
| wall_same | ref_last | 0.000 | 0/20 | 0 |
| wall_same | cand_best | 0.000 | 0/20 | 0 |
| wall_same | cand_stage10 | 0.800 | 16/20 | 0 |
| wall_same | cand_stage30 | 0.350 | 7/20 | 0 |
| wall_same | cand_stage60 | 0.050 | 1/20 | 0 |
| wall_same | cand_stage120 | 0.050 | 1/20 | 0 |

## Paired world bootstrap of HDV |ds| (row - column; negative favours the first model)

| comparison | set | h | difference [95% CI over worlds] | worlds | pairs |
|---|---|---|---|---:|---:|
| cand_last-ref_last | wall_same | 10s | -0.23 [-0.44, 0.13] | 20 | 20 |
| cand_last-ref_last | wall_same | 30s | 0.38 [-1.46, 1.36] | 20 | 20 |
| cand_last-ref_last | wall_same | 60s | 1.71 [0.55, 5.44] | 20 | 20 |
| cand_last-ref_last | wall_same | 120s | 10.44 [-23.91, 48.30] | 20 | 20 |
| cand_best-ref_last | wall_same | 10s | -0.24 [-0.53, 0.05] | 20 | 20 |
| cand_best-ref_last | wall_same | 30s | 0.06 [-1.48, 0.74] | 20 | 20 |
| cand_best-ref_last | wall_same | 60s | 1.59 [-0.62, 7.82] | 20 | 20 |
| cand_best-ref_last | wall_same | 120s | 12.02 [-20.65, 83.34] | 20 | 20 |
| cand_best-cand_last | wall_same | 10s | -0.15 [-0.26, 0.15] | 20 | 20 |
| cand_best-cand_last | wall_same | 30s | -0.45 [-1.47, 0.31] | 20 | 20 |
| cand_best-cand_last | wall_same | 60s | 0.02 [-0.94, 2.19] | 20 | 20 |
| cand_best-cand_last | wall_same | 120s | -1.01 [-20.78, 7.40] | 20 | 20 |

## Per-stage checkpoints (quick subset of `wall_same`, 60 s)

| stage | 60 s median | vs ref_last (paired) |
|---|---:|---|
| cand_stage10 | 206.57 | 193.52 [122.20, 212.81] |
| cand_stage30 | 41.19 | 27.66 [15.00, 40.85] |
| cand_stage60 | 16.14 | 3.92 [0.65, 6.57] |
| cand_stage120 | 14.80 | 1.71 [0.55, 5.44] |

## Fingerprints (instrumented rollout, quick subset of `wall_same`, arm `wall`)

| fingerprint | cand_last | ref_last |
|---|---|---|
| mean |g*delta_a| (m/s^2) [first 60 s] | 1.06 | 1.76 |
| p95 |g*delta_a| (m/s^2) [first 60 s] | 3.97 | 3.42 |
| cap-bind rate [first 60 s] | 0.0000 | 0.0000 |
| lane-change rate pred (per HDV-step) [first 60 s] | 0.0209 | 0.0098 |
| lane-change rate recorded [first 60 s] | 0.0118 | 0.0118 |
| signed speed bias (m/s; >0 too fast) [first 60 s] | -1.170 | -0.313 |
| mean |g*delta_a| (m/s^2) [full 120 s] | 1.64 | 7.46 |
| p95 |g*delta_a| (m/s^2) [full 120 s] | 6.54 | 34.20 |
| cap-bind rate [full 120 s] | 0.0000 | 0.0000 |
| lane-change rate pred (per HDV-step) [full 120 s] | 0.0446 | 0.0849 |
| lane-change rate recorded [full 120 s] | 0.0155 | 0.0155 |
| signed speed bias (m/s; >0 too fast) [full 120 s] | -2.100 | -1.142 |
| wrong-lane share of |ds| @60 s | 0.328 | 0.373 |
| wrong-lane share of |ds| @120 s | 0.261 | 0.417 |
| jam slice (n / 60s / 120s median |ds|) | 19 / 60s 14.92 / 120s 110.24 | 19 / 60s 12.84 / 120s 110.02 |
| free slice (n / 60s / 120s median |ds|) | 1 / 60s 8.62 / 120s 205.92 | 1 / 60s 8.01 / 120s 307.80 |
| recorded-slow HDV |ds| median | 60s 26.35 / 120s 144.83 | 60s 29.29 / 120s 141.10 |

## History states

States dir: `E:/rd-wt/ws-r3-evalstates3/_out/wallc1_states_h60`; synthetic short-history adapter: False; candidate has history encoder: True.
- `wall_same`: `E:\rd-wt\ws-r3-evalstates3\_out\wallc1_states_h60\wall_same.json`

## Runtime

scoring 1416.2 s; fingerprints 357.8 s; bootstrap 0.8 s; total 1807.3 s.

