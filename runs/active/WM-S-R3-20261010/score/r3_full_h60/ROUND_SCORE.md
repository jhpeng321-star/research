# Round-3 score: E:\research-dense\runs\active\WM-S-R3-20261010\box\gpuA\extracted\wms_train_r3\r3_main_a2\full vs E:\research-dense\runs\active\WM-S-C1FAST-20261006\box\full_struct

Generated 2026-10-10T03:32:29Z; mode `full`; sets `wall_same,wall_pilot`; entries `off`; workers 2. CPU only, no training, no SUMO. CAVs teacher-forced from the recorded arm; each checkpoint scored with its own TRAIN_RECEIPT physics. Metric = across-states median of the per-state HDV median |pred-recorded| (arm `wall`, `native` for seen_controls). Negative paired differences favour the candidate.

## Checkpoints

| role | source | sha256 | physics merge_cal |
|---|---|---|---|
| cand_last | E:\research-dense\runs\active\WM-S-R3-20261010\box\gpuA\extracted\wms_train_r3\r3_main_a2\full\wm_s.pt | 09602e395e58ca6c... | True |
| ref_last | E:\research-dense\runs\active\WM-S-C1FAST-20261006\box\full_struct\wm_s.pt | 87ed02e0707e5e4e... | False |
| cand_best | E:\research-dense\runs\active\WM-S-R3-20261010\box\gpuA\extracted\wms_train_r3\r3_main_a2\full\wm_s_best.pt | 9ee1bba1f4573911... | True |

## HDV |ds| (m) - median / IQR / p90 across states

### `wall_same` (arm `wall`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| cand_last | 10s | 67 | 1.72 | 0.75 | 2.78 |
| cand_last | 30s | 67 | 4.71 | 2.66 | 8.49 |
| cand_last | 60s | 67 | 13.39 | 10.49 | 27.70 |
| cand_last | 120s | 66 | 131.89 | 122.33 | 289.19 |
| ref_last | 10s | 67 | 2.04 | 0.85 | 2.89 |
| ref_last | 30s | 67 | 5.07 | 3.01 | 10.54 |
| ref_last | 60s | 67 | 8.71 | 10.40 | 27.54 |
| ref_last | 120s | 66 | 99.84 | 84.96 | 217.67 |
| cand_best | 10s | 67 | 1.77 | 0.75 | 2.72 |
| cand_best | 30s | 67 | 4.88 | 2.52 | 8.37 |
| cand_best | 60s | 67 | 14.89 | 12.00 | 29.08 |
| cand_best | 120s | 66 | 129.78 | 178.10 | 292.96 |

### `wall_pilot` (arm `wall`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| cand_last | 10s | 53 | 1.65 | 0.73 | 2.97 |
| cand_last | 30s | 53 | 4.35 | 2.42 | 8.60 |
| cand_last | 60s | 53 | 11.87 | 13.27 | 28.31 |
| cand_last | 120s | 52 | 110.77 | 128.90 | 286.12 |
| ref_last | 10s | 53 | 2.06 | 0.81 | 2.73 |
| ref_last | 30s | 53 | 5.06 | 3.34 | 9.77 |
| ref_last | 60s | 53 | 8.34 | 10.09 | 24.80 |
| ref_last | 120s | 52 | 96.76 | 102.85 | 241.01 |
| cand_best | 10s | 53 | 1.71 | 0.76 | 2.92 |
| cand_best | 30s | 53 | 4.28 | 2.44 | 7.41 |
| cand_best | 60s | 53 | 12.97 | 11.39 | 21.17 |
| cand_best | 120s | 52 | 112.39 | 154.21 | 294.96 |

## HDV |dv| (m/s) - median / IQR / p90 across states

### `wall_same` (arm `wall`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| cand_last | 10s | 67 | 0.31 | 0.10 | 0.38 |
| cand_last | 30s | 67 | 0.33 | 0.21 | 0.56 |
| cand_last | 60s | 67 | 1.04 | 0.75 | 2.24 |
| cand_last | 120s | 66 | 7.42 | 12.15 | 17.18 |
| ref_last | 10s | 67 | 0.27 | 0.14 | 0.44 |
| ref_last | 30s | 67 | 0.40 | 0.20 | 0.59 |
| ref_last | 60s | 67 | 0.72 | 0.41 | 1.15 |
| ref_last | 120s | 66 | 6.94 | 12.61 | 22.92 |
| cand_best | 10s | 67 | 0.25 | 0.07 | 0.33 |
| cand_best | 30s | 67 | 0.41 | 0.27 | 0.68 |
| cand_best | 60s | 67 | 0.97 | 0.86 | 1.72 |
| cand_best | 120s | 66 | 8.12 | 13.33 | 17.58 |

### `wall_pilot` (arm `wall`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| cand_last | 10s | 53 | 0.30 | 0.11 | 0.39 |
| cand_last | 30s | 53 | 0.33 | 0.23 | 0.55 |
| cand_last | 60s | 53 | 0.89 | 0.73 | 1.50 |
| cand_last | 120s | 52 | 9.05 | 10.84 | 15.58 |
| ref_last | 10s | 53 | 0.24 | 0.12 | 0.43 |
| ref_last | 30s | 53 | 0.37 | 0.17 | 0.54 |
| ref_last | 60s | 53 | 0.69 | 0.41 | 1.31 |
| ref_last | 120s | 52 | 7.12 | 11.40 | 22.57 |
| cand_best | 10s | 53 | 0.24 | 0.08 | 0.33 |
| cand_best | 30s | 53 | 0.40 | 0.20 | 0.69 |
| cand_best | 60s | 53 | 0.92 | 0.72 | 1.72 |
| cand_best | 120s | 52 | 9.91 | 12.54 | 17.92 |

## Blow-up rate (|ds|>50 m @60 s and >100 m @120 s)

| set | model | rate | blowups / evaluated | excluded |
|---|---|---:|---:|---:|
| wall_same | cand_last | 0.030 | 2/66 | 1 |
| wall_same | ref_last | 0.000 | 0/66 | 1 |
| wall_same | cand_best | 0.000 | 0/66 | 1 |
| wall_pilot | cand_last | 0.038 | 2/52 | 1 |
| wall_pilot | ref_last | 0.000 | 0/52 | 1 |
| wall_pilot | cand_best | 0.000 | 0/52 | 1 |

## Paired world bootstrap of HDV |ds| (row - column; negative favours the first model)

| comparison | set | h | difference [95% CI over worlds] | worlds | pairs |
|---|---|---|---|---:|---:|
| cand_last-ref_last | wall_same | 10s | -0.14 [-0.32, 0.08] | 29 | 29 |
| cand_last-ref_last | wall_same | 30s | 0.05 [-0.99, 1.26] | 29 | 29 |
| cand_last-ref_last | wall_same | 60s | 1.53 [-0.97, 4.43] | 29 | 29 |
| cand_last-ref_last | wall_same | 120s | 39.32 [-1.66, 64.48] | 29 | 29 |
| cand_last-ref_last | wall_pilot | 10s | -0.10 [-0.25, 0.14] | 27 | 27 |
| cand_last-ref_last | wall_pilot | 30s | 0.05 [-1.62, 1.18] | 27 | 27 |
| cand_last-ref_last | wall_pilot | 60s | 1.53 [-0.97, 6.73] | 27 | 27 |
| cand_last-ref_last | wall_pilot | 120s | 35.59 [-1.66, 73.99] | 27 | 27 |
| cand_best-ref_last | wall_same | 10s | -0.17 [-0.42, 0.06] | 29 | 29 |
| cand_best-ref_last | wall_same | 30s | -0.15 [-0.83, 0.60] | 29 | 29 |
| cand_best-ref_last | wall_same | 60s | 1.70 [-0.33, 3.88] | 29 | 29 |
| cand_best-ref_last | wall_same | 120s | 23.87 [-2.63, 69.74] | 29 | 29 |
| cand_best-ref_last | wall_pilot | 10s | -0.17 [-0.41, -0.06] | 27 | 27 |
| cand_best-ref_last | wall_pilot | 30s | -0.15 [-0.86, 0.60] | 27 | 27 |
| cand_best-ref_last | wall_pilot | 60s | 1.56 [-0.33, 3.88] | 27 | 27 |
| cand_best-ref_last | wall_pilot | 120s | 17.18 [-15.96, 48.44] | 27 | 27 |
| cand_best-cand_last | wall_same | 10s | -0.12 [-0.27, 0.11] | 29 | 29 |
| cand_best-cand_last | wall_same | 30s | -0.50 [-0.88, 0.20] | 29 | 29 |
| cand_best-cand_last | wall_same | 60s | 0.01 [-2.49, 1.22] | 29 | 29 |
| cand_best-cand_last | wall_same | 120s | 4.88 [-3.34, 9.01] | 29 | 29 |
| cand_best-cand_last | wall_pilot | 10s | -0.12 [-0.27, 0.11] | 27 | 27 |
| cand_best-cand_last | wall_pilot | 30s | -0.50 [-0.88, 0.20] | 27 | 27 |
| cand_best-cand_last | wall_pilot | 60s | -0.06 [-3.25, 0.69] | 27 | 27 |
| cand_best-cand_last | wall_pilot | 120s | -0.97 [-11.52, 7.86] | 27 | 27 |

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
- `wall_pilot`: `E:\rd-wt\ws-r3-evalstates3\_out\wallc1_states_h60\wall_pilot.json`

## Runtime

scoring 3337.7 s; fingerprints 0.0 s; bootstrap 0.9 s; total 3381.1 s.

