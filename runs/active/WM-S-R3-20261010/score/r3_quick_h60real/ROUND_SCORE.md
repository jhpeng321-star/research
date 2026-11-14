# Round-3 score: E:\research-dense\runs\active\WM-S-R3-20261010\box\gpuA\extracted\wms_train_r3\r3_main_a2\full vs E:\research-dense\runs\active\WM-S-C1FAST-20261006\box\full_struct

Generated 2026-10-10T02:36:06Z; mode `quick`; sets `wall_same`; entries `off`; workers 2. CPU only, no training, no SUMO. CAVs teacher-forced from the recorded arm; each checkpoint scored with its own TRAIN_RECEIPT physics. Metric = across-states median of the per-state HDV median |pred-recorded| (arm `wall`, `native` for seen_controls). Negative paired differences favour the candidate.

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
| cand_last | 10s | 20 | 2.36 | 1.13 | 4.04 |
| cand_last | 30s | 20 | 7.16 | 4.01 | 14.92 |
| cand_last | 60s | 20 | 28.31 | 18.37 | 61.94 |
| cand_last | 120s | 19 | 84.22 | 268.51 | 477.35 |
| ref_last | 10s | 20 | 2.10 | 1.41 | 4.59 |
| ref_last | 30s | 20 | 8.59 | 7.30 | 17.12 |
| ref_last | 60s | 20 | 16.80 | 21.07 | 36.86 |
| ref_last | 120s | 19 | 66.38 | 122.12 | 187.38 |
| cand_best | 10s | 20 | 2.19 | 1.58 | 4.49 |
| cand_best | 30s | 20 | 6.78 | 3.85 | 16.29 |
| cand_best | 60s | 20 | 22.22 | 8.30 | 35.59 |
| cand_best | 120s | 19 | 79.42 | 230.40 | 383.19 |

## HDV |dv| (m/s) - median / IQR / p90 across states

### `wall_same` (arm `wall`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| cand_last | 10s | 20 | 0.39 | 0.09 | 0.51 |
| cand_last | 30s | 20 | 0.58 | 0.26 | 0.82 |
| cand_last | 60s | 20 | 1.15 | 0.68 | 1.94 |
| cand_last | 120s | 19 | 3.33 | 5.22 | 13.93 |
| ref_last | 10s | 20 | 0.38 | 0.15 | 0.58 |
| ref_last | 30s | 20 | 0.51 | 0.19 | 0.69 |
| ref_last | 60s | 20 | 0.94 | 0.56 | 1.65 |
| ref_last | 120s | 19 | 4.46 | 5.36 | 9.07 |
| cand_best | 10s | 20 | 0.30 | 0.13 | 0.45 |
| cand_best | 30s | 20 | 0.57 | 0.27 | 0.71 |
| cand_best | 60s | 20 | 1.44 | 0.66 | 2.61 |
| cand_best | 120s | 19 | 6.30 | 8.28 | 15.20 |

## Blow-up rate (|ds|>50 m @60 s and >100 m @120 s)

| set | model | rate | blowups / evaluated | excluded |
|---|---|---:|---:|---:|
| wall_same | cand_last | 0.158 | 3/19 | 1 |
| wall_same | ref_last | 0.000 | 0/19 | 1 |
| wall_same | cand_best | 0.053 | 1/19 | 1 |

## Paired world bootstrap of HDV |ds| (row - column; negative favours the first model)

| comparison | set | h | difference [95% CI over worlds] | worlds | pairs |
|---|---|---|---|---:|---:|
| cand_last-ref_last | wall_same | 10s | 0.16 [0.05, 0.30] | 20 | 20 |
| cand_last-ref_last | wall_same | 30s | 0.17 [-3.52, 1.83] | 20 | 20 |
| cand_last-ref_last | wall_same | 60s | 4.42 [0.17, 16.68] | 20 | 20 |
| cand_last-ref_last | wall_same | 120s | 4.88 [-16.59, 198.21] | 19 | 19 |
| cand_best-ref_last | wall_same | 10s | 0.05 [-0.17, 0.23] | 20 | 20 |
| cand_best-ref_last | wall_same | 30s | 0.74 [-3.23, 2.07] | 20 | 20 |
| cand_best-ref_last | wall_same | 60s | 5.01 [-0.34, 10.67] | 20 | 20 |
| cand_best-ref_last | wall_same | 120s | 12.09 [-41.92, 134.99] | 19 | 19 |
| cand_best-cand_last | wall_same | 10s | -0.01 [-0.34, 0.15] | 20 | 20 |
| cand_best-cand_last | wall_same | 30s | 0.44 [-0.30, 1.10] | 20 | 20 |
| cand_best-cand_last | wall_same | 60s | -3.51 [-9.64, 3.00] | 20 | 20 |
| cand_best-cand_last | wall_same | 120s | 6.80 [-51.96, 21.35] | 19 | 19 |

## Fingerprints (instrumented rollout, quick subset of `wall_same`, arm `wall`)

| fingerprint | cand_last | ref_last |
|---|---|---|
| mean |g*delta_a| (m/s^2) [first 60 s] | 1.12 | 2.19 |
| p95 |g*delta_a| (m/s^2) [first 60 s] | 4.02 | 3.77 |
| cap-bind rate [first 60 s] | 0.0000 | 0.0000 |
| lane-change rate pred (per HDV-step) [first 60 s] | 0.0258 | 0.0115 |
| lane-change rate recorded [first 60 s] | 0.0134 | 0.0134 |
| signed speed bias (m/s; >0 too fast) [first 60 s] | -1.653 | -0.346 |
| mean |g*delta_a| (m/s^2) [full 120 s] | 1.81 | 9.62 |
| p95 |g*delta_a| (m/s^2) [full 120 s] | 7.13 | 55.77 |
| cap-bind rate [full 120 s] | 0.0000 | 0.0000 |
| lane-change rate pred (per HDV-step) [full 120 s] | 0.0529 | 0.1368 |
| lane-change rate recorded [full 120 s] | 0.0144 | 0.0144 |
| signed speed bias (m/s; >0 too fast) [full 120 s] | -2.201 | -0.715 |
| wrong-lane share of |ds| @60 s | 0.236 | 0.279 |
| wrong-lane share of |ds| @120 s | 0.123 | 0.429 |
| jam slice (n / 60s / 120s median |ds|) | 20 / 60s 28.31 / 120s 84.22 | 20 / 60s 16.80 / 120s 64.36 |
| free slice (n / 60s / 120s median |ds|) | 0 / 60s n/a / 120s n/a | 0 / 60s n/a / 120s n/a |
| recorded-slow HDV |ds| median | 60s 38.20 / 120s 84.22 | 60s 32.54 / 120s 54.64 |

## History states

States dir: `E:/rd-wt/ws-r3-evalstates3/_out/wallc1_states_h60_real`; synthetic short-history adapter: False; candidate has history encoder: True.
- `wall_same`: `E:\rd-wt\ws-r3-evalstates3\_out\wallc1_states_h60_real\wall_same.json`

History cut to the last 4.5 s (cand_hcut, primary set; difference = cut - full, positive = the long history helps):

| h | cut median |ds| | cut - full (paired) |
|---|---:|---|
| 10s | 2.40 | 0.00 [-0.03, 0.00] |
| 30s | 7.14 | -0.24 [-0.92, -0.00] |
| 60s | 27.16 | 0.00 [-2.35, 0.00] |
| 120s | 63.23 | 3.94 [-9.41, 16.30] |

## Runtime

scoring 827.8 s; fingerprints 2580.6 s; bootstrap 0.3 s; total 3870.0 s.

