# Round-3 score: E:\research-dense\runs\active\WM-S-C1FAST-20261006\box\full_struct vs E:\research-dense\runs\active\WM-S-C1FAST-20261006\box\full_struct

Generated 2026-10-09T17:59:28Z; mode `quick`; sets `wall_same`; entries `off`; workers 2. CPU only, no training, no SUMO. CAVs teacher-forced from the recorded arm; each checkpoint scored with its own TRAIN_RECEIPT physics. Metric = across-states median of the per-state HDV median |pred-recorded| (arm `wall`, `native` for seen_controls). Negative paired differences favour the candidate.

## Checkpoints

| role | source | sha256 | physics merge_cal |
|---|---|---|---|
| cand_last | E:\research-dense\runs\active\WM-S-C1FAST-20261006\box\full_struct\wm_s.pt | 87ed02e0707e5e4e... | False |
| ref_last | E:\research-dense\runs\active\WM-S-C1FAST-20261006\box\full_struct\wm_s.pt | 87ed02e0707e5e4e... | False |
| cand_stage10 | E:\research-dense\runs\active\WM-S-C1FAST-20261006\box\full_struct\wm_s_stage10.pt | 912c72dc4e25093a... | False |
| cand_stage30 | E:\research-dense\runs\active\WM-S-C1FAST-20261006\box\full_struct\wm_s_stage30.pt | 5aca526e2ce3689e... | False |
| cand_stage60 | E:\research-dense\runs\active\WM-S-C1FAST-20261006\box\full_struct\wm_s_stage60.pt | 5e78de3c7b107772... | False |
| cand_stage120 | E:\research-dense\runs\active\WM-S-C1FAST-20261006\box\full_struct\wm_s_stage120.pt | 89084b90cbf6e9b4... | False |

## HDV |ds| (m) - median / IQR / p90 across states

### `wall_same` (arm `wall`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| cand_last | 10s | 20 | 2.29 | 0.61 | 3.16 |
| cand_last | 30s | 20 | 5.08 | 4.47 | 9.82 |
| cand_last | 60s | 20 | 11.96 | 12.56 | 25.34 |
| cand_last | 120s | 20 | 111.82 | 58.60 | 154.30 |
| ref_last | 10s | 20 | 2.29 | 0.61 | 3.16 |
| ref_last | 30s | 20 | 5.08 | 4.47 | 9.82 |
| ref_last | 60s | 20 | 11.96 | 12.56 | 25.34 |
| ref_last | 120s | 20 | 111.82 | 58.60 | 154.30 |
| cand_stage10 | 10s | 20 | 2.39 | 0.50 | 3.04 |
| cand_stage10 | 30s | 20 | 39.63 | 21.77 | 61.34 |
| cand_stage10 | 60s | 20 | 526.99 | 230.16 | 716.25 |
| cand_stage10 | 120s | 20 | 3259.14 | 6682.03 | 9066.59 |
| cand_stage30 | 10s | 20 | 1.95 | 0.68 | 2.75 |
| cand_stage30 | 30s | 20 | 6.84 | 3.42 | 10.00 |
| cand_stage30 | 60s | 20 | 56.09 | 23.38 | 81.15 |
| cand_stage30 | 120s | 20 | 1897.11 | 1339.58 | 3140.68 |
| cand_stage60 | 10s | 20 | 2.17 | 0.72 | 2.80 |
| cand_stage60 | 30s | 20 | 6.14 | 3.97 | 10.95 |
| cand_stage60 | 60s | 20 | 17.10 | 20.23 | 40.21 |
| cand_stage60 | 120s | 20 | 550.10 | 434.52 | 1037.80 |
| cand_stage120 | 10s | 20 | 2.29 | 0.61 | 3.16 |
| cand_stage120 | 30s | 20 | 5.08 | 4.47 | 9.82 |
| cand_stage120 | 60s | 20 | 11.96 | 12.56 | 25.34 |
| cand_stage120 | 120s | 20 | 111.82 | 58.60 | 154.30 |

## HDV |dv| (m/s) - median / IQR / p90 across states

### `wall_same` (arm `wall`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| cand_last | 10s | 20 | 0.32 | 0.13 | 0.41 |
| cand_last | 30s | 20 | 0.40 | 0.22 | 0.59 |
| cand_last | 60s | 20 | 0.75 | 0.57 | 1.43 |
| cand_last | 120s | 20 | 6.64 | 8.53 | 19.16 |
| ref_last | 10s | 20 | 0.32 | 0.13 | 0.41 |
| ref_last | 30s | 20 | 0.40 | 0.22 | 0.59 |
| ref_last | 60s | 20 | 0.75 | 0.57 | 1.43 |
| ref_last | 120s | 20 | 6.64 | 8.53 | 19.16 |
| cand_stage10 | 10s | 20 | 0.43 | 0.08 | 0.53 |
| cand_stage10 | 30s | 20 | 5.86 | 2.59 | 8.31 |
| cand_stage10 | 60s | 20 | 26.25 | 7.40 | 37.60 |
| cand_stage10 | 120s | 20 | 104.75 | 166.32 | 228.57 |
| cand_stage30 | 10s | 20 | 0.35 | 0.14 | 0.49 |
| cand_stage30 | 30s | 20 | 0.82 | 0.29 | 1.11 |
| cand_stage30 | 60s | 20 | 6.92 | 2.94 | 9.48 |
| cand_stage30 | 120s | 20 | 54.02 | 52.22 | 98.00 |
| cand_stage60 | 10s | 20 | 0.36 | 0.14 | 0.46 |
| cand_stage60 | 30s | 20 | 0.44 | 0.30 | 0.74 |
| cand_stage60 | 60s | 20 | 1.40 | 1.02 | 2.98 |
| cand_stage60 | 120s | 20 | 16.34 | 10.58 | 31.80 |
| cand_stage120 | 10s | 20 | 0.32 | 0.13 | 0.41 |
| cand_stage120 | 30s | 20 | 0.40 | 0.22 | 0.59 |
| cand_stage120 | 60s | 20 | 0.75 | 0.57 | 1.43 |
| cand_stage120 | 120s | 20 | 6.64 | 8.53 | 19.16 |

## Blow-up rate (|ds|>50 m @60 s and >100 m @120 s)

| set | model | rate | blowups / evaluated | excluded |
|---|---|---:|---:|---:|
| wall_same | cand_last | 0.000 | 0/20 | 0 |
| wall_same | ref_last | 0.000 | 0/20 | 0 |
| wall_same | cand_stage10 | 1.000 | 20/20 | 0 |
| wall_same | cand_stage30 | 0.700 | 14/20 | 0 |
| wall_same | cand_stage60 | 0.000 | 0/20 | 0 |
| wall_same | cand_stage120 | 0.000 | 0/20 | 0 |

## Paired world bootstrap of HDV |ds| (row - column; negative favours the first model)

| comparison | set | h | difference [95% CI over worlds] | worlds | pairs |
|---|---|---|---|---:|---:|
| cand_last-ref_last | wall_same | 10s | 0.00 [0.00, 0.00] | 20 | 20 |
| cand_last-ref_last | wall_same | 30s | 0.00 [0.00, 0.00] | 20 | 20 |
| cand_last-ref_last | wall_same | 60s | 0.00 [0.00, 0.00] | 20 | 20 |
| cand_last-ref_last | wall_same | 120s | 0.00 [0.00, 0.00] | 20 | 20 |

## Per-stage checkpoints (quick subset of `wall_same`, 60 s)

| stage | 60 s median | vs ref_last (paired) |
|---|---:|---|
| cand_stage10 | 526.99 | 510.37 [436.68, 623.05] |
| cand_stage30 | 56.09 | 46.69 [38.05, 51.69] |
| cand_stage60 | 17.10 | 7.53 [4.25, 9.84] |
| cand_stage120 | 11.96 | 0.00 [0.00, 0.00] |

## Fingerprints (instrumented rollout, quick subset of `wall_same`, arm `wall`)

| fingerprint | cand_last | ref_last |
|---|---|---|
| mean |g*delta_a| (m/s^2) | 7.46 | 7.46 |
| p95 |g*delta_a| (m/s^2) | 34.20 | 34.20 |
| cap-bind rate | 0.0000 | 0.0000 |
| lane-change rate pred | 0.0849 | 0.0849 |
| lane-change rate recorded | 0.0155 | 0.0155 |
| wrong-lane share 60 s | 0.373 | 0.373 |
| wrong-lane share 120 s | 0.417 | 0.417 |
| signed speed bias (m/s; >0 too fast) | -1.142 | -1.142 |
| jam slice (n / 60s / 120s median |ds|) | 19 / 60s 12.84 / 120s 110.02 | 19 / 60s 12.84 / 120s 110.02 |
| free slice (n / 60s / 120s median |ds|) | 1 / 60s 8.01 / 120s 307.80 | 1 / 60s 8.01 / 120s 307.80 |
| recorded-slow HDV |ds| median | 60s 29.29 / 120s 141.10 | 60s 29.29 / 120s 141.10 |

## History states

States dir: `None`; synthetic short-history adapter: False; candidate has history encoder: False.
- `wall_same`: `E:\rd-wt\ws-wallc1\_out\wallc1_states_v2\wall_same.json`

## Runtime

scoring 793.9 s; fingerprints 304.6 s; bootstrap 0.1 s; total 1127.2 s.

