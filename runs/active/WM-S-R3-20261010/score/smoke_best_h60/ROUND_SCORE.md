# Round-3 score: E:\rd-wt\ws-r3-int\_out\r3_smoke\best vs E:\research-dense\runs\active\WM-S-C1FAST-20261006\box\full_struct

Generated 2026-10-09T18:42:29Z; mode `quick`; sets `wall_same`; entries `off`; workers 2. CPU only, no training, no SUMO. CAVs teacher-forced from the recorded arm; each checkpoint scored with its own TRAIN_RECEIPT physics. Metric = across-states median of the per-state HDV median |pred-recorded| (arm `wall`, `native` for seen_controls). Negative paired differences favour the candidate.

> NOTE: candidate has no wm_s.pt: wm_s_best.pt used as cand_last (cand_best not scored separately)

> NOTE: candidate has no TRAIN_RECEIPT.json: physics flags taken from the reference receipt (merge_cal etc. may be wrong)

## Checkpoints

| role | source | sha256 | physics merge_cal |
|---|---|---|---|
| cand_last | E:\rd-wt\ws-r3-int\_out\r3_smoke\best\wm_s_best.pt | 8c26d99fbbe4ba15... | False |
| ref_last | E:\research-dense\runs\active\WM-S-C1FAST-20261006\box\full_struct\wm_s.pt | 87ed02e0707e5e4e... | False |

## HDV |ds| (m) - median / IQR / p90 across states

### `wall_same` (arm `wall`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| cand_last | 10s | 20 | 4.38 | 0.54 | 5.26 |
| cand_last | 30s | 20 | 17.77 | 6.63 | 23.51 |
| cand_last | 60s | 20 | 51.84 | 23.80 | 74.34 |
| cand_last | 120s | 20 | 264.84 | 229.66 | 508.05 |
| ref_last | 10s | 20 | 2.29 | 0.61 | 3.16 |
| ref_last | 30s | 20 | 5.08 | 4.47 | 9.82 |
| ref_last | 60s | 20 | 11.96 | 12.56 | 25.34 |
| ref_last | 120s | 20 | 111.82 | 58.60 | 154.30 |

## HDV |dv| (m/s) - median / IQR / p90 across states

### `wall_same` (arm `wall`)

| model | h | n | median | IQR | p90 |
|---|---|---:|---:|---:|---:|
| cand_last | 10s | 20 | 0.74 | 0.15 | 0.89 |
| cand_last | 30s | 20 | 0.89 | 0.34 | 1.40 |
| cand_last | 60s | 20 | 2.73 | 1.98 | 3.54 |
| cand_last | 120s | 20 | 11.07 | 7.21 | 17.71 |
| ref_last | 10s | 20 | 0.32 | 0.13 | 0.41 |
| ref_last | 30s | 20 | 0.40 | 0.22 | 0.59 |
| ref_last | 60s | 20 | 0.75 | 0.57 | 1.43 |
| ref_last | 120s | 20 | 6.64 | 8.53 | 19.16 |

## Blow-up rate (|ds|>50 m @60 s and >100 m @120 s)

| set | model | rate | blowups / evaluated | excluded |
|---|---|---:|---:|---:|
| wall_same | cand_last | 0.450 | 9/20 | 0 |
| wall_same | ref_last | 0.000 | 0/20 | 0 |

## Paired world bootstrap of HDV |ds| (row - column; negative favours the first model)

| comparison | set | h | difference [95% CI over worlds] | worlds | pairs |
|---|---|---|---|---:|---:|
| cand_last-ref_last | wall_same | 10s | 2.16 [1.93, 2.68] | 20 | 20 |
| cand_last-ref_last | wall_same | 30s | 11.21 [8.55, 13.52] | 20 | 20 |
| cand_last-ref_last | wall_same | 60s | 33.50 [25.66, 45.81] | 20 | 20 |
| cand_last-ref_last | wall_same | 120s | 174.62 [85.51, 220.28] | 20 | 20 |

## Fingerprints (instrumented rollout, quick subset of `wall_same`, arm `wall`)

| fingerprint | cand_last | ref_last |
|---|---|---|
| mean |g*delta_a| (m/s^2) [first 60 s] | 0.52 | 1.76 |
| p95 |g*delta_a| (m/s^2) [first 60 s] | 1.12 | 3.42 |
| cap-bind rate [first 60 s] | 0.0000 | 0.0000 |
| lane-change rate pred (per HDV-step) [first 60 s] | 0.0009 | 0.0098 |
| lane-change rate recorded [first 60 s] | 0.0118 | 0.0118 |
| signed speed bias (m/s; >0 too fast) [first 60 s] | -0.357 | -0.313 |
| mean |g*delta_a| (m/s^2) [full 120 s] | 0.87 | 7.46 |
| p95 |g*delta_a| (m/s^2) [full 120 s] | 2.02 | 34.20 |
| cap-bind rate [full 120 s] | 0.0000 | 0.0000 |
| lane-change rate pred (per HDV-step) [full 120 s] | 0.0011 | 0.0849 |
| lane-change rate recorded [full 120 s] | 0.0155 | 0.0155 |
| signed speed bias (m/s; >0 too fast) [full 120 s] | -1.341 | -1.142 |
| wrong-lane share of |ds| @60 s | 0.365 | 0.373 |
| wrong-lane share of |ds| @120 s | 0.293 | 0.417 |
| jam slice (n / 60s / 120s median |ds|) | 19 / 60s 52.60 / 120s 257.41 | 19 / 60s 12.84 / 120s 110.02 |
| free slice (n / 60s / 120s median |ds|) | 1 / 60s 51.08 / 120s 510.69 | 1 / 60s 8.01 / 120s 307.80 |
| recorded-slow HDV |ds| median | 60s 42.78 / 120s 287.46 | 60s 29.29 / 120s 141.10 |

## History states

States dir: `E:/rd-wt/ws-r3-evalstates3/_out/wallc1_states_h60`; synthetic short-history adapter: False; candidate has history encoder: True.
- `wall_same`: `E:\rd-wt\ws-r3-evalstates3\_out\wallc1_states_h60\wall_same.json`

History cut to the last 4.5 s (cand_hcut, primary set; difference = cut - full, positive = the long history helps):

| h | cut median |ds| | cut - full (paired) |
|---|---:|---|
| 10s | 4.38 | 0.00 [0.00, 0.00] |
| 30s | 17.77 | 0.00 [0.00, 0.00] |
| 60s | 51.84 | 0.00 [0.00, 0.00] |
| 120s | 264.84 | 0.00 [0.00, 0.00] |

## Runtime

scoring 584.5 s; fingerprints 910.2 s; bootstrap 0.2 s; total 1768.9 s.

