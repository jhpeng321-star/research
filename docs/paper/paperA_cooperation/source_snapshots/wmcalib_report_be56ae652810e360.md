# S-WMCALIB — frozen-WM effect calibration
**NOT USABLE as policy long-horizon credit.** Complete 40-world fit corpus; no evaluation seed fitted; no SUMO or WM training.
Fit seeds 969400–969439; 4 scored non-native arms = CTRL0/MAPPO/PI/CIRCLES, yielding 3 arm−CTRL0 contrasts/world/horizon.
The inherited 320-task scorer has no WM-native reference; 12 older FIT worlds were unavailable. Panel transfer below is diagnostic across a changed reference/arm contract.
Label: recorded objective_h.json training J (road speed deficit + source queue), matching the SPEEDHEAD contract; cumulative team_delay_vs is a distinct integration, not silently substituted.
All 480 cumulative labels are hash-verified and paired. Mean absolute paired-J differences from cumulative team delay are 0.026/0.326/1.000 veh-s at 120/300/600 s (input_audit.json).
WM2 raw = mean of K8 paired J differences (frozen SPEEDHEAD reduction); median-of-paired-differences sensitivity is also reported.
Copies: original gpuA/gpuR world ownership, fixed before fit; 44 cross-device duplicate tasks differ. Owner-vs-all-gpuR K8 effect mean/max differences: 0.057/3.404, 2.000/60.882, 42.395/597.830 veh-s; no sign changes (canonical_manifest.json).
Features: predicted effect/spread, observable t0=300 counts/queues/speeds/CAV share/entries, arm identity; no driver truth/oracle types. 10-fold world-grouped outer CV; grouped inner ridge alpha.
Slope/isotonic/ridge/small GBM compared; action-stat models are CV-only. WM-only panel variants avoid synthesizing missing starts or unseen-arm identities.
Selected per horizon by minimum CV MAE among no-action-stat calibrators, locked before the complete SPEEDHEAD test; selection optimism in CV is not an independent qualification.
Gate frozen on resume: lower MAE AND higher sign accuracy AND higher Spearman rho than BOTH raw WM2 and arm-mean, at every horizon in both splits.
Each table cell: MAE veh-s / sign accuracy / Pearson r / Spearman rho / within-world arm-pair order accuracy. Smaller J is better.
## World-grouped CV (n=120 contrasts/horizon)
| h s | signal | MAE / sign / r / rho / arm-order |
|---:|---|---|
| 120 | raw WM2 | 263 / 0.633 / 0.103 / 0.207 / 0.642 |
| 120 | raw WM2 median | 254 / 0.625 / 0.116 / 0.225 / 0.692 |
| 120 | arm-mean (no WM) | 185 / 0.700 / 0.270 / 0.144 / 0.708 |
| 120 | ridge +start | 174 / 0.667 / 0.413 / 0.347 / 0.700 |
| 300 | raw WM2 | 1,719 / 0.467 / -0.086 / 0.021 / 0.592 |
| 300 | raw WM2 median | 1,664 / 0.450 / -0.052 / -0.002 / 0.600 |
| 300 | arm-mean (no WM) | 999 / 0.492 / 0.332 / 0.112 / 0.658 |
| 300 | median ridge +start | 865 / 0.692 / 0.602 / 0.555 / 0.725 |
| 600 | raw WM2 | 6,013 / 0.525 / 0.015 / 0.161 / 0.617 |
| 600 | raw WM2 median | 6,043 / 0.500 / 0.060 / 0.158 / 0.658 |
| 600 | arm-mean (no WM) | 3,298 / 0.617 / 0.335 / 0.219 / 0.742 |
| 600 | ridge +start | 3,152 / 0.617 / 0.493 / 0.444 / 0.733 |
## SPEEDHEAD evaluation-only seeds 963880–963887 (n=24 contrasts/horizon)
| h s | signal | MAE / sign / r / rho / arm-order |
|---:|---|---|
| 120 | raw WM2 | 176 / 0.750 / 0.593 / 0.495 / 0.750 |
| 120 | raw WM2 median | 188 / 0.708 / 0.544 / 0.350 / 0.750 |
| 120 | arm-mean (no WM) | 198 / 0.750 / 0.440 / 0.494 / 0.875 |
| 120 | ridge +start | 212 / 0.583 / 0.181 / 0.190 / 0.875 |
| 300 | raw WM2 | 1,399 / 0.625 / 0.654 / 0.489 / 0.792 |
| 300 | raw WM2 median | 1,460 / 0.708 / 0.599 / 0.474 / 0.833 |
| 300 | arm-mean (no WM) | 1,568 / 0.667 / 0.343 / 0.133 / 0.667 |
| 300 | median ridge +start | 1,783 / 0.458 / 0.073 / -0.037 / 0.750 |
| 600 | raw WM2 | 4,701 / 0.458 / 0.422 / 0.071 / 0.750 |
| 600 | raw WM2 median | 4,909 / 0.583 / 0.436 / 0.280 / 0.875 |
| 600 | arm-mean (no WM) | 4,073 / 0.500 / 0.392 / 0.288 / 0.708 |
| 600 | ridge +start | 4,754 / 0.458 / 0.308 / 0.122 / 0.708 |
## 963887 MAPPO stress case (evaluation only)
| h s | real paired J | raw WM2 | arm-mean | CV-selected calibrator |
|---:|---:|---:|---:|---:|
| 120 | -250 | -1 | 27 | 54 |
| 300 | -3,216 | 83 | 7 | -200 |
| 600 | -9,866 | 1,487 | -281 | -2,322 |
Gate by horizon (120/300/600): CV FAIL/PASS/FAIL; SPEEDHEAD FAIL/FAIL/FAIL.
## WMREQUAL3 frozen 36-world validation — diagnostic transfer only
| h s | subset (n) | raw WM2 / no-WM baseline / CV-selected WM-only calibrator: MAE, sign, rho |
|---:|---|---|
| 120 | native-lane (72) | 284, 0.528, -0.245 / 207, 0.250, nan / 149, 0.528, -0.245 |
| 120 | MAPPO diagnostic (36) | 301, 0.556, -0.149 / 153, 0.472, nan / 153, 0.556, -0.149 |
| 300 | native-lane (72) | 1,327, 0.514, -0.120 / 1,221, 0.306, nan / 1,465, 0.500, -0.204 |
| 300 | MAPPO diagnostic (36) | 1,644, 0.500, 0.129 / 1,140, 0.278, nan / 1,467, 0.444, 0.054 |
Panel WM-only choice by fit-world CV MAE: 120s=slope, 300s=GBM wm-only. Native-lane arms=ours_c1/coopoff_c1 (unseen arms, global-mean baseline); MAPPO has unmatched lane-head execution. No 600s panel exists.
Evidence: tools/wmcalib/out/full/{cv,test,transfer}.json (all variants/folds/per-world predictions, including panel); input_audit.json and wmrequal3_availability.json verify inputs. Four-world smoke preserved.
**Verdict: NOT USABLE as policy long-horizon credit; WMREQUAL3 is an out-of-contract diagnostic, not policy-credit qualification.**
