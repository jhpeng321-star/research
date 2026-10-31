# S-WMVALUE frozen decisive offline test

**VERDICT: FAIL** (MAIN21:12 short-prefix-first amendment).
Eight reserved FIT worlds969432..969439;200arm cells;400paired differences.
Starts300/360/420/480/540; full remaining fleet+source delay to absolute900s.
V/D/E unchanged SHAef3e7bce;111causal sensed features; no source queue input.
K8 WM2+mix; identical long stochastic plans truncated only after90s.
World-cluster bootstrap10,000,seed20261031;95% percentile intervals.

| Method | Sign accuracy % [CI] | Spearman [CI] | MAE veh-s [CI] |
|---|---:|---:|---:|
| A: WM to900 | NA | NA | NA |
| B: WM60 + V | 47.25 [38.00, 56.25] | -0.08 [-0.32, 0.21] | 8877.72 [4590.41, 14306.82] |
| C: WM90 + V | 43.25 [35.25, 51.50] | -0.21 [-0.40, -0.00] | 10117.34 [5832.41, 14984.31] |
| D: V + present commands | 58.75 [47.75, 67.75] | 0.29 [0.09, 0.48] | 4460.00 [3161.00, 6183.67] |
| E: FIT arm/time mean | 72.25 [56.00, 86.00] | 0.46 [0.11, 0.72] | 3038.93 [1725.06, 4769.29] |

| Contrast | Sign gain pp [CI] | MAE reduction veh-s [CI] |
|---|---:|---:|
| B-D | -11.50 [-21.75, 2.00] | -4417.72 [-8445.89, -921.72] |
| B-E | -25.00 [-41.25, -4.75] | -5838.80 [-10057.35, -2104.14] |
| C-D | -15.50 [-26.00, -3.00] | -5657.34 [-9214.69, -2159.55] |
| C-E | -29.00 [-45.25, -9.50] | -7078.42 [-10909.94, -3335.07] |

Positive strictly-above-zero lower bounds are required for BOTH sign and MAE
against EACH D/E, then A. Short methods surviving D/E: [].
Neither short method clears the necessary D/E gate; the full A/D/E
conjunction therefore FAILS regardless of A. Long-A completion is unnecessary.
Recommendation: do not plug this frozen V into policy training as a validated fix.
Keep WM-primary as the research goal; diagnose value/rollout mismatch on FIT only.
Preserve SUMO oracle as teacher/reference. These exposed worlds cannot tune a revision.

Real/imagined sensing parity, prefix causality, history, entries/exits, clocks and CRN
passed CPU gates before reserved inference. Only registered worker-driver versions differ.
Existing long outputs supply their own short checkpoints; active jobs were not duplicated.
A is NA because the complete320long-task test has not run; no partial A efficacy claim.
Evidence: tools/wmvalue/evidence/short_test.json and short_arm_predictions.json.
No SUMO, eval/prohibited paths, refit, peer PID signals, policy training or push.
