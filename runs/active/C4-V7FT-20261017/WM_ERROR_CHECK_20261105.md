# S-WMCHECK — how wrong is the world model?

**Sol · 2026-11-05 · analysis of existing training records only · no SUMO runs or charges**

## Plain-language answer

The world model's selected plans are not reliably better than the continuing policy. When its recorded gain said a candidate would beat that policy, SUMO agreed only about **four times in ten**. Its ranking of how much candidates would help was also weak. The logged gain differences are often hundreds of vehicle-seconds, with some much larger misses.

**We cannot answer “how wrong is it when it is at least 80% confident?” from these records.** None of the recovered, identity-checked pairs contains the paired `P(better)` evidence required by the accepted calibration module. A predicted saving is not a probability. Neither a positive gain, selection into the top candidates, nor `P(best)` would establish 80% confidence in beating the baseline. Raw and calibrated ≥80% success rates, probability reliability, and ECE are therefore **unavailable**, not zero and not evidence of good calibration.

The errors are **strongly seed/situation dependent**, rather than one uniform offset or demonstrably independent random noise. Half the seeds have positive mean error and half negative mean error; several individual seeds have the same error direction for every candidate. The positive overall average error does not establish a universally optimistic WM. A seed-separated validation slice is especially poor for positive-gain predictions.

There is another important limitation: many SUMO windows end before the registered WM minimum horizon. Consequently, the gain-error numbers below measure **disagreement between the logged prediction and the logged realised window**, not a fully isolated, same-horizon WM forecasting error. Restricting to records that meet the minimum horizon does not make the results reassuring, but exact prediction-horizon provenance is still absent.

## 1. Main numbers

Here, **base = the same-snapshot current policy continuing without committed plans**. Predicted gain is the candidate plan's verified `saving_vs_policy`; realised gain is `J_policy_veh_s - J_candidate_veh_s`. Positive gain means improvement; ties count as non-wins. Units are vehicle-seconds of team delay, not percentages.

The main diagnostic contains **542 candidate/outcome pairs**, from **74 complete group receipts**, **32 original training world seeds**, and **401 unique plan hashes**. It includes 401 first-panel rows and 141 additional second-CRN-panel rows. These are correlated observations, not 542 independent worlds.

| Quantity | Result |
|---|---:|
| WM predicted positive gain over base | 221 / 542 |
| SUMO won among those positive predictions | **90 / 221 = 40.7%** |
| SUMO did not win among those positive predictions | 131 / 221 = 59.3% |
| SUMO wins across all paired rows | 198 / 542 = 36.5% |
| Agreement on win versus non-win across all rows | 303 / 542 = 55.9% |
| Positive SUMO outcomes despite non-positive WM predictions | 108 / 321 = 33.6% |
| Median absolute logged gain error | **328.77 veh-s** |
| Bias, mean(predicted gain − realised gain) | **+207.29 veh-s** |
| Spearman correlation of predicted and realised gain | **0.103** |
| Mean predicted / mean realised gain | −37.54 / −244.83 veh-s |
| Rows with overprediction / underprediction | 280 / 262 |
| Usable recorded raw `P(better)` observations | **0** |
| Usable accepted calibrated probabilities | **0** |

A 10,000-resample bootstrap over **original world seeds**, keeping all candidates and CRN replicas together, gives a descriptive 95% interval of **27.1%–53.4%** for positive-prediction agreement and **−88.40 to +537.96 veh-s** for the mean bias. These intervals reflect this selected and available record set; they do not correct its selection or missingness. The bias interval crosses zero.

### Scope and horizon sensitivity

These scopes overlap; do not add their row counts.

| Diagnostic scope | n | Original seeds | SUMO wins / positive WM predictions | Median absolute error, veh-s | Bias, veh-s | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| All intact first panels plus registered all8 second panels | 542 | 32 | 90/221 (40.7%) | 328.77 | +207.29 | 0.103 |
| First panels only | 401 | 32 | 72/165 (43.6%) | 259.18 | +74.82 | 0.123 |
| Module-recognised all8 primary population, both panels | 282 | 18 | 47/112 (42.0%) | 545.44 | +275.63 | 0.082 |
| Realised window meets registered minimum horizon | 222 | 17 | 46/98 (46.9%) | 752.71 | +373.71 | 0.167 |
| Realised window shorter than registered minimum | 320 | 21 | 44/123 (35.8%) | 206.06 | +91.84 | 0.092 |

The accepted minimum is `final_horizon(D) = min(600, max(300, D + 120))`. Checking the longest candidate in each group gives the same 222 minimum-horizon-passing rows as the candidate-specific check. **Passing this lower-bound check does not prove exact horizon equality with the original WM scoring call.** No paired probability/contract envelope attests the original scoring horizon. Recorded realised windows are 150 s (227 rows), 180 s (48), 210 s (24), 240 s (21), 300 s (16), 360 s (43), and 420 s (163).

All 542 rows have `committed_plan_ids == []` in their verified branch specification. This supports the policy-continuation comparator; it does not supply the missing full-horizon probability contract.

## 2. Probability calibration and the ≥80% question

### What the accepted module permits

Used without editing its implementation:

`E:/agent-work/c4-planner-calib/tools/coopmwm/planner_calibration.py`

The module requires a sealed, candidate/snapshot-bound `plan.wm_probability_evidence` envelope with paired WM samples, or an attested paired `p_better` estimate, plus an unchanged full-horizon objective/baseline contract. Its accepted fitting path does not turn `saving_vs_policy` or `p_best` into raw `P(better)`.

An audit of all **607 recovered candidate specification files**, and probability-like keys throughout the captured receipt/result objects, found **no recorded probability evidence**. In particular:

- All 542 main rows have `wm_prediction = None`.
- The 282 all8-primary rows are refused by the module with `planner calibration unavailable: missing paired WM probability evidence`.
- The other 260 main rows are additionally `unregistered-direct`, so they are gain diagnostics rather than an accepted probability-fitting population.
- The 65 intact outcome-selected extended-confirmation rows also lack probability evidence.

| Requested quantity | Usable probability records | Result |
|---|---:|---|
| Raw `P(better) >= 0.8`: SUMO win fraction | 0 | Unavailable |
| Accepted calibrated `P(better) >= 0.8`: SUMO win fraction | 0 | Unavailable |
| Raw probability ECE on seed-separated validation | 0 | Undefined |
| Calibrated probability ECE on seed-separated validation | 0 | Undefined |

The **actual size** of the WM's originally high-confidence bucket is unknown. Zero usable probability records does not mean the WM never made a high-confidence prediction. No replacement confidence was inferred from scalar gains, and no calibrated planner artifact was fitted.

### Requested reliability table by probability decile

All 542 main rows are **unbinnable because probability is missing**. The zero counts below refer only to usable probability observations, not zero candidates or zero observed wins. Empty-bin frequencies and ECE are undefined. Bins are left-inclusive/right-exclusive, except the final bin includes 1.0.

| Probability interval | Raw P observations | Accepted calibrated P observations | Mean P | Observed SUMO win frequency |
|---|---:|---:|---|---|
| [0.0, 0.1) | 0 | 0 | N/A | N/A |
| [0.1, 0.2) | 0 | 0 | N/A | N/A |
| [0.2, 0.3) | 0 | 0 | N/A | N/A |
| [0.3, 0.4) | 0 | 0 | N/A | N/A |
| [0.4, 0.5) | 0 | 0 | N/A | N/A |
| [0.5, 0.6) | 0 | 0 | N/A | N/A |
| [0.6, 0.7) | 0 | 0 | N/A | N/A |
| [0.7, 0.8) | 0 | 0 | N/A | N/A |
| [0.8, 0.9) | 0 | 0 | N/A | N/A |
| [0.9, 1.0] | 0 | 0 | N/A | N/A |

For available probabilities, the module's ECE is the sample-weighted sum of `abs(mean_probability − observed_win_frequency)` over these ten equal-width bins. It was **not evaluated on an empty probability dataset** and was not reported as 0.

## 3. Seed-separated validation

Used the accepted `split_by_seed` helper with its existing settings: **validation fraction 0.25**, **split seed 20261104**, deterministic hash ordering of original `episode.world.seed`. Every candidate, CRN replica, and update of an original seed stays in one partition. These are withheld **training/dev seeds for this analysis**, not any sealed evaluation data.

### Broad gain-diagnostic population: 32 seeds

| Partition | n | Seeds | SUMO wins / positive WM predictions | Median absolute error, veh-s | Bias, veh-s | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Fit-designated seeds | 446 | 24 | 90/195 (46.2%) | 358.39 | +250.14 | 0.229 |
| Validation seeds | 96 | 8 | **0/26 (0.0%)** | 260.30 | +8.20 | **−0.403** |

Validation seeds: **986008, 986044, 986048, 986050, 986052, 986062, 986067, 986069**.

Fit-designated seeds: 986020, 986023, 986026, 986036, 986038, 986040, 986041, 986042, 986043, 986046, 986047, 986049, 986051, 986053, 986058, 986060, 986061, 986063, 986064, 986065, 986066, 986068, 986070, 986071.

No model was fitted to these gain scores. “Fit-designated” identifies the partition the accepted splitter would reserve for fitting; it is not an in-sample probability-calibration result. The validation rows contain 38 SUMO wins overall, but all are among non-positive WM gain predictions. The near-zero average validation bias therefore conceals poor directional/ranking performance.

### All8-primary population: 18 seeds

Applying the same fixed split rule to the module's narrower primary population produces a different partition, shown to make its intended calibration population explicit:

| Partition | n | Seeds | SUMO wins / positive WM predictions | Median absolute error, veh-s | Bias, veh-s | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Fit-designated all8-primary | 186 | 13 | 30/60 (50.0%) | 643.78 | +495.07 | 0.200 |
| Validation all8-primary | 96 | 5 | 17/52 (32.7%) | 491.13 | −149.52 | −0.236 |

All8-primary validation seeds: **986052, 986066, 986067, 986069, 986070**.

All8-primary fit-designated seeds: 986023, 986036, 986041, 986046, 986047, 986051, 986053, 986058, 986060, 986063, 986064, 986065, 986071.

Neither partition can yield raw/calibrated probability reliability or ≥80% statistics without the missing WM probability evidence. Both partition results are reported; neither was selected for a more favourable outcome.

## 4. Breakdown by plan family and recorded type

All main pairs are arm-B, WM-proposed **MPPI** plans. There is no second recorded family and no `posthoc_label` in the recovered specifications, so a comparison between named cooperative motion families is not supported. The recorded origin, numerical niche ID, duration, group size, and regime permit the following descriptive breakdowns.

**For every family/type/seed below, raw and calibrated ≥80% success rates and probability ECE remain N/A because probability evidence is absent.** Gain errors have the mixed-horizon limitation described above. A denominator of 0 means there were no positive own-policy gain predictions in that subgroup.

### Family and origin

| Recorded family/origin | n | SUMO wins / positive WM predictions | Median absolute error, veh-s | Bias, veh-s | Spearman |
|---|---:|---:|---:|---:|---:|
| Family: MPPI, all origins | 542 | 90/221 (40.7%) | 328.77 | +207.29 | 0.103 |
| Origin: sampled | 497 | 82/212 (38.7%) | 328.77 | +198.92 | 0.088 |
| Origin: library | 25 | 5/5 (100.0%) | 259.18 | −101.09 | 0.080 |
| Origin: mappo | 13 | 3/4 (75.0%) | 1215.84 | +337.40 | 0.414 |
| Origin: centre | 7 | 0/0 (N/A) | 1316.65 | +1661.51 | −0.549 |

Origin is the stored plan provenance, not the proposer: these rows all have receipt `proposer == 'wm'`. Five positive library predictions are too few to establish dependable accuracy.

### Stored niche IDs

The niche values are reported as stored IDs; no unrecorded physical interpretation is assigned to them.

| Niche ID | n | SUMO wins / positive WM predictions | Median absolute error, veh-s | Bias, veh-s | Spearman |
|---|---:|---:|---:|---:|---:|
| 0 | 98 | 20/33 (60.6%) | 428.79 | +177.70 | 0.268 |
| 1 | 61 | 13/28 (46.4%) | 481.79 | +67.02 | −0.033 |
| 2 | 52 | 13/27 (48.1%) | 457.31 | +386.13 | −0.008 |
| 3 | 46 | 19/30 (63.3%) | 550.75 | +266.14 | 0.385 |
| Not recorded | 285 | 25/103 (24.3%) | 312.83 | +205.36 | 0.008 |

### Candidate duration

| Duration, s | n | SUMO wins / positive WM predictions | Median absolute error, veh-s | Bias, veh-s | Spearman |
|---|---:|---:|---:|---:|---:|
| 30 | 287 | 50/133 (37.6%) | 183.77 | −52.49 | −0.034 |
| 60 | 58 | 9/14 (64.3%) | 590.12 | +357.93 | 0.445 |
| 90 | 50 | 6/12 (50.0%) | 857.57 | +724.47 | 0.432 |
| 120 | 20 | 5/10 (50.0%) | 376.32 | −262.80 | 0.528 |
| 150 | 27 | 6/18 (33.3%) | 410.26 | +358.88 | −0.186 |
| 180 | 26 | 4/7 (57.1%) | 316.73 | +535.86 | 0.172 |
| 240 | 21 | 3/8 (37.5%) | 884.99 | +892.57 | −0.151 |
| 300 | 53 | 7/19 (36.8%) | 1316.65 | +628.75 | −0.001 |

Longer plans can have longer realised windows, so absolute errors across these groups do not have identical exposure. Duration is not an experimentally isolated cause of error.

### Group size

| Members | n | SUMO wins / positive WM predictions | Median absolute error, veh-s | Bias, veh-s | Spearman |
|---|---:|---:|---:|---:|---:|
| 2 | 183 | 24/62 (38.7%) | 234.40 | +180.99 | 0.432 |
| 3 | 128 | 15/48 (31.3%) | 297.37 | −346.07 | −0.175 |
| 4 | 79 | 20/45 (44.4%) | 1222.83 | +857.26 | 0.101 |
| 5 | 40 | 0/8 (0.0%) | 896.59 | +1092.33 | −0.605 |
| 6 | 112 | 31/58 (53.4%) | 240.69 | +108.13 | 0.315 |

## 5. Breakdown by situation

Group composition uses only the public member-ID prefixes in candidate specifications; demand, penetration, and seed use the receipt's episode-world metadata. No latent vehicle/driver parameter truth is used.

| Situation | n | Seeds | SUMO wins / positive WM predictions | Median absolute error, veh-s | Bias, veh-s | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Mainline-only members | 250 | 18 | 35/115 (30.4%) | 250.90 | +319.45 | 0.187 |
| Includes ramp member | 292 | 19 | 55/106 (51.9%) | 410.26 | +111.26 | −0.017 |
| Recorded regime: ordinary | 478 | 29 | 73/179 (40.8%) | 289.22 | +238.49 | 0.068 |
| Recorded regime: merge-dense | 64 | 3 | 17/42 (40.5%) | 754.00 | −25.73 | 0.381 |
| Demand 1900 | 178 | 7 | 39/73 (53.4%) | 209.19 | +124.90 | 0.421 |
| Demand 2400 | 180 | 13 | 11/56 (19.6%) | 497.11 | +498.38 | −0.209 |
| Demand 2700 | 184 | 12 | 40/92 (43.5%) | 376.32 | +2.24 | 0.195 |
| Penetration 0.05 | 8 | 1 | 0/0 (N/A) | 71.31 | −74.72 | 0.756 |
| Penetration 0.10 | 46 | 3 | 7/21 (33.3%) | 783.93 | +195.81 | 0.288 |
| Penetration 0.20 | 296 | 15 | 56/120 (46.7%) | 177.87 | +14.38 | 0.235 |
| Penetration 0.40 | 192 | 13 | 27/80 (33.8%) | 958.31 | +519.19 | −0.024 |

These are descriptive slices with overlapping seed membership across some group categories. The demand-2400 and mainline-only slices show poor positive-prediction precision, but their differences are confounded by seed, cell, collector history, and horizon. There are only three merge-dense seeds and one 0.05-penetration seed.

## 6. Breakdown by original world seed

All are registered **training** seeds. “V” marks the broad 32-seed population's analysis-validation partition; “F” its fit-designated partition. “Overpredicted” counts rows with `pred − real > 0`. Spearman uses average ranks for ties and is undefined when either variable is constant.

| World seed | Split | n | SUMO wins / positive WM predictions | Median absolute error, veh-s | Bias, veh-s | Spearman | Overpredicted |
|---|---|---:|---:|---:|---:|---:|---:|
| 986008 | V | 8 | 0/0 | 71.31 | −74.72 | 0.756 | 0/8 |
| 986020 | F | 13 | 5/7 | 361.51 | −668.93 | 0.276 | 7/13 |
| 986023 | F | 25 | 2/14 | 835.31 | +958.99 | 0.630 | 22/25 |
| 986026 | F | 8 | 0/0 | 783.93 | −783.94 | N/A | 0/8 |
| 986036 | F | 45 | 14/26 | 191.51 | +127.45 | 0.693 | 29/45 |
| 986038 | F | 15 | 7/15 | 65.94 | +30.23 | −1.000 | 8/15 |
| 986040 | F | 59 | 11/11 | 203.99 | +47.00 | 0.468 | 19/59 |
| 986041 | F | 13 | 0/0 | 0.30 | −0.31 | 0.775 | 0/13 |
| 986042 | F | 28 | 0/12 | 39.39 | −47.50 | −0.880 | 12/28 |
| 986043 | F | 8 | 0/0 | 60.85 | −74.18 | −0.143 | 0/8 |
| 986044 | V | 16 | 0/8 | 263.13 | −71.86 | −0.785 | 8/16 |
| 986046 | F | 16 | 7/14 | 1148.24 | −289.81 | 0.361 | 8/16 |
| 986047 | F | 16 | 2/2 | 491.84 | −515.20 | 0.355 | 0/16 |
| 986048 | V | 8 | 0/0 | 155.57 | −154.50 | 0.615 | 0/8 |
| 986049 | F | 16 | 0/8 | 51.09 | +51.07 | 1.000 | 16/16 |
| 986050 | V | 8 | 0/0 | 154.26 | +38.43 | −0.071 | 5/8 |
| 986051 | F | 16 | 4/8 | 359.09 | +486.53 | 0.457 | 12/16 |
| 986052 | V | 16 | 0/0 | 73.73 | −56.50 | −0.160 | 5/16 |
| 986053 | F | 16 | 11/16 | 273.09 | +280.27 | −0.095 | 10/16 |
| 986058 | F | 8 | 0/0 | 1515.00 | −250.46 | 0.252 | 5/8 |
| 986060 | F | 16 | 0/0 | 3172.02 | +3292.36 | N/A | 16/16 |
| 986061 | F | 8 | 0/8 | 4823.73 | +4690.97 | −0.524 | 8/8 |
| 986062 | V | 8 | 0/8 | 588.20 | +588.20 | 0.082 | 8/8 |
| 986063 | F | 16 | 0/0 | 809.00 | +716.59 | 0.473 | 14/16 |
| 986064 | F | 16 | 2/4 | 1481.90 | +501.51 | 0.024 | 8/16 |
| 986065 | F | 16 | 0/0 | 108.21 | −116.86 | −0.018 | 3/16 |
| 986066 | F | 32 | 9/26 | 637.50 | +679.86 | 0.015 | 21/32 |
| 986067 | V | 16 | 0/10 | 395.26 | +670.54 | −0.391 | 13/16 |
| 986068 | F | 8 | 8/8 | 1536.21 | −1567.87 | 0.167 | 0/8 |
| 986069 | V | 16 | 0/0 | 701.89 | −691.67 | −0.065 | 0/16 |
| 986070 | F | 16 | 8/16 | 2555.55 | −2179.23 | 0.026 | 8/16 |
| 986071 | F | 16 | 0/0 | 1106.98 | +946.95 | 0.628 | 15/16 |

## 7. Systematic or random?

**Best supported answer: substantial seed-conditioned disagreement, with both directions represented; not one validated universal correction.**

- Mean error is positive on **16 seeds** and negative on **16 seeds**. Overall row-weighted bias is positive, but the seed-cluster bootstrap interval includes zero. Only 51.7% of individual rows overpredict, so the positive mean should not be read as an almost-always optimistic WM.
- Every row overpredicts on **986049, 986060, 986061, and 986062**. Every row underpredicts on **986008, 986026, 986041, 986043, 986047, 986048, 986068, and 986069**. Thus 12 seeds exhibit completely consistent within-seed residual signs, although candidates within a seed are correlated.
- On seed **986061**, all 8 positive-gain predictions fail, with mean error **+4690.97 veh-s**. On **986068**, all 8 positive-gain predictions win, but their benefits are underestimated by **1567.87 veh-s on average**. This is not a single global additive offset.
- Within the registered all8 panels, the same 56 positive predictions win **29/56** on the first CRN panel and **18/56** on the second. First-panel mean bias is **−32.77 veh-s**; second-panel bias is **+584.04 veh-s**. Realisation variability and nonuniform panel coverage matter.
- The 60 s and 90 s duration slices have positive mean-error seed-bootstrap intervals of approximately **[+89.94, +911.02]** and **[+123.92, +1308.12] veh-s**. These are exploratory, unadjusted subgroup results, not proof that duration causes the error. Most other examined situation/duration bias intervals cross zero.
- There is only one recorded plan family, so a stable sign difference between distinct named families cannot be established. The origin/niche/group-size results above suggest heterogeneity but have small or confounded subgroups.

These records do not identify how much disagreement is attributable to stochastic SUMO futures, horizon mismatch, systematic WM dynamics error, or selected-candidate distribution shift. They do **not** justify describing all residuals as independent random noise or applying one universal bias correction.

## 8. Sources, exclusions, and limitations

### Existing cpuQ sources inspected

Read only receipt/result JSON and ordinary candidate branch specifications under these explicit roots:

- `/home/ubuntu/coopmwm20261102_warm1/replay_bank/B/sumo_verification/{verifications,branches}`
- `/home/ubuntu/coopmwm20261102_warm1/replay_bank/B/confirmations_all8`
- `/home/ubuntu/coopmwm20261102_warm1/replay_bank/B/confirm_seed2_all8/{verifications,branches}`
- `/home/ubuntu/coopmwm20261102_warm1/replay_bank/B/confirm_extended_all8/{verifications,branches}`
- `/home/ubuntu/coopmwm20261102_warm1/replay_bank/B/confirm_extended_seed2_all8/{verifications,branches}`
- `/home/ubuntu/coopmwm20261104_v2v_coop3/isolated/u*_B`
- `/home/ubuntu/coopmwm20261104_v2v_coop3/isolated_rep/u*_B_r*`

The coop3 replay-bank metadata links back to the warm1 bank; it is not a second independent bank. The top-level warm1 `sumo_verification` directory contains only `SMOKE_PASSED.json`, not another cohort of paired verification receipts. The cpuQ gate-coop3 and launch3-coop3 roots contain relocation/binding metadata rather than an additional results population.

### Acquisition accounting

| Source class | Recovered objects | Treatment |
|---|---:|---|
| First-panel WM verification receipts | 67 | 55 usable, 401 paired rows; 12 failed receipts contribute no rows |
| all8 confirmation manifests | 19 | Hash-checked; expand to complete first/second candidate panels |
| Second-panel receipts linked by those manifests | 19 | 141 additional paired rows; all required candidate specs present |
| Additional unregistered second-panel receipts | 29 | 205 candidate positions excluded: all 205 corresponding second-panel specs unavailable |
| Extended all8 confirmations, both CRN kinds | 22 | 14 complete receipts supply 65 additional matched rows; 8 lack 41 candidate specs; kept outside main cohort |
| coop3 isolated outcome records | 28 | No WM gain/probability forecast for the evaluated intervention; excluded from WM-error pairs |
| coop3 isolated-replica outcome records | 35 | Same limitation; excluded from WM-error pairs |

In total, **607 matched candidate/outcome rows** were extractable from **88 complete group receipts**: the 542-row main cohort plus 65 selected, extended-window rows. The latter are not a comparable calibration population. There were **246 missing candidate-spec reads** and **two legacy confirmation-directory read timeouts** in acquisition; the missing-spec counts reconcile as 205 + 41. No missing outcome, prediction, or spec was imputed.

The selected extended-window subset would give **28/30 = 93.3%** wins among its positive predictions. That number is deliberately excluded from the headline: these candidates were selected using earlier SUMO outcomes and evaluated over extended windows. Pooling them with initial all-candidate panels would make apparent prediction quality substantially more favourable for the wrong reason.

Legacy `confirm_replication/verifications` and `confirm_extended/verifications` were unavailable through disconnected filesystem mounts. Early metadata-only discovery also encountered unavailable mounts; stalled inspection processes were stopped. This is an analysis of the **available intact records**, not a claim that every historical bank verification has been recovered. The 29 unregistered second panels with missing specs are not silently treated as non-wins or used to repair predictions.

### Interpretation limits

1. **Selection bias:** verification records exist for candidates the WM ranked high and sent for checking. They do not represent every imagined candidate or every possible online move. The accepted module's primary population further requires first/second all-candidate all8 panels; the broad gain diagnostic also contains 260 unregistered-direct rows. Outcome-selected extra CRN replicas are excluded before linked-file reads.
2. **No high-confidence evidence:** no genuine candidate-bound raw `P(better)` is recorded. A confident-WM failure rate cannot be inferred from these gains. No probability calibration artifact was fabricated.
3. **Horizon mismatch:** 320/542 realised windows are shorter than the registered minimum; original WM scoring contracts are not recorded. Even the 222-row minimum-horizon subset is not a certified exactly matched-horizon probability dataset.
4. **Comparator specificity:** gains here are versus each snapshot's own continuing policy. Ranking can instead favour improvement over the separately recorded MAPPO comparator. In fact, 321/542 main rows have non-positive own-policy gain, so “WM-selected” does not itself mean “WM predicted beats this base.” The accepted extractor verifies `saving_vs_policy` rather than trusting a legacy comparator fallback.
5. **Dependence and coverage:** candidates and CRN panels share worlds, snapshots, policies, and search history. Seeds have unequal numbers of rows. Only 32 training seeds, one plan family, and the recovered arm-B WM proposals are represented; random-guide arm-R candidates are not WM forecasts.
6. **Incompleteness:** unavailable legacy mounts and missing specs can introduce additional availability bias. The 63 coop3 isolated records are policy-intervention outcomes, not candidate-specific WM prediction/outcome pairs; using their cooperation gains as WM predictions would change the question.
7. **Outcome definition:** “SUMO agreed” means strictly positive team-delay gain. It does not mean statistically significant benefit, passage of a safety/admission rule, or cooperative synergy over solos.

## 9. Reproducibility and execution record

- Calibration worktree branch: **`c4/planner-calib`**; inspected HEAD: **`472db6c0`** (`coopmwm planner calibration: reconcile readiness with independent PASS`). Existing acceptance marker: `T1C_ACCEPTED.json`, referencing `REVIEW_T1C_r1.md`.
- Module SHA-256: `9a80f4b159a26a853934ede9d0e5037b1fde8cc5ca747f671c0af462c2e39c11`.
- Used accepted `load_verified_records`, `examples_from_receipt`, `split_by_seed`, and probability-eligibility checks. A read-only POSIX-path adapter lets the unchanged loader consume captured cpuQ JSON on Windows; original receipt, confirmation, plan, and branch-spec digests remain intact and are checked by the module.
- Main rows deduplicated by receipt/candidate identity; all8/direct copies do not double-count a receipt. Original-seed grouping keeps CRN replicas together. Gain summaries are row-weighted; Spearman uses average ranks for ties.
- Independent verification recomputed gains directly from receipt policy/candidate costs, checked all 542 prediction/plan matches, counted wins, recomputed median absolute error and bias, and independently reproduced Spearman using standard-library rank correlation. **PASS**.
- Seed-bootstrap settings: 10,000 resamples, RNG seed **20261105**, whole original-seed clusters with replacement.
- Captured-source snapshot SHA-256: `67619403b072380a5ec564b901e55068ed30aff41a00a00661818dcd8cf32d2d`.
- Canonically sorted paired-row dataset SHA-256: `3364b40fe58e66e0cd1b4e0c36ad728b636c7a846efb529fad1ed3373c3a979d`.
- Local acquisition, analysis, paired rows, and verification details: **`E:/Temp/opencode/wmcheck_20261105/`** (`remote_collect.py`, `receipts.jsonl`, `analyze.py`, `analysis.json`, `paired_rows.jsonl`, `verify_results.py`, `verification.json`). Recompute locally with `python -B E:/Temp/opencode/wmcheck_20261105/analyze.py` and `python -B E:/Temp/opencode/wmcheck_20261105/verify_results.py`.
- cpuQ operations were read-only JSON/metadata acquisition via SSH; scripts were streamed over stdin, with **no remote output files**. No simulator, training, rollout, GPU job, accounting reservation, or charge was executed.
- Content reads were restricted to explicitly allowlisted train/dev filename seeds and then checked against receipt roles/world seeds. All admitted rows are training seeds in the registered 986000–986071 range. Refused seed ranges: **964000–964199, 962xxx, 977xxx**. No sealed/holdout/heldout/cal_split/chunk5/j4b/inc1 data paths or hetdata_truth/vType/driver/vehicle_params truth files were read. No tapes, snapshot contents, or model checkpoints were opened for the analysis.

**Bottom line:** on the recoverable selected-candidate records, a positive WM gain is a weak signal of actual improvement, and gain ranking transfers poorly to the seed-separated validation slice. The owner’s specific **“≥80% confident”** question remains unidentifiable from these historical records because the required probability evidence was never recorded in them.
