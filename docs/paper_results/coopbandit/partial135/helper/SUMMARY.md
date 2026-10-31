# S-COOPBANDIT-LEARN STEP 3 summary

- dataset: 135 states / 28 worlds (PARTIAL - exploratory)
- rule: argmax predicted D among available options, executed only if predicted D - radius > 0, else O0; radius = finite-sample 95% quantile of per-world max signed overprediction (world-grouped 5-outer/4-inner CV); no behavioural threshold.

| model | OOF RMSE (veh-s) | full radius (veh-s) | learned D mean | oracle headroom | learned regret | cooperate rate |
|---|---|---|---|---|---|---|
| GBT | 4247.30 | 9262.55 | 0.00 | 284.87 | 284.87 | 0.000 |
| linear | 4548.95 | 10617.60 | 0.00 | 284.87 | 284.87 | 0.000 |

95% world-bootstrap CIs (GBT regret): [122.91, 484.80]; always-O0 baseline regret equals headroom 284.87.

Per-cell descriptive regret/headroom and feature diagnostics: report.json, feature_importance.csv/png, partial_dependence.csv/png. Live-test callable: policy.json + tools/coopbandit_learn/runtime.py (no sklearn needed remotely).
