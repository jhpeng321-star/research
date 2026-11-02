# S-COOPBANDIT-LEARN STEP 3 summary

- dataset: 300 states / 60 worlds (complete panel)
- rule: argmax predicted D among available options, executed only if predicted D - radius > 0, else O0; radius = finite-sample 95% quantile of per-world max signed overprediction (world-grouped 5-outer/4-inner CV); no behavioural threshold.

| model | OOF RMSE (veh-s) | full radius (veh-s) | learned D mean | oracle headroom | learned regret | cooperate rate |
|---|---|---|---|---|---|---|
| GBT | 5262.43 | 11905.25 | 0.00 | 588.48 | 588.48 | 0.000 |
| linear | 7396.94 | 62933.77 | 0.00 | 588.48 | 588.48 | 0.000 |

95% world-bootstrap CIs (GBT regret): [295.05, 961.64]; always-O0 baseline regret equals headroom 588.48.

Per-cell descriptive regret/headroom and feature diagnostics: report.json, feature_importance.csv/png, partial_dependence.csv/png. Live-test callable: policy.json + tools/coopbandit_learn/runtime.py (no sklearn needed remotely).
