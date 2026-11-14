# RAW_RESULTS index

Raw result files are NOT copied (to avoid duplicating multi-MB files); they stay at their original paths, listed here with
sha256 in `../MANIFESTS/SHA256SUMS.txt`. The one exception is `e1_bank_box4_extracted/bank/`, extracted unchanged from
`e1_bank_box4.tgz` (sha256 a69cc94b...) because FROZEN_BANKS.json (the E1 primary) exists on the laptop only inside that tgz.

Paths relative to `E:/research-dense/experiments/claim_validation_20261113_0231/`:

| Stage | Files |
|---|---|
| E1 pools, Random orders, WM job files | `out/e1/pool/POOL_seed*.json`, `out/e1/order/ORDER_RANDOM_seed*.json`, `out/e1/wm_jobs/` |
| E1 WM orders, candidates, VERIFY_PLAN (box copy) | inside `e1_out_box4.tgz` |
| E1 WM scores (registered two-stage) | `wm_scores/WM_SCORES_seed*.json` (+ `.two_stage.screen/final.json`, logs) |
| E1 verification results | `e1_verify/VERIFY_RESULTS_seed*.json` |
| E1 admission + frozen banks | `deliverables/RAW_RESULTS/e1_bank_box4_extracted/bank/` (from `e1_bank_box4.tgz`) |
| E1 confirmation | `e1_conf/CONFIRM_RESULTS.json`, `e1_conf/E1_ANALYSIS.json` |
| WM-ACT scores (sensitivity) | `wm_act_scores_reuse/` (probe), `wm_act_scores_all/` (used) |
| E2 identity, tuning, selection, confirmation, analysis | `box4_out/ident/IDENT_RAW.json`, `box4_out/e2/*.json` |
| E2 secondaries + Holm (other agent) | `e2_secondary/` |

Post hoc items (relative to `E:/research-dense/runs/active/C4-V7FT-20261017/`): `a69_lcside_confirm/`,
`a68_research60c/a69/`, `a70_smallbank/root/`, `a71_wideadmit/root/`, `a73_lcside20_out/root/`, `a73_search_p20/`.

Not on the laptop: per-episode raw logs (Box 3 / Box 4; some deleted with owner approval, `REG:11638`, `REG:11659`),
WM-ACT verification outputs (running on Box 3), box-side CHARGES for A72 and A69 Box 4 stages.
