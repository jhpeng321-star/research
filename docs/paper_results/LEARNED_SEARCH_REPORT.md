# LEARNED SEARCH REPORT (plan proposer; informational, never auto-activated)

Generated 2026-11-02T11:56:57+00:00 by `python -m tools.coopmwm.learned_search_report` (CPU only; no SUMO/GPU).
**NOT ACTIVATED: requires registered minimum + offline PASS + explicit owner approval via MAIN.**

## Data (read-only copies; TRAIN worlds 986000-986071 only)
- Collection: 8 remote dirs, 3 new files; errors: none.
- Request/response pairs 421; usable B search receipts 111 (skipped {'search without scored perturbations (rejected or unavailable)': 314}; non-TRAIN dropped 0); SUMO entries 1.
- Searches 111 (111 with elites) from 32 worlds; full-pool receipts 111 (curve needs >= 48).
- Exact SUMO verifications: 0 receipts, 0 arm-B graded, 0 joined -> 0 verification-labelled searches from 0 worlds.
- Post-safety command mirror: OFF: no receipt carries a command-mirror flag (live evaluate final_chain=False; handoff: command mirror still OFF) -> all searches pre-fix, weights unchanged; post-fix searches 0.

## Offline check (held-out worlds; vs the search's own initial distribution)
- Train searches 87 (selected step 100); held-out 24 searches / 8 worlds; enough data: False; registered minimum (160 full pools, 32 worlds) met: False.
- PASS: **False**. LL gain -0.585 nats/elite CI [-1.3495363924764276, 0.0893952705412781]; nearest-elite distance change vs initial -0.036 m/s^2 CI [-0.04778956735792596, -0.020515191713118552]; vs centre -0.047 CI [-0.06208196422043807, -0.03313088766641759].
- Checkpoint (local cache) .pt SHA256 `7f7ab001ea720ad3f55aaefede81e292a298b0937561294dff34f4351d66053d`.

## Offline check on VERIFICATION-labelled held-out searches (exact SUMO outcomes; separate)
- Lists 0 / worlds 0; enough data: False; PASS: **False** (harm: None). no held-out verification-labelled search
- Share of verified decisions where a WM rank 5-8 candidate would be exact best (top-8 verification): {'estimable': False, 'verified_lists': 0, 'reason': 'needs >= 10 verified lists with K8 finalists'}.
- Nested best-of-4 vs best-of-8 (exact k=8 lists): {'estimable': False, 'k8_lists': 0}.

## Evaluations needed to reach the plain search's final elite quality (surrogate replay)
- 23 held-out searches / 8 worlds; mean screen budget 169.6; median surrogate LOO R^2 0.40.

| mode | evaluations used (x plain) | reached plain elite quality | evals to it | evals - plain_alt (95% CI) | final elite quality (95% CI) | quality - plain_alt (95% CI) | elite coverage | top-8 quality (95% CI) |
|---|---|---|---|---|---|---|---|---|
| plain | 1.00 | 1.00 | 165.3 | 11.1 [2.1, 24.3] | 1.00 [1.00, 1.00] | -0.04 [-0.26, 0.07] | 1.00 | 1.00 [1.00, 1.00] |
| plain_alt | 1.00 | 0.43 | 154.2 | 0.0 [0.0, 0.0] | 1.04 [0.93, 1.25] | 0.00 [0.00, 0.00] | 0.47 | 1.05 [0.94, 1.25] |
| seeds | 1.00 | 0.57 | 163.7 | 9.5 [3.6, 18.3] | 0.99 [0.95, 1.02] | -0.06 [-0.26, 0.06] | 0.74 | 0.99 [0.95, 1.02] |
| warm | 1.00 | 0.13 | 166.8 | 12.6 [5.9, 23.5] | 0.81 [0.64, 1.00] | -0.24 [-0.34, -0.13] | 0.68 | 0.83 [0.68, 1.02] |
| both | 1.00 | 0.26 | 164.5 | 10.3 [3.6, 22.0] | 0.83 [0.67, 1.03] | -0.21 [-0.31, -0.13] | 0.67 | 0.86 [0.71, 1.04] |
| warm_it1 | 0.33 | 0.00 | 170.6 | 16.3 [8.7, 28.9] | 0.31 [0.17, 0.41] | -0.73 [-0.94, -0.64] | 0.52 | 0.34 [0.21, 0.43] |
| warm_it2 | 0.67 | 0.04 | 168.0 | 13.8 [6.3, 24.6] | 0.55 [0.42, 0.68] | -0.49 [-0.63, -0.40] | 0.62 | 0.59 [0.46, 0.72] |

Elite quality: 0 = centre seed (unchanged policy), 1 = plain 3-iteration final top-E mean. plain_alt (independent plain seed) is the fair reference for savings and calibrates elite coverage. Horizons are identical in every mode (screen K4/H300, final K8/H_final); warm_it1/2 use fewer CEM iterations only. Warm modes keep 20% generic + 10% novelty slices.

## Caveats
- Surrogate replays are noise-free fits of each search's own WM screen pool; they estimate search efficiency, not SUMO benefit.
- Top-k-only service receipts carry no screened pool; the evaluations curve needs `RecordingEvaluator` logging (`<queue_root>/proposer_receipts/<state_id>.json`).
- WM-pool check uses held-out full-pool searches only; the verification check uses held-out searches with exact SUMO branch outcomes (top-4 + policy continues) only. Exact outcomes also train the proposer (listwise ranking + gold likelihood targets, weight 2 per search vs 1 for WM pools).
- The script never activates the proposer and never writes to remote hosts.

## Watch refresh B_u003
- READY: **False** (WM-pool False, verification False); expected saving "none non-inferior"; wall-clock {"screen_share_top4": 0.8917400469543915, "screen_share_top8": 0.8219054163143522, "basis": "saving_fraction ~ gpu_busy_share x screen_share x evaluation_saving (GPU-bound upper bound)", "expected_wall_saving_fraction": 0.0, "expected_wall_saving_s": 0.0, "top8_final_overhead_note": "top-8 doubles the K8 finalists (16 vs 8): screen share 0.892 -> 0.822 of WM futures"}; search-time saving {"available": true, "evaluation_saving_fraction": 0.0, "mean_saved_s_per_search": 0.0, "mean_wall_s_per_search": 720.6918828828832, "saved_s_by_group_size": {"2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0}, "basis": "SEARCH_WALL_S (COOPMWM.md measured) x screen share (top-4) x saving; GPU-bound, final K8 re-score unchanged; top-8 adds 8 finalists x K8 futures"}; nested best-of-4 vs 8 {"estimable": false, "k8_lists": 0}.
