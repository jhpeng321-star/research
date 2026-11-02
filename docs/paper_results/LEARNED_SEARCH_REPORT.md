# LEARNED SEARCH REPORT (plan proposer; informational, never auto-activated)

Generated 2026-11-02T10:46:06+00:00 by `python -m tools.coopmwm.learned_search_report` (CPU only; no SUMO/GPU).
**NOT ACTIVATED: requires registered minimum + offline PASS + explicit owner approval via MAIN.**

## Data (read-only copies; TRAIN worlds 986000-986071 only)
- Collection: 8 remote dirs, 53 new files; errors: none.
- Request/response pairs 416; usable B search receipts 106 (skipped {'search without scored perturbations (rejected or unavailable)': 314}; non-TRAIN dropped 0); SUMO entries 1.
- Searches 106 (106 with elites) from 32 worlds; full-pool receipts 106 (curve needs >= 48).
- Exact SUMO verifications: 0 receipts, 0 arm-B graded, 0 joined -> 0 verification-labelled searches from 0 worlds.
- Post-safety command mirror: OFF: no receipt carries a command-mirror flag (live evaluate final_chain=False; handoff: command mirror still OFF) -> all searches pre-fix, weights unchanged; post-fix searches 0.

## Offline check (held-out worlds; vs the search's own initial distribution)
- Train searches 82 (selected step 100); held-out 24 searches / 8 worlds; enough data: False; registered minimum (160 full pools, 32 worlds) met: False.
- PASS: **False**. LL gain -0.376 nats/elite CI [-1.0034330542402832, 0.31615561768420664]; nearest-elite distance change vs initial -0.040 m/s^2 CI [-0.04768144499177992, -0.031526220558957214]; vs centre -0.051 CI [-0.06579015977760871, -0.036465782018897774].
- Checkpoint (local cache) .pt SHA256 `ddf1c5d2c98b2601c3c29e2437b2e8d38f006f24c5bd918dd54c1708f109440b`.

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
| seeds | 1.00 | 0.48 | 164.8 | 10.6 [4.0, 20.1] | 0.98 [0.94, 1.01] | -0.06 [-0.26, 0.04] | 0.72 | 0.98 [0.95, 1.01] |
| warm | 1.00 | 0.09 | 167.0 | 12.8 [4.7, 23.8] | 0.76 [0.63, 0.95] | -0.28 [-0.37, -0.22] | 0.65 | 0.79 [0.66, 0.98] |
| both | 1.00 | 0.22 | 165.5 | 11.3 [3.9, 22.4] | 0.83 [0.70, 1.02] | -0.21 [-0.30, -0.17] | 0.68 | 0.85 [0.75, 1.03] |
| warm_it1 | 0.33 | 0.00 | 170.6 | 16.3 [8.7, 28.9] | 0.31 [0.16, 0.42] | -0.74 [-0.94, -0.64] | 0.56 | 0.34 [0.21, 0.44] |
| warm_it2 | 0.67 | 0.04 | 168.0 | 13.8 [6.3, 24.6] | 0.55 [0.42, 0.68] | -0.49 [-0.63, -0.41] | 0.62 | 0.59 [0.47, 0.72] |

Elite quality: 0 = centre seed (unchanged policy), 1 = plain 3-iteration final top-E mean. plain_alt (independent plain seed) is the fair reference for savings and calibrates elite coverage. Horizons are identical in every mode (screen K4/H300, final K8/H_final); warm_it1/2 use fewer CEM iterations only. Warm modes keep 20% generic + 10% novelty slices.

## Caveats
- Surrogate replays are noise-free fits of each search's own WM screen pool; they estimate search efficiency, not SUMO benefit.
- Top-k-only service receipts carry no screened pool; the evaluations curve needs `RecordingEvaluator` logging (`<queue_root>/proposer_receipts/<state_id>.json`).
- WM-pool check uses held-out full-pool searches only; the verification check uses held-out searches with exact SUMO branch outcomes (top-4 + policy continues) only. Exact outcomes also train the proposer (listwise ranking + gold likelihood targets, weight 2 per search vs 1 for WM pools).
- The script never activates the proposer and never writes to remote hosts.

## Watch refresh B_u003
- READY: **False** (WM-pool False, verification False); expected saving "none non-inferior"; wall-clock {"screen_share_top4": 0.8907744759522065, "screen_share_top8": 0.8203803140645681, "basis": "saving_fraction ~ gpu_busy_share x screen_share x evaluation_saving (GPU-bound upper bound)", "expected_wall_saving_fraction": 0.0, "expected_wall_saving_s": 0.0, "top8_final_overhead_note": "top-8 doubles the K8 finalists (16 vs 8): screen share 0.891 -> 0.820 of WM futures"}; search-time saving {"available": true, "evaluation_saving_fraction": 0.0, "mean_saved_s_per_search": 0.0, "mean_wall_s_per_search": 718.0848584905665, "saved_s_by_group_size": {"2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0}, "basis": "SEARCH_WALL_S (COOPMWM.md measured) x screen share (top-4) x saving; GPU-bound, final K8 re-score unchanged; top-8 adds 8 finalists x K8 futures"}; nested best-of-4 vs 8 {"estimable": false, "k8_lists": 0}.
