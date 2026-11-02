# LEARNED SEARCH REPORT (plan proposer; informational, never auto-activated)

Generated 2026-11-02T08:48:45+00:00 by `python -m tools.coopmwm.learned_search_report` (CPU only; no SUMO/GPU).
**NOT ACTIVATED: requires registered minimum + offline PASS + explicit owner approval via MAIN.**

## Data (read-only copies; TRAIN worlds 986000-986071 only)
- Collection: 8 remote dirs, 3 new files; errors: none.
- Request/response pairs 409; usable B search receipts 97 (skipped {'search without scored perturbations (rejected or unavailable)': 314}; non-TRAIN dropped 0); SUMO entries 1.
- Searches 97 (97 with elites) from 32 worlds; full-pool receipts 97 (curve needs >= 48).
- Exact SUMO verifications: 0 receipts, 0 arm-B graded, 0 joined -> 0 verification-labelled searches from 0 worlds.
- Post-safety command mirror: OFF: no receipt carries a command-mirror flag (live evaluate final_chain=False; handoff: command mirror still OFF) -> all searches pre-fix, weights unchanged; post-fix searches 0.

## Offline check (held-out worlds; vs the search's own initial distribution)
- Train searches 73 (selected step 100); held-out 24 searches / 8 worlds; enough data: False; registered minimum (160 full pools, 32 worlds) met: False.
- PASS: **False**. LL gain -0.381 nats/elite CI [-1.0514364398862297, 0.32397325992058507]; nearest-elite distance change vs initial -0.037 m/s^2 CI [-0.0488421509273341, -0.02370300222195241]; vs centre -0.048 CI [-0.06431873927859683, -0.032269894388194595].
- Checkpoint (local cache) .pt SHA256 `fee3ff71a0a0ed6efd924db6a1d3492d3243791b2477f9c326e318aac8322e40`.

## Offline check on VERIFICATION-labelled held-out searches (exact SUMO outcomes; separate)
- Lists 0 / worlds 0; enough data: False; PASS: **False** (harm: None). no held-out verification-labelled search
- Share of verified decisions where a WM rank 5-8 candidate would be exact best (top-8 verification): {'estimable': False, 'verified_lists': 0, 'reason': 'needs >= 10 verified lists with K8 finalists'}.

## Evaluations needed to reach the plain search's final elite quality (surrogate replay)
- 23 held-out searches / 8 worlds; mean screen budget 169.6; median surrogate LOO R^2 0.40.

| mode | evaluations used (x plain) | reached plain elite quality | evals to it | evals - plain_alt (95% CI) | final elite quality (95% CI) | quality - plain_alt (95% CI) | elite coverage | top-8 quality (95% CI) |
|---|---|---|---|---|---|---|---|---|
| plain | 1.00 | 1.00 | 165.3 | 11.1 [2.1, 24.3] | 1.00 [1.00, 1.00] | -0.04 [-0.26, 0.07] | 1.00 | 1.00 [1.00, 1.00] |
| plain_alt | 1.00 | 0.43 | 154.2 | 0.0 [0.0, 0.0] | 1.04 [0.93, 1.25] | 0.00 [0.00, 0.00] | 0.47 | 1.05 [0.94, 1.25] |
| seeds | 1.00 | 0.52 | 164.7 | 10.5 [3.7, 20.1] | 0.98 [0.95, 1.01] | -0.06 [-0.26, 0.05] | 0.74 | 0.98 [0.96, 1.01] |
| warm | 1.00 | 0.04 | 169.6 | 15.4 [8.0, 26.9] | 0.75 [0.64, 0.86] | -0.30 [-0.42, -0.22] | 0.66 | 0.77 [0.68, 0.88] |
| both | 1.00 | 0.17 | 166.7 | 12.4 [5.2, 23.3] | 0.76 [0.66, 0.87] | -0.28 [-0.42, -0.20] | 0.66 | 0.78 [0.68, 0.88] |
| warm_it1 | 0.33 | 0.00 | 170.6 | 16.3 [8.7, 28.9] | 0.33 [0.21, 0.42] | -0.72 [-0.92, -0.62] | 0.51 | 0.36 [0.25, 0.45] |
| warm_it2 | 0.67 | 0.00 | 170.6 | 16.3 [8.7, 28.9] | 0.55 [0.44, 0.65] | -0.49 [-0.65, -0.40] | 0.61 | 0.59 [0.48, 0.68] |

Elite quality: 0 = centre seed (unchanged policy), 1 = plain 3-iteration final top-E mean. plain_alt (independent plain seed) is the fair reference for savings and calibrates elite coverage. Horizons are identical in every mode (screen K4/H300, final K8/H_final); warm_it1/2 use fewer CEM iterations only. Warm modes keep 20% generic + 10% novelty slices.

## Caveats
- Surrogate replays are noise-free fits of each search's own WM screen pool; they estimate search efficiency, not SUMO benefit.
- Top-k-only service receipts carry no screened pool; the evaluations curve needs `RecordingEvaluator` logging (`<queue_root>/proposer_receipts/<state_id>.json`).
- WM-pool check uses held-out full-pool searches only; the verification check uses held-out searches with exact SUMO branch outcomes (top-4 + policy continues) only. Exact outcomes also train the proposer (listwise ranking + gold likelihood targets, weight 2 per search vs 1 for WM pools).
- The script never activates the proposer and never writes to remote hosts.

## Watch refresh B_u003
- READY: **False** (WM-pool False, verification False); expected saving "none non-inferior"; wall-clock {"screen_share_top4": 0.8886731168261561, "screen_share_top8": 0.8171348798615599, "basis": "saving_fraction ~ gpu_busy_share x screen_share x evaluation_saving (GPU-bound upper bound)", "expected_wall_saving_fraction": 0.0, "expected_wall_saving_s": 0.0, "top8_final_overhead_note": "top-8 doubles the K8 finalists (16 vs 8): screen share 0.889 -> 0.817 of WM futures"}.
