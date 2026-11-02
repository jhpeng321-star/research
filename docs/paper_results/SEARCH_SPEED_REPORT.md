# SEARCH SPEED REPORT (plain-search speed-ups, registered v2; Paper A)

Continuation of 2026-11-02 (~16:00 UTC). Branch `c4/plan-proposer`, CPU only (no SUMO, no GPU),
TRAIN worlds 986000-986071 only, no sealed paths. Protocol: `BUDGET_SEARCH_PREREG_V2.md`
(committed 29bb908e BEFORE any held-out result). Ranked recommendations: `SEARCH_SPEED_RESULT.json`.
**Nothing here activates anything: owner-approved only (MAIN).**

## 0. Setup
- **Held-out:** the fixed 18 of 72 TRAIN worlds (`learned_screen_cv.HELD72`); developed/calibrated on the other 54 only.
- **Noise calibration (dev only, registered rule):** alpha from {0.35, 0.5, 0.7} whose simulated screen-vs-final
  Spearman among finalists is closest to the observed 0.44 → **alpha = 0.5** (0.35→0.506, 0.5→0.457, 0.7→0.353;
  `E:/agent-work/_plan_proposer_cache/ls/calib.json`). All results below use `unit='pool_sd'` (v2), horizons
  unchanged in every config (screen K4/H300, final K8/H_final; only iterations / population / screen_K change).
- **Data:** 28 held-out full-pool searches / 11 worlds; **27 evaluated** (1 skipped: surrogate pool too small).
  Registered minimum is >= 40 searches (>= 8 worlds): **NOT ENOUGH DATA** — treat every verdict below as provisional.

## 1. v2 held-out check (budget/screen configs vs independent plain search `plain_alt`)

Seed-noise margins on the same held-out searches: d8 = 0.504, d1 = 0.513 pool-sd.
Non-inferior = q8 CI-lo > -d8 AND q1 CI-lo > -d1 AND elite clusters >= 0.75 x plain_alt (8.81 → gate 6.61).
Held-out noise-model sanity: simulated screen-vs-final Spearman 0.348 / top-1 agreement 0.357 (observed dev 0.44/0.51).

| config | q8 - plain_alt [95% CI] | q1 - plain_alt [95% CI] | WM futures x | wall s/search (x plain) | elite clusters | top-8 patterns | spread x | top-1 most-common | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **it2+K2** | -0.240 [-0.419, -0.068] | -0.304 [-0.477, -0.119] | **0.409** | **282 (2.47x)** | 10.81 | 3.59 | 1.039 | 0.444 (6 distinct) | **PASS (rank 1)** |
| K2 | -0.121 [-0.285, +0.101] | -0.137 [-0.344, +0.112] | 0.557 | 386 (1.81x) | 8.74 | 3.04 | 1.002 | 0.296 (9 distinct) | PASS (rank 2) |
| it2 | -0.122 [-0.271, -0.028] | -0.095 [-0.256, +0.025] | 0.704 | 490 (1.42x) | 10.70 | 3.19 | 1.044 | 0.259 (6 distinct) | PASS (rank 3) |
| K2+pop50 | -0.439 [-0.695, -0.205] | -0.510 [-0.792, -0.276] | 0.347 | 239 (2.92x) | 8.04 | 3.33 | 0.995 | 0.370 (8 distinct) | FAIL (CI below margin) |
| pop50_it2 | -0.507 [-0.807, -0.129] | -0.516 [-0.814, -0.107] | 0.425 | 294 (2.37x) | 8.59 | 3.44 | 0.985 | 0.296 (6 distinct) | FAIL (CI below margin) |
| pop50 | -0.138 [-0.353, +0.142] | -0.122 [-0.356, +0.178] | 0.581 | 403 (1.73x) | 6.56 | 2.56 | 0.967 | 0.296 (7 distinct) | FAIL (elite-cluster gate) |
| plain | +0.260 [+0.056, +0.425] | +0.248 [+0.029, +0.415] | 1.000 | 698 (1.00x) | 9.78 | 2.96 | 1.000 | 0.259 (8 distinct) | reference |
| plain_alt | 0 (ref) | 0 (ref) | 1.000 | 698 (1.00x) | 8.81 | 2.78 | 1.010 | 0.222 (7 distinct) | reference |

Wall per search = COOPMWM.md measured wall (by group size) x the config's WM-futures ratio.

**Recommendation (ranked, full detail in `SEARCH_SPEED_RESULT.json`):**
1. **it2_K2** (2 iterations, screen K2; population unchanged): 2.45x fewer WM futures, 2.47x wall (698 → 282 s/search),
   non-inferior on both q8 and q1, elite clusters and sample spread HIGHER than plain_alt (no diversity loss);
   top-1 repetition rises (0.444 most-common share) — watch if the downstream verifier prefers distinct picks.
2. **K2** (screen K2 only): 1.80x futures / 1.81x wall, q8 CI straddles 0.
3. **it2** (2 iterations only): 1.42x, q8 CI strictly below 0 but well inside the seed-noise margin.

Population cuts (pop50 in any combination) stay unreliable on held-out too — every pop50 variant fails its
registered gate (pop50 via the elite-cluster gate; pop50_it2 and K2_pop50 via CIs below the seed-noise margin),
consistent with the v1 dev read.

Caveats: (a) NOT ENOUGH DATA (27/40 searches) — re-run before any activation; (b) q8 deltas of the passing configs
are negative in mean (small real quality loss cannot be excluded; it2_K2's whole CI is below 0 but within the
registered seed-noise margin — the registered non-inferiority criterion); (c) plain beats plain_alt by +0.26 on these
searches, i.e. seed asymmetry widens the effective safety margin; (d) surrogate replays measure WM-pool quality
(best-of-final-top-8 by the search's own truth surrogate), not exact-SUMO benefit.

## 2. State triage (v2 B: skip states unlikely to hold best-plan saving)

Registered protocol: target y = max(0, best WM-finalist saving vs 'policy continues') per search; ridge (alpha 10,
standardised) fitted on DEV only (83 searches / 22 worlds); held-out = the fixed 18 worlds (**26 searches / 11 worlds
evaluated — NOT ENOUGH DATA, < 40**). Score inputs are decision-time fields the collector already has:
request rows/group (positions, speeds, lanes, group size, regime, merge-zone occupancy) plus the K1/H300 centre
rollout costs it computes before any screening (MAPPO - policy cost, policy cost, centre open-loop gap).

| searched fraction | kept best-plan saving [95% CI] | searched wall fraction | kept per GPU-second | random | oracle |
|---|---|---|---|---|---|
| top 50% by score | **0.644 [0.146, 0.902]** | 0.519 | **1.24x** | 0.500 | 0.984 |
| top 33% by score | 0.557 [0.076, 0.815] | 0.369 | 1.51x | 0.333 | 0.897 |

- Single features kept 0.697 (50%) / 0.534 (33%) — the fitted ridge (`mappo_minus_policy` dominant, +`sd_v`,
  +`zone_occupancy`) beats the best single feature.
- Registered PASS (CI-lo of kept at 50% > 0.5): **FAIL** (0.146) — better than random in mean but not separably so.
- **Deliverable (portable JSON coefficients):** `docs/paper_results/TRIAGE_SCORE.json`
  (11 features, means/sds/coef/intercept), score via
  `budget_search_eval.triage_score(request, centre_costs, model, centre_gap=None)` — higher = search first.
- Recommendation: **use it for ordering states only** (e.g. which state to search first under contention); do NOT
  gate searches on it until it passes with enough data.

## 3. Per-episode warm start (v2 C)

(added by step 3 below)
