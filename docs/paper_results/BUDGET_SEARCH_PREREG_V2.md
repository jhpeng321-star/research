# PRE-REGISTRATION v2: fast search (budget cuts, screen K2, state triage, per-episode warm start)

Registered 2026-11-02 ~22:45 +08 (owner-approved speed-ups; MAIN). This file is committed
BEFORE any held-out result for the v2 configurations, the v2 metric and the triage
score. Disclosure: v1 held-out numbers for pop50/it2/pop50_it2 were seen (27 searches,
NOT ENOUGH DATA). v2 changes the normalisation because v1's (divide by plain's gain)
explodes on small gains (dev margins 0.8-1.4). Horizons are never changed. The work
is CPU only, TRAIN worlds only, with no sealed paths.

## Shared rules
- **Held-out set:** the fixed 18 of 72 TRAIN worlds by SHA256 rank (as in v1). Required:
  >= 8 worlds and >= 40 searches, otherwise NOT ENOUGH DATA. Development and
  calibration use only the other 54 worlds.
- **Noisy surrogate replay** (`budget_search_eval.noisy_replay`, real `mppi_search` code):
  - truth = KRR of the search's own recorded WM screen pool;
  - K4 screen sd = alpha x sqrt(1 - LOO R^2) x the recorded pool sd, scaled by
    sqrt(4/screen_K) for other screen K;
  - K8 final sd = that K4 sd / sqrt(2);
  - **alpha is calibrated on DEV ONLY**: from {0.35, 0.5, 0.7}, the value whose
    simulated screen-vs-final Spearman among finalists is closest to the observed
    0.44.
- **v2 quality unit:** sd of the surrogate truth over the search's recorded pool
  (`unit='pool_sd'`).
  - q8: best of the final top-8 (truth) minus the centre seed, in pool-sd units.
    This is the primary metric.
  - q1: the same for the final top-1.
- **Margins:** seed noise, d8 = mean |q8_plain - q8_plain_alt| and d1 likewise, on
  the same held-out searches.
- **Non-inferior** when all three hold:
  - world-cluster bootstrap 95% CI lower bound of mean(q8_cfg - q8_plain_alt) > -d8;
  - the same for q1 > -d1;
  - mean elite clusters >= 0.75 x plain_alt.

## A. Budget/screen configurations
- **Configurations:** pop50, it2, pop50_it2, K2 (screen_K=2), K2_pop50, it2_K2.
- **Recommended:** the non-inferior configuration with the fewest WM futures.
- **Speed-up:** WM futures ratio, and the wall per search from COOPMWM.md measured
  wall x ratio.
- **Also reported:** diversity (elite clusters, top-8 patterns, spread, top-1
  repetition).

## B. State triage (decide, after the cheap centre/comparator rollout, whether to search)
- **Target y:** the max over the search's WM final finalists of max(0, saving vs
  'policy continues'). When exact SUMO k=8 labels exist, the exact version is
  reported as well.
- **Score:** ridge regression (alpha 10, standardised features) fitted on DEV worlds
  only. Features available at decision time:
  - from the K1/H300 centre rollout: MAPPO cost - policy cost; policy cost;
    `centre_open_loop_gap` when present;
  - observable: group size, regime, mean speed, member speed sd, group spread
    (rel_s range), lane count, leader position, merge-zone occupancy.
- **Held-out metrics:**
  - the fraction of the total y kept when only the top 50% / 33% of states by
    score are searched;
  - kept saving per GPU-second (kept fraction / searched wall fraction, wall from
    COOPMWM.md by group size);
  - comparisons against random selection (= the fraction itself) and the oracle;
  - world-bootstrap CIs.
- **PASS (worth deploying):** CI lower bound of the kept fraction at 50% > 0.5
  (better than random).
- **Deliverable:** the score as portable JSON coefficients, plus
  `budget_search_eval.triage_score(request, centre_costs)`.

## C. Per-episode warm start (only if consecutive same-episode searches exist)
- **Pairs:** same world, same members, the later search 10-300 s after the earlier
  one.
- **Treatment:** the later search's replay gets the earlier search's top-8
  perturbations, time-shifted by the gap and zero after the end, as `extra_seeds`.
  The control is the same replay without them; both use the same noise model.
- **Same non-inferiority/benefit report as A.** Evaluated only when >= 20 such pairs
  exist.
