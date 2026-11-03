# SEARCH SPEED REPORT (plain-search speed-ups, registered v2; Paper A)

Continuation of 2026-11-02 (~16:00 UTC); **topped up 2026-11-03 (~07:00 UTC)** with the `lscollect1`
receipts that landed overnight. Branch `c4/plan-proposer`, CPU only (no SUMO, no GPU),
TRAIN worlds 986000-986071 only, no sealed paths. Protocol: `BUDGET_SEARCH_PREREG_V2.md`
(committed 29bb908e BEFORE any held-out result). Ranked recommendations: `SEARCH_SPEED_RESULT.json`.
**Nothing here activates anything: owner-approved only (MAIN).**

## 0. Setup
- **Held-out:** the fixed 18 of 72 TRAIN worlds (`learned_screen_cv.HELD72`); developed/calibrated on the other 54 only.
- **Noise calibration (dev only, registered rule):** alpha from {0.35, 0.5, 0.7} whose simulated screen-vs-final
  Spearman among finalists is closest to the observed 0.44 → **alpha = 0.5** (0.35→0.506, 0.5→0.457, 0.7→0.353;
  `E:/agent-work/_plan_proposer_cache/ls/calib.json`, frozen BEFORE any v2 held-out look). All results below use
  `unit='pool_sd'` (v2), horizons unchanged in every config (screen K4/H300, final K8/H_final; only iterations /
  population / screen_K change).
- **Data (top-up):** the overnight refresh (read-only collect, 2026-11-03) brought held-out full-pool receipts from
  28 → **62 searches over all 18 held worlds**; **56 evaluated / 17 worlds** (6 skipped: surrogate unavailable or
  degenerate pool scale). This clears the registered minimum (>= 40 searches, >= 8 worlds): **ENOUGH DATA.**

## 1. v2 held-out check (budget/screen configs vs independent plain search `plain_alt`)

Seed-noise margins on the same held-out searches: d8 = 0.417, d1 = 0.439 pool-sd.
Non-inferior = q8 CI-lo > -d8 AND q1 CI-lo > -d1 AND elite clusters >= 0.75 x plain_alt (8.07 → gate 6.05).
Held-out noise-model sanity: simulated screen-vs-final Spearman 0.400 / top-1 agreement 0.323 (observed dev 0.44/0.51).

| config | q8 - plain_alt [95% CI] | q1 - plain_alt [95% CI] | WM futures x | wall s/search (x plain) | elite clusters | top-8 patterns | spread x | top-1 most-common | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **K2** | -0.136 [-0.252, -0.035] | -0.136 [-0.274, -0.017] | 0.556 | 388 (**1.81x**) | 8.00 | 2.70 | 0.988 | 0.250 (11 distinct) | **PASS (rank 1)** |
| **it2** | -0.192 [-0.393, -0.071] | -0.162 [-0.365, -0.041] | 0.704 | 493 (**1.43x**) | 9.93 | 3.27 | 1.039 | 0.232 (9 distinct) | **PASS (rank 2)** |
| it2+K2 | -0.235 [-0.418, -0.128] | -0.247 [-0.435, -0.123] | **0.408** | 283 (2.48x) | 9.91 | 3.52 | 1.049 | 0.357 (11 distinct) | FAIL (q8 CI-lo misses margin by 0.001) |
| K2+pop50 | -0.369 [-0.643, -0.181] | -0.392 [-0.677, -0.195] | 0.346 | 239 (2.92x) | 7.62 | 3.00 | 0.992 | 0.286 (9 distinct) | FAIL (CIs below margin) |
| pop50_it2 | -0.540 [-0.836, -0.284] | -0.543 [-0.844, -0.273] | 0.424 | 294 (2.37x) | 8.77 | 3.34 | 0.999 | 0.196 (9 distinct) | FAIL (CIs below margin) |
| pop50 | -0.261 [-0.478, -0.085] | -0.241 [-0.510, -0.062] | 0.580 | 405 (1.73x) | 6.88 | 2.64 | 0.968 | 0.232 (10 distinct) | FAIL (CIs below margin) |
| plain | +0.130 [-0.030, +0.255] | +0.118 [-0.061, +0.240] | 1.000 | 702 (1.00x) | 8.43 | 2.88 | 1.000 | 0.286 (11 distinct) | reference |
| plain_alt | 0 (ref) | 0 (ref) | 1.000 | 702 (1.00x) | 8.07 | 2.62 | 0.990 | 0.250 (10 distinct) | reference |

Wall per search = COOPMWM.md measured wall (by group size) x the config's WM-futures ratio.

**Recommendation (ranked, full detail in `SEARCH_SPEED_RESULT.json`):**
1. **K2** (screen futures 2, everything else unchanged): 1.80x fewer WM futures, 1.81x wall (702 → 388 s/search).
   The ONLY config that clears both registered CI gates with room (q8 lo -0.252 vs -0.417 margin; q1 lo -0.274 vs
   -0.439) and the elite-cluster gate. Mean deltas are negative but small relative to seed noise.
2. **it2** (2 iterations, full population): 1.42x futures / 1.43x wall; CI inside the seed-noise margin with more
   headroom than at n=27; elite clusters 9.93 (above plain_alt) — no diversity loss.
3. **it2_K2 (2.48x)** is the fastest config but **fails its registered gate at n=56**: q8 CI-lo -0.418 vs margin
   -0.417 — a 0.001 miss. It would need its own pre-registered re-check (e.g. more held searches) to be considered;
   NOT usable now under the registered rule.

Population cuts stay unreliable with enough data: every pop50 variant now fails via CIs below the seed-noise margin
(pop50 no longer fails only the elite-cluster gate — its q8/q1 CIs are below margin too). Consistent with v1 dev.

What changed at n=56 vs the provisional n=27 read: it2_K2 loses non-inferiority (its q8 CI tightened around a more
negative mean, -0.240 → -0.235 with lo -0.419 → -0.418 while the margin narrowed 0.504 → 0.417); K2 and it2 keep it
with margin to spare. The ranking flips: **K2 first, it2 second**. Registered criterion satisfied: ENOUGH DATA.

Caveats: (a) mean q8/q1 deltas of ALL reduced configs are negative — the registered criterion tolerates small losses
within seed noise, but a small real quality cost cannot be excluded, and plain still beats plain_alt by +0.13 on
these searches; (b) top-1 quality drops more than top-8 (K2 q1 lo -0.274): if the downstream verifier needs the best
single pick, prefer it2; (c) surrogate replays measure WM-pool quality (best-of-final-top-8 by the search's own truth
surrogate), not exact-SUMO benefit.

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

## 3. Per-episode warm start (v2 C: seed from the same episode's previous decision elites)

Registered protocol: pairs = same world, same members, the later search 10-300 s after the earlier one
(request `time_s`); treatment = the later search's replay with the earlier search's final top-8 perturbations,
time-shifted by the gap and zero after the end, as `extra_seeds` (they displace sampled candidates, so WM
futures are unchanged); control = the same replay without them; same truth surrogate, same noise model
(alpha 0.5, same tag), quality in pool-sd units. Coded in `budget_search_eval.py`
(`episode_pairs`, `shifted_seeds`, `evaluate_warm_start`; noisy_replay gained a default-off `extra_seeds`
argument — byte-identical when None), CPU tests in `test_budget_search_cpu.py` (58 tests pass).

- **Pair availability:** 87 consecutive same-episode pairs exist among logged requests (56 episodes), but only
  **6 pairs have both sides as full-pool receipts** (2 held-out / 4 dev).
- **Registered gate (evaluate only when >= 20 such pairs exist): NOT MET → NOT EVALUATED**; the numbers below
  are exploratory only and carry no recommendation.
  - held-out 2 pairs: q8 warm - control -0.397 [-0.781, -0.014], q1 -0.238 [-0.461, -0.015]
  - dev 4 pairs: q8 +0.066 [-0.236, +0.463], q1 +0.275 [-0.010, +0.483]
- Interpretation: at 10-100 s gaps the shifted previous elites sometimes help (1 of 4 dev pairs gains +0.67
  pool-sd) and sometimes hurt (one held-out pair loses -0.78); n is far too small to separate.
- To make this evaluable, the collector needs to keep full-pool receipts for BOTH searches of a pair
  (~40+ such pairs required; currently most pairs have a top-k-only or missing response on one side).

