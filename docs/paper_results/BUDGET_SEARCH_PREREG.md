# PRE-REGISTRATION: budget-reduced PLAIN search (fewer candidates/iterations, horizons unchanged)

Registered 2026-11-02 ~21:50 +08 (MAIN request). This file is committed BEFORE any
held-out result for these configurations. Code: `tools/coopmwm/budget_search_eval.py`.
It runs on CPU only and uses no SUMO or GPU. Only TRAIN worlds are used and no sealed
paths are read.

**Why this exists.** Learned screens and learned pre-filters failed TRAIN-only inner CV
(docs/paper_results/LEARNED_SEARCH_REPORT.md):
- Cross-search Spearman is 0.13-0.16, and the top-16 is worse than a random 16.
- A learned pre-filter performs the same as random dropping.
- **Noise ceiling** (TRAIN dev searches): among finalists, the K4/H300 screen and the
  K8/H_final final agree with Spearman 0.44, and the screen's top-1 is the final's
  top-1 only 51% of the time.

Most of the screen's fine ranking is therefore noise. The question is whether a
smaller plain budget is enough.

## Configurations (all with screen K4/H300 and final K8/H_final unchanged; top_k=4 -> 8 finalists)
- **pop50**: population x0.5 (pop_min 16, 1 x dim), 3 iterations.
- **it2**: population x1, 2 iterations.
- **pop50_it2**: population x0.5, 2 iterations.
- **References**: plain (registered) and plain_alt (the same search with an independent seed).

## Held-out set
- The FIXED 18 of 72 TRAIN worlds with the lowest SHA256 rank of (20261102, world):
  986002 986003 986004 986007 986008 986017 986024 986036 986038 986039 986040
  986044 986047 986056 986058 986063 986068 986069.
- Use all full-pool searches in those worlds; at least 8 worlds and 40 searches are
  required, otherwise NOT ENOUGH DATA.
- The 54 other worlds were used only for development: the noise-ceiling estimate and
  the code sanity checks.

## Replay
- Each held-out search is replayed through the real `mppi_search` code with a NOISY
  surrogate:
  - truth = RBF kernel-ridge regression of the search's own recorded WM screen pool;
  - screen = truth + N(0, s^2), with s = sqrt(1 - LOO R^2) x the recorded pool sd;
  - final = truth + N(0, s^2 / 2).
- The noise model is checked against the observed agreement (Spearman 0.44, top-1
  0.51) and reported.

## Quality
- Normalisation: 0 = the centre seed (unchanged policy); 1 = plain's best of the final
  top-8 (truth).
- **q8** (primary): the truth of the best of the final top-8. Exact SUMO picks among
  these.
- **q1**: the truth of the final top-1.

## Pass rule, per configuration
Margins are the seed noise of the registered plain search on the same held-out
searches: d8 = mean |q8_plain - q8_plain_alt| and d1 likewise. A configuration
passes when all three hold:
- world-cluster bootstrap (2000) 95% CI lower bound of mean(q8_cfg - q8_plain_alt) > -d8;
- the same for q1 > -d1;
- no collapse: mean elite clusters >= 0.75 x plain_alt.

The recommended configuration is the passing one with the fewest WM futures per search.

## Reported
- WM futures and wall time per search (measured COOPMWM.md wall x futures ratio).
- Diversity: elite clusters, top-8 member patterns, pool spread, top-1 repetition.
- Noise-model check.
- Refreshes on more data re-run the identical rule and record every attempt.
