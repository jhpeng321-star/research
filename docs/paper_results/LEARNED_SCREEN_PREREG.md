# PRE-REGISTRATION: learned screen (amortised WM screen) for arm B — offline check

Registered 2026-11-02 ~19:20 +08. This was committed BEFORE any held-out learned-screen
result was computed or seen. Code lives in `tools/coopmwm/learned_screen.py` (written after
this file) and the flag in `mppi_search.search(..., screen=)`. CPU only: no SUMO, no GPU,
TRAIN worlds only, no sealed paths. Activation is owner-gated. The owner has approved
switching only if this check PASSES; MAIN performs the switch.

## Object
- **Model.** A network predicts a candidate plan's WM screen score from the
  observable state features (the proposer's 41 member features) and the plan
  (canonical member x 10 perturbation, knot mask, duration).
- **Architecture.** Set transformer (d=64, 2 layers) over member tokens, each token
  being [state features | own perturbation row | knot mask | D/300]. Mean+max
  pooling feeds a scalar head.
- **Targets.** Within-search standardised WM screen saving vs 'policy continues'
  (K4/H300) for every screened candidate of every TRAIN full pool.
- **Loss.** MSE plus a listwise ListNet term within each search. Exact SUMO
  verification lists (k=4 or k=8 candidates plus the zero 'policy' item) add a
  ListNet term with weight 2 on the standardised exact savings.
- **Training.** Deterministic (seed 20261102, single thread), at most 2000 AdamW
  steps, lr 1e-3, wd 1e-4. Early stopping on an inner world split of the TRAIN
  worlds only.
- **Search use** (`screen=` flag; default None leaves the search byte-identical):
  - Candidates are sampled exactly as in the plain search: same population and
    iterations, same MPPI/CEM updates, same centre/MAPPO/library seeds.
  - The network scores them; the K4/H300 WM rollouts are skipped.
  - The top 16 (`top_k=8`, finalists = 2 x top_k) go to the unchanged WM final
    stage (K8, H_final = min(600, max(300, D+120))).
  - Top-k and Thompson are unchanged; exact SUMO verification picks among the
    top-8.
  - No horizon is shortened.

## Held-out evaluation (all numbers on held-out worlds only)
- **Split.** `plan_proposer.world_split(searches)`: 25% of worlds by SHA256 rank of
  (20261102, world). Minimum >= 8 held-out worlds and >= 40 held-out full-pool
  searches; otherwise NOT ENOUGH DATA, never PASS.
- **P (primary, non-inferiority).** Each held-out search is replayed through the
  real `mppi_search` code on its own surrogate. The surrogate is RBF kernel ridge
  regression of the search's recorded WM screen pool, standing in for the WM, which
  cannot run offline; this is disclosed.
  - Modes:
    - **plain**: registered search, surrogate as screen, 8 finalists;
    - **plain_alt**: the same with an independent seed;
    - **LS**: network as screen, 16 finalists, surrogate as the WM final stage.
  - Quality: q = (best finalist surrogate value - centre seed value) / (plain best
    finalist value - centre seed value).
  - Margin delta = mean |q_plain - q_plain_alt| over the same held-out searches.
    This is the seed-to-seed noise of the registered plain search, measured
    symmetrically and never using LS.
  - **PASS-P**: world-cluster bootstrap (2000) 95% CI lower bound of
    mean(q_LS - q_plain_alt) > -delta.
- **S1 (real WM labels, no surrogate).** On each held-out search's recorded pool:
  - Spearman rho between network scores and the actual WM screen savings;
  - q16 = (best actual saving among the network's top-16 - centre) / (best actual
    in the pool - centre).
  - **PASS-S1**: CI lower bound of mean rho > 0, AND CI lower bound of mean q16 >
    the mean q16 of a random 16 (expectation over 200 random subsets per search).
- **S2 (no collapse).** The mean number of distinct elite clusters of LS must be
  >= 0.75 x that of plain_alt. Clusters are linked at the median plain-pool NN
  spacing, as in `learned_search_diversity`.
- **READY** = enough data AND PASS-P AND PASS-S1 AND S2.
- **Reported, not gates:**
  - elite coverage vs recorded elites (LS vs plain_alt);
  - diversity/collapse metrics (spread, patterns/clusters in the top-8/elites,
    top-1 repetition);
  - learning curve of P and S1 vs number of training searches, extrapolated to
    the volume needed if not passing;
  - estimated search wall time per search: WM futures plain = (N+2)*4*3 +
    (8+2)*8; LS = (16+2)*8. This is scaled by the measured wall per query
    (COOPMWM.md: 2/4/6 CAVs 559.6/1030.0/898.3 s) plus measured network CPU time.
- **Exact SUMO labels.** When verification lists exist on held-out worlds, the
  exact-best rank of the network order vs the WM-screen order is reported.
- **No gate shopping.** The configuration is frozen at the commit that adds
  `learned_screen.py`. Refreshes re-run the identical check on more data (hourly /
  per B update / per LS-COLLECT batch), and every attempt is recorded in
  `docs/paper_results/LEARNED_SEARCH_REPORT*`. A configuration change needs a new
  registration before its held-out results are seen.
