# Learned search for the MPPI plan search (arm B only; never auto-activated)

`tools/coopmwm/plan_proposer.py` is an AlphaZero-style prior. From the observable situation of a CAV group it proposes joint plan perturbations. The search can use it in two flag-only ways (default off, output byte-identical): **seeds** (extra iteration-0 seeds) and **warm start** (the starting distribution for MPPI/CEM). The WM still scores every candidate, and horizons are never shortened. Everything runs on CPU (numpy/torch), deterministically, without SUMO or GPU.

## Design
- **Inputs:** one token per member from the rows the plan service already receives (`vid, s_m, v_mps, lane, is_cav[, length_m]`): relative/road position, merge-zone flag, speeds, lane one-hot, nearest vehicle ahead/behind per lane within 100 m, HDV/CAV counts, and group context (regime, size, mean speed, zone occupancy).
- **Excluded:** world/seed/demand/share labels, clock, driver parameters and truth.
- **Model:** set transformer (d=64, 2 layers), permutation-equivariant for 2-6 members. It outputs M=4 Gaussian joint proposals in `mppi_search`'s canonical space (members x 10 knots over 300 s, relative to the policy), plus a duration categorical. 0.1 of the mixture stays on the search's own N(0, Sigma_0) as a defensive component. A mixture-entropy floor of 0.5 log M prevents collapse onto one plan type.
- **Targets:** every B search with a full screened pool, including searches that were not accepted. Each search's top-E elites (E = ceil(N/4)) are ranked by utility against its comparators, min(saving vs policy, saving vs MAPPO). The weight is the utility above the next-ranked candidate, and each search sums to one unit. A SUMO-positive executed plan adds +1; SUMO-negative executed plans are never imitated. Top-k-only receipts are a fallback.
- **Training:** world-grouped split (25% of worlds); early stopping on inner TRAIN worlds only; seed 20261102. Checkpoints are `.pt` + `.json` with SHA256, and the loader verifies both.
- **Warm start** (`warm_start`; `mppi_search.search(warm_start=...)`): the mixture is moment-matched into one Gaussian. In every iteration, **20% generic** policy-centred samples and a **10% novelty** slice are kept. The novelty slice takes, from 4x as many generic draws, those with the lowest proposer/generic density ratio (rank-based), so unexpected moves are still tried.
- **Novelty per executed plan:** `plan_novelty` / `novelty_report` give the distance to the nearest proposer mode and library entry, plus the post-hoc label including 'new'.

## Checks
- **Offline check** (`offline_check`, held-out worlds, vs the search's own initial distribution): elite log-likelihood gain CI lower > 0; nearest-elite distance vs initial samples and vs the 'centre' seed, CI uppers < 0; >= 8 worlds and >= 40 searches.
- **Evaluations saved** (`learned_search_eval`): each held-out search is replayed through the real `mppi_search` code on a kernel-ridge surrogate of its own full WM screen pool. Modes: plain, plain_alt (different seed), seeds, warm, both, and warm_it1/warm_it2 (1 or 2 CEM iterations, same K4/H300 screen and K8/H_final). It reports evaluations needed to reach plain's final elite quality (mean of the top E), the quality curve, and elite coverage, all with world-bootstrap CIs.
- **Synthetic results** (`docs/paper_results/LEARNED_SEARCH_SYNTH.json`: 40 held-out searches, 8 worlds, sizes 2-6, 30% unaccepted, 240 screened candidates per search). All comparisons are against an independent plain search (plain_alt):
  - **warm** reaches plain's final elite quality 33 evaluations sooner (-14%, CI -41 to -24) and ends +0.22 higher (CI 0.10 to 0.36).
  - **both** saves 38 evaluations (+0.18 quality). **seeds** saves 9 (CI includes 0).
  - **warm_it2** uses 67% of the evaluations and ends -0.06 (CI -0.12 to 0.01), roughly equal.
  - **warm_it1** is too weak (-0.41).
  - Elite coverage is 0.62 for warm and 0.55 for warm_it2, against 0.47 for an independent plain search.
  - Data needed: with full pools, the check passes from 96 training searches (`PLAN_PROPOSER_SYNTH_CURVE.json`).

## Real data (`python -m tools.coopmwm.learned_search_report`)
- **Collect:** copies read-only (SSH `find` / `tar -c` only, with forbidden names pruned) from cpuQ `requests/` + `responses/` and from `proposer_receipts/` on cpuQ/gpuA/gpuR into `E:/agent-work/_plan_proposer_cache`. Only TRAIN worlds are kept.
- **Run:** trains, runs the check and the evaluations curve (once at least 48 full-pool receipts exist), and writes `docs/paper_results/LEARNED_SEARCH_REPORT.md` + `.json`. It never activates anything.
- **Status 2026-11-02 04:12:** 3 B searches found on cpuQ (2 accepted, 1 rejected). Real full pools: 0, because the live service does not log pools.
- **To get full-pool data,** MAIN would approve this logging-only patch in `mppi_runtime.plan` (search output byte-identical, no WM cost): `rec = plan_proposer.RecordingEvaluator(score)`; `search(..., rec, ...)`; `plan_proposer.log_receipt(root, record, result, rec)`.
- **Registered minimum** for the first check: >= 160 full-pool B searches (>= 40 held out) from >= 32 TRAIN worlds.

## Integration after explicit owner approval (`mppi_runtime.plan`)
1. Run `model, meta = plan_proposer.load_checkpoint(path)` and check the SHA.
2. For the **seeds** mode: `seeds = proposal_seeds(record, group, policy, 4, model, bounds, regime=...)`, then `search(..., extra_seeds=seeds)`.
3. For the **warm** mode: `search(..., warm_start=plan_proposer.warm_start(record, group, model, regime))`. Mode **both** combines steps 2 and 3. Iteration cuts need a separate amendment.

## Activation (`docs/paper_results/PLAN_PROPOSER_PREREG.md`)
Activation needs both (a) a PASS of the offline check under the registered schedule and (b) explicit owner approval recorded by MAIN. It is never automatic. Arm R never uses the proposer, and activation is recorded as an engineering amendment at an update boundary.

## Tests (rl_driving, CPU)
- `python -m pytest -q tools/coopmwm/test_plan_proposer_cpu.py tools/coopmwm/test_learned_search_cpu.py tools/coopmwm/test_mppi_search_cpu.py`: 43 passed.
- Coverage: permutation invariance, determinism, SHA tamper detection, schema tolerance; unaccepted searches as training data; flag-off and prior-equal warm start byte-identical; generic/novelty slice counts and novelty ranking; entropy floor; unchanged horizons with fewer iterations; elite coverage; read-only report path on a synthetic cache.
