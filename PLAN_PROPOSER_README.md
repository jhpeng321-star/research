# tools/coopmwm/plan_proposer.py: learned prior for the MPPI plan search (arm B only)

The plan proposer is an AlphaZero-style prior. It sees the observable situation of a CAV group and proposes joint plan perturbations, so searches can start near good plans. It never scores plans. Its proposals are extra iteration-0 seeds, and the WM still scores every candidate. It runs on CPU only (numpy and torch), uses no SUMO or GPU, and is deterministic.

## Design
- **Inputs:** one token per member from the rows the plan service already receives (`vid, s_m, v_mps, lane, is_cav[, length_m]`): relative/road position, merge-zone flag, speed and relative speed, lane one-hot, nearest vehicle ahead/behind in each of 3 lanes within 100 m, HDV/CAV counts within 100 m, and group context (regime, size, mean speed, zone occupancy).
- **Excluded:** world/seed/demand/share labels, clock, driver parameters and truth. `world_id` only groups the split.
- **Model:** set transformer (d=64, 2 layers, no positional encoding), permutation-equivariant for 2-6 members.
  - Output: M=4 Gaussian joint proposals in `mppi_search`'s canonical space (members x 10 knots over [0, 300] s, relative to the policy trajectory), plus a per-component duration categorical.
  - Defensive mixture: 0.1 of the mass stays on the search's own initial N(0, Sigma_0), uniform over durations.
- **Training targets:** weighted likelihood of candidates whose WM-predicted saving vs 'policy continues' is > 0. Each search has unit weight, shared in proportion to saving. A SUMO-positive executed plan adds +1. SUMO-losing executed plans are never targets.
- **Training setup:** world-grouped split (25% of worlds by SHA rank); early stopping on an inner world split of the TRAIN worlds only; AdamW, at most 1500 steps, seed 20261102, 1 thread, deterministic algorithms.
- **Checkpoint:** `<out>.pt` + `<out>.json` (cfg, .pt SHA256, own SHA256); the loader verifies both.
- **Offline check** (`offline_check`) on held-out-world searches, against the search's own initial distribution: elite log-likelihood gain, and nearest-elite RMS distance change (16 samples each) vs initial samples and vs the zero 'centre' seed, with world-bootstrap 95% CIs. PASS needs >= 8 worlds, >= 40 searches, LL CI lower > 0 and both distance CI uppers < 0.

## Data needed (schema `coopmwm.plan_proposer.receipt/1`, documented in the module docstring)
- **Already enough:** plan-service `requests/` + `responses/` (or `forecasts/`). `receipt_from_service` uses the top-k plans and the `search_learning` covariance. It skips arm R, rejected searches and missing rows. `attach_sumo` adds the `search_learning.update` entries.
- **Better:** wrap `evaluate` in `RecordingEvaluator`. The search output stays byte-identical (tested) and the whole screened pool is logged (372 candidates, 0.26 MB JSON for 6 members).
- **How many searches?** From the synthetic planted-direction curve (`docs/paper_results/PLAN_PROPOSER_SYNTH_CURVE.json`, 12 held-out worlds):
  - Top-k-only receipts pass from 96 searches. Log-likelihood gain is +2.0 nats per elite at 192 searches and +4.3 at 384.
  - Full-pool receipts pass from 48 searches (+3.8 at 192).
  - With fewer searches, the CI includes 0.
- **Registered minimum** before the first check: >= 200 B searches with elites from >= 32 TRAIN worlds, so that >= 8 worlds are held out. Real WM noise is unknown, so the gate decides.
- **Cost:** inference takes about 5 ms per group on CPU; training takes minutes on CPU.

## Integration recipe (only after activation; in `mppi_runtime.plan`)
1. At service start, run `model, meta = plan_proposer.load_checkpoint(path)` and check `meta['pt_sha256']` against the registered SHA.
2. For each search, run `seeds = plan_proposer.proposal_seeds(record, group, policy, 4, model, bounds, regime=regime(record['rows']))`.
   - Seeds are made with `build_plan` and `validate_plan`: clipped to bounds, <= 10 knots, registered durations, origin `'proposer'`.
   - They are deterministic per record.
3. Call `search.search(..., extra_seeds=seeds)`.
   - This patch adds an optional `extra_seeds` argument and the `'proposer'` origin to `mppi_search`. With `None` the output is unchanged and gains no key.
   - Seeds take population slots, so the WM cost per search does not change.
4. Optional logging, usable now: `rec = plan_proposer.RecordingEvaluator(score)`, then `search(..., evaluate=rec)`, then write `receipt_from_search(record, result, rec)`.
5. At each boundary, run `python -m tools.coopmwm.plan_proposer train <queue roots|receipt dirs> --learning-entries <search_learning JSONs> --out <ckpt>`. It prints the SHAs and the offline check.

## Activation rule (full text: `docs/paper_results/PLAN_PROPOSER_PREREG.md`)
- The proposer joins B's seed population only after it passes the offline check on held-out searches.
- Checks run only at TRAIN-check boundaries (every 4 updates), at most once each, up to u40, with a frozen config. Every attempt is recorded.
- Arm R never uses the proposer.
- Activation is an engineering amendment `ACTIVATE_PLAN_PROPOSER_u<k>.json` at an update boundary.
- The current frozen launch is unchanged.

## Tests
- `python -m pytest -q tools/coopmwm/test_plan_proposer_cpu.py` (rl_driving, CPU): **14 passed** in about 35 s.
- Coverage: groups of 2-6; permutation invariance/equivariance; only row fields read; planted direction learned with a held-out PASS; untrained model and too little data FAIL; world-grouped split; deterministic training and sampling; checkpoint SHA with tamper detection; schema tolerance; SUMO weighting; service adapter and queue loader; search byte-identical with the recorder; seeds within bounds/knots/durations and WM-scored via `extra_seeds`.
- `test_mppi_search_cpu.py` still passes 15/15.
