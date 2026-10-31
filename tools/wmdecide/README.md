# WM-PHYS probabilistic decisions and credibility

CPU-only, NumPy-only tooling. No simulator, model inference, fitting, driver-label
access or training. Team delay is full-population road delay + source waiting in
veh-s on the evaluator's registered horizon and clock convention.

## Picker API (also for S-C1SWITCH2)

```python
from tools.wmdecide import Future, decide

# Producer supplies identical hidden-driver and future-arrival draws for all actions.
actions = {
    "incumbent": [Future("draw0", 100., False), Future("draw1", 200., True)],
    "candidate": [Future("draw0", 90., False), Future("draw1", 180., True)],
}
choice = decide(actions, incumbent="incumbent")
```

`summarize(actions)` reports mean/SD/quantiles/range/Monte Carlo SE of delay,
breakdown probability, and ordered paired action comparisons. It aligns by
`draw_id`, rejects duplicate/missing/nonfinite futures, and never drops failures.
Unknown breakdown labels yield an unknown probability, not a reduced-K estimate.
Actual common-random-number coupling is the producer's responsibility: identical
ID strings alone cannot prove identical draws. IDs must identify the full shared
hidden-driver/arrival realization and action-independent random streams.

**Registered switch rule:** switch only when strict paired
`P(candidate delay < incumbent delay) >= 0.8` **and** the candidate has lower
expected delay. Select the lowest expected-delay eligible candidate; lexical
action name breaks equal candidate means. Otherwise retain the incumbent.
Ties count as neither win; no epsilon, dwell time or behavioural threshold.
The **0.8 is the owner's registered confidence level**, not a fitted parameter.
For K8 this requires at least 7/8 strict wins. It does not mean an 80% confidence
interval on an expected-value difference, nor establish real-world calibration.
MC SE measures finite-future sampling only, not shared dynamics-model error.

Verify with `python -B -m unittest tools.wmdecide.test_decision -v`.

## Credibility integration (S-RB-EVAL → S-RB-INT morning report)

Run once per identity-frozen model: WM2, WM-PHYS-A, B and C. Input directories
are the **already produced** evaluator shards; this command launches no inference.
Use the same evaluator version as the run's registered validation contract.

```powershell
python -B -m tools.wmdecide credibility --eval-root E:/agent-work/c4-rb-eval --inputs SHARD0 SHARD1 SHARD2 SHARD3 --out E:/Temp/opencode/NEW_wm2_credibility
```

On Linux use the corresponding staged evaluator/runtime directory. Output parent
must exist, output directory must be NEW. `--allow-partial` permits **only complete
K8 worlds** and labels the result diagnostic. Default requires the full 20-world
panel. Duplicated tasks, mixed constructor identities, changed BIC/prehistory or
unbalanced delay ledgers are rejected by the evaluator's own `read_predictions`.
Python `-O` is rejected because that validator uses assertions. Inputs must be
trusted local harness files. No raw recording paths are searched or opened.

For model comparisons, add `--worlds 969432 969433 969434 969436` (or the frozen
matched set) **to both invocations**. Every requested world must have validated
complete K8 futures; missing, empty or duplicate selections fail before writing.
This selects score rows only: original input validation still checks all shards,
and climatology still uses the full permitted development receipt. The audit
distinguishes all available complete worlds from scored/excluded worlds. Without
`--allow-partial`, the full-panel input requirement still applies.

Outputs:
- `CREDIBILITY.md`: ready-to-include morning report section per model, referencing
  three reliability SVGs, interval coverage, selected switches, and qualification limits.
- `credibility.json`, `summary.csv`: CRPS, fair-ensemble CRPS, Brier scores and
  model-minus-climatology differences; interval coverage/width; unknown counts.
- `reliability_breakdown.svg`, `reliability_preference.svg`, `reliability.csv`:
  fixed probability deciles, mean prediction vs empirical frequency, bin support,
  pointwise whole-world intervals. Includes all three registered breakdown regions.
- `reliability_selected_preference.svg`: calibration of the **selected** switch
  versus its incumbent, in the chosen direction (including reverse directions).
  Selected-switch probability, observed win rate, and Brier vs same-pair
  climatology are separate from the unordered all-pair diagnostic.
- `distributions.csv`, `events.csv`, `coverage.csv`: per-world outcome receipts.
- `decisions.csv`: the registered 0.8 rule evaluated from each possible incumbent
  using predictions only; switch rate and expected/recorded paired saving are
  summarized by split/horizon/incumbent/cell. No-switch cases have zero saving
  and contribute to switch rate, but are excluded from selected-win calibration.
  Recorded savings are retrospective original-state diagnostics, not closed-loop
  switching-policy efficacy. Outcomes never enter selection.
- `audit.json`, `FILES.json`: input identities/hashes, complete/missing tasks,
  evaluator and decision source hashes, output checksums. Outputs are immutable.

### Statistical contract

- Strict pairwise event: `J(A) < J(B)`; real and predicted ties are reported and
  are non-wins. One orientation per unordered pair follows frozen EVAL arm order.
  Each delta distribution is **paired A−B**, never independent cross-products.
  Selected-switch checks additionally follow the exact selected direction; a
  tied future is a non-win in either direction (not `1 - P(reverse wins)`).
- Baselines: same-arm empirical real-delay climatology and same-pair empirical
  paired-delay / win-rate climatology. Pooled baseline conditions on arm/pair;
  cell baseline additionally matches demand×penetration. Only development worlds
  969400–969431, excluding the **entire target world**, enter either baseline.
  Reserved labels cannot enter climatology. Missing baselines remain unavailable.
- Proper scores: empirical CRPS = mean|X−y| − 0.5 mean|X−X′|; fair CRPS uses
  K(K−1) instead of K² for off-diagonal pair correction (K≥2). Brier = (p−y)²
  for both breakdown and strict preference. Lower scores and negative
  model-minus-climatology differences are better; no post-hoc probability fit.
- Coverage: central 50/80/90% **linear empirical quantile** intervals and inclusive
  min–max, for both absolute delay and paired effects. Central nominal levels do
  not imply exact finite-K exchangeability coverage. Min–max's continuous-outcome
  reference is (K−1)/(K+1)=7/9 at K8; ties may raise inclusive coverage.
- Report separately by development/reserved, horizon, region, action/pair and
  cell×action/pair. Pooled summaries cannot hide conditional failures. Diagrams
  are descriptive equal-width deciles, not learned bins or traffic thresholds.
- Pointwise 95% percentile CIs use 10,000 split×demand×penetration-stratified
  whole-world resamples, seed20261031, retaining every action/pair inside each
  world. Singleton/constant groups have unavailable uncertainty. These CIs
  condition on saved futures; they do **not** estimate finite-K sampling error
  or shared dynamics-model error, and are not simultaneous qualification bands.
- Sparse overnight pilot evidence does not establish calibrated same-state
  probabilities. Adoption stays with the registered **S-RB-EVAL gate**, including
  the 60 s HDV non-regression requirement; this module does not redefine that gate.

### CPU verification

```powershell
python -B -m unittest tools.wmdecide.test_decision tools.wmdecide.test_credibility -v
python -B -m ruff check tools/wmdecide
```

Standalone tests use synthetic futures and independent CDF-integral / finite-rank
checks. Integration uses the trusted local evaluator (`WMS_RBEVAL_ROOT` overrides
the sibling worktree), creates a clearly named **synthetic plumbing fixture** from
its allowed observable receipt, and exercises full validation → CSV/SVG/report.
Scratch artifacts are preserved under `E:/Temp/opencode`, never deleted.

## Portable report publication

```powershell
python -B -m tools.wmdecide.publish --reports WM2_REPORT CANDIDATE_REPORT --matched --out NEW_PUBLICATION
```

This validates every source output checksum, requires scored K8 evidence, and
copies exact Markdown/figure/reliability bytes into per-model directories so the
relative links survive inclusion in the morning report. `--matched` rejects
different scored-world sets. `POOLED.json` contains compact numeric summaries;
`REPORTS.json` binds the full source reports/audits/manifests, and `FILES.json`
binds the publication. Output must be NEW under an existing parent. No scoring,
inference, fitting or input overwriting occurs during publication.
