# S-BDRANK evidence index

Primary report: [BDRANK.md](../BDRANK.md). Definitions and reproduction:
[tools/bdrank/README.md](../../../tools/bdrank/README.md).

## Q1: recorded breakdown cost

- `real_episodes.csv`: 4,500 episode×region×horizon rows. J120/J300/J600 all
  include source waiting. `status` separates event, no-new-event, pre-existing.
  `depth_mps` is the negative fitted downward step **only if selected as an event**.
  `duration_s` and `queue_extent_m` are NA: absent from the registered definition.
- `cost_by_status.csv`: J/road/source/onset/depth means, counts, world CIs, at
  pooled, cell, controller and cell×controller levels. Event/no-event group
  comparisons are descriptive, not a treatment effect. Status-conditional
  bootstrap draws retain all worlds in the parent group, including zero-count
  worlds; a singleton contributing stratum suppresses the CI.
- `cost_by_exact_onset.csv`: per-cell delay and depth by exact detected onset;
  no hand-selected early/late cutpoints.
- `cost_associations.csv`: event-only onset/depth correlations and linear slopes
  versus matched-horizon team delay, plus raw event/no-event contrasts.
- `cost_standardized_contrasts.csv`: event-minus-no-event team delay standardized
  by observed episode counts in overlapping cell×controller groups. `supported_*`
  columns disclose the often very small common-support population.
- `cost_controller_pairs.csv`: ALL within-world controller pairs, factual J
  difference and occurrence/timing/depth feature differences. Pre-existing pairs
  are retained with an exclusion flag; they are not silently treated as no events.
- `cost_variance_decomposition.csv`: through-origin paired regression, nested
  R², Shapley occurrence/timing/severity fractions of total squared within-world
  controller differences, residual, coefficients/rank and leave-world-out R².
  No causal mediation or regression confidence interval is claimed. Cell models
  are deliberately descriptive and can be rank-deficient or have negative CV R².
- `occurrence_controller_shifts.csv`: paired changes versus native and CTRL0;
  **probability_shift_pp and its ci_low/ci_high are percentage points**; delta_J
  is chosen-minus-reference delay (negative helps). This sign differs from saving.

## Q2: ranking fidelity

- `controller_cache_manifest.json`:160 unique identity-bound FULL-v4 tasks,
  canonical SHA256s, source22cc32f6, recorded GPU-worker seconds.
- `controller_future_draws.csv`:7,200 factual-contract imaginary draw reductions
  (20×5×8×3regions×3horizons), coupled future indices preserved.
- `controller_predictions.csv`:900 controller/region/horizon rows. Prefixes
  `pred_` and `real_` identify the ensemble mean and recorded outcome. Event-only
  onset/depth averages have eight-draw event-support counts; pre-existing
  probability is separate. `mean_delay` is absolute source-inclusive team delay.
- `controller_pairs.csv`: every within-world/controller pair, predicted/real
  values, matching-signal and real-delay targets, concordance/tie categories.
  Later onset is preferable, lower probability/depth/delay preferable.
- `controller_concordance.csv`: pooled, cell, controller-pair and per-world
  summaries. `concordance` excludes pairs tied on either side.
  `tie_adjusted_concordance` gives half credit for predicted ties on real-distinct
  pairs, and excludes real ties. Separate outcomes are:
  concordant, discordant, predicted-only tie, real-only tie, both tie; missing
  comparisons have their own count. Empty informative bootstrap samples produce
  NA, not filtered-out draws. Singleton cells, including pooled claims containing
  one, have NA intervals. CIs are pointwise, not simultaneous qualification bands.
- `option_cache_audit.json`: read-only gpuA audit of all96 completed futures and
  24 labels for EACH version. Exact task/identity hashes, scalar saved-field keys,
  `breakdown=None` for every outcome, task wall times and cached absolute expected
  delays. This script imports only remote Python stdlib and writes nothing remote.
- `option_predictions.csv`: available O0–O4, H120/H300, v4/v5. `pred_mean_delay`
  and `real_mean_delay` are J_option−J_O0, not absolute J; a common O0 cost cancels
  in every pair ordering, selection and regret. `pred_absolute_mean_delay` is
  also supplied from the completed cached label. Exact absolute option J is not
  supplied by these diagnostic records. All breakdown fields are NA.
- `option_pairs.csv`, `option_concordance.csv`: same pairwise calculations;
  absent detector fields remain explicitly missing, never inferred from delay.
  `cell_*` IDs preserve the fixed one-state-per-cell selection ordering, not a
  demand/share inference from outcomes. One state/world per cell means CI=NA.

## Q3: decisions, selection and missing items

- `development_selection.json`: frozen development-only horizon-wise rule
  selection; written before constructing reserved decision results. No
  reserved/evaluation or exact option labels select the controller rule.
- `controller_decisions.csv`: five frozen rules on all20 worlds; actions, full
  prediction-tie sets, MAPPO/selected/oracle J, saving, regret and tie sensitivity.
  Hypothetical five-controller menu includes native. It is not a live policy.
- `controller_decision_summary.csv`: pooled and cell saving/regret/oracle/tie
  ranges, full-world counts, tie counts, choice counts and appropriately scoped
  world CIs. Positive saving=MAPPO−chosen; regret=chosen−same-menu factual minimum.
- `controller_rule_contrasts.csv`, `controller_rule_contrast_summary.csv`:
  paired extra saving relative to current mean-delay selection and changed-choice
  counts. No outcome-based pair exclusions.
- `option_decisions.csv`, `option_decision_summary.csv`: delay-only selection
  against available-option exact labels, saving/regret/oracle/strict-gate
  sensitivity and exact-choice agreement. All12 states retained for both versions.
  This does not authorize fitting or live-policy selection on diagnostic targets.
- `blocked_items.csv`: exact missing fields, stopped items, required additional
  inference and occupied GPU-minute estimates. Existing96-task totals are71.4883
  minutes v4/91.7741 v5 (163.2624 total); instrumentation overhead and exact
  trajectory recovery are not included. A conditional new controller spatial cache
  would require160five-arm600s futures, approximately688.1085 occupied GPU-min
  from observed task totals, if no separate archive exists and after a queue
  definition is registered. Existing regional bins cannot identify queue extent.
  No such inference was run.
- `round2_availability.json`: metadata-only fixed16:16:43 snapshot,7/216
  complete registered round2 world jobs; incomplete controller-v5 panel unscored.
  Existing producer/collector jobs are never touched or duplicated.
- `summary.json`, `input_manifest.json`, `verification.json`: counts, source
  hashes, development rule selection, independent checks and output hashes.
- `REGISTRATION.md`: one-line copy of the append-only shared analysis registration.

All source worktrees are read-only. This output contains no hidden truth files,
new rollouts, fitted model, live result, or new behavioral thresholds. Historical
panel exposure and sparse-cell limitations are retained in the primary report.
