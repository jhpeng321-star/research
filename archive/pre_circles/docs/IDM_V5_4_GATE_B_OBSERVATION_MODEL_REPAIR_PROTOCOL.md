# IDM v5.4 Gate-B observation/model repair protocol

Status: **frozen before the post-primary-failure model screen is executed.**

Date: 2026-08-31

## Scientific role

The primary 56-feature regularized-logistic Gate-B screen failed its frozen
grouped out-of-fold policy gate. This protocol is one bounded development-only
repair phase. It does not change the v5.4 wall, Gate F, paired outcomes, safety
or benefit labels, development cases, probability threshold grid, validation
partition, or final paper partition.

The analysis uses only the 179 Gate-F-admitted, compact-dose-qualified cases
from seeds 4401--4520. Seeds 4601--4660 and 5001--5060 remain sealed. Results
from this phase are model-selection evidence; only a later untouched validation
can support a deployable-predictor claim.

## Frozen diagnostics

The screen first reports, without feature selection:

- finite/missing/constant feature counts and forbidden-name leakage checks;
- univariate direction-invariant ROC AUC for safety and conditional benefit;
- absolute Spearman correlations at or above 0.98;
- label rates by the four declared decision times;
- within-seed label mixing across repeated decision windows;
- dose-qualified treatment mechanism categories as diagnostics only.

Postdecision dose, release and outcome fields are never model inputs.

## Frozen candidate library

All candidates use the identical five grouped folds, with every window from a
physical seed retained in one fold. The safety head is trained on all
dose-qualified cases. The conditional-benefit head is trained only on safe
training cases and predicts every held-out case.

The six candidates are:

1. `primary_logistic_reference`: the original 56-feature standardized
   logistic model (`C=0.1`, balanced, liblinear).
2. `expanded_logistic`: the same model with the frozen 101-feature allow-list.
3. `primary_additive_spline`: 56 features, zero-variance removal, three-knot
   quadratic quantile splines and balanced logistic regression (`C=0.05`).
4. `expanded_additive_spline`: the same additive model with 101 features.
5. `primary_shallow_boosting`: 56 features and a depth-two histogram gradient
   booster (100 iterations, learning rate 0.05, minimum leaf 15, L2=1).
6. `expanded_shallow_boosting`: the same booster with 101 features.

There is no hyperparameter search, feature selection, neural network or
outcome-driven candidate addition. Candidate ordering above breaks exact metric
ties in favor of the earlier, simpler candidate.

## Unchanged policy gate

Each candidate uses the already frozen safety grid 0.50--0.95 and conditional
benefit grid 0.30--0.90. A threshold pair passes only if grouped OOF commands
include at least 10 cases across eight seeds with at least 90% observed safety
and at least 50% joint-benefit precision. Passing candidates are ranked by
recovered beneficial cases, precision, safety, seed coverage and then the
fixed candidate order.

For a selected candidate the report also freezes, without retuning:

- command coverage across decision times;
- fixed-threshold leave-one-selected-seed-out rate ranges;
- fixed-threshold leave-one-decision-time-out predictions;
- primary-label results under tighter safety and stricter-benefit sensitivity
  labels.

These are robustness diagnostics. A candidate that passes the aggregate policy
gate but is dominated by one seed or one decision time is not automatically
allowed to open validation; its evidence must first be documented in a frozen
validation protocol.

## Metastability-tolerant exploratory gate

The original 90% safety / 50% benefit gate remains the confirmatory development
gate and its failure is retained. At the user's direction, this protocol also
freezes a secondary exploratory gate before inspecting selected-case severe
safety outcomes. It may nominate a candidate for untouched validation, but it
cannot erase or relabel the primary failure.

The exploratory candidate must command at least 10 cases across eight seeds,
achieve at least 80% safety-admissibility and at least 45% joint-benefit
precision, and improve both rates by at least 15 percentage points over the
unconditional dose-qualified development population. Selected commands must
contain zero cases with excess collisions or teleports. Candidate ranking uses
the same recovered-benefit, precision, safety, seed-coverage and fixed-order
rule as the primary gate.

A candidate passing only this gate is called an **exploratory selective
enrichment policy**, not a verified safety gate. Its thresholds, model and
feature contract must be frozen before seeds 4601--4660 are opened, and an
untouched validation failure ends this predictor claim rather than triggering
threshold retuning.

## Stop rules

- If no candidate passes either gate, no model bundle is produced and untouched
  validation remains locked. Diagnose whether one predecision temporal-
  observation repair is justified; do not add more identical paired outcomes.
- If only the metastability-tolerant gate passes, a separately marked
  exploratory bundle may be produced for one untouched validation; it is not a
  primary-gate success.
- If a candidate passes, write the full-development model bundle and freeze an
  untouched validation protocol before opening seeds 4601--4660.
- Final paper seeds 5001--5060 remain sealed until prospective validation
  passes.

## Command

```powershell
python tools/screen_idm_v5_4_gate_b_models.py
```
