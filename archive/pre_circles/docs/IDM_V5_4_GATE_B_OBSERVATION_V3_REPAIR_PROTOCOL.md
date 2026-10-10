# IDM v5.4 Gate-B observation-v3 repair protocol

Status: **frozen before observation-v3 model outcomes are computed.**

Date: 2026-08-31

## Trigger and scope

The original 56-feature logistic predictor failed the 90% safety / 50% benefit
development gate. The bounded 101-feature and nonlinear screen also failed that
gate. A metastability-tolerant shallow-boosting policy achieved 80% safety and
50% benefit precision in aggregate grouped OOF predictions, but its
leave-one-decision-time-out result was only 50% safe and 12.5% beneficial.
Untouched validation therefore remains locked.

This is the one permitted predecision-observation repair. It changes no SUMO
branch, treatment, paired outcome, label, group split, threshold grid or seed
partition. It uses safety-history fields already persisted from the common
control prefix in all 480 development rows. No treatment replay is required.

## Physics-selected observation contract

`idm_counterfactual_escalation_observation/3` contains 204 finite decision-time
features: the frozen 101-field observation-v2 allow-list plus fields selected by
mechanism before computing v3 model results:

- current minimum TTC and aggregate merge/inflow/queue state;
- follower/leader gap means and neighbor composition;
- distance-bin CAV fractions, minimum speed and acceleration;
- lane-resolved vehicle/CAV counts, speed dispersion, acceleration, queue and
  hard-braking state;
- 60-step hard-brake/low-TTC exposure and path range/change summaries;
- 5/10/20/30/60-step density, inflow, speed, queue and CAV-lane history.

Every field is from the control branch at or before the declared decision.
Postdecision dose, release, safety and treatment-effect values remain targets or
diagnostics and are forbidden as inputs. The frozen feature hash is
`0C717F281A707799CE727E71DEA34BA8BD881A92AC69121420548A501563568B`.

## Frozen v3 candidates

Exactly three v3 candidates use the identical five seed-grouped folds and the
same two-stage fitting rule:

1. standardized balanced logistic regression (`C=0.1`, liblinear);
2. zero-variance removal, three-knot quadratic additive splines and balanced
   logistic regression (`C=0.05`);
3. zero-variance removal and depth-two histogram gradient boosting (100
   iterations, learning rate 0.05, minimum leaf 15, L2=1).

There is no feature selection, neural network or hyperparameter search.

## Gates and stop rule

The primary 90% safety / 50% benefit gate and the separately documented
exploratory 80% safety / 45% benefit, +15-point enrichment, zero-catastrophic-
excess gate are unchanged. Both require at least 10 commands across eight
seeds. Primary passing candidates outrank exploratory-only candidates.

For a selected candidate, fixed-threshold leave-one-seed-out and
leave-one-decision-time-out results are mandatory diagnostics. Validation may be
opened only after its model, thresholds, feature hash and abstention policy are
frozen in a prospective validation protocol. If v3 produces no candidate or
again collapses across decision-time holdouts, do not add another feature/model
repair or more identical development outcomes; report the observation boundary
and retain deterministic Gate F plus conservative escalation as the honest
paper result.

