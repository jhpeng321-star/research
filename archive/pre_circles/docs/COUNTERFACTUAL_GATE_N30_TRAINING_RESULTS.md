# Preliminary counterfactual gate training result (N=30 cases, 10 seeds)

Generated 2026-08-25 from the tau-free paired dataset.

## Evaluation contract

- Target: readiness-selected action (`off=0`, `tau-free soft=1`).
- Labels: 21 off and 9 soft.
- Outer evaluation: leave one entire SUMO seed out.
- Inner regularization selection: leave one seed out within the outer-training
  seeds, over L2 values 0.1, 1, and 10.
- Activation threshold: fixed at 0.5 before evaluation.
- No future outcome or seed identifier is used as a model feature.
- All three decision windows from one seed remain in the same fold.

Three L2-logistic models were evaluated:

1. Health only: instantaneous health, one-step change, and 30-step change.
2. Traffic pattern: health plus current queue/speed and 60-step queue/speed change.
3. Placement pattern: traffic pattern plus local CAV fraction, soft-band CAV
   count, and soft-band lane coverage.

## Monitored result

The original health signal and both health-change features are exactly zero in
all 30 hyper-density cases. Consequently, the health-only baseline always
selects off and has balanced accuracy 0.500, soft recall 0, and MCC 0.

At the predeclared threshold, neither richer model generalizes to held-out
seeds:

- Traffic pattern: balanced accuracy 0.476, soft precision 0, soft recall 0,
  MCC -0.122, ROC AUC 0.328.
- Placement pattern: balanced accuracy 0.476, soft precision 0, soft recall 0,
  MCC -0.122, ROC AUC 0.243.

Each richer model activates once, on an off-labeled case. The resulting policy
does not improve on always-off: it leaves all eight sustained-congestion cases,
raises mean queue area from 11,895.0 to 11,930.8 veh-s, raises mean delay from
17,061.0 to 17,112.8 s, and lowers mean discharge from 319.6 to 319.43.

A diagnostic threshold sweep from 0.20 to 0.50 does not reveal a deployable
operating point. Lower thresholds gain recall only by activating in 20--29 of
30 cases, produce up to five safety-inadmissible activations, remain at or
below chance balanced accuracy, and worsen delay/discharge relative to off.
The threshold sweep is diagnostic only and was not used to select or report a
post-hoc model.

## Interpretation

The simulation experiment establishes treatment-effect headroom, but the
current 10 independent seed groups do not identify a state-to-treatment-effect
mapping that generalizes out of seed. The 30 windows are correlated within
seed, and the nine positive labels are concentrated: seed 102 supplies three
of them and four seeds contain no positive label.

This is not evidence that counterfactual gating is impossible. It shows that:

- The original scalar health gate is unusable in this hyper-density regime
  because the signal is saturated at zero.
- Aggregate traffic and placement descriptors are not yet sufficient with ten
  independent realizations.
- Reporting a random-row split would be misleading because seed leakage would
  make the dataset look larger than it is.

## Next experiment

Do not deploy or claim the current predictor works. Preserve this result and:

1. Generate at least 20 additional independent development seeds, keeping the
   three fixed decision times.
2. Reserve a later block of 10 unseen seeds as a locked test set.
3. Replace the single discontinuous deployment label with two supervised
   targets: continuous traffic effect and safety-admissibility risk.
4. Add causal-at-decision local shock features (hard-brake onset count, TTC
   exposure, speed variance, and CAV/HDV spacing around the wall band).
5. Continue using seed-grouped model selection and policy-outcome evaluation.

## Artifacts

- Training report: `runs/probe/counterfactual_gate_tau0_n30.report.json`
- Out-of-fold predictions: `runs/probe/counterfactual_gate_tau0_n30.predictions.csv`
- Training CLI: `experiments/counterfactual/train_counterfactual_gate.py`
- Model implementation: `counterfactual/gate_model.py`
