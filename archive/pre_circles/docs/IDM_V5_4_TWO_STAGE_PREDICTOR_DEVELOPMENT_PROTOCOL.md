# IDM v5.4 two-stage predictor development protocol

Status: **frozen before predictor-development wall outcomes are opened.**

Date: 2026-08-31

## Estimand and architecture

At a declared high-density IDM merge decision, choose between continuing the
frozen analytical P30 damping controller and temporarily adding the frozen p30
v5.4 coordinated compact moving wall.

Gate F is the deterministic predecision physical supervisor. It checks full
occupied-lane CAV coverage, compact and predicted trajectory span, kinematic
authority, projected duration, and activation gap/TTC/acceleration. It is not a
learned classifier. Refused starts are availability records, not negative wall
effects.

Gate B is learned only from Gate-F-admitted, compact-dose-qualified paired
cases. Its two primary heads are:

1. probability that the wall is safety-admissible under the frozen collision,
   teleport, hard-braking and low-TTC budgets;
2. conditional probability, trained on safe cases, that the wall is both
   service-noninferior and meaningfully beneficial.

Meaningful benefit is frozen to hard-brake-onset reduction, hard-brake-exposure
reduction, low-TTC reduction, or merge-discharge gain. Delay is only a
noninferiority guardrail. Queue area is never a selection criterion.

## Data and partitions

- Scenario: IDM, 2700 veh/h mainline, 400 veh/h ramp, 30% CAV.
- Decisions: 620, 660, 680 and 700; common terminal step 960.
- Development: 4401--4520 (120 independent seeds).
- Untouched validation: 4601--4660.
- Final paper test: 5001--5060.
- All windows from one seed stay in one group.

The treatment-blind development scan must finish all 480 windows and admit at
least 80 windows across 60 seeds and three decision times. Only admitted starts
open paired wall outcomes. Four independent admitted starts receive exact
damping and treatment repeats.

Model fitting is forbidden unless the completed development artifacts contain:

- at least 60 Gate-F-admitted seeds;
- at least 50 dose-qualified seeds;
- safe and unsafe cases spanning at least 20 seeds each;
- joint-beneficial and joint-negative cases spanning at least 15 seeds each;
- exact replay/identity, zero feature leakage and all v5.4 integrity gates.

## Frozen model and threshold selection

Both heads use standardized regularized logistic regression (`C=0.1`, balanced
class weights, liblinear). The primary 56-feature observation-v2 input is
frozen; the 101-field expanded allow-list is sensitivity only. Five-fold
stratified grouped cross-fitting produces one out-of-fold probability per
development case. No neural network or outcome-driven feature addition is
allowed.

The safety threshold grid is 0.50--0.95 in 0.05 increments; the conditional
benefit grid is 0.30--0.90 in 0.05 increments. A candidate threshold pair must
select at least 10 cases across eight seeds, achieve at least 90% observed
safety and at least 50% joint-benefit precision in grouped OOF development
predictions. Among passing pairs, maximize recovered true beneficial cases,
then precision, safety and seed coverage. If none pass, no predictor is frozen.

The chosen thresholds and models are fit on all authorized development data and
then frozen. Validation is opened once and cannot tune features, labels, model
regularization or thresholds. Final paper seeds remain sealed until validation
passes.

## Commands and expected runtime

```powershell
python -m experiments.idm.scan_idm_wall_v5_4_predictor_development --workers 4
python tools/analyze_idm_wall_v5_4_predictor_development.py `
  --scan-root runs/probe/idm_wall_v5_4_predictor_development_feasibility_p30_s4401_4520 `
  --output runs/probe/idm_wall_v5_4_predictor_development_feasibility_p30_s4401_4520/analysis.json
python -m experiments.idm.run_idm_wall_v5_4_predictor_development --workers 4
python tools/train_idm_v5_4_two_stage_gate.py
```

The scan is expected to take 1--2 hours and paired development approximately
5--10 hours on four workers. Both simulation stages are resumable.
