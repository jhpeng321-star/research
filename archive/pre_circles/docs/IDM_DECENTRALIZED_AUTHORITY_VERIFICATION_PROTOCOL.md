# Decentralized IDM Candidate-and-Mode Authority Verification

Status: frozen development-only pretraining gate, version
`idm_decentralized_authority/1`. No PPO run is authorized unless every
mandatory gate below passes without post-outcome threshold changes.

## Paired treatment

At the first development-seed state containing at least two independently
admissible CAVs in one disturbance lane, replay the exact predecision state into
ten branches:

- all-abstain control and exact all-abstain replay;
- admissible candidate slot 1 with mild, nominal, strong, and conservative
  reference modes;
- admissible candidate slot 2 with the same four modes.

Only the target CAV bids at the decision step. The selected option remains
latched for 20 controller steps; subsequent actor requests abstain. All other
CAVs remain native IDM. Safety outcomes use 40 steps and service outcomes use
60 steps. The controller configuration, candidate ranking, admission, V2V
contention, safety shield, and release contract are identical across branches.

## Cohort

Smoke: demand 2200, seeds 4107 and 4109, one decision per seed.

Initial authority pilot: demand 2200, seeds 4107--4112, one decision per seed.
These are inspected development seeds and can never support independent claims.
Seeds 4201 and above remain unopened. Failure does not authorize adding seeds.

## Outcomes

Primary physical effect is the equal-weight mean of reference-normalized
reductions in corridor severe braking and acceleration energy over 40 steps.
Replay fixes a symmetric deadband of at least 0.02 reference units. Report TTC,
collisions, teleports, delay rate, arrivals, discharge, collapse exposure,
projection survival, command dose, and follower/influence/lane response
separately. No favorable composite may hide a failed safety or service bound.

## Mandatory integrity and dose gates

- every branch matches discovery and all branches share one decision hash;
- all-abstain versus replay is exact;
- all horizons complete with zero action leakage;
- control branches issue zero IDM commands;
- zero collisions and teleports;
- at least five of six pilot seeds expose a two-candidate decision;
- at least 80% of treatment branches realize at least 16 of 20 command steps;
- at least 90% of treatment branches pass all safety/service constraints.

An individual treatment is admissible only when it completes its horizon, has
at least 16 command steps, adds no collision/teleport, adds at most two low-TTC
steps, adds no more than 1.0 s/vehicle corridor delay rate, and loses no more
than one arrival or merge discharge.

## Mandatory learnability gates

- at least 30% of dose-qualified effects exceed the replay deadband;
- helpful admissible effects occur in at least three independent development
  seeds;
- neutral or harmful effects occur in at least three independent development
  seeds, proving a nontrivial decision rather than an always-act rule;
- at least two modes are the best admissible choice in different cases;
- candidate-slot contrasts exceed the deadband in at least 20% of matched
  mode/case comparisons, and each candidate slot wins in at least two matched
  comparisons;
- at least 30% of dose-qualified branches change influence-region acceleration
  energy above numerical noise;
- median projection survival for nonnominal commands is at least 50%.

Passing these gates authorizes observation/reward freeze and a small resumable
PPO smoke only. It does not prove independent performance, calibration,
generalization, collapse prevention, or escalation benefit.

## Commands

Smoke:

```powershell
python -m tools.run_idm_decentralized_authority_verification `
  --output runs/eval/idm_decentralized_authority_smoke_v1 `
  --seeds 4107 4109 --workers 3
```

Pilot, only after smoke integrity/dose passes:

```powershell
python -m tools.run_idm_decentralized_authority_verification `
  --output runs/eval/idm_decentralized_authority_pilot_v1 `
  --seeds 4107 4108 4109 4110 4111 4112 --workers 3
```
