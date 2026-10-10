# Fixed-P30 Limited-Domain Damping Protocol

**Status:** frozen limited-domain development protocol for the fixed-P30
damping program after the closed-loop bilateral pulse mechanism no-go
(`docs/IDM_REAR_AWARE_FEEDBACK_V2_FAILURE_DIAGNOSIS.md`, accepted by the
user on 2026-08-30). No RL training is authorized by this document; it
authorizes the causal observation/value screen only.
**Date:** 2026-08-30
**Workspace:** `D:\research-dense`

## 1. Supported candidate domain

- CAV penetration limited to **5% and 10%** (the empirically supported P30
  operating domain; recorded development evidence: 25.9% held coverage,
  +0.0434 conditional reward, 7 helpful / 3 harmful / 5 neutral activations
  on the one-time held development rows).
- No critical traffic density is assumed. Demands **1900, 2000, 2100 veh/h**
  are retained in the screen; the screen may determine whether demand or an
  observable traffic state further limits the domain.
- Only exact paired P30-versus-abstain decisions are used (matched
  predecision identity by construction of the paired-option grid).
- Seeds 4107–4132 are development-only; 4101–4106 and 4107–4132 are excluded
  from independent/prospective claims. Every no-decision, constraint failure,
  harmful outcome and neutral outcome is retained and reported.
- Operations outside the domain (penetration outside 5–10%, unsupported
  states) abstain and remain candidates for later moving-wall escalation.

## 2. Deployment interpretation of penetration

CAV penetration is a configuration/operating variable known to the traffic
management system at deployment. It is **not** a post-treatment feature and
it is **not** inferred from any outcome. Frozen interpretation:

> Penetration is used **outside the actor** as the operating-domain
> admission rule. The act of screening/training within domain-specific data
> makes penetration part of the deployment context, not a learned policy
> input. The actor observation remains penetration-free.

This preserves the causal treatment-independence of the observation and
avoids any risk that the model learns a penetration label from realized
outcomes. A deployment-time admission check (`penetration ∈ {0.05, 0.10}`)
gates activation; outside it the policy always abstains.

## 3. Dataset and leakage contract

Permitted causal predecision observations (all from the frozen paired grid
discoveries; no postdecision input):

- ego speed and acceleration (candidate observation block, 134-field local
  observation; classic RSU block allowed);
- leader gap and relative speed;
- immediate-follower gap, closing speed, TTC and secure-gap headroom;
- candidate runway and predicted disturbance-interception time
  (`predicted_dose_s`);
- local density, speed variance, wave energy and severe-braking state
  (`pattern_context`, spatial front/wave fields);
- merge-zone state available from the RSU (classic RSU observation block);
- causal trajectories ending at the decision (temporal context only if the
  existing causal temporal contract is reused; the failed p15 temporal screen
  does not imply the same for this domain);
- projected P30-versus-native command separation and expected dose
  (`terminal_speed_previews`, `preview_nominal_acceleration_mps2`,
  `predicted_dose_s`).

Forbidden:

- seed;
- future observations;
- realized dose;
- treatment outcome or reward;
- release reason;
- branch identity;
- postdecision safety or service measurements;
- configured demand when used across cells (demand is a deployment-known
  cell variable, not a per-decision post-hoc feature; the screen reports
  per-demand results but does not train a demand-selective model unless
  demand is declared a deployment-available admission variable in a future
  versioned protocol).

Splits: exclusively grouped by seed/group. The frozen split is
train/development 4121–4128 and one-time held development 4129–4132; a
leave-one-seed-out grouped screen is reported only as diagnostic. No
random branch-row split.

## 4. Screen targets (evaluate separately)

### A. Outcome classification (diagnostic)

Helpful versus harmful among qualified activations (frozen contracts:
helpful = composite > +0.02 with all constraints passing; harmful =
composite < −0.02 or any constraint failing; neutral otherwise). Report
grouped AUROC, average precision, balanced accuracy and calibration. Retain
the existing AUROC 0.70 and balanced-accuracy 0.65 targets as diagnostic
evidence. AUROC alone does not authorize training.

### B. Conditional expected option value (primary authorization question)

Question: would activating P30 in an observable state improve expected paired
reward relative to abstention?

Fit simple models first, in this order:

1. penetration/domain rule (the existing 5–10% rule — baseline);
2. ridge regression;
3. logistic regression;
4. shallow tree / monotonic model where appropriate.

A neural policy is evaluated only after these baselines. Outcome labels and
paired rewards come from the frozen paired-option contract
(`idm/option_reward.py`, reference
`runs/eval/idm_decentralized_lower_reward_refs_v1.json`); training labels in
a supervised sense are the paired reward / classification from the paired
grid, not a variable-lifetime return.

Model thresholds and hyperparameters are selected on grouped development
predictions (seeds 4121–4128) only; seeds 4129–4132 are opened once for the
report and are never used for further tuning.

## 5. Frozen authorization gates (RL authorization)

Authorize RL only if a causal deployable model achieves, on held development
groups:

- activation coverage at least 20%;
- positive conditional mean paired reward;
- helpful activations more frequent than harmful activations;
- harmful activation fraction no greater than 25%;
- policy value above never activate;
- policy value above always activate;
- policy value above the fixed 5–10% domain rule by a practically meaningful
  margin (predeclared: mean conditional reward at least +0.01 above the rule);
- positive value in more than one demand/operating cell;
- no added safety or service violations;
- treatment effects beyond replay noise.

AUROC alone must not authorize training. If helpful/harmful classification
remains difficult but a leakage-safe expected-value policy clearly and
reproducibly beats the simple rule, that distinction is documented and the
policy is evaluated under the frozen expected-value gate.

## 6. Reward audit

If the observation/value gate passes, reproduce the existing paired 60-step
binary option reward on the limited-domain dataset with a fresh exact audit:

- abstain reward exactly zero;
- helpful qualified P30 positive; harmful qualified P30 negative;
- constraint failures receive the hard negative override (−2);
- exact paired horizon (60 steps);
- no agent-lifetime, seed, future-state, queue or branch leakage;
- exact replay and effect reproduction (within 1e-12);
- component balance (no component above 80% mass);
- all constraint-failed rewards rank below admissible rewards;
- focused tests plus the complete repository suite pass.

## 7. Training

Only after observation/value and reward gates pass: train a paired
contextual-bandit/semi-Markov PPO policy, action 0 = abstain, action 1 =
activate frozen P30, frozen causal observation, frozen reward, counterfactual
paired branches during training only. The rejected ordinary per-step MAPPO
reward is never used.

Sequential gates (each must pass before the next):

1. wrapper determinism and leakage tests (branch-order invariance,
   worker-count invariance, exact replay identity, selected-CAV departure,
   resume/restart, normalization freeze, no postdecision leakage);
2. 2,000-sample learning smoke (finite rewards/losses/gradients, both actions
   sampled, policy probabilities change, no collapse to always activate or
   always abstain caused by implementation bias);
3. 10,000-sample value test (grouped policy value above never-activate,
   always-activate, the fixed 5–10% rule, health-threshold and feasibility-only
   gates; acceptable coverage; bounded harmful fraction; repeatable);
4. full campaign only if both pass.

Every checkpoint is compared against never activate, always activate, the
fixed 5–10% rule, the health threshold and the best simple expected-value
model. If a ridge, logistic, tree or other simple learned model performs
better than PPO, the simpler trained model is retained. The paper requires a
defensible learned controller, not unnecessary neural complexity.

## 8. Prospective acceptance

After freezing model, activation threshold, observation normalization,
reward, treatment and checkpoint, allocate untouched seeds (fresh set; none
of 4107–4132). Freeze the seed list, cells, metrics, deadband, limits and
analysis before opening them. Require per seed-grouped results:

- positive mean paired policy value;
- better policy value than never activate, always activate and the fixed
  5–10% rule;
- helpful more frequent than harmful;
- at least 70% helpful-or-neutral activations;
- bounded harmful tail (predeclared bound);
- no added collision or teleport;
- acceptable TTC, follower braking and service;
- no single seed responsible for the average gain.

The controller does not need to improve every metastable seed.

## 9. Failure path

If the limited-domain causal observation cannot outperform the fixed 5–10%
rule, do not force PPO. Freeze the rule-based P30 controller as the strongest
honest decentralized damping arm and document that learned activation was not
supported. The paper architecture remains: limited low-authority decentralized
damping, an empirically demonstrated authority boundary, and coordinated
moving-wall escalation for unsupported states.

## 10. Seed exclusions and prohibited claims

Seeds 4107–4132 are development-only; no seed in this range may be used for
prospective claims. Prohibited until prospective evidence exists: claiming
the damping controller works or generalizes; calibrated probabilities;
demonstrated collapse prevention; validated moving wall; working escalation
predictor. Prohibited actions: training the escalation predictor, generating
paper-scale data, ordinary per-step MAPPO with the rejected reward, discarding
harmful/neutral/constraint-failed cases, or changing thresholds after viewing
held outcomes without creating a new version.
