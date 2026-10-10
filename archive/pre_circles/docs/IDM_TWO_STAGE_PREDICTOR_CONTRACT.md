# IDM two-stage counterfactual-escalation predictor contract

Status: frozen downstream design; no model fitting is authorized until the
prospective treatment gate passes.

## Question answered by the predictor

At a declared decision instant, given only the damping-control history and the
current IDM traffic/CAV geometry, should the controller temporarily escalate
from decentralized analytical damping to the frozen coordinated moving wall?

This is not a generic collapse predictor. It is a conditional treatment-choice
model: it estimates whether this particular wall can be delivered safely and
whether its paired effect is preferable to continuing identical damping.

## Decision pipeline

1. **Gate F is deterministic, not learned.** The predecision supervisor applies
   the frozen full occupied-lane, compact-geometry, predicted-span, kinematic
   authority, projected-dose and activation-safety checks. Vetoed opportunities
   remain refusal records and can never be used as treatment-effect negatives.
2. Gate-F-admitted attempts must meet the compact dose contract: at least 80%
   duration and lane coverage, completed assembly, at least 50% HOLDING time,
   and compact geometry within 50 m for at least 80% of samples. Dose failures
   diagnose the treatment; they are excluded from Gate-B effect fitting.
3. **Gate B is learned conditionally on Gate F and dose qualification.** One
   regularized logistic head estimates safety admissibility. A second head,
   trained on safe cases, estimates joint service-noninferior meaningful
   safety/discharge benefit relative to identical damping.
4. Hard-brake-onset, hard-brake-exposure, low-TTC, and merge-discharge
   improvements can define benefit. Delay is a service-noninferiority guardrail
   only; queue area is diagnostic and never selects the wall.
5. The deployment policy commands v5.4 only after Gate F and frozen lower
   probability thresholds for both Gate-B heads. Thresholds are selected once
   from grouped development OOF predictions and frozen before validation.

The probability threshold, uncertainty margin, model family, and benefit rule
must be selected using predictor-development seeds only and then frozen before
predictor-validation or final-test outcomes are opened.

## Causal records and features

Each case is one `(scenario, seed, decision_step)` paired replay. Every decision
window from the same physical seed stays in the same split. The feature vector
is generated from the control trace through the decision snapshot using
`counterfactual.pilot.features_from_trace`; the treatment branch may never
supply an input.

The persisted expanded allow-list is `IDM_DECISION_FEATURES` in
`counterfactual/idm_predictor_contract.py`; it contains 101 auditable fields.
The primary model is frozen to the 56 physics-informed fields in
`IDM_PRIMARY_FEATURES`. The expanded set is a sensitivity analysis and cannot
replace the primary model based on validation or test performance. The inputs
include health history,
aggregate traffic evolution, local shock/queue structure, current CAV
placement/lane coverage, neighboring gaps and speeds, decision time, and CAV
penetration. It also includes deterministic pre-override formation diagnostics
for the actually selected wall members: member/lane counts, initial and
predicted spans, front/rear gaps and TTCs, and projected minimum acceleration.
These are computed before any wall override and are therefore deployable; the
raw timing-scan admission label itself is not a learned input. Local traffic
bins extend through 800 m, while selected-member diagnostics represent the
full frozen 200--1000 m recruitment calculation. Treatment dose, engagement,
aborts, wall commands, realized duration, and paired outcomes are targets or
diagnostics and are forbidden as features.

The v5.4 multi-window runner persists the control-prefix `features` mapping,
feature/profile hashes, predecision scan, exact causal identity, dose,
mechanics, outcomes and the versioned safety-first Gate-B label in every row.

## Data partitions

- Treatment development and support, permanently excluded: 4101--4352.
- Predictor development: 4401--4520.
- Predictor validation: 4601--4660.
- Final untouched test: 5001--5060.

No seed can cross partitions. Multiple windows from one seed are correlated
samples, not additional independent groups.

## Authorization gates before fitting

Predictor-scale generation is forbidden unless the v5.4 p30 prospective support
pilot passes every causal-identity, treatment-dose and safety-first class-support
gate in `IDM_WALL_V5_4_EMERGENCY_HANDOFF_SUPPORT_PROTOCOL.md` and its p30
amendment.

After development data are generated, fitting additionally requires:

- at least 60 independent development seeds with a Gate-F-admitted opportunity;
- at least 50 independent seeds with dose-qualified treatments;
- safe and unsafe dose-qualified outcomes each spanning at least 20 seeds;
- Gate-B joint-benefit and nonbenefit outcomes each spanning at least 15
  independent seeds;
- zero feature-schema violations and zero split overlap;
- paired effects larger than exact-repeat numerical noise.

If these minima fail, add no model complexity. Diagnose treatment support or
collect more predeclared development seeds while leaving validation/test sealed.

## Models, baselines, and evaluation

Gate F is the frozen deterministic supervisor. Both primary Gate-B heads use
standardized regularized logistic regression; model complexity is deliberately
bounded for the available grouped sample size. A shallow tree is permitted only
as a predeclared sensitivity after the primary model is frozen. Thresholds use
grouped out-of-fold development predictions; uncertainty uses grouped seed
bootstrap. No neural network is the primary model.

Required baselines are damping-only, always-command after deterministic veto,
health-threshold escalation, deterministic geometry-only escalation, and an
oracle upper bound. Development reporting includes Stage-1 average precision,
ROC AUC, Brier score, calibration slope/intercept, and grouped bootstrap
intervals; Stage 2 includes MAE/RMSE, sign accuracy, and interval coverage for
each effect. Policy reporting includes activation rate, unsafe activation rate,
beneficial precision/recall, safety/service outcomes, and paired bootstrap
confidence intervals versus every baseline.

Predictor validation is used once to confirm the frozen rule; it cannot tune the
model. The final-test partition is opened once for the paper. A publishable
claim requires the frozen gate to remain safety-admissible and improve a
predeclared safety/service endpoint over damping-only without violating service
noninferiority. Otherwise the honest paper result is treatment-mechanism and
predictability limits, not a verified online gate.
