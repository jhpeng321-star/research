# IDM Decentralized Lower-Loading Observability Validation Protocol

Status: frozen before opening paired treatment outcomes at 1900, 2000 and
2100 veh/h. All evidence produced by this protocol is development-only.

## Question

The fixed p30 activation is dose-feasible and safety/service admissible at the
2200 veh/h transition cell, but the current 134 physical observation fields do
not distinguish helpful from harmful qualified activations (seed-grouped LOSO
AUROC 0.492, balanced accuracy 0.579 on 16 nonneutral cases).

A post-hoc 2200 diagnostic identified one compact, causal hypothesis:

```text
existing 12 local-zone fields + persistent-risk recurrence index
```

This 13-field family achieved development-only LOSO AUROC 0.841 and balanced
accuracy 0.675. The result is not validation because it selected the family on
the same 2200 outcomes. The lower-loading demand/seed cells below are the
prospective-within-development check. No additional feature family may be
selected after their outcomes are opened.

## Frozen treatment and branches

- Demands: 1900, 2000 and 2100 veh/h mainline; ramp demand remains 400 veh/h.
- Seeds: 4107--4120, all permanently development-only.
- Landmarks: first, second and third persistent `risk_onset`, separated by at
  least 60 one-second controller steps.
- Candidate: one admissible CAV ranked by upstream runway.
- Control: abstain and retain native IDM.
- Replay: a second exact abstain branch.
- Treatment: fixed `activation_speed_ceiling_v1` p30 ceiling for 20 steps.
- Adequate dose: at least 16 command steps.
- Outcome horizon: 60 steps.
- Outcome, safety and service classification: unchanged from
  `IDM_DECENTRALIZED_BOUNDED_HARM_CONTRACT.md`.

The discovery-only scan found 105 opportunities before treatment outcomes were
opened: 35 at 1900, 33 at 2000 and 37 at 2100 veh/h. Missing recurrences remain
missing; no replacement seed or landmark is selected.

## Frozen observation hypotheses

1. **Current actor:** all 134 physical fields, excluding action-mask bits.
2. **Proposed compact upgrade:** the already present 12 local-zone fields plus
   a normalized persistent-risk recurrence index. The recurrence count is
   causal episode history available at the decision time. It is not configured
   demand, seed, a future outcome, realized dose or branch identity.

The proposed upgrade is admitted only if it generalizes across held-out seeds
and is not solely a pooled demand separator. No admission/controller geometry
features will be added from this experiment.

## Pass/fail criteria

General evidence gates:

- exact discovery, branch and abstain-replay identity;
- zero collisions and teleports;
- at least 80% adequate treatment dose;
- at least 90% safety/service admissibility excluding dose;
- at least 24 qualified nonneutral decisions;
- helpful and harmful outcomes in at least three seeds each;
- coverage of all three predeclared demand cells.

Current-observation gate:

- seed-grouped leave-one-seed-out AUROC at least 0.70; and
- seed-grouped balanced accuracy at least 0.65.

Proposed-upgrade gate:

- pooled seed-grouped leave-one-seed-out AUROC at least 0.70;
- pooled seed-grouped balanced accuracy at least 0.65;
- at least two of the three individual demand cells have AUROC at least 0.60
  and balanced accuracy at least 0.55; and
- no individual cell with both classes has AUROC below 0.45.

Passing the proposed-upgrade gate authorizes implementing and testing the
single recurrence field. It does not authorize MAPPO training. After the field
is implemented, observation-schema, gradient, mask separation, reset and
checkpoint-compatibility tests must pass, followed by reward-alignment
verification on the unchanged paired rows.

If treatment dose or safety/service fails, repair the actuator rather than the
observation. If dose and admissibility pass but neither observation hypothesis
passes, do not train; redesign the causal state or accept a conservative
deterministic abstention rule. Results at 2700 veh/h are excluded because
structural overload is reserved for moving-wall escalation.

