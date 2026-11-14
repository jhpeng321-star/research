# IDM P30 mechanism diagnosis and modified-FollowerStopper validation results

Status: completed development diagnosis on 2026-08-29. The treatment did not
pass its frozen efficacy/headroom gate. Bounded residual-RL training remains
unauthorized.

## Scope

This report closes the development gate defined by
`IDM_MODIFIED_FOLLOWERSTOPPER_P30_DEVELOPMENT_PROTOCOL.md` and
`IDM_MODIFIED_FOLLOWERSTOPPER_P30_VALIDATION_PROTOCOL.md`. It uses only the six
already inspected development seeds 4123, 4127, 4129, 4130, 4131 and 4132.
Those seeds remain excluded from later prospective evaluation.

The work did not train a policy, run the moving wall, generate predictor-scale
data, or consume prospective evaluation seeds.

## Frozen P30 diagnosis

The 15 matched P30 cases contained six helpful, five neutral and four harmful
outcomes. Phase-aligned replay showed that the sign of the early active-phase
response did not reliably predict the final post-release effect. Two of the
four ultimately harmful cases still appeared helpful at the end of the active
phase, while none of the six ultimately helpful cases met the same early
benefit test. Corridor and upstream-follower acceleration-energy changes at
steps three and five had overlapping signs across helpful, neutral and harmful
cases.

The diagnosis therefore rejected an efficacy-conditioned online hold rule.
Only causal hard-safety aborts are justified by the available online signals.
The same development evidence also rejected fixed P30 extensions and the
projected-release-only treatment: changing duration or release did not
consistently remove delayed harm.

This localized the next mechanism question to the active command itself. The
separately versioned treatment
`activation_modified_followerstopper_p30_ceiling_v1` preserved P30 admission,
target, reference, 20-command dose, safety projection and controlled release,
while adding a modified-FollowerStopper-shaped speed ceiling.

## Viewed design smoke

The known harmful case `m2000.seed4129.t354` was used only as a viewed design
smoke and was excluded from validation. The treatment changed six of 20
projected commands and changed its native-IDM safety-composite effect from
-0.13818 under P30 to +0.08021. It also removed two stable-cohort
severe-braking exposure steps.

The original strict design gate nevertheless failed. Stable-cohort minimum TTC
decreased from 17.89 s to 13.80 s, immediate-follower minimum TTC decreased
from 20.48 s to 15.75 s, and immediate-follower minimum acceleration changed
from -0.463 to -0.666 m/s2. These remained outside the frozen hard-risk region,
so the separate bounded-safety diagnostic passed. The result was preserved as
a design observation and was not relabeled as a validation success.

## Treatment-naive validation

The frozen validation comprised the other 14 P30 cases: five first onsets,
five second onsets and four third onsets. Every treatment branch:

- matched its P30 source, predecision state, declared target and protocol;
- completed 60 evaluation steps and all 20 requested treatment commands;
- passed the runner's collision, teleport, low-TTC, delay, arrival, discharge,
  dose and unexpected-command constraints;
- had zero treatment safety aborts and zero infeasible projection intervals;
- kept the stable upstream cohort and immediate follower above the 4.0 s TTC
  and -4.5 m/s2 hard-braking floors.

The efficacy result was negative:

- P30: six helpful, five neutral and three harmful cases;
- modified FollowerStopper-P30: six helpful, five neutral and three harmful
  cases;
- mean safety-composite effect: 0.0361709 for P30 and 0.0361030 for the new
  treatment;
- tenth-percentile effect: -0.1073607 for P30 and -0.1073627 for the new
  treatment;
- stable-cohort severe-braking exposure: one step for each treatment;
- immediate-follower severe-braking exposure: zero steps for each treatment;
- only one of 14 cases (7.14%) had at least two projected-command differences
  of 0.05 m/s2 or greater, below the frozen 30% requirement.

The treatment failed three predeclared gates:

1. harmful-case count was not lower than P30;
2. the tenth-percentile effect was slightly lower than P30; and
3. meaningful command differentiation occurred in fewer than 30% of cases.

The complete machine-readable result is
`runs/probe/idm_modified_followerstopper_p30_m2000_validation_analysis_v1/summary.json`.

An independent post-run integrity audit found three future-run guardrail gaps,
none of which occurred in these artifacts. The implementation now blocks a
same-lane replacement while an unsafe feedback member releases, rejects any
feedback run whose option duration differs from the frozen 20 steps, and
prevents the generic development runner from authorizing RL for this treatment.
Feedback release telemetry is also retained at branch level. The analyzer now
enforces the exact 14-case set, candidate abstain-replay identity, comparison of
every treatment command and presence of local-safety diagnostics. Reanalysis
under those stricter checks produced the same three failed efficacy gates.

Focused verification passed 53 tests. The complete repository suite passed
661 tests.

## Mechanistic interpretation

The new nominal command was the minimum of the analytical P30 acceleration and
the modified-FollowerStopper ceiling. It could therefore make P30 more
restrictive, but it could never relax a P30 command. In 13 of the 14
treatment-naive cases, P30 or the unchanged safety projection was already the
binding constraint. The added feedback law consequently had no meaningful
projected-command authority. The encouraging design case was an outlier, not a
replicated treatment mechanism.

This is a treatment failure rather than a simulation, dose, pairing or safety
failure. Adding seeds or tuning the FollowerStopper headways on these viewed
cases would not repair the missing action authority and would invalidate the
frozen development decision.

## Decision

The modified-FollowerStopper P30 ceiling is rejected as the action foundation
for residual-RL training. `rl_training_authorized` remains false. The frozen
P30 controller remains a development comparator, not a prospectively accepted
trained damping controller.

Before any new training, a materially different action must be discussed and
frozen. It must be able to both relax and strengthen P30 within the existing
hard safety projection, or use a genuinely rear-aware/bilateral local feedback
law. It then needs a new action-credit screen on declared development seeds,
followed by observation/reward alignment. High-density collapse prevention
continues to belong to the later moving-wall stage rather than being imposed as
a decentralized-damping acceptance requirement.
