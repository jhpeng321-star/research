# IDM-to-P30 Projected-Blend Authority Results

**Date:** 2026-08-29  
**Status:** completed development screen; authority failed; no observation,
reward, RL, wall, or predictor work authorized

## Decision

The versioned `activation_idm_p30_projected_blend_v1` action is implemented and
mechanically identifiable, but it is rejected as the action foundation for the
IDM decentralized RL controller.

The blend solved the earlier action-collapse problem: intermediate strengths
produce genuinely different projected commands. It did not demonstrate enough
safe, seed-diverse incremental outcome headroom to justify learning which
strength to choose. The predeclared authority gate is therefore failed, and the
contingent observation/reward audit was not run.

This is a development result, not prospective controller evidence.

## Frozen treatment

The treatment uses true native-IDM abstention and four active strengths:

```text
0.00 = native IDM, no TraCI speed command
0.25 = 25% of the projected native-IDM-to-P30 interval
0.50 = 50%
0.75 = 75%
1.00 = exact frozen projected P30 command
```

Both endpoints are projected from the same pre-action state, their feasible
accelerations are convexly blended, and the result is projected again through
the unchanged safety layer. Active strength is latched for the existing
20-command option and uses the existing release path. Candidate selection and
execution remain decentralized; nonselected CAVs receive no treatment command.

The frozen protocol is `IDM_P30_PROJECTED_BLEND_AUTHORITY_PROTOCOL.md`.

## Implementation and artifacts

The implementation slice is in:

- `idm/projected_blend.py`;
- `idm/decentralized_damping.py`;
- `env/idm_residual.py`;
- `tools/run_idm_decentralized_authority_verification.py`;
- `tools/analyze_idm_p30_projected_blend.py`;
- the corresponding focused tests.

The completed treatment roots are:

- `runs/probe/idm_p30_projected_blend_m2000_l1_development_v1`;
- `runs/probe/idm_p30_projected_blend_m2000_l2_development_v1`;
- `runs/probe/idm_p30_projected_blend_m2000_l3_development_v1`.

The aggregate analysis is
`runs/probe/idm_p30_projected_blend_m2000_analysis_v1`. Its authoritative
outputs are `summary.json`, `case_strengths.jsonl`, `oracle_cases.jsonl`,
`command_diversity.csv`, and `summary.md`.

## Integrity and action credit

All identity gates passed:

- exact frozen 15-case set and matched source decisions;
- exact abstain-versus-abstain replay;
- identical target CAV and predecision state;
- exact diagnostic strength identity;
- strength 1.00 reproduced every frozen P30 projected command and outcome
  within replay tolerance.

The action space also has real command authority:

- strengths 0.25, 0.50, and 0.75 were meaningfully distinct from P30 in
  12, 11, and 10 of 15 cases, respectively;
- 10/15 cases realized at least three projected command levels;
- all command diagnostics were present and final projection was idempotent.

This is an important positive result: projection no longer collapses the
candidate modes into one action. It is not sufficient evidence of useful
state-conditioned control.

## Mechanics and safety failures

One 0.25-strength treatment at seed 4129, decision step 354, aborted after 15
of 20 commands. At the abort step, the comfortable speed floor was
23.7569 m/s while SUMO's front-safe follow-speed cap was 23.7504 m/s. The
required acceleration was about -1.0065 m/s2, slightly outside the frozen
-1.0 m/s2 comfort bound. Front safety correctly dominated and release returned
the CAV to IDM. This is a real constraint, dose, and zero-conflict failure, not
a numerical or identity defect.

The same case produced executed command changes of about 0.953 m/s2 and
0.898 m/s2 for strengths 0.75 and 1.00. The existing projector first applies
the 0.5 m/s3 jerk preference, then permits a stricter newly binding front
follow-speed cap to override it. That safety priority is defensible, but it
violates the frozen protocol's unconditional executed-jerk gate. The current
projector also does not label front-versus-jerk infeasibility as a projection
conflict. The result remains a literal gate failure; it was not waived after
viewing outcomes.

There were no collisions or teleports, and aggregate severe-braking exposure
did not exceed P30. A null immediate-follower TTC in seed 4132, decision 291,
strength 0.25 was corrected in analysis: 31 follower samples were present and
all were nonclosing, so null represents infinite/not-applicable TTC rather than
missing telemetry. The corrected local safety gate passes. A future trace
version should separately count valid nonclosing and missing-kinematics samples.

## Outcome headroom

Frozen P30 across the 15 cases had:

- 6 helpful, 5 neutral, and 4 harmful outcomes;
- mean effect 0.02455;
- tenth-percentile effect -0.13182.

The safety/service-admissible per-case active oracle had mean effect 0.06414,
one harmful case, and a tenth percentile close to zero. Including abstention
raised the deployable oracle mean to 0.07408. These optimistic values are not a
learned policy result.

The apparent oracle gain is mostly the value of abstaining from harmful P30,
not evidence for choosing a graded intermediate authority. Native/P30 endpoint
selection alone has mean oracle effect 0.06192; intermediate strengths add only
0.01217 on average.

Only two cases exceeded the 0.02 deadband relative to the better endpoint:

- seed 4123, step 61: strength 0.50, incremental gain +0.05911;
- seed 4132, step 291: strength 0.25, incremental gain +0.11075.

The protocol required at least three cases and actionable winners across at
least three seeds. Three P30 harms became nonharmful under an intermediate
strength, but all three became neutral rather than helpful; the protocol
required at least one helpful rescue. Seed 4127, step 200 remained harmful
under every strength, with best active effect -0.14839.

The final failed gates are:

1. all branches satisfy constraints;
2. all active arms deliver 20 commands;
3. zero projection conflicts;
4. all executed command jerk remains within 0.5 m/s3;
5. blend-specific gain in at least three cases;
6. actionable winners across at least three seeds;
7. at least one harmful P30 case becomes helpful.

## Why observation and reward verification stopped

The frozen protocol permits the 30-decision expansion and grouped
observation/reward audit only after authority passes. It did not pass.

The existing 15 cases contain 26 qualified nonneutral option-versus-abstain
contrasts and 68 informative option pairs, but they cover only six development
seeds. The grouped audit requires at least eight. Expanding after a failed
treatment gate would be post-hoc model fishing, so no extra blend branches,
feature model, option reward, encoder repair, or RL training were run.

## Scientific interpretation and next design boundary

Projected interpolation is useful as a mechanism diagnosis: it proves that
the controller can deliver graded decentralized authority and that weaker
actions can bound several harmful P30 outcomes. It does not establish that a
policy can reliably select a beneficial intermediate action.

Pruning strength 0.25 is not a repair: it supplies one of only two actionable
gains and two harm neutralizations. Another post-hoc fixed grid, percentile,
duration, or winning-strength selection is not supported by this cohort.

If decentralized action development continues, the next scientific treatment
must be separately frozen and materially different: a closed-loop bilateral
law with explicit safety-compatible transition behavior, rather than another
open-loop strength grid. That change alters the treatment definition and must
be discussed before implementation. Until then, RL training and the later wall
and escalation-predictor phases remain unauthorized.

## Verification

Focused blend and authority tests pass. The complete repository suite passes:

```text
677 passed
```

No prospective evaluation seed was consumed.
