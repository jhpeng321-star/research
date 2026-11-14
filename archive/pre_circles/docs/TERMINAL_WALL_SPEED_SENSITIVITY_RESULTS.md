# Terminal Wall-Speed Sensitivity Results (v5.2 Development Audit)

## Status

Completed on 2026-08-27. This is excluded-case treatment development, not a
prospective pilot and not predictor verification.

Artifact:

runs/probe/counterfactual_terminal_speed_v5_2_development/summary.json

## Main result

Terminal wall speed is a meaningful treatment dimension. The previous 20%
maximum speed reduction was limiting effectiveness in some traffic states.
However, no tested speed is universally safe or effective, and the 30% level
does not yet pass the frozen safety threshold needed to proceed directly to a
fresh predictor/headroom pilot.

All 15 primary rows passed predecision identity and the overlay action contract,
received 100% treatment duration and lane coverage, and produced zero excess
collisions or teleports.

| Maximum speed reduction | Mean requested target (m/s) | Mean terminal actual speed (m/s) | Strict dose + geometry | Safety-admissible among strict | Selected among strict | Strict collapse prevention |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 24.92 | 20.94 | 3/5 | 1/3 | 1/3 | 0 |
| 20% | 22.16 | 19.35 | 4/5 | 1/4 | 1/4 | 2 |
| 30% | 19.59 | 18.47 | 4/5 | 3/4 | 3/4 | 1 |

The stronger level therefore improved safety/service action headroom but did
not create adequate safe collapse-prevention diversity. Its 75% safety rate
among strict cases is below the frozen 80% development threshold and the later
90% prospective-pilot threshold.

## Primary case results

### Seed 504, step 180

No tested wall is safety-admissible. The 20% wall prevents sustained congestion
and improves discharge by 14, but adds six hard-brake exposure steps. The 30%
wall accurately reaches about 18.7 m/s but adds four low-TTC exposure steps and
does not prevent collapse. The correct action under the frozen safety/service
criterion is damping.

### Seed 606, step 180

The relationship is nonlinear. The 10% and 30% walls are selected, while the
20% wall is unsafe. The 30% wall is the strongest safety result: eight fewer
hard-brake exposure steps and four fewer low-TTC steps. It costs two discharged
vehicles and about 164 seconds of delay but remains within the frozen service
budget.

### Seed 607, step 180

The 10% wall prevents collapse but fails compact geometry. The 20% and 30%
walls both pass strict geometry, are safe, and prevent collapse. The 30% wall
is fully compact, reduces hard-brake exposure by eight and low-TTC exposure by
two, adds eight discharged vehicles, and improves delay by about 858 seconds.

### Seed 609, step 120

The 10% and 20% walls are safety-inadmissible. The 30% wall reaches its target,
has no excess hard-brake exposure, reduces low-TTC exposure by seven, adds six
discharged vehicles, and improves delay by about 933 seconds. It is a strong
safety/service win but does not prevent collapse.

### Seed 612, step 120

All levels receive full duration but fail compact geometry. The fraction within
the 50 m span remains 0.20. Stronger terminal reduction improves maximum span
from 80.5 m to 66.8 m but does not repair formation. The 30% wall also violates
the safety budget. This activation should remain infeasible.

## Scientific interpretation

The experiment supports three conclusions:

1. The old 20% terminal-speed ceiling was a real action-space limitation.
2. Treatment response is state dependent and nonmonotonic; speed cannot be
   selected with a global fixed rule.
3. The evidence currently supports a safety/service escalation estimand more
   strongly than a primary collapse-prevention estimand.

The present development cases contain a useful binary pattern for a 30% wall:
three strict safe/beneficial cases and two damping/infeasible cases. That is
promising predictor headroom, but it is deliberately inspected evidence and
cannot establish predictive performance.

## Required next decision

Do not train a predictor or launch the fresh 12–20-seed pilot yet. First
diagnose the 30% wall's low-TTC failure in seed 504 and determine whether it
occurs during formation, holding, or release. Also diagnose why seed 612 loses
compact geometry despite full dose. If those are repairable without changing
the treatment's scientific meaning, rerun this bounded audit. Otherwise freeze
the 30% wall with an explicit feasibility/safety refusal definition and frame
the paper as counterfactual safety/service escalation, with collapse prevention
as a secondary outcome.
