# IDM wall v5.2 p30 prospective pilot results

Status: **executed; treatment gate failed; predictor not authorized**.

Date: 2026-08-30

## Outcome

The compact full-occupied-lane v5.2 wall was evaluated at 30% CAV
penetration, selected without treatment outcomes by the preceding p10--p30
feasibility scan. The paired arms were frozen analytical IDM damping versus
the identical damping controller plus the temporary wall.

- 12 seeds and 48 predeclared decision windows were evaluated.
- 12 windows were admitted by the treatment-blind feasibility gate.
- Damping-versus-damping and treatment-versus-treatment exact replay passed on
  two independent seeds.
- Predecision identity and the damping-preserving overlay contract passed.
- 10 of 12 admitted cases passed the composite realized-dose contract
  (83.3%, required at least 80%).
- Beneficial completed treatments spanned seeds 4302, 4307, and 4311.
- Nonbeneficial completed treatments spanned seven seeds.
- Only 5 of 10 dose-qualified treatments were safety-admissible (50%, required
  at least 90%).
- No excess collision or teleport occurred.
- Collapse prevention was 0 of 10 completed cases; collapse remains a
  secondary outcome.

The safety gate is the decisive no-go. `predictor_authorized` is false. No
predictor-scale generation or training is permitted from this result.

## Dose failures

Both failures delivered the full 20-second duration and had zero aborts; the
failure was spatial delivery rather than early termination.

- Seed 4301, step 660: one of two latched wall members changed lanes at step
  664 and was removed without a same-lane handoff. Coverage fell to 23 of 40
  requested lane-steps (57.5%), leaving a one-lane action for 17 seconds.
- Seed 4312, step 660: both members and full lane coverage remained, but their
  span grew from 21.0 m to 118.3 m. Only 55% of wall samples stayed within the
  frozen 50 m compactness cap. Front-TTC projection clips and front/rear
  constraint conflicts prevented the servo from preserving compact geometry.

## Safety failures among dose-qualified cases

All five failures are caused only by excess hard-brake vehicle-steps. None has
excess collision, teleport, or low-TTC exposure beyond its budget.

- 4302/620: 35 to 142 hard-brake steps, +107 versus budget 4.
- 4307/680: 27 to 38, +11 versus budget 3.
- 4308/680: 33 to 158, +125 versus budget 4.
- 4309/620: 30 to 35, +5 versus budget 3.
- 4311/700: 14 to 17, +3 versus budget 2.

The additional events occur predominantly after wall release while both
branches have returned to native analytical damping. They are not direct
braking by active wall members or their immediate followers. Seeds 4302 and
4308 show severe wall-induced downstream breakdown and service loss. Seeds
4307 and 4311 produce large service gains but cross the global full-horizon
hard-brake budget. Seed 4309 is a marginal safety-boundary failure with nearly
neutral service.

## Completed treatment diversity

Safe, meaningful, service-noninferior wall outcomes occurred at:

- 4302/700: +1251.6 s delay benefit, +9 discharge, -3 hard-brake steps.
- 4307/660: +1782.5 s, +13 discharge, +4 hard-brake steps at its budget.
- 4311/660: +384.7 s, +6 discharge, +2 hard-brake steps at its budget.

This proves heterogeneous treatment effect and gives genuine escalation
headroom. It does not yet justify a predictor because half of completed
treatments violate the frozen safety margin.

## Required next decision

Do not add seeds or train the benefit model under v5.2. A new version would
need an explicit scientific choice between:

1. a mechanism redesign that preserves full-lane membership/compactness and
   reduces post-release wave creation; or
2. a formally redefined estimand in which the learned second stage predicts
   joint safety and service, with the current full-horizon hard-brake outcome
   retained as a label rather than required to be safe for 90% of all raw
   activations.

Either option changes the frozen scientific protocol and must be approved and
versioned before another pilot.

Artifacts:

- `runs/eval/idm_wall_v5_2_prospective_p30_s4301_4312/summary.json`
- `runs/eval/idm_wall_v5_2_prospective_p30_s4301_4312/seeds/`

