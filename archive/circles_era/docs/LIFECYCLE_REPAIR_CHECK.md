# Part B: finite remaining wall commitment

Owner authorization: execute the stages of NEXT_EXPERIMENT_PLAN.md and report
after each part. Part A has selected the unchanged original network and control
region; its acquired spatial/traffic evidence is frozen into this registration.
On-ramp CAV damping is intentional. No shared asset or CIRCLES change is made.

One version: finite_lifecycle_wall/1. Retain -0.3 m/s² voluntary wall braking,
+1.5 m/s² maximum acceleration, 0.15 m/s³ ramp, 50-second assembly deadline,
three consecutive measured settled samples, 20-second maximum compact HOLD and
at most 10 seconds for exit. The original acquired mean-speed-minus-2 target,
24 m/s ceiling, two registered mainline members, gains and sensing remain fixed.

The controller integrates its own declared command law over only the remaining
assembly, HOLD and exit. Every forecast uses current acquired member state and
the exact resolver, assuming commanded acceleration response; it calls no future
CIRCLES inference and no simulator shadow. Predicted readiness never substitutes
for measured readiness. Forecast compactness is checked at half-second HOLD
boundaries. Map displacement and reported front/downstream relative clearance
are checked at each half-second forecast prefix, not only the terminal sample.

Other acquired vehicles are conditional constant-speed constraints. Current and
one-control-interval rear clearance/TTC protection remains; full future rear
response is NOT certified by assuming followers never respond. Missing traffic
beyond sensing is unknown. Receipt age and missing report/lane/member checks
remain enforced. Finite feasibility is neither a safety proof nor a benefit gate.

Exit is an explicit WALL-owned transition toward zero acceleration. Successful
terminal handback requires each member's previous issued wall acceleration to
be zero and its acquired measured acceleration within one ramp step of zero.
Then empty wall output returns to the identical live CIRCLES instance, including
actual override release when its native branch is selected. Zero is not an
invented IDM/native acceleration or a speed-matched native prediction. Failure
to reach this condition by the exit deadline is an abort, not successful release.

At entry, reject an infeasible full commitment. During execution, if continuation
is infeasible but immediate finite exit is feasible, start that exit. If neither
is feasible, preserve protective abort and handback. If already at the coast
terminal, immediate explicit handback is allowed. Ongoing feasibility uses only
remaining phases, never charges elapsed HOLD again and never forecasts a distant
slow reporter to a 78–107-second stop beyond the wall-owned horizon.

Physical budget: exactly two separately charged mechanics fixtures, each at most
120 simulated seconds including warmup; zero retries. Fixture 1 must demonstrate
positive entry acceleration, nonvacuous assembly and full HOLD/exit. Fixture 2
must demonstrate a physically developing, acquired downstream constraint during
HOLD and a verified early finite exit. The mechanics registration records exact
IDM route/stop geometry and pure preflight estimates; those estimates are not
SUMO evidence. No fixture outcome tuning. Stop on implementation or fixture
verification failure and preserve all charged artifacts.

Only after both fixtures pass: two full 2400-second wall episodes, exposed seeds
4107 and 4108, mainline2700/p20, decision600, identical registered original
cohorts. Reuse the exact verified damping references after source/scenario/hash
and complete-prefix checks. Corridor ceiling4800 simulated seconds; total Part B
ceiling5040 seconds. No automatic retries. Full episodes continue after physical
refusal/abort to retain delayed service and safety outcomes. Other staged blocks
require separate outcome-dependent registrations; they are not hidden successors
inside this runner.

Primary physical labels and margins are identical to the original panel. Report
all-request service/source waiting, measured severe braking a<-3, effort integral
a²dt, TTC<3 exposure, collisions, teleports and unfinished trips. Preserve the
original gentle results as the preceding-policy comparator. Independently score
full nominal HOLD, actual HOLD, successful declared early exit, abandoned assembly,
protective abort and actual handback separately. A valid early exit is not a
completed20-second HOLD. All-refusal is not positive escalation headroom.

Additional descriptive success: all five continuous costs nonworsening (1e-9
numeric comparison tolerance), at least one original material primary benefit,
all service completed and zero treatment collision involvements/teleports.
This does not alter prior labels or turn a baseline incident into wall-caused harm.
Two exposed seeds do not establish generalization, a fitted benefit gate or safety.
