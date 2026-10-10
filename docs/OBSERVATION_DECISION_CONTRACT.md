# Observation and decision contract v2

Successor to the [gate feature contract](GATE_FEATURE_CONTRACT.md) (which
remains v1 for the entry feature packet). This version separates the two
decision problems, fixes the label vector and evidence taxonomy, and states
what information exists at each decision. It binds every dataset and analysis
in `output/observation_decision_v1/`.

## The two decisions

**Decision A — entry.** At an opportunity (currently 600 s), continue damping
or attempt one concrete wall proposal (per-CAV alternatives: native release and
PI external-U; FollowerStopper retired 2026-09-25 and no longer an action;
per-lane/per-cell mix layer retired 2026-09-25 WM-39 - per-CAV actions,
wall = gated group action) (members, lanes, target law, H20
commitment). Information available: the v1 65-field acquired packet (member
self state, anonymous same-lane neighbors within 100 m, 20 s histories, ages,
coverage, detection flags, proposal geometry/runway estimates, reporting-CAV
count) plus everything reconstructable offline from received report frames
alone (per-region/per-lane reporter speeds, slow-reporter fractions, trends,
queue proximity), each with its acquisition clock. Post-entry measurements
(member response to wall commands, follower compression, realized assembly)
are **excluded** from the entry dataset.

**Decision B — continuation.** At a tick inside an executed coordination
(currently measured HOLD start + 5 s = 617 s in 4107, 623 s in 4108), retain
the target, lower it, raise it, or begin release. Information available: the
entry packet frozen at 600 s, all reports received since, the member's measured
response to commands so far, follower/rear compression observed through member
rear-neighbor reports, downstream reporter proximity/speeds at the decision
tick, remaining commitment geometry, and the pre-entry damping history of the
same members (for post-handback motion uncertainty). Outcomes of other
branches are labels, never inputs.

## Label vector (both decisions)

Every attempt carries the matched full-horizon (2400 s) vector of paired
physical differences versus its genuine damping reference, per requested
vehicle: severe-braking exposure (vehicle-s), squared-acceleration effort
(m²/s³), total service/system time (s), source waiting (s), TTC<3 exposure
(vehicle-s), plus collision involvements, starting teleports and unfinished
requested service (counts), with requested-population and completion flags.

- The **original benefit criteria** are unchanged: a useful attempt reduces
  service by ≥1 s/request or braking by ≥0.02 vehicle-s/request, with no
  increase beyond 0.5 s service, 0.1 s source wait, 0.005 vehicle-s braking,
  0.5 m²/s³ effort, 0.02 vehicle-s TTC, and zero treatment collisions,
  teleports or unfinished service.
- The **strict joint diagnostic** is separate and descriptive: all five
  continuous costs non-worsening (tolerance 1e-9) plus absolute safety and
  complete service plus material primary improvement. It never changes the
  original label.
- Harmful attempts and failed assemblies keep their outcomes as valid
  negative labels. A refusal is a policy outcome of the entry decision; it is
  never a label for the unexecuted wall.

## The two prediction problems (never merged)

1. **Whether a wall beats damping** (entry). With six delivered walls across
   two seeds, five harmful and one nonbeneficial, this problem currently has
   zero positive examples and no trainability claim.
2. **Which continuation is less harmful** (continuation). Evidence that one
   branch is less harmful than another does not establish successful
   escalation while every branch fails against damping; it establishes action
   tradeoff structure only.

## Evidence taxonomy for every feature

Each feature concludes in exactly one of: **acquired** (receivable under the
sensing/report contract), **state-informative** (distinguishes traffic states
across seeds/times), **feasibility-informative** (predicts delivery/execution),
**outcome-associated** (descriptively co-varies with matched physical
differences), or **action-validated** (held-out, whole-seed-grouped prediction
of the label with physical decision value). These categories do not collapse.

## Runtime boundaries (unchanged)

Inputs: timestamped CAV self position/lane/speed/acceleration; anonymous
neighbors within the existing 100 m same-lane sensing model; acquired history
with receipt age, missingness and coverage; the proposed action and controller
state. Excluded: true density, penetration, driver type, global simulator
state, private lane-change parameters, future arrivals, offline counterfactual
outcomes, future rollouts, seed identity. Counts describe reporting CAVs;
missing reports do not imply empty lanes; overlapping detections are not an
HDV census. Wall actuation keeps −0.3 m/s² voluntary braking, 0.15 m/s³
ramp, compactness, finite exit, identical damping handback; HDVs receive no
commands; ramp-CAV damping remains allowed.

## Splitting rule

Whole seed groups are the only split unit. Time samples, branches and
vehicles from one traffic realization never straddle training and evaluation.
Preprocessing and thresholds fit only on the training portion. With two
exposed seed groups, everything is exploratory; no calibration or
generalization claim is admissible.
