# Escalation implementation and experiment tests

Execution update (2026-09-06): formation and paired runner are now implemented; the14-job natural panel is complete and S05 has0/8 qualifying attempts. Later stages remain gated. See [the execution record](ESCALATION_EXECUTION_STATUS.md); the stages below remain the scientific framework, not completed-result claims.

Owner objective, clarified 2026-09-06: keep the external damping controller fixed
and let an RSU use CAV-acquired reports to select a physically feasible wall when
it offers incremental benefit. **Any traffic state can benefit.** Unresolved
congestion, a density cutoff and poor damping performance are not prerequisites.

The [machine-readable catalog](../protocols/escalation_test_plan_v1.json) contains
12 ordered stages, tests, dependencies, deliverables and advance rules. The first
panel is executable. Offline geometry, formation, paired-effect, split and
probability evaluators have automated regressions. Passing these synthetic tests
does not mean that physical wall experiments or a trained predictor are finished.

## S00: Controller, actuator and counterfactual integrity

Keep the exact model SHA, embedded wrapper, raw SI singleton input, 1 s control,
0.5 s physics, 100 m sensing and speedMode31. Verify command ownership, measured
response, native fallback, HDV isolation and identical damping restoration.
Before interpreting wall effects, verify paired predecision trajectories and a
zero-effect sham. Existing actor/ownership regressions are reused. Stop on an
implementation or replay defect, not because damping fails to outperform IDM.

## S01: Acquirable observations and RSU receipt

Test map lanes, nearest-visible front/rear detections, coverage, missing data,
timestamps, duplicate and delayed packets. Reports are unavailable until actual
receipt. Stale or lost reports cannot become safe empty space. Hidden simulation
changes cannot change identical received evidence. Do not infer an HDV census or
coverage union from overlapping detections. Realistic occlusion, missed detections
and noise need explicit sensor assumptions; ideal-sensor regressions do not
validate those physical assumptions.

## S02: Fixed damping and opportunity capture

Initial panel: exposed seeds **4107 and 4108**, each at **2000 and 2700 mainline
veh/h**, ramp 400 veh/h, p20, native and damping: **8 full trajectories** with a
maximum **19,200 simulated seconds**, no automatic retry or expansion. Both seeds
are historical development seeds, never fresh/prospective. Using the same seeds
at both loads avoids the previous one-seed-per-load confound.

Measure severe braking, acceleration effort, TTC, requested service including
source waiting, unfinished trips, collisions and teleports. Retain favourable
and unfavourable outcomes; neither selects a new damping controller. Small panels
characterize development cases and do not establish population acceptance.

The separate `traffic.jsonl` records measured density in vehicles/km/lane, speed
mean/dispersion and stopped counts in each map region. Nominal demand is not
called density. This full-state channel is never fed to the actor or RSU.
`telemetry.jsonl` retains CAV-only reports. These same damping trajectories can
serve as wall counterfactuals when scenario, source and decision-prefix identities
are verified; reuse is not assumed from a similar directory name.

## S03: Spatial separation and actual formation

Read-only capture uses the fixed times **300, 600, 900, 1200, 1500 s**, two map
lanes and a provisional 200-1900 m recruitment band. At most 256 candidate groups
are enumerated per snapshot; truncation is reported and cannot prove absence of
an opportunity. Repeated snapshots are not independent seeds.

The physical initial-span test points are **10, 25, 49, 50, 51, 75, 100, 150 m**,
with selected speed mismatches, front/rear clearances and runway cases. A 50 m
formed-wall span is a provisional hypothesis. Initial candidate separation and
final compactness are distinct quantities. Merely rejecting inputs above a
programmed 50 m limit is not a measurement of physical assembly feasibility.

Use bounded synthetic SUMO mechanics to isolate geometry, then natural candidate
groups to test mixed-traffic validity. Changing natural member identities does
not isolate distance alone. The retained `wall.py` diagnostic only settles
already compact members. The active `formation.py` now implements the required
relative-position controller and has bounded controlled and natural evidence;
see the execution record for measured delivery and failures.

Record measured member position/lane/speed, front/rear gap and closing speeds,
actual acceleration, initial/max/final span, readiness time, compact HOLD,
refusal/abort and remaining runway. Test cut-ins, member lane changes/loss,
report loss, short runway and deadline misses. Formation failures remain data.
Missing trajectory samples cannot count as successful formation.

## S04: Authority, HOLD, taper and release

Register one report-based target rule, acceleration envelope, assembly/HOLD/taper
and hard-abort definition before wall outcomes. Use a few one-factor variants
only when diagnosing a specific failure. Avoid a full Cartesian search.

Verify compactness, lane coverage and actually delivered control. Measure follower
compression, service loss and delayed release-wave effects beyond the controlled
CAVs. Use sham and assembly-only ablations when causal attribution is unclear.
A smooth command is not proof of a smooth measured response or safe traffic.
Stop for a failed mechanism or invalid comparison; do not reinterpret dispersed
slowdowns as successful compact walls after seeing a beneficial outcome.

## S05: Incremental benefit at any traffic state

Compare attempted wall with continued identical damping for the same proposed
group, starting state and exogenous future at a common full service horizon.
Assembly, failed attempts, HOLD, release and delayed effects all count. Formation
delivery is a separate label, not a filter that removes costly attempts.

`BenefitCriteria` requires a material improvement margin for one or more primary
physical costs and a maximum permitted increase for every tracked cost. Units
are explicit and per-request normalization retains source waiting and unfinished
work. This allows declared trade-offs in an overall improvement claim. Scientific
numerical margins need a recorded rationale before wall outcomes; no defaults
are silently chosen to approve exposed results. New collisions/teleports and
additional unfinished service are reported as violations by this evaluator.

Labels are beneficial, nonbeneficial or harmful; no congestion field is required.
If useful wall effects do not exist, diagnose the treatment before generating
predictor-scale data. If a simple feasible-wall rule already captures all useful
effects, report it instead of manufacturing class diversity to justify learning.

## S06: Penetration and the availability chain

An optional explicit extension uses the same seeds/loads at **p5, p10 and p30**,
paired native/damping: **24 additional trajectories**, maximum **57,600 simulated
seconds**, no automatic launch or retry. This complements p20 rather than
repeating it. Wall branches require their own registered profile and allocation
after S04/S05; this initial CLI cannot yet launch them.

Report all registered decision opportunities, member availability, sufficient
information, feasible proposals, approvals, attempted assemblies, completed walls
and physical outcomes. Distinguish missing members, unknown information, observed
unsafe geometry and insufficient expected benefit. Low penetration is not
programmed to fail and true penetration is not an online input.

## S07: Dataset, splits and observation sufficiency

Pair proposal/evidence hashes with continuous physical differences. Use local
geometry, self/neighbor histories, report counts, ages, coverage and missingness.
Any health summary needs an explicit acquired-input definition. True density,
penetration, demand, hidden driver parameters, future outcomes, episode IDs and
privileged simulator rollouts are excluded from online model features.

Split entire seeds across training, calibration and prospective evaluation; all
windows and cohorts of one seed stay together. Independently audit archive
exposure. Registry absence does not prove freshness, and existing sealed seed
blocks remain sealed. Compare current observations with short histories and a
transparent rule before increasing model complexity. An offline full-state oracle
can diagnose information limitations but cannot supply online features.

## S08: Feasibility reliability

Compare the gate's predecision assessment with actual formation/HOLD and local
safety evidence. Measure false approvals, false rejections and missing-information
abstention by penetration, starting span and report quality. A reliable transparent
rule is sufficient; a learned feasibility gate is optional. Calibrate probabilities
on separate episodes. Predicting a programmed geometric threshold is not physical
validation.

## S09: Benefit prediction and threshold

Compare a small learned predictor with report-based selection on identical
proposals. Measure false escalations, missed helpful opportunities, coverage,
probability reliability/Brier and physical policy value. Abstentions remain in
the overall recall denominator. Freeze physical labels and the decision threshold
using development/calibration only. Test benefits in uncongested as well as
congested states; a traffic-state label must not force either gate's answer.

## S10: Complete escalation lifecycle

Both gates approve the same proposal and current report snapshot. Test each
rejection, stale authorization, repeated opportunities, cooldown, no duplicate
wall command ownership, lost members/reports, abort and identical damping return.
One active wall is the initial scope. Gate-composition regressions do not replace
an integrated SUMO policy runner, which is still to be built after wall qualification.

## S11: Prospective physical policy and robustness

Freeze treatment, features, gates and scheduling. Use untouched audited seeds at
p5/p10/p20/p30 in the declared traffic conditions. Compare damping, a report-based
rule, feasibility-only escalation with a declared action budget, and both gates.
Report physical effects, refusal/intervention frequency, failure cases and
seed-cluster uncertainty. Never tune from prospective results.

Separately thin reports while holding controllable membership fixed, and thin
control membership while holding observation access fixed, as clearly labeled
diagnostics. Test registered delay/loss, sensor degradation, flow and IDM
perturbations. Information loss may change decisions; measure degradation and
abstention rather than demanding artificial invariance.

Simultaneous multiwall operation is optional additional work: disjoint membership,
joint spacing/storage/catch-up, combined control, coordinated release and benefit
conditional on another active wall. It is not required for the initial paper.

## Execute only a declared stage

Use the existing `rl_driving` Python and SUMO environment. These commands inspect,
test or register; registration starts no simulation:

```powershell
python -m pytest -q
python -m ruff check traffic_shepherd tests
python -m traffic_shepherd.escalation_study plan
python -m traffic_shepherd.escalation_study register --output runs/active/escalation_pilot_v1
python -m traffic_shepherd.escalation_study analyze --registration runs/active/escalation_pilot_v1/registration.json
```

One explicit job starts exactly one trajectory:

```powershell
python -m traffic_shepherd.escalation_study run --registration runs/active/escalation_pilot_v1/registration.json --job-id s4107_m2000_p20_native
```

The registration lists the other seven exact IDs. An exclusive allocation charges
2400 seconds before launch and remains on failure. Existing outputs cannot resume
or retry silently. Changed model, adapter, scenario or source invalidates the
frozen registration; register a new version rather than changing its hashes.
No blanket damping-performance pass condition exists.

After a complete damping job, capture only its received CAV reports:

```powershell
python -m traffic_shepherd.escalation_study geometry --run runs/active/escalation_pilot_v1/jobs/s4107_m2000_p20_damping --times 300 600 900 1200 1500
```

This initial collector assumes zero communication delay. Geometry output carries
the source telemetry hash and says that physical assembly is untested.

`score --kind formation`, `probabilities`, or `seed-summary` reads a JSON object
of arguments to the functions in
[escalation_evidence.py](../traffic_shepherd/escalation_evidence.py). Formation rows
have `time_s`, `phase`, and member `id`, `lane`, `distance_m`, `speed_mps`.
Definition fields are `required_lanes`, `assembly_deadline_s`, `hold_duration_s`,
`formed_span_m`, `speed_spread_mps`, `readiness_samples`, optionally `dt_s`.
These evaluators score supplied measurements; they do not create trajectories
or certify their provenance on their own.

Concrete input/failure examples live in the
[evidence tests](../tests/test_escalation_evidence.py),
[observation tests](../tests/test_escalation_observations.py), and
[panel tests](../tests/test_escalation_study.py). None of their synthetic outcomes
may be used as a paper result.
