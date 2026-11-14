# IDM v5.5 t=880 Gate-F and matched-wall pilot results

Status: completed development result, 2026-09-01. Do not expand the dataset or
train the escalation predictor from this result.

## Frozen timing and endpoint

The density-mapper confirmations are 966, 975, 999 and 999 s. The matched
intervention decision is the traffic state at absolute SUMO time 880 s. Wall
commands act from 880 through 899 s, their result states are 881 through 900 s,
and the return to identical analytical damping begins at 900 s. Both arms use
the common 1080 s endpoint or proven natural clearance.

The primary endpoint is physical sustained premerge congestion: at least five
premerge vehicles below 15 m/s together with premerge mean speed at or below
15 m/s for 60 consecutive seconds. Prevention or shortening must also improve
raw predicate exposure and the longest contiguous run by at least 10 s. Health
below 0.15 is retained only as a diagnostic.

## Treatment-blind Gate F

Gate F passes with four of six independent development seeds. Every admitted
case selects one CAV in each of the two occupied lanes, passes the frozen
front/rear activation-safety projection, and projects 20/20 s of treatment.
All six Gate-F files state `wall_was_applied=false` and
`treatment_outcomes_read=false`.

| Seed | Decision | Members | Maximum projected span | Terminal runway | Gate-F result |
|---:|---:|---:|---:|---:|---|
| 4401 | 880 s | 2 | 40.07 m | +485.80 m | admitted |
| 4402 | 880 s | 2 | 27.23 m | +333.14 m | admitted |
| 4403 | 880 s | 0 | 451.89 m | not reached | rejected: formation span exceeded |
| 4404 | 880 s | 2 | 4.23 m | +60.11 m | admitted |
| 4405 | 880 s | 2 | 38.92 m | +386.58 m | admitted |
| 4406 | 880 s | 0 | 37.29 m | not reached | rejected: insufficient kinematic headroom |

The unreported safety/runway layers for rejected seeds were not evaluated
after formation rejection; they are not observed safety failures.

## Matched treatment result

All four admitted treatments completed 20/20 command seconds with full lane
coverage, assembly completion, a five-step monotone taper and return to
identical damping. The two rejected cases applied no override and remained
exact physical no-op matches; their branch bookkeeping still records that a
wall command was requested and rejected.

| Seed | Physical congestion result | Safety/service result | Frozen label |
|---:|---|---|---|
| 4401 | no sustained control event; raw predicate and longest run 36 to 11 s, so no primary benefit | safety fails: +3 hard-brake steps and +3 onsets; service passes | safety failure |
| 4402 | sustained congestion 91 to 0 s: physical prevention | safety fails: +4 hard-brake steps/onsets; service passes with +2003.3 s delay benefit | safety failure |
| 4403 | 101 to 101 s; zero wall override | no added harm; service passes | infeasible formation |
| 4404 | no sustained event, but raw exposure worsens 21 to 51 s | safety fails: +5 hard-brake steps and +2 onsets; service fails with 1471.5 s delay loss | safety failure |
| 4405 | sustained duration 100 to 76 s and raw exposure 100 to 87 s: material 24 s shortening | safe; service passes, with 70.8 s delay loss inside the 300 s budget | safe beneficial |
| 4406 | 70 to 70 s; zero wall override | no added harm; service passes | infeasible formation |

There are two physical-primary benefits, but only seed 4405 is also safe and
service-admissible. Seed 4402 is important mechanism evidence—it shows that the
wall can prevent the physical event—but it cannot count as a deployable win
because the same treatment violates the frozen hard-braking guardrail.

The first-stage counts are therefore:

- infeasible formation: 2;
- safety failure: 3;
- safe beneficial: 1;
- safe nonbeneficial: 0.

## Dose and compactness caveat

The frozen duration/lane/taper/release gate passes for all four admissions.
Seed 4404 nevertheless fails the broader compact-composite diagnostic: only
75% of its commanded samples remain within the 50 m span and its maximum span
reaches 127.16 m. All five violations occur during the planned taper at action
times 895--899; assembly and holding remain within the cap, with a holding
maximum of 46.61 m. Its predeclared first-failure label remains `safety_failure`
rather than being changed after outcomes were seen.

The original `treatment_mechanics` convenience field in completed case files
reports zero taper rows because the helper was not given the stored controller
rows. The authoritative `treatment_contract.treatment_dose` and
`treatment_lifecycle_telemetry` correctly record five taper steps with blend
fractions 0.8, 0.6, 0.4, 0.2 and 0.0. The runner has been corrected for future
artifacts; completed scientific outcomes and labels were not rewritten.

## Integrity and repeat reconciliation

The pilot contains exactly seven paired case files and 14 branches. It passes
predecision identity, causal action-path checks, absolute timing, common
horizon/natural-clearance checks, 1 Hz telemetry completeness, out-of-band
damping identity, release to damping and zero override in rejected cases.

The exact repeat of seed 4401 passes all 12 trace, controller, telemetry,
outcome, endpoint, treatment-contract and classification comparisons. Repeat
noise is 0 s.

The density controls and pilot controls initially had different omnibus
`ReplayTrace` hashes despite exact aggregate telemetry. The post-execution,
no-wall reconciliation audit reran both control wrappers for all six seeds and
bound them to their original case files, reset records, controller-audit
hashes, scenarios and frozen controller contracts. For every seed:

- complete microscopic vehicle trajectories, active-CAV sets, collisions and
  teleports are exact;
- dynamic damping actions and diagnostics are exact;
- all supervisor observations are exact after excluding only index 10;
- the only raw snapshot difference is index 10, the non-actuating episode
  progress value (`step / max_episode_steps`).

Thus corrected physical/control execution integrity passes. The original raw
omnibus-hash comparison remains recorded as failed; it is not silently waived.

## Frozen decision

Expanded counterfactual generation fails independently on four scientific
requirements:

- only one safe independent benefit, below the required two;
- zero safe nonbeneficial outcomes, below the required two;
- only one of four admitted treatments is safety-admissible;
- one admitted treatment is service-inferior.

Therefore expanded data generation is not recommended, predictor training is
not authorized, and validation/final partitions remain unopened. This pilot
does not establish reliable congestion prevention, safe prevention,
generalization or a working escalation predictor.

The next defensible step is mechanism diagnosis, not more seeds: localize the
hard-braking pathways in 4401, 4402 and 4404; decide prospectively whether
compactness must hold through the complete taper; and redesign/revalidate wall
safety before repeating this frozen headroom gate.
