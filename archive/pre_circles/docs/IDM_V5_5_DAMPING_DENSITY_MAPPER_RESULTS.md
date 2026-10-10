# IDM v5.5 damping-only density mapper results

Status: complete, 2026-09-01. Development identification evidence only.

## Result

The frozen 21-branch mapper completed successfully. All integrity gates pass,
all three repeats are exact, and independent read-only reconstruction of the
physical endpoint found zero discrepancies.

The identified regimes are:

- 2200 veh/h: subcritical candidate, 0/6 incident, 0/6 prevalent, 6/6
  no-incident/recovered.
- 2400 veh/h: subcritical candidate, 0/6 incident, 0/6 prevalent, 6/6
  no-incident/recovered.
- 2700 veh/h: upper-transition candidate, 4/6 incident, 0/6 prevalent, 2/6
  no-incident/recovered.

This is a useful development result: 2700 supplies independent-seed outcome
diversity for one later matched wall question, while the two lower cells serve
as nonincident contexts. It is not an estimate of a universal critical density.

## Integrity

All frozen gates pass:

- exactly 18 independent primary branches and three repeat branches;
- exact long-network, scenario, route, p30 and analytical-controller hashes;
- uninterrupted 1 Hz absolute-clock coverage with the 561--620 s prevalence
  lookback, or valid natural clearance;
- zero prevalent or censored primary cases;
- no wall command, member or override in any branch;
- zero collisions and teleports;
- exact seed-4401 repeat identity at every demand across trace digest,
  controller-audit digest, telemetry digest, physical endpoint, health
  diagnostic, terminal state, and flow/population.

Repeat/numerical noise is zero at stored precision. Every trajectory ended by
natural clearance, at 1014--1036 s for 2200, 1017--1032 s for 2400, and
1036--1065 s for 2700.

The run took approximately 606 s elapsed with up to eight workers, 0.90 summed
worker-hours, and produced 66.6 MB of bounded artifacts. Pre-execution tests
were 46/46 focused and 903/903 for the complete repository.

## What the 2700 events mean

The four physical incidents are late:

- seed 4405: onset 907 s, confirmation 966 s;
- seed 4403: onset 916 s, confirmation 975 s;
- seed 4402: onset 940 s, confirmation 999 s;
- seed 4406: onset 940 s, confirmation 999 s.

Thus the exact confirmation sequence is **966, 975, 999 and 999 s**. The
proposed wall timing uses **880 s as the activation decision** and **900 s as
the planned release boundary**: twenty wall commands act over the transitions
880->881 through 899->900, after which identical damping resumes. Activation
at 880 must not be described as release at 880, and release at 900 must not be
described as a second activation.

Seeds 4401 and 4404 have only transient predicate runs of 36 s and 21 s. All
four sustained incidents later recover before natural clearance.

These are mixed late post-demand/drain-down congestion episodes after scheduled
inflow ends at 900 s. They are not evidence of irreversible gridlock or
persistent structural overload. The defensible future endpoint is prevention
or shortening of the late sustained premerge queue-speed episode.

The physical primary outcome is frozen as prevention or shortening of the
sustained premerge queue-speed episode, including incidence, duration and
recovery. Health remains a diagnostic feature only and is not an outcome gate.

The common 1080 s/natural-clearance horizon is mandatory for later work. A
terminal step of 960 would miss three of four confirmations and censor the
primary effect.

## Health diagnostic

The secondary `h < 0.15` diagnostic reports events in 3/6 seeds at 2200 and
6/6 seeds at both 2400 and 2700. It disagrees with the physical endpoint in
3/6, 6/6 and 2/6 seeds, respectively. Health therefore overlabels the lower
density cells and must remain a controller/diagnostic feature, not the paper's
collapse label.

## Exactly one selected next treatment question

The next question is timing-aligned, not a duration sweep:

> At 2700 veh/h and 30% CAV penetration, does the unchanged frozen v5.4
> 20-second wall activated at absolute SUMO time 880 s—covering the final
> 20 seconds of scheduled inflow—prevent or shorten post-release sustained
> premerge congestion relative to identical analytical damping, without
> worsening full-horizon hard-brake/TTC exposure, collisions/teleports, delay,
> or discharge through the common 1080 s/natural-clearance horizon?

This isolates one timing change while preserving wall duration, strength,
geometry, damping arm and outcomes. The wall pulse ends at the demand boundary
and precedes the earliest physical onset by 27 seconds. Treatment-period
safety and service costs still count, while the sustained-congestion endpoint
is evaluated post-release so intended temporary wall storage is not mislabeled
as treatment failure.

Before opening a wall branch, run a treatment-blind Gate-F formation/runway
check at 880 s on these six already-opened development seeds. Failure means the
question is infeasible, not that duration should be changed. If it passes, the
small matched pilot must carry bounded 1 Hz safety, follower-chain, flow,
release and delayed-event telemetry through the full horizon. Duration,
strength, geometry and additional densities remain frozen.

## Claims not authorized

This result does not verify damping efficacy, wall benefit, collapse
prevention, predictor accuracy/calibration, a critical density, or
generalization. It authorizes only the treatment-blind 880 s feasibility check
for the one selected 2700 development question. Predictor training and
validation/final partitions remain closed.

## Artifacts

- Protocol: `docs/IDM_V5_5_DAMPING_DENSITY_MAPPER_PROTOCOL.md`
- Manifest: `runs/probe/idm_v5_5_damping_density_mapper_p30_s4401_4406/manifest.json`
- Machine-readable summary:
  `runs/probe/idm_v5_5_damping_density_mapper_p30_s4401_4406/summary.json`
- Branch artifacts:
  `runs/probe/idm_v5_5_damping_density_mapper_p30_s4401_4406/cases/`
