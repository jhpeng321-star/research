# IDM v5.5 damping-only density mapper protocol

Status: frozen and executable, 2026-09-01. Development identification only.

## Scientific question

Under the exact current analytical IDM damping control arm, which tested
loading cells contain post-anchor sustained-congestion incidence diversity
that can support one later matched damping-versus-wall treatment question?

This mapper has no no-op or wall arm. It cannot establish damping efficacy,
estimate a universal critical density, verify wall headroom, train a predictor,
or open validation/final partitions.

## Frozen contract

- Long IDM/LC2013 corridor and demand assets already used by v5.4.
- Mainline demand: 2200, 2400 and 2700 veh/h; ramp: 400 veh/h.
- CAV penetration: exactly 0.30, seeded scatter.
- Control: `idm_analytical_disturbance_p30_v1`, fingerprint
  `75649f14ea355869a40558f8dfb0afc6057385b55c182a8235b6fb2c9b8fe2c7`.
- Intervention schedule: damping only at every controller step; no wall
  command, member or override is permitted.
- Development seeds: 4401--4406. These remain excluded from validation/final.
- Independent runs: 3 demands x 6 seeds = 18.
- Determinism repeats: seed 4401 once per demand, after primary completion.
- Total branches: 21. Repeats are never counted as independent seeds.
- SUMO integration: 0.5 s; controller and bounded telemetry: 1 Hz.

The frozen contract fingerprint is
`0989359d85ababf579444bb79859935220fe3ccb5a9b4f799544ede10c8d4054`.

## Absolute clock and endpoints

The analysis clock is absolute SUMO time, not controller steps after reset.
The anchor is 620 s and the common endpoint is 1080 s. Every valid branch must
contain uninterrupted 1 Hz observations from 561 through 1080, except that a
natural empty-network termination after the anchor is a proven recovery and
needs no fabricated post-clearance samples. An exact 620 sample and the full
60 s prevalence lookback are mandatory.

The primary physical endpoint is at least five premerge vehicles below
15 m/s together with premerge mean speed at or below 15 m/s for 60 consecutive
seconds. Labels are mutually distinguished as:

- prevalent at the 620 s anchor;
- incident after the anchor;
- no incident or recovered;
- censored or invalid.

Only incident-after-anchor cases count toward actionable transition diversity.
Congestion prevalent at the anchor is not wall-prevention headroom. A strict
`h < 0.15` for 60 seconds is reported independently as a legacy merge-health
diagnostic and never combined with the primary label.

Natural clearance after a sustained event retains the event and separately
records recovery. Any post-1080 replay samples are discarded before labels,
terminal state, flow, and population accounting.

## Integrity and interpretation

All of the following are mandatory:

- exact controller, scenario hashes, p30 and development partition;
- exactly 18 primary artifacts plus three repeat artifacts;
- exact repeat trace, controller-audit, telemetry, endpoint, terminal and
  flow/population identity in every demand cell;
- complete absolute clock/lookback or proven natural clearance;
- classified terminal reason;
- zero wall command or override;
- zero collision and teleport.

With six independent seeds per demand, actionable incident counts are mapped
only as development categories: 0--1 subcritical candidate, 2 lower-transition
candidate, 3--4 upper-transition candidate, and 5--6 overload candidate.
Any censoring invalidates the cell. Any anchor prevalence contaminates the
prevention interpretation; prevalence in at least four seeds means the anchor
is too late.

A mixed transition cell selects a later wall-treatment regime but does not
prove wall headroom. No mixed cell stops duration/predictor expansion and
requires a separately frozen density/anchor revision.

## Execution

```powershell
python -m experiments.idm.run_idm_v5_5_damping_density_mapper `
  --mainline-demands 2200 2400 2700 `
  --ramp-demand 400 `
  --cav-penetration 0.30 `
  --seeds 4401 4402 4403 4404 4405 4406 `
  --analysis-clock sumo_time `
  --analysis-start 620 `
  --analysis-end 1080 `
  --prevalence-lookback-s 60 `
  --primary-queue-count 5 `
  --primary-speed-threshold-mps 15 `
  --primary-sustain-s 60 `
  --secondary-health-threshold 0.15 `
  --secondary-health-sustain-s 60 `
  --repeat-seed 4401 `
  --workers 8 `
  --output runs/probe/idm_v5_5_damping_density_mapper_p30_s4401_4406
```

The runner is resumable. It prints an incremental classification after each
completed primary manifest. Expected elapsed time is 20--40 minutes with eight
workers. Bounded aggregate output is expected below 20 MB; reserve 100 MB.

Pre-execution verification: 46 focused tests and 903 complete repository tests
passed. Independent clock/scenario and analyzer falsification reviews issued
GO before launch.
