# S-SETUPAUDIT2 metric contract

Named real reports were absent from this branch; read as Git objects only:
`266d5e37:docs/paper_results/CTRLARM.md` and
`69b4524d:docs/paper_results/E3FINAL_A.md`.
No real episode/seed files were read and no simulator was run.

Both reports' physical team delay is the full road fleet's unnormalised
speed deficit at 31.29 m/s plus authoritative source waiting, including entrants.
CTRLARM uses 23 equal-weight worlds, 300 s native warmup / 600 s intervention;
its MAPPO-CTRL0 interval includes zero. E3FINAL_A uses 228 worlds, equal weight
over twelve penetration/demand cells, and compares cycle-1 C1 with MAPPO.
The requested panel instead averages 24 distinct restored TRAIN-sampler worlds,
time-in-episode 300..1180 s, demands 1900/2100/2400/2700, shares 5/10/20/40%.
The different initial states, policies, sampling weights and dynamics preclude
an exact-world real/WM validation or attribution of the real effect to a lane head.

Physical WM J has the same population, speed reference, additive road/source
terms and units. The trained actor's weighted group/individual/cooperation
suffix-time, joint-K8-q10 credit is a different functional, against learned-native
rows rather than CTRL0. The old t0-HDV-road qualification is a third estimand;
it must not be mistaken for either report's full-team metric.

## Birth/exit boundary recomputation

`traffic_shepherd/c4/branching.py:262-295,321-367` defines the real road cohort:
previous boundary occupants; newborns wait until the next interval; exiting
vehicles carry their last observed speed. `traffic_shepherd/metrics.py:30-55`
integrates requests at their continuous due times and insertions at the reported
boundary. These two source files have identical Git blobs at E3FINAL_A's
69b4524d and this branch (7e69a8e9 / d2c8bdd7 respectively).

WM insertion happens before a physics transition, and objective14 charges that
newborn immediately; its source count is the remaining post-insertion backlog.
For `real_J`, panel2 excludes that newborn road tick, substitutes prior speed
for exits, and charges `dt * pre-insertion pending + sum(now - due_time)`.
The due-time observer wraps the existing renewal headway call without drawing
another random number. This is a metric-only reconstruction on WM trajectories,
not another simulation or an intervention on the WM's insertion dynamics.

## Decomposition and interventions

Road spatial bins are disjoint **current locations**: merge zone s in [0,75.44],
ramp lane 2 outside that interval, mainline remainder. They differ from the real
reports' requested-route origin groups. Source mainline/ramp labels are the
actual two entry processes, not a spatial assignment.
HDV/CAV road costs use observable is_cav; queued requests have no realised
is_cav identity yet, so their source cost is explicitly untyped. No truth/type
assignment file is used. Initial means on-road at t0; entrant_total is entrant
road plus all source waiting (including vehicles already queued at t0).

The two crossed actors are closed-loop policies on their own evolving geometry,
not replayed actions from a different trajectory. MAPPO_native_lanes preserves
absolute longitudinal ownership but delegates lanes and ramp first passage to
the same native WM law as CTRL0. CTRL0_MAPPO_lanes executes the exact CTRL0
longitudinal law and uses the MAPPO lane bridge, observing the actual hybrid's
last longitudinal command. All share the prior audit's cap/Euler alignment.

MAPPO_CTRL0_admissions retains MAPPO's lane/longitudinal policies but supplies
CTRL0's admission geometry to the same keyed entry process. Thus insertion
times/states, newborn identities and source queues are fixed to CTRL0 while
existing road vehicles continue on the hybrid's own state. This deliberately
breaks queue/road feedback; it is an attribution intervention, not deployable
traffic physics. Source waiting is preserved rather than zeroed. Virgin slots
are used once. The paired arrivals and learned entrant-mix keys remain shared.
