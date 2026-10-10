# IDM escalation fallback routes

Status: development guidance only. These routes do not alter the frozen v5.4
treatment or authorize validation/final evaluation. Complete the current shadow
observation analysis before changing the scientific treatment definition.

## Current diagnosis

The v5.4 development cohort uses one nominal operating condition: 2700 veh/h
mainline, 400 veh/h ramp, 30% CAV penetration, four decision windows and a
fixed 20 s compact moving wall. This is the project's structural-overload cell,
not its primary collapse-prevention cell. Earlier matched IDM evidence found
wall collapse counts of 2/12 at 2200, 4/12 at 2400 and 12/12 at 2700 veh/h.
At 2700, the wall still improved delay and TTC outcomes in many seeds, so that
cell remains useful for safety/service management rather than a primary claim
of preventing collapse.

Density is material but insufficient. In the existing development-only
univariate audit, the strongest density feature had conditional-benefit AUC
0.638, whereas change in pre-merge speed variability reached AUC 0.725.
Relevant confounders and effect modifiers include temporal instability, ramp
arrival phase, lane imbalance, pre-merge headway dynamics, disturbance
propagation, CAV placement/runway, follower compression, treatment duration,
and post-release recovery. Because nominal demand is constant in the v5.4
cohort, it cannot establish a cross-demand density policy.

## Ordered fallback hierarchy

1. **Adaptive duration and closed-loop release.** Retain a minimum wall probe,
   continue while recovery is measurable and safety is admissible, and release
   when discharge/speed-variance recovery is achieved or a hard maximum is
   reached.
2. **Sequential successor walls.** Maintain one effective inflow-control front;
   assemble an upstream successor before the current wall reaches its release
   boundary. Prefer this over simultaneous independent walls at a single merge.
3. **Variable control area and placement.** Select the upstream wall location
   and controlled-area length from predicted congestion severity, storage and
   available CAV geometry.
4. **Adaptive wall strength.** Test a small discrete action library spanning
   abstain, mild/moderate/strong speed reduction and scientifically justified
   lane coverage before considering continuous optimization or RL.
5. **Merge-specific gap shaping.** If merge insertions dominate the failures,
   test gap closure, batched ramp merging and mainline gap redistribution rather
   than relying only on a full moving wall.
6. **Model-predictive moving-actuator control.** Predict queue accumulation,
   discharge, recovery time and spillback, then choose wall speed, duration and
   placement under safety constraints.
7. **Hybrid ramp/mainline coordination.** Coordinate the mainline wall with
   ramp platoon formation, gap creation or virtual ramp metering when ramp
   bursts are the disturbance source.
8. **Honest outcome pivot.** If dose and safety pass but collapse headroom does
   not exist, make safety/service harm avoidance the primary escalation target
   and retain collapse prevention as a conditional stress outcome.

## Smallest discriminating experiment after the current screen

Use the same 8--12 development seeds at 2200, 2400 and 2700 veh/h, with the
frozen 400 veh/h ramp and 30% penetration. Compare damping only with matched
20 s, 30 s and 40 s walls. Freeze safety, service, collapse and treatment-dose
criteria before execution. Measure safety admissibility, hard-braking and
low-TTC effects, merge discharge, recovery time, collapse prevention,
wall-induced upstream storage and post-release persistence.

Interpretation:

- Longer duration helps: proceed to adaptive duration, then sequential
  successor walls.
- Duration does not help and merge disturbances dominate: proceed to gap
  shaping or ramp/mainline coordination.
- Walls work around 2200--2400 but not 2700: train a regime-aware escalation
  gate across the transition and retain 2700 as structural-overload stress.
- Dose passes but no outcome changes: redesign the actuator instead of adding
  more identical seeds.

## Research basis

- Y. Du et al., *Adaptive control with moving actuators at motorway
  bottlenecks with connected and automated vehicles*, Transportation Research
  Part C 156 (2023) 104319. https://doi.org/10.1016/j.trc.2023.104319
- M. Cicic et al., *Platoon-actuated variable area mainstream traffic control
  for bottleneck decongestion*, European Journal of Control 68 (2022) 100687.
  https://doi.org/10.1016/j.ejcon.2022.100687
- S. Vishnoi et al., *CAV Traffic Control to Mitigate the Impact of Congestion
  from Bottlenecks: A Linear Quadratic Regulator Approach and Microsimulation
  Study*. https://arxiv.org/abs/2306.10418
- D. Chen, A. Srivastava and S. Ahn, *Harnessing connected and automated
  vehicle technologies to control lane changes at freeway merge bottlenecks in
  mixed traffic*, Transportation Research Part C 123 (2021) 102950.
  https://doi.org/10.1016/j.trc.2020.102950
- Y. Han and S. Ahn, *Stochastic modeling of breakdown at freeway merge
  bottleneck and traffic control method using connected automated vehicle*,
  Transportation Research Part B 107 (2018) 146--166.
  https://doi.org/10.1016/j.trb.2017.11.007
