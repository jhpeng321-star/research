# Frozen IDM-Native Decentralized-Control Contract

Status: Phase-0 development contract, version `idm_decentralized_escalation/1`.
This contract controls the IDM-native pivot described in
`docs/IDM_DECENTRALIZED_ESCALATION_MASTER_PLAN.md`. Old Krauss paired rows are
not primary evidence under this contract.

## Simulator and network

- SUMO: 1.27.1 (Windows release build).
- Network: `D:\research\highway_merge_shepherd_long\osm.net.xml.gz`.
- Network SHA-256: `F58C4EF5F0497CFB887A298CD0FA4876BFE7BCEF101CC0ED2FA6801D503E93F5`.
- SUMO integration step: 0.5 s.
- Observation/control interval: 1.0 s.
- Collision action: warn; time-to-teleport: disabled (`-1`).
- Demand interval: 0--900 s; ramp demand: 400 veh/h.
- Mainline cells: 1900, 2000, 2100, and 2200 veh/h.
- Workspace-local scenarios: `scenarios/idm_transition/`.

The initial scientific roles are 1900--2100 veh/h for lower/near-critical
actuator evaluation and 2200 veh/h as the measured stochastic-transition
candidate. Phase 0 must reproduce this ordering before the role labels are
retained. The pre-existing 2400 cell remains an escalation/service-management
cell and 2700 remains structural-overload stress, not primary decentralized
collapse-prevention evidence.

## IDM and lane-changing types

All three HDV types use `carFollowModel="IDM"` and SUMO's default LC2013 lane
changer. Unspecified IDM parameters retain SUMO 1.27.1 defaults (`delta=4`,
internal `stepping=0.25 s`; passenger-car emergency deceleration remains the
SUMO default). Unspecified lane-change parameters retain LC2013 defaults,
including `lcStrategic=1`, `lcSpeedGain=1`, and `lcKeepRight=1`.

- Aggressive, probability 0.10: tau 0.95 s, sigma 0.6, speedFactor 1.08,
  accel 3.0, decel 4.5 m/s2, minGap 1.5 m, lcCooperative 0.3,
  lcAssertive 0.6, impatience 0.7.
- Normal, probability 0.70: tau 1.1 s, sigma 0.4, speedFactor 1.0,
  accel 2.6, decel 4.5 m/s2, minGap 2.5 m, lcCooperative 0.6,
  impatience 0.3.
- Cautious, probability 0.20: tau 1.5 s, sigma 0.2, speedFactor 0.92,
  accel 2.0, decel 3.0 m/s2, minGap 3.5 m, lcCooperative 1.0,
  impatience 0.0.

CAVs are selected independently at departure with seeded Bernoulli probability
0.20 (`scatter` placement). Paired evaluation uses the same integer for SUMO
and CAV selection. Selected CAVs are retyped to `normal_car`, retaining IDM and
LC2013. HDVs retain the route-file draw.

## Observation and action timing

Each policy call observes the state after the preceding 1 s environment step.
It computes one command, which is applied before SUMO advances another 1 s
(two internal 0.5 s ticks). The fixed physical observation blocks are ego[9],
zone[12], and 20 neighbor slots of five values. Optional context follows in the
single order declared by `env.features.context_feature_names`. Action-mask bits
follow the complete physical vector and are never embedded as measurements.

The new residual action is one scalar acceleration correction in
`[-0.5,+0.3] m/s2`. It affects only deterministic selected absorbers. A zero
residual must reproduce the analytical selected-absorber controller exactly.
Nonselected CAVs remain under native IDM.

## Analytical actuator

For a selected absorber:

```text
a_nom = 0.15 (v_ref - v)
      + 0.02 clip(gap - (s0 + 1.4 v), -20, 20)
      + 0.20 (v_lead - v)

a_cmd = safety_project(a_nom + absorber_gate * delta_a_rl)
```

The downstream reference is the lane-specific 30th speed percentile in the
frozen preview window, low-pass filtered with a 20 s time constant and bounded
in rate and magnitude. Selection permits at most one primary absorber per lane
or non-overlapping influence region, requires at least 150 m runway, enforces
150 m absorber spacing, and latches membership for at least 10 s.

The deterministic projection enforces comfortable acceleration, jerk, leader
follow speed/secure gap, front TTC, follower secure gap/rear TTC, allowed speed,
and SUMO speed mode 31. Front collision avoidance dominates a follower/leader
constraint conflict. Release blends toward `getSpeedWithoutTraCI` before
`setSpeed(-1)` returns full authority to native IDM.

## Metrics

Wave: lane/corridor speed-variance amplification, oscillation energy,
low-speed-wave duration, minimum speed, propagation duration, and exposed
vehicles.

Safety: SUMO collisions/teleports, all-vehicle hard and severe braking,
integrated negative acceleration, same-lane TTC count/minimum, jerk, and
follower response following absorber commands.

Service: mean time loss, travel time, arrived/unfinished vehicles, discharge,
throughput, and spillback.

Control: controlled CAV-seconds, selected/simultaneous absorber count, command
magnitude, projection reasons/frequency, hold time, and handoffs.

Sustained collapse initially means at least five pre-merge vehicles below
15 m/s while pre-merge mean speed is at most 15 m/s for 60 consecutive 1 s
controller steps. Phase 0 may reject the scientific role of a demand cell, but
must not tune this definition against treatment outcomes.

Replay tolerances are absolute state tolerance `1e-6` and scalar metric
tolerance `1e-9` for identical branches.

## Seed partitions

- Initial development: 4101--4120; analytical pilot opens only 4101--4106.
- Controller feature validation: 4121--4132, frozen only after the initial
  feature-family diagnosis and excluded from checkpoint acceptance.
- Controller validation: 4201--4240.
- Moving-wall treatment development/headroom: 4301--4320.
- Predictor development: 4401--4520.
- Predictor validation/calibration: 4601--4660.
- Untouched final prospective test: 5001--5060.

Every opened development seed must be recorded in its run manifest. No final
seed may be inspected before all controller, treatment, predictor, and protocol
gates are frozen.

## Phase-3 analytical gate

At least one empirically confirmed transition cell must show a 10% or greater
reduction in wave energy or severe-braking exposure (15% preferred), benefit in
at least four of six development seeds, zero collisions, no material TTC or
follower-braking deterioration, no more than 1.0 s/vehicle additional mean time
loss (0.5 s preferred), no material discharge loss, and effects above replay
noise. The analytical and zero-residual-interface arms must be replay-identical.

The existing checkpoint
`runs/rllib/shepherd_pruned_v1_dense_mpen/checkpoints/best` is a Krauss-trained
comparison arm only. Its `algorithm_state.pkl` SHA-256 is
`1191F787C6AA9CF3EB969FA6729E16D0A1B31A58F21FBCA8ABAA9F9463FA6F71`.
The previously documented `champ_idm_band/checkpoints/best` does not exist and
cannot be treated as evidence.
