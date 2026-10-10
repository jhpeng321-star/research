# IDM wall v5.2 release and downstream-safety diagnosis

Status: **executed on existing artifacts only; no new SUMO branches**.

Date: 2026-08-30

Source: `runs/eval/idm_wall_v5_2_prospective_p30_s4301_4312/`

## Question

The v5.2 pilot failed because five of ten dose-qualified cases exceeded the
full-horizon hard-brake budget. This diagnosis asks whether those failures are
localized at the wall-release boundary, which would support a simple smooth
taper, or whether they are delayed changes in the downstream traffic state.

The analysis uses the already recorded paired event onsets, treatment phases,
roads, vehicle classes, and terminal wall mechanics. No outcomes, thresholds,
labels, or treatment parameters were changed.

## Result

- 48 declared windows, 10 dose-qualified wall cases, and five unsafe cases
  were recovered exactly from the frozen artifacts.
- The five unsafe cases contain +251 hard-brake vehicle-steps relative to their
  paired damping controls.
- Only 2.3% of treatment-arm hard-brake exposure occurred while assembling or
  holding the wall; 97.7% occurred after active wall control.
- Paired hard-brake onset excess was zero during the wall, +1 in the first 20
  seconds after release, -5 from 20--60 seconds after release, and +248 more
  than 60 seconds after release.
- All five unsafe cases are classified as delayed downstream divergence, not
  release-boundary-localized harm.
- Of the aggregate paired onset excess, +231 events occur on downstream edge
  `1487606178`; the excess is distributed across both HDVs (+158) and CAVs
  (+86).
- Safe and unsafe cases have overlapping ranges for every inspected terminal
  mechanical indicator: wall span, member speed mismatch, final-five-sample
  span growth, projection clips, target error, and fraction within 1 m/s of
  target.

The result supports the statement that the wall changes the later metastable
traffic trajectory. It does **not** isolate abrupt release as the cause. A
five-second taper by itself is therefore not a sufficiently supported v5.3
mechanism.

## Per-case temporal result

- 4302/620: +107 hard-brake vehicle-steps; one positive excess onset in the
  first 20 seconds and 104 after 20 seconds, primarily downstream.
- 4307/680: +11; zero in the first 20 seconds and 11 after 20 seconds.
- 4308/680: +125; zero in the first 20 seconds and 126 after 20 seconds,
  primarily downstream.
- 4309/620: +5; zero in the first 20 seconds and four after 20 seconds.
- 4311/700: +3; zero in the first 20 seconds and three after 20 seconds.

Event-onset differences need not equal vehicle-step differences because an
onset can persist for multiple steps and paired branches contain different
vehicle histories after divergence.

## Literature implication

The result is consistent with moving-bottleneck research that treats the
actuator as a closed-loop traffic-flow controller rather than a fixed-duration
pulse:

- Vishnoi et al. model full-lane CAV platoons as rolling roadblocks and compare
  LQR, PI, and MPC control. Their microsimulation shows that control horizon
  and input variation materially affect whether the controller creates new
  upstream jams; penalizing control variation makes changes more gradual, but
  the traffic-state horizon remains central. See
  [CAV Traffic Control to Mitigate the Impact of Congestion from Bottlenecks](https://arxiv.org/abs/2306.10418).
- Cicic et al. compute platoon speed and configuration from predicted traffic
  and platoon state immediately upstream of the bottleneck, dynamically
  changing the controlled area and which platoons act according to predicted
  congestion. See
  [Platoon-actuated variable area mainstream traffic control](https://doi.org/10.1016/j.ejcon.2022.100687).
- Piacentini et al. formulate moving-bottleneck speed control using a
  capacity-drop-aware traffic model and model-predictive control rather than a
  fixed open-loop duration. See
  [Highway Traffic Control with Moving Bottlenecks of CAVs](https://doi.org/10.1109/CDC40024.2019.9029994).

These papers support feedback based on bottleneck state, predicted congestion,
and control horizon. They do not prove that any particular feedback law will
work in the present SUMO/IDM scenario.

## Scientific decision

Two defensible paths remain:

1. **Selective escalation (recommended for the present paper).** Preserve the
   wall as a candidate action, repair only its spatial fail-safe behavior, and
   define the learned conditional gate as joint safety-and-benefit selection.
   The gate must learn to abstain on the delayed harmful trajectories. This
   changes the old requirement that 90% of *all raw activations* be safe and
   therefore requires a separately frozen estimand protocol.
2. **New closed-loop moving-bottleneck controller.** Replace the fixed 20-second
   action with a traffic-state feedback/MPC treatment that regulates downstream
   density or discharge over time. This is a larger actuator research project
   and must be verified before any predictor work.

A taper-only v5.3 should not be run. The existing feasibility-versus-benefit
separation remains correct, but the benefit gate must include the full-horizon
safety constraint if Path 1 is approved.

## Reproducibility

- Analyzer: `tools/analyze_idm_wall_v5_2_release_safety.py`
- Tests: `tests/test_analyze_idm_wall_v5_2_release_safety.py`
- Machine-readable result:
  `runs/eval/idm_wall_v5_2_prospective_p30_s4301_4312/release_safety_diagnosis.json`
