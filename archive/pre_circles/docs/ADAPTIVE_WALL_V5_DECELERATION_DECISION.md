# Adaptive compact wall v5: holding-deceleration decision

Status: **approved 2026-08-26; v5.1 implementation under verification**.

## Question

Should the full-cross-lane adaptive compact wall retain the current
0.3 m/s2 deceleration cap during HOLDING, or may it use the already configured
1.0 m/s2 assembly cap throughout the temporary 20 s treatment?

This is a treatment-intensity decision, not a predictor hyperparameter. It
must be frozen before a fresh treatment-dose/headroom cohort is generated.

## Invariants that do not change

- Control is frozen decentralized damping.
- Treatment is the identical damping controller plus a temporary coordinated
  moving-wall overlay.
- Only latched in-band wall CAVs receive member-specific speed/tau overrides.
- Out-of-band CAVs retain bitwise-identical damping actions.
- Release returns wall members to damping; it never releases them to
  uncontrolled SUMO.
- Feasibility requires every occupied pre-merge lane to be covered.
- Primary compact geometry is a 50 m maximum cross-lane longitudinal span.
- The common target is latched from the predecision state and is limited to a
  20% reference-speed reduction.
- Front/rear TTC target projection, influence-zone monitoring, collision,
  teleport and hard-brake aborts remain active.

## Exact development evidence

Both artifacts pass damping replay identity, predecision identity, treatment
repeat identity, in-band-only override, out-of-band damping identity and
release-to-damping:

- runs/probe/counterfactual_adaptive_compact_v5_pd_kd05_exact_s504_step180.json
- runs/probe/counterfactual_adaptive_compact_v5_pd_kd05_exact_s506_step180.json

These seeds are development cases and must remain excluded from prospective
evaluation.

### Seed 504: effect headroom, but loss of compactness and safety

- Realized dose: 20/20 s; lane coverage: 100%; no abort.
- Compact fraction: exactly 80%; maximum span: 82.82 m.
- The span contracted from 30.57 m to 0.72 m and then expanded to 82.82 m.
- The traffic-constrained member decelerated by an average 0.798 m/s2 during
  steps 183--200.
- The freer member was capped at exactly 0.300 m/s2.
- Their speed gap grew from 1.89 to 10.36 m/s.
- The three TTC projections clipped the already traffic-constrained member's
  upper target because of its leader. They did not prevent additional gentle
  deceleration of the freer member in the observed trajectory.
- Treatment versus damping: delay -337.55 s, queue area -494 vehicle-s and
  discharge +7 vehicles, but hard-brake exposure increased from 6 to 13
  vehicle-steps.

The proportional-derivative servo cannot remove this failure while its
actuator is saturated at 0.3 m/s2. The observed constrained-lane trajectory
requires approximately 0.8 m/s2 merely to stop the cross-lane speed gap from
growing.

### Seed 506: excellent geometry, but no useful effect

- Realized dose: 20/20 s; lane coverage: 100%; no abort.
- Compact fraction: 100%; maximum span: 14.72 m.
- Hard-brake exposure remains 10 vehicle-steps in both arms.
- Low-TTC exposure remains one step in both arms.
- Treatment versus damping: delay +8.12 s, queue area -11 vehicle-s and
  discharge unchanged.
- Neither arm develops sustained congestion.

The current low-authority treatment works in a benign state but has no
meaningful outcome to improve.

## Interpretation

The two exact cases expose a treatment-overlap problem:

1. where lane trajectories are naturally coherent, 0.3 m/s2 preserves the
   wall but the action can be causally negligible;
2. where the wall has service headroom, asymmetric leaders can demand more
   than 0.3 m/s2 to preserve full-cross-lane compactness.

Training a predictor now would mostly learn this feasibility boundary, repeat
the failure of the earlier dataset, and confound formation feasibility with
treatment benefit.

Published moving-actuator and coordinated-platoon work supports treating CAV
speed/formation as an explicit control input and using it to postpone or shape
bottleneck inflow. It does **not** by itself validate this project's numeric
1.0 m/s2 choice. Here, that value is justified as a predeclared bounded
actuator envelope above the empirically required 0.8 m/s2, and must be checked
prospectively for safety:

- Du et al., Adaptive control with moving actuators at motorway bottlenecks
  with connected and automated vehicles, TR-C 156 (2023), 104319,
  https://doi.org/10.1016/j.trc.2023.104319
- Cicic et al., Coordinating Vehicle Platoons for Highway Bottleneck
  Decongestion and Throughput Improvement, IEEE T-ITS 23(7), 8959--8971,
  https://doi.org/10.1109/TITS.2021.3088775
- Piacentini et al., Traffic control via moving bottleneck of coordinated
  vehicles, IFAC-PapersOnLine 51(9), 13--18,
  https://doi.org/10.1016/j.ifacol.2018.07.003

## Alternatives

### A. Keep 0.3 m/s2 during holding

Declare seed-504-like states infeasible using predecision leader constraints.
This preserves the lightest actuation envelope, but current evidence suggests
that it may remove precisely the states with meaningful service/collapse
headroom. A predictor could still estimate feasibility, but a publishable
benefit predictor is unlikely without outcome diversity among feasible cases.

### B. Permit up to 1.0 m/s2 throughout adaptive treatment (approved)

Add a separately named and frozen compact_servo_a_max_dec_mps2=1.0 parameter.
Do not overload the assembly parameter. The actuator may use *up to* this
bound; member targets remain subject to TTC projection, and the treatment
aborts on unsafe influence-zone behavior.

The user approved this option before implementation. The headroom artifact
schema advances to version 6 and the exact-twin schema advances to version 2
so pre-v5.1 results cannot be mistaken for evidence using the revised
treatment.

The first verification is a development replay, not paper evidence:

1. focused tests and full repository tests;
2. exact twin seed 504;
3. exact twin seed 506;
4. only if both preserve the causal contract, a bounded replay of already
   inspected development seeds.

The change is rejected before a fresh pilot if it fails any of:

- 20/20 s requested dose in feasible cases;
- at least 80% lane coverage;
- at least 80% of treatment samples within 50 m;
- no collision or teleport increment;
- no treatment-attributable unsafe TTC/hard-brake pattern;
- identical prefix and out-of-band damping;
- treatment-repeat identity.

If development checks pass, the parameter is frozen and evaluated on a new,
excluded 12--20-seed cohort. It is not tuned again on that cohort.

## Predictor implication

No benefit predictor is trained until a fresh pilot establishes:

- adequate safe treatment dose;
- effects above replay/numerical noise;
- beneficial completed outcomes on at least three independent seeds;
- nonbeneficial completed outcomes on at least three independent seeds;
- collapse prevention and nonprevention diversity if collapse prevention
  remains the primary claim.

If dose passes but collapse diversity remains absent, the paper must pivot to
a prospectively verified safety/service or recovery estimand rather than
relabelling the same composite outcome.
