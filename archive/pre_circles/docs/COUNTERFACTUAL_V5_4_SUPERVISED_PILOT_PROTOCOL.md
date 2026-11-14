# Counterfactual escalation v5.4 supervised prospective pilot

Frozen on 2026-08-27 after diagnosing the failed, permanently excluded v5.3 cohort (seeds 701--716) and before inspecting seeds 801--820.

## Unchanged treatment and estimand

The causal arms remain frozen decentralized damping versus identical damping plus the same 30% adaptive compact moving wall. Geometry, assembly, hold, release, safety projections, 10 s planned dose, 90 s outcome window, and the global safety/service oracle are unchanged. Collapse prevention remains secondary.

## Versioned predecision safety envelope

The wall command is admitted only when all three observable conditions hold:

1. `local_bin_100_200m_cav_count >= 2` (local CAV support),
2. `traffic_lag_10_density_scaled <= 0.80` (not deeply saturated),
3. `formation_soft_cav_speed_std_mps <= 7.0` (speed-field coherence).

The profile name is `local_support_density_coherence_v1`. These conditions use only information available before the decision. All paired potential outcomes may still be replayed for scientific audit, but treatment readiness is evaluated only among commands admitted prospectively by this profile. Seeds 701--716 are development evidence only and can never be used to claim supervisor performance.

## Fresh cohort and unchanged pass criteria

Seeds 801--816 are the prospective cohort; the entire 801--820 block is permanently excluded from modeling and locked evaluation. Decision steps remain 120, 180, and 240.

The pilot passes only if admitted feasible commands satisfy the v5.3 gates: at least 80% qualified dose, zero excess collisions/teleports, at least 90% global safety-admissibility, beneficial completed treatments in at least three independent seeds, nonbeneficial completed treatments in at least three independent seeds, and at least six completed-treatment seeds. Collapse results remain reported but are not mandatory.

## Exact command

```powershell
python 'D:\research-dense\experiments/counterfactual/generate_counterfactual_escalation_headroom.py' --seeds 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 --decision-steps 120 180 240 --outcome-steps 180 --treatment-steps 20 --cav-penetration 0.2 --demand hyper --workers 2 --dataset-role headroom_gate --primary-estimand safety_service --predecision-supervisor local_support_density_coherence_v1 --wall-speed 10.0 --wall-d-lo 200.0 --wall-d-hi 1000.0 --wall-tau 3.0 --wall-a-max-dec 0.3 --tau-ramp-step 0.05 --geometry-mode compact_moving --max-longitudinal-span-m 50.0 --compact-release-boundary-m 50.0 --compact-selection-mode trajectory_coherent --compact-assembly-duration-s 10.0 --compact-assembly-speed-tolerance-mps 2.0 --compact-assembly-a-max-dec-mps2 1.0 --compact-control-mode adaptive_servo --compact-servo-a-max-dec-mps2 1.0 --compact-max-activation-speed-spread-mps 4.0 --compact-max-speed-reduction-fraction 0.3 --compact-shock-gap-lo-mps 5.0 --compact-shock-gap-hi-mps 15.0 --compact-position-kp-s-inv 0.05 --compact-position-kd 0.5 --compact-position-speed-offset-cap-mps 1.0 --compact-safety-min-front-ttc-s 3.0 --compact-safety-min-rear-ttc-s 4.0 --compact-influence-upstream-m 100.0 --compact-influence-max-followers 3 --compact-influence-min-ttc-s 1.5 --output-dir 'D:\research-dense\runs\probe\counterfactual_v5_4_supervised_prospective_headroom'
```

No predictor may be trained if this supervised gate fails. A pass permits model-development data generation; it does not itself verify a learned predictor.
