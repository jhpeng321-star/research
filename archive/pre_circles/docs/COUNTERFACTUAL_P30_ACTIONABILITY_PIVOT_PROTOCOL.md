# Hyper/p30 actionability pivot protocol

Declared on 2026-08-27 after the hyper/p20 v5.3 treatment and v5.4 supervised prospective pilots failed. This is a bounded treatment-development diagnostic, not predictor verification.

## Question

Does increasing CAV penetration from 20% to 30% make the unchanged 30% adaptive compact moving-wall action sufficiently feasible, consistently dosed, and outcome-diverse to justify a fresh prospective p30 headroom cohort?

The causal arms, treatment duration, geometry, outcome window, safety/service oracle, and damping checkpoint are unchanged. No predecision supervisor is used. Seeds 701--708 have already been inspected and permanently excluded from all model and locked evaluation roles.

## Development interpretation

The diagnostic is promising only if it materially improves on hyper/p20 by showing at least 80% qualified dose among feasible activations, at least 80% global safety-admissibility as a development screen, and beneficial completed treatments across at least three seeds alongside nonbeneficial outcomes. The formal prospective p30 pilot will retain the stricter 90% safety threshold.

## Exact command

```powershell
python 'D:\research-dense\experiments/counterfactual/generate_counterfactual_escalation_headroom.py' --seeds 701 702 703 704 705 706 707 708 --decision-steps 120 180 240 --outcome-steps 180 --treatment-steps 20 --cav-penetration 0.3 --demand hyper --workers 2 --development --dataset-role treatment_development --primary-estimand safety_service --predecision-supervisor none --wall-speed 10.0 --wall-d-lo 200.0 --wall-d-hi 1000.0 --wall-tau 3.0 --wall-a-max-dec 0.3 --tau-ramp-step 0.05 --geometry-mode compact_moving --max-longitudinal-span-m 50.0 --compact-release-boundary-m 50.0 --compact-selection-mode trajectory_coherent --compact-assembly-duration-s 10.0 --compact-assembly-speed-tolerance-mps 2.0 --compact-assembly-a-max-dec-mps2 1.0 --compact-control-mode adaptive_servo --compact-servo-a-max-dec-mps2 1.0 --compact-max-activation-speed-spread-mps 4.0 --compact-max-speed-reduction-fraction 0.3 --compact-shock-gap-lo-mps 5.0 --compact-shock-gap-hi-mps 15.0 --compact-position-kp-s-inv 0.05 --compact-position-kd 0.5 --compact-position-speed-offset-cap-mps 1.0 --compact-safety-min-front-ttc-s 3.0 --compact-safety-min-rear-ttc-s 4.0 --compact-influence-upstream-m 100.0 --compact-influence-max-followers 3 --compact-influence-min-ttc-s 1.5 --output-dir 'D:\research-dense\runs\probe\counterfactual_v5_5_p30_actionability_development_s701_708'
```

If dose or safety remains poor, do not add seeds; redesign the wall or pivot the paper claim. If dose, safety, and action diversity improve, freeze a new 12--20-seed p30 prospective cohort before model-scale generation.
