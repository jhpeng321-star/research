# Counterfactual escalation v5.3 prospective pilot protocol

Frozen before inspecting seeds 701--720 on 2026-08-27.

## Scientific treatment

- Control: the frozen decentralized damping checkpoint.
- Treatment: the identical damping controller plus one temporary compact moving wall.
- Action space: binary. The only wall action uses an adaptive terminal target capped at a 30% reduction from the observed common predecision speed.
- Geometry: trajectory-coherent, full occupied-lane coverage, 50 m maximum longitudinal span, adaptive servo assemble--hold--release.
- Release: every wall member returns to the identical damping controller. Out-of-band CAVs always retain their damping actions.

The primary estimand is the conditional safety/service effect of commanding this fixed wall instead of continuing damping. Collapse prevention remains a reported secondary outcome. The global full-horizon safety oracle and its 90% admissibility threshold are unchanged; direct wall-member attribution is an additional diagnostic and does not erase downstream causal effects.

## Frozen cohort and exclusions

The prospective pilot uses seeds 701--716 at hyper demand and 20% CAV penetration. The whole 701--720 block is permanently excluded from model development, validation, and locked testing. Earlier inspected protocol-development blocks, including 501--512 and 601--620, remain excluded.

Each seed is evaluated at decision steps 120, 180, and 240. The planned treatment is 20 controller steps (10 s), followed by a 180-step (90 s) outcome window.

## Pass/fail criteria

The pilot passes treatment readiness only if all of the following hold:

1. At least one feasible activation occurs.
2. At least 80% of feasible activations receive at least 80% of planned duration and at least 80% lane coverage; compact treatments must complete assembly, spend at least half their override in hold, and remain within the 50 m span for at least 80% of treatment.
3. No excess collision or teleport occurs.
4. At least 90% of completed treatments satisfy the unchanged global safety oracle.
5. Beneficial completed treatments occur in at least three independent seeds.
6. Nonbeneficial completed treatments occur in at least three independent seeds.
7. Completed treatments span at least six independent seeds.
8. Meaningful effects exceed deterministic replay/numerical noise.

Collapse-prevention and nonprevention seed counts are always reported, but they are not mandatory for this safety/service-primary pilot. If a later paper makes collapse prevention the primary target, the mandatory diversity gate reverts to at least three prevention seeds, at least three nonprevention seeds, and at least a 5% prevention-case fraction.

## Exact command

```powershell
python 'D:\research-dense\experiments/counterfactual/generate_counterfactual_escalation_headroom.py' --seeds 701 702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 --decision-steps 120 180 240 --outcome-steps 180 --treatment-steps 20 --cav-penetration 0.2 --demand hyper --workers 2 --dataset-role headroom_gate --primary-estimand safety_service --wall-speed 10.0 --wall-d-lo 200.0 --wall-d-hi 1000.0 --wall-tau 3.0 --wall-a-max-dec 0.3 --tau-ramp-step 0.05 --geometry-mode compact_moving --max-longitudinal-span-m 50.0 --compact-release-boundary-m 50.0 --compact-selection-mode trajectory_coherent --compact-assembly-duration-s 10.0 --compact-assembly-speed-tolerance-mps 2.0 --compact-assembly-a-max-dec-mps2 1.0 --compact-control-mode adaptive_servo --compact-servo-a-max-dec-mps2 1.0 --compact-max-activation-speed-spread-mps 4.0 --compact-max-speed-reduction-fraction 0.3 --compact-shock-gap-lo-mps 5.0 --compact-shock-gap-hi-mps 15.0 --compact-position-kp-s-inv 0.05 --compact-position-kd 0.5 --compact-position-speed-offset-cap-mps 1.0 --compact-safety-min-front-ttc-s 3.0 --compact-safety-min-rear-ttc-s 4.0 --compact-influence-upstream-m 100.0 --compact-influence-max-followers 3 --compact-influence-min-ttc-s 1.5 --output-dir 'D:\research-dense\runs\probe\counterfactual_v5_3_prospective_headroom_safety_service'
```

Expected wall time is approximately 25--40 minutes with two workers. A failed dose gate triggers geometry/membership/handoff diagnosis. Passed dose with no outcome diversity triggers treatment redesign, not more seeds. Predictor-scale generation remains prohibited until this pilot passes.
