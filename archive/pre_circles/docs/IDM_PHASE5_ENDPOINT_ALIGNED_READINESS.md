# IDM Phase-5 endpoint-aligned training-readiness gate (v3)

Status: **training blocked until all mandatory gates pass**.

The v2 safety-first audit is preserved at `runs/eval/idm_phase5_safety_reward_audit_v2/safety_readiness.json`. It passed every aggregate criterion but failed the predeclared harmful-case directionality gate: 5/9 cases that worsened both endpoint wave energy and severe-braking counts received nonpositive reward. Diagnosis showed that current-vehicle normalization can improve while raw integrated event/energy totals worsen when corridor vehicle-time changes.

## Frozen correction

The treatment/controller is unchanged. Only reward normalization changes:

- Severe-braking signal: raw number of corridor vehicles below -3.0 m/s² per controller step, divided by a fixed analytical-zero rate for the active demand.
- Wave signal: raw sum of squared corridor accelerations per controller step, divided by a fixed analytical-zero rate for the active demand.
- No clipping and no division by the treatment arm's current vehicle count.
- Demand-specific references are medians from excluded seeds 4101–4106:
  - 2200: 0.2093507902 severe events/s; 12.38990232 acceleration²-units/s.
  - 2700: 0.4463338745 severe events/s; 34.44865898 acceleration²-units/s.
- Weights remain: severe braking 1.0, corridor wave energy 1.0, TTC 1.0, delay 0.10, efficiency 0.10, local risk 0.0, residual effort 0.0.

All other treatment definitions and v2 mandatory gates remain unchanged. In particular:

1. The locked 2700 discrimination must assign negative reward to +0.05 on seed 4108 and positive reward on seed 4112.
2. The 48-case audit must be complete, safe, service-preserving, and exactly accounted.
3. +0.10 at 2200 must retain positive mean physical utility and reward with at least three positive-utility seeds.
4. All nonzero fixed arms at 2700 must have nonpositive mean physical utility and nonpositive mean reward.
5. At least 80% of 2700 cases worsening both endpoint wave and braking metrics must have nonpositive reward.
6. Reward/physical-utility Spearman rho must be >=0.50; no component may exceed 80% absolute magnitude; reward winners must be diverse.
7. Focused/full tests and the RLlib learning-signal smoke must pass before training.

Passing v3 authorizes only the balanced Phase-5 training run. It does not validate a checkpoint or the downstream escalation predictor.