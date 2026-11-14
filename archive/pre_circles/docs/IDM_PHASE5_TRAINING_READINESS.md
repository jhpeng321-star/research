# IDM Phase-5 training-readiness gate

Status: **training blocked until every mandatory gate below passes**.

This gate is frozen before the fixed-action reward-ranking experiment. Seeds 4101–4106 remain development-only because they were used to construct the scenario reward references. Seeds 4107–4112 are also development-only after this audit and must not be reused for prospective paper evaluation.

## Frozen controller and scenarios

- Baseline: the verified analytical IDM damping controller with an exactly zero residual.
- Candidate probes: constant executed residual requests of -0.10, +0.10, and +0.20 m/s², all passed through the same absorber gate, safety projection, and 5 s residual filter used for training.
- Scenarios: IDM mainline demand 2200 and 2700 veh/h, ramp demand 400 veh/h, paired seeds 4107–4112.
- Reward references: `runs/eval/idm_phase5_reward_refs_v1.json`; these must not be recomputed from audit outcomes.

## Mandatory reward-accounting gates

1. The scalar reward equals the sum of local, efficiency, delay-budget, corridor-TTC, wave, and residual-effort components at every selected agent-step (unit-test tolerance).
2. Nonselected CAVs receive exactly zero IDM residual-learning reward.
3. Every evaluation case reports cumulative and per-selected-agent-step reward components; missing diagnostics are a failure.

## Mandatory action-ranking gates

1. A physically beneficial candidate must not rank below zero residual because of the residual-effort term alone.
2. Across all paired candidate/zero cases, reward improvement must have positive rank association with a frozen physical utility: mean of wave reduction and severe-braking reduction, minus 0.05 per second/vehicle of positive delay, 0.25 per lost arrival beyond one, and 1.0 per added collision. Spearman rho must be at least 0.50; ties are average-ranked.
3. Mean reward improvement for +0.10 m/s² must be positive in at least one demand where it provides positive mean physical utility.
4. Reward must preserve action diversity: it may prefer zero in low-headroom cases and a nonzero probe in high-headroom cases. A single arm winning every paired case is a failure unless its physical utility also wins every paired case.
5. No single reward component may contribute more than 80% of the mean absolute component magnitude across all selected agent-steps without a documented scale justification.

## Mandatory physical and execution gates

1. No added collisions relative to paired zero-residual controls.
2. No material discharge loss: arrived-vehicle delta is at least -1 per case.
3. At 2200 veh/h, delay increase is at most 1.0 s/vehicle per case for any candidate considered trainable.
4. At 2700 veh/h, the audit must expose measurable safety or wave headroom rather than only numerical noise: at least two independent seeds with >=10% improvement in wave energy or severe-braking events for some admissible probe.
5. Requested and executed residual diagnostics must be nonzero for nonzero arms and respect the configured bounds/filter.

## RL interface gates after any reward correction

1. Observation shape and seven controller-context features match the frozen model contract.
2. Preview/step identity and analytical zero-residual replay identity pass.
3. A 512–2048-step RLlib smoke completes without worker restart or NaN/Inf values.
4. At least one learner iteration has nonzero policy loss or KL, selected action samples are present, and the executed residual distribution is not identically zero.
5. Focused tests and `python -m pytest -q` pass.

Only after all mandatory gates pass may a 100k-step training run begin. Passing this document authorizes training readiness, not a paper claim; prospective independent seeds are still required for the trained policy.