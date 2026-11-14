# IDM Phase-5 safety-first training-readiness gate

Status: **training blocked until every mandatory gate below passes**.

This is a prospective v2 protocol frozen after the v1 audit failed. The v1 failure remains recorded in `IDM_PHASE5_TRAINING_READINESS.md` and `runs/eval/idm_phase5_reward_action_audit_v1/phase5_readiness.json`. The observed reasons were (a) an effort tax reversing beneficial corrections, (b) delay-driven reward wins despite worse hard braking and corridor oscillation, and (c) no incremental fixed-residual headroom beyond analytical damping at structural overload (2700 veh/h).

## Scientific treatment definition

The learned controller is a bounded contextual residual around the frozen analytical IDM damping controller. Its intended behavior is:

- at 2200 veh/h metastable onset, learn when a small residual can improve corridor wave energy and severe braking;
- at 2700 veh/h structural overload, learn to preserve the analytical controller with a near-zero residual when corrections are harmful;
- escalation to coordinated moving-wall control remains downstream and is not part of this training gate.

The analytical reference, absorber selection, residual bounds [-0.5, +0.3] m/s², 5 s low-pass filter, and front/rear safety projection remain unchanged.

## Frozen reward definition

- Corridor severe braking: fraction of corridor vehicles below -3.0 m/s², normalized by 0.05; weight 1.0.
- Corridor wave energy: mean squared corridor acceleration, normalized by 4.0 m²/s⁴; weight 1.0.
- Corridor TTC weight: 1.0.
- Local safety/stability weights: 2.0 / 0.5.
- Delay-budget and efficiency weights: 0.10 / 0.10.
- Local onset-risk reward weight: 0.0; it remains an observation and absorber-gate signal.
- Residual-effort weight: 0.0; bounds, filtering, gating, and the safety projection constrain the actuator.

## Confirmation audit

- Paired seeds: 4107–4112, permanently development-only after this audit.
- Demands: 2200 and 2700 veh/h mainline, 400 veh/h ramp.
- Arms: zero, -0.10, +0.10, and +0.20 m/s² requests through the identical filter/gate/shield.
- Additional locked discrimination evidence: 2700 seeds 4108 and 4112, zero versus +0.05 m/s².

## Mandatory gates

1. All 48 confirmation cases and all reward-component diagnostics are present; maximum accounting error <= 1e-12.
2. No added collisions and no paired arrival loss worse than one vehicle.
3. At 2200, +0.10 has positive mean safety-first physical utility and positive mean reward delta; at least three seeds have positive physical utility.
4. At 2700, zero is the preservation target: mean physical utility of each fixed nonzero arm is nonpositive, and its mean reward delta must also be nonpositive.
5. For 2700 paired cases that worsen both corridor wave energy and severe braking, at least 80% must have nonpositive reward delta; no arm may obtain a positive mean reward solely from delay improvement.
6. Across all paired cases, Spearman correlation between reward delta and safety-first physical utility is >=0.50. Utility is the mean of proportional wave and severe-braking reductions, minus 0.01 per positive second/vehicle of delay, 0.25 per arrival lost beyond one, and 1.0 per added collision.
7. No reward component contributes more than 80% of aggregate absolute magnitude without a documented scale justification.
8. Action diversity is present: the reward selects at least two different winning arms across paired demand/seed cells.
9. Locked discrimination cases pass: seed 4108 assigns negative reward to +0.05 when both safety metrics worsen; seed 4112 assigns positive reward when both improve.
10. Focused tests and the complete repository suite pass.
11. A 512–2048-step RLlib smoke completes without restart or NaN/Inf and shows nonzero policy loss or KL plus nonzero selected actions.

Only after these gates pass may the balanced 2200/2700 Phase-5 training begin. Passing authorizes training readiness only; a trained checkpoint still requires prospective evaluation on untouched validation seeds.