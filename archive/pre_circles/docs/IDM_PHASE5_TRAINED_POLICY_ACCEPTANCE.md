# IDM Phase-5 trained-policy acceptance protocol

Status: frozen before the v3 RLlib smoke and full training.

## Claim standard

The learned residual is not required to improve every metastable realization. Acceptance means **bounded-risk aggregate improvement over the frozen analytical zero-residual controller**, with hard safety invariants and an explicitly limited harmful tail. This is consistent with the earlier Krauss controller: useful damping can have occasional adverse instances without making a universal-benefit claim.

## Data partitions

- Reward/reference construction: seeds 4101–4106 (excluded).
- Controller/reward development: seeds 4107–4120 (excluded from independent claims).
- First independent acceptance set: seeds 4201–4220, paired zero residual versus frozen trained checkpoint.
- Reserved confirmation set: seeds 4221–4240; untouched until checkpoint and acceptance calculations are frozen.
- Demands: 2200 and 2700 veh/h mainline plus 400 veh/h ramp.

## Metrics

For each paired run:

- `wave_reduction = (E_zero - E_policy) / E_zero`, where E is integrated corridor acceleration-squared energy;
- `braking_reduction = (B_zero - B_policy) / B_zero`, where B is the count of acceleration events below -3.0 m/s²;
- safety composite `S = 0.5 * (wave_reduction + braking_reduction)`;
- service deltas: mean time loss, arrived vehicles, and discharge;
- collisions and safety-shield diagnostics;
- executed residual distribution and activation windows.

## Mandatory gates

### A. Hard safety protection

1. Zero collisions on every independent policy run and no collision increase relative to paired zero residual.
2. Every executed command stays inside the residual and acceleration bounds.
3. No executed command bypasses the front-TTC 3.0 s or rear-TTC 4.0 s safety projection; rejected/projected requests are logged rather than counted as unsafe execution.
4. No NaN/Inf action, observation, reward, or controller diagnostic.

### B. Frozen service bounds

1. Mean paired delay regression is at most +2.0 s/vehicle at each demand.
2. Per-seed paired delay regression is at most +5.0 s/vehicle.
3. Paired arrived-vehicle loss is no worse than one vehicle per seed, and mean arrived delta is nonnegative within numerical tolerance.
4. A safety improvement is not counted as a win when any hard service bound is breached.

### C. Observation sufficiency

Before full training, zero-residual selected-absorber observations are collected from the development seeds. A cell is labeled helpful when the best audited nonzero residual has safety composite `S > +0.02`, harmful when all audited nonzero residuals have `S <= 0`, and ambiguous cells are excluded.

A leave-one-seed-out classifier using only policy-visible features (no demand, seed, arm, future metric, or filename) must distinguish helpful from harmful labels with:

- pooled AUROC >= 0.70;
- balanced accuracy >= 0.65;
- both classes represented in at least three independent development seeds.

This is a diagnostic of available information, not evidence that the final policy is calibrated.

### D. Reward alignment

The frozen endpoint-aligned v3 reward audit must retain:

- reward/physical-utility Spearman rho >= 0.50;
- >=80% correct nonpositive reward direction for 2700 cases worsening both safety endpoints;
- positive mean reward for the physically helpful +0.10 arm at 2200;
- nonpositive mean reward for every harmful fixed arm at 2700.

### E. Independent aggregate benefit

At 2200:

1. Mean safety composite `S` is positive.
2. Either mean severe-braking reduction is >=5% with wave reduction no worse than -2%, or mean wave reduction is >=5% with severe-braking reduction no worse than -2%.
3. At least 60% of independent seeds have `S > 0` without a service-bound breach.

At 2700, where fixed residuals have no incremental headroom:

1. Mean safety composite is no worse than -2%.
2. Mean absolute executed residual is lower than at 2200, demonstrating preservation of analytical damping rather than indiscriminate intervention.
3. All hard safety and service bounds still apply.

### F. Rare and bounded harmful activations

An activation is `|executed residual| >= 0.03 m/s²` for at least three consecutive controller steps. Each activation is compared with the same-seed zero-residual 10 s time window. It is labeled harmful only when both severe-braking count and acceleration-energy sum worsen by more than 10%; near-zero baseline windows require a nonzero absolute increase and are not classified from ratios alone.

1. Harmful activation-window rate is at most 25%.
2. No harmful activation window contains a collision or hard service-bound breach.
3. At the paired-run level, no more than 25% of 2200 seeds may have `S < -0.05`, and none may have `S < -0.20`.
4. At 2700, nonzero activation rate must be lower than at 2200; occasional activations are allowed, but systematic residual intervention is a failure.

## Interpretation

Passing supports: “the trained residual controller improves safety/stability on average near metastable onset while hard safety constraints and frozen service/tail bounds limit adverse instances; under structural overload it learns to preserve analytical damping.”

It does not support perfect intervention, universal per-instance benefit, calibrated counterfactual probabilities, or successful moving-wall escalation. Those require separate evidence.