# IDM residual damping Phase-4 protocol

Status: training interface verified; no trained checkpoint has been selected or frozen yet.

## Frozen foundation

- Analytical controller: selected-absorber IDM damping with `reference_mode=disturbance_p30`.
- Evidence authorizing learning: `runs/eval/idm_disturbance_reference_probe_v1/summary.json`, `phase3_pass=true` and exact analytical/zero-residual replay identity on seeds 4101–4106.
- Residual: one scalar acceleration in `[-0.5, +0.3] m/s^2`.
- Physical command: `safety_project(a_analytical + absorber_gate * residual)`.
- Deterministic absorber selection, hold/handoff, follower-aware projection, and smooth release remain outside the learned policy.
- Actor inputs add `wave_risk`, `wave_forming`, and `idm_absorber_gate` after the fixed physical observation.
- The centralized critic uses at most 48 CAV slots.

## Learning contract

Only a deterministically selected absorber can affect traffic. Nonselected actions receive zero reward and zero PPO advantage. Active advantages are centered before RLlib batch standardization so inactive zeros remain zero. The critic may still learn state value from all traffic observations.

RLlib action normalization is disabled because asymmetric bounds would map normalized zero to `-0.1 m/s^2`. Actions are clipped directly in physical units. The continuous head begins with exact mean zero and state-independent `log_std=-2` (`std=0.1353`). Campaign entropy regularization is zero so the much larger inactive sample population cannot inflate exploration.

Reward priority:

1. local collision/TTC/severe-braking exposure;
2. local wave risk and acceleration/jerk stability;
3. scenario-matched service-budget and corridor TTC penalties;
4. scenario-matched discharge benefit;
5. sparse residual effort.

Corridor terms are divided across simultaneous absorbers so reward scale does not grow with lane count. Per-demand references come from medians of no-op IDM development seeds 4101–4106 in `runs/eval/idm_phase4_reward_refs_v1.json`. These seeds are excluded from validation.

## Verified smoke evidence

- Complete repository suite before the final actor mask: 523 passed.
- Real SUMO contract: `runs/eval/idm_phase4_env_smoke_v1.json` passed 240/240 steps, 104 controlled CAV-seconds, zero preview/action mismatches, zero nonselected reward leakage, exact zero residual, and 24 maximum active CAVs (<48 critic slots).
- RLlib unmasked corrected smoke: `runs/rllib/idm_phase4_smoke_v2`; three completed 240-step episodes, finite metrics, KL about 0.005, checkpoint save and restore passed.
- RLlib selected-advantage smoke: `runs/rllib/idm_phase4_smoke_v3_masked`; one completed 240-step episode, no worker restart, finite loss/gradient/value statistics, checkpoint saved, actor mask enabled.

## Serious development campaign

Training cells are the explicit IDM configs for 1900, 2000, 2100, and 2200 veh/h mainline demand with 400 veh/h ramp demand. CAV penetration is sampled only from 0.10 and 0.20. Episode seed and Bernoulli CAV placement vary independently. Development seeds and campaign checkpoints are not final prospective evidence.

```powershell
python train.py `
  --baseline shepherd `
  --control-mode idm_residual_smoothing `
  --density dense `
  --sumo-root D:\research-dense\scenarios\idm_transition `
  --sumo-config osm.dense.idm_m2200r400.sumocfg `
  --sumo-config-choices `
    osm.dense.idm_m1900r400.sumocfg `
    osm.dense.idm_m2000r400.sumocfg `
    osm.dense.idm_m2100r400.sumocfg `
    osm.dense.idm_m2200r400.sumocfg `
  --scenario-refs D:\research-dense\runs\eval\idm_phase4_reward_refs_v1.json `
  --cav-penetration 0.2 `
  --penetration-choices 0.1 0.2 `
  --max-episode-steps 1200 `
  --num-env-runners 8 `
  --total-timesteps 1000000 `
  --train-batch-size 6000 `
  --sgd-minibatch-size 128 `
  --num-sgd-iter 2 `
  --max-cavs-critic 48 `
  --lr 0.0001 `
  --entropy-coeff 0.0 `
  --save-every 5 `
  --seed 42 `
  --logdir D:\research-dense\runs\rllib\idm_residual_phase4_v1
```

## Stop conditions

Stop and diagnose rather than extending training if any of these occurs:

- NaN/Inf reward, value, loss, KL, entropy, or gradient;
- worker restarts or exploration gridlock;
- KL remains above 0.03 for two iterations;
- critic capacity is reached (`active_cavs >= 48`);
- residual mean drifts toward a bound without reduced safety/wave cost;
- no checkpoint outperforms exact zero residual on development evaluation after the first scheduled evaluation block.

## Checkpoint selection gate

No checkpoint becomes the paper control arm until untouched validation seeds 4201–4240 show repeatable severe-braking or wave reduction, zero collisions, service-budget compliance, reasonable cross-seed consistency, and near-zero residual in healthy traffic. Then record checkpoint hash, code/diff identity, schema, bounds, scenario parameters, and seed partitions; finally run damping-versus-damping replay identity. Only after that is the moving-wall overlay rebased.