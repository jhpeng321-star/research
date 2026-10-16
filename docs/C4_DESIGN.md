# C4: learned per-CAV control trained inside the world model

C4 is a shared per-CAV policy that outputs **absolute accelerations and lane decisions directly**
(no designed controller underneath) and is trained Dreamer-style: by differentiating imagined
WM-S rollouts ([ARCHITECTURE.md](ARCHITECTURE.md)) with respect to the CAV actions. The same
observation, action, safety and lane code runs in imagination and in SUMO. A MAPPO baseline with an
identical actor is trained directly in SUMO for comparison. Code: `src/wms/control/`; the shared
constants and signatures are in `src/wms/control/contract.py`. Paths are relative to the repository
root; where a design note and the code disagree, the code is authoritative.

Design principles enforced in code:

- **No privileged actor inputs.** No true driver types, penetration rate or global density. The
  critic may use the full imagined state during training only.
- **No imitation.** The policy starts from scratch. Designed controllers (CIRCLES damping, PI,
  wall) appear only as whole-episode evaluation baselines, never as actions, demonstrations or
  check candidates. The WM training data contain native driving, undirected random direct-output
  CAV commands and the policy's own rollouts only.
- **Penetration-agnostic.** One shared actor per CAV; 5 % and 40 % penetration differ only in the
  number of acting slots.

## 1. Timing and actuation

| constant | value | meaning |
|---|---|---|
| `CONTROL_DT_S` / `PHYSICS_DT_S` | 1.0 / 0.5 s | one decision per CAV per second, held for two physics steps |
| `ACCEL_MIN_MPS2` | -3.0 | actuator floor |
| actor `accel_max` | per checkpoint | stored in the actor's `accel_bounds` buffer; fresh default 2.6, set by `[policy] accel_max_mps2` (current v5/v6 configs: 1.2); checkpoints without the buffer decode with the legacy 1.5 |
| `DIRECT_ACCEL_MAX_MPS2` | 3.0 | ceiling of direct commands in `CavSumoEnv` (the actor's own bound binds first) |
| `SENSOR_RANGE_M` / `REPORT_RANGE_M` / `MAX_REPORTS` | 100 m / 400 m / 16 | local sensing and RSU relay |

## 2. Observation contract (`obs.py`, `rsu_field.py`)

`build_observations(state)` returns an `Observation` with:

- `own [B,K,13]`: `v, a`, lane one-hot x3, signed distance to the merge end, same-lane front and
  rear bumper gap / relative speed / presence within 100 m, in-merge-zone flag (`OWN_FEATURES`).
- `reports [B,K,16,13]` + `report_mask`: up to 16 other CAVs within 400 m, sorted by `|ds|`, each
  with relative position/speed/lane, its own front-gap sensing, receipt age, and its **previous
  commanded acceleration and lane intent** (`REPORT_FEATURES`; `obs.CommandMemory` in SUMO,
  bridge memory in imagination). Actors built with the older 11 report features are zero-padded
  on load.
- RSU own-sensor field `[B, N_cells, 6]` (`build_rsu_field`): 50 m cells over lanes 0/1 for
  `s in [-800, 225.44]` m and lane 2 over the whole ramp `[-378.66, 75.44]` m, with density, mean
  speed, speed std, queue flag, occupancy and CAV share (`RSU_FIELD_FEATURES`). Only vehicles
  inside the zones contribute.

Features keep physical units; network scaling is a single normalisation table without clipping.

## 3. Networks (`policy.py`)

- `CavPolicy`: own-state MLP + multi-head attention over reports + the 8-dim RSU group signal ->
  a tanh-Gaussian acceleration mapped to `[accel_min, accel_max]` (reparameterised, so
  gradients flow through the WM) and a masked categorical `stay / left / right` head
  (zero-initialised, i.e. uniform over feasible choices). `init_zero_accel` optionally centres
  the initial mean acceleration at 0.
- `RsuGroupEncoder`: self-attention over CAV reports plus the RSU field, conditioned per recipient
  on position and soft upstream/merge/downstream zone memberships -> `[B,K,8]` broadcast signal.
  This is the channel through which synchronised multi-CAV moves can be expressed.
- `TeamCritic`: privileged set transformer over all present vehicles -> team value; optionally
  also fed the per-entrance entry backlog (`trainer.critic_entry_backlog`).

## 4. Action execution (`lanes.py`, `safety.py`, `bridge.py`)

- **Lane feasibility** (`lane_feasibility`, identical in imagination and SUMO): stay is always
  allowed; mainline CAVs move only between lanes 0 and 1; ramp CAVs may merge left only inside the
  merge zone; target-lane front gap `>= gap_base_m + gap_time_s * v + closing_time_s * closing`
  and the analogous rear check; a minimum dwell between changes; no change that would straddle a
  SUMO edge boundary. Defaults in `LaneRules` (front 1.0 m + 1.2 s + 4.5 s, rear 3.0 m + 1.25 s +
  6.0 s, dwell 3 s, edge guard 10 m); simultaneous changes are re-checked against each other.
- **Safety layer** (`safe_accel`): a same-lane stopping-distance bound (action held for 1 s, then
  braking at 3 m/s^2, leader may brake immediately, 2 m reserve) applied as a soft-min; HDV
  proposals are untouched. It is conditional, not a collision guarantee.
- **Execution cap** (`trainer.cav_exec_cap`): in imagination a commanded CAV realises
  `min(safe command, its own IDM acceleration)`, as SUMO speed mode 27 does
  (`idm_normal:textbook` uses the fixed CAV vType settings).
- `bridge.make_cav_controller` packages all of this as the WM-S CAV callback
  (`state -> [B,K,4]`): decisions on even physics steps, safety every step, the forced ramp merge
  applied after the lane head and never blocked.

## 5. Reward and constraint (`reward.py`, `srcgrad.py`)

`team_reward(state, next_state, accel, RewardWeights)` per physics step, summed over the two
steps of a control step:

- `delay`: `dt * relu(1 - v_next / v_limit)` summed over the present cohort (including exits);
- `throughput`: soft count of crossings of the merge end;
- `smooth`: `sum a^2`; `hard_brake`: `sum relu(-a - 3)^2` (HDVs and CAVs);
- `fairness`: soft 90 % CVaR of per-vehicle delay;
- `source_wait`: vehicles waiting at an entrance (imagined entry backlog) cost `dt` each, weighted
  like delay and included in the normalisation and the CVaR. With `trainer.source_wait_grad` the
  value stays exact while the gradient comes from a differentiable insertion-blocking surrogate,
  so blocking an entrance is not free for the actor.

The weighted sum is divided by the vehicle count. **TTC is a constraint, not a reward term:**
`cost_ttc` (soft exposure below `ttc_limit_s` over closing same-lane pairs) enters the actor loss
through a Lagrange multiplier updated by projected dual ascent toward `trainer.ttc_budget`.
Default weights (`RewardWeights`): delay 1.0, throughput 0.5, smooth 0.05, hard brake 1.0,
fairness 1.0, `ttc_limit_s` 3.0.

## 6. Imagination training (`dreamer.py`, `train_cli.py`)

1. **Start states** (`start_states.py`, `c4_starts.py`): TRAIN-world windows of the WM config's
   C4 episode sources (validation worlds never start a rollout), sampled uniformly over
   penetration buckets then windows, optionally weighted by source (`[data] start_sources`,
   `start_source_weights`). Only the t0 state, the observed pre-t0 histories and entry features
   are passed; recorded futures are refused.
2. **Rollout.** Frozen WM-S members (`[world_model] checkpoints` or `--wm-ckpt`, an ensemble is
   allowed) are rolled out once over `horizon` **control** steps (horizon 30 = 30 s = 60 physics
   steps) with the policy driving CAVs, HDV driver settings from the WM encoder, and entrants from
   the observed entry history (`imagine_entries`). `imagination_grad_decay` overrides the WM's
   per-step gradient damping during imagination only. `checkpoint_imagination` recomputes pure
   per-step calls in backward to bound memory (same gradients).
3. **Returns.** TD(lambda) with a target-critic bootstrap; reward minus
   `uncertainty_weight * ensemble disagreement`; a percentile return scale. Optional sampled
   futures (`future_samples`, `src/wms/control/futures.py`) perturb driver settings within the
   encoder uncertainty and add residual-scaled acceleration noise, with common random numbers.
4. **Updates.** The actor minimises the negative scaled return (analytic gradients through the
   WM) minus an entropy bonus plus the TTC Lagrangian term; the lane head is trained by REINFORCE
   with the same advantages. With `entropy_target` set, the bonus weight is a learned SAC-style
   temperature. The critic regresses detached lambda-returns. Separate Adam optimisers, gradient
   clipping, EMA target critic.

## 7. Counterfactual cooperation objective (`counterfactual.py`)

With `[counterfactual] enabled`, each selected start is replayed in one batched rollout per member
with common random numbers in several variants: `B` (all CAVs native: simple IDM or hold speed),
`D_i` (only CAV `i` acts), `D_G` (a close group acts: CAVs chained within `group_radius_m`, size
2-6), `D_{G\i}` (group without `i`), `D_all` (all CAVs act), each relative to `B`. Derived:
cooperation gain `C_G = D_G - sum_i D_i`, marginal-in-team `M_i = D_G - D_{G\i}`. The actor
maximises `w_all D_all + w_group D_G + w_ind sum D_i + w_coop C_G + w_team sum M_i`, implemented as
per-row return coefficients. Validation enforces `w_coop <= w_ind` at every curriculum knot
(otherwise sabotaging solo behaviour would pay); weights follow a piecewise-linear schedule over
updates. A fraction of each batch is drawn from group-rich, congested starts. The same quantities
are measured in SUMO by `traffic_shepherd/c4/branching.py`, and `coop_imagine.py` re-computes
them in imagination from the same saved states to check WM-vs-SUMO agreement.

## 8. Checkpoints and outputs

`python -m wms.control.train_cli --config configs/<policy>.toml --wm-ckpt <wm_s_best.pt> ...`
(`--init-policy` loads actor/encoder weights only, `--resume` restores the full trainer, `--smoke`
runs two tiny CPU updates). Output directory: `updates.jsonl`, `validation.jsonl`,
`policy_NNNNNN.pt`, `policy_latest.pt`, `policy_best.pt` (best validation `D_all`, or imagined
return without counterfactuals), `policy_best_coop.pt` (best validation `C_G`), `RUN_SUMMARY.json`.
A `STOP` file in the output directory ends training after the current update.

Format `c4_dreamer/3` (`/1` and `/2` remain readable for policy-only loading): `config`,
`lane_rules`, `lane_head`, `counterfactual`, `reward_weights`, `policy`, `group_encoder`, actor
bounds, `critic`, `target_critic`, both optimisers, optional entropy temperatures, `update_count`,
`lagrange_multiplier`, return percentiles, best scores, RNG states, `ensemble_size`,
`world_model_ids` and sampler states. Loading uses `weights_only=True`. A full resume requires the
same WM ensemble unless `WMS_LOOP_REFRESH=1` (the data loop, [DREAMER_LOOP.md](DREAMER_LOOP.md)).

## 9. Deployment in SUMO (`deploy.py`, `traffic_shepherd/c4/`)

`policy.load_policy_fn(ckpt)` returns a `PolicyExecutor` (`state [1,K] -> accel`, or
`(accel, lanes)`), the deterministic actor with the same lane rules and safety layer. It is used
by `python -m traffic_shepherd.c4.collect --policy <ckpt>` and by:

```
python -m wms.control.deploy --controller {native|damping|pi|wall|policy|policy_check} \
    --policy <policy.pt> --wm-ckpt <wm_s_best.pt> [...] --seed <s> --demand <veh/h> \
    --penetration <p> --warmup 300 --horizon 600 --out <new dir>
```

`CavSumoEnv.step(accel, lanes=...)` writes direct accelerations through the simulator's command arbiter and
issues lane targets with `changeLane` under lane-change mode 513 (strategic changes stay on, TraCI
requests respect others' gaps); SUMO may refuse a request whose gap is below its own secure gap.

`policy_check` is the run-time verification mode: every `--decision-period` seconds the RSU imagines
`--imagine-horizon` seconds for direct-output candidates only (the policy, the policy with a zeroed
or perturbed group signal, stochastic policy samples, hold speed) in the frozen WM ensemble
(`rsu_check.RsuChecker`) and keeps the policy unless an alternative's paired improvement clears
`--margin` with a positive lower confidence bound. The imagined state comes from either the true
simulator state (`--state oracle`, an upper bound) or an RSU belief (`--state belief`,
`belief.BeliefTracker`): CAV self reports, their anonymous front/rear detections and RSU zone
detections, associated into tracks and propagated by WM-S between sightings. A vehicle no sensor
has seen does not exist in the belief.

## 10. MAPPO baseline (`src/wms/control/mappo/`)

Same `CavPolicy` + `RsuGroupEncoder`, lane rules, safety layer, `team_reward` and TTC constraint;
only the learning method differs: on-policy PPO in SUMO with a centralised `TeamCritic` on the
privileged simulator state, team GAE, and per-episode randomisation over penetration, demand and
development seeds. See `mappo/trainer.py` and `configs/c4_mappo.toml`.

## 11. Known limitations (from the code)

- Lane changes are discrete; their effect reaches the actor only through REINFORCE.
- Saturated safety/actuator bounds give zero action gradient.
- SUMO accepts only part of the lane requests that imagination executes, and imagined lane changes
  are instantaneous.
- At t0 of each training start the command/intent report features are zero.
- WM errors on unusual CAV behaviour can be exploited; mitigations are the ensemble penalty,
  sampled futures, the execution cap and the SUMO data loop.
