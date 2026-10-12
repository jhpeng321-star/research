# MAPPO baseline for WM-S C4 — implementation notes

Status: implemented and unit-tested on CPU with fake environments (no SUMO run yet, per the
no-training rule). Branch `c4/mappo`. Design reference: `docs/WM_C4_DESIGN.md` section 3
(final comparison, step 4b), shared interfaces: `src/wms/control/contract.py`.

## What this is

The model-free comparator: **MAPPO** (parameter-shared per-CAV actor + centralised critic)
trained **directly in SUMO** on exactly the C4 policy's observation, action and team-reward
interface. The actor architecture class is the *same* `wms.control.policy.CavPolicy` (with
`RsuGroupEncoder`), injected by dotted name, so the comparison isolates the learning method
(model-based gradients through the WM vs on-policy samples from SUMO).

- Per-CAV rewards = team reward (shared): one team reward stream per episode; the team GAE
  advantage is broadcast to every acting CAV slot; CAV entries/exits are handled by the
  `cav_mask`, not by episode dones.
- PPO: clip 0.2, value clip 0.2, entropy bonus, GAE(0.99, 0.95), minibatch epochs, batch
  advantage normalisation, return normalisation (Welford running stats; the critic predicts
  scaled returns, unscaled for bootstraps/GAE), grad clip, approx-KL early stop.
- Safety constraint: Lagrangian multiplier on the team `cost_ttc` exposure
  (PPO-Lagrangian, zero-baseline discounted cost-to-go, batch-normalised; multiplier α
  updated by dual ascent against `cost_limit_per_step`, no separate cost critic).
- Budget: dual-unit accounting (SUMO-seconds and env-steps) so the run can be matched to
  the C4 method's SUMO budget (`max_sumo_seconds`); an episode that would exceed the budget
  is never started. Default charges controlled seconds (`steps × 1 s`), warmup excluded
  (`charge_warmup` to change).
- Rollouts: N parallel SUMO workers, one env per process (spawn `multiprocessing.Pool` with
  an initializer; actor/encoder weights shipped per job as state dicts — no checkpoint-file
  races). Episodes sampled uniformly over the penetration mix (0.05/0.1/0.2/0.4) and
  demands (2400/2600/2700 veh/h), seeds derived from the episode index (deterministic,
  resume-safe).

## Layout / API

| Module | Contents |
|---|---|
| `src/wms/control/mappo/buffer.py` | `EpisodeData`, `flatten_episodes`, `compute_gae`, `discounted_returns`, `episode_gae`, `ReturnScaler` |
| `src/wms/control/mappo/ppo.py` | `PPOConfig`, `PPOUpdater` (clipped surrogate + value clip + entropy + Lagrangian cost term), `evaluate_actions`, `clipped_surrogate`, `clipped_value_loss`, `update_lagrange` |
| `src/wms/control/mappo/rollout.py` | `collect_episode` (one CavEnv episode), `resolve_name` (lazy dotted import + `fake*` shortcuts), `build_rollout_deps`, `run_job`, `make_pool` |
| `src/wms/control/mappo/trainer.py` | `MappoConfig`/`RolloutConfig`/`Budget`, `MappoTrainer` (iterations, resume, `log.jsonl`, `checkpoints/`, `RUN_SUMMARY.json`) |
| `src/wms/control/mappo/config_helpers.py` | TOML → config objects → trainer wiring (`build_trainer`, `merge_override`) |
| `src/wms/control/mappo/train.py` | CLI: `python -m wms.control.mappo.train --config configs/c4_mappo.toml --workers N --out DIR [--max-sumo-seconds X] [--smoke]` |
| `src/wms/control/mappo/eval_bridge.py` | `load_policy_fn(ckpt) -> policy_fn(state) -> accel [1, K]` — drops the saved actor into the same evaluation harness as the C4 policy (deterministic action + safety layer) |
| `src/wms/control/mappo/fakeenv.py` | `FakeCavEnv` + fake obs/reward/safety/actor/encoder/critic implementing the contract protocols (CPU tests, `--smoke`) |

Output directory: `RUN_SUMMARY.json` (status, budget totals, config, last diagnostics —
rewritten every iteration), `log.jsonl` (one `config` line, then one `iter` line per
iteration with losses/entropy/approx-KL/clip fraction/cost/α), `checkpoints/latest.pt` and
per-iteration `ckpt_XXXXX.pt` (models, optimisers, α, scaler, budgets, numpy/torch/python
RNG states — fully resumable; re-running the CLI with the same `--out` continues where the
budget stopped).

## Config (`configs/c4_mappo.toml`)

Sections `[mappo]` (PPO hyperparameters), `[rollout]` (seed, iterations, episodes per
iteration, horizon/warmup, penetration/demand mix, `max_sumo_seconds`, `max_env_steps`,
`charge_warmup`), `[env]` (`traffic_shepherd.c4.CavSumoEnv` + kwargs), `[policy]` (dotted
names + kwargs for `CavPolicy`, `RsuGroupEncoder`, `TeamCritic`, `build_observations`,
`team_reward`, `safe_accel`), `[reward_weights]` (`RewardWeights` fields), `[smoke]` (flat
overrides merged by `--smoke`; defaults to the fake env + a 64 SUMO-second budget).

## What was ported from the archive (copied, never imported)

- `archive/circles_era/traffic_shepherd/ppo_train.py`: `compute_gae` (done/bootstrapping
  handling), the clipped-surrogate minibatch-epoch update (advantage normalisation with
  degenerate-std guard, grad clipping, diagnostics, fixed-seed permutation idea), and the
  round-start checkpoint/episode-completion resume pattern.
- `archive/circles_era/traffic_shepherd/ppo_features.py`: the Welford running-statistics
  pattern (`RunningNormalizer`), re-targeted from observations to **returns** (value
  normalisation as in the MAPPO paper's implementation).
- `archive/circles_era/traffic_shepherd/ppo_env.py` / `ppo_worker.py`: the
  bounded-chargeable-episode discipline and worker-per-episode pattern, upgraded to one
  persistent SUMO env per worker process (a Pool initializer) instead of one subprocess
  per episode.
- `archive/pre_circles/code/env/mappo.py`: **not** ported — its flat local|peers|trust obs
  packing is replaced by the C4 `Observation` contract (own features + masked reports +
  group signal).

## Adapter assumptions (to reconcile when the parallel modules land)

Single-place adaptation points, all in `wms.control.mappo`:

1. `ppo.evaluate_actions` calls `actor.evaluate_actions(obs, group_signal, accel)` or
   `actor.evaluate(...)` for stored-action log-probs/entropy — `CavPolicy` needs one such
   method (the tanh-Gaussian reference implementation is `fakeenv.FakeActor`).
2. `group_encoder(obs) -> [B, GROUP_SIGNAL_DIM]` is assumed callable on the Observation
   (RSU computes it from CAV reports); change the one call site in `rollout.collect_episode`
   / `PPOUpdater.update` if `RsuGroupEncoder` takes the raw state instead.
3. `critic(state_dict) -> [B]` (privileged centralised `TeamCritic`); adjust the two call
   sites (`PPOUpdater.values`, `PPOUpdater.update`) if the real signature differs.
4. `team_reward` is called once per **control step** on `(s_t, s_{t+1}, accel, weights)`
   (the contract's per-physics-step function applied over the 1 s transition, the same
   quantity the C4 imagination loop accumulates).
5. Episodes end at `horizon_s` (time-limit truncation, bootstrapped) unless
   `info["terminated"]` is set by the env.

## Limitations / not done

- No SUMO run, no GPU, no real training performed here (rule: no training without owner
  discussion); the real `wms.control.{obs,reward,policy,safety}` and
  `traffic_shepherd.c4.CavSumoEnv` modules were still being written in parallel — the CLI
  is wired for their dotted names, and everything was verified against the fakes.
- Cost advantage uses a zero baseline (discounted cost-to-go) instead of a separate cost
  critic — lower-variance variant can be added if the constraint binds during real runs.
- `cost_limit_per_step` (default 0.02) needs a physical calibration pass once real
  `cost_ttc` magnitudes are measurable in SUMO.
- The multi-worker Pool path is exercised structurally (spec/job round-trip in-process in
  the unit tests) but not with N spawned processes here; first real SUMO run should
  smoke-test `--workers 2` before scaling.
- Warmup seconds are not charged to the budget by default; set `charge_warmup = true` if
  the C4 budget accounting includes warmup simulation.
