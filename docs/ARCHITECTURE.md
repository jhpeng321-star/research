# Architecture: the WM-S world model and the package map

This document describes **WM-S**, the physics-informed per-step world model of mixed CAV/HDV
traffic on a highway on-ramp merge, and how the repository is organised around it. The learned
CAV controller trained inside WM-S is described in [C4_DESIGN.md](C4_DESIGN.md), the SUMO data
loop in [DREAMER_LOOP.md](DREAMER_LOOP.md) and the on-disk formats in [DATA_FORMAT.md](DATA_FORMAT.md).
Paths are relative to the repository root. Where a design note and the code disagree, the code is
authoritative.

## 1. What the model predicts

WM-S is a **next-state model**: every 0.5 s step it maps the current state of all vehicles on the
modelled road to their next position, speed, acceleration and lane. Traffic quantities (delay,
queues, flow) are computed from predicted states; there is no separate delay head.

- **State.** `K` vehicle slots per batch row. Each slot carries `veh_kin [B,K,7]` =
  (`s`, lane one-hot x3, `v`, `a`, length) plus `present` and `is_cav` masks
  (`src/wms/physics/state.py`, `src/wms/core/constants.py`). `s` is the merge-relative
  longitudinal coordinate: the merge zone is `[0, 75.44]` m, upstream is negative. Lane 0 is the
  left mainline lane, lane 1 the right mainline lane, lane 2 the on-ramp/acceleration lane.
- **CAVs are exogenous.** The model never learns CAV motion. A CAV slot follows, in priority
  order, a `direct` action-timeline command, a recorded teacher trajectory, or a controller
  callback (`src/wms/model/cav.py`). This is what lets controllers that were never in the training
  data be evaluated zero-shot: the same physics and learned HDV behaviour respond to whatever the
  CAVs do.
- **HDVs are physics plus learned corrections.** Each human-driven vehicle moves by an IDM
  car-following core whose parameters are inferred per driver, with learned lane-change decisions
  (`lane_rule = learned_discretionary`, `lane_model = hazard_v5`; physics only masks unsafe changes;
  `lane_rule = mobil` remains for legacy configs), plus a gated, bounded learned residual.

## 2. One model step (`WmSModel.forward` / `step`, `src/wms/model/wms.py`)

1. **Driver encoding** (`src/wms/model/driver_encoder.py`). An attention encoder pools each
   vehicle's observed pre-t0 motion history (optionally a long history of up to 60 s,
   `src/wms/data/history.py`) and neighbour-interaction cues into per-driver settings
   `v0, time_headway, min_gap, accel, decel, politeness, mobil_threshold, merge_gap,
   merge_urgency`, an uncertainty per setting and a driver embedding. In `physical` mode
   (`model.driver_param_mode`) the five longitudinal settings are bounded physical ranges
   (`v0` as a ratio of the speed limit) whose deviations shrink toward a learned shared centre
   when history is short. **No driver labels are used:** the encoder is trained only through the
   prediction error of rollouts. Driver ground truth in `src/wms/data/driver_truth.py` is
   evaluation-only. An optional in-rollout update (`model.driver_update = "ema"`,
   `src/wms/model/driver_update.py`) refines settings when a vehicle is observed again.
2. **Physics core** (`src/wms/physics/`):
   - `idm.py`, `accel.py`: IDM acceleration with per-driver parameters, accel-lane end obstacle,
     optional emergency braking (`physics.emergency_decel_mps2`) and stop-time-aware integration
     (`physics.stop_time_integration`). `config.py` holds `PhysicsConfig`, including the
     SUMO-vType IDM means used under `sumo_fidelity`.
   - `mobil.py`: lane-change logits. `model.lane_rule` selects `mobil`, `forced_merge_only` (only the
     mandatory ramp merge 2 -> 1) or `learned_discretionary` (physics proposes only the forced
     merge; safe adjacent changes are open at a low prior for the learned lane model). Feasibility
     uses either a fixed gap test or the speed-dependent, one-step-anticipated secure-gap rule
     (`model.lane_gap_rule = "speed"`, constants `GAP_*` in `config.py`). Masked moves stay masked
     after every learned correction.
   - `merge_v6.py` plus switches in `mobil.py`/`accel.py`: opt-in merge-zone rules.
     `physics.merge_end_wait` lets a slow vehicle near the accel-lane end enter lane 1 only into a
     secure gap and never drive past the lane end; `physics.mainline_merge_yield` makes the
     nearest lane-1 follower of such a blocked merger brake (bounded) as if it were a virtual
     leader. Both apply equally to every vehicle and add no parameters.
   - `merge_cal.py` (calibrated merge yielding, off in C4 configs), `neighbours.py`,
     `lane_conflict.py` (optional front-to-back arbitration of simultaneous lane changes),
     `projection.py` (optional feasibility projection), `batched.py` (batched numpy-parity
     physics used by the entry process), `wall_exec.py` (rollout-local executor of the wall-group
     laws for evaluating the wall baseline).
3. **Gated residual** (`src/wms/model/residual_net.py`, `gate.py`, `residual_bound.py`). A
   transformer over neighbouring vehicles (conditioned on the driver embedding in `hybrid`
   mode) outputs an acceleration correction and a log-sigma. The correction is multiplied by a
   learned gate `g(k)` of the number of steps `k` since the last real observation (`g(0) = 0`,
   non-decreasing; `cumsum` or `saturating` kind) and softly bounded
   (`model.residual_soft_bound_mps2`). `model.rollout_mode`: `residual` (no driver settings),
   `driver_params` (settings only, no residual) or `hybrid` (both; current C4 configs).
4. **Lane decision.** `model.lane_model = "hazard_v5"` (`src/wms/model/lane_v5.py`) is the current
   choice: one shared network with one logit per road transition (overtake 1 -> 0, return 0 -> 1,
   merge 2 -> 1), driven by situation features and recurrent per-vehicle accumulators, never by
   driver identity; `lane_decision = "gated"` turns logits into discrete changes. Older options are
   the residual lane delta and the `hazard` model (`lane_hazard.py`).
5. **Entries** (`src/wms/model/entries.py`, default off). New vehicles enter from the RSU-observable
   entrance counters (Poisson, or a renewal process fitted to the observed pre-t0 entry history),
   with seeded common random numbers so paired comparisons are not diluted by arrival noise.
   `entry_insert_rule` selects the insertion test (`sumo_like_v2` mimics SUMO's saturated
   entrance).

`WmSModel.rollout` chains steps, carries histories, lane ages and lane accumulators, and can apply
forward-exact gradient damping per step (`model.grad_decay_per_step`) and per-step activation
checkpointing (`model.checkpoint_step`).

## 3. Training (`src/wms/train/`)

- `cli/assemble.py` turns a `TrainConfig` into datasets and a model; `train/trainer.py` runs the
  curriculum. Each stage unrolls the model open-loop from recorded start states over a longer
  horizon (`curriculum.unroll_s`, e.g. 30/60/120 s) with chunked truncated BPTT
  (`train/unroll.py`, `curriculum.truncation_steps_per_stage`). CAVs are teacher-forced from their
  recorded realised motion; all losses supervise HDVs and traffic, never controller labels.
- Loss terms (`train/losses/`, composed in a fixed order by `compose.py`): Huber position/speed per
  car (`trajectory.py`), lane cross-entropy or the teacher-forced lane-event loss
  (`lane_event.py`), a flip-rate hinge (`flip_rate.py`), residual-size penalties (`residual.py`),
  aggregate consistency with the RSU cell field and reward components the controller uses
  (`aggregate.py`), the paired-difference loss on branches that share a start state
  (`paired_diff.py`), plus braking, traffic-level, stability and driver-prior terms that are off
  in current configs.
- `train/selection.py` validates at several horizons and writes the selected checkpoint;
  `train/gates.py` and `train/preflight.py` provide a fail-closed smoke gate and a side-effect-free
  preflight / two-step CPU dry run.
- Run directory: `wm_s_best.pt` (plus `wm_s_latest.pt`, `wm_s_stage<N>.pt`), `physics.json` (the
  physics switches and constants the checkpoint must be rolled out with), `train_config.json`,
  `train_log.jsonl`, `val_log.jsonl`, `TRAIN_RECEIPT.json`. Checkpoints use format `wm_s/1`
  (`src/wms/model/checkpoint.py`): plain state dict plus the flat config, loaded with
  `weights_only=True`.

## 4. Evaluation (`src/wms/eval/`, `src/wms/cli/`)

`eval/predictor.py` (`WmsPredictor.predict_states`) rolls a checkpoint from an evaluation state
with CAVs teacher-forced from the recorded arm. `metrics.py`/`score.py` compute identity-matched
car errors and per-section traffic metrics at fixed horizons, `report.py` horizon tables and
blow-up rates, `compare.py` a paired candidate-vs-previous-best comparison, `ranking.py`
wall-vs-native choice and regret, `fingerprints.py` failure diagnostics. Reference predictors
(constant speed) live in `models.py`. Entry point:

```
python -m wms train --config configs/<preset>.toml [--set a.b=c ...] [--mode preflight|stage|run]
python -m wms preflight --config ... | python -m wms eval --config ... --checkpoint ...
python -m wms score --config ...     | python -m wms fetch list|status|verify|get
```

## 5. The SUMO side (`traffic_shepherd/`)

`traffic_shepherd/` is the simulation layer. It never imports `wms` except through
`traffic_shepherd/c4/wms_glue.py` (lazy imports), and `src/wms/sim_bridge.py` is the only `wms`
module that imports the simulation layer. `tests/unit/test_import_graph.py` checks these edges.

| module (`traffic_shepherd/c4/`) | role |
|---|---|
| `env.py` | `CavSumoEnv`: one TraCI episode on the merge network, 0.5 s physics / 1 s control, per-CAV direct accelerations and optional lane targets, frozen baselines (`native`, `damping`, `pi`, `wall`) |
| `sumo_io.py`, `sensing.py`, `state.py`, `geometry.py` | subscription-based reads, the CAV local sensor, SUMO -> model-frame slot state, rolling history, RSU detections |
| `population.py` | `legacy3` (three discrete driver types) or `textbook` (continuous per-vehicle IDM/LC2013 parameters) HDV population |
| `recorder.py`, `collect.py` | write-once episode recording; parallel, resumable collection |
| `genrand.py`, `genrand_driver.py` | undirected random CAV actions for WM training, optional paired branches |
| `branching.py` | saved-state branching: measure cooperation of any controller in SUMO |
| `seeds.py` | seed guard (only registered development/training/evaluation blocks) |
| `wallregen.py`, `wallregen_states.py` | regenerate unseen-wall evaluation states |
| `baselines.py`, `parity.py`, `smoke.py` | episode runner and baselines, featurizer parity, smoke checks |

Baseline controllers, used only for evaluation and as whole-episode comparators: CIRCLES damping
(`baselines/circles/`, ONNX weights, `traffic_shepherd/damping.py`), PI controllers
(`traffic_shepherd/pi_session.py` with the laws under `output/controller_verification_v{2,4,7}/`),
and the coordinated wall group (`traffic_shepherd/wall_group.py`). A single command arbiter in
`traffic_shepherd/` is the only longitudinal writer for CAVs; HDVs are never commanded.

## 6. Package map (`src/wms/`)

| package | contents |
|---|---|
| `core` | constants (controller vocabulary, `KIN_F`), errors, IO, seeds, restricted-path guard, TOML reader |
| `config` | typed config dataclasses (`train.py`, `data.py`, `eval.py`), preset loader, legacy flag map |
| `physics` | differentiable IDM core, lane safety mask (+ legacy MOBIL) and merge rules (section 2) |
| `model` | `WmSModel`, driver encoder, residual net, gates, lane models, entries, checkpoint IO |
| `data` | data registry, converted C4 episode sources, packs, windows, datasets, collation, hygiene |
| `train` | trainer, unroll, curriculum, losses, selection, gates, preflight |
| `eval` | evaluation states, predictor, metrics, scoring, reports, comparison |
| `cli` | `python -m wms` entry points |
| `control` | C4 policy, Dreamer trainer, reward, safety, RSU check, belief, deployment, MAPPO ([C4_DESIGN.md](C4_DESIGN.md)) |
| `legacy` | black-box baselines and vendored readers kept to score older checkpoints |

## 7. Configuration system

- **Typed, strict.** Training configs are TOML files deserialised into frozen dataclasses
  (`src/wms/config/schema.py`, `train.py`, `data.py`). Unknown keys and wrong types are errors.
  Sections: `data`, `model`, `loss`, `curriculum`, `optim`, `selection`, `runtime`, `physics`.
- **`extends` chains** (`src/wms/config/load.py`). A preset names its parent file (resolved next to
  it; depth <= 16, cycles refused). The parent is loaded first and the child is deep-merged on top:
  **tables merge recursively, arrays replace**. A child that declares `[[data.c4_sources]]`
  therefore replaces the whole inherited source list, which is why the source list is restated in
  several presets.
- **Overrides.** `--set a.b=c` (value parsed as TOML) is applied after the chain. The resolved
  config and its SHA-256 fingerprint are written to `config.resolved.json`.
- **Logical data names.** Configs never contain local paths; dataset and checkpoint names are
  resolved by `src/wms/data/registry.py` ([DATA_FORMAT.md](DATA_FORMAT.md#6-data-registry)).
- **World-model chain** (each file changes one aspect): `c4_wm.toml` -> `c4_wm_v2_loss.toml` ->
  `c4_wm_v2.toml` -> `c4_wm_v3.toml` -> `c4_wm_v4_lane.toml` -> `c4_wm_v4_data.toml` ->
  `c4_wm_v4.toml` -> `c4_wm_v5_base.toml` -> `c4_wm_v5.toml` (C4 episode sources replace the
  legacy packs via `data.c4_base`; `hazard_v5` lanes) -> `c4_wm_v6.toml` (physical driver
  settings) -> `c4_wm_v6_full.toml` (stop-time integration, emergency braking, long paired
  branches) -> `c4_wm_v6_cont.toml`, with loop-refresh descendants `c4_wm_v6_refresh.toml` and
  `c4_wm_v6m_refresh.toml` (adds the two merge-zone switches).
- **Policy chain** (`[run]`, `[world_model]`, `[trainer]`, `[data]`, `[reward]`, `[lanes]`,
  `[counterfactual]`, `[policy]`; loaded by `src/wms/control/train_cli.py` with the same merge
  rule): `c4_policy.toml` -> `c4_policy_base.toml` -> `c4_policy_base_p4.toml` -> `c4_policy_final.toml` ->
  `c4_policy_v5.toml` -> `c4_policy_v6.toml` -> `c4_policy_v6_full.toml` ->
  `c4_policy_v6_loop.toml`. `[data] wm_config` names the WM config whose data sources supply the
  imagination start states.
