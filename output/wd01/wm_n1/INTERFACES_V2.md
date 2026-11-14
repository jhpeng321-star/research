# WM-N1 INTERFACES v2 — Idea A (controller-in-the-loop, strategy-agnostic)

**Frozen:** 2026-10-01 on branch `zcode/wmn1ifv2` (base `zcode/revwm3`,
which carries `zcode/wmn1int` with TRAIN/BUNDLE/ADAPTER reviewed).
Amendment: `output/wd01/wm_n1/REGISTRATION_WM_N1_AMENDMENT_3_IDEA_A.md`
(relayed by owner decision; Idea A is the headline model). This supersedes
the piece-layer contract in `INTERFACES.md` (v1) and adds nine parallel
pieces. Design: `WM_N1_DESIGN.md`. v1 tensors/APIs are UNCHANGED and remain
the consumer surface.

Read with `git show` (do not merge): `zcode/wmn1d1:.../model/rollout.py`
(D1 position fix), `zcode/wmn1corpus2:.../corpus/**` (packed corpus),
`zcode/wd01-ctrl:.../partA_ctrl/**` and `zcode/wd01wall3:.../partA_v2g/**`
(the controller mirrors and the Part A v2g baseline engine).

Constraints: torch 2.5.1-compatible ops only; NO SUMO; no training and no
data reads inside tests; `WD_FORBID_D=1`; never touch `model/rollout.py`
position code in place — P3 owns the D1 integration.

## Tensors (new channels; the v1 shard tensors are unchanged)

Per-step world state, batched `[B, K]`, `K = 128`, `DT = 0.5 s`, 3 lanes
(0 main_left, 1 main_right, 2 accel). Defined in `rollout2/state.py`:

| field | dtype | shape | meaning |
| --- | --- | --- | --- |
| `s_m` | float32 | `[B,K]` | downstream position |
| `v_mps` | float32 | `[B,K]` | speed |
| `a_mps2` | float32 | `[B,K]` | last-step acceleration |
| `lane` | int64 | `[B,K]` | 0/1/2, -1 empty |
| `length_m` | float32 | `[B,K]` | bumper-to-bumper length |
| `present` | bool | `[B,K]` | slot occupied |
| `is_cav` | bool | `[B,K]` | CAV flag (never an HDV-net input) |
| `wall_group` | int64 | `[B,K]` | wall id, -1 = none |
| `slot_id` | int64 | `[B,K]` | vehicle identity |

New per-run featurizer channels (P9, `features/idea_a.py`), appended to the
v1 shard; the frozen `writer.py` is NOT edited (P9 owns `corpus/idea_a.py`):

| array | dtype | shape | meaning |
| --- | --- | --- | --- |
| `veh_kin` | float16 | `[N_ticks,128,7]` | kinematics only: `s, lane(3), v, a, length`; no CAV flag, no action one-hot |
| `hdv_base` | float16 | `[N_ticks,128,6]` | baseline per-HDV: `idm_accel, mobil_gain_left, mobil_gain_right, merge_urgency, lc_accept_left, lc_accept_right` |
| `cav_plan_ch` | float16 | `[N_ticks,128,8]` | controller planned channels: `tgt_speed/40, a_cmd/3, wall_speed/40, wall_id/4, hold/600, lane_target(3)` |
| `event_tok` | float16 | `[N_ticks,8,8]` | P6 event tokens (below) |
| `event_mask` | uint8 | `[N_ticks,8]` | P6 event mask |

Event token layout (`events/tokens.py`, `EVENT_F = 8`, `EVENT_MAX = 8`):
0-2 kind one-hot `{stopped_obstacle, blocked_lane_segment, speed_limit_change}`,
3 `s_down_m/1000`, 4 `lane/2`, 5 magnitude (`speed_delta/40` or
`segment_len/1000`), 6 `t_start_s/600`, 7 reserved 0.
Amendment-2 J4 road-end relabel is a run-level `meta.json` key
`amendment2_road_end_relabel` (P9 `apply_road_end_relabel`).

## Step contract (one 0.5 s tick; `rollout2/step.py`)

`step_once(world, actions, *, bundle, config, t_s, control_t_s=None, rng=None,
teacher_state=None) -> (WorldState, StepAux)`, in fixed order:

1. **(a) CAV ctrl (P1, 1 s cadence).** On integer control ticks build the
   `ctrl.CavStepInput` from `world`; `CavMover.step` returns per-CAV target
   speeds (NaN = no command). The engine executes them with
   `ctrl.execution_speed` once per control tick and holds the command for the
   two 0.5 s physics steps (baseline convention: partA_v2g/engine.py:568-572).
2. **(b) HDV baseline + residual (P2).** Baseline IDM + MOBIL/merge accel and
   lane logits; `HdvResidualNet.forward(kin, kin_mask)` adds
   `accel_delta_mps2` and `lane_logit_delta`. `is_cav` is never an input
   (headline); CAVs already have a target this tick.
3. **(c) Arrivals (P4).** Sample counts/speeds from
   `IdeaAArrivalHead(counter)` and insert into empty slots.
4. **(d) Validity + D1 (P3).** `project_rollout_step` clips acceleration,
   integrates `integrate_d1` (trapezoidal on speed), rejects illegal lane
   changes, projects same-lane rear bumper gaps >= 0; exits `s > lane_len`.

`StepAux`: `cav_target_speed_mps [B,K]`, `cav_reason_code [B,K]`,
`hdv_accel_delta_mps2 [B,K]`, `hdv_lane_logit_delta [B,K,3]`,
`baseline_idm_accel_mps2 [B,K]`, `counts` (v1 `ValidityCounts`),
`teacher_forced: bool`. Teacher forcing (training only) replaces the CAV
trajectory with the recorded `teacher [n_steps+1, B, K, 4]`.

## Existing APIs reused (read-only; consumers must not edit v1 semantics)

- `features/constants.py` indices and `FORMAT.md` layout (P9 reads).
- `model.net.{WmN1Config, WmN1Net, CounterEncoder, parameter_count,
  build_member}`; `model.losses.{pinball, outcome_crps, contrast_loss,
  gmm_accel_nll, neg_binom_nll}`; `model.rollout.ValidityCounts`
  (`rollout2/validity.py:26` imports, never edits).
- `evalh.protocol.{PredictorBase, Predictor, OutcomeRow, RolloutRow,
  RolloutBatch, QUANTILE_LEVELS, HORIZONS_S}`; `evalh.metrics`, `evalh.seals`,
  `evalh.run_eval` registration/sha plumbing.
- `combine.crps`, `combine.stacking`, `combine.oof_tree`,
  `combine.select_variant` (candidate set extended by P8).
- `train.config.TrainConfig`, `train.dataset.MemmapCorpus`,
  `train.trainer.Trainer` (extended, not rewritten).
- `partA_ctrl.emulators` (`ControllerEmulators`, `FrameBatch`, `PIState`,
  `wall_pace`, `wall_gap_to_front`, `execution_speed` rule) and
  `partA_v2g.engine` rule functions (`idm_accel`, `mobil_*`, `merge_*`,
  `segment_limits`, `execution_speed`).
- `corpus.{pack, verify, manifest}` and `corpus.source` (P9 passthrough).
- `inventory`/`features.writer` for run selection and byte accounting.

## New stub APIs per piece (signatures + docstrings; bodies `NotImplementedError`)

**P1 `ctrl/`** — `config.CtrlConfig`; `config.{CAV_ACTIONS, ACTION_CODES,
CIRCLES_MODEL_SHA256, WALL_LAW_V21, WALL_LAW_V22}`;
`wall_hook.{WallLaw, register_wall_law, resolve_wall_law, wall_law_status,
clear_registry, WallLawUnavailable}`;
`mover.{CavStepInput, CavStepOutput, CtrlState, execution_speed, CavMover}`;
`CavMover.init_state(batch, slots=128)` and `CavMover.step(state, inp,
actions, *, wall_speed_mps=None) -> CavStepOutput`. Wall v2.2 is resolved via
the hook; `partA_ctrl` keeps `WALL_LAW_VERSION = "wall-v2.1-closure"`.

**P2 `hdv/`** — `config.HdvConfig` (`use_cav_flag=False`), `config.KIN_FIELDS`;
`net.HdvResidualNet.forward(kin, kin_mask, *, cav_flag=None) -> {accel_delta_mps2,
lane_logit_delta, attn?}`; `net.lane_logit_delta_to_logits`.

**P3 `rollout2/`** — `state.{WorldState, StrategyActions, Rollouts}`;
`validity.{ValidityCounts, integrate_d1, project_rollout_step}`;
`step.{StepAux, step_once}`; `engine.{IdeaABundle, Rollout2Engine, rollout2}`;
`smoke.{j4_smoke_world, j4_smoke_contract}`. D1 (`integrate_d1`) is the
frozen trapezoidal integration.

**P4 `arrival/`** — `head.{ArrivalConfig, IdeaAArrivalHead.forward(counter)}`;
`sampler.{ArrivalSampler.sample_counts, sample_speeds, insert}`.

**P5 `corr/`** — `summary.{CorrConfig, SUMMARY_RAW_F, cav_motion_summary,
CavMotionSummary}`; `head.CorrectionHead.forward(summary, global_token) ->
{outcome_quantiles [B,5,3,19], collision_logits [B,5]}`, `CorrectionHead.crps`.
Input is the controller-produced CAV-motion summary only; never labels.

**P6 `events/`** — `tokens.{EVENT_F, EVENT_MAX, EVENT_KINDS, EventTokens,
encode_events}`; `masks.{event_token_mask, blocked_lane_mask, speed_limit_mask}`;
`synthetic.synthetic_event_batch` (implemented fixture).

**P7 `train/idea_a.py`** — `IdeaAConfigError`, `IdeaATrainConfig` (with
`seed_for_member`, `loss_weights`, `model_configs`, `from_dict`),
`ensemble_members(k)`, `MEMBER_ORDER`, `DEFAULT_SEEDS`, `IdeaATrainer`
(`fit`, `residual_nll`, `rollout_loss`, `arrival_loss`, `correction_crps`,
`paired_response_contrast`, `save_checkpoint`, `resume`, `evaluate_cal`).

**P8 `evalh/idea_a.py`** — `VARIANT_A`, `IDEA_A_VARIANTS = ("T","A","C","N")`,
`HeldOutStrategyTest`, `held_out_protocol()` (tests i/ii/iii, A3.4),
`IdeaAVariantPredictor(PredictorBase)`, `OodConfig`, `calibrate_ood`,
`ood_score`. P8 also owns the (documented) edits to
`combine.select_variant.VARIANTS` (add `"A"`) and the `adapter` bundle loader.

**P9 `features/idea_a.py` + `corpus/idea_a.py`** — `KIN_SOURCE_INDICES`,
`KIN_F`, `IDEA_A_CHANNELS`, `ROAD_END_RELABEL_FLAG`, `kinematics_view`
(implemented), `baseline_predictions`, `cav_plan_channels`,
`apply_road_end_relabel`, `IdeaAChannelSpecs`, `expected_bytes_idea_a`;
`corpus.idea_a.{IdeaAPassthrough, write_idea_a_channels,
road_end_relabel_enabled}`.

## Config keys (defaults)

Architecture/losses come from `train.IdeaATrainConfig`:
`hdv_d_model=128, hdv_n_heads=4, hdv_n_layers=3, hdv_ffn=512,
hdv_use_cav_flag=False, corr_summary_dim=64, arrival_counter_channels=128,
member=1, n_members=5, seeds=(101,202,303,404,505), ensemble_k=5,
epochs=15, steps_per_epoch=2000, batch_size=8, lr=1e-3, grad_clip=1.0,
lambda_residual_nll=1.0, lambda_rollout=1.0, lambda_arrival=1.0,
lambda_correction_crps=1.0, lambda_response_contrast=1.0,
rollout_horizon_steps=20, teacher_forcing_cavs=True, device="cpu", out_dir,
resume`.
`CtrlConfig`: `dt_s=0.5, control_dt_s=1.0, sensor_range_m=100.0,
accel_min_mps2=-3.0, accel_max_mps2=1.5, model_sha256=<circles>,
pi_u_mps=31.29, wall_law="wall-v2.2", wall_close_max_delta_mps=2.0,
wall_close_gain_per_s=0.1, wall_assembly_timeout_s=60.0,
wall_buffer_decel_mps2=-3.0, wall_comfort_accel_mps2=1.5, nominal_length_m=5.0`.
`HdvConfig`: `d_model=128, n_heads=4, n_layers=3, ffn=512, radius_m=150.0,
dropout=0.0, use_cav_flag=False, accel_delta_scale_mps2=1.5, lane_logit_scale=1.0`.
`ArrivalConfig`: `n_entrances=2, counter_bins=30, counter_channels=128,
d_model=256, max_insert_per_entrance=3`.
`CorrConfig`: `summary_dim=64, d_model=256, n_tau=5, n_outcome_targets=3,
n_quantiles=19`.
`OodConfig`: `cal_quantile=0.95, min_summary_std=1e-6`.
v1 `model.net.WmN1Config` / `train.TrainConfig` keys are unchanged.

## File ownership (disjoint)

Every path below is owned by exactly one piece; no piece edits another's.

| piece | owned paths (under `output/wd01/wm_n1/`) |
| --- | --- |
| P1 CTRL | `ctrl/**` |
| P2 HDV | `hdv/**` |
| P3 ROLLOUT2 | `rollout2/**` |
| P4 ARRIVAL | `arrival/**` |
| P5 CORR | `corr/**` |
| P6 EVENTS | `events/**` |
| P7 TRAIN | `train/idea_a.py`, `train/tests/test_idea_a_*.py` |
| P8 EVALH | `evalh/idea_a.py`, `evalh/tests/test_idea_a_*.py`; owns the documented `combine/select_variant.py` and `adapter/predictor.py` edits |
| P9 FEATURES | `features/idea_a.py`, `features/tests/test_idea_a_*.py`, `corpus/idea_a.py`, `corpus/tests/test_idea_a_*.py` |
| INTEGRATOR | `INTERFACES_V2.md`, ledger/registration records |

Consumers that need a change are assigned to one piece: `combine/` and
`adapter/` -> P8; `bundle/` stage list -> P7. `model/` is NOT edited (P2/P5
consume it).

## Mismatches between existing code and Idea A (`file:line`)

1. **D1 — rollout never integrated position.** `model/rollout.py:84`
   (`s_new = s.clone()`) fixed only on `zcode/wmn1d1`; this branch still has
   the defect. P3 owns `rollout2.validity.integrate_d1` (trapezoidal).
2. **v1 rollout moves every vehicle with the learned decoder.**
   `model/rollout.py:234-265` samples GMM accel for all slots; Idea A moves
   CAVs with the real controllers. P3 replaces it (`rollout2/engine.py`).
3. **Existing net conditions on the CAV flag.** `model/net.py:346-349`
   (`is_cav = veh[..., C.IS_CAV_IDX]`) feeds `PlanUpdater`/`WallUpdater`; the
   Idea A HDV net must not. P9 `kinematics_view` strips index 7 and 8-12.
4. **Action labels leak through v1 features.** `features/constants.py:19-27`
   put action one-hots in `veh_step`; `train.idea_a` must consume `veh_kin`
   (P9), never `veh_step`, for the HDV residual.
5. **Baseline integrator is ballistic / scalar.** `partA_v2g/engine.py:661`
   (`s_new = s + v_new*dt`) and the per-vehicle loops (engine.py:562-697) are
   incompatible with the batched D1 layer; P3 reimplements batched.
6. **No wall-v2.2 in the mirrors.** `partA_ctrl/emulators.py:437`
   `WALL_LAW_VERSION = "wall-v2.1-closure"`; P1 routes v2.2 through
   `ctrl.wall_hook` (registry) until `zcode/wallmr` lands.
7. **Correction/selection ignores variant A.** `combine/select_variant.py:21`
   `VARIANTS = ("T","C","N")`; P8 adds `"A"` and the A3.3 tie order.
8. **Adapter loads one net only.** `adapter/predictor.py` (`WmN1Predictor`)
   has no composed Idea A bundle; P8 extends it.
9. **Arrival head shape differs.** v1 `model/net.py:262-271 ArrivalHead`
   consumes the global token; P4's head is counter-driven and feeds the P3
   rollout directly.
10. **D1 regression tests live on another branch.**
    `zcode/wmn1d1:model/tests/test_rollout.py` (+88 lines) is NOT on this
    branch; P3 must add equivalent coverage for `integrate_d1`.
11. **`features.writer` is frozen.** `features/writer.py:43,84`
    `expected_bytes`/`write_shard` know only the v1 arrays; P9 adds
    `features/idea_a.expected_bytes_idea_a` + `corpus/idea_a` passthrough and
    does not mutate the writer.
12. **Outcome target names.** v1 mismatch 3 stands: `model/constants.py:57`
    `OUTCOME_TARGETS` vs `features/outcomes.py:23-28` label columns; P5/P7
    map them offline only.
13. **`train.TrainConfig.device` defaults to `"cuda"`.**
    `train/config.py:69`; Idea A CPU smoke uses `IdeaATrainConfig.device="cpu"`.

## Test fixtures

- **60 s J4-world CPU smoke contract (integration gate).**
  `rollout2.smoke.j4_smoke_contract()` is frozen:
  `batch=2, slots=32, n_steps=120 (60 s at 0.5 s), n_samples=1, seed=0,
  device="cpu"`. `j4_smoke_world()` builds the synthetic kinematics-only
  batch. When P1-P5 land, `Rollout2Engine.run` on this batch must return
  finite `positions/speeds/lanes/masks`, all `ValidityCounts` reported, and
  no forbidden input — the first end-to-end CPU check before any GPU job.
- **P6 synthetic events.** `events.synthetic_event_batch(batch=B)` (implemented)
  builds 3 live event tokens per world for shape/mask tests.
- **P1/P3 contract tests.** `ctrl/tests/test_ctrl_interface.py` pins
  `execution_speed`; `rollout2/tests/test_rollout2_interface.py` pins the D1
  trapezoidal rule and the stub boundaries.

## Torch-2.5 notes

GPU box pins `torch>=2.7,<2.13` from the CUDA 12.8 (cu128) wheel index
(Blackwell-class sm_120; local CPU box 2.12.1+cpu). The "2.5" in this heading is
historical. Every new module uses only 2.5-compatible ops that are also valid
across 2.7-2.12: `nn.{Linear,LayerNorm,GELU,Conv1d,ModuleList}`,
`F.{softmax,logsigmoid?,...}` avoided where possible, `torch.{clamp,stack,
gather,index_select,where,zeros,ones,full,arange,tensor}` and
`Tensor.scatter_reduce` (used by `partA_ctrl.wall_gap_to_front`, 2.5-OK).
No `torch.compile`, `torch.cuda.amp`/`autocast` (deprecated; use `torch.amp`),
`scaled_dot_product_attention`, `flex_attention` or `torch.export`. Checkpoint
loaders must pass `torch.load(..., weights_only=...)` explicitly (supported
Torch now defaults `True`); save plain `state_dict`s + JSON config. Rollout
randomness uses CPU `torch.Generator` + `numpy.random.default_rng`, so results
are device-independent.
