# ARCHITECTURE_V4 changes and notes from WP-A

Deviations from, and additions to, `docs/ARCHITECTURE_V4.md` made while implementing
WP-A (branch `wms/clean-a`). Public signatures of section 2 are unchanged unless listed
under "Interface changes". Other work packages should read sections 1 and 2 before
rebasing on this branch.

## Interface changes (only these)

| area | doc | now | why |
|---|---|---|---|
| `DataConfig.cav_drive` | `Literal["teacher","native","damping","pi"]` | `Literal["teacher","closed_loop"]` | the legacy trainer/CLI only has `teacher` and `closed_loop` (`CAV_DRIVE_MODES`); per-CAV kinds come from the action timeline |
| `TrainConfig.to_legacy_flat()` | no arguments | `to_legacy_flat(resolve=None)` | optional callable mapping logical data names to local paths; checkpoints store resolved paths. Default keeps names as written |
| `CurriculumConfig.stage_steps` default | `(2500, 2500, 1875, 1250)` | `None` | schema defaults equal the legacy `WmSTrainConfig`/CLI defaults so unspecified flags behave as before; recipes live in presets |
| `ModelConfig.merge_cal` default | `True` | `False` | same reason (legacy default OFF); `r3.toml` turns it on |
| `SwitchingConfig.mix_ratio/event_oversample` defaults | 0.3 / 1 | 0.5 / 2.0 | legacy CLI defaults; `r3.toml` sets 0.3 / 1 |
| `ResidualLossConfig` | `residual_mag_weight` only | adds `hybrid_magnitude_weight: float|None` | flat key `residual_magnitude_weight` (hybrid identifiability guard) is a different field from `residual_mag_weight` |
| `ModelConfig` | no flags for physics options | adds `sumo_fidelity`, `geometry_only`, `merge_cal`, `physics_params`, `allow_rejected_hard_cap` | `--sumo-fidelity`, `--geometry-only`, `--merge-cal`, `--merge-cal-params` shape `PhysicsConfig` in the legacy CLI |
| `CurriculumConfig` | no `epochs_per_stage`, `steps_per_epoch` | both added | flat keys of the legacy config |
| hard residual caps | "validator warns, trainer refuses" vs "validate() rejects" (doc is inconsistent) | constructor warns; `validate_for_training()` raises unless `allow_rejected_hard_cap` | old receipts with caps must stay loadable for re-scoring |

## Additions not in the doc

* `src/wms/core/guard.py` (`find_restricted_token`, `refuse_restricted`): the single restricted-token
  matcher used by the registry and `TrainConfig.validate_for_training()`. WP-C's source-tree scan
  (`tests/unit/data/test_no_sealed_tokens.py`) must exempt this file: it contains the token patterns.
* `src/wms/core/toml.py` + `toml_subset.py`: `tomllib` -> `tomli` -> vendored strict subset reader
  (Python 3.10 env has none of the first two). `core/logging.py` and `core/tensors.py` from the layout
  tree were not created (nothing to put in them yet).
* `src/wms/physics/config.py`: skeleton home of `PhysicsConfig` (doc section 2.4 lists the class but no file).
* `config/legacy_flat.py` (flat key <-> nested path table), `config/legacy_flags.py` (all 88 legacy flags (89 option strings),
  old defaults, `parse_legacy_argv`), `config/presets.py`, `config/eval.py` (`EvalConfig`, `RolloutFlags`,
  `EvalSetConfig`; fields beyond the doc's list are minimal and WP-E may extend them).
* `TrainConfig.required_log_terms()` (chain semantics: position, speed, lane, then traffic, asym_speed,
  decel_weight, propagation, stability, conservation when enabled) and `validate_for_training()`
  (num_workers == 0, caps, long-horizon selection needs a val list, restricted tokens).
* `DataConfig.effective_start_shift_prob`: the legacy CLI forces start-shift 0 with `--leave-pi-wall-out`;
  WP-C's dataset must use this property, not the raw field.
* `sim_bridge`: `load_controller_dependencies`, `make_sensor_frame`, `clamp_acceleration`,
  `make_cav_controller(kind, deps=None)` with `DampingController` / `PiFleetController` implementing
  `target_speed(vehicle_id, frame, *, now_s)` and `invalidate(vehicle_id)`. Numerics equal
  `wms.train_controllers` / `wms_plugin` (same PI setpoint 31.29 m/s, 1 s sample period).
* `data/registry.py`: `lookup`, `env_var_name`, `split_name` besides `DataItem/resolve/require_present`.
  A name is `<dataset>[/<relative path>]`; the dataset part is looked up, the rest is joined. Absolute
  paths are accepted as already resolved (box overrides and old argv). All results pass the restricted guard.

## CLI contract for WP-D / WP-E

`cli/train.py` (WP-A) assembles training from these interfaces and nothing else:

```python
corpora   = {role: PackCorpus.open(require_present(name), name=name) for role, name in cfg.data.packs.items()}
selection = select_windows(cfg.data, corpora)            # dataclass with .train, .val
train_set = TrajectoryDataset(cfg.data, selection, corpora, unroll_steps=first_stage_steps, seed=seed)
val_set   = TrajectoryDataset(cfg.data, dataclasses.replace(selection, train=selection.val, val=[]), ...)
model     = build_model(cfg.model)                       # physics built inside from cfg.model flags
result    = WmSTrainer(cfg, model, train_set, val_dataset=val_set, out_dir=out).fit()
```

`WindowSelection` must therefore be a dataclass with `train`/`val` fields. `--mode preflight` calls
`wms.train.preflight.preflight(cfg)` and needs `.ok` and `.findings`. `cli/eval.py` keeps two names that
WP-E replaces the body of: `add_arguments(parser)` and `run(args) -> int`. `cli/score.py` calls
`wms.eval.score.build_result(cfg.models, cfg.sets, config=cfg)`.

## Data names introduced by the presets (for the data owner to register)

`pack_v3 pack_dspeed pack_damping pack_switching_train identities_pack_v3 identities_dspeed
identities_damping overlay_dspeed overlay_damping sidecars_packv3_v2 (identities/, outcomes/)
lpwo_v3 (windows_*, excluded_runs_*) lpo_c1 (windows_train_noPI.json, windows_val.json, pi_excluded_runs.json)
identities_switching_train driver_params_switching_train windows_train_hetdata
action_timeline_index_train zero_shot_heldout ckpt_c1fast eval_states_wall_same eval_states_wall_pilot
eval_states_seen_controls`; `merge_params` is registered in `configs/data_registry.toml`
(`configs/merge_params.json`, byte copy of `_out/merge_params.json`). `scripts/box/launch.sh` exports
the box locations as `WMS_DATA_<NAME>`.

## Facts found in the real artefacts (differ from the doc text)

* **C1FAST receipt**: `physics.merge_cal` is `false`, `n_controllers` is 6 and `num_workers` is 5.
  The doc says "merge_cal on" and the R3 chain comment says "merge_cal as C1"; the receipt says otherwise.
  `configs/c1fast.toml` follows the receipt; `r3.toml` (chain default `MERGE_CAL=1`) turns merge_cal on
  and sets `n_controllers = 7`, `num_workers = 0`. Training `c1fast.toml` as is is refused (workers != 0).
* The R3 preset is the chain's `R3_FULL` defaults (no soft bound, no history, no jam, no selection);
  the numbers in the doc's `r3.toml` example were placeholders.
* The legacy `--batch-size` default is 4 while `WmSTrainConfig.batch_size` is 8; the legacy argv parser
  keeps 4, the schema default is 8.
* The R3 chain checks `SW_DRIVER` exists but never passes it to the CLI; it is kept as
  `data.switching.driver_params` for the existence check in preflight.

## Environment

Python 3.10 (`rl_driving`): `pyproject.toml` declares `requires-python >=3.10` and keeps ruff `py310`
(the doc proposed py311 for `tomllib`). Tests that compare with the legacy package run it in a
subprocess (the two packages are both named `wms`) and skip if torch or the legacy tree is absent.

## WP-D (train) deviations from the section 2.5 signatures

* `build_loss(settings: TrainSettings, *, physics)` instead of `build_loss(cfg: LossConfig)`: the traffic term needs the
  model's dt and merge-zone bounds. `TrainSettings` (`train/settings.py`) is the single mapping point from
  `TrainConfig.to_legacy_flat()` (legacy field names); the trainer needs nothing else from `wms.config`.
* `run_gate(name, cfg, out_dir, *, model, dataset, ...)` (needs the model/dataset); `check_log(log, spec, terms)` is the
  pure log checker. One `GateSpec` covers check400 / gate_run2 / gate_s2 / gate_stages / the chain's inline checks.
* `plan_steps(cfg, budget)` -> `stage_plan(settings, dt_s)` (explicit plan), `plan_from_log`, `project_hours`.
* `preflight(cfg, *, max_hours, out_dir, inputs, scripts)`; `dry_run(cfg, model, dataset, out_dir=)` lives in `preflight.py`.
* `WmSTrainer.__init__` gains `allow_rejected_hard_cap=False` (hard `residual_cap_*` refused for new runs, change-log #3);
  `fit()` returns `TrainResult(receipt, checkpoint, history, val_by_stage)`. Per-chunk entry points: `forward_backward`.
* `LossContext` carries no `stage_s`/`batch` (unused); `TermValue(value, valid_count, log)`.
* Loss-share diagnostic: legacy mislabelled the traffic share with the hybrid residual-penalty tensor when
  braking was off in hybrid mode; the new code always uses the traffic tensor (diagnostic only, no numerics).
* `loss_share.jsonl` rows are now key-sorted; `TRAIN_RECEIPT.json` gains optional `config_fingerprint`.
* Imports expected from siblings: `wms.physics.projection`, `wms.model.{stability,entries,driver_encoder,checkpoint}`,
  `wms.data.{collate,action_timeline,driver_truth}`, `wms.core.errors.WmsError`. `TimelineControllers` imports the frozen
  sim-layer controllers lazily itself (`train/cav_drive.py`); the integrator may route that through `sim_bridge`.
* Not ported: the shell chain's CPU-contention/RSS timing guard (`guarded_train`, Linux `/proc`); `preflight` checks
  config, restricted-token paths, LF-only scripts, fresh output dir and free disk.

---

# ARCHITECTURE_V4 changes (WP-E)

Interface notes from the eval rewrite (`src/wms/eval/`). No public signature of another WP was changed.

* `src/wms/__init__.py` is NOT created by WP-E (owned by WP-A); `wms` is a PEP 420 namespace until A lands.
* `eval.states.materialize_set(spec, out)` is realised as `materialize_windows(out_path, set_name, windows, packs, identities, *, horizon_s, ...)`
  because `EvalSetConfig` (WP-A) was not available. Set builders (`eval/sets/*`, unseen/seen splits) and `walls.py`, `decide.py`,
  `driver_recovery.py` are NOT ported in this pass.
* `WmsPredictor` takes a `WmsBackend(model, config, tools, projection)`; `RolloutTools` bundles helpers from other WPs. Required names
  (WP-I must provide them): `wms.data.history.{history_spec,long_history_features,history_step_features}`,
  `wms.data.presence.junction_carry_presence`, `wms.data.action_timeline.{ActionTimeline,CONTROLLERS}`, `wms.model.entries`,
  `wms.model.checkpoint.load_checkpoint` (returning `.model`, `.train_cfg.to_legacy_flat()`), `wms.model.stability.{FeasibilityProjection,ProjectionConfig}`.
  The model must expose `rollout(...)`, `hold_teacher`, and `step(...)` (traced by `eval.trace`).
* Not ported: closed-loop CAV mode and black-box kinds (predictor raises for `cav_mode != "teacher"`); `rollout_mode == "driver_params"` is refused (change-log #15).
* The `recorded_paths` physics-log loader of the legacy `data.recorded_for` is dropped.
* `wms.eval.__init__` does not re-export `compare`/`ranking` (would shadow the submodules).
* Pre-integration tests use `tests/support/legacy_eval_bridge.py`, which loads the legacy `wms` package as `wms_legacy` and fakes the
  `wms.data.*` names above; delete it once WP-B/WP-C are merged.

---

# ARCHITECTURE_V4 changes recorded by WP-B (physics + model)

Interface deviations from section 2.4 (all because WP-A types did not exist yet):

* `WmSModel.__init__(spec: ModelSpec | None = None, *, physics=None, hdv=None, physics_step=None)`.
  `wms.model.spec.ModelSpec` (frozen dataclass) carries every legacy constructor option
  (black_box, rollout_mode, lane_head, history_*, k_hidden, n_controllers, caps / soft bounds,
  lane_logit_cap, history_s, history_stride_s). The integrator maps `ModelConfig` (+ `history_s`
  from `DataConfig`) onto it; `ModelSpec.from_mapping(flat_legacy_config)` already does so for
  `checkpoint["config"]`. `hdv` (residual-net width) stays a separate kwarg as in the legacy code.
* `build_model(spec, physics=None, *, kind="structured"|"blackbox_id"|"blackbox_traj", hdv=None,
  action_conditioned=True)`.
* `load_checkpoint(path, *, device, strict, model_kind="structured", physics=None,
  config_factory=None) -> LoadedWms`; `LoadedWms` gained `legacy_config` (flat dict) and
  `projection` (a `FeasibilityProjection` when the run used it). `train_cfg` is
  `config_factory(legacy_config)` (pass `TrainConfig.from_legacy_flat`), else `None`.
  `strict=True` rejects unexpected keys; missing `driver_encoder.*` (residual models) and
  `command_proj.*` (pre-action-timeline) stay allowed, exactly as in the legacy loader.
* `save_checkpoint(path, model, cfg, ...)`: `cfg` may be a flat dict or an object with
  `to_legacy_flat()` / `to_dict()`.
* Physics functions take explicit tensors plus `PhysicsConfig`:
  `lane_neighbour(kin, present, query_lane, physics, *, ahead)`, `lead_neighbour(kin, present, physics)`,
  `idm_accel(v, leader_v, gap, IDMParams, *, accel_min, accel_max)` (v0 lives in `IDMParams`),
  `mobil_merge_logits(kin, present, physics, driver_params)`, `baseline_accel(...)`,
  `merge_cal_yield(...)`, `project_gaps(...)`. `PhysicsConfig` lives in `wms/physics/config.py`
  (still a plain mutable dataclass, `idm()` tuple kept). `state_kin` / `state_present` replace the
  private `_state_kin` / `_present`.
* Constants without a `wms.core` yet: `KIN_F`, `DEFAULT_LENGTH_M` in `wms.physics.state`;
  `HISTORY_H/F`, `LONG_HISTORY_F`, `history_spec`, `history_step_features` in `wms.model.history`;
  `CONTROLLER_VOCAB`, `N_CONTROLLERS`, `controller_index` in `wms.model.controllers`.
  WP-A's `core.constants` should re-export these (physics/model must not import config upward).
* `FeasibilityProjection` lives in `wms.physics.projection` (it needs no model code); the loop reference
  implementation `_project_gaps_reference` was dropped from the module (still in the legacy tree).
* `balanced_lane_loss` stayed in `wms.model.lane_head` (WP-D `losses/lane.py` should import it).
* `physics/batched.py`: legacy `physics_torch.PhysicsConfig` renamed `BatchedPhysicsConfig`.
* The numpy `rollout2` merge calibration is not vendored (only the torch implementation is used).
* `wms/__init__.py` is intentionally not created by WP-B; the packages work as namespace packages with
  `src` on `sys.path`. Legacy-vs-new tests load the legacy tree as `wms_legacy` and remove `wm_n1` from
  `sys.path` afterwards so it cannot shadow `src/wms`.

---

# ARCHITECTURE_V4 interface changes

## WP-C (wms/clean-c) - data layer

Deviations from the sketches in ARCHITECTURE_V4 section 2.3 (all additive or forced):

* `TrajectoryDataset` keeps the legacy classmethod constructors (`from_arrays`, `from_pack`,
  `from_real_packs(...)` with unchanged kwargs); the sketched `__init__(cfg: DataConfig, ...)`
  and `set_unroll_steps` were not built (no DataConfig exists yet; the unroll horizon is the
  `horizon_steps` argument, as today). The integrator maps `DataConfig` -> kwargs.
* `PackCorpus`: `open(pack_dir, *, extra_pack_dirs, extra_overlay_dirs, extra_identity_dirs,
  horizon_steps)` (multi-pack composite); `run_meta/item(run,tick)` were replaced by
  `meta(i)`, `item(i)`, `pack_index(i)`, `manifest_position(i)`, `array(name)`, `window(item)`,
  `run_bounds/run_start/run_end`, `shifted_item`. `resolve_pack_name` lives in
  `windows.resolve_window_pack_names(corpus, names, keep_raw)` (same fail-closed rule, #12).
* `windows.py`: `select_windows(corpus, *, window_list, exclude_window_list, names)` ->
  `WindowSelection(positions, receipt, kept_by_pack)`; `window_counts(...)` -> `{pack: (n_train, n_val)}`;
  `ControllerExclusion(horizon_steps=, heldout=, leave_pi_wall_out=)` is the former
  `TrainingWindowGuard` (item_filter callable, not `refuses(meta)`).
* `mixdata.MixedTrajectoryDataset` / `jam.JamOversampledDataset` keep their names; the
  architecture names `MixedDataset` / `JamOversampler` are aliases. Legacy `python -m wms.mixdata`
  CLI removed (`preflight`, `check_receipt` stay as functions; CLI belongs to `wms.cli`).
* `driver_truth.py` merges `driver_sidecar` + `dense_truth` (CLIs dropped). `driver_recovery`
  needs the encoder parameter names; `driver_truth.PARAMETER_NAMES` is a copy of
  `wms.model.driver_encoder.PARAMETER_NAMES` (data may not import model). WP-A may move it to
  `core.constants`.
* `pack_history.entry_features` lazily imports `wms.model.entries.build_entrance_history`
  (the one data->model edge; only for `entries_v2`, default off).
* `action_timeline.py`: `load_timeline`, `ControllerStateSemantics`, `sample_window_indices`
  dropped (no caller outside legacy tests); `from_pack` no longer carries the A11 `veh_kin`
  manifest branch (it passed raw metadata to `window()` and could not work).
* `window_lists.build_lpo` / `lpwo.build_lpwo` are ports; `build_lpwo` and `classify_window`
  take an injected `read_wall_timeline` callable (wall timeline reader belongs to eval).
* `tools/data_guard.py`: self-contained `DataGuard(spec, allowed_roots, workspace=)`; the
  external "canonical guard" tool argument is gone (no such tool exists in this tree).
* Provisional `src/wms/{__init__,core/*}.py` stubs were committed because no wms/clean-a
  skeleton existed; on merge take wms/clean-a's versions (names used: `core.constants`
  `CONTROLLER_VOCAB, controller_index, HISTORY_H, HISTORY_F`; `core.errors.TrainDataRefusal`;
  `core.io.run_key`).
* Latent legacy hazard kept as is: `val_split.materialise_holdout` writes `*_holdout.json`, a
  name `validate_train_source` refuses; use `mixdata.write_world_split`.
* Vendored reader: `src/wms/legacy/vendor/{train,dspeed,model,corr,features}` (see headers).
  `plan_horizon_s` of `IdeaACorpus.window` is unsupported (needs unvendored `model.paired_effect`).


---

# ARCHITECTURE_V4 changes recorded by WP-I (integration, branch `wms/clean-int`)

Everything below is what the integrator had to change or decide to make the six packages (and the
golden harness) work together. Full narrative, counts and commands: `docs/INTEGRATION_REPORT.md`.

## Merge facts

* WP-D had merged *earlier snapshots* of WP-B and WP-C, so the tree carried WP-A stubs where B/C had
  later implementations (`model/{checkpoint,factory}.py`, `legacy/blackbox/*`, `data/window_lists.py`,
  `data/lpwo.py`, `tools/data_guard.py`, the B/C unit and equivalence tests). `wms/clean-b` and
  `wms/clean-c` were merged again; implementing packages win over WP-A stubs (`src/wms/**`).
* `core/{constants,errors,io}.py`: WP-A's versions kept (supersets of WP-C's provisional ones).
* Constants now have ONE definition in `wms.core.constants` (`KIN_F`, `HISTORY_H/F`, `LONG_HISTORY_F`,
  `DEFAULT_LENGTH_M`, `CONTROLLER_VOCAB`, `N_CONTROLLERS`, `controller_index`); `physics.state`,
  `model.history`, `model.controllers`, `data.history`, `eval.rollout_inputs` re-export them
  (`tests/unit/test_constants_single_source.py`). `history_spec` still exists three times (config / data /
  model, same arithmetic); a test pins them to each other.

## Interface decisions (where the documents disagreed with the code that landed)

| topic | document / WP-A contract | what exists | decision |
|---|---|---|---|
| dataset assembly | `TrajectoryDataset(cfg.data, selection, corpora, ...)`, `select_windows(cfg.data, corpora)` | WP-C kept the legacy API: `TrajectoryDataset.from_real_packs(paths, identity_dirs=, overlays=, window_list=, ...)` over ONE composite `PackCorpus` | `wms/cli/assemble.py` reproduces the legacy `wms.cli` assembly call for call (registry names resolved there; `ControllerExclusion`/`HeldOutGuard` item filter; `mixdata.build_switch_datasets` + `mix_datasets`; jam oversampling without a switching pack) |
| model assembly | `build_model(ModelConfig)` | `build_model(ModelSpec, PhysicsConfig, kind=)` | `assemble.build_model` = `ModelSpec.from_mapping(cfg.to_legacy_flat())` + `assemble.build_physics` (merge-cal params file, `sumo_fidelity`, `geometry_only`) |
| preflight report | `.ok`, `.findings` | `PreflightReport.ok/.failures/.plan` | CLI prints `failures` and `plan`; every declared data name is resolved and passed as `inputs` (unresolvable names become failures, not crashes) |
| `score` | `wms.eval.score.build_result` | `score.py` has `evaluate_state` / `score_states` only | new `wms/eval/run.py::build_result(models, sets, *, config)` (score every model on every configured set once, compare each with the single previous best through `compare.compare`); `cli/score.py` and `cli/eval.py` call it |
| eval set subsets | none | needed for the 20-state quick check | `EvalSetConfig.subset = "first:N" | "first_per_world:N"` (`eval/subsets.select_states`) |
| eval backend | `load_backend` via bridge | real modules | `eval.backend.load_backend` -> `model.checkpoint.load_checkpoint(config_factory=TrainConfig.from_legacy_flat)`; `WmsBackend.config` is the checkpoint's flat dict verbatim |
| physics of a checkpoint | embedded / sibling `physics.json` | C1/R3 checkpoints carry no physics; their receipt does | `load_checkpoint` also reads `TRAIN_RECEIPT.json["physics"]` next to the file; `WmSTrainer.fit` writes `physics.json` at start so stage checkpoints are scoreable mid-run |
| sim layer access | `sim_bridge` only | `train/cav_drive.py` imported `traffic_shepherd` itself | `cav_drive.load_controller_dependencies` goes through `sim_bridge` (PI classes only loaded when a PI id occurs) |
| layer rule exceptions | none | 3 narrow edges | `tests/unit/test_import_graph.py::EXCEPTIONS`: `data/corpus.py -> legacy.vendor` (vendored pack reader adapter), `data/pack_history.py -> model.entries` (lazy entries-v2 features), `model/factory.py -> legacy.blackbox` (lazy black-box baselines); a stale exception fails a test |
| `r3.toml` | one file | `r3_packs_only.toml` (base) + `r3.toml` (adds `[data.switching]`); resolved config of `r3.toml` is unchanged | the switching pack is not available on every machine; smoke / golden configs need an R3 without it |
| `wms` name clash | shim + legacy tests | root `tests/test_wms_eval_*.py` and `test_wms_decide.py` import the LEGACY flat modules (`from wms.driver_sidecar import ...`) | cannot share a process with `src/wms`; `tests/conftest.py` runs them as their own group (`WMS_TEST_GROUP=legacy`), the default group skips them with a header note |

## Tolerances and exceptions in the golden comparison (`tests/golden/test_golden_new.py`)

The table `golden_compare.TOLERANCES` is unchanged. Four subsets are treated separately, each with a reason (item 3 is the only place a tolerance is looser than the table):

1. `selection.json`: the `window_filter.window_list` / `exclude_window_list` strings are absolute paths of the
   capturing worktree; compared by file name. Kept keys, per-pack counts (911/151; pack_v3 294/47), order: exact.
2. `forward_c1.json`: `state_checksum.per_tensor_sum` / `per_tensor_abs_sum` of `driver_encoder.*` are skipped.
   The C1 checkpoint predates the driver encoder; those tensors keep their *unseeded random construction values*
   in the legacy loader and in the new one, and residual-mode rollouts never read them. The assertion that the
   unstored set is exactly `{driver_encoder.}` is in the test; every stored tensor is compared at 1e-6. The rollout
   arrays (`forward_c1.npz`) match at the stored tolerance.
3. `train3_final_params.npz` / `train3_r3.json`: the three parameters `residual_net.blocks.{0,1,2}.attn.bias.2.bias`
   are checked at 1e-4 instead of 1e-5. They are the last bias of the pair-bias MLP, i.e. a constant added to every
   logit of a softmax row: a null direction whose true gradient is exactly 0, so autograd returns ~1e-9 rounding noise
   and Adam (`g / (|g| + 1e-8)`) turns the noise into a step. Observed max difference 1.4e-5; every other parameter
   agrees to 1e-5 or better.
4. Hash keys (`checksum`, `sha256_float32`, `init_checksum`) are compared only with `STRICT_HASHES=1` (golden README rule).
   The initial R3 parameters (`r3_init_state.npz`, tolerance 0) are bit-identical, so the construction/RNG order of the
   rewritten model is the same as the legacy model.

`tests/golden/test_golden_current.py` (re-capture from the legacy code and compare) was renamed to
`tests/golden/recapture_legacy_check.py`: the legacy code moved to `archive/wms_era/` (WP-F) and importing
`capture_golden` prepends the legacy tree to `sys.path`, which would shadow `src/wms`. `capture_golden.py` stays as the
record of how `data/` was produced (run it from commit bf7d4cb).

## Equivalence-test loaders (WP-F paths)

Legacy modules are imported through `archive/wms_era/wm_n1/wms` (spec-from-file alias `wms_legacy` in
`tests/equivalence/legacy.py`) or the dotted shim `output.wd01.wm_n1.wms`; the sibling legacy packages (`hdv`, `features`,
...) still live in `output/wd01/wm_n1`, which is only ever APPENDED to `sys.path`. `tests/unit/train/_compat.py` no longer
installs any stand-in module (B/C/D are real); it keeps `legacy()` and `FlatConfig`. `tests/support/legacy_eval_bridge.py`
remains for reference but is unused: the eval equivalence test drives the real `load_backend`.
