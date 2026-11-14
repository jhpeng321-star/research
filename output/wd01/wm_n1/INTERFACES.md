# WM-N1 INTERFACES (frozen contract for parallel pieces)

Integration branch `zcode/wmn1int`.  Frozen after `--no-ff` merges of
`zcode/wmn1eval` (1ce7f7d), `zcode/wmn1oof` (67436c7) and `zcode/wmn1inv`
(42f7e6d) on top of `zcode/wmn1model` 43a3e06 / features d20120f.

This file documents **what exists** and the **stubs** the next pieces must
implement.  It does not redesign.  Design: `WM_N1_DESIGN.md`; frozen
registration: `REGISTRATION_WM_N1.md` (LF-sha256 `6662…3246`); training order
amendment: `REGISTRATION_WM_N1_AMENDMENT_1.md` (commit 79e686e);
chunk-5 guard sha: `f8c7…b31e`.

## Tensors (frozen; copied from `features/FORMAT.md` and the writer)

Per run = one `(chunk/world/arm)` shard directory.  `DT = 0.5 s`, `K = 128`,
`W = 4`.  Written by `features/writer.write_shard`; the model
(`model/constants.py`) never imports the featurizer at runtime — the equality
is asserted by `model/tests/test_contract.py`.

| array | dtype | shape | meaning |
| --- | --- | --- | --- |
| `veh_step` | float16 | `[N_ticks, 128, 16]` | vehicle-step features (indices below) |
| `veh_mask` | uint8 | `[N_ticks, 128]` | 1 = slot occupied |
| `slot_index` | int32 | `[N_ticks, 128]` | index into `slot_vid` or -1 |
| `plan` | float16 | `[N_dec=20, 128, 19, 8]` | per-CAV 30 s action segments |
| `decision_t` | float64 | `[20]` | `t0 + 30k` |
| `wall` | float16 | `[N_ticks, 4, 11]` | wall tokens |
| `wall_mask` | uint8 | `[N_ticks, 4]` | active wall mask |
| `counter` | float16 | `[N_ticks, 2, 30]` | entrance counts, 10 s bins, 300 s |
| `accel_next` | float16 | `[N_ticks, 128]` | step target |
| `target_mask` | uint8 | `[N_ticks, 128]` | same vehicle present next tick |
| `lane_next` | int8 | `[N_ticks, 128]` | {0 stay, 1 left, 2 right, -1 none} |
| `arrivals` | int16 | `[N_ticks, 2]` | departures per entrance per tick |

`veh_step` index map (`features/FORMAT.md` §2 = `features/constants.py`):
0 `s_down_m`, 1 `lane_main_left`, 2 `lane_main_right`, 3 `lane_accel`,
4 `v_mps`, 5 `a_mps2`, 6 `nominal_length_m`, 7 `is_cav`, 8-12 action one-hot
`{native,damping,pi,wall,unknown}`, 13 `wall_member`, 14 `junction`,
15 `present`.  `plan` slots: 0-4 action one-hot, 5 `wall_id_index`,
6 `wall_speed/40`, 7 `planned_hold/600`.  `wall` fields 0-4 phase one-hot,
5 speed/40, 6 time-in-phase/600, 7 hold/600, 8 n_members/4, 9 mean s_down/1000,
10 lane span/2.  Sidecars: `slot_vid.json`, `meta.json`, `outcomes.parquet`,
`DONE.json` (input + output shas); tree index `INDEX.json` keys runs by
`<chunk>/<world>__<arm>` and groups siblings by `pair_key` =
`<world>@t0=<t0>`.

## Existing APIs (`file:line`)

**`features/`** (frozen; do not edit).
`writer.expected_bytes(n_ticks, n_slots, n_dec=20, has_outcomes=True) -> int`
(writer.py:43); `writer.write_shard(shard, data_root, *, force=False) ->
{"record", "status"}` (writer.py:84); `writer.load_index(data_root) -> dict`
(writer.py:162); `writer.pair_key(world, t0_s) -> str` (writer.py:169);
`writer.update_index` (writer.py:173); `writer.verify_index(data_root, index)
-> {"ok","n_runs","errors"}` (writer.py:216).  CLI: `python -m features
{featurize,estimate,verify}` (cli.py).  Guard: `splits.decide_split`,
`splits.run_dir`, `splits.SplitRefusal` (splits.py:107,151,56).

**`model/`** (e1, no training).  `net.WmN1Config(d_model=256,n_heads=4,
n_layers=4,ffn=768,n_mix=3,radius_m=150.0,bias_hidden=64,counter_channels=128,
dropout=0.0)` (net.py:32); `net.WmN1Net.forward(veh_step, veh_mask, plan, wall,
wall_mask, counter) -> dict` with keys `slot_hidden [B,K,256]`, `global_token
[B,256]`, `step` (GMM accel `accel_mu/accel_log_sigma/accel_mix_logits [B,K,3]`
+ `lane_logits [B,K,3]`), `arrival` (`count_mu/count_log_alpha [B,2]`,
`speed_mu/speed_log_sigma [B,2]`), `outcome_quantiles [B,5,3,19]`,
`collision_logits [B,5]` (net.py:333-359).  `net.build_member(seed,cfg)`,
`net.parameter_count` (net.py:363-371).  `losses`: `gmm_accel_nll`,
`lane_change_nll`, `neg_binom_nll`, `insertion_speed_nll`, `arrival_loss`,
`pinball`, `outcome_pinball`, `outcome_crps`, `collision_bce`, `contrast_loss`
(losses.py:116), `total_loss(pred, targets, weights=None)` (losses.py:144).
`rollout.rollout(net, batch, *, n_steps, n_samples=1, seed=None, generator=None)
-> {positions,speeds,lanes,masks,accel,arrivals,insertions,validity}`
(rollout.py:277); `rollout.project_step` (rollout.py:68); `ValidityCounts`
(rollout.py:31).  `ensemble.WmN1Ensemble(cfg, seeds=DEFAULT_SEEDS=(101,202,303,
404,505))` with `.mixture_quantiles`, `.mixture_collision_logit`,
`.mixture_predictive` (ensemble.py:18-65).

**`evalh/`** (CPU, synthetic).  `protocol.Predictor` protocol: attributes
`name: str`, `quantile_levels: np.ndarray`; methods
`predict_quantiles(rows: Sequence[OutcomeRow]) -> np.ndarray [len(rows), 19]`
and `sample_rollouts(rows, horizons_s=HORIZONS_S) -> RolloutBatch`
(protocol.py:100-118).  `PredictorBase` supplies defaults + `_validate`
(protocol.py:121).  `OutcomeRow`/`RolloutRow`/`RolloutBatch` dataclasses
(protocol.py:36-96); `QUANTILE_LEVELS` 0.05..0.95, `GATING_CONTRASTS`,
`GATING_TAUS`, `HORIZONS_S` (protocol.py:25-33).
`run_eval.main(argv)` / `run(args)` writes `rows.csv` before aggregation then
`REPORT.json`/`REPORT.md` (run_eval.py:134,227); parser `--split
{train_oof,cal,final_test} --out --predictor mod:attr --reference mod:attr
--source --final-test --registration-sha --chunk5-sha` (run_eval.py:204).
`timing_gate.run_gate(cfg, *, device, tiny, budget_hours_max=50,
gate_limit_hours=1, ...) -> TIMING_GATE.json dict` (timing_gate.py:221);
`BUDGET_HOURS_MAX=50`, `GATE_LIMIT_HOURS=1`, `ROLLOUT_FRACTION=0.30`,
`EVAL_HOURS=2` (timing_gate.py:30-33).  Seals: `seals.SPLITS`, `guard_split`,
`assert_holdout` (seals.py:44,169).

**`combine/`** (pure CPU).  `crps.QuantilePred(levels, qvals)` (crps.py:42),
`crps_pinball`, `crps_cdf`, `mixture_quantiles`, `cdf_distance` (crps.py:122-179).
`oof_tree.build_oof_predictions(rows, model_factory, ds_builder=None, *, k=5,
...)` (oof_tree.py:118); `OofResult.save(prefix)` (oof_tree.py:93);
`load_oof(prefix)` (oof_tree.py:228); `RowTableDataset` rows have `world`,
`split`, `tau`, `target`, `policy`, `row_id` (oof_tree.py:36-54).
`stacking.fit_stack(fn, ft, y, worlds, splits, *, levels, folds=2) -> StackModel`
(stacking.py:174); `cross_fit_pool`, `conformal_shifts`, `fit_conformal_pool`;
`DEFAULT_QUANTILES = (0.05,0.1,0.25,0.5,0.75,0.9,0.95)` (stacking.py:27).
`disagreement(fn, ft, *, x_grid=None, scale=...)` (disagreement.py:19).
`select_variant.select_planner_variant(*, crps, coverage, ...)` (select_variant.py:68);
`write_planner_variant(path, record)` (select_variant.py:105).  CLI:
`python -m combine {version,synthetic}` (cli.py).

**`inventory/`** (metadata only).  `inventory.parquet` columns = the dict from
`inventory_lib.read_run` (inventory_lib.py:352-378): `batch, chunk, world,
split, root_rel, loading_veh_h, penetration, t0_s, arm, policy, policy_source,
is_fleet, wm72_relabel, schedule_arm, n_schedule_events, duration_s, present,
receipt_status, is_complete, physics_rows_written, physics_file_bytes,
saved_state_id, is_smoke, mode, run`.  `pairs.parquet` columns = `build_pairs`
(inventory_lib.py:443-454): `world, t0_s, pair_key, category, native_run,
native_batch, other_run, other_batch, other_policy, same_base_state`.  Summary:
2114 TRAIN / 311 CAL runs, 2 629 082 ticks, 1629 pairs.  CLI:
`python build_inventory.py` (build_inventory.py:169).

## New stub APIs (piece-owned; signatures + docstrings only)

All new modules have `NotImplementedError` bodies and are importable
(`import train`, `corpus`, `adapter`, `bundle`).

**`train/`** — `config.TrainConfig` dataclass (defaults in the Config-keys
table), `TrainConfig.load(path)` / `.save(path)` / `.to_dict()`;
`dataset.MemmapCorpus(pack_dir, split="TRAIN", *, pair_sampling=False)` with
`__len__`, `__getitem__(i) -> dict` (six tensors + step targets + outcomes),
`sibling_pairs(i) -> (a, b)` (same `(world, t0)`, different arms, for
`losses.contrast_loss`), `worlds() -> list[str]`;
`trainer.Trainer(config, corpus, *, member, seed, device=None)` with
`fit() -> dict`, `save_checkpoint(path)`, `resume(path)`, `evaluate_cal()`;
`cli.main(argv)` with `--member {1..5} --seed --config --corpus --out
--resume --device`, `cli.record_seed(out_dir, member, seed)`.

**`corpus/`** — `featurize_all.select_runs(inventory_parquet, *, data_root,
splits)` and `featurize_all(inventory_parquet, data_root, out_root, *,
workers=4, force=False) -> dict`; `pack.pack_shards(data_root, out_dir, *,
workers=4) -> dict`, `pack.convert(in_dir, out_dir, *, workers=4) -> dict`;
`verify.verify_pack(pack_dir) -> {"ok","n_runs","errors"}`;
`manifest.build_upload_manifest(pack_dir, *, exclusions_path=None,
tools_dir=None) -> dict` and `write_manifest(manifest, path) -> sha256`
(`ManifestRefusal` on HOLDOUT / excluded / chunk-5); `cli.main` with
`featurize-all|pack|verify|manifest`.

**`adapter/`** — `predictor.WmN1Predictor(checkpoint=None, corpus=None, *,
device="cuda")` extending `evalh.protocol.PredictorBase`, with
`from_checkpoint(checkpoint, corpus=None, *, device) -> WmN1Predictor`,
`predict_quantiles(rows) -> [N, 19]`, `sample_rollouts(rows, horizons_s=None)
-> RolloutBatch`; `export.export_cal(predictor, rows, out_prefix, *,
levels=None) -> dict` and `export_oof(predictor, rows, out_prefix, *,
folds=5, levels=None) -> dict` producing `q<level>` rows for
`combine.stacking` / `combine.oof_tree`.

**`bundle/`** — `approval.approval_present(path) -> bool`,
`require_approval(path, *, stage) -> dict` (`ApprovalMissing`);
`queue.STAGE_ORDER = ("verify-upload","convert+verify","timing-gate",
"approval","member-1","members-2-5")`, `bundle_queue(pack_manifest, work_dir,
approval_path, *, device="cuda", stop_after=None) -> dict`, `run_stage(...)`;
`cli.main` with `--manifest --work-dir --approval --stop-after --device`.
Stage 3 (timing gate) is a hard STOP; no member trains before stage 4.

## Config keys (defaults from `WM_N1_DESIGN.md` §3, §7; `model.net.WmN1Config`)

`d_model=256, n_heads=4, n_layers=4, ffn=768, n_mix=3, radius_m=150.0,
bias_hidden=64, counter_channels=128, dropout=0.0` (architecture);
`n_members=5, seeds=(101,202,303,404,505)` (design §3 ensemble; seeds recorded
before training per amendment A1); `epochs=15, steps_per_epoch=2000,
batch_size=8, lr=1e-3, weight_decay=0.0, grad_clip=1.0` (design §7 +
`evalh.timing_gate.MemberConfig`); `lambda_contrast=1.0, w_step=1.0, w_lane=1.0,
w_arrival=1.0, w_outcome=1.0, w_collision=1.0` (design §3 / `losses.
DEFAULT_WEIGHTS`); `rollout_horizon_steps=20, rollout_fraction=0.30,
scheduled_sampling=0.0` (design §3b/§7); `early_stop_patience=3,
calibrate_on_cal=False, device="cuda", out_dir, resume` (design §3 selection).
A YAML/JSON file overrides any subset.

## File ownership per piece (disjoint)

| piece | owned paths | depends on |
| --- | --- | --- |
| FEATURES | `output/wd01/wm_n1/features/**` (frozen) | — |
| MODEL | `output/wd01/wm_n1/model/**` (frozen e1) | FEATURES |
| EVALH | `output/wd01/wm_n1/evalh/**` (frozen) | MODEL (protocol only) |
| COMBINE | `output/wd01/wm_n1/combine/**` (frozen) | MODEL/EVALH (protocol) |
| INV | `output/wd01/wm_n1/inventory/**` (frozen) | — |
| TRAIN | `output/wd01/wm_n1/train/**` | CORPUS, MODEL |
| CORPUS | `output/wd01/wm_n1/corpus/**` | INV, FEATURES |
| ADAPTER | `output/wd01/wm_n1/adapter/**` | TRAIN, EVALH, COMBINE |
| BUNDLE | `output/wd01/wm_n1/bundle/**` | CORPUS, TRAIN, EVALH |
| INTEGRATOR | `output/wd01/wm_n1/INTERFACES.md`, ledger/registration records | all |

Each piece owns its own `tests/` directory.  No piece edits another's; shared
records are owned by the integrator.

## Mismatches (`file:line`)

1. **Quantile levels 7 vs 19.** `combine.stacking.DEFAULT_QUANTILES` has 7
   levels (stacking.py:27, Part B), but the neural/evalh heads use 19
   (`model/constants.py:61`, `evalh/protocol.py:25`).  The C variant cannot
   pool until `adapter.export` maps the 19-level neural quantile curve onto the
   7 levels.  Not fixed (needs the adapter).
2. **Timing gate measures a toy, not WM-N1.** `evalh/timing_gate.py:82-102`
   builds an ad-hoc `Toy` Transformer with `MemberConfig.d_in=16,
   n_layers/...` (timing_gate.py:39-56), not `model.net.WmN1Net` /
   `WmN1Config`.  The gate cannot yet bound the real member's cost; BUNDLE
   must wire the real model (or record the proxy explicitly).
3. **Outcome target names.** `model/constants.py:57` `OUTCOME_TARGETS =
   ("delay_s","ttc_lt3_s","hard_brake_s")` vs the v0.4a label columns
   `total_delay_ref_s`, `ttc_lt3_veh_s`, `hard_brake_s`
   (`features/outcomes.py:23-28`).  No code maps the shard `outcomes.parquet`
   to the model target tensors; TRAIN must define it.
4. **Rollout shape.** `model.rollout` returns `positions/... [n_samples,
   n_steps+1, B, K]` (rollout.py:277-304) while
   `evalh.protocol.RolloutBatch.pos/speed` is `[N, H, V]` (protocol.py:90-96).
   `adapter.WmN1Predictor.sample_rollouts` must collapse sample/batch/steps to
   the harness shape.
5. **No real EvalSource.** `evalh/sources.py` only ships `SyntheticSource`
   (sources.py:42); `run_eval` needs a source that reads real rows.  CORPUS /
   ADAPTER must add one; none exists yet.
6. **Inventory micro-sample layout.** `inventory_lib._featurize`
   (inventory_lib.py:458-479) writes a different index order (e.g. idx3 =
   `is_cav`, idx4 = length) than `features/FORMAT.md:61-79`.  It is a
   byte-count-only micro-sample and never feeds the model; left as is.

## Torch-2.5 notes

The GPU box pins `torch>=2.7,<2.13` from the CUDA 12.8 (cu128) wheel index
(Blackwell-class sm_120; local dev box: 2.12.1+cpu, Python 3.10/3.11).  The
"2.5" in this heading is historical.  A grep over `model/` and `evalh/` finds
**no API newer than 2.5 and none deprecated across 2.7-2.12**: only
`nn.{Linear,LayerNorm,GELU,Conv1d,ModuleList,TransformerEncoder,
TransformerEncoderLayer}`, `F.{softmax,log_softmax,cross_entropy,
binary_cross_entropy_with_logits,logsumexp,lgamma,softplus}`, `torch.{eye,
nan_to_num,gather,logit,Generator,rand,randn,manual_seed,stack,where,clamp}`.
No `torch.compile`, `torch.cuda.amp`/`autocast` (deprecated; use `torch.amp`),
`scaled_dot_product_attention`, `flex_attention` or `torch.export`.  No
`torch.save`/`torch.load` exists yet: the TRAIN/ADAPTER checkpoint loaders must
set `torch.load(..., weights_only=…)` explicitly so every supported Torch
(default `True`) agrees — save plain `state_dict`s + JSON config where
possible.  Rollout randomness uses CPU `torch.Generator` +
`numpy.random.default_rng`, so it is device independent.
