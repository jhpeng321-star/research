# CHANGES_V03_NET.md — B3b v0.3 net (BRIEF_B3_V03 items A1-A5, B6-B11, C13, D14)

2026-09-29, worktree `wd01w4` (branch `zcode/wd01w4`). Scope: trainer + model
code and CPU smoke tests only. ZERO simulation; no net training beyond tiny
CPU smoke runs. Ported from the cancelled wd01w3/wd01w attempts, reviewed and
corrected (deviations listed at the end).

## Item table

| Item | Where (file / function) | Test |
| --- | --- | --- |
| A1 per-tau target normalisation | `partB_v0/data.py` `fit_target_medians`, `MedTable` (exact piecewise-linear in tau), `V03TensorData.y_norm` / `med_row` | `tests/test_train_net.py::test_data_v03_normalisation_and_gather`; `tests/test_tau_conditioning.py` |
| A2 twohot grid = realistic range only | `partB_v0/model.py` `SYM_MIN_V03/SYM_MAX_V03` (-2..12 symlog, normalised), `twohot_encode_grid` / `twohot_encode_v03`, `PmfGrid` | `tests/test_model_v03.py::test_grid_covers_realistic_normalised_range_only`, `::test_twohot_v03_rows_sum_to_one_two_bins_and_clamp` |
| A3 tau conditioning in every head layer + per-tau output scale/bias | `partB_v0/model.py` `FiLMHead` (FiLM per hidden layer; `softplus(out_scale(tau))` and `out_bias(tau)`; Gaussian prior init at the typical normalised target) | `tests/test_model_v03.py::test_tau_and_policy_conditioning_change_the_head`; `tests/test_tau_conditioning.py` (< 5 % fit, monotone in tau) |
| A4 per-feature standardisation from TRAIN stats (saved with weights) + summary-feature token | `partB_v0/data.py` `summary_features_base` (exact diag_tau30 features), `fit_feature_stats`, `_apply_stats`, `fit_v03_stats` -> `stats.json`; `partB_v0/model.py` `summary_proj` (token type 9); `train_net.py` writes `stats.json` next to the weights | `test_data_v03_normalisation_and_gather` (unit variance on TRAIN, summary shape 40); `tests/test_cli_net.py` (stats.json ships + zipped) |
| A5 median point forecast | `partB_v0/model.py` `PmfGrid.quantile_torch(., 0.5)` / `PmfGrid.median`; `partB_v0/evaluate.py` net evaluation decodes the mixture median | `tests/test_model_v03.py::test_point_estimate_is_the_median_of_the_pmf` |
| B6 selection on TRAIN world-grouped OOB, all tau + both policies; eval every 500 steps, patience 3, max 8000; CAL never used | `partB_v0/train_net.py` `TrainNetConfig` (`eval_every_steps=500`, `patience=3`, `max_steps_cap=8000`), `data.set_members` (OOB row sets), `ConcurrentNetTrainer.eval_oob` (TRAIN OOB rows only, CRPS real units), `_eval_and_select` (best tracking + early stop; `m{k}_best.pt` persisted) | `test_oob_covers_all_tau_and_both_policies`, `test_oob_selection_never_touches_cal` (CAL guard: corrupting every CAL label leaves eval_oob bit-identical), `test_oob_early_stop_patience_and_best_write`, `test_eval_oob_uses_eval_mode` |
| B7 `--ablation` (base / wd / dropout 0.1 / d_model 128) on OOB | `partB_v0/train_net.py` `VARIANTS`, `run_ablation` (short runs, mean best-OOB-CRPS per variant, choice in `ablation.json`, artifacts under `out/ckpt/ablation/` so the bundle stays clean) | `test_ablation_runs_variants_and_chooses_by_oob` |
| B8 MC weight rising with tau; n-step on-grid targets; consistency penalty; no off-grid interpolation | `partB_v0/train_net.py` `mc_weight`, `_member_losses` (TD target = on-grid chain `sum(d30) + label(tau-30 grid point)` where feasible, else one-step bootstrap at the EXACT `tau-30` via continuous tau conditioning; `MedTable` gives exact normalisers at off-grid tau; huber consistency penalty `V(s,tau) ?~ D30 + V(s+30,tau-30)`); `partB_v0/data.py` `TD_CHAIN_STEPS`, `V03TensorData._build_td` | `test_mc_weight_rises_with_tau`, `test_td_chains_and_exact_normalisers` |
| B9 shared encoder over damping+native rows | `partB_v0/model.py` `V03ValueNet.encode` (policy is a conditioning token, not a split); trainer pools contain both sources; native metrics scored on native rows only at evaluation (`evaluate_net_v03` per-policy tables) | `test_data_v03_normalisation_and_gather` (both sources gathered); per-policy rows in the smoke `eval_report.md` (`test_cli_net.py`) |
| B10 M=5 concurrent members on one device + bitwise per-member resume | `partB_v0/train_net.py` `build_member_nets`, `stack_members` (`torch.func.stack_module_state` + vmap), `StackedAdamW` (elementwise, per-member active mask freezes early-stopped members bitwise), `ConcurrentNetTrainer.try_resume` (stacked params/moments/target/RNG/best states), `LossScaler` | `test_resume_is_bitwise_per_member` (8 straight == 3 + resume(5), BITWISE on params and AdamW m/v per member, and on the written `member_{k}.pt`); `tests/test_cli_net.py::test_net_smoke_resume_picks_up_checkpoint` |
| B11 per-member budget re-planned from measured ms/step (incl. eval/ckpt overhead); device-resident dataset; bf16 Ampere+ / fp16+GradScaler-equivalent on T4 | `partB_v0/train_net.py` `_plan_steps`, re-plan block in `train()` (cosine LR follows the re-planned total), `V03TensorData` (all blocks resident on the device), `resolve_dtype`, `_autocast_ctx`, `LossScaler` (manual scaling; the stacked optimizer is not a `torch.optim.Optimizer`) | `test_budget_replan_follows_measured_rate`; dtype resolution is exercised by `run_net_training` on CPU (fp32) |
| B12 persistence (brief item 12) | `m{k}_best.pt` + meta on EVERY new best; `member_{k}.pt` + sidecar from the best state on EVERY path (natural finish, skip path, budget stop — `write_members`); v0.3 loader `evaluate.load_net_v03` requires `stats.json`, format-gates member files | `test_oob_early_stop_patience_and_best_write`, `test_resume_with_smaller_max_steps_writes_all_members`, `test_cli_net.py` (interrupt + smaller `--max-steps` resume writes all members and re-evaluates) |
| C13 paired Delta head (CRN restore pairs, damping_fleet - switching_mix; v0.3.2 fix #13 relabel: the campaign '__native' arms are random per-CAV switching schedules, not a fleet-native policy) | `partB_v0/data.py` `build_delta_pairs` (t0 pairs, offset recorded, `pair_kind`, REP-P `label_fn` hook), `DeltaScale`; `partB_v0/model.py` `head_delta`; `partB_v0/train_net.py` pair batches (vmap-safe masked mean); `partB_v0/evaluate.py` `net_delta_pred` + item 5 (sign agreement, Spearman, MAE vs the delta=0 baseline) | `test_delta_pairs_offset_and_normalisation`, `test_delta_head_train_step_moves_prediction`, `tests/test_cli_net.py` (`item5_delta_head.n_pairs > 0`, report section 5) |
| D14 post-hoc per-tau calibration fitted on OOB | `partB_v0/calibration.py` (per-tau quantile SHIFT map, or ISOTONIC; `CalMap` json, coverage/CAL/PIT helpers); `partB_v0/evaluate.py` `fit_run_calibration` (pooled OOB only; OOB-empty fallback documented), CAL reported BEFORE/AFTER on CAL, never selected on | `tests/test_calibration.py` (5 tests: shift reduces CAL, isotonic monotone + calibrating, coverage/PIT helpers, json round-trip, thin-tau left raw); `tests/test_cli_net.py` (calibration.json + section 2 before/after) |
| BAR vs v0.3.1 selected deployable (task addition) | `partB_v0/evaluate.py` `V031_BAR_DIR`, `bar_section` (reads `metrics.json` of `E:/research-dense/output/partB_results/v0.3.1_selected_deployable`; absent -> "bar not available"); rendered as report section 6 | Present in this workspace (with `metrics.json`), so the smoke report renders the 5-tau comparison table; smoke numbers pass 0/5 as expected — the verdict comes from the real GPU run |
| D15/D16 permanent ridge/GBM baselines + HARD acceptance vs ridge in the NET report | `partB_v0/evaluate.py` `evaluate_net_v03` item 7 calls `gbm.baselines_section` (identical estimator as the v0.2/GBM report) and renders section 7/7a | `tests/test_cli_net.py` (full report renders); numbers only from the real (GPU) run — smoke runs are labelled smoke-only |

## Deferred (with reason)

- **Real acceptance numbers / GPU run.** Per the brief (item 16, addendum 7):
  smoke runs cannot show HARD acceptance; the trainer ships ready for the
  owner's GPU run which produces the verdict tables.
- **`make_v0_bundle.py --value-model` switch** (brief item 21): owned by B1 in
  `E:/rd-wt/wd01u`; not edited here. The v0.2 `train --value-model {net,gbm}`
  CLI is untouched; the v0.3 net is exposed as the separate `net` subcommand.
- **REP-P replicate-mean delta targets** (addendum 5 item 5 / addendum 6):
  hook only — `build_delta_pairs(label_fn=...)` rebuilds pair frames with
  replicate-mean labels when those data exist.
- **lambda-returns**: the plan's MC/TD mixing uses the tau-rising MC weight +
  n-step chains (addendum 4/5 wording); forward-view traces remain deferred
  as in v0.2.

## Deviations from the unverified reference (wd01w3 working tree)

The wd01w3 attempt was reviewed before porting; these defects were found and
fixed:

1. **`PmfGrid.crps_real` scaled the pairwise CRPS term by c^2.** CRPS is
   linear under joint scaling: CRPS(F*c, y*c) = c*(E|X-y| - 0.5 E|X-X'|).
   The c^2 version goes massively NEGATIVE at per-tau scales ~1e3 (measured
   -4e9 on the tiny fixture), which corrupts OOB selection and every report
   table. Fixed in `model.py`; regression test
   `test_crps_real_is_linear_in_scale_and_nonnegative` (spread pmfs, not just
   point masses — the old test's point mass could not catch it).
2. **`fit_run_calibration` raised on an empty pooled OOB set** (deterministic
   for members=1 on small data: the bootstrap can cover every TRAIN world).
   Now falls back to TRAIN-in-bag rows with the fit source recorded in
   `calibration.json` info; CAL is still never touched.
3. **Net eval report lacked the permanent ridge/GBM baselines and the HARD
   acceptance table** (task requirement); added as report section 7/7a via
   the same `baselines_section` used by the v0.2/GBM report.
4. Minor: timing probe now always measures >= 1 step (`max(3, probe_steps)`;
   the old `max(2, ...)` could leave zero measured steps and log NaN ms);
   lint cleanups; `WD_TEST_THREADS` cap (default 4) in the partB_v0 test
   conftest so the suites stay small on the box shared with the SUMO
   campaign (both resume comparisons run inside one process, so bitwise
   equality is unaffected).
5. **CLI wiring**: `_run_net` read `args.replan` but the flag is
   `--replan-after` (`args.replan_after`) — every `net` invocation crashed
   with AttributeError; fixed.
6. **`CalMap.apply` level mismatch**: the D14 report applied the map
   (fitted at 19 FIT_LEVELS) to a 9-column CAL_LEVELS quantile matrix ->
   broadcast ValueError; the report now applies the map at its own levels
   and reads the CAL_LEVELS columns back out.
7. **numpy/torch mixups in the net TD-consistency table**: `data.td[...]`
   is numpy; the report called `.to(torch.float64)/.cpu()` on it ->
   AttributeError; fixed with plain numpy indexing.
