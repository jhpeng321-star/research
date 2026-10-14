# INFRAV5: non-lane v5 world-model fixes (2026-10-14)

Branch `c4/infrav5` (from `c4/int` 1b249be). Code, tests and a CPU smoke only: no GPU training, no
SUMO. Every switch defaults OFF, and the off path is bit-identical (tests below). The lane model is
rebuilt separately on `c4/lanev5`; no lane files were touched.

Evidence: `E:/agent-work/tmp/v4speed/`, `E:/agent-work/tmp/MORNING_WM_DIAG_NOTES.md` ("v4 speed
bias root cause", "v4 stage-30 diagnosis"), and the v4_cont `RUN_SUMMARY.json`.

## 1. Selection fix: `curriculum.selection = "multi_horizon_v2"`

**The v4 defect.** `multi_horizon` scored each validation event over the horizons that event had.
- v4's 60 s-stage events had 10/30/60 s.
- The final 120 s-stage event also had 120 s, normalised by the run's own first 120 s validation.
  That is the same event, so its 120 s ratio was exactly 1.0.
- So the step-200 checkpoint (0.829) beat the final one (0.880).
- On unseen walls the final checkpoint is far better (60/120 s: 12.56/71.7 vs 18.10/142.4).

**v2** (`wms.train.selection.SameHorizonSelection`, wired in `WmSTrainer.fit`):
- **Online, each validation event.** The score is the mean of position/reference over the common
  horizons:
  - Common horizons = selection horizons <= the first eligible stage. Every eligible event
    validates them (v5: 10/30/60 s).
  - Only fixed references are used.
  - This tracks the best event of each eligible stage and the last event of each eligible stage
    (CPU weight snapshots, at most 2 per stage). These are the candidates.
  - A provisional `wm_s_best.pt` is written, so a run that is cut short still has one.
- **End of run** (`_final_same_horizon`):
  - Every candidate is completed on the full horizon set F (the horizons of the last eligible
    event, v5: 10/30/60/120 s). Missing horizons are re-validated on the same fixed
    `val_max_windows` subset (rows marked `revalidated` in `val_log.jsonl`).
  - Score = mean over F of position_h / norm_h. norm_h = the fixed reference if > 0, else the best
    candidate's value at h. The run's own first validation is never used.
  - The lowest score wins; ties go to the earlier candidate.
  - `wm_s_best.pt` is rewritten with that candidate's weights, and the model is restored to its
    final weights.
  - The full table (online events + `final_comparison`, with per-candidate positions, ratios,
    re-validated horizons and the selected flag) goes to TRAIN_RECEIPT
    (`long_horizon_selection.multi_horizon_v2`) and RUN_SUMMARY (`selection`).
- **Requirements.** `select_by_long_horizon` and `val_all_horizons`.
- **Cost in v5.** The two 60 s-stage candidates are re-validated at 120 s, about 15 min on gpuW.

**On the v4_cont log** (test `test_v2_on_the_v4_log_...`):
- Online, on 10/30/60 s only, step 200 still leads: 0.829 vs 0.840 for the final checkpoint.
- So the deciding information is the 120 s value of the 60 s-stage checkpoints. v4 never measured
  it on validation.
- With a synthetic re-validated 120 s of 210 m (walls: ~2x the final's), v2 picks the final
  checkpoint. If the 60 s checkpoint were as good at 120 s, v2 would pick it.

## 2. Jam oversampling and the C4 / GENRAND sources

**Coverage: yes, already.**
- With `data.jam.oversample > 1`, every path flags jam windows per source:
  - `mix_datasets` (switching)
  - `JamOversampledDataset`, which is the v4/v5 path because DATAHYG disables switching
  - `add_c4_source` (`c4_episodes`)
  - `add_c4_sources` (`data.c4_sources`, GENRAND plain + branch ordinary windows)
- `MixedTrajectoryDataset.stage_weights` multiplies jam windows inside each source and keeps the
  source shares.
- Paired batches (`PairedBatchDataset.paired_batch`) draw pair groups directly and do not see the
  weights. Only which batch becomes paired depends on the sampled indices, and the paired fraction
  is unchanged.
- No code change was needed. Tests: `test_jam_oversampling_covers_c4_sources`,
  `test_paired_batches_are_unaffected_by_jam_weights`.

**New receipt fields** (only when jam is on, so off receipts are unchanged):
- `stage_sampling[].sources[*].jam_fraction_sampled_before` / `jam_fraction_sampled`
- `stage_sampling[].jam_share_total.{before, after}`

**Finding: the legacy jam rule saturates.**
- The legacy rule flags a window when any HDV observation in it is below 5 m/s. That flags 96-97 %
  of the native TRAIN windows: ramp mergers stop at the end of the acceleration lane, and the
  median window has 31 % of its HDV observations below 5 m/s.
- So factor 2 barely changes anything. Local TRAIN, packs + GENRAND smoke, mix 0.80 / 0.15 / 0.05:

| | jam share of draws, factor 1 | jam share of draws, factor 2 |
|---|---|---|
| native packs | 96.1 % | 98.0 % |
| GENRAND smoke | 100 % | 100 % |
| total | 96.8 % | 98.4 % |

  The full GENRAND set is only on gpuW. The receipt reports the real shares at launch.

**Added switch `data.jam.min_fraction`** (default 0 = legacy rule, bit-identical flags and cache
bytes):
- A window is a jam window when more than this fraction of its HDV observations are below
  `speed_mps`. The cache records the rule.
- Measured with 0.3 (same local TRAIN): 312 of 583 native windows flagged (53.5 %). Factor 2 moves
  native draws 53.5 % -> 69.7 % jam and all ordinary draws 57.8 % -> 70.8 %. On the smoke, GENRAND
  plain is 19/19 flagged and branch 0/1.
- **Integrator decision.** `configs/c4_wm_v5_base.toml` keeps 0 (as specified) and has the 0.3 line
  commented. `E:/agent-work/tmp/infrav5_scratch/jam_mf03/` holds the measured shares with 0.3.

## 3. Residual magnitude

- `loss.residual.residual_mag_weight` (flat `residual_mag_weight`) is an L2 penalty on the raw,
  pre-gate learned residual (`ResidualMagnitudeLoss`).
- In hybrid mode it is built next to the hybrid `ResidualPenalty` (`hybrid_magnitude_weight`,
  default 1e-3).
- Raising it to 0.05 works: the term scales 5x and the forward pass is unchanged (test). No code
  change.

## 4. Optional driver-estimate prior: `loss.driver_prior` (OWNER DECISION, default off)

**Definition.**
- prior = weight × Σ_p ((mean_batch(x_p) − target_p) / target_p)²
- p ∈ {accel, time headway, decel, minGap, v0 / speed limit}.
- x = the t0 encoder output over valid, observed t0 HDV slots of the batch.
- Only the batch mean is pulled. Every valid slot gets the same gradient, so zero-mean per-driver
  deviations leave the prior unchanged and per-driver identification stays free (tests).

**Gradient path.**
- The prior's gradient w.r.t. the t0 encoder-output leaf is added before the single push through
  the encoder graph. It therefore reaches only `driver_encoder.*`; the gradients of all other
  parameters are bit-identical (test).
- Requires `curriculum.driver_grad_across_chunks` and an encoder-driven rollout mode (config
  validation).

**Logging.** `driver_prior` (added to `loss`), `driver_prior_means`, and
`driver_prior_grad_share` = ‖g_prior‖ / ‖g_total‖ at the encoder output.

**Targets** (`loss.driver_prior.target`):
- `"explicit"`: the five values `accel`, `time_headway`, `decel`, `min_gap`, `v0_ratio`.
- `"legacy"`: the legacy SUMO population (`scenarios/idm/driver_routes.rou.xml`, normal 70 % /
  aggressive 10 % / cautious 20 %). Every median is the normal_car value: a 2.6, T 1.1, b 4.5,
  s0 2.5, v0 ratio 1.0. The means would be 2.52 / 1.165 / 4.2 / 2.6 / 0.992.
- `"textbook"`: c4/popfix `TEXTBOOK_POPULATION_MEDIANS`: a 1.2, T 1.2, b 2.0, s0 2.0, ratio 1.0.
  This is a local copy with a TODO to import it once `traffic_shepherd/c4/population.py` is merged;
  a test compares the two when the module is importable.

v5 sets weight 0 and target "legacy" (the v5 data are legacy-population data). For example,
weight 0.05 turns it on.

## 5. CAV execution cap with known CAV settings: `cav_exec_cap = "idm_normal[:table]"`

**What it does.**
- `WmSModel._exec_cap_normal` uses the EXECCAP ceiling with the same physics-core IDM: leader
  search, accel-lane-end obstacle and plant bounds.
- The ceiling is computed with fixed CAV vType settings: min(IDM, (v0 − v)/dt, accel), where v0 =
  speedFactor × speed limit. It replaces the WM-inferred settings.
- Tables (`wms.model.cav.CAV_SETTINGS_TABLES`):
  - `textbook`: `cav_typical`, a 1.2, b 2.0, T 1.2, s0 2.0, speedFactor 1.0. Default for a bare
    `"idm_normal"`. Local copy with a TODO to import popfix `TEXTBOOK_CAV_PARAMS`.
  - `legacy`: normal_car, a 2.6, b 4.5, T 1.1, s0 2.5, speedFactor 1.0.

**What it leaves alone.** HDVs and recorded teachers. Only command executors apply the cap, as
with "idm" (tests).

**Wiring.** It goes through the existing switch paths: `[trainer] cav_exec_cap` (DreamerConfig),
`WMS_CAV_EXEC_CAP` (`load_backend`), `deploy --cav-exec-cap` (choices extended) and
`RsuChecker(cav_exec_cap=)`.

**Config.** `configs/c4_policy_final.toml` now sets `"idm_normal:textbook"`. Use
`"idm_normal:legacy"` if the final policy runs against the legacy SUMO population.

## 6. `configs/c4_wm_v5_base.toml`

**Changes from v4:**
- extends v4
- stage_steps [1200, 900, 800, 250]
- mixed_horizon_frac 0
- lr_per_stage [1e-3, 3e-4, 3e-4, 1e-4]
- val_max_windows 512
- selection multi_horizon_v2 (val_all_horizons and selection_min_stage_s 60 inherited)
- data.jam.oversample 2.0
- genrand_plain mix_ratio 0.15 (branch source unchanged)
- residual_mag_weight 0.05
- driver_prior weight 0 / target legacy
- loss.paired_diff weight 2.0 / fraction 0.25, as the integrator asked. The reason given was "v4 ~2 %
  of the logged loss; target ~10 % on paired batches", but the premise does not hold:
  - In v4's `train_log.jsonl`, `paired_value_share` on PAIRED batches is already 8.5 / 9.2 / 9.4 %
    at the 10 / 30 / 60 s stages. The ~2 % is the mean over all batches, of which 20 % are paired.
  - Weight 2.0 therefore means ~28 % on paired batches (~7 % overall). The CPU smoke gives
    25-32 % at 2.0 vs 8.5-11 % at 0.5.
  - For ~10 % on paired batches, use a weight of ~0.5-0.6 (integrator decision).
- out_dir `runs/active/c4_wm_v5_base`
- max_hours 10.5

**Estimated wall time on gpuW** (measured v4 pure-horizon step times 2.45 / 7.31 / 15.0 / 34.7 s):
- training ≈ 8.4 h
- validation ≈ 1.3 h (32 batches per horizon, val_every 200, plus re-validation and startup)
- total ≈ 9.9 h, which is over the 9 h target

**Proposed cuts (not applied):**
- val_every 200 → 400: −0.5 h
- 60 s stage 800 → 650: −0.6 h
- together → about 8.8 h
- alternative: 120 s stage 250 → 200, −0.5 h

## Verification

- `tests/unit/test_infrav5.py` (new): off = bit-identical (flat forms, one training step, a real
  fit); v2 on the v4 log; fake-fit candidate re-validation and weight restore; a real end-to-end v2
  fit; jam coverage of the C4 sources at factor 1 and 2; paired batches unaffected; `min_fraction`
  rule and cache; the residual-mag key; the prior (formula, batch-mean-only gradient,
  encoder-only, targets); `idm_normal` tables, HDVs and teachers untouched; wiring.
- `tests/unit/test_execcap.py`: the final-config expectation is now `"idm_normal:textbook"`.
- Suites run with OMP_NUM_THREADS=1: test_infrav5, test_sched, test_pairloss, test_execcap,
  test_wmloss, test_c4_core, test_c4_dreamer, test_import_graph, unit/config, unit/data, unit/train.
  Result: everything passes except `data/test_drive_registry.py::test_committed_registry_matches_the_generator_output`.
  That failure is unrelated: data_ops registry text vs the generator, a file not touched here.
- CPU smoke (`E:/agent-work/tmp/infrav5_scratch/run_smoke.py`, as v4smoke but importing this
  worktree, 2 threads), data = local packs + GENRAND smoke:
  - 2 optimizer steps at 10 s, finite (loss 12.34 -> 12.37).
  - Paired batch at weight 2.0: paired_value_share 0.32 / 0.25.
  - Fwd/bwd profiles: 25.8 s for the 10 s unroll and 107.8 s for the 30 s unroll (laptop,
    throttled).
  - Results are in `E:/agent-work/tmp/infrav5_scratch/smoke2/`.
- Also run: test_c4_genrand, test_datahyg, test_entryfix, test_r5_switches, test_c4_deploy,
  test_c4_rsu: pass.
  - `test_junctionfix.py` fails/errors on the base too. It loads `configs/c4_wm_v4.toml`, which
    since 2ca1f8a needs the GENRAND data (not local) and differs from v3 by more than
    junction_fix. Unrelated.
