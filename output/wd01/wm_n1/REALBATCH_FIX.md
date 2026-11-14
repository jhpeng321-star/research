# REALBATCH_FIX — real-corpus Idea A training path (Amendment 3 A3.2)

Date: 2026-10-01. Branch `zcode/m1realbatch` (base `zcode/wd01` a64438a).
Scope: `output/wd01/wm_n1/{train,bundle,preflight}`. No K_max/feature/pack
change; no re-featurization; synthetic tmp packs only; zero SUMO.

## What was missing

`train/idea_a.py:main()` trained through
`train.dataset.MemmapCorpus -> IdeaATrainer._iter_batches ->
batch_from_corpus_items`. That builder read only the frozen v1 tensors and:

* set `baseline_idm = zeros` — so the one-step HDV residual and the rollout
  residual were fitted against a **zero** IDM baseline, not `hdv_base[:,0]`;
* produced **no** `teacher [T+1,B,K,4]` / `rollout_accel` / `rollout_baseline_idm`
  / `rollout_mask`, no `s0/v0/a0/lane0/length0/present0` rollout world;
* produced **no** `summary_raw` (the CAV planned-motion summary for the P5
  correction head), no `event_tok`/`event_mask`;
* so `loss_terms()` silently skipped `rollout` and `correction_crps`, and the
  paired `response_contrast` had only its HDV sub-term. A real run trained on
  a reduced objective with no error and no receipt flag.

## What changed (all additive; frozen v1 surfaces preserved)

* `train.idea_a.IdeaACorpus` — a read-only view over a packed corpus that adds
  the five P9 channels (`veh_kin`, `hdv_base`, `cav_plan_ch`, `event_tok`,
  `event_mask`) and builds the **full canonical window** per item:
  `kin`/`baseline_idm`/`hdv_base` from the pack, `teacher` and rollout targets
  from the consecutive-tick windows of `veh_kin`/`accel_next` at the registered
  horizon, `summary_raw = corr.summary.cav_motion_summary` rebuilt from the
  CAV planned channels (`cav_plan_ch` speed/accel) and recorded positions, and
  the event tokens/masks. Time tensors are zero-padded to the registered
  horizon with `rollout_mask=False` so runs of different lifetimes collate.
  Construction **fails closed** when the pack lacks any Idea A channel.
* `build_idea_a_bundle` now composes the real P5 path (`CavMotionSummary` 16->64
  projection + `CorrectionHead`) so the canonical 16-d `summary_raw` is
  consumable (previously the raw 64-d head was built, which the batch contract
  could never feed).
* `IdeaATrainer.fit()` sets `_require_all_terms` on any corpus run unless
  `allow_incomplete_losses` is set: a missing registered term raises
  `IdeaAConfigError` instead of being skipped. `TRAIN_RECEIPT.json` now records
  `active_loss_terms`, `required_loss_terms` and `allow_incomplete_losses`.
* `main()`/`evaluate_cal()` open `IdeaACorpus`; the box CLI gained the
  TEST-ONLY `--allow-incomplete-losses` flag (recorded in the receipt).
* `bundle.e2e.write_synthetic_pack` writes the Idea A channels + `INDEX.json`;
  `run_e2e_smoke`/`run_config_job` train through `IdeaACorpus`.
* `bundle.timing.run_timing_gate` now measures the **real Idea A corpus
  member** (`measure_idea_a_corpus` / `loss_decrease_idea_a`) and keeps the v1
  `model.net.WmN1Net` as `v1_member`; `measured_model` records the real path.
* `train.idea_a.overfit_report` (pre-flight G4) now writes a tiny real-format
  pack (`write_g4_tiny_pack`) from the crafted windows and overfits the REAL
  corpus path (`corpus_overfit_probe`), i.e. through `IdeaACorpus`.

## Active loss terms

Registered A3.2 terms (`IdeaATrainConfig` weights all 1.0; `REQUIRED_LOSS_TERMS`):

```
residual_nll, rollout, arrival, correction_crps, response_contrast
```

Tiny-pack `main()`-path fit receipt (2 steps, real P2/P4/P5 bundle) shows
exactly those five active:
`active_loss_terms = ["arrival","correction_crps","residual_nll","response_contrast","rollout"]`,
`allow_incomplete_losses = false`.

## Per-step CPU time on the tiny synthetic pack

Measured with 1 CPU thread, real bundle (default config, batch 4, 20-step
rollout), 3 warmup + 20 timed steps, `bundle.e2e.write_synthetic_pack`:

| path | s/step | terms active |
| --- | --- | --- |
| before (`MemmapCorpus` + `batch_from_corpus_items`) | 0.148 | residual_nll, arrival, response_contrast (silent drop) |
| after (`IdeaACorpus` full canonical batch) | 1.177 | all five registered terms |

The ~8x is the real cost of the previously-dropped terms (20-step
differentiable rollout + P5 projection/CRPS + paired contrast), not a
regression of the one-step path.

## Review fixes (2026-10-01)

Four correctness fixes on top of the full-canonical batch path (branch
`zcode/m1realbatch2`; no K_max/pack change, no re-featurization).

1. **Cumulative rollout mask (slot reuse).** `IdeaACorpus.window()` built
   `rollout_mask = target_mask[tick:tick+h]` per step. `target_mask` is 1 iff
   the SAME vehicle is in the slot next tick (`features/FORMAT.md` §6), so a
   slot reused mid-window returned to 1 for the NEW vehicle while the rollout
   kept integrating the departed vehicle's state - the loss compared the wrong
   vehicle. The mask is now alive-since-t0:
   `alive_state[t] = present0 AND prod_{u<t} target_mask[tick+u]`, and
   `rollout_mask = alive_state[1:h+1]` (step `t` predicts `tick+t -> tick+t+1`).
   In `IdeaATrainer.rollout_loss()` `present` (the HDV attention mask) is
   updated every step to `present0 & alive`, so a departed vehicle leaves the
   neighbour set and a mid-window entrant is never added; a CAV only
   teacher-forces the recorded trajectory while alive (`cav = cav0 & present`),
   so its teacher columns are not consumed after departure.
2. **`summary_raw` alive steps.** The CAV-motion summary now uses a per-step
   CAV mask (`is_cav & alive_state`); `corr.summary._flat_mask` uses a
   `[T,B,K]` mask ELEMENTWISE instead of `any(dim=0)`, and the corpus
   forward-fills the trajectory so the displacement feature uses the slot's
   LAST ALIVE position, never a departed/reused slot's later state.
3. **INDEX run ranges fail closed.** `_load_run_ranges()` raised nothing and
   silently fell back to one range on a missing/unparsable `INDEX.json` or a
   run without `tick_start`/`tick_end`. It now raises `IdeaAConfigError` in
   every such case; the single-range fallback is reachable only through the
   explicit TEST-ONLY `allow_single_run_fallback` constructor flag.
4. **`evaluate_cal` requires the CAL half.** A non-empty CAL corpus with no
   (or an unknown) `half` now raises (amendment A4.2) instead of scoring the
   whole CAL set; `IdeaACorpus.window()` carries a top-level `world`/`arm` so
   the `evalh.cal_split` filter can resolve the item. Empty/unavailable CAL
   still returns `empty`/`unavailable` without scoring.

### Per-step CPU time (1 thread, real bundle, batch 4, 20-step rollout)

3 warmup + 20 timed steps, `bundle.e2e.write_synthetic_pack`, real P2/P4/P5
bundle:

| path | s/step | terms active |
| --- | --- | --- |
| v1 silent-drop (`MemmapCorpus` + `batch_from_corpus_items`) | 0.148 | residual_nll, arrival, response_contrast |
| full canonical, before review fixes | 1.177 | all five registered terms |
| full canonical, after review fixes | 1.038 | all five registered terms |

The review fixes are within run-to-run noise of the prior full-canonical
measurement (no regression); the ~7x over the v1 path is the real cost of the
previously-dropped terms.

## Verification

* `train/tests/test_idea_a_realbatch.py` (10 tests): main()-path fit activates
  every term and writes the receipt; `baseline_idm == hdv_base[:,0]`; teacher
  CAV slots equal the recorded kinematics; a pack without the Idea A channels
  is refused; a v1 corpus fails closed and the TEST-ONLY flag relaxes it; and
  the four review-fix tests - slot 3 vacated at step 4 / reused at step 6
  contributes zero rollout loss after step 4 and is excluded from attention; a
  departing CAV's teacher columns are not consumed; `summary_raw` ignores dead
  steps; a window never crosses a run boundary; a missing/unparsable/
  incomplete `INDEX.json` fails closed (fallback only via the test flag).
* `train/tests/test_idea_a_cal_split.py` (5 tests): `evaluate_cal` scores only
  the named half on a synthetic two-half CAL pack and refuses a missing/unknown
  half.
* Full suite `python -m pytest -q` -> **751 passed** (864 s).
* G4 real-corpus 1500-step overfit: `final 0.0233375 <= 0.05 on 2 windows`.
* `python -m preflight all --data-root E:/research-dense` -> **6/6 PASS**
  (zero-residual, ctrl-parity, label-leak, overfit, e2e-smoke, heldout-dryrun).
* `ruff check` clean on the changed files.
