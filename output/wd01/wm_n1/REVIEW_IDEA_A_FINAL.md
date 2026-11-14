# WM-N1 Idea A — FINAL REVIEW (integration stage 3 + close-out)

**Reviewer:** integrator-executor (single process, maxSubagents 0).
**Branch:** `zcode/wmn1integC`.
**Scope:** STAGE 3 real-module G4 + final review of `CODE_READY.md`, the box
configs, torch 2.5 compatibility and the `HOLDOUT`/`CAL`/chunk-5/`J4B` guards.
**Environment:** CPU only, zero SUMO, `WD_FORBID_D=1`, priority BelowNormal,
one worker. No `HOLDOUT`/`CAL`/chunk-5/`J4B` file was read; no
`evaluate_summary`, `dispatcher_rows` or `insertions` was opened.

Verdict table
-------------

| # | item | verdict | evidence |
| --- | --- | --- | --- |
| 1 | merge `zcode/feat1fix` + `zcode/wmn1docs`, conflict list | **PASS** | two clean `--no-ff` merges, **0 conflicts** (§1) |
| 2 | G4 overfit with REAL modules | **PASS** | `final 0.00965234 <= 0.05`, all 5 terms, real modules (§2) |
| 3 | stand-in fallback FAILS the gate | **PASS** | `overfit_report(standins=True)` -> `ok=False` + warning (§2) |
| 4 | combined pytest (one process) | **PASS** | `601 passed in 414.73s` (§3) |
| 5 | `preflight all` 6/6 | **PASS** | `PREFLIGHT_SUMMARY.json n_pass=6 n_fail=0 all_passed=true` (§4) |
| 6 | CODE_READY train/eval/score argv dry-run | **PASS** | 105 argv arrays parse; chunk-5 final TEST parses (§5) |
| 7 | box configs vs `GPU_PLAN.md` + `multibox` loader | **PASS** | 7/7 boxes match `plan_fleet` (§6) |
| 8 | torch 2.5 API scan + `torch.load` | **PASS** | 0 forbidden hits; all `torch.load` use `weights_only=True` (§7) |
| 9 | guard grep (HOLDOUT/CAL/chunk-5/J4B) | **PASS** | all access paths gated (§8) |

## 1. Merges and conflicts

Both merges were `--no-ff` onto `zcode/wmn1integC` (from `5d765cf`), and both
applied cleanly with the `ort` strategy. **Conflict list: NONE** — no file had
`<<<<<<<` markers and `git diff --name-only --diff-filter=U` was empty.

| merge | commit | content |
| --- | --- | --- |
| `zcode/feat1fix` | `b89e849` | FEAT1 shard-key fix (`batch/world/arm`); supersedes corpus/features code; 28 files |
| `zcode/wmn1docs` | `fbcb5d6` | Idea A as-built design doc + owner summary; 2 files |

The FEAT1 merge brought the FEAT1 corpus/features/inventory updates
(`corpus/*`, `features/*`, `inventory/*`, `train/tests/test_dataset.py`,
`forbid_d_hook.py`) on top of the stage-2 tree without conflict because the
stage-2 files are additive relative to the shared base `b3a6f6c`.

## 2. G4 overfit — REAL modules (defect 2a fixed)

`train/idea_a.overfit_report()` now defaults to the REAL P2/P4/P5 bundle from
the new `train/g4_real.py`:

* P2 `hdv.HdvResidualNet` (`d_model=16, n_heads=2, n_layers=1, ffn=32`);
* P4 `arrival.IdeaAArrivalHead` (`d_model=16, counter_channels=8`);
* P5 `corr.CorrectionHead` with the REAL registered 64-d motion summary
  (`summary_dim=64, d_model=16`), fed through the REAL
  `corr.CavMotionSummary` projection inside `RealCorrBundle` (the same
  16 -> 64 path as `evalh.idea_a._outcome`).

The total A3.2 loss is optimised on 2 tiny TRAIN windows (real
`IdeaATrainer` losses/optimiser, cosine LR, 1500 steps, `grad_clip=0`), so
gradients flow through the REAL residual NLL, differentiable rollout, arrival
NLL, correction CRPS and paired response contrast. Result
(`preflight/reports/overfit.json`):

```
PASS overfit: final 0.00965234 <= threshold 0.05 on 2 windows (cpu)
terms = {residual_nll 0.00048, rollout 0.00132, arrival 0.0,
         correction_crps 0.00700, response_contrast 0.00086}
terms_seen = [arrival, correction_crps, residual_nll,
              response_contrast, rollout]
real_modules = True
```

The documented linear STAND-INS (`train/g4_standins.py`, now marked
`G4_STANDIN=True`) are retained only as a fallback: `overfit_report(...,
standins=True)` (or a stand-in bundle) is FORCED to `ok=False` with a warning,
so the fallback FAILS the gate. New tests pin both behaviours
(`train/tests/test_idea_a_overfit.py`).

## 3. Combined pytest (one process)

```
$env:WD_FORBID_D="1"
python -m pytest -q
601 passed in 414.73s (0:06:54)
```

(was 572 before stage 3; the FEAT1 merge + the new G4/held-out tests add the
rest; 0 failures, 0 collection errors.)

## 4. Pre-flight 6/6 (`python -m preflight all --data-root E:/research-dense`)

```
PASS zero-residual: pos<= 0 m, speed<= 0 m/s, tol 1e-06, 1 window(s)
PASS ctrl-parity: all strategies within tolerance on >= 2 windows
PASS label-leak: kinematics-only schema; output invariant to labels/flags/strategies
PASS overfit: final 0.00965234 <= threshold 0.05 on 2 windows (cpu)
PASS e2e-smoke: queue e2e stages complete on CPU; torch scan clean (version gate: skipped)
PASS heldout-dryrun: all 3 held-out splits build; excluded arms absent from train
```

`preflight/reports/PREFLIGHT_SUMMARY.json`: `n_pass=6`, `n_fail=0`,
`all_passed=true`; G4 entry point `train.idea_a.overfit_report` (real modules).

## 5. CODE_READY argv dry-run

Every `train_argv` / `eval_argv` / `score_argv` / `box_cli` in
`bundle/box_configs/{3box,4box}/*.json` was parsed with its own parser
(`train.idea_a.build_parser`, `train.cli.build_parser`,
`evalh.run_eval.build_parser`, `bundle.multibox.build_parser`) or executed with
`evalh.idea_a heldout-config`: **105/105 parse-ok**. The CODE_READY §5.5
chunk-5 final TEST argv also parses.

J4B wording corrected (defect 2b): CODE_READY, the G6 note in
`train/idea_a.py` and the box config for `heldout-i-j4b` no longer call J4B
"the planned chunk-5 TEST wall arm". J4B is the **J4 wall panel re-run under
wall law v2.2** (`REGISTRATION_J4B.md` on `zcode/wd01`, LF `2463f5be`), read
ONLY by held-out test (i); chunk 5 is the separate sealed final TEST
(`f8c7d760`). The held-out (i) eval in CODE_READY / the box config is now the
registered A3.4 `heldout-config` check (no data read, no seal consumed); the
chunk-5 final TEST has its own §5.5 command. A new subcommand
`python -m evalh.idea_a heldout-config --test i` validates the (i) split and
the `assert_j4b_read_allowed` guard, writing `HELDOUT_CONFIG.json`.

## 6. Box configs vs GPU plan and loader

`bundle.multibox.plan_fleet(3|4)` reproduces every `assigned_configs` order,
`box_id`, `wall_clock_hours`, `box_finish_hours`, per-config `start_hours` and
`gpu_hours`, and both `index.json` files. 7/7 boxes validated; schema
`wm_n1_box_config/1`. 3-box makespan 54.0 h, 4-box 41.0 h, matching
`GPU_PLAN.md`.

## 7. torch 2.5 API scan

`preflight.torch_compat.scan_and_check` over the whole `wm_n1` tree: **0
forbidden hits** (`torch.compile`, `torch.amp`/`autocast`,
`scaled_dot_product_attention`, `flex_attention`, `torch.export`), clean.
Runtime here is torch `2.12.1+cpu`, so the version-gated op test is documented
as skipped (G5 still PASS). All 9 `torch.load` call sites pass
`weights_only=True` explicitly.

## 8. Guard grep (no unguarded HOLDOUT/CAL/chunk-5/J4B read)

* `evalh.seals.guard_split` — CAL read-once marker; chunk-5 `final_test`
  requires `--final-test` + BOTH frozen LF-sha256; HOLDOUT refused.
* `evalh.idea_a.assert_j4b_read_allowed` / `held_out_eval_runs` — J4B reads
  allowed ONLY for config (i); `FORBIDDEN_READ_NAMES` blocks
  `dispatcher_rows`/`insertions`/`evaluate_summary`/`physics_steps`.
* `train.dataset.MemmapCorpus` refuses any pack containing HOLDOUT/TEST/chunk-5;
  `corpus/verify.py FORBIDDEN_SPLITS`, `features/splits.py` chunk-5 gate,
  `adapter/export.py` chunk-5 refusal, `bundle/queue.py _SEALED_TOKENS`,
  `combine/guards.py SEALED`, `inventory/inventory_lib.py assert_clean`.
* `train.idea_a.heldout_split_report` reads ONLY the TRAIN split of the
  inventory; J4B rows are DECLARED metadata tags (never opened).
* `ctrl/parity.py` and `preflight` gates read TRAIN only.

No active code path opens the sealed bytes outside these guards; no
`HOLDOUT`/`CAL`/chunk-5/`J4B` file was touched during this review.

## 9. Residual observations

* `bundle/box_configs` are enriched outside the repo (the `plan_fleet`
  generator emits assignments only), so their `eval_argv` for `heldout-i-j4b`
  was hand-corrected to the registered (i) check; a future regeneration must
  preserve it.
* `train.idea_a.overfit_report` seeds the real correction head's initial biases
  near the tiny-window outcome scale (`train/g4_real.G4_CORR_BASE_SEED`); this
  is a documented G4 probe optimisation aid only — module code, forward
  contracts and losses are unchanged.
* J4B outcome evaluation remains gated by `REGISTRATION_J4B.md` and cannot run
  before the J4B panel is produced and authorized.
