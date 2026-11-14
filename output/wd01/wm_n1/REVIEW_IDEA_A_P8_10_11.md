# REVIEW: WM-N1 Idea A pieces P8 + P10/P11 (first pass)

**Reviewer task:** first-pass review of P8 (`zcode/wmn1p8` @ `4f8ec08`) and
P10+P11 (`zcode/wmn1p1011` @ `9abf0aa`), merged `--no-ff` into `zcode/revA2`
(base `zcode/wmn1base2` @ `99ad833`), against `INTERFACES_V2.md`, amendment 3
(`zcode/wd01:.../REGISTRATION_WM_N1_AMENDMENT_3_IDEA_A.md`) and
`PREFLIGHT_GATES_IDEA_A.md` (`zcode/wd01` @ `7ce56a1`).

**Scope:** no SUMO, no real data, no HOLDOUT/CAL/chunk-5/J4B bytes;
`WD_FORBID_D=1`; CPU only. Python
`C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`.

**Result: PASS with two review fixes applied and three non-blocking
observations.**

---

## 1. Merge

| branch | commit | merge commit | conflicts |
| --- | --- | --- | --- |
| `zcode/wmn1p8` | `4f8ec08` | `1237bb6` | none (`ort`) |
| `zcode/wmn1p1011` | `9abf0aa` | `7187f74` | none (`ort`) |

The two pieces own disjoint paths (P8: `adapter/`, `combine/`, `evalh/`;
P10/P11: `bundle/multibox.py`, `preflight/**`, amendment 2/3 records), so no
conflicts. `git log -5`:

```
f936ffd review fix (preflight): import PREFLIGHT_GATES_IDEA_A.md spec (zcode/wd01 7ce56a1, byte-identical 99767ef) referenced by preflight README/__init__
81f6cae review fix (preflight): rename tests/test_cli.py -> tests/test_preflight_cli.py (unique basename; restores combined wm_n1 pytest collection, REVIEW_WMN1_PIECES X3)
7187f74 Merge zcode/wmn1p1011 (P10: multibox one-config-per-box queue; P11: Idea A pre-flight gates G1-G6) into zcode/revA2
1237bb6 Merge zcode/wmn1p8 (P8: Idea A evalh predictor, held-out tests, CAL-only OOD, combine A tie order, adapter bundle loader) into zcode/revA2
4f8ec08 wd01 WM-N1 P8: Idea A evalh (variant A predictor, held-out tests, CAL-only OOD) + combine A tie order + adapter Idea A bundle loader
```

---

## 2. P8 - `combine.select_variant` (A3.3)

| # | Item | Verdict | Evidence |
| --- | --- | --- | --- |
| P8-1 | `VARIANTS == ("T","A","C","N")` | **PASS** | `select_variant.py:22`. |
| P8-2 | Tie order exactly `T < A < C < N` (simplest first) | **PASS** | `COMPLEXITY_RANK={"T":0,"A":1,"C":2,"N":3}` (`:23`); `chosen=min(tied,key=rank)` (`:93`). Independent run (coverage 0.80, all-tie `T`; `{T1.02,A1,C1,N1}`->`T`; `{A1,C1,N1}`->`A`; `{C1,N1}`->`C`; `{N1}`->`N`): T, T, A, C, N. `test_idea_a_tie_order_beats_c_and_n`, `test_tie_within_two_percent_goes_to_simpler`. |
| P8-3 | CAL-only (never sees TEST/HOLDOUT split) | **PASS** | Consumes CAL score/coverage dicts only; no split input. `combine/tests/test_select_variant.py`. |
| P8-4 | `combine/cli.py` synthetic set includes A + coverage | **PASS** | `cli.py:73-83` adds `A` to `crps` and `coverage`. |

## 3. P8 - held-out protocol (A3.4)

| # | Item | Verdict | Evidence |
| --- | --- | --- | --- |
| P8-5 | (i) HEADLINE: train J4 (v2.1), eval J4B (v2.2) | **PASS** | `train_excludes`: `source != "J4"`; `eval_includes`: `source == "J4B"` (`idea_a.py:93,115`). |
| P8-6 | (ii) drop every PI arm from train; eval CAL PI | **PASS** | `train_excludes`: `policy == "pi"`; `eval_includes`: `split=="cal" and policy=="pi"`. |
| P8-7 | (iii) drop one wall-speed factor; eval that factor's CAL wall arms; factor a priori highest J4 | **PASS** | `HIGHEST_J4_WALL_SPEED_FACTOR="mid"` (`:54`, J4 W-mid 0.85 > W-lo 0.60); filters match. `test_iii_drops_highest_j4_wall_factor`. |
| P8-8 | J4B read refused outside (i) | **PASS** | `assert_j4b_read_allowed` raises for every test with `reads_j4b=False`; `held_out_eval_runs` guards the whole candidate pool. `test_j4b_refused_outside_config_i`. |
| P8-9 | Held-out tests are generalisation evidence, not selection inputs | **PASS** | No coupling to `select_planner_variant`; `select_variant` never imports `idea_a`. |

## 4. P8 - OOD (A3.5), predictor, adapter, seals

| # | Item | Verdict | Evidence |
| --- | --- | --- | --- |
| P8-10 | OOD calibration is CAL-only | **PASS** | `calibrate_ood` raises `SealError` for any `split != "cal"`, and routes `worlds` through `seals.assert_holdout`. `test_calibration_is_cal_only_guarded`. |
| P8-11 | Threshold = CAL quantile; larger score = less familiar | **PASS** | `np.quantile(cal_scores, cal_quantile)`; `hold_back = score >= threshold`. `test_threshold_is_cal_quantile_and_unfamiliar_scores_high`. |
| P8-12 | Variant A predictor conforms to `Predictor` | **PASS** | `IdeaAVariantPredictor(PredictorBase)`, `name="A"`, 19 levels; fakes exercise both heads; production path raises `IdeaANotLoadedError` without a bundle. `test_idea_a_variant.py`, `test_idea_a_evalh_interface.py`. |
| P8-13 | Adapter composed-bundle loader (INTERFACES_V2 Mismatch 8) | **PASS** | `load_idea_a_bundle` accepts engine `IdeaABundle`, mapping, or `idea_a.json` dir; `load_idea_a_predictor` wraps it as `IdeaABundlePredictor`. Signatures match `rollout2.engine.IdeaABundle(ctrl,hdv,arrival,corr,cfg)` and `rollout2(bundle,batch,*,n_steps,n_samples,seed)`. `test_idea_a_adapter.py`. |
| P8-14 | `evalh/seals.py` UNCHANGED | **PASS** | `git diff zcode/wmn1base2..zcode/revA2 -- output/wd01/wm_n1/evalh/seals.py` is empty. |
| P8-15 | Final TEST path needs `--final-test` + BOTH frozen registration shas; read-once | **PASS** | `seals.guard_split(...,final_test=True)` refuses without the flag and on wrong `LF_SHA_CHUNK5`/`LF_SHA_WM_N1`; marker is read-once. `test_missing_final_test_refused`, `test_wrong_*_sha_refused`, `test_test_is_read_once` (unchanged, passing). |
| P8-16 | Combine/adapter/evalh tests pass | **PASS** | 126 passed (see section 8). |

### P8 overclaims / gaps (non-blocking)

- **O1 (reported, not fixed).** The acceptance phrase "final-test path refuses
  `k<5`" (amendment 1 A2: the single final TEST uses the **full 5-member
  ensemble**; the early 1-member model is never run on chunk 5) is **not
  implemented in P8**. There is no ensemble-size guard in `evalh/idea_a.py`,
  `adapter/predictor.py` or `combine/`. The only `k` bound is
  `train.IdeaATrainConfig.ensemble_k in 1..5` (P7, partial ensembles for CPU
  smoke), which *allows* `k<5` and has no final-test refusal. Interpreted in
  the review as a genuine gap; it is a cross-piece (P7/P8/run_eval) change, so
  it is reported rather than invented here. The companion clause "needs both
  registration shas" is satisfied by the unchanged seal path (P8-15).
- **O2 (minor, latent).** `IdeaAVariantPredictor._outcome` reads
  `getattr(self.bundle, "corr_cfg", None)`, but no bundle defines `corr_cfg`
  (`rollout2.engine.IdeaABundle` exposes `cfg`); it silently falls back to the
  default `CorrConfig()`. The production path is still a stub
  (`cav_motion_summary`/`CavMotionSummary.forward` raise), so no test catches
  it. Flag for P3/P5 wiring.
- **O3 (minor).** Held-out (i) train filter is `source == "J4"` only, not also
  `wall_law == "wall-v2.1-closure"`. A mis-tagged J4 run with law v2.2 would
  be admitted; registration defines J4 as v2.1, so risk is low.

---

## 5. P10 - `bundle.multibox` (one config per box)

| # | Item | Verdict | Evidence |
| --- | --- | --- | --- |
| P10-1 | Exactly one config per box | **PASS** | `DEFAULT_BOX_ASSIGNMENT` maps 8 boxes -> 8 singleton tuples; `assignment_receipt()["one_config_per_box"] is True`. `plan` output below. |
| P10-2 | Unassigned-config refusal | **PASS** | `run_config` -> `UnassignedConfigRefusal` when `config not in configs_for_box(box)`; unknown box -> `MultiboxError`. `test_run_config_refuses_unassigned_config`, `test_run_config_refuses_unknown_box`. |
| P10-3 | Per-config approval binds written TIMING_GATE sha + 4 frozen LF-shas | **PASS** | `require_config_approval` checks declared fields == frozen constants and re-hashes present records (CRLF-safe `seals.lf_sha256`) + raw-bytes gate sha. `test_approval_wrong_sha_refused` parametrised over all 5 fields, `test_approval_tampered_record_refused`, `test_approval_accepts_crlf_copies`, `test_approval_wrong_config_refused`. |
| P10-4 | Copied amendment/registration LF-shas equal `666233aa`/`decb9656`/`56fa99a0`/`a597d986` | **PASS** | Independently recomputed (CRLF->LF) at `zcode/wmn1p1011`: REGISTRATION `666233aa...93246`, AMD1 `decb9656...16c57`, AMD2 `56fa99a0...c83ed`, AMD3 `a597d986...04eef6` - all OK; the ADDED copies in the merge hash identically. |
| P10-5 | Member-1-first | **PASS** | `run_config` refuses member k>1 unless `member1_completed`; `run_fleet` runs member-1's box in phase 1 before the rest. `test_member_2_refused_before_member_1`, `test_member_1_first_within_a_box`, `test_run_fleet_runs_every_box_member_1_first`. |
| P10-6 | Resume (idempotent, per-box + fleet state) | **PASS** | Per-box `boxes/<box>/state.json` + `fleet_state.json`; completed config returns `skipped=True`, executor not re-called. `test_resume_from_per_box_state`, `test_box_state_is_per_box`. |
| P10-7 | No hardcoded `E:/` (or any drive) path on the box side | **PASS** | grep for `[A-Za-z]:[\\/]`, `D:`, `E:`, `/research` over `bundle/multibox.py`, `bundle/tests/test_multibox.py`, `preflight/**`: no drive paths (only `https://download.pytorch.org` in docs and the literal `(i) HEADLINE:` text). Paths are built from `work_dir`. |

`python -m bundle.multibox plan` (run from `output/wd01/wm_n1`):

```
{
 "one_config_per_box": true,
 "n_boxes": 8,
 "n_configs": 8,
 "boxes": {
  "box-member-1": ["member-1"],
  "box-member-2": ["member-2"],
  "box-member-3": ["member-3"],
  "box-member-4": ["member-4"],
  "box-member-5": ["member-5"],
  "box-heldout-i-j4b": ["heldout-i-j4b"],
  "box-heldout-ii-leave-pi": ["heldout-ii-leave-pi"],
  "box-heldout-iii-leave-highest-wall-speed": ["heldout-iii-leave-highest-wall-speed"]
 }
}
```

---

## 6. P11 - pre-flight gates G1-G6 (PREFLIGHT_GATES_IDEA_A.md)

| # | Item | Verdict | Evidence |
| --- | --- | --- | --- |
| P11-1 | G1 `zero-residual` wording/threshold (residual 0, arrivals off, v2g, <=1e-6 or explained tolerance) | **PASS** | `BASE_TOLERANCE=1e-6`; checks integrator is v2g, non-empty J4 windows, `pos/spd <= tol`, and refuses `tol>1e-6` without `tolerance_explained`. |
| P11-2 | G2 `ctrl-parity` (native/damping/PI/wall-v2.1, >=2 TRAIN windows, supports v2.1 AND v2.2) | **PASS** | `REQUIRED_STRATEGIES=("native","damping","pi","wall-v2.1")`, `MIN_TRAIN_WINDOWS=2`, `wall-v2.2` must be present. |
| P11-3 | G3 `label-leak` (no action/strategy/CAV channel; perturbation invariant) | **PASS** | `FORBIDDEN_TOKENS=("action","strategy","cav","one_hot","onehot")` over the tensor schema fields; requires `identical is True and max_abs_delta==0` over >=1 perturbation. |
| P11-4 | G4 `overfit` (1 member, total loss near 0, 2 tiny TRAIN windows, CPU) | **PASS** | `MIN_WINDOWS=2`; requires `device=="cpu"` (or unset), finite loss curve, `final <= threshold`. |
| P11-5 | G5 `e2e-smoke` (convert->timing(30s)->train->ckpt->resume->eval, CPU; torch-2.5.1 check) | **PASS** | `STAGES` exactly the six queue stages; `MIN_TIMING_SECONDS=30`; runs `torch_compat.scan_and_check` (static scan + version-gated op test). |
| P11-6 | G6 `heldout-dryrun` (configs i-iii build; excluded arms absent from train) | **PASS** | `EXCLUDED_ARM={heldout-i-j4b:"j4b", heldout-ii-leave-pi:"pi", heldout-iii-leave-highest-wall-speed:"wall-speed-highest"}`; requires non-empty train+eval, arm declared excluded and absent from train. |
| P11-7 | `piece missing` -> FAIL (never PASS) | **PASS** | Every gate catches `PieceMissing` -> `fail(..., error="piece-missing")`. `test_g*_piece_missing` (6 tests). Live `all` run: all six FAIL with `piece missing: ...` / `entry point missing: ...`. |
| P11-8 | Reports written under `preflight/reports/` | **PASS** | `python -m preflight all` wrote `zero-residual.json`, `ctrl-parity.json`, `label-leak.json`, `overfit.json`, `e2e-smoke.json`, `heldout-dryrun.json`, `PREFLIGHT_SUMMARY.json`. `test_run_gates_all_pass_and_reports_written`, `test_main_all_with_missing_pieces_writes_summary`. |
| P11-9 | torch 2.5 scan present | **PASS** | `torch_compat.scan_sources` forbids `torch.compile/amp/autocast/scaled_dot_product_attention/flex_attention/torch.export`; `version_gate` runs on 2.5.x else documented skip. Live scan: 108 files, 0 hits, version gate `skipped` (local torch 2.12.1+cpu). |
| P11-10 | Spec doc present for the runbook | **PASS (fixed)** | `PREFLIGHT_GATES_IDEA_A.md` was missing though referenced by `preflight/README.md` and `preflight/__init__`; imported byte-identical (blob `99767ef`) from `zcode/wd01` @ `7ce56a1` in `f936ffd`. |

`python -m preflight all --dry` (run from `output/wd01/wm_n1`):

```
WM-N1 Idea A pre-flight gates (dry run; nothing executed)
  G1 zero-residual      -> rollout2.equivalence.zero_residual_report
  G2 ctrl-parity        -> ctrl.parity.strategy_parity_report
  G3 label-leak         -> hdv.input_schema.main_config_schema, hdv.input_schema.perturbation_check
  G4 overfit            -> train.idea_a.overfit_report
  G5 e2e-smoke          -> bundle.e2e.run_e2e_smoke
  G6 heldout-dryrun     -> train.idea_a.heldout_split_report
  (6 gates)
```

Live `python -m preflight all` (CPU; P1-P7 not yet merged) - all FAIL, not PASS:

```
FAIL zero-residual: piece missing: rollout2.equivalence.zero_residual_report (module 'rollout2.equivalence' not importable)
FAIL ctrl-parity: piece missing: ctrl.parity.strategy_parity_report (module 'ctrl.parity' not importable)
FAIL label-leak: piece missing: hdv.input_schema.main_config_schema (module 'hdv.input_schema' not importable)
FAIL overfit: entry point missing: train.idea_a.overfit_report (module train.idea_a has no attribute 'overfit_report')
FAIL e2e-smoke: piece missing: bundle.e2e.run_e2e_smoke (module 'bundle.e2e' not importable)
FAIL heldout-dryrun: entry point missing: train.idea_a.heldout_split_report (module train.idea_a has no attribute 'heldout_split_report')
[preflight] summary -> .../preflight/reports/PREFLIGHT_SUMMARY.json
```

G5 torch scan captured in `preflight/reports/e2e-smoke.json`:

```
"torch": {"ok": true,
 "scan": {"ok": true, "n_files": 108, "forbidden_hits": []},
 "version_gate": {"status": "skipped", "ok": true,
                  "torch_version": "2.12.1+cpu", "expected": "2.5.1"}}
```

### P11 observations (non-blocking)

- **O4.** G5 trusts the entry-point report for `n_worlds==1`, the 60 s horizon
  and `train_steps>0`; it validates only stages/device/timing. Acceptable, as
  those live in the P3/bundle e2e contract.
- **O5.** `preflight/__init__.__all__ = ["GATE_ORDER","GATES"]` names objects
  that are never imported at package level (harmless documentation).

---

## 7. Review fixes applied

| fix | commit | change | reason |
| --- | --- | --- | --- |
| F1 | `81f6cae` | rename `preflight/tests/test_cli.py` -> `test_preflight_cli.py` | P11 reintroduced a duplicate `test_cli` basename (vs `combine/tests/test_cli.py`), regressing `REVIEW_WMN1_PIECES` X3: a combined `pytest output/wd01/wm_n1` failed collection (`import file mismatch`). After the rename, combined collection is clean. |
| F2 | `f936ffd` | add `PREFLIGHT_GATES_IDEA_A.md` (blob `99767ef`, from `zcode/wd01` @ `7ce56a1`) | referenced by `preflight/README.md` and `preflight/__init__` but absent from the merged branch; the P11 acceptance requires review against it. |

## 8. Test evidence (pasted)

Environment: `WD_FORBID_D=1`, `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`.

Pre-fix combined collection (expected regression, hits F1):

```
ERROR collecting output/wd01/wm_n1/preflight/tests/test_cli.py
import file mismatch: imported module 'test_cli' ... combine\tests\test_cli.py ...
1 error in 8.53s
```

P8 target suites:

```
$ pytest -q output/wd01/wm_n1/combine/tests output/wd01/wm_n1/evalh/tests output/wd01/wm_n1/adapter/tests
126 passed, 2 warnings in 125.65s (0:02:05)
```

P10/P11 target suites:

```
$ pytest -q output/wd01/wm_n1/bundle/tests output/wd01/wm_n1/preflight/tests
81 passed in 15.96s
```

Whole wm_n1 tree after F1 (regression guard):

```
$ pytest -q output/wd01/wm_n1
379 passed in 134.56s (0:02:14)
```

## 9. Verdict summary

- **P8: PASS** for all listed acceptance items (select_variant T,A,C,N tie
  order; held-out (i)-(iii); J4B read guard; CAL-only OOD; seals unchanged;
  final-test needs both shas). Gap **O1** (no `k<5` final-test guard) and minor
  latent **O2**/**O3** reported.
- **P10: PASS** (one config per box; unassigned refusal; approval binds gate
  sha + 4 frozen LF-shas, copies verified; member-1-first; resume; no
  hardcoded box paths).
- **P11: PASS** (G1-G6 match spec wording/thresholds; piece-missing -> FAIL;
  reports written; dry-run pasted; torch 2.5 scan present), with the missing
  spec doc imported (F2) and the test-name collision fixed (F1).
