# WM-N1 Idea A — CODE READY (integration stage 3)

**Branch:** `zcode/wmn1integC` (stage-2 base `zcode/wmn1integB` `5d765cf`;
stage-1 = `zcode/wmn1integA` `bac9f33`).
**Status:** stage-3 integration complete. **ALL 6 pre-flight gates PASS with the
REAL-module G4** (`PREFLIGHT_SUMMARY.json`: `n_pass=6`, `n_fail=0`,
`all_passed=true`). Combined suite: **601 passed in one process, 0 collection
errors**. Zero SUMO; CPU only; TRAIN data only. Final review:
[`REVIEW_IDEA_A_FINAL.md`](REVIEW_IDEA_A_FINAL.md).

## 1. Merges (`--no-ff`, reviewed versions preferred)

| merge | commit | content |
| --- | --- | --- |
| `zcode/feat1fix` | `b89e849` | FEAT1 shard-key fix (`batch/world/arm`); supersedes corpus/features code (28 files) |
| `zcode/wmn1docs` | `fbcb5d6` | Idea A as-built design doc `WM_N1_IDEA_A_DESIGN.md` + owner summary (2 files) |
| `zcode/revA1g` | `99f0188` | (stage 2) P1 ctrl + second-pass: per-strategy G2 tolerances, `law_version` wall-v2.1-closure + frozen v2.2 merge-release |
| `zcode/revA3g` | `7d31388` | (stage 2) P3 rollout2 second-pass fixes (supersedes `zcode/wmn1p3`); real J4 TRAIN G1 evidence |
| `zcode/wmn1ncmp` | `eefe49e` | (stage 2) variant-N comparator configs + `GPU_PLAN.md` 3/4-box planner |

**Conflicts: NONE** (stage 3). `zcode/feat1fix` and `zcode/wmn1docs` both merged
cleanly with the `ort` strategy (`git diff --diff-filter=U` empty: no
`<<<<<<<` markers); the FEAT1 merge is additive relative to the shared base
`b3a6f6c`, so stage-2 files were preserved. Stage 2 fixed one **semantic**
conflict after its merges: `revA3g` renamed `select_j4_train_run` →
`select_j4_train_windows`, but the auto-merge kept stage 1's stale call site in
`zero_residual_report` (undefined name); corrected in integration (see §2).

## 2. Stage-3 wiring (integration only; no piece internals rewritten)

| file | change |
| --- | --- |
| `train/g4_real.py` (new) | REAL-module G4 bundle: `HdvResidualNet`/`IdeaAArrivalHead`/`CorrectionHead` tiny-but-real config + real 64-d `CavMotionSummary` projection (`RealCorrBundle`) |
| `train/idea_a.py` | `overfit_report` defaults to the REAL bundle (1500 steps, cosine LR, `grad_clip=0`); the stand-in path is forced to FAIL the gate; G6 J4B wording corrected; `build_parser` + `--dry-run` added to the member CLI |
| `train/g4_standins.py` | stand-ins retained as a documented fallback; `G4_STANDIN` markers; docstring updated |
| `evalh/idea_a.py` | new `heldout-config` CLI: registered A3.4 (i) split + `assert_j4b_read_allowed` check, no data read |
| `bundle/box_configs/**` | `heldout-i-j4b` eval/score corrected off the chunk-5 seal to the registered (i) check |
| `train/tests/test_idea_a_overfit.py`, `evalh/tests/test_idea_a_held_out.py` | pin real-module G4, stand-in gate failure, and the held-out config check |

## 3. Code-root tree shas (`git rev-parse HEAD:<path>`)

Code commit `5088c6ae7f058ff425374cb20857f705b62e3d15`:

```
HEAD                        5088c6ae7f058ff425374cb20857f705b62e3d15
root tree                   b8db17416df19b74dcadbde2c38e059e1e908db2
output/wd01/wm_n1 subtree   8020437432248d0c96fdb7eceab88a64f902bcdd
ctrl/                       a74cf9e7f8b3afddebd1a0939740b51ad220d3d5
rollout2/                   aa68e159ac2b637138aecfcafa07dd7e869fb90f
train/                      13360eda63b46231e3a9cdefc4789d29a67078fc
preflight/                  28ffce742d8c58b8b01e40ae8b90322655d6829f
bundle/                     a8f39c5c623f58e6668e0eb91c0516d483fa4382
bundle/box_configs/         861db7836240ae91eba57f3f5adeb77b73f66982
adapter/                    f45205bac4b87d283788324f426de7c8016f31cc
evalh/                      205ea18f41243d3a915f77535ea3942eb0c61c42
```

Key blobs (`git rev-parse HEAD:<path>`):

```
ctrl/parity.py                    ae12387d3acfcd628f78f25c199b7f9401e0c3ed
rollout2/equivalence.py           f69221ca0f426e90142604676e3a48af8513c8fa
train/idea_a.py                   593558d0e7dd6065afb911f91063ef1962c6876d
train/g4_real.py                  e05ca81aa163eeb95d71bfd17274df1533b54a65
train/g4_standins.py              876277deef2af53084195ea4cad7d634a5ddee4b
bundle/requirements-wmn1.txt      5b05d995a1f9f38fae95b33591251138ce9e1f86
bundle/GPU_PLAN.md                3c1b93b2163a02bbf57f206d1e1260fb17708a12
preflight/reports/PREFLIGHT_SUMMARY.json  7999c830c81a95696b2750a51bd384ac92a0bcc6
```

## 4. Dependencies

- GPU box: **`torch>=2.7,<2.13` from the cu128 wheel index**, Python 3.10/3.11
  (`>=3.10,<3.12`), pinned in
  [`bundle/requirements-wmn1.txt`](bundle/requirements-wmn1.txt)
  (`torch>=2.7,<2.13`, `numpy>=1.24`, `pandas>=2.0`, `pyarrow>=14.0`). Install:
  `python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7,<2.13"`.
  This supports Blackwell-class sm_120 GPUs (vGPU-48GB, driver CUDA <= 13.2).
- Only torch-2.7..2.12-compatible ops are used (no `torch.compile` /
  `torch.cuda.amp` / deprecated `torch._six` / SDPA / flex_attention / export).
  The damping controller runs the vendored torch ONNX graph
  (`partA_ctrl/onnx2torch.py`), never `onnxruntime`.
- Local CPU box: torch 2.12.1+cpu (inside the supported window, so the G5
  version-gated op test now RUNS rather than skips). Verify the box with
  `python -m bundle.envcheck` (see [`TORCH27_COMPAT.md`](TORCH27_COMPAT.md)).
  No SUMO. `WD_FORBID_D=1` on every command.

## 5. GPU-box configs — exact argv per config

Run every command **from `output/wd01/wm_n1`**; `$WORK` is the box-local run
directory. **Idea A / held-out configs train with
`train.idea_a.IdeaATrainer` (NOT the v1 trainer); the variant-N comparator
trains with the v1 CLI `python -m train`.** The machine-readable source of
truth (per-box packing + the same argv arrays) is
`bundle/box_configs/{3box,4box}/box{1..N}.json`; the tables below spell them
out. `$PACK` is the box's packed TRAIN corpus (see §8).

Common eval/score reference for all configs:
`--reference evalh.climatology:ClimatologyPredictor`.

### 5.1 Idea A ensemble members (member-1 first)

Predictor `adapter.predictor:IdeaABundlePredictor`; seeds `101, 202, 303, 404,
505` for members 1..5. `$K` = member index.

```
train  python -m train.idea_a --member $K --seed <SEED> --corpus $PACK \
         --out $WORK/members/member_$K --device cuda --ensemble-k 5
eval   python -m evalh.run_eval --split cal --out $WORK/eval/member_${K}_cal \
         --predictor adapter.predictor:IdeaABundlePredictor \
         --reference evalh.climatology:ClimatologyPredictor
score  python -m evalh.run_eval --split train_oof \
         --out $WORK/score/member_${K}_oof \
         --predictor adapter.predictor:IdeaABundlePredictor \
         --reference evalh.climatology:ClimatologyPredictor
```

| config | member | seed | expected outputs |
| --- | --- | --- | --- |
| `member-1` | 1 | 101 | `$WORK/members/member_1/member_1.pt`, `member_1.json`, `TRAIN_RECEIPT.json`, `$WORK/eval/member_1_cal/{rows.csv,REPORT.json}` |
| `member-2` | 2 | 202 | `$WORK/members/member_2/{member_2.pt,member_2.json,TRAIN_RECEIPT.json}` |
| `member-3` | 3 | 303 | `$WORK/members/member_3/{member_3.pt,member_3.json,TRAIN_RECEIPT.json}` |
| `member-4` | 4 | 404 | `$WORK/members/member_4/{member_4.pt,member_4.json,TRAIN_RECEIPT.json}` |
| `member-5` | 5 | 505 | `$WORK/members/member_5/{member_5.pt,member_5.json,TRAIN_RECEIPT.json}` |

Member-k (k>1) refuses until `member-1` has completed (amendment 1 A1).

### 5.2 Variant-N comparator members (v1 pure-neural; member-1 first in-family)

Predictor `adapter.predictor:WmN1Predictor`; same seeds `101, 202, 303, 404,
505`. Trains with the **v1 CLI**.

```
train  python -m train --member $K --seed <SEED> --corpus $PACK \
         --out $WORK/members/n_member_$K --device cuda
eval   python -m evalh.run_eval --split cal \
         --out $WORK/eval/n_member_${K}_cal \
         --predictor adapter.predictor:WmN1Predictor \
         --reference evalh.climatology:ClimatologyPredictor
score  python -m evalh.run_eval --split train_oof \
         --out $WORK/score/n_member_${K}_oof \
         --predictor adapter.predictor:WmN1Predictor \
         --reference evalh.climatology:ClimatologyPredictor
```

| config | member | seed | expected outputs |
| --- | --- | --- | --- |
| `n-member-1` | 1 | 101 | `$WORK/members/n_member_1/{member_1.pt,SEEDS.json,TRAIN_RECEIPT.json}` |
| `n-member-2` | 2 | 202 | `$WORK/members/n_member_2/{member_2.pt,SEEDS.json,TRAIN_RECEIPT.json}` |
| `n-member-3` | 3 | 303 | `$WORK/members/n_member_3/{member_3.pt,SEEDS.json,TRAIN_RECEIPT.json}` |
| `n-member-4` | 4 | 404 | `$WORK/members/n_member_4/{member_4.pt,SEEDS.json,TRAIN_RECEIPT.json}` |
| `n-member-5` | 5 | 505 | `$WORK/members/n_member_5/{member_5.pt,SEEDS.json,TRAIN_RECEIPT.json}` |

`n-member-k` (k>1) refuses until `n-member-1` completed (in-family rule).

### 5.3 Held-out-strategy tests (1-member models, seed 101)

All three train with `train.idea_a.IdeaATrainer` (`--member 1 --seed 101
--ensemble-k 1`) and evaluate with `adapter.predictor:IdeaABundlePredictor`.

**(i) `heldout-i-j4b` — train J4 (law v2.1) → eval J4B (law v2.2):**

J4B is the J4 wall panel **re-run under wall law v2.2** (`merge-release`;
`REGISTRATION_J4B.md` on `zcode/wd01`, LF `2463f5be`), read ONLY by held-out
test (i). It is **not** the chunk-5 sealed final TEST (that is the separate
5-member run in §5.5, `f8c7d760`). Test (i) is evaluated through the registered
A3.4 held-out protocol (`evalh.idea_a.held_out_protocol()` config (i) /
`assert_j4b_read_allowed`); no other config may read J4B.

```
train  python -m train.idea_a --member 1 --seed 101 --corpus $PACK \
         --out $WORK/members/heldout-i-j4b --device cuda --ensemble-k 1
eval   python -m evalh.idea_a heldout-config --test i \
         --out $WORK/eval/heldout-i-j4b \
         --predictor adapter.predictor:IdeaABundlePredictor \
         --reference evalh.climatology:ClimatologyPredictor
score  = the same registered held-out (i) protocol command
```

The `heldout-config` check validates the registered (i) train/eval filters and
the J4B guard, reading NO CAL/HOLDOUT/chunk-5/J4B bytes and consuming no seal
marker. The J4B outcome evaluation runs once the J4B panel exists and its
registration authorizes the read.

**(ii) `heldout-ii-leave-pi` — train without every PI arm → eval CAL PI:**

```
train  python -m train.idea_a --member 1 --seed 101 --corpus $PACK \
         --out $WORK/members/heldout-ii-leave-pi --device cuda --ensemble-k 1
eval   python -m evalh.run_eval --split cal \
         --out $WORK/eval/heldout-ii-leave-pi \
         --predictor adapter.predictor:IdeaABundlePredictor \
         --reference evalh.climatology:ClimatologyPredictor
score  = the same cal command
```

**(iii) `heldout-iii-leave-highest-wall-speed` — train without the a-priori
highest J4 wall-speed factor → eval those arms:**

```
train  python -m train.idea_a --member 1 --seed 101 --corpus $PACK \
         --out $WORK/members/heldout-iii-leave-highest-wall-speed \
         --device cuda --ensemble-k 1
eval   python -m evalh.run_eval --split cal \
         --out $WORK/eval/heldout-iii-leave-highest-wall-speed \
         --predictor adapter.predictor:IdeaABundlePredictor \
         --reference evalh.climatology:ClimatologyPredictor
score  = the same cal command
```

Expected outputs for (ii)/(iii): `$WORK/members/<config>/{member_1.pt,
TRAIN_RECEIPT.json}` + `$WORK/eval/<config>/{rows.csv,REPORT.json}`; for (i)
additionally `$WORK/eval/heldout-i-j4b/HELDOUT_CONFIG.json` (the registered
A3.4 config check; J4B outcome reads stay gated by `REGISTRATION_J4B.md`).

### 5.5 Chunk-5 sealed final TEST (full 5-member ensemble, separate run)

The chunk-5 TEST is the separate sealed final evaluation
(`REGISTRATION_CHUNK5_TEST.md`, LF `f8c7d760`), run ONLY once the full
5-member Idea A ensemble exists (`require_full_ensemble`). It is NOT held-out
test (i) and it does NOT read J4B.

```
python -m evalh.run_eval --split final_test --final-test \
  --out $WORK/eval/final_test \
  --predictor adapter.predictor:IdeaABundlePredictor \
  --reference evalh.climatology:ClimatologyPredictor \
  --chunk5-sha f8c7d760baedeb359c501cd90a3e2a599fb0ddfef390ea555cf304c38214b31e \
  --registration-sha 666233aa1d1e466750d6d944b9102c84a2f32850ebeb4a97c9f3c9b56dd93246
```

### 5.4 Per-box packing (`bundle/box_configs/`)

`bundle.multibox.plan_fleet` schedules the 13 logical configs onto physical
boxes (one `boxN.json` per box; `index.json` records the plan):

- **3-box** (`bundle/box_configs/3box/box1..box3.json`, ≈54.0 h): box1 =
  `member-1, heldout-iii, n-member-5, member-4`; box2 = `n-member-1,
  heldout-ii, n-member-3, member-2, member-5`; box3 = `heldout-i, n-member-2,
  n-member-4, member-3`.
- **4-box** (`bundle/box_configs/4box/box1..box4.json`, ≈41.0 h): box1 =
  `member-1, n-member-2, n-member-5, member-5`; box2 = `n-member-1,
  heldout-iii, member-2`; box3 = `heldout-i, n-member-3, member-3`; box4 =
  `heldout-ii, n-member-4, member-4`.

Each file follows `GPU_PLAN.md` (`member-1` is box 1's first job; each
family's member-1 gates its k>1 members) and carries per-config `train_argv` /
`eval_argv` / `score_argv`, `seed`, `expected_outputs`, the owner
`APPROVAL.json` / `TIMING_GATE.json` paths and the
`bundle.e2e.run_config_job` executor. A reconstructor drops the box's file into
its `gpu_bundle`. See `bundle/box_configs/README.md` for the
`wm_n1_box_config/1` schema.

### 5.6 Amendment 4 binding (CAL-tune / CAL-cal) — required on every approval

`REGISTRATION_WM_N1_AMENDMENT_4_M1_TUNING.md` (LF `d392b300…0822f`) and the
materialised split `CAL_SPLIT_A4.json` (LF `bc4e32f2…20d46`, `A4_FREEZE.json`)
are FROZEN. Every owner `APPROVAL.json` (schema `wm_n1_approval/1`), every
per-config `APPROVAL.json` (schema `wm_n1_multibox_approval/1`, including the
`2box/c_m1only` boxA/boxB configs; `bundle.multibox.require_config_approval`)
and `APPROVAL_A_M2_5.json` (schema `wm_n1_approval_a_m2_5/1`) binds BOTH the
Amendment 4 LF-sha and the CAL split LF-sha (CRLF-safe) on top of the existing
shas; members 2-5 additionally bind `member_1.pt` and the
`WINNING_CONFIG.json` sha. The single CAL split loader/filter is
`evalh.cal_split`: the improvement-loop selection / quick-tune scoring reads
CAL-tune ONLY, while early stopping, calibration, blend weights and variant
choice read CAL-cal ONLY; `train`, `evalh.run_eval` (`--cal-half`) and
`combine.guards` all refuse a world from the other half. The winning
configuration is written by `bundle.winning_config` (schema
`wm_n1_winning_config/1`; A4.2 knobs, round log, CAL-tune scores, sha).

## 6. Exact pre-flight gate commands

```
cd output/wd01/wm_n1
$env:WD_FORBID_D="1"
python -m preflight all --data-root E:/research-dense
python -m preflight zero-residual
python -m preflight ctrl-parity
python -m preflight label-leak
python -m preflight overfit
python -m preflight e2e-smoke
python -m preflight heldout-dryrun
```

`PREFLIGHT_SUMMARY.json` (`preflight/reports/`): **`n_pass=6`, `n_fail=0`,
`all_passed=true`** — G1 zero-residual PASS (`pos<= 0 m, speed<= 0 m/s`, real
J4 TRAIN window), G2 ctrl-parity PASS (all strategies within `1e-3` m/s on
>=2 TRAIN windows), G3 label-leak PASS, G4 overfit PASS with the **REAL
modules** (`final <= 0.05`, `train/g4_real.py`), G5 e2e-smoke PASS, G6
heldout-dryrun PASS. The exact G4 number is in
`preflight/reports/overfit.json` / `PREFLIGHT_SUMMARY.json`.

Tests (one process):

```
$env:WD_FORBID_D="1"
python -m pytest -q            # 601 passed in 414.73s
```

## 7. Notes / open items

- **G2 is a real recomputation by default.** `strategy_parity_report()` replays
  the registered chunk-4 **TRAIN** windows with the REAL `ctrl.CavMover`
  (`native` 3 windows, `damping`/`pi`/`wall-v2.1` 2 each; max command
  |diff| `0` / `7.2e-7` m/s, 100 % of command rows matched) and falls back to
  the frozen `ctrl/results/parity_g2.json` when run data is absent. It never
  overwrites the frozen file and never reads HOLDOUT/CAL/chunk-5/J4B,
  `dispatcher_rows`, `insertions` or `evaluate_summary`. `wall-v2.2` has no
  recorded run yet; it is pinned by the real mover step + the ctrl unit tests.
- **G4 uses the REAL modules** (`train/g4_real.py`): the real
  `hdv.HdvResidualNet`, `arrival.IdeaAArrivalHead` and `corr.CorrectionHead`
  (with the real 64-d `corr.CavMotionSummary` projection) in a tiny-but-real
  config (`hdv d_model=16, n_heads=2, n_layers=1, ffn=32`;
  `arrival d_model=16, counter_channels=8`; `corr summary_dim=64, d_model=16`),
  optimizing the total A3.2 loss (residual NLL, differentiable rollout, arrival
  NLL, correction CRPS, paired response contrast) on 2 tiny TRAIN windows on
  CPU (real `IdeaATrainer` losses/optimiser, cosine LR schedule, 1500 steps).
  Gradients flow through the real contrast and rollout terms. The linear
  **stand-ins** (`train/g4_standins.py`) are kept only as a documented fallback:
  using them forces the report to `ok=False` and FAILS the gate.
- **G6 (i)** J4B is the **J4 wall panel re-run under wall law v2.2**
  (`REGISTRATION_J4B.md` on `zcode/wd01`, LF `2463f5be`), read ONLY by held-out
  test (i); its eval tags are DECLARED metadata (never opened by pre-flight).
  It is **not** the chunk-5 sealed final TEST (`f8c7d760`), which is the
  separate 5-member run in §5.5. Splits (ii)/(iii) use the real TRAIN PI / J4
  W-mid arms.

- **Variant N** is the v1 pure-neural comparator (v1 train CLI + the v1
  `adapter.predictor:WmN1Predictor`); it cannot handle unseen strategies by
  design.

## 8. Corpus path placeholder (FEAT1 in progress)

The GPU boxes consume the packed TRAIN corpus at the FEAT1 staging path

```
D:/rd-staging/wmn1_feat/pack_v2      # placeholder; FEAT1 featurization in progress
```

The CPU box never mounts `D:/`; the placeholder appears in
`bundle/box_configs/**/box*.json` as `corpus_source`, and the per-box argv uses
the box-local `$WORK/pack` (populated by the upload/convert stages from that
staging pack).
