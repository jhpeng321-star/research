# partB_v0 — training + evaluation CLI (WD-01 Part B round 0)

Platform-independent implementation of `PLAN_PARTB_V0_TRAINING.md`
(+ addenda 2/3/3a; the Colab addendum is superseded).  One CLI trains
the policy- and tau-conditioned value function `V(s, tau, policy)` on
the V0 dataset and evaluates it on CALIBRATION.  No Dockerfile here —
the Docker wrap is a separate task.

```
python -m partB_v0 train  --data <dataset_root> --out <dir> [flags]
python -m partB_v0 eval   --data <dataset_root> --out <dir> [flags]
python -m partB_v0 verify --out <dir>
```

Run from `output/wd01/` (or put it on `PYTHONPATH`).  CPU is for smoke
tests only (owner rule: no CPU training); real runs target a rented T4 /
L4 (`--device cuda`, fp16 autocast + GradScaler, T4 has no bf16).

`verify` re-hashes every OUTPUT_MANIFEST entry, checks that results.zip
contains exactly the manifest files plus the manifest itself, and that
results.zip.sha256 pins the zip — run it after downloading the bundle.

## Flags (train)

| flag | default | meaning |
|---|---|---|
| `--data` | `$WD01_V0_DATA` else `<RESEARCH_DENSE_ROOT>/output/wd01_partB_v0data` | dataset root (MANIFEST.json is verified: its own sha256 is the data version tag and every listed file is sha256-checked) |
| `--out` | `$OUT_DIR` else `/out` | output dir (addendum 3a: set `OUT_DIR=/workspace/out` on RunPod/Vast) |
| `--members` | 5 | ensemble size M (bootstrap over TRAIN worlds, distinct seeds) |
| `--seed` | 20260928 | master seed; member m uses `seed + 1000*m` |
| `--batch-size` | 512 | MC batch (TD batch is half) |
| `--device` | auto | `auto` (cuda if available), `cuda`, `cpu` (`WD_DEVICE` env respected) |
| `--time-budget-min` | 75 | wall-clock budget; a timing probe (K steps first) prints ms/step, the projected total, and scales steps/member so training fits ~85% of the budget (rest is eval + overhead) |
| `--ckpt-every-min` | 10 | wall-clock checkpoint cadence (also saved at member end) |
| `--max-steps` | – | fixed steps/member (overrides budget scaling) |
| `--value-model` | `net` | `net` = the M-member value net; `gbm` = the calibrated quantile-GBM fallback value model (B3 item D17) — fits on CPU in minutes, no net training, then runs the SAME evaluation and bundle |
| `--gbm-max-iter` | 300 | max HGB iterations per quantile model for `--value-model gbm` (early-stops after 15 flat iterations; smoke runs use a small value) |
| `--features` | `oracle` | `oracle` = the historical V0 features (demand/context include the exact future-arrival schedule — offline upper bound, PARTB_LEAK_AUDIT.md); `deployable` = RSU-observable inputs only (v0.3.1 sidecar `<src>_feat_dep.npz`: observed entrance/CAV-entry counts over (t−60,t], (t−180,t], (t−300,t], configured demand level; no future schedule, no future CAV share, no time-to-branch, no true penetration).  gbm path only (the net refuses deployable) |
| `--compare-metrics` | – | metrics.json of the oracle run; with `--features deployable` the eval report gains the with-vs-without-schedule 4-way table (section 8) |
| `--smoke` | off | tiny run (6 steps/member, capped eval, small batch) — numbers are NOT meaningful |
| `--chunks` | all | e.g. `chunk1` (v0.1 trains chunk 1; v0.2 adds chunk 2 + `pre_fix`) |
| `--no-verify` | off | skip per-file sha256 verification (speed) |
| `--exclude-superseded` / `--include-superseded` | exclude on | skip MANIFEST sources marked `superseded_by` |
| `--no-resume` | off | ignore existing checkpoints, start fresh |
| `--no-eval` | off | skip the evaluation phase |

`eval` flags: `--data --out --device --chunks --no-verify
--exclude-superseded --value-model --features --compare-metrics --smoke`.  It re-evaluates the model
already in `--out` (must match how it was trained) and rewrites
`eval_report.md`, `metrics.json` (eval section), the env report and the
results bundle.

## The GBM value model (`--value-model gbm`)

`train --value-model gbm` fits the calibrated quantile-GBM fallback
value model (BRIEF_B3_V03 item D17) and then runs the SAME CAL
evaluation, metrics, OUTPUT_MANIFEST and results.zip as the net path —
**CPU is fine and expected** (it is not net training; minutes, not
hours).  Per tau it fits `HistGradientBoostingRegressor(loss="quantile")`
at quantiles 0.05/0.1/0.25/0.5/0.75/0.9/0.95 for `total_delay_ref_s`
(0.1/0.5/0.9 for the diagnostic exposure/brake heads) plus a collision
classifier, on the diagnosis summary features
(`partB_diag/diag_tau30.py`: per-lane counts + mean speeds,
corridor-third counts, context, demand, `policy_id`, `pre_fix`), TRAIN
rows only.  Quantiles are conformal-shifted per (tau, target, quantile)
on a 2-fold TRAIN hold-out, monotone-enforced, and shipped with a
`median + quantiles` predict API for the planner.

Outputs: `gbm_value.pkl` (pickled models; sha256 pinned in the sidecar
and OUTPUT_MANIFEST) + `gbm_value.json` (sklearn version, feature list,
shifts, data MANIFEST sha).  `eval --value-model gbm` re-evaluates;
re-running the same train command skips the fit when the sidecar's data
MANIFEST sha matches.  All sklearn fits run thread-limited (2 OpenMP
threads) with early stopping — unbounded HGB thrashes on a box whose
CPU is saturated (measured 33 min/fit unlimited vs ~5-9 s limited).

```bash
# AutoDL (CPU is fine for gbm; pick the CPU-rich instance type)
cd /root/autodl-tmp/partB_v0_export          # this package on the volume
export OUT_GBM=/root/autodl-tmp/out_gbm_v03
nohup python -m partB_v0 train --value-model gbm \
  --data /root/autodl-tmp/wd01_partB_v0data \
  --out $OUT_GBM > ${OUT_GBM}.train.log 2>&1 &
tail -f ${OUT_GBM}.train.log
# re-evaluate / rebuild the bundle later:
python -m partB_v0 eval --value-model gbm --data \
  /root/autodl-tmp/wd01_partB_v0data --out $OUT_GBM
```

The net path on AutoDL (GPU instance) is unchanged:

```bash
export OUT=/root/autodl-tmp/out_v03
nohup python -m partB_v0 train \
  --data /root/autodl-tmp/wd01_partB_v0data \
  --out $OUT --device cuda > ${OUT}.train.log 2>&1 &
tail -f ${OUT}.train.log
```

## The v0.3 net (`net` subcommand, B3b)

`python -m partB_v0 net` trains the REDESIGNED v0.3 net (BRIEF_B3_V03
items A1-A5, B6-B11, C13, D14; see `output/wd01/CHANGES_V03_NET.md`):
per-tau-normalised targets over a realistic twohot range, FiLM tau
conditioning in every head layer with a learned per-tau output
scale/bias, TRAIN-standardised inputs + the diag summary-feature token,
median point forecast, M members trained CONCURRENTLY on one device
(vectorised ensemble), early stopping on TRAIN world-grouped OOB CRPS
averaged over ALL tau and both policies (CAL is reported only, never
selected on), tau-rising MC weight with on-grid n-step TD targets and a
consistency penalty, a shared encoder over damping+native rows, a
paired Delta head on the t0 CRN restore pairs, post-hoc per-tau
calibration fitted on OOB, and the acceptance tables (ridge/GBM
baselines + HARD acceptance, and the v0.3.1 deployable BAR).

```bash
export OUT=/root/autodl-tmp/out_v03net
nohup python -m partB_v0 net \
  --data /root/autodl-tmp/wd01_partB_v0data \
  --out $OUT --device cuda > ${OUT}.train.log 2>&1 &
tail -f ${OUT}.train.log
# optional regularisation search (~15 min), winner becomes the default:
#   python -m partB_v0 net ... --ablation
```

Key flags (all others as `train`): `--members 5`, `--batch-size 512`,
`--td-batch 256`, `--pair-batch 128`, `--lr 3e-4`,
`--weight-decay 0.01`, `--dropout 0.0`, `--d-model 192`,
`--eval-every 500` (OOB eval interval), `--patience 3`,
`--max-steps-cap 8000`, `--replan-after 500` (re-plan the budget from
the MEASURED ms/step incl. eval/ckpt overhead; the cosine LR follows),
`--dtype auto` (bf16 on Ampere+, fp16 + loss scaling on T4, fp32 on
CPU), `--w-cons 0.25` (consistency penalty), `--w-delta 1.0`,
`--mc-min 0.5` (MC weight = mc-min + (1-mc-min)*tau/570), `--ablation`,
`--variant {auto,base,wd,dropout,d_model128}`, `--cal-method
{shift,isotonic}`, `--features {oracle,deployable}` (the net sizes its
input projections from the saved stats, so deployable sidecars work),
`--smoke`, `--no-resume`, `--no-eval`.

Outputs add `stats.json` (TRAIN standardisation + per-tau target
medians + delta scales — REQUIRED by the eval loader, ships with the
weights), `calibration.json` (D14 maps) and `ckpt/net_state.pt` +
`ckpt/m{k}_best.pt` (stacked resume state + per-member best).  Resume
is bitwise per member (stacked params/moments/RNG restored exactly).
The eval report contains: median/CRPS/coverage per tau, CAL
before/after, member spread, TD-vs-bootstrap consistency, the Delta
head vs the dV=0 baseline, the v0.3.1 deployable BAR, and the permanent
ridge/GBM baselines with the per-tau HARD acceptance table.

## Outputs (written to `--out`)

- `member_{i}.pt` — final weights per member (plain `state_dict`,
  loadable with `torch.load(..., weights_only=True)`), selected by CAL
  native-row CRPS at tau=300 (best eval, else last state).  Written from
  the best state on EVERY path — natural finish, the skip path
  ("already >= planned; skipping"), and deadline/budget stops — and
  rebuilt from the persisted best checkpoint when the member file is
  missing (addendum 4c; the sidecar records `weights_source`).
- `member_{i}.json` — sidecar: model config, member seed, bootstrap
  worlds, steps, CAL CRPS, data MANIFEST sha, git commit.
- `gbm_value.pkl` + `gbm_value.json` — the `--value-model gbm` model and
  sidecar (instead of member files).
- `ckpt/` — resume checkpoints per member:
  `m{i}_state.pt` (model state_dict) + `m{i}_state.prev.pt`,
  `m{i}_optim.pt` (AdamW moments + EMA target net + scaler) + `.prev`,
  `m{i}_meta.json` (config, step, RNG states, data MANIFEST sha) +
  `.prev`, plus `m{i}_best.pt` + `m{i}_best.json` — the best-state
  weights, saved at every new best so a restart never loses them
  (addendum 4c).
- `metrics.json` — train config, probe (incl. the effective batch after
  any OOM fallback), per-member report, OOM events, eval section.
- `eval_report.md` — the plan's evaluation items 1–5 (CRPS vs TRAIN
  climatology per tau x policy x chunk, randomized PIT with 3-decimal
  coverage/PIT means and a PIT histogram per tau; TD residuals; policy
  contrast at t0 with sign agreement / Spearman / MAE vs the dV=0
  baseline plus the single-realisation + 10 s-offset caveat; diagnostic
  P_coll Brier + exposure CRPS (G-1/G-2 gated); planner-call speed
  50 plans x M5 x K16), PLUS the B3a permanent sections: **6. Baselines
  on CAL** (climatology / ridge / point-GBM on the diagnosis summary
  features, fitted on TRAIN — MAE(median) and CRPS per tau x policy x
  chunk) and **7. HARD acceptance vs RIDGE per tau** (the evaluated
  model must beat ridge on BOTH MAE and CRPS at every tau, or the table
  shows exactly where it fails; overall PASS/FAIL line).  Sections 6-7
  run for BOTH value models.
- `logs/train.log` (or `eval.log`) — everything printed is teed here.
- `env_report.json` — python/torch/CUDA/cuDNN/driver/GPU name+memory/
  CPU, data MANIFEST sha256, git commit (git optional; `unknown` in a
  container), effective batch + OOM events.
- `OUTPUT_MANIFEST.json` — sha256 of every DELIVERABLE file, chosen by
  an explicit whitelist: `member_*.pt/.json`, `gbm_*.pkl/.json`,
  `metrics.json`, `eval_report.md`, `env_report.json`, `logs/` and the
  manifest itself.  `ckpt/` (resume-only state) and anything else in
  OUT_DIR (e.g. a stray `ckpt_m0_old/`) are excluded from the manifest
  and the zip (addendum-5 item 6 / B3 item D19).  The manifest covers
  exactly the contents of results.zip minus OUTPUT_MANIFEST.json itself
  (the manifest ships inside the zip but cannot self-list);
  `python -m partB_v0 verify --out <dir>` checks both relations.
- `results.zip` + `results.zip.sha256` — the bundle to download
  (deliverables only, no checkpoints).

After success the CLI prints `DONE — download <OUT_DIR>/results.zip`,
then rebuilds the bundle silently one last time so every
OUTPUT_MANIFEST hash matches the finished log; the final zip size and
sha256 are printed to stdout (the log is frozen at that point) and
self-checked.  With `KEEP_ALIVE=1` it then sleeps until stopped (cloud
containers; unset/0 exits normally — the Docker entrypoint decides).

## Re-running the SAME command (cloud restart case)

Re-running the identical `train` command on an OUT_DIR whose training
already completed does NOT retrain: completed members are skipped
(logged `already at step N >= planned N; skipping`), the evaluation is
re-run only when its outputs are missing or stale (any member
retrained, or eval_report/metrics missing) and is otherwise skipped
with the old eval section kept, and the OUTPUT_MANIFEST + results.zip
are rebuilt consistently at the end.  metrics.json keeps the original
training records for skipped members.  If a member was interrupted
mid-way, the restart resumes exactly that member and leaves earlier
members' weights untouched.

## Resume semantics

Resume is automatic: a `train` run with checkpoints in `--out` picks up
every member at its saved step (weights + AdamW moments + EMA target +
RNG states) and skips members already at the planned step count.
`--no-resume` forces a fresh start.  Details:

- Each save rotates the previous checkpoint to `.prev`; a truncated or
  corrupt latest triple falls back to `.prev` automatically (logged).
- A checkpoint whose `data_manifest_sha256` differs from the loaded
  dataset is refused (delete `ckpt/` to retrain on other data).
- Batch sampling is a pure function of (member seed, step), so a
  resumed run reproduces the uninterrupted run exactly (tested to
  < 1e-6 on CPU; bitwise in practice) when the plan (steps/member) is
  unchanged.  Changing `--max-steps`/budget between runs re-plans the
  LR-cosine and lambda_TD schedules for the remaining steps.
- `KEEP_ALIVE` never affects checkpoints; a killed container loses at
  most `--ckpt-every-min` minutes.
- `load_members` (eval) requires ALL expected members and names every
  missing `member_k.pt` in its error — it never stops silently at the
  first gap (addendum 4c).  A member skipped because its checkpoint is
  already at the planned step still gets its `member_k.pt` rewritten
  from the persisted best state, so the v0.2 "resume wrote no member
  file" failure cannot recur.

## CUDA OOM fallback

If the probe or a training step hits `torch.cuda.OutOfMemoryError`, the
batch halves (`512 -> 256 -> ...`, floor 64; TD batch floor 32), the
step is retried, and the event is logged and recorded in
`metrics.json` (`oom_events`, `probe.batch_size`), `env_report.json`
(`batch_size_effective`, `oom_fallback`, `oom_events`) and the
checkpoint sidecars — a resumed run keeps the reduced batch.  At the
batch floor the error is re-raised (not fixable by shrinking).

## Data contract notes

- Splits TRAIN/CALIBRATION only; requesting HOLDOUT is a hard error and
  no HOLDOUT source exists in the V0 manifest (never built).
- `policy`: source `base` -> `damping_fleet`, `native` -> `native`
  (extensible table; MANIFEST policy prose is provenance-only).
- Row weights equalise native vs damping per tau (native tau=570 TRAIN
  rows are thin: 141 chunk1 + 147 chunk2).
- TD applies at tau >= 60 with the target net queried at `tau - 30`
  (learned interpolation for the off-grid 90/270/540; tau=60->30 is
  on-grid).  A grid-chained TD target would equal the MC label exactly
  (builder-verified identity), so it adds no signal.  TD rows are used
  even where the MC label is incomplete — that is how supervision
  extends past the data end.  lambda_TD mixes MC/TD, 0.5 -> 0.8.
- P_closing_pair and future_min_ttc_q05 have NO V0 labels: optional
  heads, disabled by default (no invented labels).

## Local tests

```
python -m pytest output/wd01/partB_v0/tests -q          # CPU, tiny steps
WD_FORBID_D=1 python -m pytest output/wd01/tests -q     # path-hook suite
```

Canonical dataset tests run READ-ONLY against
`E:/research-dense/output/wd01_partB_v0data` and are skipped when it is
absent; everything else uses tmp dirs.
