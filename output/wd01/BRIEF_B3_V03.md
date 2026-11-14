# B3 brief: Part B V0 trainer v0.3

Written by the world model session on 2026-09-29, for ZCode.

**Target:** a verified v0.3 bundle on data v0.3 (chunk 3 + fix27) by Tuesday
around 10:00 local. This task builds and tests the trainer on CPU only. The
owner trains it on a rented GPU.

## Read first

- `output/wd01/PLAN_PARTB_V0_TRAINING.md`, all addenda. Addendum 5 is the
  work order; 5a is the diagnosis.
- `output/wd01/partB_diag/diag_tau30.py` and `.json`, the diagnosis.
- `output/wd01/partB_v0/` at commit ef4064a (v0.2 plus the B2 fixes). This
  is the code you extend.
- v0.2 run-1 results: `E:/research-dense/output/partB_results/v0.2_run1/extracted`
  (`eval_report.md`, `metrics.json`).

## Key finding to fix

On the SAME features, ridge and GBM beat v0.2 by about 9× at τ = 30:

| τ | ridge MAE | v0.2 MAE | v0.2 bias |
| --- | --- | --- | --- |
| 30 | 84 | 744 | +610 |
| 300 | 1361 | 4205 | −704 |

v0.2 squeezes every horizon toward the middle one. The data and features are
fine; the trainer and model are not.

## Items (all required)

### A. Model and targets

1. **Per-τ target normalisation.** Divide y by a TRAIN typical value for that
   τ (the median per τ and policy) before symlog/twohot.
2. **Twohot grid.** Cover only the realistic range, e.g. symlog [−2, 12],
   or the normalised range. Keep 255 bins or tune the count.
3. **τ conditioning.** Condition every head layer on τ (FiLM or concat), and
   add a learned per-τ output scale and bias.
4. **Inputs.**
   - Standardise every feature from TRAIN statistics, and save the
     statistics with the weights.
   - Add a summary-feature token with the same features the diagnosis used:
     per-lane counts and mean speeds, corridor-third counts, context, demand,
     policy and pre_fix.
5. **Point forecast.** Use the median of the predictive distribution.

### B. Training

6. **Selection.**
   - Each member bootstraps TRAIN worlds. Its out-of-bag (OOB) worlds are
     its validation set.
   - Early-stop on OOB mean CRPS averaged over ALL τ and both policies.
   - Evaluate every 500 steps, with patience 3 and a maximum of 8000 steps.
   - CAL is reported only, never used for selection.
7. **Regularisation ablation.** Compare weight decay (wd) / dropout 0.1 /
   small width (d_model 128) against base. The `--ablation` flag runs a
   short version of each. Choose the default by OOB.
8. **TD.**
   - The MC weight rises with τ.
   - Use n-step targets from chained on-grid labels where the data allow.
   - Add an explicit consistency penalty,
     V(s, τ) ≈ D30 + V̄(s', τ − 30).
   - Avoid off-grid τ in targets, or make the interpolation exact.
9. **Shared encoder.** Train on the damping-base rows and the native rows
   together. Native metrics are scored only on native rows.
10. **Concurrent members.** Train M = 5 members concurrently on one GPU
    (vectorised ensemble or streams). Resume must be bitwise-exact per
    member.
11. **Budget.** The budget is per member. Re-plan after about 500 steps
    from the measured ms/step, including eval, checkpoint and logging
    overhead. The cosine LR follows the re-planned total. Use a
    device-resident dataset, and bf16 on Ampere+ (fp16 + GradScaler on T4).
12. **Persistence.**
    - Save `m{k}_best.pt` whenever a new best occurs.
    - Write `member_k.pt` from the best state on EVERY path: natural finish,
      skip, or budget stop.
    - `load_members` requires all M members and names any missing.
    - Test: interrupt, then resume with a smaller `--max-steps` than the
      current step.

### C. Policy contrast

13. **Paired Δ head.**
    - Train it on CRN pairs: at a world's t0, base (damping) vs native
      restore the same state, so labels at the same (t, τ) exist for both
      sources.
    - It predicts Δdelay = damping − native directly.
    - Evaluate sign agreement, Spearman, and MAE against the Δ = 0
      baseline.
    - Leave a hook for REP-P replicate-mean targets.

### D. Calibration and evaluation

14. **Post-hoc per-τ calibration.** Fit a quantile shift/scale or an
    isotonic map on OOB TRAIN predictions. Report CAL before and after.
15. **PERMANENT baselines in the eval report.** Ridge and GBM on the summary
    features, fitted on TRAIN, reported on OOB and CAL per τ, policy and
    chunk. Reuse `partB_diag/diag_tau30.py` logic.
16. **HARD acceptance.** On OOB, the v0.3 net must beat RIDGE (MAE of the
    median, and CRPS where a ridge CRPS is defined via residual quantiles)
    at EVERY τ. Otherwise the report says clearly, per τ, where it doesn't.
    Smoke runs cannot show this. Implement the check and the report table,
    and have the final GPU run produce it.
17. **Fallback value model.**
    - A calibrated quantile-GBM (e.g. HistGradientBoosting with quantile
      loss at 0.1/0.5/0.9, or LightGBM if pinned) per τ and policy, on the
      summary features.
    - It is exposed through the same predict interface (median + quantiles
      → the planner's distribution fields).
    - `make_v0_bundle.py --value-model {net,gbm}` selects which model ships
      as V0.
    - `train` with `--value-model gbm` fits the GBM in minutes on CPU or
      GPU hosts. This IS allowed on CPU: it is not net training.
18. **Report output.** 3 decimals for coverage and PIT, a PIT histogram per
    τ, member spread, and TD-vs-MC consistency per τ.
19. **results.zip.** Whitelisted outputs only: no stray directories such as
    `ckpt_m0_old`, and no `ckpt/`.

### E. Packaging

20. **README_TRAIN / README_PLAIN.** Use `nohup` and `tail -f`, plus the new
    flags.
21. **`make_v0_bundle.py`.** Being written by B1 in `E:/rd-wt/wd01u`; do not
    edit it there. Add the `--value-model` switch only after B1 is merged,
    or leave a clearly named hook and a note.

## Constraints

- Never kill processes by image name or pattern. Only stop PIDs this task
  spawned. Never touch sumo.exe processes you didn't start.
- ZERO simulation.
- NO net training on CPU beyond smoke runs. GBM fitting is fine.
- HOLDOUT is never opened.
- No D: paths. `WD_FORBID_D=1` must stay green.
- Never write into dataset roots or backups.
- Do not edit shared records.
- maxSubagents 0.
- Paste real stdout.
