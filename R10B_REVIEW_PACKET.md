# R10B_REVIEW_PACKET - round 10b = round 10 (round 9 + fix B) + learned pre-merge follower courtesy

S-COURTESY, 2026-10-23. Branch `c4/v10r10b` (not pushed). Base: round-10 prep 4bdbbf18 + `c4/v10r9excl` + `c4/v10r10`
(d19c2066 is_cav fix, hardened round-10 gate). Nothing run on gpuR; no SUMO; local CPU, TRAIN data only
(read-only E:/agent-work/c4-v10r9fix/data/r9_train). Design: R10B_DESIGN.md.

## 1. Design in one paragraph
For every ramp-lane car its target follower is the nearest lane-1 car with `s_f <= s_ramp` (geometry, no window); a
follower owned by several ramp cars takes the nearest. A 10 -> 32 -> 32 -> 1 MLP (1,441 parameters, zero last layer,
unbounded output) maps smooth current-state inputs (bumper gap, ramp car's distance to the lane end, both speeds, relative
speed, ramp car wait/time-in-lane, detached merge-head intent, follower's own leader gap/closing) to an acceleration
offset added to that HDV follower before the plant clamp, outside the residual gate. No partner -> offset exactly 0.
Same function in training rollouts, the recorded-state replays and every runtime/scoring rollout. Default off (no tensor).

## 2. Training, trainable set, weights
| term | rows | prediction | trains | weight |
|---|---|---|---|---|
| round-9 terms (unchanged) | rollouts | full model | temporal + gate scalars | as round 9 |
| fix B `cutin_reaction` (unchanged) | post-insertion follower rows + lane-1 controls | full model | temporal + gate scalars | 2.174 |
| R10B `follower_courtesy` | lane-1 HDV transitions whose follower has a partner | `clamp(a_pre.detach() + head)` vs recorded a[t+1], Huber 1.0 | head only | 4.719 |

- Head weights get gradient ONLY from the courtesy term (`follower_courtesy_rollout_grad=false`: forward uses the head
  with detached parameters, the state adjoint through it is kept). Tests check both directions.
- Courtesy and cut-in share one recorded-state replay (detached outputs collected by the cut-in replay; identical values).
- Trainable: `temporal.*` (lr x5), `gate_g_max_logit`, `gate_tau_raw` (lr x100), `follower_courtesy.*` (lr x1000).
  Frozen bitwise: merge_pivot, merge_core, driver encoder/updater, lane gates, lane_v5, history_proj, k_embedding, residual_net.
- Weight 4.719 = gradient-norm parity with the weighted fix-B term (G_cutin 0.737 / G_courtesy 0.156, 8 batches, per-batch
  0.05-0.27); the weight only enters through the global grad clip and the logged total (docs/r10b_weight_rule.json).
- lr scale 1000 = best held-out loss in the declared grid {1, 10, 100, 1000} under the production schedule (held loss
  0.870 / 0.858 / 0.676 / 0.578; zero head 0.871). 1000 is the grid edge; not extended.
- Config `configs/c4_wm_v10round10b.toml`: extends round 2 with round 10 inline (16-file extends cap); differs from round 10
  only in name, `model.follower_courtesy*`, `loss.follower_courtesy.*`, `optim.lr_scale.follower_courtesy`, out_dir (test).
  init = `/root/autodl-tmp/c4/v5/runs/c4_wm_v10_round9/wm_s_stage300.pt`.

## 3. Tests (CPU, PYTHONPATH="src;.;tests")
- New: `tests/test_r10b_courtesy.py` (21): geometric pairing / nearest owner / zero inputs without partner; inputs at t are
  unchanged when recorded rows > t change (no leakage); flag off = bit-identical state dict + step outputs, no tensor;
  enabled zero-init head = exact no-op and every other tensor/RNG identical; offset exactly 0 without partner and on CAVs;
  no clamp on the head output; courtesy loss reaches only the head; zero-init head gets a gradient; weight 0 / no_grad off;
  shared replay gives the identical loss; cut-in and rollout losses reach the head only when rollout_grad=true; runner
  composes both terms + telemetry; refusals (weight without model flag, loss without head); **window longer than the unroll**
  (direct, shared replay, runner 12-row windows / 6-step unroll); config pins vs round 10; trainable set; weight/lr from the
  rule file; gate content; legacy-flat round trip.
- New: `tests/test_round10b_gate_verdict.py` (28) and `tests/test_round10b_gate_launcher.py` (10, via WSL).
- Final run (2026-10-23 17:37, all new + regression: cut-in, r10 config, round-10 gate verdict/launcher, merge context/core/
  core-v2 config/pivot/pivot runtime/pivot training, tests/unit/config, tests/unit/train, wmloss, wmlane, equivalence):
  **657 passed, 17 skipped** (pre-existing data-availability skips). ruff clean on all touched files.

## 4. Micro-smoke (tools/r10b_micro_smoke.py; real WmSTrainer.train_batch, local CPU)
PASS (docs/r10b_micro_smoke_summary.json, docs/r10b_micro_smoke_train_log.jsonl). Round-10b config, init round-8
stage-300 (INIT_FROM: 149 loaded; new = 8 merge_pivot + 6 courtesy-head tensors), 2 steps, batch 2, dataset windows 60 steps
with a 40-step unroll (window > unroll, the shape that crashed round 10's first gpuR smoke), each with a recorded insertion.
Run started before the weight/lr were measured, so it used weight 1.0 / lr scale 100 and the micro-smoke's base lr (mechanics only).

| step | loss | cut-in | cut-in events / rows / controls | courtesy | courtesy rows | shared replay | grad norm |
|---|---|---|---|---|---|---|---|
| 1 | 28.59 | 0.943 | 6 / 17 / 2837 | 0.610 | 229 | 1 | 4.80 |
| 2 | 23.84 | 1.175 | 7 / 20 / 2936 | 0.662 | 1260 | 1 | 1.07 |

All finite; head gradient > 0 on both steps; changed tensors = exactly the 6 head tensors + gate_g_max_logit, gate_tau_raw,
temporal.0/2 weight+bias; no frozen tensor and no merge_pivot tensor changed. (Step 1 wall time 9,628 s includes a laptop
sleep; step 2 took 1,313 s on the shared CPU.)

## 5. Offline check on recorded TRAIN windows (tools/r10b_courtesy_offline.py; docs/r10b_offline_fit.json)
104 windows x 120 s (53 v5_native, 51 genrand_plain; 26 held-out by episode), round-8 stage-300 weights (HDV acceleration
identical to round 9, which trains only merge_pivot), teacher-forced replay. "Excess" = mean minus the same-cell mean of
C3-type controls (lane-1 HDV, waiting ramp car in the zone but not adjacent), cells = speed x time headway x closing x
position (evaluation device only). All 104 windows (the head saw the fit episodes; held-out alone is too small: n 1,212):

| rows (target followers) | n | recorded excess | model before | after production schedule (90 steps, x1000) | after converged fit (400 steps) |
|---|---|---|---|---|---|
| dA 6-15 m | 7,697 | **-0.29** (SUMO full TRAIN -0.40) | +0.21 | **-0.35** (head -0.61) | -0.67 (head -0.90) |
| dA < 20, d_end < 10 m | 4,746 | -0.53 | +0.48 | -0.15 | -0.57 |
| dA < 20, d_end 10-20 m | 707 | -0.29 | +0.24 | -0.11 | -0.38 |
| dA < 20, d_end > 40 m | 14,632 | -0.02 | -0.39 | -0.11 (head +0.25) | -0.33 |
| alongside dA < 6 m | 13,834 | -0.06 | -0.35 | -0.05 (head +0.26) | -0.27 |
| far partner dA > 30 m | 2,916 | -0.19 | -0.03 | **-0.45** (head -0.47) | -0.36 |

- The round-8/9 model has the wrong sign before the merge at matched state too (+0.21 vs -0.29; R9_YIELD rollouts +0.15).
- At the production budget the head brings dA 6-15 m to -0.35 (recorded -0.29, SUMO -0.40); one-step bias on those rows
  +0.90 -> +0.29 m/s^2; held-out courtesy loss 0.871 -> 0.578.
- Converged, the head overshoots (-0.67): its target is recorded minus the model, and the model also over-predicts
  acceleration of NON-adjacent cars in the same cells near the merge (C3-cell base bias +0.40), which the head can only fix
  on partner rows. Far partners get -0.47 (recorded -0.19): the main risk (section 6).
- Event profile (excess vs t - t_merge, after production schedule): recorded -0.23 / -0.29 / -0.40 / -0.25 at -6 / -3 / -2 /
  -1 s; model before +0.24 / +0.18 / +0.16 / +0.31; after -0.27 / -0.62 / -1.08 / -0.95. Onset right, last 2 s about
  3x too strong at matched state.
- Option B (learned own-state control regressor to remove non-adjacent bias) was tried and rejected: not identified (head
  target at dA 6-15 m moved from -1.14 to +0.12 with the regressor's training length).

## 6. Risks / what the reviewer should decide
1. **Overshoot and far partners** (section 5): a one-step-correct head can make adjacent followers slow ~0.5-1 m/s^2 more
   than identical non-adjacent cars in the last 2 s and for far partners. The 90-step budget limits it (-0.35 at dA 6-15 m)
   but production windows are larger than the offline batches. Required post-run reads: rollout event profile and
   negative controls (dA > 30 m, d_end > 40 m, alongside, ramp car behind) vs SUMO; reject if far-partner or tail braking
   gets worse than round 10.
2. **lr x1000 on a fresh head** (lr 0.05 -> 0.03): chosen by the pilot rule; the head only sees its own loss, but its
   output acts in every rollout from step 1. If the reviewer prefers caution, x100 gave held loss 0.676 (vs 0.578).
3. **Base errors absorbed** (design risk 2) and **weak wait input** (time in lane saturates without merge_impatience).
4. **Interaction with the merge rule and fix B**: more follower slowdown + softer post-insertion reaction -> watch collisions,
   TTC, min gaps, stuck ramp cars (R9_STUCK_DIAG), merge counts.
5. **Attribution**: courtesy term moves only the head (pre-merge partner rows); fix B and rollout terms move temporal+gate.
   Read pre-merge [-6, 0) s target-follower excess for the head, post-insertion [0, 6] s min accel for fix B.
6. Offline numbers are teacher-forced at recorded state on 104 local windows with round-8 weights; not rollouts.

## 7. Gate and launch
`scripts/round10b_gate.sh [REPO_DIR]` (LF, `bash -n` clean): round-10 hardened structure (set -u/pipefail/noclobber, stale
artifact refusal, lock, R10B_INIT/R10B_ROOT/R10B_PYTHON), smoke batch 4 at 120/300 s, 2+2 steps, `selection.val_max_windows=4`,
`curriculum.max_hours=1.5`; verdict `python -m wms.train.round10b_gate` = round-10 checks (merge TF counters, frozen pivot,
cut-in census) + courtesy finite/positive with rows > 0, init report clean except the new zero-init head tensors, strict
reload with the head enabled, frozen + merge_pivot bitwise, a fix-B tensor and the head read-out moved; then exec the
full run (`runs/c4_wm_v10_round10b`, `selection.val_max_windows=16`). Default repo dir `$L/repo_r10b`. Not run anywhere.
Commits: SEE_COMMIT.
