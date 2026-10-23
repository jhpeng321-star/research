# R11_REVIEW_PACKET - round 11 = round 10b + the structural courtesy fix (launch-ready, NOT launched)

S-R11, 2026-10-24. Branch `c4/v10r11` (from `c4/v10r10b` 8ee9341a; not pushed). Design: R11_DESIGN.md.
Nothing was run on gpuR (round 10b is still training there) and there was no SUMO run. Local CPU, TRAIN data only:
read-only filtered roots of `E:/agent-work/c4-v10r9fix/data/r9_train` (`c4-v10r10b/tmp_r10b/roots`). No sealed,
holdout or eval-seed path was opened. The coordinator decides the launch after round 10b's reject checks and an
independent review.

## 1. What changes (all default-off switches; flag off is bit-identical)
| part | switch | effect |
|---|---|---|
| inputs | `model.follower_courtesy_version = 2` | 3 inputs appended to the 10 round-10b ones: ramp partner's acceleration at t and t-1 (state / short motion history), `asinh(t_zone/10 s)` with a carried, causal merge-zone time (`courtesy_zone_s`, carried by rollout and replay, t0 value from the observed history). The dead lane-age column is kept, so a round-10b head zero-expands exactly. |
| loss | `loss.follower_courtesy.contrast = true` | target = recorded minus the kernel-matched residual of UNPARTNERED lane-1 HDV rows of the same batch (R9_YIELD covariates: speed, time headway, closing speed, position; Gaussian kernel, Scott bandwidth, one pseudo-row shrinkage). The head learns the adjacency effect only. |
| fix B | `loss.cutin_reaction.exclude_partnered_controls = true` | partnered lane-1 rows (head pairing on the recorded state) are removed from the control anchor. Temporal/gate never see `a_pre + head`. |
| init | `model.follower_courtesy_init_zero` (default false) | false: round-10b final head, zero-expanded to version 2 (`INIT_FROM.json`: `expanded = [follower_courtesy.net.0.weight]`, `follower_courtesy_init = "checkpoint"`); true: fresh zero-init head. |
Kept: geometric pairing, offset exactly 0 without a partner and on CAVs, unbounded read-out, Huber 1.0,
`rollout_grad = false`, merge_pivot frozen, round-9 decisions, fix-B weights. A stochastic head was not added (design
section 4).

## 2. Trainable set and rules (docs/r11_weight_rule.json)
- Trainable as in 10b: `temporal.*` x5, `gate_g_max_logit` / `gate_tau_raw` x100, `follower_courtesy.*` x s
  (12 tensors). Everything else is frozen bitwise.
- Weight = `G_cutin(0.737) / G_courtesy(w = 1)` at the round-11 init (geometric mean over 8 production-like batches):
  r10b-head init 1.086 (config); zero-init head 9.067 (the gate passes it when `R11_HEAD_INIT=zero`).
- lr scale = best held-out contrast loss in {1, 10, 100, 1000} after the production schedule, from the round-11 init:
  r10b-head init **x100** (held 0.656 / 0.604 / **0.566** / 0.585; init alone 0.666); zero-init head **x1000**
  (0.698 / 0.682 / 0.606 / **0.578**). Config = x100; the gate passes x1000 with `R11_HEAD_INIT=zero`.

## 3. Tests (CPU, PYTHONPATH="src;.;tests")
- `tests/test_r11_courtesy.py` (24) covers:
  - v2 features: appended, exact values, zero without a partner, bad version refused.
  - zone counter: causal, geometric, resets, pre-t0 history.
  - no leakage (future rows changed: inputs at t identical); rollout carries the zone state; replay zone input = the
    counter.
  - version 1: unchanged, no zone state. All defaults give the round-10b losses and plans bit-identically.
  - v2 head: exact no-op at init, exact 0 without a partner and on CAVs.
  - contrast on synthetic rows recovers the injected adjacency effect (-0.4) under a state-generic base error. No
    controls gives a zero shift; no support shrinks to the control mean.
  - matching features = R9_YIELD covariates, not model inputs. Contrast loss = explicit recomputation and reaches the
    head only. Window longer than the unroll (direct and shared replay).
  - fix-B exclusion: controls minus partnered rows, events unchanged, pairing = the head's pairing.
  - runner composes the R11 terms and their telemetry; refusals; gradient scope (rollout_grad off/on).
  - r10b head zero-expands to the identical function; the trainer init loads the expanded head or a zero head and
    writes the report.
  - config pins vs 10b (only the R11 keys differ; round-9 decisions and fix B inherited); trainable set; weight/lr
    from the rule file; gate content; legacy-flat round trip.
- `tests/test_round11_gate_verdict.py` (25) and `tests/test_round11_gate_launcher.py` (14, via WSL, including the
  head-init switch and its weight/lr overrides).
- Regression: the new tests plus r10b courtesy, cut-in, round-10/10b gate verdict/launcher, merge context/core/core-v2
  config/pivot/runtime/training/exclusion/hybrid, tests/unit/{config,train,model}, equivalence. Result: **726 passed,
  17 skipped** (data-availability skips) before the rule file existed. The one failure then was the rule-file test,
  which passes now (38/38 on the R11 + launcher files after the rule). ruff is clean on all touched files.

## 4. Offline check (tools/r11_courtesy_offline.py; docs/r11_offline_fit.json)
207 TRAIN windows x 120 s (54 held out by episode; 66,024 partner rows). Round-8 stage-300 base (HDV acceleration as
round 9), teacher-forced replay with the version-2 model. Four arms on identical windows with the production schedule
(lr scale by rule), plus round 11 started from a round-10b-head stand-in (arm A's production head, zero-expanded).
"Excess" is R9_YIELD's matched excess vs C3-type controls in speed x headway x closing x position cells
(evaluation only; the same definition as M1/M3). **Held-out episodes:**

| event time t - t_merge (s) | -6 | -3 | -2 | -1.5 | -1 | -0.5 |
|---|---|---|---|---|---|---|
| recorded excess | -0.25 | -0.36 | -0.47 | -0.43 | -0.28 | -0.08 |
| base, head 0 | +0.12 | +0.13 | +0.17 | +0.21 | +0.18 | +0.23 |
| A round-10b loss (x1000) - reviewer's problem | -0.56 | -0.81 | -1.24 | -1.13 | -1.06 | -0.89 |
| B round-10b loss + v2 inputs | -0.64 | -0.87 | -1.24 | -1.17 | -1.19 | -1.08 |
| C contrast, v1 inputs (x100) | -0.07 | -0.13 | -0.16 | -0.14 | -0.19 | -0.17 |
| **D round 11, zero-init head (x1000)** | -0.18 | -0.15 | -0.38 | -0.36 | -0.29 | -0.19 |
| **D round 11 from the r10b-head init (x100, default)** | -0.16 | -0.18 | -0.31 | -0.18 | -0.10 | -0.06 |
| D zero init, long fit (400 steps) | +0.00 | -0.04 | -0.11 | -0.07 | -0.07 | -0.03 |

Mean of the -3..-0.5 s bins (the M1 statistic, here one-step): recorded -0.32; A -1.03; D zero -0.27; D from r10b
-0.17. The overshoot is gone in both round-11 variants. Head output in the last 3 s: A -1.21, D zero -0.49, D from
r10b -0.37.

| held-out group | recorded excess | head 0 | A (r10b loss) | D zero init | D from r10b head | abs bias after, D from r10b / A |
|---|---|---|---|---|---|---|
| dA 6-15 m | -0.28 | +0.13 | -0.43 | -0.23 | -0.09 | +0.56 / +0.14 |
| dA < 20, d_end < 10 m | -0.55 | +0.28 | -0.72 | -0.44 | -0.25 | +0.32 / -0.02 |
| far partner dA > 30 m (neg. control) | -0.10 | -0.02 | -0.36 | -0.03 | +0.12 | +0.51 / +0.19 |
| dA < 20, d_end > 40 m (neg. control) | -0.01 | -0.38 | -0.14 | -0.09 | -0.07 | +0.07 / -0.03 |
| alongside dA < 6 m (neg. control) | -0.04 | -0.37 | -0.13 | -0.09 | -0.06 | +0.03 / -0.04 |
| head p0.1 / p99.9, share < -3 | - | - | -2.09 / +3.58, 0 | -2.39 / +1.96, 0 | -1.85 / +1.85, 0 | |

Findings:
1. **The structural fix works in excess terms.** The new inputs alone do not fix it (B is as bad as A). The contrast
   does (C, D). D with the v2 inputs follows the recorded peak at -2..-1 s better than C (C is flat at about -0.15).
   On the negative controls the excess stays within 0.1-0.2 of recorded; the round-10b loss is off by 0.1-0.26.
2. **Absolute caveat (important for M3 and M5).** The round-8/9 base over-predicts the target follower by
   +1.2..+1.6 m/s^2 in the last 3 s. Unpartnered cars matched on headway carry about +0.85..+1.1 of that. The contrast
   head now leaves that generic part to fix B (whose controls are now unpartnered rows only; `r_hat` is recomputed
   with the live model). Offline, with the base frozen, the one-step ABSOLUTE prediction in the last 3 s stays about
   +1.0 vs recorded 0.0. Far partners keep an absolute bias of +0.5, already +0.46 with head 0. The round-10b head
   masked this (and over-brakes in excess terms). Whether fix B removes the generic over-prediction within 90 steps is
   untested offline. If it does not, rollout followers may still accelerate into the merge gap (fewer merges, R9_STUCK
   bistability). Read M3 (absolute) together with the head-zero arm, and M5 stuck cars.
3. **The matching metric matters.** Matching on (v, gap) instead of (v, headway) finds base-accurate unpartnered cars
   and gives a last-2-s head target of -1.2. Headway follows the registered evaluation (design section 2). A
   zone-occupancy coordinate destroys support (median effective controls ~1 in the last 2 s).
4. **Init.** From the r10b-like head (x100) the excess undershoots at -1.5..-0.5 s (-0.18/-0.10/-0.06 vs
   -0.43/-0.28/-0.08), because 90 steps do not fully re-fit the inherited head. The zero-init head (x1000) is closest
   to recorded at every event time and on dA 6-15 m. The real round-10b head is more extreme than the stand-in
   (training telemetry -3.9..+4.7), so inheriting it carries more to unlearn. **Recommendation for the coordinator:**
   prefer `R11_HEAD_INIT=zero` (with `R11_INIT` = the round-10b checkpoint, so temporal/gate come from 10b) unless the
   round-10b checks favour keeping its head. The default stays "checkpoint" as instructed.
5. Offline numbers are teacher-forced one-step at recorded states with a frozen base, on local windows. They are not
   rollouts.

## 5. Micro-smoke (tools/r11_micro_smoke.py; docs/r11_micro_smoke_summary.json)
Real `WmSTrainer.train_batch`, round-11 config, local CPU, 2 steps, batch 2, 60-step windows with a 40-step unroll
(window > unroll), each with a recorded insertion. Init: a local round-10b stand-in (round-8 weights + the R10B
offline production head, 10 inputs; `tmp_r11/make_r10b_like.py`, mechanics only). Init report: loaded 163, expanded 1
(`follower_courtesy.net.0.weight`), new 0, shape-mismatch 0, unused 0. Results: see section 5a (filled at completion).

## 6. Gate
`scripts/round11_gate.sh [REPO_DIR]` (LF, `bash -n` clean, blob has no CR). It has the round-10b hardened structure:
set -u/pipefail/noclobber, stale-artifact refusal (smoke_r11, c4_wm_v10_round11, logs, `.round11_gate.lock`), atomic
lock, R11_ROOT/R11_PYTHON/R11_INIT (default `$L/runs/c4_wm_v10_round10b/wm_s_stage300.pt`), R11_HEAD_INIT
(checkpoint|zero; zero also passes the zero-init weight/lr). The smoke runs batch 4 at 120/300 s, 2+2 steps,
`selection.val_max_windows=4`, `curriculum.max_hours=1.5`. Verdict `python -m wms.train.round11_gate`: round-10b checks
(round-10 merge/fix-B census, courtesy finite/positive/rows) + contrast controls > 0 and finite shift + the
partnered-control exclusion logged + an init report matching the requested mode + strict reload with a VERSION-2 head
and rollout_grad off + frozen and merge_pivot bitwise vs the (zero-expanded) init + the head moved vs its init. The full
run then execs (`runs/c4_wm_v10_round11`, `selection.val_max_windows=16`). Default repo dir `$L/repo_r11` (ship as for
10b). Not run anywhere.

## 7. Reject criteria for the round-11 run (register BEFORE exec; same tools as 10b)
Reuse `tools/r10b_checks` (run_all.sh) with `R10B_RUN=$L/runs/c4_wm_v10_round11` and `R9_CK=<round-10b final>`. The
reference arm is labelled `r9` in the output and the round-11 arm `r10b`; the head-zero arm is round 11 with
`follower_courtesy.net.4.*` = 0 (make_headzero works for the version-2 head). Rollouts carry the zone counter through
`model.rollout`. Thresholds as registered for 10b:

| id | reject the head (zero it) if | round-11 specific reading |
|---|---|---|
| M1 | rollout target-follower excess, mean over [-3, 0) s < -0.9 | offline expectation -0.2..-0.3 (recorded -0.32) |
| M2 | far placebo dv(10 s) < -0.5 or CI excludes 0 (braking) | - |
| M3 | \|model - recorded\| > 0.3 on dA > 30 / d_end > 40 / alongside, one-step and rollout; ramp car behind must be exactly 0 | far partners can fail on the base alone (+0.46 offline with head 0). Attribute with the head-zero arm: reject the HEAD only if the head makes the group worse than head-zero; flag the base otherwise |
| M4 | head regularly < -3 m/s^2 on any group | offline p0.1 -1.85..-2.4, share < -3 = 0 |
| M5 | collisions or TTC worse than round 10b / round 9: reject the round | also stuck ramp cars and merge count vs 10b (absolute caveat, section 4.2) |
| M6 | fix-B reads as R10_REVIEW | on round 11 and head-zero |
| M7 | 90 steps completed; contrast telemetry (`follower_courtesy_contrast_*`, `cutin_reaction_n_partnered_controls_excluded`) finite every step | - |
| M8 (new) | - (read) | one-step absolute last-3-s target-follower bias and the C3-control bias at the merge (does fix B remove the generic over-prediction?) |

## 8. Commits
86fb7f12 (code), 42f62ca4 (contrast matching, config, gate, tests), 50204cff (verdict tests), 76fe0e65
(offline fit, rules, config values, design). See `git log c4/v10r11`.
