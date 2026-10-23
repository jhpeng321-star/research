# R12_REVIEW_PACKET - round 12 = round 9 + calibrated car-following base + round-11 courtesy head (launch-ready, NOT launched)

S-R12, 2026-10-24. Branch `c4/v10r12` (from `c4/v10r11` c476cad4 + S-BASEDIAG final 116824b7; not pushed).
Design: R12_DESIGN.md. Nothing ran on gpuR or cpuN; the only remote access was one read-only `scp` of the round-9 final
checkpoint (`runs/c4_wm_v10_round9/wm_s_stage300.pt`, sha256 af9bc643...; INIT_FROM.json) into `tmp_r12/ck/`.
Local CPU only, TRAIN data only:
- the filtered r9_train roots (`c4-v10r10b/tmp_r10b/roots`), relabelled train for the smoke only in `tmp_r12/roots`;
- S-BASEDIAG's TRAIN captures.

No sealed, holdout, cal-split or eval-seed path, hetdata_truth or vType file was opened. Launch right after round 11,
if an independent review says GO.

## 1. What changes (default-off switches; flag off = round-11 behaviour, tests)
| part | switch | effect |
|---|---|---|
| A | `model.driver_param_mode = "physical"` + `model.driver_param_init = "centre"` | V6ENC full-range settings with a learned centre. After loading round 9, the head rows of the five longitudinal means are zeroed, so every driver starts at the centre (physics defaults a 2.49, b 4.2, T 1.135, s0 2.6) and per-driver deviation is re-learned from prediction error. |
| B | `model.residual_accel_reinit = true` | Delta row of `residual_net.accel[-1]` zeroed: the dead tanh head restarts at 0 with a live gradient. |
| C | courtesy v2 + contrast, `follower_courtesy_init_zero = true` | The round-11 head, unchanged code, fresh zero head. Weight 9.067, lr x1000 (R11 zero-init rule). |
| D | `loss.recorded_replay_present = "filled"` | Replay STATE = filled window presence. Supervision = `loss_valid` only. |
| fix B | `loss.cutin_reaction.weight = 0` | Dropped (R10_DIAG: bought nothing). Kept as a diagnostic in the checks. |
| gate | `gate_` frozen | R10_DIAG: non-converging downward drift under the clipped, position-dominated gradient. |

Kept from round 9 (pinned by tests): A-only first-passage hybrid merge law (merge_pivot FROZEN), TF + F2 (merge_core
TF 0.6, on-policy 0), counted motion exclusion (recomputed per batch), lane-end stop + risk-retain,
merge_pivot_state_st = true, checkpoint_forward = true. The motion objective equals round 9's effective config.

## 2. Trainable set, rules, schedule
- **Trainable** (99 tensors, 0.74 M parameters): `driver_encoder.head` x10, `driver_encoder.centre_logits` x1000,
  `residual_net.*` except `.lane` x0.5, `temporal.*` x5, `follower_courtesy.*` x1000.
- **Frozen bitwise:** merge_pivot, merge_core, the gate scalars, the driver-encoder trunk (embedding unchanged),
  driver_updater, lane_v5, the lane gates, history_proj, k_embedding, residual_net.lane.
- **lr rule** (docs/r12_lr_rule.json): best held-out objective of {1,10} x {10,100,300,1000} after the production
  schedule, on S-BASEDIAG's TRAIN captures (96 windows, 24 held out by episode). Chosen: head x10, centre x1000.

  | head \ centre | 10 | 100 | 300 | 1000 |
  |---|---|---|---|---|
  | x1 | 1.034 | 0.860 | 0.757 | 0.742 |
  | x10 | 0.779 | 0.753 | 0.738 | **0.732** |

- **Courtesy weight/lr:** the R11 zero-init rule (9.067 / x1000). The weight acts only through the global clip
  (`rollout_grad = false`, Adam scale invariant).
- **Clip 1.0 kept.** Every step is clipped (micro-smoke pre-clip norms 181 / 173; round 10: 3-79). Adam on the
  normalised gradient keeps the per-parameter step at about lr.
- **Driver prior** (owner decision flagged to main): the existing population-mean prior (literature IDM values, equal
  to the textbook generator medians), weight 0.05. Default kept. Gate switch `R12_PRIOR=off|weak` (0 / 0.01).
- **Schedule** `[40, 70]` steps (120 s x 4, then 300 s x 2; lr 5e-5 / 3e-5). Round-9 pace was about 190 s/step on
  the 4090. Estimate about 5.8 h training + about 1 h final validation + about 1 h smoke, about 7.8 h in total.
  Latest checkpoint every 20 steps plus the stage checkpoints. If the smoke projects more than 8 h:
  `--set curriculum.stage_steps=[40,60]`.

## 3. Offline TRAIN check (tools/r12_offline_base.py; docs/r12_offline_base.json)
One-step, teacher-forced, on 24 held-out TRAIN windows (113k lane-1 HDV rows). Bias = model - recorded, as
[mean, MAE] in m/s^2. Arms:
- **r9:** round 9 as is.
- **r12 init:** physical encoder at the centre; attention read-out 0; round-9 gate x temporal kept.
- **r12 fit:** r12 init after the production-schedule CPU fit of the driver head + centre with the prior.
- **_core:** the same without the old temporal term, which the run retrains.

RESULT_TABLE

Readings:

READINGS

Encoded settings on held-out rows (p1 / p10 / p50 / p90 / p99; the p90-p99 values equal the centre for drivers without
history):

ENCODED

## 4. Merge-law support recount (tools/r12_support_recount.py)
The counted exclusion is recomputed in every training batch from the WM's one-step geometry (never reused). The offline
recount runs the identical `MergePivotTerm.prepare` on the same TRAIN windows with round 9 and the round-12 init:

SUPPORT

Exclusions are rare: 113 of 14,676 merges (0.77 %) over all 192 TRAIN episodes, 15 of them WM-motion. A CPU sample of
tens of windows can therefore show that the geometry class does not move, but cannot measure the motion class.
Registered post-run: `python tools/r12_support_recount.py --extra <round-12 final> --windows 192 --n-shards N` on cpuN
(CPU, TRAIN roots). The motion-class count and the retained cohort are compared with round 9. Geometry-class
(`event_vetoed_explained`) counts must be identical.

## 5. Tests and micro-smoke
- `tests/test_r12_base.py` (20):
  - new switches default to round 11 and are omitted from legacy flat; bad values are refused;
  - round 12 vs round 9 differs only by the registered keys; the round-9 decisions and the effective motion objective
    are pinned;
  - trainable set, loss composition (no cut-in term) and frozen merge law;
  - the lr rule file matches the config; the gate is launchable;
  - centre init puts every observed driver exactly on the centre, with lane and uncertainty rows kept;
  - the physical range is reached below the legacy floors and never left;
  - the residual re-init zeroes only the delta row and keeps a live gradient; switches off load bit-identically;
  - the switches need init_from and physical mode;
  - filled replay keeps junction vehicles in the state and does not reset their recurrent state, for both window ==
    unroll and window > unroll; supervision rows are unchanged; it is causal (future rows cannot change step t); the
    runner supplies presence only when switched on;
  - courtesy offset is exactly 0 without a partner under the filled replay.
- `tests/test_round12_gate_verdict.py` and `tests/test_round12_gate_launcher.py`: 138 passed together with the round-11
  verdict tests (WSL launcher tests ran).
- `tests/test_r12_checks.py`: 10 passed, 2 slow skipped. They pass with R12_SLOW=1.
- **Regression** (r12 + r11 + r10b + cut-in + round-9/10/10b/11 verdicts + merge pivot/context/core config + r9 lane-end /
  risk-retain + v6enc + unit config/model): **984 passed, 2 skipped** (slow), 0 failed. ruff is clean on all touched
  files.

**Micro-smoke** (tools/r12_micro_smoke.py; docs/r12_micro_smoke_summary.json). Setup:
- the real `WmSTrainer` with the round-12 config, init = the real round-9 final checkpoint;
- local CPU, 2 steps, batch 2, 60-step windows, 40-step unroll (window > unroll), each window with a recorded
  insertion;
- production lr (`optim.lr = 5e-5`). It ran with the pre-rule scales (head x1, centre x100), a mechanics check only.

SMOKE

## 6. Gate (scripts/round12_gate.sh; sha256 GATE_SHA)
- **Script hygiene:** LF, `bash -n` clean, no CR. Hardened like round 11: set -u/pipefail/noclobber; stale-artifact
  refusal (smoke_r12, c4_wm_v10_round12, logs, `.round12_gate.lock`); atomic lock.
- **Variables:** R12_ROOT / R12_PYTHON. R12_INIT defaults to the round-9 final stage300, falling back to wm_s_best.
  R12_PRIOR = config|off|weak.
- **Smoke:** batch 4 at 120/300 s, 2+2 steps, `selection.val_max_windows=4`, `max_hours=1.5`.
- **Verdict** (`python -m wms.train.round12_gate`):
  - round-10 merge census without the cut-in part, plus fail if any cut-in term is logged;
  - courtesy loss finite/positive with rows, contrast controls > 0, shift finite, and the R11 C1 head-saturation rule
    (plant clamp from the config);
  - fix-D telemetry present on courtesy steps;
  - driver-prior batch means finite and strictly inside the physical ranges (absent when the prior is off);
  - init report exact: 157 loaded, the 9 new keys (centre/bounds + 6 head tensors), zero head, centre rows 0-4 zeroed,
    accel row 0 zeroed, sha256 of the init;
  - strict reload of all 3 checkpoints: physical mode, v2 head, rollout_grad off, state ST, lane-end stop, batch 4;
  - frozen prefixes, merge_pivot and the gate scalars bitwise equal to the init;
  - at the final checkpoint: the centre moved and stays strictly inside the ranges, the driver head rows 0-4 are
    nonzero, the accel row is nonzero and the head moved;
  - saved prior weight = the requested mode.
- **Full run:** `runs/c4_wm_v10_round12`, `selection.val_max_windows=16`.
- **Ship:** `repo_r12` = repo_r11 plus the files changed since 4d8de32b (`git diff --name-only 4d8de32b..HEAD`), as
  LF blobs. Not run anywhere.

## 7. Reject criteria (registered BEFORE exec; `bash tools/r12_checks/run_all.sh full` with `SEEDS2=1`, CPU, on gpuR or cpuN)
- **Labels in the reused tools:** r9 = ROUND 9 (reference), r10b = ROUND 12, headzero = round 12 with
  `follower_courtesy.net.4.*` = 0.
- **Replay:** all arms are replayed with the filled presence (`--replay-present filled`). Round 9 is loaded with
  `driver_param_mode=legacy` (`--arm-set`).
- **Seed replication** (R10_DIAG): rollouts run seeded (offset 0), plus a second seed set (offset 1000) for round 9 and
  round 12. Behaviour metrics are reported per set and pooled, with 95 % state-bootstrap CIs and the paired
  round 12 - round 9 CI.
- **Cost:** about 75-90 min, plus about 35 min with SEEDS2=1.

| id | reject if | notes |
|---|---|---|
| M5 (round) | collisions, or TTC < 1.5 s share, worse than round 9: point estimate (analyze.py), AND/OR the seed-replicated paired CI lies above 0 | the hard safety rule |
| BASE-1 | slow-start one-step mean bias: \|.\| > 0.3, or not smaller than round 9's | tools/r12_checks/base_calibration.py |
| BASE-2 | free-road \|bias\| > 0.2, or far-upstream slow-start \|bias\| > 0.2 | offline r12-fit core: see section 3 |
| BASE-3 | recorded-braking-row bias worse than round 9's by > 0.1 | b moves from 3.0 to about 2.0-2.4 |
| BASE-4 (read) | last-3-s target follower: round 12 / head-zero / round 9 | expectation: about +0.8..+1.0 with the head zeroed; within +-0.3 with the head |
| M8-R12 | round 12's mean ABSOLUTE target-follower bias over [-3, 0) s (rollout or one-step) is not below round 9's, AND any of merges/run, stuck ramp cars, TTC < 1.5 s, min insertion gap p5, post-insertion hard brakes is worse (pooled point estimates) | tools/r12_checks/m8_r12.py |
| M7 | any of the 110 steps violates the C1 head-saturation rule; steps incomplete; contrast telemetry non-finite | m7_head_range.py with the round-12 config |
| M1-M4 | as round 11 (head behaviour; M3 attribution against head-zero) | the head is rejected (zeroed), not the round |
| support | geometry-class exclusions differ from round 9 on the post-run recount | section 4 |
| encoded settings (read) | > 20 % of rows within 1 % of an a/b range bound | base_calibration.py |

- **cpuN scoring** (72-state check, paired ranking): run with 2 seed sets for round 12 AND round 9 (re-scored with the
  current code). Accept round 12 as the new base only if:
  - M5 and BASE-1/2/3 pass;
  - the pooled stuck-car and merge-count differences vs round 9 have CIs that do not lie on the worse side.

  The 48-pair rank metrics are read with their bootstrap CIs, never as single scorings.
- **Round 11 vs round 12:** compared in the same scoring if round 11 passes its own checks. Round 12 inherits no
  round-11 weights.

## 8. Risks (details: R12_DESIGN.md section 7)
1. The start point over-accelerates more than round 9 (all drivers at a 2.49 / b 4.2). The centre must move early:
   x1000 = 0.05 logit per step at stage 1.
2. a may not be identifiable without the prior; the prior equals the generator medians (owner decision).
3. The re-initialised residual works inside the frozen gate envelope.
4. The merge law was fitted on the old base's geometry and is not refitted; the cohort is recounted.
5. The R11 head lr rule was measured on the old base.
6. Fix D: frozen pack-frame junction positions. Teacher-forced terms only.
7. Seed noise: acceptance is seed-replicated.

## 9. Commits
`git log c4/v10r12 ^c476cad4`. The packet commit is the last one.
