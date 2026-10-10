# WM-N1 IDEA A — AS-BUILT DESIGN (controller-in-the-loop, strategy-agnostic world model)

**Status:** as-built description of the code on `zcode/wmn1docs` (snapshot of
`zcode/wmn1integA` at `bac9f33`, stage-1 integration). Documentation only: no
code, data or registration text is changed by this file.

**Headline registration:** `REGISTRATION_WM_N1.md` (LF-sha256 `666233aa…93246`)
as amended by Amendment 1 (`decb9656…`, member-1-first, dedicated GPU box),
Amendment 2 (`56fa99a0…`, J4 disclosure) and Amendment 3 (`a597d986…`, the
headline model becomes Idea A). All four files live beside this file; their
LF-sha256 values are the frozen identifiers quoted here.

**Companion documents:** `INTERFACES_V2.md` (piece contracts, tensors, step
order, config defaults), `PREFLIGHT_GATES_IDEA_A.md` (gate specs),
`CODE_READY.md` (integration evidence, CLIs, gate status), `WM_N1_DESIGN.md`
(the superseded v1 draft design; kept for history).

## 0. Frozen registrations and amendments cited (LF-sha256)

| record | path | LF-sha256 | status |
| --- | --- | --- | --- |
| WM-N1 registration | `output/wd01/wm_n1/REGISTRATION_WM_N1.md` | `666233aa1d1e4667…93246` | FROZEN, never edited |
| Amendment 1 | `output/wd01/wm_n1/REGISTRATION_WM_N1_AMENDMENT_1.md` | `decb96562ebfb549…16c57` | FROZEN |
| Amendment 2 | `output/wd01/wm_n1/REGISTRATION_WM_N1_AMENDMENT_2_J4_DISCLOSURE.md` | `56fa99a04fdd38e3…c83ed` | FROZEN (disclosure) |
| Amendment 3 | `output/wd01/wm_n1/REGISTRATION_WM_N1_AMENDMENT_3_IDEA_A.md` | `a597d9861dcc859c…4eef6` | FROZEN, owner-acknowledged |
| chunk-5 TEST | `output/wd01/wm_n1/REGISTRATION_CHUNK5_TEST.md` | `f8c7d760baedeb35…4b31e` | final text; freezes on the fresh-seed audit |
| wall law v2.2 | `output/wd01/walls_design/merge_release/REGISTRATION_WALL_V22.md` (branch `zcode/wallv22`) | `e7fd353616eea9ef…f01e7` | APPROVED and FROZEN (owner ack 2026-10-01) |
| J-PULSE | `output/wd01/REGISTRATION_JPULSE.md` (branch `zcode/wd01` `36766cd`) | `a4d1c5f9beb40dc1…21ea7` | APPROVED WITH the 240 s option, FROZEN |
| Part B v0.4b | `output/wd01/REGISTRATION_PARTB_V04B.md` (branch `zcode/pbv04b2`) | `e00cd98edadbdfda…5ad9d` | **ACK PENDING**; no fit before the ack |

The full 64-hex values and the verification commands are recorded at the end
of this file ("Sha verification").

## 1. What Idea A is

Each rollout step (dt = 0.5 s; control at 1 s) does four things, in a fixed
order implemented by `rollout2/step.py:step_once`:

- **(a) CAVs are moved by our REAL controller code** (native / CIRCLES
  damping / PI external-U / wall law), exactly and never learned — P1
  `ctrl/mover.py` (`CavMover`) over the verified `partA_ctrl` mirrors; wall
  v2.1/v2.2 selection via `ctrl/wall_hook.py`. The returned target speed is
  executed with `execution_speed` once per control tick and held for the two
  0.5 s physics steps (the `partA_v2g` convention). As built, the `CavMover`
  body is still a stub (`ctrl/mover.py:148` `NotImplementedError`); G1
  currently runs on the `rollout2/equivalence.py` `NativeMover`/`ReplayMover`
  stand-ins until P1 lands.
- **(b) HDVs are the Part A v2g baseline** (IDM + MOBIL/merge) **PLUS a
  neural residual** — an acceleration correction and a lane-change-logit
  correction from `hdv/net.py:HdvResidualNet`, computed by attention over
  neighbours' KINEMATICS only. The main-config input has NO CAV flag and NO
  action/strategy channel (`hdv/input_schema.py`, gate G3).
- **(c) Arrivals** come from the learned arrival head
  (`arrival/head.py:IdeaAArrivalHead`, counter-driven; `arrival/sampler.py`
  inserts new vehicles into empty slots).
- **(d) Validity** applies the D1 layer (`rollout2/validity.py`): clip
  acceleration, trapezoidal position integration (`integrate_d1`), reject
  illegal lane changes, project same-lane rear-bumper gaps ≥ 0, retire
  vehicles past the lane end.

Long-horizon outcomes (delay, TTC < 3 s, hard-brake; τ ∈ 5 horizons, 19
quantiles + collision logit) come from the rollout PLUS a strategy-independent
correction head (P5) that is fed a SUMMARY OF THE CAVs' PLANNED MOTION —
never a strategy name or action label.

Two strategy-agnostic input rules hold throughout: the HDV network never
receives a strategy name or action labels, and in the headline model it has
no CAV flag (the CAV-flag variant is an ablation only, `HdvConfig.use_cav_flag=False`).

## 2. As-built pieces (paths under `output/wd01/wm_n1/`)

| piece | paths | as-built state |
| --- | --- | --- |
| P1 CTRL | `ctrl/config.py`, `ctrl/mover.py`, `ctrl/wall_hook.py` | interfaces + wall registry in; `CavMover.step` body stubbed; `ctrl/parity` missing (G2 pending) |
| P2 HDV | `hdv/net.py`, `hdv/input_schema.py` | implemented; kinematics-only schema, G3 entry points |
| P3 ROLLOUT2 | `rollout2/state.py`, `step.py`, `engine.py`, `validity.py`, `baseline.py`, `equivalence.py`, `smoke.py` | implemented; D1 validity, G1 zero-residual report, 60 s J4 smoke contract |
| P4 ARRIVAL | `arrival/head.py`, `arrival/sampler.py` | implemented; counter-driven head + insertion sampler |
| P5 CORR | `corr/summary.py`, `corr/head.py`, `corr/ood.py` | implemented; 16-feature `cav_motion_summary`, correction head, OOD score |
| P6 EVENTS | `events/tokens.py`, `masks.py`, `synthetic.py` | implemented encoders + synthetic fixture; not yet consumed by the P3 step (v1 shard carries no event channels) |
| P7 TRAIN | `train/idea_a.py` | implemented; losses, held-out splits, ensemble flags, checkpoints |
| P8 EVALH | `evalh/idea_a.py`, `combine/select_variant.py`, `adapter/predictor.py` | implemented; variant A predictor, held-out protocol, OOD calibration, action descriptors |
| P9 FEATURES | `features/idea_a.py`, `corpus/idea_a.py` | implemented; `veh_kin` / `hdv_base` / `cav_plan_ch` / event channels; Amendment-2 relabel flag |
| P10 MULTIBOX | `bundle/multibox.py`, `bundle/e2e.py`, `bundle/queue.py` | implemented; one config per box, member-1-first, approval binding |
| P11 PREFLIGHT | `preflight/**` | implemented; gates G1–G6 with JSON reports |

## 3. HDV residual on kinematics only (P2)

- Input: `veh_kin` `[N_ticks,128,7]` = `s, lane(3 one-hot), v, a, length`
  (`features/idea_a.py:kinematics_view` strips the CAV flag and action
  one-hots from the v1 `veh_step` tensor). No `is_cav`, no action one-hot,
  no strategy label anywhere in the main config (`hdv/input_schema.py`).
- Architecture: lane-aware attention over neighbours within `radius_m=150`
  (`HdvConfig`: d_model 128, 4 heads, 3 layers, ffn 512), outputs
  `accel_delta_mps2` and `lane_logit_delta` added to the v2g baseline.
- Gate G3 asserts the tensor schema is free of action/strategy/CAV channels
  and that permuting labels/flags/strategies leaves the output identical.

## 4. Arrival head (P4)

`IdeaAArrivalHead.forward(counter)` consumes the entrance-counter history
(30 bins × 2 entrances) and returns per-entrance counts and insertion speeds;
`ArrivalSampler` inserts up to `max_insert_per_entrance=3` vehicles per step
into empty slots. Config: `ArrivalConfig` (`counter_channels=128`,
`d_model=256`). This replaces the v1 global-token arrival head.

## 5. Correction head fed by the CAV planned-motion summary (P5)

- `corr/summary.py:cav_motion_summary(positions, speeds, accel, cav_mask)`
  builds a fixed 16-feature summary of the CAVs' PLANNED trajectory
  (active fraction, per-CAV speed/accel/displacement statistics over the
  horizon). It is a deterministic function of the CAV trajectory only: two
  different controllers producing the same trajectory give identical
  summaries. No strategy, action or label channel exists in the signature.
- `corr/head.py:CorrectionHead.forward(summary, global_token)` returns
  outcome quantiles `[B,5,3,19]` and collision logits `[B,5]`, trained with
  CRPS. It supplies the long-horizon outcomes alongside the rollout.

## 6. Validity and D1 (P3)

`rollout2/validity.py` implements `integrate_d1` (the frozen trapezoidal
position integration that fixes the v1 defect where rollout positions never
moved, `model/rollout.py:84`) and `project_rollout_step` (acceleration clip,
lane-change rejection, rear-bumper gap ≥ 0, end-of-lane retirement), with all
interventions counted in `ValidityCounts`. Regression coverage:
`rollout2/tests/test_d1_regressions.py`. G1 proves the zero-residual,
arrivals-off rollout reproduces the Part A v2g rollout on real J4 TRAIN
windows (max position/speed error 0.0; `zcode/revA3g 31232fd`).

## 7. Event-ready inputs (P6)

`events/tokens.py` encodes stopped obstacles, blocked lane segments and
speed-limit changes as `[N_ticks,8,8]` tokens + masks; `events/synthetic.py`
provides the synthetic fixture. The current data does not use them: the v1
shard carries no event channels and the P3 step does not consume them yet.
They are supported and tested so an incident campaign can be added later as
a separately registered follow-up.

## 8. Training (P7, `train/idea_a.py`; Amendment 3 A3.2)

- **CAV trajectories:** teacher-forced from the data (`teacher_forcing_cavs=True`),
  identical to the controller output in SUMO; no gradients or labels flow
  into the HDV path from them.
- **Losses (all weights default 1.0 in `IdeaATrainConfig`):**
  - one-step HDV residual NLL (`residual_nll`);
  - multi-step rollout loss (`rollout_loss`, horizon 20 steps = 10 s,
    gradients reach HDV parameters);
  - arrival loss (`arrival_loss`);
  - outcome/correction-head CRPS (`correction_crps`);
  - the PAIRED-BRANCH CONTRAST loss (`paired_response_contrast`) on the HDV
    response and outcomes, pairing sibling arms of the same world + t0
    (treatment minus reference).
- **Training data:** the TRAIN corpus as registered (Amendment 2 applies to
  J4 labels). CAL is used only for early stopping, calibration and variant
  selection — never for gradients.
- Config: `IdeaATrainConfig` (epochs 15, 2000 steps/epoch, batch 8, lr 1e-3,
  grad clip 1.0; `seed_for_member` gives independent per-member seeds).

## 9. Ensemble: 5 members, member-1-first, partial-ensemble rule

- **Member-1-first (Amendment 1):** after the 1-h timing gate and the owner
  approval file, member 1 is trained ALONE to completion (an early working
  model for internal checks only), then members 2–5 with independent recorded
  seeds (`DEFAULT_SEEDS=(101,202,303,404,505)`, `MEMBER_ORDER`).
- **Partial-ensemble rule (as built):** `ensemble_flags(k)` records
  `ensemble_k`, `members`, `partial=True` and `final_test_eligible=False`
  for any k < 5 (`train/idea_a.py:203`); `require_full_ensemble`
  (`train/idea_a.py:216`, wired into the bundle queue,
  `bundle/queue.py:320`) REFUSES the final test for any partial ensemble.
  The early 1-member model supports internal TRAIN/CAL diagnostics only and
  is never run on chunk 5; the single final TEST uses the full 5-member
  ensemble. Members combine by the equal-weight mixture
  (`model/ensemble.py:mixture_quantiles`); CAL-only conformal calibration is
  applied downstream, never inside the ensemble.

## 10. Variants A/N/T/C and selection (Amendment 3 A3.3)

| variant | meaning |
| --- | --- |
| A | the headline controller-in-the-loop model |
| N | the label-conditioned pure neural model (original design; comparator that cannot handle unseen strategies) |
| T | the tree Part B (v0.4a; v0.4b registered as `e00cd98e…`, ACK PENDING) |
| C | the CAL linear-pool blend of the neural and tree predictors |

- As built, `combine/select_variant.py` has `VARIANTS=("T","A","C","N")` and
  `COMPLEXITY_RANK={"T":0,"A":1,"C":2,"N":3}`.
- The planner variant is chosen on CAL BEFORE the TEST by the registered rule
  (lowest CAL mean ΔV CRPS over the gating contrasts at τ ∈ {60,120,300}
  subject to 10–90 % coverage in [0.75, 0.85]); a tie within 2 % goes to the
  SIMPLER model, tie order **T < A < C < N**. The choice is frozen in
  `PLANNER_VARIANT.json` before the TEST read.

## 11. Held-out-strategy tests (Amendment 3 A3.4; 1-member models)

| test | train | evaluate | as-built |
| --- | --- | --- | --- |
| (i) HEADLINE | J4 (wall law v2.1) | J4B (law v2.2), before any J4B outcome is read | `held_out_protocol()` test i; J4B reads refused outside (i) (`assert_j4b_read_allowed`) |
| (ii) | every PI arm removed | the PI arms of CAL | test ii (`policy == "pi"` excluded from train) |
| (iii) | the wall-speed arms of one factor removed | that factor's CAL wall arms | factor fixed a priori to the highest J4 factor (`HIGHEST_J4_WALL_SPEED_FACTOR="mid"`) |

Metrics: the registration §4 paired ΔV CRPS and sign accuracy, against the
Part A v2g rollout and variant N where applicable. Status: generalisation
evidence only — never selection inputs. G6 dry-runs the splits and asserts
the excluded arms are absent from training.

## 12. OOD score (Amendment 3 A3.5)

`corr/ood.py` + `evalh/idea_a.py:calibrate_ood/ood_score`: an "is this motion
familiar?" score over the CAV planned-motion summaries, measured by density /
ensemble disagreement against the TRAIN summaries. Calibration is CAL-only
(seal-guarded: any non-CAL split raises), threshold = CAL quantile 0.95,
larger = less familiar; it feeds the planner's hold-back rule and is reported
on the TEST as a diagnostic.

## 13. Pre-flight gates G1–G6 (P11; `PREFLIGHT_GATES_IDEA_A.md`)

| gate | checks | status on `zcode/wmn1integA` `bac9f33` |
| --- | --- | --- |
| G1 `zero-residual` | residual 0, arrivals off → Idea A rollout ≡ Part A v2g on a real J4 TRAIN window, tol 1e-6 (or an explained, documented tolerance) | PASS (max error 0.0) |
| G2 `ctrl-parity` | P1 mover reproduces recorded SUMO CAV trajectories, ≥ 2 TRAIN windows per strategy, wall v2.1 AND v2.2 supported | **FAIL — pending P1** (`ctrl.parity.strategy_parity_report` missing; `CavMover.step` stubbed) |
| G3 `label-leak` | no action/strategy/CAV-flag channel; label-perturbation invariance | PASS |
| G4 `overfit` | 1 member drives total loss (incl. contrast + rollout) near 0 on 2 tiny TRAIN windows, CPU | PASS (final 0.00706 ≤ 0.05; deterministic stand-ins `train/g4_standins.py`) |
| G5 `e2e-smoke` | convert → timing ≥ 30 s → train → checkpoint → resume → eval on a real Idea A bundle; torch 2.5.1 compat check | PASS |
| G6 `heldout-dryrun` | splits (i)–(iii) build; excluded arms absent from train | PASS (J4B tags are declared metadata, never opened) |

Summary report: `preflight/reports/PREFLIGHT_SUMMARY.json` (`n_pass=5`,
`n_fail=1`). Gate commands are in `CODE_READY.md` §6.

## 14. GPU plan (owner: ~3 GPUs, 3–4 boxes, final count after the timing gate)

- All rented GPUs (3–4 boxes) go to WM-N1: the Idea A members, the N
  comparator and the held-out tests (owner decision 2026-10-01 night). The
  final box count is set after the 1-h timing gate.
- P10 `bundle/multibox.py` implements **one config per box**: 8 default
  boxes — `box-member-1..5`, `box-heldout-i-j4b`,
  `box-heldout-ii-leave-pi`, `box-heldout-iii-leave-highest-wall-speed` —
  run in parallel; every member-k (k>1) box waits for `box-member-1`
  (member-1-first is enforced in `run_config`). Boxes may be consolidated
  onto fewer physical machines as long as each config keeps its own
  `APPROVAL.json` binding its `TIMING_GATE.json` raw-bytes sha256 plus the
  four frozen registration/amendment LF-shas.
- Budget (Amendment 3 A3.7, replaced by the timing gate): members ≈ 45–85
  GPU-h, held-out tests ≈ 20–40, evaluation ≈ 3; total ≈ 70–130 GPU-h;
  member 1 alone ≈ 9–17 GPU-h. A dedicated WM-N1 GPU box (Amendment 1) is
  the default home; the queue order is verify upload → memmap verify →
  timing gate STOP → owner approval → member 1 → members 2–5.

## 15. Action-descriptor requirement (owner, 2026-10-01)

The action descriptor of a paired comparison is the PLANNED SCHEDULE, never
the arm name: 19 × 30 s per-CAV/per-group segment actions
(`evalh/idea_a.py`, `ActionDescriptor`, `descriptor_from_events`,
`descriptor_from_schedule_dict`; owner stratification merged from
`zcode/wmn1p8s`). An `ActionDescriptor` deliberately has no
arm/contrast/policy field — "damp 60 s then native" and "damp 570 s" are
different descriptors. Evaluation reports are stratified by descriptor
(`evalh/reports/ACTION_REPRESENTATION.*`).

## 16. Paper framing (owner decision, 2026-10-01)

- The WM is **trained and evaluated on the TRUE state**. This is the honest
  description of what is built: inputs are simulator ground-truth kinematics
  and counter histories, not a sensing pipeline.
- The **sparse-sensing → reconstruction input path (C4L2) is FUTURE WORK**.
  It is kept in the design as an INTERFACE only (the deployment-honest input
  tables of `WM_N1_DESIGN.md` §1 and the RSU-style input rules), not built:
  no reconstruction model, no recon-sample evaluation, no claim.

## 17. Known limitations (disclosed)

- **J4 is wall law v2.1 with 28 road-end relabels** (Amendment 2,
  `56fa99a0…`): the registered road-end truncation never fired in J4
  (edge-blind `road_left_m`, root-caused in `zcode/wallrt eee755f`); the 28
  `member_left_network` aborts are relabelled as road-end events for WM-N1
  training labels and wall tokens only (run-level meta
  `amendment2_road_end_relabel`). The chunk-5 wall arms run under law v2.2
  (`e7fd3536…`), so the trained wall tokens describe a different release
  rule than the TEST arms; wall-arm results are reported separately.
- **96 % whole-window action pairs until J-PULSE** (`a4d1c5f9…`): of 4,952
  TRAIN pairs, 96.1 % are whole-window; only 5 are ≤ 60 s, 189 are 60–240 s
  and 2 differ only in the first 60 s. Short-commitment strategy effects are
  barely represented in training until the J-PULSE arms exist; this is
  disclosed here and in the eval.
- **G2 tolerances:** the controller-parity gate (and the parity tolerance it
  must state) is not yet satisfiable — P1's real mover body and
  `ctrl.parity` are missing, so G2 fails cleanly (`piece missing`) and stays
  FAIL until P1 lands. G1's 1e-6 equivalence currently holds exactly
  (error 0.0) on the stand-in mover.
- **Synthetic vs real evidence:** the P6 event inputs are exercised only on
  synthetic fixtures (no real incident data); G4's overfit machinery uses
  deterministic CPU stand-ins rather than the real attention net; the
  long-horizon correction head and variant selection are validated on
  synthetic and micro TRAIN fixtures so far. No GPU training, no CAL/TEST
  read and no J4B read has happened.
- v0.4b (`e00cd98e…`) is registered but **ACK PENDING**; until the ack, T is
  v0.4a in any reported comparison, and no v0.4b fit happens.

## 18. Sha verification (evidence)

LF-sha256 (CRLF→LF, bytes) recomputed 2026-10-01 on `zcode/wmn1docs`:

```
666233aa1d1e466750d6d944b9102c84a2f32850ebeb4a97c9f3c9b56dd93246  output/wd01/wm_n1/REGISTRATION_WM_N1.md
decb96562ebfb549b56406d980be9c78bf4a377bd50c5eadeea08aac16e16c57  output/wd01/wm_n1/REGISTRATION_WM_N1_AMENDMENT_1.md
56fa99a04fdd38e37a518dfc65972bc5e2e6d227210372a75258bf28249c83ed  output/wd01/wm_n1/REGISTRATION_WM_N1_AMENDMENT_2_J4_DISCLOSURE.md
a597d9861dcc859cbcb9c10b65069a81031732516ec12a0296fd68aff604eef6  output/wd01/wm_n1/REGISTRATION_WM_N1_AMENDMENT_3_IDEA_A.md
f8c7d760baedeb359c501cd90a3e2a599fb0ddfef390ea555cf304c38214b31e  output/wd01/wm_n1/REGISTRATION_CHUNK5_TEST.md
e7fd353616eea9ef2ab0440c751c8c9211f11a3f5de97f1b911b7a528f6f01e7  zcode/wallv22:output/wd01/walls_design/merge_release/REGISTRATION_WALL_V22.md
a4d1c5f9beb40dc117c2f53ef64ab176bab434f3dcab8481ad9cd279e3621ea7  zcode/wd01:output/wd01/REGISTRATION_JPULSE.md
e00cd98edadbdfdafc9267eefdd9f004fb72cc7de8c531e00ef27e8be8b5ad9d  zcode/pbv04b2:output/wd01/REGISTRATION_PARTB_V04B.md
```

Check command (Python, repo root):

```
python -c "import subprocess,hashlib;p='output/wd01/wm_n1/REGISTRATION_WM_N1.md';b=open(p,'rb').read();print(hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest())"
```

(cross-branch records via `git show <branch>:<path>`; the same hashes are
independently recorded in `bundle/multibox.py` frozen constants, the
`zcode/wd01` `MORNING_OWNER_ITEMS_20261001.md`, and
`walls_design/merge_release/WALL_V22_FREEZE.json`.)
