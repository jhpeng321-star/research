# REGISTRATION_WM_N1 — AMENDMENT 3: the headline model becomes IDEA A, "controller-in-the-loop, strategy-agnostic world model"

**Status:** owner decision relayed by main discussion, 2026-10-01. Registered
BEFORE any WM-N1 featurization run or training.

- Amends `REGISTRATION_WM_N1.md` (LF-sha256 `666233aa…93246`), Amendment 1
  (`decb9656…`, member-1-first, dedicated GPU box) and Amendment 2
  (`56fa99a0…`, J4 disclosure).
- **UNCHANGED:**
  - the §4 targets and the decision rule;
  - chunk 5 (`REGISTRATION_CHUNK5_TEST.md`, `f8c7d760…`) as the single
    final test, sealed until then;
  - CAL-only calibration and variant selection;
  - HOLDOUT sealed;
  - member-1-first.

## A3.1 Headline model (variant "A")

Each rollout step (0.5 s) does four things:

- **(a) CAVs:** moved by our REAL controller code (native / damping /
  PI / wall law v2.2, via the partA_ctrl mirrors), exactly and never
  learned.
- **(b) HDVs:** the Part A v2g baseline (IDM + MOBIL/merge) PLUS a neural
  RESIDUAL (an acceleration correction and a lane-change-logit correction),
  computed by attention over neighbours' KINEMATICS only.
- **(c) Arrivals:** the learned arrival head from the entrance counters.
- **(d) Validity:** the validity layer (non-negative speed, no overlap,
  lane bounds, position integration as in the D1 fix).

Two input rules for the HDV network:

- It NEVER receives a strategy name or action labels.
- In the headline model it has NO CAV flag. The CAV-flag variant is an
  ablation only.

**Long-horizon outcomes** (delay, TTC < 3 s, hard-brake) come from the rollout
PLUS a strategy-independent correction head. That head is fed a SUMMARY OF
THE CAVs' PLANNED MOTION, a trajectory summary produced by the controllers,
and never labels.

**Event-ready inputs:** tokens and masks for stopped obstacles, blocked lane
segments and speed-limit changes. The current data doesn't use them, but they
are supported and tested with synthetic fixtures. An incident campaign is a
later, separately registered follow-up.

## A3.2 Training

- **CAV trajectories:** teacher-forced from the data (identical to the
  controller output in SUMO).
- **Losses:**
  - the one-step HDV residual NLL;
  - a multi-step rollout loss;
  - the arrival loss;
  - the outcome/correction-head CRPS;
  - the paired-branch CONTRAST loss on the HDV response and outcomes, with
    sibling arms at the same world + t0.
- **Ensemble:** 5 members, member 1 first (Amendment 1).
- **Training data:** the TRAIN corpus as registered (Amendment 2 applies).
  CAL is used only for calibration and variant selection.

## A3.3 Variants reported (the chunk-5 TEST, identical windows)

| variant | description |
| --- | --- |
| A | the headline, controller-in-the-loop |
| N | the label-conditioned pure neural model, the original design. A comparator; it cannot handle unseen strategies |
| T | the tree Part B |
| C | the blend (CAL linear pool) |

- The planner variant is chosen on CAL before the TEST, by the registered
  rule in REGISTRATION_WM_N1 §3, with A added to the candidate set.
- **Tie order**, simplest first: T < A < C < N.

## A3.4 Held-out-strategy tests (registered now; 1-member models)

| test | train | evaluate | notes |
| --- | --- | --- | --- |
| (i) HEADLINE | J4 (wall law v2.1) | J4B (law v2.2), before any J4B outcome is read | J4B is a separate rental job. Its outcomes are read ONLY by this test's registered evaluation |
| (ii) | with every PI arm (native-vs-PI pairs included) removed | the PI arms of CAL | |
| (iii) | with the wall-speed arms of one speed factor removed | that factor's wall arms of CAL | the factor is fixed a priori as the highest J4 factor |

- **Metrics:** those of REGISTRATION_WM_N1 §4 for paired dV CRPS and sign
  accuracy, against two baselines: the Part A v2g rollout, and variant N
  where it can be applied.
- **Status:** reported as generalisation evidence. They are NOT selection
  inputs.

## A3.5 OOD score

- An "is this motion familiar?" score on the CAV planned trajectories,
  measured by density or ensemble disagreement against the TRAIN CAV-motion
  summaries.
- Calibrated on CAL only, and fed to the planner's hold-back rule.
- Reported on the TEST as a diagnostic.

## A3.6 Idea B (an action-conditioned breakdown-hazard head)

DEFERRED. It would be included only if it didn't delay A, and at this
amendment it would.

## A3.7 Compute (to be replaced by the timing gate)

| item | GPU-h |
| --- | --- |
| Members (5), incl. the multi-step rollout loss | ≈ 45–85 |
| Held-out tests (3 × 1 member) | ≈ 20–40 |
| Evaluation | ≈ 3 |
| **Total** | **≈ 70–130** |

Member 1 alone is ≈ 9–17 GPU-h.
