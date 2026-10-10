# REGISTRATION: WD-01-PB-V04A: Part B value model v0.4a on TRUE policy references (no walls)

**Registered:** 2026-09-30 by the world model session, BEFORE execution.

**Authority:** owner decision, 2026-09-30 (via main discussion).

- GO for v0.4a now; walls are deferred to v0.4b, after J4.
- Open items 1–3 of REGISTRATION_PARTB_V04_DRAFT.md are resolved as
  recommended (§2, §4, §3).
- This supersedes that draft for the no-wall scope.

## 1. Question

Can a deployable value model V(s, τ, π) do two things?

- **Q1:** predict remaining delay with calibrated uncertainty under TRUE
  fleet policies (native, damping, PI) and under per-CAV switching / fixed
  mixed assignment.
- **Q2:** give SKILFUL policy contrasts ΔV = V(s, τ, a) − V(s, τ, native) on
  held-out worlds with true references.

v0.2–v0.3.2 could not test Q2, because their "native" arms were switching
mixes (WM-72).

**Closest prior:** v0.3.2 (CHANGES_V032.md; partB_results/v0.3.2_strict),
with the same model family and the same deployable_strict inputs.

**The differences:**

- true-policy sources;
- per-CAV action inputs (the 9734eee layer);
- a policy-conditioned target;
- contrast acceptance.

**Input from PB-RESCORE1:** if its report lands before the build, its label
audit is cited here and its findings are applied. A finding that changes the
design becomes a disclosed addendum BEFORE any fit.

## 2. Sources and splits (HOLDOUT sealed everywhere)

| source | policy label | role |
| --- | --- | --- |
| chunk-4 `<w>__base` (after t0) | damping (fleet) | primary |
| chunk-4 `<w>__native_base` | native (fleet) | primary |
| chunk-4 `<w>__native` (restored) | native (fleet) | primary |
| chunk-4 `<w>__pi` (restored) | PI (fleet) | primary |
| chunk-4 `<w>__fixed*` (2 arms) | fixed mixed assignment (per-CAV actions) | primary |
| chunk-4 `<w>__switching*` (3 arms) | switching (per-CAV actions) | primary |
| chunk-2/3 J3 truearms `<w>__native`, `<w>__pi` | native / PI (fleet) | primary |
| chunk-2/3 bases | damping | primary |
| chunks 1–3 + fix27 `__native`/`__pi` switching arms | switching (per-CAV actions) | AUXILIARY training only |

- **Chunk-4 arm verification:** before the build, every chunk-4 arm's policy
  is verified from its switch_schedule.json plus suffix_commands.jsonl
  owners (kind == "command" only). A mismatch refuses the source.
- **Per-CAV actions** come only from switch_schedule plus suffix_commands,
  via partB_v0data/actions_v04.py. NEVER from dispatcher_rows.jsonl or
  insertions.jsonl.
- **Splits:** each chunk's WORLD_PLAN split.
  - **TRAIN:** fitting and all selection/calibration (world-grouped OOB).
  - **CALIBRATION:** the acceptance report only, read once.
  - **HOLDOUT:** never opened. The split is checked before any file access,
    and tools/holdout_guard.py plus protocols/holdout_exclusions.json apply.
- **Chunk-4 CAL (9 worlds)** is also C4L1's look set for Part A, a different
  model. It is DISCLOSED here; Part B's use does not feed Part A.
- **Chunk-2/3 CAL** was read once each by v0.3.2 and RESCORE1 for reports
  only. It is disclosed.
- **Auxiliary chunks 1–3** (owner item 1): they enter TRAINING only, with
  their relabelled per-CAV actions. They never enter scoring, acceptance or
  selection. An ablation without them is reported.
- **New data root:** E:/research-dense/output/wd01_partB_v0data_v04a.
  - COPY only (never hard-link);
  - MANIFEST with the per-source policy and its verification evidence;
  - SHA256SUMS;
  - a backup under E:/research-dense-backups/wd01/.
  - Older roots are read-only.

## 3. Inputs

- **deployable_strict (HEADLINE):** CAV self-state and CAV-detected
  neighbours; no all-vehicle entrance counts.
- **deployable_detector:** a SENSITIVITY variant.
- **Both variants carry:**
  - the per-CAV action tokens, with the unknown mask (CAV_RAW 20);
  - the policy conditioning π for the continuation. For fleet arms it is
    the fleet label. For switching/fixed arms it is the planner-known
    per-CAV schedule summary.
  - Wall tokens are all empty (v0.4b adds walls).
- **Prohibited inputs:** penetration, pre_fix, speed_mode and world strings
  (owner, 2026-09-29). The ideal-sensing premise is stated: lane bins come
  from the true state, and the reconstruction gap is measured separately.
- **Carried over from v0.3.2:**
  - t0 is unrecoverable (ctx col 1 = 0);
  - world-grouped bootstrap and early stopping;
  - CAL guard on selection.

## 4. Targets and contrast

- **V target:** the remaining total delay over (t, t + τ], all vehicles, for
  τ ∈ {30, 60, 120, 300, 570}. The safety heads are as in v0.3.x.
- **ΔV from PAIRED predictions** of the same model (owner item 2; no Δ head):
  - point: ΔV(s, τ, a) = V_med(s, τ, a) − V_med(s, τ, native);
  - predictive distribution: independent-quantile pairing, as in the v0.3.x
    contrast code. A comonotone pairing is reported as a sensitivity.
- **Realised TRUE contrast:** ΔY = Y(a arm) − Y(native arm), from the same
  saved state at the same t0. The native reference is the RESTORED
  `__native` arm (chunk 4) or the J3 truearm `__native` (chunks 2/3).
- **Contrasts reported:** damping−native, PI−native, switching−native and
  fixed−native.
- **Gating contrasts:** damping−native and PI−native.

## 5. Model and selection

- **Model:** partB_v0 gbm-variant auto, per τ (ridge / hybrid / pure),
  policy-conditioned.
- **Selection and calibration:** TRAIN world-grouped OOB only.
- **Ridge baseline:** on the SAME inputs, including the policy conditioning.
- **Comparators in the report:** v0.3.2_strict (reference) and climatology
  (TRAIN, per τ × policy × demand).

## 6. Acceptance rule (fixed BEFORE running; CAL; strict = HEADLINE)

- **A1 calibration.** At every τ, CRPS < climatology; AND overall 10–90 %
  coverage in [0.75, 0.85].
- **A2 vs ridge (v0.3.2 failed this at long τ).**
  - Per τ: the HARD comparison, which needs strictly better MAE(median) AND
    CRPS than ridge (a tie is a TIE).
  - **Gating:** PASS at τ = 30, 60 and 120.
  - **Reported with its verdict, not gating:** τ = 300 and 570.
- **A3 contrast skill, for each gating contrast c ∈ {damping−native,
  PI−native}, at τ ∈ {60, 120, 300}:**
  - **(i) Sign accuracy:** of sign(E[ΔV]) vs sign(ΔY), over CAL pairs with
    |ΔY| above the TRAIN median |ΔY| for that τ, so near-ties don't count.
    The world-bootstrap 90 % CI lower bound must be > 0.5.
  - **(ii) CRPS:** the ΔV CRPS must be below BOTH the zero-contrast baseline
    (a point mass at 0) and the TRAIN-cell-mean baseline (the TRAIN mean ΔY
    per demand × penetration cell as a point mass; the cell key is a label
    for the baseline only, not a model input).
  - c passes A3 if (i) AND (ii) hold at ≥ 2 of the 3 τ.
- **VERDICT:**
  - **ACCEPT** iff A1, AND A2-gating, AND A3 holds for BOTH gating contrasts.
  - **PARTIAL** iff A1 and A2-gating hold and A3 holds for exactly one
    contrast.
  - **FAIL** otherwise.
  - The detector variant is judged by the same rule, labelled SENSITIVITY.
- **Always reported:**
  - per-policy coverage;
  - the switching−native and fixed−native contrasts;
  - the auxiliary-data ablation;
  - the comonotone ΔV sensitivity;
  - breakdowns by demand and penetration (as report strata only);
  - the TRAIN-OOB versions of A1–A3 (labelled in-sample-selection-free).

## 7. Budget

- Zero SUMO.
- **CPU on the laptop ≤ 4 workers** (WM-43b is running). Estimates:
  - build about 2 h;
  - two refits (strict, detector) about 1 h;
  - auxiliary ablation plus report about 1 h.
- One executor task, with the owner's time limit per continuation.
- **Stop rule:** no refit after the CAL read. A crash before the CAL read may
  be continued; a crash after it is reported as is.

## 8. Outputs

- partB_results/v0.4a_strict and partB_results/v0.4a_detector (NEW dirs),
  each with eval_report + verify.
- output/wd01/CHANGES_V04A.md.
- REPORT_V04A.md: the verdict table under §6.
- LEDGER_DRAFT_PB_V04A.json in the experiment_history ENTRY schema. An
  allocation entry is committed before the build.
