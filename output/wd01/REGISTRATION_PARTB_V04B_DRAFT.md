# REGISTRATION DRAFT: WD-01-PB-V04B: Part B direct-difference tree model v0.4b on J4 wall pairs (baseline T for Idea A)

**Status:** DRAFT, 2026-10-01, world model session. NOTHING has been fitted.
This becomes the registration when the owner accepts it; execution starts
only after that, with the allocation entry committed BEFORE any build.

**Authority:** owner decision, 2026-10-01 (direct-difference tree on J4 wall
data; the fair baseline "T" of REGISTRATION_WM_N1_AMENDMENT_3_IDEA_A.md
§A3.3).

**Closest priors:**

- WD-01-PB-V04A (`REGISTRATION_PARTB_V04A.md`, `REPORT_V04A.md`): same
  pairing machinery and A3 rule; verdict FAIL on every variant — sign
  skill (damping−native sign accuracy 0.87–1.00 on |dY| > median pairs),
  no value skill (dV CRPS never beat the zero-contrast / TRAIN-cell-mean
  baselines at the gating taus). The unanswered difference: v0.4a scored
  the difference of TWO level predictions (dV from paired V(s, τ, π));
  v0.4b predicts the difference dY DIRECTLY, so the level-prediction
  error cannot enter the contrast.
- WM-N1 Amendment 3 (Idea A, variant T) and Amendment 2 (J4 disclosure).
- WM-59 J4 wall pilot (`E:/research-dense/output/campaign_walls_v1`,
  stage `walls_pilot_chunk4`).

## 1. Question

Can a tree model that predicts the ARM DIFFERENCE directly — dY between
paired sibling arms (same world + t0) from deployable state features plus
request-time action/wall descriptors — produce a contrast distribution
with genuine VALUE skill (CRPS better than predicting "no difference")
and sign skill on J4 wall pairs, where the v0.4a paired-level model had
sign skill but no value skill?

It is registered as baseline T for the WM-N1 Idea A comparison (A3.3):
WM-N1's paired dV must beat this direct-difference tree on the same
evaluation rule for the neural contrast to be worth its cost.

## 2. Sources and splits (HOLDOUT never; chunk-5 never)

**Data: the J4 wall pilot only** (chunk-4 wall branches and their chunk-4
native siblings). Split counts are counted BY NAME from the J4 plan
`E:/research-dense/output/campaign_walls_v1/results_plan/WORLD_PLAN.json`
(schema `wm59_walls_pilot_world_plan/1`, plan_seed 20261101, 44 worlds,
4 wall branches per world, 176 branches). No run directory, receipt
outcome or measurement of any CAL or HOLDOUT world was opened for this
registration.

- **TRAIN: 39 worlds** (156 wall−native pairs per tau):
  w8659m2700p10t420, w8660m2700p10t300, w8661m2600p10t240,
  w8662m2700p10t300, w8663m2700p10t480, w8664m2600p10t360,
  w8665m2700p10t240, w8666m2700p10t480, w8667m2700p10t420,
  w8668m2600p10t240, w8669m2700p10t360, w8670m2700p10t300,
  w8676m2700p20t360, w8677m2700p20t360, w8678m2700p20t300,
  w8679m2400p20t540, w8680m2600p20t300, w8681m2700p20t420,
  w8682m2400p20t240, w8683m2400p20t540, w8684m2700p20t360,
  w8685m2700p20t480, w8686m2600p20t240, w8687m2600p20t480,
  w8688m2700p20t420, w8693m2600p40t420, w8694m2600p40t240,
  w8695m2400p40t480, w8696m2600p40t420, w8697m2700p40t240,
  w8698m2600p40t300, w8699m2700p40t240, w8700m2600p40t420,
  w8701m2700p40t540, w8702m2700p40t300, w8703m2600p40t540,
  w8704m2700p40t480, w8705m2700p40t300, w8706m2400p40t360.
- **CALIBRATION: 5 worlds** (20 wall−native pairs per tau), read ONCE for
  the report only (§7): w8657m2700p10t300, w8658m2600p10t360,
  w8674m2600p20t300, w8675m2700p20t300, w8692m2600p40t300.
- **HOLDOUT: 0 worlds.** The J4 plan itself contains no HOLDOUT split
  (10 holdout worlds of the source chunk-4 plan were excluded at campaign
  design, `holdout_worlds_in_source_excluded: 10`). No HOLDOUT of any
  chunk is enumerated; `tools/holdout_guard.py` +
  `protocols/holdout_exclusions.json` apply.
- **Chunk-5 (sealed for WM-N1):** never touched — no chunk-5 file, plan
  or outcome is read. The chunk-5 TEST arms do not exist yet (J4B, law
  v2.2, is pending the owner's wall-v2.2 ack).

**Wall arms.** One wall branch = one pair per tau:
`dY = Y(wall branch) − Y(native sibling)`, both at the same world + t0
from the SAME saved state (the plan records one `saved_state_id` /
`state_xml_sha256` per world and per-branch `restores_state.from_run =
<world>__base` at t0; the native sibling carries the same
`saved_state_id` — asserted at build time, a mismatch refuses the world).
The native reference is the restored TRUE fleet-native chunk-4 arm
(`<world>__native`, `native_sibling_run` in the plan; schedule verified
native from switch_schedule + kind=="command" suffix owners, as v0.4a).
Wall arms (176): W-lo-early, W-lo-late, W-mid-early, W-mid-late;
`arm_speed_factors` W-lo 0.6, W-mid 0.85; `timing_offset_s` early 0.0,
late +60.0; hold draws over the 176 branches: 60 s ×52, 120 s ×70,
240 s ×54; branch duration 600 s; runs root
`E:/research-dense/runs/campaign_walls_pilot_chunk4`.

**Non-wall chunk-4 contrasts — CHOICE: AUXILIARY TRAINING ROWS
(training only; never scoring, selection-gating or acceptance), with the
wall-only ablation always reported.** Justification: (a) the headline
contrast has only 39 TRAIN worlds × 4 branches = 156 pairs per tau; the
non-wall chunk-4 arms (base/damping, pi, fixed×2, switching×3 vs the same
native sibling, ≈ 7 pairs/world ≈ 273 TRAIN pairs per tau) share the
identical paired-at-t0 structure and label machinery, and direct-difference
training is otherwise starved; (b) arm-family descriptors (wall tokens vs
policy/action tokens) let the model separate the contrast families, so the
wall effect is identified within the wall subset; (c) v0.4a's auxiliary
ablation showed aux rows did not change any verdict (coverage 0.880 →
0.879), i.e. the risk is bounded and measured, not assumed; (d) acceptance
is wall−native only, so aux rows cannot inflate the headline. Aux pairs
come only from the SAME 44 J4 worlds, split as in the J4 plan (never the
chunk-4 plan split — no split mixing, no CAL bleed).

**New data root:** `E:/research-dense/output/wd01_partB_v0data_v04b`
(NEW; older roots read-only). COPY only (never hard-link); MANIFEST with
per-source verification evidence; SHA256SUMS; backup under
`E:/research-dense-backups/wd01/`. Never read: dispatcher_rows.jsonl,
insertions.jsonl, any outcome block of branch_record.json.

## 3. Inputs (deployable + request-time only; no outcome-derived inputs)

Per pair, at t == t0, featurized from the SHARED base trace (the same
saved state; identical for both pair members):

- **deployable_strict (HEADLINE):** demand[6] (designated-CAV entrance
  counts, 2 entrances × 3 windows) + context[8] with the entrance-queue
  columns zeroed — `partB_v0data/deployable_features.py` SourceObs,
  variant "strict"; no all-vehicle entrance counts.
- **deployable_detector (SENSITIVITY):** demand[12] + context[8] with the
  all-vehicle counts and queues — variant "detector".
- **Wall descriptors (request-time planner state only),** reconstructed
  by `partB_v0data/wall_tokens_v04.py` from wall_events.jsonl +
  branch_record.json ALLOWED_RECORD_FIELDS (`wall_id, members,
  wall_speed_mps, hold_s, request_s, timing, speed_factor`): wall speed
  factor (0.6/0.85) and commanded wall_speed_mps, timing (early/late,
  offset 0/+60 s), hold draw (60/120/240 s), request tick, phase at t0
  (5-phase one-hot), planned_hold_remaining (PLANNER_OWNED, flagged),
  n_members, members_mean_s_down_m and members_lane_span from the MEMBER
  CAVs' self-reports at t only (members are CAVs; a non-CAV member is a
  loud error). FORBIDDEN_RECORD_FIELDS (outcome, assembled/released/
  aborted times, abort_reason) are never read.
- **Action/policy descriptors for aux non-wall pairs:** the v0.4a per-CAV
  action tokens with the unknown mask (CAV_RAW 20) and the planner-known
  schedule summary π (pi_native/pi_damping/pi_pi action shares), from
  switch_schedule.json + suffix_commands.jsonl via
  `partB_v0data/actions_v04.py`, plus an arm-family one-hot (wall /
  damping / pi / fixed / switching) so contrast families are separable.
- **Prohibited inputs (carried from v0.4a):** penetration, pre_fix,
  speed_mode, world strings. Nothing with timestamp > t. t0 recoverability
  stays fixed (ctx col 1 = 0). The ideal-sensing premise is stated (lane
  bins from the true state; the reconstruction gap is Part A's result).

## 4. Targets and pairs

- **Pair unit:** (chunk4, world, t == t0, arm) matched to its native
  sibling at the same t0 — the v0.4a `build_pairs` machinery, split fixed
  to the J4 plan.
- **Targets (per tau, as differences):** dY_delay = total_delay_ref_s
  (veh·s), dY_ttc = ttc_lt3_veh_s (TTC < 3 s exposure, veh·s), dY_brake =
  hard_brake_s (a < −3 m/s², WM-44 contract) — each
  `Y(arm window) − Y(native window)` over the same (t, t+τ]. Negative dY
  = the arm lowers the quantity.
- **tau set (as v0.4a):** τ ∈ {30, 60, 120, 300, 570} s, complete
  horizons only (at t0 the 600 s branches give complete horizons for all
  taus; truncated rows are dropped, tau_available recorded).
- **Contrasts:** wall−native (HEADLINE/gating; all four wall arms pooled
  per §6), and aux damping−native, pi−native, fixed−native,
  switching−native (training + reported breakdowns only).
- dY here is a REALISED single-draw difference per pair (one physics
  continuation per sibling), not a distribution pairing.

## 5. Model and selection

- **Model:** one quantile-GBM (HistGradientBoosting, quantile loss) per
  (tau × head × sensing variant) on dY DIRECTLY — no level head, no Δ
  head, no pairing of two predictions. Quantile levels
  (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95) as `partB_v0/gbm.py` QUANTILES.
  Pure quantile GBM (the v0.4a hybrid ridge-residual machinery is dropped:
  at 156 headline rows a residual decomposition has nothing to correct —
  disclosed as a deliberate simplification). The predictive dY
  distribution is the model's own 7-quantile function (piecewise-linear,
  flat tails), sampled on the same K = 40 midpoint grid as
  `partB_v0/acceptance_v04a.py` — identical CRPS code path for model and
  baselines.
- **Tuning:** world-grouped CV on TRAIN ONLY (6 folds by world; no CAL,
  no aux-family leakage across folds), fixed small grid of 6 candidates —
  (max_iter, learning_rate, max_leaf_nodes, min_samples_leaf):
  (150, 0.05, 8, 20), (300, 0.05, 8, 20), (150, 0.1, 8, 20),
  (300, 0.1, 16, 10), (150, 0.05, 16, 10), (300, 0.05, 4, 40) — selected
  per tau by mean pinball loss over the 7 levels (delay head). No other
  search. Conformal shifts for the dY quantiles are fitted per tau on
  TRAIN world-grouped OOB residuals only (the v0.4a CalMap method), then
  frozen.
- **Reported references:** ridge-on-dY on the SAME inputs (reported
  only); the zero-contrast and TRAIN-cell-mean baselines of §6.

## 6. Acceptance (fixed BEFORE running; strict = HEADLINE; ONE CAL read)

Baselines per (contrast, tau), from TRAIN pairs only
(`train_references`): **zero-contrast** = point mass at 0;
**TRAIN-cell-mean** = TRAIN mean dY per demand × penetration cell as a
point mass (cell key is a baseline label, never a model input; pooled
TRAIN mean as fallback). Sign threshold = the TRAIN median |dY| of that
(contrast, tau).

- **A1 calibration.** At every τ ∈ {30, 60, 120, 300, 570}:
  CRPS(model dY) < CRPS(zero-contrast); AND the overall (all τ pooled)
  10–90 % coverage of the conformal dY band in [0.75, 0.85].
- **A3 contrast skill, contrast wall−native, τ ∈ {60, 120, 300}:**
  - **(i) Sign accuracy:** of sign(median predicted dY) vs sign(dY) over
    CAL pairs with |dY| above the TRAIN median |dY| for that τ
    (near-ties don't count). The world-bootstrap 90 % CI lower bound must
    be > 0.5 (whole worlds resampled; N_BOOT 2000, seed 20261001).
  - **(ii) CRPS:** the model dY CRPS must be below BOTH the zero-contrast
    baseline AND the TRAIN-cell-mean baseline.
  - A3 passes if (i) AND (ii) hold at ≥ 2 of the 3 gating taus.
- **VERDICT:**
  - **ACCEPT** iff A1 AND A3 (wall−native).
  - **PARTIAL** iff exactly one of A1, A3 holds.
  - **FAIL** otherwise.
  - The detector variant is judged by the same rule, labelled
    SENSITIVITY. The wall-only (no-aux) ablation is reported with its own
    labelled verdict.
- **Gating is on the delay head only** (v0.4a precedent); the TTC<3 and
  hard-brake heads are reported, not gating (owner question 4).
- **Always reported:** per-arm (W-lo-early … W-mid-late) and
  per-speed-factor / per-timing breakdowns of (i)+(ii); aux-ablation
  verdict; the aux contrasts' own A3-style numbers (reported only);
  TRAIN-OOB versions of A1/A3 (labelled in-sample-selection-free);
  demand/penetration strata as report strata only.

## 7. CAL use (read ONCE)

- The 5 J4 CAL worlds (§2) are read ONCE, for the acceptance report only,
  after tuning, fitting, conformal calibration and every selection step
  are frozen. Lock file `CAL_READ_ONCE_V04B.json` written before the
  first CAL row, with start/finish timestamps.
- **No refit after the CAL read.** A crash before the CAL read may be
  continued; a crash after it is reported as is.
- Bookkeeping disclosure: these CAL worlds were read once by v0.4a
  (2026-09-30, its report) and are C4L1's Part A look set; this run adds
  ONE more disclosed read, for THIS report only.

## 8. Budget and stop rule

- **Zero SUMO. CPU only.** Laptop, ≤ 4 workers (owner standing limit).
- Estimates: build (44 worlds, t0-row-only features/labels) ≤ 1 h;
  tuning + 2 variant fits + ablation ≤ 1.5 h; CAL read + report ≤ 0.5 h.
  Declared cap: **6 CPU-hours total**, one executor task with the owner's
  time limit per continuation.
- **Stop rule:** no refit after the CAL read (§7).

## 9. Disclosures

1. **Wall law v2.1.** J4 ran under wall law v2.1 ("walls held through the
   merge"), not the owner's intended v2.2 (walls release at the merge
   start; `REGISTRATION_WALL_V22`). The chunk-5 TEST arms will run v2.2:
   a train/test law shift, already disclosed in WM-N1 Amendment 2 §5.
   v0.4b never claims v2.2 validity.
2. **Road-end relabel.** In the 176 completed J4 runs, 28
   `member_left_network` aborts (hold 60: 1, 120: 12, 240: 15; both
   members last on exit edge 1487606178) are RELABELLED road-end events
   for the WM-N1 corpus (`road_end_relabelled=true`); J4 receipts keep
   their original labels. The relabel changes event labels only; v0.4b
   consumes trajectories and request-time fields, and no event label is a
   model input. Disclosed for traceability (WM-N1 Amendment 2).
3. **Small n.** 39 TRAIN worlds / 156 wall pairs per τ; 5 CAL worlds /
   20 wall pairs per τ. Every CI is world-clustered; coverage on 20 CAL
   pairs is coarse — this is why A1's verdict and the small-sample risk
   are owner question 3.
4. **Ideal-sensing premise** carried from v0.4a (§3).
5. **CAL exposure** history disclosed in §7.
6. **No D: paths** (WD_FORBID_D=1); no outcome-derived input anywhere;
   HOLDOUT never opened (J4 contains none); chunk-5 never (sealed for
   WM-N1).

## 10. Outputs

- `partB_results/v0.4b_strict` and `partB_results/v0.4b_detector` (NEW
  dirs), each with eval_report + verify.
- `output/wd01/CHANGES_V04B.md`; `REPORT_V04B.md` (the §6 verdict table);
  `LEDGER_DRAFT_PB_V04B.json` in the experiment_history ENTRY schema,
  ledger id **WD-01-PB-V04B**; `LEDGER_DRAFT_PB_V04B_ALLOCATION.json`
  committed BEFORE the build. Data root per §2.

## 11. Open questions for the owner

1. **Aux choice (§2).** I fixed: non-wall chunk-4 contrasts ENTER
   TRAINING ONLY (aux), with the wall-only ablation reported alongside.
   Confirm, or flip the headline to wall-only training.
2. **Gating contrast (§6).** I fixed: one pooled wall−native contrast
   (all four arms), with per-arm / per-factor / per-timing breakdowns
   reported. Confirm, or gate per speed factor (2 contrasts).
3. **A1 with 5 CAL worlds.** The 10–90 % coverage band [0.75, 0.85] is
   estimated from ~100 CAL rows per tau (20 pairs × 5 taus) pooled to one
   number. Keep A1 gating (proposed), or demote A1 to reported-only given
   the small CAL?
4. **Safety heads.** TTC<3 and hard-brake dY heads are fit and reported,
   NOT gating (proposed). Confirm.
5. **Sequencing.** Fitting starts only after this draft is accepted as
   the registration. Confirm that no dependency on the wall-v2.2 ack
   decision is intended (v0.4b is offline on existing J4 v2.1 data; zero
   SUMO).
