# REGISTRATION: WD-01-PB-RESCORE1: re-score Part B (≤ v0.3.2) against the chunk-2/3 TRUE native/PI references (zero simulation)

**Registered:** 2026-09-30 by the world model session, before execution.

**Authority:** owner decision, 2026-09-30 about 09:40: "re-run old analyses vs
the new true references (zero-sim) — Part B contrast re-scoring against the
chunk-2/3 truearms where applicable".

**Type:** ANALYSIS. No training, no refit, no model selection.

## 1. Questions

- **Q1 (label audit).** Per world and per τ ∈ {30, 60, 120, 300, 570}, how
  different is the realised remaining delay under three policies?
  - **switching_mix:** the chunk-2/3 `__native` / `__pi` arms that were
    mislabelled "native" (WM-72).
  - **TRUE fleet native:** J3 / WM-73, `campaign_stage1_chunk{2,3}_truearms/<w>__native`.
  - **TRUE fleet PI:** J3 / WM-73, `…/<w>__pi`.

  All three share the same saved state at t0, and the damping base is the
  common reference. This quantifies how much the relabel changes the
  "damping vs native" facts that earlier reports stated.
- **Q2 (proxy skill).** The v0.3.2 deployable models (strict = HEADLINE,
  detector = SENSITIVITY) predict V(s, τ, damping) and V(s, τ, switching_mix).
  Their old-interpretation contrast is ΔV_old = V(damping) − V(switching_mix).
  - Score ΔV_old against the REALISED TRUE contrast
    ΔY = Y(damping base) − Y(true native).
  - Measures: sign accuracy, CRPS of ΔY under the paired predictive
    distribution, bias, and a comparison vs the zero-contrast baseline.
  - This answers whether any earlier "damping vs native" value claim survives
    when native is true native. (Q2 uses existing models only.)
- **Q3 (inventory).** List every earlier WD-01 report or claim that relied on
  the `__native` arms: Part B v0.2–v0.3.2 policy contrasts, A2, PA2G-A,
  stage1C-2 Part B/C where WM-owned, and C2L1 (never run).
  - For each: its status after Q1/Q2 (stands / corrected / withdrawn), with a
    one-line reason.
  - Planner-owned analyses (PL-02a) are out of scope and only listed.

## 2. Data

- **References:**
  - runs/campaign_stage1_chunk2_truearms and runs/campaign_stage1_chunk3_truearms
    (J3/WM-73; 140 worlds × {native, pi}; graded 280/280, reg b9fe37e5);
  - the switching arms and damping bases from runs/campaign_stage1_chunk2 and
    runs/campaign_stage1_chunk3 (plus chunk1_fix27 / verify27 if chunk-1
    truearms exist; otherwise chunk 1 is out of scope).
- **Splits:**
  - TRAIN and CALIBRATION worlds only, from each chunk's WORLD_PLAN.json.
  - HOLDOUT is NEVER opened: the split is checked BEFORE any file access, and
    tools/holdout_guard.py plus protocols/holdout_exclusions.json apply.
  - The truearms roots contain HOLDOUT worlds; those are skipped by name
    before opening.
  - w8477 (disclosed exposure) follows its recorded status in
    holdout_exclusions.json.
- **Labels:**
  - Built with the v0.3.2 label code (partB_v0data build_v0data / v0lib, the
    physics departed_ids, bases_root-aware; no insertions.jsonl, no
    dispatcher_rows).
  - They go into a NEW root E:/research-dense/output/wd01_partB_truearms_labels_v1,
    COPY only (never hard-link), with SHA256SUMS and a backup.
  - v0.2–v0.3.2 roots are read-only.
- **CAL use:** Q2 scores v0.3.2 on CAL pairs. This is a report-only re-read of
  CAL, so no selection or calibration follows. It is disclosed as the second
  CAL read for v0.3.2. TRAIN-pair results are reported alongside,
  in-sample-labelled.

## 3. Method

- **Q1:**
  - Per (world, t0) and per τ, compute Y under each policy from the same
    decision grid used by v0.3.2.
  - Report distributions of Y_switch − Y_native, Y_PI − Y_native and
    Y_damping − Y_native, by demand × penetration.
  - Report the sign agreement between the old (damping vs switching) and the
    true (damping vs native) contrast.
- **Q2:** v0.3.2 predict on matched states for both policies.
  - The CRPS of ΔY uses the M × quantile predictive difference, with
    independent-member pairing as in the v0.3.x contrast code.
  - Sign accuracy of E[ΔV_old] vs ΔY, with a world-bootstrap 90 % CI.
  - Baselines: zero contrast, and the TRAIN-mean true contrast per
    demand × penetration cell.
- Raw rows are saved before aggregation.

## 4. Budget

- Zero SUMO, no training.
- CPU: label build (estimate from v0.3.2: ≤ 2 h at 4 workers) plus scoring
  (≤ 30 min).
- One ZCode task (Flash, 240 min).

## 5. Outputs

- output/wd01/partB_rescore1/REPORT_PB_RESCORE1.md: Q1 tables, Q2 table, and
  the Q3 inventory.
- LEDGER_DRAFT_PB_RESCORE1.json in the protocols/experiment_history.json ENTRY
  schema. Put answers and next steps inside `results` / `limitations`, not in
  extra keys.
