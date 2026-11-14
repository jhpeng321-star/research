# REGISTRATION_WM_N1 — AMENDMENT 4: member-1-first improvement loop, CAL-only selection

**Status:** FROZEN, 2026-10-01. Owner-approved, verbatim: "approve
amendment 4 with the split" (relayed by main discussion).

- Amends `REGISTRATION_WM_N1.md` (`666233aa…`) and amendments 1–3
  (`decb9656…`, `56fa99a0…`, `a597d986…`).
- **UNCHANGED:**
  - the §4 targets and the decision rule;
  - the chunk-5 TEST as the single final test (`f8c7d760…`), sealed;
  - HOLDOUT sealed;
  - the variants and the tie order (T < A < C < N);
  - the held-out-strategy tests (A3.4).

## A4.1 Order

1. Train Idea A member 1 (Amendment 1, A1) and STOP: members 2–5 are not
   started.
2. On gpuB, train only held-out test (i) (J4 → J4B, 1 member), then STOP.
   Tests (ii), (iii) and the N comparator stay unqueued until a later owner
   decision.
3. Run the improvement loop below on member 1.
4. Members 2–5 then train with the WINNING configuration and NEW seeds,
   recorded before training. They are gated by an owner
   `APPROVAL_A_M2_5.json` bound to the winning-config sha and the m1
   checkpoint sha.

## A4.2 Improvement loop (CAL-only selection)

- **Knobs (at most 4, all a priori):**
  1. learning rate: {0.5×, 1×, 2×} the registered default;
  2. epochs: {1×, 1.5×} the registered default;
  3. multi-step rollout-loss weight: {0.5×, 1×, 2×};
  4. paired-contrast-loss weight: {0.5×, 1×, 2×}.
  Architecture, inputs, data, splits and all other losses are frozen.
- **Rounds:**
  - at most 3 rounds;
  - at most 4 configurations per round, including the current best;
  - round 0 = the registered default config.
- **Quick-tune:** rounds may use the registered quick-tune setting: a fixed,
  world-stratified TRAIN subset and/or a fraction of the epochs, seeded. This
  is allowed only after a rank-fidelity check in round 0. That check runs 3
  configs both quick and full, and requires Spearman ρ ≥ 0.8 between the
  quick and full CAL scores; otherwise rounds use full runs.
- **Selection metric (single, a priori):** the CAL mean paired dV CRPS over
  the gating contrasts at τ ∈ {60, 120, 300}. This is the same quantity as
  the design §5 variant rule, subject to 10–90 coverage in [0.75, 0.85].
  - Ties within 2 % go to the configuration closest to the default.
  - No other metric is consulted for selection.
- **CAL split (owner-approved, MANDATORY):** the CAL worlds are split ONCE,
  BEFORE any training, into two halves:
  - **CAL-tune**, used ONLY for this loop's selection;
  - **CAL-cal**, used ONLY for the registered early stopping, calibration,
    blend weights and variant choice.

  How the split is made:
  - It is world-level, by NAME from the chunk WORLD_PLANs only (no outcome is
    read), and stratified by chunk × loading × penetration.
  - Seed 20261001.
  - Within each stratum, worlds are sorted by name and shuffled with the
    seed. The strata are concatenated in sorted order, and worlds are then
    assigned ALTERNATELY (tune, cal, tune, …; the first label is drawn with
    the seed). This keeps the halves balanced to ±1 and spreads every stratum
    across both.
  - It is materialised as `wm_n1/CAL_SPLIT_A4.json`. Its LF-sha is recorded
    in `A4_FREEZE.json` before training starts.

## A4.3 Sealing and disclosure

- TEST (chunk 5) and HOLDOUT are never read by the loop.
- J4B is read only by held-out test (i)'s registered evaluation, never for
  tuning.
- Every round's configs, CAL-tune scores and the winner are logged with shas
  before the next round starts.
- Rounds and configs that were tried are reported in full, including the
  losers.

## A4.4 Compute

- **Per full m1 run:** ≈ 9–17 GPU-h on an A10G-class card. The 4090 D figure
  comes from the timing gate.
- **Per quick-tune round:** target ≈ 2–4 h.
- **Loop cap:** 3 rounds × 4 configs.
