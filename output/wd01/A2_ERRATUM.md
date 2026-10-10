# ERRATUM (FINAL, 2026-09-29): A2 step 2 ("30 s action effects are below single-realisation noise")

Status: FINAL. H1 was confirmed by the reconstructor on 2026-09-29: 3 fresh w8488 runs from t=0 with seeds 84881/84882/84881 gave identical digests over 1200/1200 ticks. TraCI reports imperfection −1 (IDM: sigma is not applied), speedDeviation 0 and actionStepLength 0.5.

## Original claim

From A2 step 2 (ZCode e5265607), repeated in REPLICATE_DESIGN_DRAFT.md §0:

> The paired 30 s deltas (arm − native) are below SUMO's single-realisation
> stochastic noise; the naive "CAVs as native" predictor wins.

## Correction (under H1)

**Truth is exact per root.**

- The campaign SUMO configuration has no stochastic element after t0 (the
  REP-P gate, w8488).
- Each arm-vs-native pair restored from the same t0 state is therefore the
  exact effect of the action change for that root.

**The spread is not noise.** The σ ≈ 15 veh·s of paired deltas is the
variation of true effects ACROSS roots, not within-pair noise.

**The zero-Δ predictor won because of model error.**

- Part A's predicted paired deltas had MAE 11–16 veh·s.
- The true effects were small (mean −0.15; 95th percentile of |Δ| about 34).
- So the naive zero-Δ predictor won because Part A's paired-Δ error exceeded
  the effect sizes. That is model error.

**Consequence.**

- Paired-Δ (action) validation needs no replicates.
- The existing chunk-1–3 branch pairs are exact targets.
- The G-2 plan uses them (G2_VALIDATION_PLAN_DRAFT.md).

## Records

- **WD-01-DEV-2 note:** an addendum records this re-interpretation. The
  validation-look count is unchanged.
- **REPLICATE_DESIGN_DRAFT.md §0 reading 1–2:** superseded by this erratum.

## Related correction: p3e "dawdle" (Part A)

p3e's residual of about −σ·accel·dt/2 per vType was read as SUMO dawdling. On
the campaign data it cannot be dawdling, because SUMO applies no imperfection
under IDM.

The most likely source is a systematic mismatch between SUMO's IDM update
(actionStepLength 0.5; SUMO's velocity/position update order and ballistic vs
Euler integration) and our emulator's integration. The fitted "dawdle"
absorbed that mismatch. Part A v2 must reproduce SUMO's exact update; the
oracle-vType exactness test will show it. Fitted dawdle is set to 0 by
default.
