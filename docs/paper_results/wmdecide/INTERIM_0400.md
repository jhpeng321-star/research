# Credibility interim — matched 04:00 saved-forecast snapshot

**FULL-v1 improves delay CRPS but does not establish calibrated action confidence.**
This analysis uses four complete K8 worlds shared by WM2 and WM-PHYS-FULL-v1:
969432, 969433, 969434, 969436. All five registered actions and horizons
120/300/600 s are retained. The snapshot has 17 complete WM2 worlds and five
complete FULL-v1 worlds; only these four are compared. Availability selection is
operational, not an outcome-based candidate/world selection or full-panel result.

## Portable sections and receipts

- [WM2 credibility](matched_0400/00_wm2/CREDIBILITY.md).
- [WM-PHYS-FULL-v1 credibility](matched_0400/01_WM-PHYS-FULL-v1/CREDIBILITY.md).
- [Publication index](matched_0400/INDEX.md); `REPORTS.json` binds full raw reports,
  source audit/manifests and compact pooled summaries; `FILES.json` binds exports.
- Full outputs remain at `runs/wmdecide_wm2_matched_0400_e8302db3` and
  `runs/wmdecide_full_v1_matched_0400_e8302db3` in this worktree. These reuse INT's
  immutable `runs/rbint_evidence/{wm2_0400,full_v1_0400}` outputs, revalidated by
  EVAL's `read_predictions`. No inference or real-data extraction was repeated.

## Scores, in 120 / 300 / 600 s order

- Absolute-delay empirical CRPS (veh-s, lower better): WM2 **2082 / 8508 / 29894**;
  FULL-v1 **1297 / 4996 / 18852**. Both still lose to the same cell-conditioned
  climatology, approximately **697 / 3026 / 6110**. Point improvement over WM2
  is not sufficient distributional qualification.
- Paired-delay CRPS across all ten unordered action pairs: WM2
  **396 / 1992 / 3988**; FULL-v1 **136 / 510 / 2176**. FULL-v1's model-minus-cell
  climatology differences are **-7 / -309 / -268** veh-s (point estimates).
- All-pair preference Brier: WM2 **0.357 / 0.203 / 0.233**; FULL-v1
  **0.334 / 0.289 / 0.174**. FULL-v1 worsens the 300 s preference score despite
  improving paired-delay CRPS. Pairwise probability quality needs its own check.
- Upstream breakdown Brier: WM2 **0.220 / 0.0180 / 0.00234**; FULL-v1
  **0.299 / 0.0164 / 0.00313**. The 120 s score worsens. Merge/ramp and
  cell/action results remain in the model sections and full summaries.
- Absolute-delay inclusive K8-range coverage: WM2 **0 / 0 / 0%**; FULL-v1
  **10 / 40 / 25%**, versus **77.8%** under continuous exchangeability. Central
  50/80/90% coverage is also published. Reduced delay error has not fixed coverage.

## The actual 0.8-rule calls, not arbitrary pair orientations

For each original state/horizon, evaluate the registered rule from all five
hypothetical incumbents, using predictions only. Each selected action must have
lower expected delay and at least 7/8 strict paired wins. These are clustered
retrospective checks against recorded paired-arm outcomes, not switching-policy
performance. Selected-call cohorts differ between models; their Brier scores
must not be mistaken for a same-target model-comparison score.

- WM2: mean selected win probabilities **95.5 / 92.9 / 88.9%**; observed strict
  wins **8/11 (72.7%) / 6/7 (85.7%) / 8/9 (88.9%)**.
- FULL-v1: mean selected win probabilities **94.2 / 96.9 / 95.5%**; observed strict
  wins **4/13 (30.8%) / 6/12 (50.0%) / 10/11 (90.9%)**. Selected preference Brier
  **0.603 / 0.462 / 0.0739**; differences versus its same-pair cell climatology
  **+0.240 / -0.0540 / -0.0852**. Early-horizon confident calls warrant attention
  even though the 600 s point result looks much better.

## Uncertainty and adoption limits

The four matched worlds occupy singleton demand/penetration strata. The frozen
stratified whole-world bootstrap has no within-stratum resampling variation, so
these credibility CIs are **unavailable**, not zero-width confidence. Neither
20 action outcomes nor 40 action pairs are independent worlds. All figures are
diagnostic with no finite-K or shared-model-error qualification; one recorded
continuation per observation cannot identify a same-state conditional probability.
The original EVAL distributional and HDV60 non-regression gate remains authoritative.
FULL-v1 is distinct from the newer FULL-v3-gpu runtime; never transfer this result
to a changed physics/sampling source. No model adoption is inferred here.
