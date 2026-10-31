# Paired uncertainty correction — 2026-11-01 05:25 +08

The point estimates are unchanged. The published ~04:00 FULL-v1 and first FULL-v3
paired confidence intervals are RETRACTED. Their implementation bootstrapped the
full registered factual panel and then restricted its weights to available scored
worlds. Drawing an unscored peer from the same cell changed the scored subset's
demand/penetration composition; it supplied false variation even when every
scored stratum had exactly one world.

Correct implementation resamples ONLY scored complete paired worlds, preserving
demand x penetration x split composition. The four currently matched worlds
969432/433/434/436 each occupy a distinct cell. Conditional on saved K8, this
supplies NO between-world sampling variation. Reported 95% paired CIs are NA,
not zero-width intervals and not evidence of statistical superiority.

The same defect existed in EVAL's partial-panel bootstrap construction. The new
local scorer now builds its bootstrap from scored targets, keeping factual
development rows available for baseline construction. Existing immutable remote
scorers/runtimes are not modified. Finite-K contributions explicitly supported by
EVAL remain separate from world sampling; our paired diagnostics condition on K8.
Constant bootstrap draws are explicitly rejected even if floating-point np.std
produces a tiny nonzero number. Missing precision cannot imply qualification.

Corrected outputs: full_v3_gpu/paired_0525_corrected,
full_v3_gpu/distribution_0525_corrected, hdv60/paired_v3_0525_corrected.
Original paired_0400 / *_0545 outputs remain retained for audit, superseded.
Snapshot directory label0545 was administrative and premature: actual FULL-v3
snapshot Unix1793481355.3513765 =05:15:55.351377+08; no timing-based selection.
Likewise the pre-canary registration's05:30 label was premature; registration and
smoke preceded actual canary dispatch Unix1793481252.7919495 (~05:14:12+08).

Verification: comparator/endpoint/EVAL CPU suite21PASS/1historical-canary skip,
including regression with two scored singleton cells and two unscored same-cell
peers: weights remain EXACTLY1, means retained, no confidence intervals invented.

Current point results FULL-v3 minus WM2: CRPS120/300/600−573/−3359/−11150veh-s;
300s breakdown-distance+2.5pp (worse); HDV60−33.94m on574common HDV/arm endpoints.
No switching eligibility or calibrated confidence is established.
