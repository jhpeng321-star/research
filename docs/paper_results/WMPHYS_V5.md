# WM-PHYS v5 — integration progress; final comparison pending

**The four fixes are integrated, but round-2 performance is not yet established.**
FULL-v5 and FULL-v5-LP pass the local physical integration tests and two-world
60s recorded-start smoke. FULL-v5 also passes the all-five-arm600s development
canary; LP passes its separate full canary too. Both are engineering-ready for the
teacher-quality rerun, frozen before new final outcomes.
The matched final evaluation is now running on both GPUs; distributional scores,
confidence intervals and switching eligibility are pending. This is not the
completed Nov2 delivery.

## What exists

- Immutable engineering model stage: sourceb2df7458 on gpuA; exact roots, owned
  PIDs, resource admission and completion receipts: `../rb2int/CURRENT_JOBS.md`.
- FULL-v5 combines causal fitted LC continuation, physical latent-spread fitting,
  current-lane SUMO IDM speed limits and consistent CAV admission/execution settings.
- FULL-v5-LP differs only by the learned physical departure-parameter mixture.
  Neither candidate adds a neural residual or learned acceleration/lane/merge/
  admission correction. The original WM2 checkpoint/encoder is preserved.
- Integration repaired an actual composition defect: attaching LC parameters
  rebuilt the base hidden state and silently removed CAL's entrant-mixture wrapper.
  The joint object now preserves both behaviors; regression gate passes.
- All-four-source integration135CPU PASS; infrastructure56PASS/1historical-canary
  skip; LANE intake92PASS/1CUDA skip; no successful final forecast repeated.

## Registered comparison and limits

- WM2 versus FULL-v4 versus FULL-v5 versus FULL-v5-LP; all five original arms,
  K8, H120/300/600. Reserved8 plus all50 recorded WMDATA2 worlds, metadata-selected
  before new final outcomes. Exact design/registration: `../rb2int/`.
- Primary54 worlds occupy eight replicated traffic cells. Four remaining cells
  are singletons and descriptive. They cannot supply full-panel fixed-stratum CIs.
  The original full12-cell gate and its rate-precision standard remain unchanged;
  this recorded panel cannot establish full qualification.
- Paired whole-world fixed-cell uncertainty; finite-K uncertainty for forecast
  rate means; conservative simultaneous interval propagation for absolute rate
  distances. No arm/future pseudoreplication or unscored-peer bootstrap variation.
- Staged parameter-artifact declarations and inherited hashes show no direct
  final-panel fit/selection overlap. Historical aggregate exposure is disclosed;
  these are not historically untouched research data. Final release remains gated.

## Remaining work / teacher status

Baseline/source/input parity and final release are complete. Execute only the
scheduled missing identity-bound tasks; score all models on matched
complete K8 worlds, distributional components, credibility and switching rule.
Keep source-bound v4 decision maps separate from v5. The FULL-v5 READY signal and
immutable source/recipe/smoke binding in HANDOFF.md are now published for
S-WMTEACH's engineering/quality rerun; final qualification is pending.

<!-- RB2-INTERIM:BEGIN -->
## INTERIM — owner16:28 early-delivery addendum

The final panel remains active. A cache-only publisher now prepares the first
matched K8 scored cell with at least two distinct recorded worlds, and the complete
reserved8 comparison as soon as available, then refreshes each view about every3h.
It reports120/300/600s delay CRPS, breakdown-rate distance, HDV60, controller-effect
signs,0.8call reliability and range coverage. Singleton confidence intervals are
NA; available-case replication never substitutes for complete registered54.

The existing supervisors have no live queue-reorder interface. Their in-memory
queues, claims and frozen sources are preserved; safe reordering without restart
is unsupported. Only the original collector transfers forecasts; the new publisher
reads completed local cache entries. Exact continuation/final handover:
`../rb2int/INTERIM_CONTINUATION.md`.

Short30/60/90s forecasts preserve exact total-delay increments, but recorded tapes
have no saved total/source clocks at those horizons. Reconstructed road clocks
match all290 recorded120s road totals. Total-delay direction accuracy will be
reported as identification bounds using nonnegative source waiting bounded by its
recorded120s total, plus resolved-pair fraction and separate road-only accuracy.
These five controller options do not establish unrecorded cooperation/group
effects. Planner/cooperation short-horizon qualification remains NOT ESTABLISHED.

Later development handoffs flag two frozen-candidate limits: LANE rejects fitted
LC adoption on its standalone free-rate/recall evidence; BREAK reports116b2147
targets a retired mode2 mirror, not the actual shared CAV actuator. Thus verified
mode2 actuator consistency must not be claimed for this frozen version. Both
findings are disclosed while the fixed four-model comparison continues unchanged.
<!-- RB2-INTERIM:END -->
