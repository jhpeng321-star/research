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
