# Credibility interim — saved outputs collected 05:04–05:05 +08

**WM-PHYS variants improve delay scores on the matched pilot; probability
credibility remains mixed.** No qualification or picker adoption is established.

## Evidence and report entrypoints

- [Available-evidence sections](available_0505/INDEX.md): WM2 all20 complete worlds;
  FULL-v1 and A2 each6 (969432–969437); FULL-v3-gpu3 (969432–969434).
  These sections have different scored populations; use them independently.
- [Matched three-world comparison](matched3_0505/INDEX.md): all four model
  identities scored on EXACT969432/969433/969434, five actions, K8, H120/300/600.
- B2 has16 saved tasks but **zero complete K8 worlds**: no probability scores.
  C2 has no output JSON at collection: no score, no replacement dispatch.
- Collection snapshots/full raw reports are preserved under `E:/Temp/opencode/`:
  `wmdecide_{wm2,full_v1,full_v3,a2,b2}_0510_c5297f7a` and corresponding
  `wmdecide_*_report_0505` / `wmdecide_*_matched3_0505` directories. The `0510`
  snapshot basename is a version label; exact UTC collection times are in
  `SNAPSHOT.json` (21:04:40–21:05:04Z on October31).
- Each portable publication contains `REPORTS.json`, `FILES.json`, per-model
  `POOLED.json`, Markdown sections and three reliability SVGs. Full raw reports
  bind all row-level outputs, evaluator validation, source hashes and task bytes.

## Identical three-world results (120 / 300 / 600 s)

Absolute team-delay CRPS, veh-s, lower better:
- WM2: **1984 / 7046 / 26307**.
- FULL-v1: **1186 / 3222 / 14681**.
- A2: **1150 / 2596 / 11940**.
- FULL-v3-gpu: **1285 / 3158 / 14077**.
- Same-arm cell climatology: approximately **793 / 3351 / 6941**.
  All three physics variants beat WM2 point scores at all horizons, but only
  the300s point scores beat this cell-conditioned absolute-delay baseline.

All-ten-pair preference Brier (same targets for every model), lower better:
- WM2: **0.371 / 0.209 / 0.231**.
- FULL-v1: **0.431 / 0.307 / 0.114**.
- A2: **0.466 / 0.251 / 0.183**.
- FULL-v3-gpu: **0.346 / 0.241 / 0.102**.
- Thus every physics variant improves600s preference Brier, while every variant
  worsens300s preference Brier. Lower delay CRPS alone misses this tradeoff.

Selected0.8-rule calls (each hypothetical incumbent; model-specific selected
cohorts, clustered within world):
- FULL-v1 mean forecast **94.4 / 97.7 / 95.5%**, observed wins
  **3/9 / 5/11 / 10/11**.
- A2 mean forecast **96.3 / 95.3 / 95.3%**, observed wins **6/10 / 7/8 / 8/8**.
- FULL-v3-gpu mean forecast **93.1 / 93.1 / 98.8%**, observed wins
  **5/9 / 5/9 / 9/10**. These are retrospective original-state comparisons,
  not closed-loop policy efficacy or same-target selected-score comparisons.

Absolute K8 range coverage remains poor: WM2 **0/0/0%**, FULL-v1
**13.3/53.3/33.3%**, A2 **0/40/26.7%**, FULL-v3-gpu **26.7/53.3/33.3%**;
continuous exchangeability reference77.8%. Upstream120s breakdown Brier is
worse than WM2 for all three physics variants; region/cell details are published.

The common three-world panel has singleton cell strata, so the registered
stratified-bootstrap CIs are unavailable, not zero-width. No finite-K/shared-model
error or simultaneous-gate claim. S-RB-EVAL's full distributional and HDV60
non-regression gate still governs adoption. All input inference jobs and existing
finalizers retain their owners; this publication only collected/scored saved JSON.
