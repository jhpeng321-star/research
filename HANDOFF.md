# S-RB-INT handoff — 2026-11-01 05:15 +08

Branch `c4/rb-int`. Retain all prior work and immutable jobs. Earlier detail:
`docs/paper_results/WMPHYS_INT_HANDOFF_ARCHIVE_{0034,0515}.md`.
Owner schedule: interim04/08, freeze09:30, evaluation through11:30, report11:45.

## READY FOR EVAL / existing ownership
- FULL-v3-gpu LIVE, sourcef378ac6033079bdd87c0059627e27e394a1d7cbe:
  gpuA `/root/rbeval20261031/phys_code_f378ac603307/configs/wm_phys.toml`.
  Own370037/370038/370039, monitor370088; root `/root/rbeval20261031/int/WM-PHYS-FULL-v3-gpu`.
  Canary223.28s reused; causal motion posterior, learned arrivals, physics-native
  CAVs and fast lane arbitration. Original observed-ID60s endpoints retained.
- FULL-v1 LIVE source4015b641; own333463/333464/333465, monitor335983,
  gpuA `/root/rbeval20261031/int/WM-PHYS-FULL-v1`. Saved-ID crosswalk unverified;
  newer FULL-v3 removes that dependence. Never redispatch either source.
- A2/C2 owned by RUNA; B2 owned by RUNB. Shared `rb-shared/evalclaims/` binds claims.
  Old failed A/C sources and all STOPs retained. No peer process action.
- WM2-HDV60 LIVE gpuR, original WM2 runtime, short120step prefix-matched protocol;
  own812962/812963/812964, monitor814672; root `/root/rbeval20261031/int/WM2-HDV60`.
  Last05:03 audit45tasks, FULL-v3 34tasks; all workers/monitors live, no STOP.
  Original WM2 H600 panel/finalizer remain EVAL-owned. SUMOCORE144/144 complete.

## Next candidate (not READY until fresh smoke)
- Merged LANE b8037c1d/aefc4bd9: signed keep-right/cooperative early returns,
  float32 endpoint normalization, urgency/message presence and blending.
  Three-value step output reconciled with INT ramp ownership and fast arbitration.
  Targeted66CPU PASS/1CUDA skip. Full lane precision still poor (default3.006%,
  encoder2.400%; recall37.942%/40.492%, conditional FIT32/fivearms).
- MERGE3ce7aa6a intake: post-movement source phase available; retain INT's
  default-OFF exact merge query switch. Merge/phase/receipt/combined/query suite
  59CPU PASS/1CUDA skip. Multi-step evaluation wiring now explicit below.
- NEW explicit recipe rollout.departure_phase='after_movement_and_lanes' now
  selects inference-only multi-step adapter (shared model default unchanged).
  Carries settings/history/absolute controller clock; repairs cost observer's
  old-road cohort, preserving exit speed/no birth charge and source ordinals.
  Phase/recipe/source suite27CPU PASS; fresh smoke/canary still required.
- Complete touched-module CPU gate287PASS/4skips/60.55s (two local CUDA,
  optional SciPy prefix diagnostic, unfetched historical GPU canary). Smoke next.
- CF47b7742c and UNC84ee27b9 already integrated; FULL uses frozen
  FIT shrinkage_v1 population and causal condition_motion, not saved tape IDs.
  Learned strength19.35568 already applied. Fitted LC remains OFF.
- All recipes SUMO IDM/LC2013/highway merge; learned motion/lane/merge/courtesy/
  admission/noise corrections OFF. A encoder+uncertainty, B population,
  C fitted population point, FULL fitted joint posterior+learned Gamma-Poisson arrivals.
- Current SUMO-merge HDV integrator ALREADY uses Euler; CAVs retain registered
  executor plant. Legacy/checkpoint paths default unchanged. No training tonight.

## Evidence and next actions
- Recovered100episode factual60s observable targets, committed ee6bb6d2:
  `docs/rbint/hdv60/FACTUAL_065e9853.json`. Extraction not repeated.
  Exact t0 identity/junction carry matching; future targets isolated from forecasts.
  Existing Miniconda endpoint/protocol suite9PASS. Paired common-K8 scorer next.
- ~04 report:4paired reserved worlds, FULL-v1−WM2 delay CRPS120/300/600
  −785[−1264,−306]/−3512[−5825,−1200]/−11041[−16268,−5815]veh-s;
  300s rate-distance+1.25pp[−4.17,+6.67], HDV60CI missing. NOT switch eligible.
- Existing20min read-only peer observer retained (`runs/rbint_poll`); do not start
  another observer. Both GPUs busy in last audit; preserve live evaluation jobs.
- NEW FULL-v4 canary395510/source22cc32f6 running gpuA (no redispatch), explicit
  move→LC→emit +final LANE. Registered before inference; shared exclusive claim.
- NEW05:45 paired FULL-v3 endpoints and delay, SAME4complete reserved worlds:
  HDV60 mean-world-median error−33.94m[−43.91,−23.97],574common endpoints;
  CRPS120/300/600−573[−979,−166]/−3359[−5620,−1099]/−11150[−16387,−5913].
  Rate-distance300+2.5pp[−1.67,+6.67] (worse point); NOT switch eligible.
  Endpoint/protocol/comparator suite18PASS; immutable manifests/attrition retained.
  WM2 full panel160/160tasks fetched read-only. No inference repeated.
- Next: complete source phase integration + CPU suite +8x60s recorded smoke,
  publish new immutable config/source, collect complete K8 matched worlds,
  endpoint CIs and DECIDE credibility, update ≤60line WMPHYS_MORNING.md.
- Remaining parity gaps: synchronous LC, incomplete serial negotiation/internal
  junction state and cold lane memory; no full distributional qualification.
