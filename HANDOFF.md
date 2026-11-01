# S-RB2-BREAK — active, 2026-11-01 ~13:18 +08

Branch `c4/rb2-break`, FULL-v4 base `22cc32f6`. Plan `docs/rb2break/PLAN.md`.
Prior notes/inherited INT history preserved in `docs/rb2break/HANDOFF_1318_ARCHIVE.md`.
**No quantified congestion improvement or qualification claimed yet.**

## Available source-backed corrections (default OFF)
- `8addc19e`: `physics.sumo_cf_lane_speed_limits=true`. Current-lane22.22m/s
  ramp-approach limit × inferred speedFactor was missing in SUMO CF (insertion/
  merge already mapped it). Mainline/acceleration-lane CF unchanged, persistent
  driver v0 untouched. Does NOT change the frozen external CAV executor.
- `116b2147`: `variants.FULL.match_cav_execution_settings=true`. CAV admission/
  initial/lane geometry uses SAME registered motion-fitted physical vector that
  current executor already forces. HDV particles, controller, executor and RNG
  unchanged. Helper `wms.control.execution_settings`; minimal recipe hook.
- Tests: CF28PASS; targeted latest16PASS; exact query/parity9PASS/2CUDA skips.
  CAV/limit traffic screens queued; do not adopt on tests alone.

## Development evidence (all5arms, H120/300/600, per-cell detail in JSON)
- Frozen baseline96tasks/dev12/K8: `docs/rb2break/BASELINE_DEV_V2.json`.
  600s bias+13,485 =road+2,154 +source+11,331veh-s (84% accounting, NOT causality).
  Native mainline admissions−50.646vehicles/600s vs recorded, ramp−2.146;
  mean rate-draw excess only+0.713/−0.109 requests (`SOURCE_SERVICE_DEV.json`).
- Matched future0/K1 baseline600s bias+14,980, CRPS16,726. Don't compare K1/K8 scores.
- LC execution held: total delay+11,219, source+26 only; rate100%; reject shutdown.
  Wishes/advice retained: this is lateral-execution-only, not total LC-family attribution.
- Merge speed advice removed (stop/acceptance retained): delay+3,436, source+1.5;
  reject shutdown. `LC_COMPLETE.json`, `MERGE_COMPLETE.json`.
- departLane first: event rate100→90% but bias→+44,024; reject.
- departSpeed last: source bias→−33,595, road→+17,686, total→−15,908;
  CRPS→17,851 (worse); violates scenario max policy, reject cancellation fix.
  `DEPART_FIRST_COMPLETE.json`, `DEPART_LAST_COMPLETE.json`.
- Birth-population motion fit fa4a2793:300drivers/35,707rows,154birth-disjoint
  validation drivers/13,635rows;910birth matches/0unmatched. Physical acceleration
  CRPS0.163291→0.165448m/s²: motion gateFAIL, rejected BEFORE traffic inference.
  Hash-bound artifact `docs/rb2break/birthfit_v1_bound/` (no neural fitting).
- Tau homogenization changes600s delay−241/source−85; population-only current
  drivers−327/source+2. Neither fixes over-breakdown. `TAU/POPULATION_COMPLETE.json`.

## Live continuation / ownership
- gpuA own max2CUDA workers; root `/root/rbeval20261031/rb2break`.
- Initial immutable source6b2d4c01; remaining population/tau/v0/fixed-rate source
  `1c5fe0cf`, supervisors440884/440882. Population/tau COMPLETE;v0/rate now live.
  Read-only `python -B -m tools.rb2break.ops status`.
- CPU attempts retained, no completed task; exact own pidfds retired for runtime
  (not efficacy): import-shadow failure then12/30min budgets. Never mix CPU/CUDA.
- Raw-posterior (zero shrinkage), lane-limit and CAV-consistency CUDA screens
  follow in sourceb4012f02, supervisors450241/450254, max2 total.
- Bounded waits and hash-verified snapshots in tools/rb2break; no automatic retries.

## Exposure / integration
New fitting/selection ONLY969420–969431. Frozen population/shrinkage inherit
FIT969400–969431 and registered CAV TRAIN artifact; no additional TRAIN tapes.
No reserved969432–969439 raw tasks/starts used; no WMDATA2 access. Historical
owner-required aggregates read; mixed-split JSON parsed only to extract DEV records.
Learned corrections OFF, no neural residuals, no outcome offsets/jam thresholds.
INT owns final freeze, exposure audit, replicated recorded-world gate and valid CIs.
