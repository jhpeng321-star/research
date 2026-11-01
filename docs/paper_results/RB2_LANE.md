# S-RB2-LANE — source repair and frozen LC continuation
Source fixes delivered; frozen fitted LC **NOT ADOPTED**. Base FULL-v4 `22cc32f6`.

## Code and parameters
- Commits `a29116e9`, `6b4bfbd5`; source comparison `docs/RB2_LANE_SOURCE_AUDIT.md`.
- SUMO1.27.1 keepRight/speedGain thresholds, fixed-point accumulation, signed gaps,
  lookahead and branch-specific decay audited. No generic `changeWish` decay exists.
- Fixed non-action lookahead updates; documented actionStepLength controls wishes,
  while simulation dt controls CF. Nondefault action-step secure gaps extrapolate
  conservative reaction-time speeds as MSLaneChanger851–925 does.
- Exact frozen `lane_raw_c5602102891e` population; SHA256 `b903fa49fe1f3f1a15428ae48de5c8f86863aa5c5821dbb0acc8275c77d4483f`.
- Five LC coordinates drawn jointly/persistently per observable driver and future;
  entrants use population. Independent IDM block unchanged; learned corrections OFF.
- Explicit `configs/wm_phys_rb2_lane.toml`: fitted_lc=true + lane_parameter_directory.
  Base recipe stays OFF. Own-start causal240–300 LC inference attaches after IDM;
  no native saved-ID posterior transplant. Replay sampled settings to warm wishes.

## Completed FIT diagnosis (conditional proposals, not free-running rates)
- All80 episodes: native969400–969431,H60/300; development969420–969431,fivearms,H60.
  K8 persistent LC draws, same frozen pooled IDM in both arms; no population refit.
- Native60: recorded626; default12,379 proposals/262TP, fitted507.25/60TP;
  precision2.116→11.828%, recall41.853→9.585%.
- Native300: recorded3,478; default58,237/1,443TP, fitted2,891.25/395TP;
  precision2.478→13.662%, recall41.489→11.357%; false positives fall95.6%.
- Native300 keep-right FP48,368→763.125 (−98.4%); precision0.761→1.437%:
  still poor. Fitted strategic/cooperative/speed-gain precision30.13/16.92/6.98%;
  all-event recall contributions8.054/1.021/1.962/0.320pp (last is keep-right).
- Development12/fivearms60: precision2.282→10.879%, recall43.168→8.959%;
  recorded1,193 vs22,568→982.375 proposal steps. Recorded reason labels unavailable.
- Native300 compressed positive runs3,716→706.5 vs3,478 events: count proximity of
  repeated positive steps hides under-frequency. Upstream lane0 step rate650.0→12.28
  vs recorded11.01/veh-h, but fitted run rate0.538. Counts/timing remain unsolved.
- Native300 ±2s run matching precision29.55→55.36%, recall31.57→11.25%; matched-only
  timing MAE.394→.326s does not establish overall timing fidelity.
- Native300 event Brier: empirical K8.043621→.002648, fair-K8.002436;
  never-change.002579. Fair correction removes sample-mean Monte Carlo variance,
  not model bias; no independent calibration or world-CI claim.

## Verification and handoff
- Entrants-ON native development12,K2,H300:48/48 free futures finite/conserved,
  no_slot=0, common IDM/arrival draws; both variants causally warm. Actual1,554 LC;
  default1,986 vs fitted1,048.5 expected. Count bias+36→−42.125/world,
  MAE36→44.625; empirical CRPS33.125→41.3542 (fair29.4167→37.75): worse.
- Full-traffic mainline upstream/merge/downstream rates per veh-h:
  real16.456/16.718/7.893; default26.046/13.899/9.017; fitted10.494/8.070/3.849.
  `free_v1/RESULTS.json` + lane_rate_10s.csv. No CI claim for K2/singleton cells.
- Four artificial initial internal→normal counts per variant excluded by ONE-step
  source-bound replay; no physical forecast changed. First-tape-frame target checked.
- 85 core CPU tests PASS (`CPU_FINAL.xml`) plus5 guard/scorer/ramp regressions PASS;
  Ruff PASS. Four real-checkpoint60s futures
  FIT969420/969421,K2 PASS finite/conserved/no learned corrections/warm LC state.
  Actual smoke source `6b4bfbd5`, CPU-only; no arrivals, so no distributional gate.
- Independent targets:4,432 nonboundary HDV events match, max |Δs|.005m;
  seven additional events at300.5 reflect the known first-tape-frame boundary.
- `docs/rb2lane/fit_v2/RESULTS.json` verifies exact shard reassembly and versions;
  adjacent motivation.csv / lane_segment.csv / timing_10s.csv provide detail.
  fit_v1 retained; v2 adds fair-K8 scoring/zero-count presentation, no inference rerun.
- Source/runtime differences from the older fitted kernel are explicitly hashed.
  Serial negotiation, critical internal leaders and nonlegacy acceptance incomplete.
- New exposure only FIT969400–969431; no WMDATA2 or reserved selection. All own
  workers exited; CPU-only. INT owns final adoption/evaluation; shared adverse result
  requests integrated-source LC ON/OFF development comparison before final selection.
- Byte-bound delivery: `docs/rb2lane/DELIVERY.json`; all12 owned jobs exited.
