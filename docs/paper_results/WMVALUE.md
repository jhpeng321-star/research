# S-WMVALUE — decisive-test inference RUNNING (2026-10-31)

**MAIN21:12 short-first execution active; complete B/C/D/E gate pending inference.**
The WM-trained cooperative policy remains the target. Original freeze12685912;
amendment03dd03e0; frozen real fit a0acedb6; adapter3d863697; pipeline226b0a8e.

## Registered comparison: remaining fleet+source delay to absolute t900, veh-s

| Method | Sign accuracy | Spearman | MAE (veh-s) | World-cluster CI |
| --- | --- | --- | --- | --- |
| A: WM2 b8797242 + mix to endpoint | NA | NA | NA | NA |
| B: WM 60 s + real-trained V | NA | NA | NA | NA |
| C: WM 90 s + real-trained V | NA | NA | NA | NA |
| D: no-WM V + current issued commands | NA | NA | NA | NA |
| E: FIT arm/time means | NA | NA | NA | NA |

NA = not evaluated, never zero. Reserved200 real cells exported AFTER adapter audit.
PASS remains: B or C beats EACH A,D,E on sign AND MAE with all six positive
95% lower bounds; 400 arm pairs, 10,000 whole-world bootstrap draws. No tuning.

## Completed: the original telemetry blocker is resolved

460 FIT episodes,82 worlds: WMDATA160 + WMDATA2 300;21 states/episode =9660.
111 features: sensed lane/segment aggregates, known CAV count, time remaining,
cumulative/trailing60 boundary FLOW; no source queue/request/demand feature.
CAV nearest same-lane front/rear <=100m + existing mainline/ramp RSU zones;
nominal5m length only; neighbour acceleration outside RSU unobserved (zero).
Internal-junction lane=-1 contributes only to known CAV count, no invented lane.
Future poisoning/truncation and existing projection-helper parity tests PASS.
Full real LABELS include union-cohort births, carried exit speeds, post-step
source waiting; label-only independent request RNG never enters features.
All1380 objective120/300/600 checks pass: max discrepancy7.87e-6veh-s;
source discrepancy exactly0. No interpolation or on-road-only target substitute.
Frozen HGB150/15/.05/L2=10 V and command-only D fit once in9.89 CPU seconds;
FIT-only E fitted too; equal world/arm/time weighting; no selection or test use.
Model: `_out/wmvalue_frozen_value6f334469.joblib`; hash in `frozen_fit.json`.

## Adapter and registered inference execution

MAIN18:52 authorizes missing matched A to900 and B/C60/90. Cached A was
keys-only coverage, not cost/CRN certification; aggregate-only exit-clock mismatch
cannot be repaired. Preserve it; new matched A/B/C share trajectories and CRNs.
Native121frame60s history, PI/CIRCLES prefix-memory replay, entrant controls,
real/imagined sensing, boundary-flow splice, exit/source clocks CPU audited.
40 CPU tests + actual frozen WM CPU repeat +134source cross-host parity PASS.
Actual WM90s CPU synthetic run: both60/90 terminal paths and all five arms PASS.
All120 reserved real objective clocks reconcile; arrays frozen SHAa61d1846.
No late true source backlog in WM initialization: warmup queue + prefix-flow prior.

**Recommendation:** complete B/C/D/E first; neither short method clears D/E => FAIL.
Durable short dispatcher resumed PID27340; _out/wmvalue_short_v1/RESUME2.json.
Old schedulers28924/27688 stopped by exact ownership; live workers preserved.
CPU caps gpuA4/gpuR8 include old workers; GPUs only absent peer compute PIDs.
Exact90vs120s checkpoint parity PASS on CPU; immutable short_v1 on both hosts.
Complete320K8-prefix tasks trigger once-only400pair gate/report/verdict commit.
Only D/E survivors require long A; V/D/E frozen, no efficacy verdict yet.
No SUMO, refit, eval/prohibited-path access, peer signals, policy run or push.
