# S-WMVALUE — decisive-test inference RUNNING (2026-10-31)

**MAIN18:12/18:52 adapter gates passed; complete-only A–E test is pending inference.**
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
36 CPU tests + actual frozen WM CPU repeat +134source cross-host parity PASS.
Actual WM90s CPU synthetic run: both60/90 terminal paths and all five arms PASS.
All120 reserved real objective clocks reconcile; arrays frozen SHAa61d1846.
No late true source backlog in WM initialization: warmup queue + prefix-flow prior.

**Recommendation:** finish the immutable320task test before any policy coupling.
Durable local dispatcher PID28924; _out/wmvalue_inference_v1/LAUNCH.json.
gpuA host CPU<=4 nice19/single-thread lanes; gpuR GPU<=3 only absent peer compute.
MAIN19:52 CPU release: disjoint969439/40tasks, gpuR CPU<=8, assistPID27688.
Complete tasks automatically trigger frozen400pair test, report/verdict and commit.
Keep V/D/E frozen. No efficacy PASS/FAIL or policy-training authorization yet.
Evidence: tools/wmvalue/evidence/{cpu_runtime_audit,cross_host_cpu}.json.
No SUMO, refit, eval/prohibited-path access, peer signals, policy run or push.
