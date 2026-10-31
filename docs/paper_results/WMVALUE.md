# S-WMVALUE — INCOMPLETE after causal reconstruction + real fitting (2026-10-31)

**The MAIN18:12 observation amendment is implemented; the A–E test is not run.**
The WM-trained cooperative policy remains the target. Original freeze12685912;
amendment03dd03e0; reconstruction6f334469; frozen real fit a0acedb6.

## Registered comparison: remaining fleet+source delay to absolute t900, veh-s

| Method | Sign accuracy | Spearman | MAE (veh-s) | World-cluster CI |
| --- | --- | --- | --- | --- |
| A: WM2 b8797242 + mix to endpoint | NA | NA | NA | NA |
| B: WM 60 s + real-trained V | NA | NA | NA | NA |
| C: WM 90 s + real-trained V | NA | NA | NA | NA |
| D: no-WM V + current issued commands | NA | NA | NA | NA |
| E: FIT arm/time means | NA | NA | NA | NA |

NA = not evaluated, never zero. No reserved969432..969439 real outcomes opened.
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

## Remaining mechanical/authorization gate

Local WMCALIB cache:240/1600 future keys,28/200 complete arm/start cells.
All are t300 and only CTRL0/MAPPO/PI/CIRCLES; no native or late restart.
Even completion of peer jobs supplies at most32/200 cells:168 structurally
missing (8 native@300 +160 cells at360/420/480/540). SPEEDHEAD seeds differ.
Unconditional cumulative tails cannot represent later-state conditional A.
MAIN18:12 allows new inference only60/90s, so new long A inference needs an
explicit amendment or matching cached predictions. Coverage is not parity proof.
The WM restart/controller/entry adapter is also unfinished; B/C not launched.

**Recommendation:** supply matched A or authorize missing A continuations while
retaining this exact test; then bind/audit real-vs-imagined sensing, history,
entry/exit/cost/CRN and controller continuation before spending the reserved test.
Keep fitted V/D/E frozen. No PASS, policy coupling/run or GPU-hour claim yet.
24 CPU tests and ruff PASS. All exports foreground-complete; no pending task job.
Evidence: `tools/wmvalue/evidence/{frozen_fit,amended_status,a_coverage_final}.json`.
No SUMO, WM inference, evaluation-seed/prohibited-path access, peer signals or push.
