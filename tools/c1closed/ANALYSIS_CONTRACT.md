# S-C1CLOSED analysis/collection contract

`analyze.py` refuses a selected partial panel: all 480 unique model/plant/world/
future tasks, all seven clocks, immutable runtime closure and 200 paired recorded
labels must validate. Outputs are three CSVs, a PNG, <=60-line report and hashes.

Endpoint draw aggregation uses the initial HDV IDs surviving the recorded arm
and **all** model draws in that comparison. Paired curves/differences additionally
intersect WM and BB cohorts. Median AE within each draw, then median K8, then
median world; export surviving counts including zeros. No-ID errors are missing,
never zero. BB-id's PI curve uses its own paired cohort/ID contract; MAPPO absent.
All models' initial features derive from the same genuine observed history;
the legacy checkpoints retain their native short-context encoder, while WM2
uses its learned long-history path. This is not a matched-history-capacity or
matched-training-data architecture attribution. Original-ID entry exclusion is
enforced by `_entry_process`'s padding-only pool (slots >= initial k0).

Aggregate fields/queues/crossings use pointwise K8 medians. Throughput error is
also exported as cumulative-crossing error / horizon x3600 veh/h. J is the float64
sum of the four exported component integrals, avoiding an intermediate float32
road-sum ULP discrepancy; `J_rounding_correction` preserves its difference from
the inference writer's J. This is arithmetic normalization, not outcome fitting.
Common total-J median-order weights determine all component forecasts. Signed
component bias uses equal-world means so the components remain additive;
endpoint/aggregate absolute errors and paired error changes use world medians.

E2 sign is treatment-minus-native, negative improvement. Exactly-zero true
effects are excluded from direction denominators only. True pairwise ties are
excluded; predicted ties on a decisive true pair are not successes. Native-first
deterministic action ties; regret and ranking statistics are equal-world means.
Paired direction/ranking/regret changes are also computed within whole worlds.
Bootstrap is 2,000 whole-world draws/seed0, never tick/vehicle/controller sampling.
Aggregate 60/120/180/300 s clocks are registered secondary; retained 10/30/90 s
aggregate rows are additional diagnostics, not additional confirmatory tests.

`verify_analysis.py` tests attrition/missingness, ties, common median weights and
whole-world bootstrap against independently known examples. `verify_delivery.py`
checks population uniqueness, additive/paired arithmetic, hashes, finite CIs,
the report limit and PNG decoding/dimensions. Actual panel inputs also validate.

`watch.py --launch` starts one local detached collector/finalizer with a Windows
creation-time receipt and exclusive lock; no inference launch, PID signals or
new environment. It polls the exact remote dispatcher incarnation every5min.
On full success it syncs, analyzes, verifies and commits only generated outputs
plus an additive registration line. Existing staged work causes a review stop.
If the dispatcher fails/disappears, it writes `WATCH_BLOCKED.json`, preserves all
outputs and launches no replacement. A frozen-session resume must inspect
`WATCH_LAUNCH.json`/`WATCH_EVENTS.jsonl`/`WATCH_COMPLETE.json` before doing anything.

Commands: `python -B tools/c1closed/watch.py --status` (local collector);
`python -B tools/c1closed/operate.py status` (remote panel). Do not launch again.
An actual registered independent WM-fix PASS and a provenance-bound floor must
be supplied to a separately bound secondary extension; neither is fabricated by
the finalizer or used to replace the registered primary checkpoint.
