# S-SETUPAUDIT2 execution receipt

Branch c4/distill; first plan commit eff5230f; scoring implementation ce6a32d3.
No optimizer update/training or simulator call is present in panel2.py.
gpuA was not used; no push or destructive Git operation was issued.

## Inputs and sampling

Named remote root: gpuR:/root/autodl-tmp/c4/setupaudit20261030/fix.
WM2 b8797242, MAPPO aecd7d70, candidate update 190 de235a78.
The candidate only supplies the exact-zero CTRL0 residual architecture/configuration
and the saved TRAIN sampler; candidate A itself is not a scored arm here.
Restore the same saved sampler and focus-cell fraction as S-SETUPAUDIT, then draw
24 consecutive starts (no outcome selection). The first two reproduce the old
audit's starts. Native/genrand TRAIN data registry overrides are:
`WMS_DATA_V5_NATIVE_Q=/root/autodl-tmp/c4/v5/data/v5_native_q` and
`WMS_DATA_GENRAND_PLAIN_Q=/root/autodl-tmp/c4/v5/data/genrand_plain_q`.
`inspect_inputs.py --candidate ../policy_000190.pt --export ../starts24.pt --n-starts 24`.
The existing metadata/path guard rejects prohibited tokens, 977xxx and 964000..964199.
No real evaluation-world episode files, driver truth or assignment files were opened.

## Controller and scoring checks

`remote2.sh` preserves the prior audit's exact-execution/leader-rule flags and the
same qualified forward shortcuts. Primary command:
`bash remote2.sh panel2.py --out panel2_complete.json --device cuda`.
This executes K8 keyed futures at 600 s and reads 90/120/300/600 s prefixes.
24 distinct worlds, six closed-loop actor/intervention rows; future-major order.
No group/solo rows are needed for the physical MAPPO-CTRL0 J contrast; this receipt
does not claim to reproduce the weighted group/solo suffix-q10 actor credit.
The horizons share one 600 s stochastic plan and entrant padding. Thus prefixes
need not numerically repeat an independent 300 s plan with different slot geometry.
Every future is durably written with metadata, input/source hashes, components,
30 s time bins, controller requests, births and terminal queues.

CPU: 21 cases in test_setupaudit2.py + test_loop14c.py passed (2.40 s);
the additional 10 existing test_native_lane.py cases passed; two standalone
world-cluster/bootstrap cases passed locally (0.38 s). Ruff passed.
Real-checkpoint 2 s / K8 CPU scorer agrees with the original aligned adapter
on CTRL0, MAPPO and native arrays (all eight futures, atol 5e-5 veh-s;
CTRL0/MAPPO maximum observed discrepancy 1.52587890625e-5 veh-s).
`audit2_probe_K8.json` and `audit2_probe_original_adapter.json` retain those checks.
The CPU native-cap regression also prevents an actor-only cap from leaking into
the native reference. Source/birth freezing and exact due-time observer are tested.

## Owned GPU processes

Initial idle nvidia-smi: 2026-10-30 14:46:41; 1 MiB/24564 MiB, no compute PID.
Idle checked again before launches. All GPU scoring uses per-process fraction .85.
Owned preflight PID 432401 failed before scoring (CPU index on CUDA; repaired).
Owned 432480 and 432925 were terminated to install sliced callbacks, the frozen
admission row and metric-boundary telemetry; their completed receipts remain remote.
Owned 433243 was terminated to add explicit GC of the entry-observer cycle.
Primary scorer: owned PID **433299**, output panel2_complete.json/log.
Owned serial waiter **434647** validates primary completeness, waits for an idle
GPU, then execs the K1 strategic-merge repair scorer under its same PID.
The original bridge files in fix/src are untouched by repair staging; isolated
repair modules are loaded only by strategic_test2.py / strategic2.py.
Only the above owned PIDs were signalled; no peer process was killed.
Partial preflight values are not pooled into the primary panel.

## Offline analysis

`python tools/setupaudit/summarize2.py tools/setupaudit/evidence/audit2_panel.json
--out tools/setupaudit/evidence/audit2_summary.json
--csv tools/setupaudit/evidence/audit2_per_start.csv`
(one shell line). Requires all 24 x K8 rows, checks partitions and frozen queues,
and uses a shared 5,000-draw WORLD-cluster plan, seed 20261030. Futures are
averaged inside each world; they are never treated as independent worlds.

Repair verification: 37 CPU cases passed together (1.58 s), including the seven
new strategic-merge cases and existing bridge/native/lane tests; two standalone
bootstrap cases also passed. The full-WM request-priority regression ensures a
fallback cannot overwrite an explicit policy crossing: delegation is decided
after the callback, not in the pre-forward native-lane mask. Final real-checkpoint
2 s CPU repair probe PID 434850 completed (earlier plumbing probe 434622 retained).
K1 fix inference is prospective on all 24 starts and is diagnostic;
it is not pooled with K8 futures. No real evaluation or training is requested.
