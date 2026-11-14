# Parallel execution of the registered damping campaign

The owner requested greater parallelism on 2026-09-05. The three independent
optimizer seeds now run concurrently. Scientific configuration, seed sets,
checkpoint schedule and total simulation budget are unchanged.

## Current process ownership

Observe tool session **96488**, running `tools/parallel_damping_campaign.py`.
Its coordinator PID was **16952** at verification. Authoritative status is in
`runs/rllib/native_damping_campaign_v1/parallel_execution/state.json` and the
three `optimizer_71.json`, `optimizer_72.json`, `optimizer_73.json` files beside
it. Verify their PIDs against live processes; files alone do not prove liveness.

- Optimizer 71 continues in the canonical campaign root. Existing trainer
  PID 30172 was preserved and adopted while working toward episode 62.
- Optimizer 72 runs in `parallel_workers/optimizer_72/` under that root.
  Initial trainer PID was 5600, through episode 20.
- Optimizer 73 runs in `parallel_workers/optimizer_73/` under that root.
  Initial trainer PID was 33112, through episode 20.

The previous serial coordinator PID 32444 was intentionally stopped **without
stopping trainer 30172**. Its old state and session 41027 are historical;
do not restart it. Session 71265 is also terminal. Transition evidence is in
`parallel_execution/transition.json`. New worker stdout/stderr goes directly
to files in `parallel_execution/logs/`, without an undrained subprocess pipe.

## Isolation and evidence integrity

Each lineage uses the unchanged original trainer and its own exclusive output
directory. Episodes within a lineage remain sequential. Every seed still has
exactly the registered 20/62/100 checkpoint schedule and 100-episode endpoint.
There are at most three concurrent lineage trainers.

Worker initialization produced exactly the canonical manifest digest:
`87bd7fb663fa35350a1c6080c44de77909dfdaa38a2308b1b2a59f814f4d008a`.
The 32 completed cached rollouts (16 training native, 8 development native,
8 analytical) were copied as 64 JSON/NPZ files per worker with byte verification.
Seed, scenario identity, population and completion checks precede copying.
They are reused observations, not additional simulated data. Missing or
conflicting cache files fail preparation. Acceptance cases remain ungenerated.

The wrapper and its registration are recorded in
`parallel_execution/manifest.json` and `source_snapshot.py`. An earlier
preparation-only registration is archived as `preflight_v1_*`; it generated
zero parallel simulation and was superseded to strengthen cache/adoption checks
before launch. Do not edit the active wrapper or frozen scientific sources
while it is running.

The old SUMO launch-lock helper has an existing append-position weakness;
concurrent launches must not be described as proven race-free. Initial worker
launches are staggered, and the unchanged launch retry logic remains in use.
Each live worker was verified with a separate SUMO process and remote port.
Incomplete rollouts retain the original no-silent-repeat rule.

After all workers finish successfully, the wrapper takes the canonical lock
and copies only the completed 72/73 lineage and development folders into the
canonical campaign. Different existing bytes are rejected. Raw summaries and
checkpoints retain their hashes. It rebuilds selection from all nine records
and stores consolidation hashes. The candidate freezer already resolves
checkpoint paths to the canonical directory and still performs its full
source, trajectory, identity, action and acceptance-registration checks.
No parallel worker can accept a checkpoint or launch wall/predictor work.

On worker failure, no further stages are launched after the failure is observed;
already running stages are allowed to finish. Failure details are retained.
Do not start a duplicate coordinator or rerun an incomplete rollout after an
observation timeout. Inspect the live state and reconcile any actual failure.

## Capacity, estimate and readiness

The machine has 22 logical processors and approximately 32 GB RAM. Before
parallel launch, a process-level sample showed roughly one CPU core of combined
Python/SUMO work, with about 14 GB RAM free and 49 GB disk free. A single
processor-load sample elsewhere differed; these are momentary observations,
not a sustained-utilization benchmark.

At the forecast snapshot, optimizer 71 had completed 27 episodes. Its 23 new
completed training rollouts averaged 219.6 seconds (median 163.9, range
122.7–826.1). Remaining serial training was approximately 16.7 hours at that
mean, plus 5.3 hours for 64 remaining development policy rollouts at the first
evaluation's mean. This excludes unusual slowdown and small optimizer/I/O costs.

With three lineages concurrent, the longest remaining lineage has 100 episodes
plus 24 development rollouts: approximately 8.1 hours at those same per-rollout
rates. **8–12 hours is an initial planning estimate for completing this campaign,
not a measured speedup or a promise of an accepted controller.** CPU contention,
thermal limits and changing traffic behavior can alter throughput; compare the
new completed rollout times before tightening that estimate.

The implementation exists, but there are zero accepted checkpoints. The first
of nine development assessments failed; see the episode-20 results document.
Completion requires an eligible selected checkpoint and the separate 32-case,
three-arm untouched acceptance evaluation (96 rollouts). If the campaign has
no eligible checkpoint, more compute alone is not a completed damping result:
a concrete diagnosis and versioned successor are required. Parallel scheduling
reduces elapsed time to that scientific decision, not the standard of evidence.

Focused validation: 22 new parallel-orchestration tests passed, including
cache corruption, assigned-stage order, no repeat after completion, failure
handling, adoption identity and incomplete-consolidation rejection. The five
existing serial-coordinator tests also passed; Ruff passed. These were synthetic
orchestration checks with no SUMO generation. Three live Python/SUMO pairs
were subsequently observed; no final multiworker completion is claimed yet.
