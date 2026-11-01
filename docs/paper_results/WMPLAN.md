# S-WMPLAN — world-model-guided cooperative planner

## Plain-language status
Preparation is in progress. No S-WMPLAN live episode has run and no claim of
beating MAPPO is supported yet. The owner-approved question is whether the
world model can choose cooperative actions online better than MAPPO alone.
The exact199200sim-s plan is registered. Complete full-prefix C_G twins require
fixed2950sim-s credit per world-arm: at most one early group can execute, and
none aftert600 fits. This is explicitly a budget-limited planner comparison;
raw proposals, executions and resource-forced O0 decisions will all be reported.

## Recovered authority and evidence
- Branch `c4/wmplan`, clean initial HEAD `df52de61`, inherited from S-WMTEACH.
- Shared registration: `S-WMPLAN BUDGET APPROVED`, 2026-11-01 13:45;
  maximum 200,000 actual simulation seconds.
- Same fresh DEV panel as S-DISTILL: 984000–984023, 24 worlds, two per cell;
  canonical panel SHA256
  `72776ff122956f021d1caf2db9622674b0cde2a74af8bd9f20c3de5ba0a383a6`.
- Reuse completed S-DISTILL MAPPO episodes from gpuA
  `/root/autodl-tmp/distill20261101_v1`; do not dispatch MAPPO again.
- Predictor/executor contract: `tools/coopplan`, S-WMTEACH's
  `tools/wmteach/runtime.py` and `marginal.py`; K8 complete paired futures,
  H300, decisions every 60 seconds at t300..960, inherited O0..O4 and safety.
- S-WMTEACH FULL-v4 quality is complete on 12 prospectively chosen exact
  states: H300 helpful agreement 4/8 and helpful exact chosen saving
  +1020.461280948498 veh-s/state; all-state chosen saving +680.307521.
- Matching FULL-v5 quality is already running in the peer-owned immutable root
  `/root/autodl-tmp/wmteach_v5quality20261101_v2`. Read completed evidence only;
  never duplicate its forecasts or control its process.
- Before any live outcome, select FULL-v5 source `b2df7458` only if its same-12
  H300 helpful-state agreement AND all-12 chosen saving are at least FULL-v4;
  otherwise select frozen FULL-v4 `22cc32f6`. Preserve both comparison results.
- Exact live budget, immutable sources, baseline hashes, model choice, CPU and
  prefix-integrity gates must be registered before dispatch. No live dispatch
  is armed by this preparation note.

## Required final evidence
Paired J600/J900 savings (MAPPO minus planner, source-inclusive team delay),
whole-world bootstrap intervals, executed option counts and budget refusals,
complete exact C_G quartets for executed groups, collisions/TTC/hard braking,
and matched-panel comparison to S-DISTILL and S-WMTEACH when available.
S-COOPPLAN's existing oracle panels differ; label those comparisons unpaired.

The inherited exact-quality adapter is roster-only with frozen FIT population,
sampled FIT source priors, and cold motion/controller memory. Its quality is not
qualification of a richer history-conditioned observation adapter or calibration
of empirical K8 win probabilities. Preserve that limitation in live reporting.

<!-- WMPLAN-PROGRESS:BEGIN -->
## Current continuation status
- Fresh-session recovery verified local continuation PID21412 and stopped only that
  owned process before revising its obsolete launch order. Its claim and log are retained.
- Both remote S-WMPLAN roots are absent: no live dispatch or SUMO has occurred.
  Existing FULL-v5 quality is alive with75/96 complete futures; it is not duplicated.
- Owner14:48 amendment is being implemented: gpuA+gpuR inference, all24 expected
  worlds and their primary report before the strict arm. Registered199200sim-s upper retained.
- gpuR has16GiB free on its root filesystem, but only1.8GiB on `/root/autodl-tmp`;
  inference outputs must use the root filesystem. No peer process is modified.
<!-- WMPLAN-PROGRESS:END -->
