# S-WMPLAN — world-model-guided cooperative planner

## Plain-language status
Preparation is in progress. No S-WMPLAN live episode has run and no claim of
beating MAPPO is supported yet. The owner-approved question is whether the
world model can choose cooperative actions online better than MAPPO alone.
The exact 199,200 sim-s plan is registered. Complete full-prefix C_G twins require
fixed 2,950 sim-s credit per world-arm: at most one early group can execute, and
none after t600 fits. This is explicitly a budget-limited planner comparison;
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

## Owner14:48 speed amendment
The expected arm now runs all24 worlds first. Its complete paired efficacy,
safety and group-twin report is independently recomputed and committed before
the strict arm is released, regardless of the primary result. One WM service
uses gpuA and one uses gpuR, with the identical serial teacher-quality function
and byte-frozen model/FIT32 artifacts. Exclusive forecast claims and all SUMO
charges remain on gpuA. The gpuR service uses `/root/wmplan20261101_v1` on the
root filesystem; no inference output goes to its nearly-full autodl-tmp mount.

Local contract gate:29 tests PASS, including primary-only full-panel accounting,
rejection of incomplete/secondary jobs at primary reporting, exact choices,
full-prefix credit quotas and generated remote-script compilation. Host/mirror
gates and the final model comparison still precede live dispatch.

<!-- WMPLAN-PROGRESS:BEGIN -->
## Current continuation status
- Model fixed as FULL-v4; 0/48 episodes complete.
- Charged 2400; closed actual 0 sim-s; 2 open jobs. Registered upper199200/cap200000.
- Complete K8 forecast states 0; future tasks 5.
- Full paired efficacy and safety report waits for all24 pairs per arm. No claim of beating MAPPO is supported by progress counts.
<!-- WMPLAN-PROGRESS:END -->
