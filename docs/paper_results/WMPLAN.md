# S-WMPLAN — world-model-guided cooperative planner

## Plain-language status
The live evaluation is running, with the primary expected-saving arm first and
decision-time world-model inference on both gpuA and gpuR. No complete paired
24-world result is available yet, so no claim of beating MAPPO is supported.
The owner-approved question is whether the world model can choose cooperative
actions online better than MAPPO alone. FULL-v4 was selected before live outcomes:
helpful-state agreement4/8 vs FULL-v5 2/8, all12 exact chosen saving+680.308 vs
+172.704veh-s/state. These are teacher-quality diagnostics, not live efficacy.
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
- At intake, matching FULL-v5 quality was already running in the peer-owned immutable root
  `/root/autodl-tmp/wmteach_v5quality20261101_v2`. Read completed evidence only;
  never duplicate its forecasts or control its process.
- Before any live outcome, select FULL-v5 source `b2df7458` only if its same-12
  H300 helpful-state agreement AND all-12 chosen saving are at least FULL-v4;
  otherwise select frozen FULL-v4 `22cc32f6`. Preserve both comparison results.
- Exact live budget, immutable sources, baseline hashes, model choice, CPU and
  prefix-integrity gates passed and were registered before the one-use dispatch;
  see `wmplan/FINAL_FREEZE.json`, `CPU.json`, and `DISPATCH.json`.

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
full-prefix credit quotas and generated remote-script compilation. Host29 tests/
24dual guards and gpuR32FIT bootstrap also PASS; final model/source/budget freeze
was registered before live dispatch. The asset-path/package startup repairs used
identical bytes and the unchanged teacher runtime; original manifests and failed
gate evidence are retained in `wmplan/`.

<!-- WMPLAN-PROGRESS:BEGIN -->
## Current continuation status
- Model fixed as FULL-v4; 0/48 episodes complete.
- Charged 2400; closed actual 0 sim-s; 2 open jobs. Registered upper199200/cap200000.
- Complete K8 forecast states 0; future tasks 11.
- Full paired efficacy and safety report waits for all24 pairs per arm. No claim of beating MAPPO is supported by progress counts.
<!-- WMPLAN-PROGRESS:END -->
