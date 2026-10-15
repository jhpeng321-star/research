# C4 v5 evaluation plan — 2026-10-15

## Readiness: partial; SUMO smoke still required

Branch: `c4/evalv5`. No simulator episodes were started during this execution
(0 of the maximum 6 consumed). No collision result, observed SUMO acceleration
range, lane-command count, or episode wall-time measurement is claimed.

Two scope blockers prevent the requested smoke:

1. The specified `mappo_ft26_it106.pt` is outside this worktree; the execution
   policy forbids reading outside it. No in-worktree copy was found. Copy it to
   the worktree, preferably `mappo_ft26_it106.pt` at the root.
2. The required output root `E:/agent-work/tmp/evalv5_smoke/` is outside the
   workspace. Authorize that specific output exception, or approve an in-worktree
   output root such as `runs/active/evalv5_smoke/` instead. The report CLI currently
   accepts only in-worktree paths, so external results would need an authorized
   copy into the worktree for analysis.

The missing measurements mean the capacity calculation below is **conditional**,
not a measured prediction that the full grid will finish in eight hours.

## Verified offline

- The actual `_policy_v5_u50.pt` loads through
  `traffic_shepherd.c4.baselines.make_policy`, the collector's default dispatch.
  Format: `c4_dreamer/3`; executor: `PolicyExecutor`; lane head: enabled;
  stored acceleration decoding bounds: **[-3.0, 1.2] m/s²**.
- On a synthetic isolated-CAV observation, the unchanged checkpoint decoded
  **-0.6608538032 m/s²** and requested WM lane **1**. The HDV acceleration was
  NaN (uncommanded). This is a single static observation, not an episode range
  or a SUMO lane command/completion count.
- MAPPO deployment exports use the same `c4_dreamer/3` schema, tagged
  `trainer="mappo"`; they do not require filename-based dispatch. Synthetic
  Dreamer and MAPPO exports load with their own bounds and lane heads using
  the default loader. The actual MAPPO file's format is still unverified.
- A full `c4_mappo_trainer/2` checkpoint is different: the existing opt-in
  `--policy-loader wms.control.mappo.eval_bridge:load_policy_fn` accepts it.
  `eval_v5.sh` does not expose that option. For the mixed-controller launcher,
  use the trainer's `policy_latest.pt`/`policy_XXXXX.pt` deployment export.
  Do not rename a full trainer checkpoint and assume it becomes an export.
- Existing textbook env selection forces CAVs to `cav_typical`. Direct command
  mapping filters present CAVs; there is no position/control-zone filter.
  Lane-head execution uses laneChangeMode **513** and restores the original
  mode at handback. Offline tests verify CAV-only mapping and mode changes.
- No production code or launcher defaults were changed: no verified controller
  blocker justified a behavior change.

Local Python 3.14, Python 3.13, and base Conda lacked torch. A workspace-local
CPU environment was prepared at `runs/active/evalv5_venv/`; torch is available
there. Its Python executable on Windows is
`runs/active/evalv5_venv/Scripts/python.exe`. Install the repo's SUMO runtime
dependencies in that environment before the eventual smoke (in particular TraCI).

## Remaining bounded laptop smoke

Once both path blockers are resolved, run exactly the product of:

- Controllers: `native`, `./_policy_v5_u50.pt`, `./mappo_ft26_it106.pt`.
- Seed **964199 only**, demand **2400**, penetrations **0.2, 0.4**.
- Population **textbook**, warmup **60 s**, controlled horizon **120 s**.
- At most **2 workers concurrently**. Three sequential collector invocations,
  each running its two penetration jobs, consume the entire 6-episode budget.
- Retain normal TTC evaluation. Use `--no-physics-log` to match the launcher;
  `episode.npz` and action timelines are still written.

Template for each invocation (substitute approved paths; not run here):

```text
python -m traffic_shepherd.c4.collect --policy <controller> --population textbook --seeds 964199 --demands 2400 --penetrations 0.2,0.4 --warmup 60 --horizon 120 --workers 2 --no-physics-log --out <approved-smoke-root>/<controller-label>
```

Count every started episode against the budget, including failures. Do not
automatically retry failed episodes once six starts have been consumed. Keep
seed 964199 out of the main comparison grid.

For each episode require `manifest.status=complete`, a `summary.json`, and
`collisions=0`. Record teleports and dropped-CAV steps as well. Verify textbook
metadata in `manifest.scenario.driver_population`, the CAV vType in the scenario,
and `summary.lane_change_mode_policy=513` for the lane-head policies.

For Dreamer, report min/max finite `requested_acceleration_mps2` in
`episode.npz`, restricted to control times >=60 s and present CAVs at state
index `2 * control_index`. These are decoded executor commands after policy
safety but before SUMO-side clipping; do not confuse them with raw actor values
or measured physical accelerations. Also report the summary's complete
`lane_commands` dictionary (issued, completed, not_completed, deferred/refused)
and the count of nonnegative `lane_request_target` entries. Check that HDV
slots have no finite requested acceleration and no nonnegative lane request.

Retain both timings: `summary.wall_seconds` (environment runtime) and the
collector's per-job `INDEX.json.run_records[].wall_seconds` (includes loading
and recording overhead). Per-episode wall times and the compact smoke table
are generated by the report tool; collector overhead is not part of that table.

## Tonight's recommended controller list and grid

Use seven controllers for the main comparison:

```bash
export POPULATION=textbook
export RUN_ROOT="$REPO/runs/active/eval_v5"
export CONTROLLERS="native,damping,pi,maxaccel_lc513,maxaccel_sumolc,$REPO/_policy_v5_u50.pt,$REPO/mappo_v5_final.pt"
export DEMANDS=1900,2100,2400,2700
export PENETRATIONS=0.05,0.1,0.15,0.2,0.3,0.4,0.6
export WARMUP=300 HORIZON=600 WORKERS=46
```

`mappo_v5_final.pt` denotes the remote **textbook fine-tuned deployment export**.
Freeze and record its checkpoint hash before launching. The legacy-population
`mappo_ft26_it106.pt` verifies mechanics only; if evaluated for comparison, label
it as a separate legacy-trained comparator rather than calling it v5 textbook.
Likewise freeze the Dreamer checkpoint and its hash. Do not mix checkpoint
versions in a resumed output root. Clear inherited `C4_POLICY_STOCHASTIC` and
`C4_POLICY_ACCEL_NOISE` settings for deterministic deployment.

Shard **seeds**, not controllers, to retain native pairing on each box:

- Box A: `SEEDS=964000-964020` (21 seeds).
- Box B: `SEEDS=964021-964041` (21 seeds).
- Each box runs `bash scripts/boxops/eval_v5.sh <distinct-run-id>` with the
  same controller paths/order and grid. These are remote-run recommendations,
  not authorization for additional laptop simulations.
- The launcher runs controller blocks sequentially, respects the product grid,
  resumes complete jobs, and relays whitelisted summaries after evaluation.
  No launcher or remote transfer was executed in this work session.

Total: **7 × 4 × 7 × 42 = 8,232 episodes**, or 4,116 per box.
Per controller per box: 588 episodes / 46 workers = **13 scheduling waves**.
The seven sequential controller blocks require **91 waves per box**.

### Capacity calculation to populate from smoke

The smoke schedule (60+120) scales both phases by five to the full schedule
(300+600). Let `t_c` be the slower of the two collector wall times for controller
`c`; initially use `5 * t_c` as its full-episode estimate. This is deliberately
conservative for one-time loading cost, but can still understate congested
late-window traffic and 46-worker contention. The smoke measures only native,
Dreamer and legacy MAPPO at one demand; baseline costs and box throughput remain
unmeasured.

Use a contention/traffic allowance `f >= 1` and evaluate:

```text
box_seconds ≈ 13 * sum(full_episode_seconds[c] for c in seven_controllers) * f
full_episode_seconds[c] ≈ 5 * max(smoke_collector_wall_seconds[c])
```

For unmeasured damping/PI/constant controllers, use the slower measured learned
controller as a provisional estimate, then replace it with box telemetry.
Reserve **20% of eight hours** for startup, tail jobs, report generation and
transfer: usable compute is 23,040 s. A uniform effective full-episode time
must therefore be **<=253.19 s** (23,040/91); equivalently, a fivefold-scaled
smoke time must satisfy **t × f <=50.64 s**. Without reserve the threshold is
316.48 s/full episode. These are budget thresholds, not measurements.

If initial box throughput cannot meet the reserve-adjusted threshold, prioritize
the same seven controllers and all 28 demand/penetration cells with **20 seeds**:
964000-964009 on A, 964010-964019 on B. That is 3,920 total episodes,
7 waves/controller/box, 49 waves/box, and a 470.20 s effective full-episode
threshold at the same reserve. Complete paired seed blocks before extending
to 42 seeds. Do not truncate an in-flight SUMO episode to force the deadline.

## Report generation and interpretation

From complete local episode trees (including manifests):

```text
python scripts/boxops/eval_v5_report.py runs/active/eval_v5/box_a runs/active/eval_v5/box_b --out runs/active/eval_v5/comparison
```

The launcher's summary-only relay omits manifests. For those trees, append
`--driver-population textbook` after verifying each run's `COLLECT_SUMMARY.json`
and `RUN_SPEC.json`. This explicitly records the caller's population assertion;
it does not verify metadata that was not relayed. Use identical controller
indices and checkpoint basenames on both boxes: without manifests, direct-policy
identities are `controller_NNN/<policy-tag>`.

Outputs: `report.md` and `report.json`, including the smoke table, per-episode
wall times, cell/group means and 95% CIs, and matched treatment-minus-native
differences. Seen penetrations are 0.05/0.1/0.2/0.4; unseen are 0.15/0.3/0.6.
Light is 1900; heavy is 2400/2700; 2100 remains separate. Grouped observations
are averaged within seed before computing seed-level Student-t intervals.
For the single smoke seed, CIs are unavailable, not zero-width evidence.

System-seconds and source-wait are episode-wide ledger metrics; speed deficit,
arrivals and braking are controlled-window metrics. Window source-wait and
episode arrivals are also included explicitly. Braking is vehicle-seconds
below -3 m/s², not an event count. TTC is the mean of episode minima.
Negative paired differences favor treatment for costs; positive favors treatment
for arrivals and minimum TTC. Missing values stay unavailable. Diagnostics
identify incomplete/malformed inputs and conflicting duplicates; mixed schedules
and driver populations are rejected.

## Verification completed

- 46 focused tests passed: report statistics/CLI and offline controller contracts.
- Ruff passed on all three new Python files.
- Actual Dreamer checkpoint loaded and evaluated on one synthetic observation.
- Synthetic CLI generation produced both Markdown and strict JSON reports.
- Real SUMO smoke/report-on-smoke and measured throughput remain blocked as above.

Record the delivery commit with `git log -1 --oneline` after committing this plan.
