# C4 v5 evaluation plan — 2026-10-15

## Readiness: smoke complete; full grid unlocked

Branch: `c4/evalv5`. The bounded laptop smoke finished on 2026-10-15: all 6
episodes completed with **0 collisions, 0 teleports, 0 dropped-CAV steps**.
Both earlier scope blockers were resolved: `mappo_ft26_it106.pt` was copied to
the worktree root, and `runs/active/evalv5_smoke/` was approved as the
in-worktree output root. The capacity calculation below now uses **measured**
per-episode wall times instead of conditional estimates.

Execution record: the previous session started native (2/2 complete) and the
Dreamer block, whose p0.4 episode completed while p0.2 was interrupted
(`manifest.status=running`, no summary); the leftover attempt was moved to
`s964199_m2400_p0200.incomplete-1` by the collector and re-run as attempt 2.
Total episode starts consumed: 7 (including the failed attempt); completed
episodes: 6, exactly the approved grid. No other simulations were run.

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
  the default loader.
- A full `c4_mappo_trainer/2` checkpoint is different: the existing opt-in
   `--policy-loader wms.control.mappo.eval_bridge:load_policy_fn` accepts it.
   `eval_v5.sh` does not expose that option. For the mixed-controller launcher,
   use the trainer's `policy_latest.pt`/`policy_XXXXX.pt` deployment export.
   Do not rename a full trainer checkpoint and assume it becomes an export.
- The actual `mappo_ft26_it106.pt` was verified this session: format
   `c4_dreamer/3`, `trainer="mappo"`, stored acceleration bounds
   **[-3.0, 2.6] m/s²**, lane head enabled; it loads through the default
   `make_policy` dispatch (`PolicyExecutor`). No filename-based dispatch needed.
- Existing textbook env selection forces CAVs to `cav_typical`. Direct command
  mapping filters present CAVs; there is no position/control-zone filter.
  Lane-head execution uses laneChangeMode **513** and restores the original
  mode at handback. Offline tests verify CAV-only mapping and mode changes.
- No production code or launcher defaults were changed: no verified controller
  blocker justified a behavior change.

Local Python 3.14, Python 3.13, and base Conda lacked torch. A workspace-local
CPU environment was prepared at `runs/active/evalv5_venv/`; torch, numpy and
TraCI (via `SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo\`) are available
there and were used for the smoke. Its Python executable on Windows is
`runs/active/evalv5_venv/Scripts/python.exe`.

## Smoke executed (2026-10-15)

Three sequential collector invocations under
`runs/active/evalv5_venv/Scripts/python.exe`, each with `--workers 2`:

```text
python -m traffic_shepherd.c4.collect --policy <controller> --population textbook --seeds 964199 --demands 2400 --penetrations 0.2,0.4 --warmup 60 --horizon 120 --workers 2 --no-physics-log --out runs/active/evalv5_smoke/<label>
```

with `<controller>`/`<label>` = `native`/`native`, `./_policy_v5_u50.pt`/`dreamer`
(prior session) plus the `.\_policy_v5_u50.pt` re-run, and
`.\mappo_ft26_it106.pt`/`mappo`.

All six episodes report `manifest.status=complete`,
`manifest.scenario.driver_population=textbook`, `summary.lane_change_mode_policy=513`,
`collisions=0`, `teleports=0`, `dropped_cav_steps=0`. TTC evaluation stayed on.

### Measured per-episode wall times (summary.wall_seconds)

| Controller | p=0.2 (s) | p=0.4 (s) | max (s) |
|---|---:|---:|---:|
| native | 61.09 | 61.61 | 61.61 |
| `_policy_v5_u50.pt` (Dreamer) | 34.28 | 27.58 | 34.28 |
| `mappo_ft26_it106.pt` (legacy MAPPO) | 21.21 | 23.77 | 23.77 |

Collector per-job `INDEX.json.run_records[].wall_seconds` (includes loading and
recording overhead; the Dreamer p0.4 value comes from `collect_log.jsonl`
because the interrupted session never rewrote `INDEX.json` for that block):

- native: 68.91 / 69.43 s
- Dreamer: 41.37 (p0.2, attempt 2) / 34.78 (p0.4)
- MAPPO: 28.76 / 31.51 s

### Controller diagnostics from `episode.npz` (control times >= 60 s, present CAVs)

Decoded executor commands (after policy safety, before SUMO clipping);
HDV slots verified clean (no finite requested acceleration, no nonnegative
lane request) in all four policy episodes:

- Dreamer p0.2: accel **[-1.4368, -0.5239]** m/s² (n=1589); lane commands
  {issued 146, completed 128, not_completed 18}.
- Dreamer p0.4: accel **[-0.6340, -0.5324]** m/s² (n=2973); lane commands
  {issued 374, completed 349, not_completed 25}.
- MAPPO p0.2: accel **[-2.7424, 2.5546]** m/s² (n=1498); lane commands
  {issued 9, completed 9, deferred_edge_end 1}.
- MAPPO p0.4: accel **[-0.5989, 2.5499]** m/s² (n=3384); lane commands
  {issued 20, completed 16, not_completed 4, deferred_edge_end 2}.

Both stay inside their stored decoding bounds (Dreamer [-3.0, 1.2];
MAPPO [-3.0, 2.6]).

### Smoke report

`scripts/boxops/eval_v5_report.py runs/active/evalv5_smoke/native runs/active/evalv5_smoke/dreamer runs/active/evalv5_smoke/mappo --out runs/active/evalv5_smoke/comparison`
accepted 6 episodes (0 duplicates; the `.incomplete-1` leftover was pruned).
Note: the interrupted first session recorded Dreamer p0.4's policy as
`./_policy_v5_u50.pt` and the re-run used `.\_policy_v5_u50.pt`, so the report
splits Dreamer into two identity tags for the same checkpoint. For future
invocations pass the identical path string on every box to keep one identity.

## Bounded laptop smoke specification (executed; do not repeat)

The approved product grid, fully consumed on 2026-10-15:

- Controllers: `native`, `./_policy_v5_u50.pt`, `./mappo_ft26_it106.pt`.
- Seed **964199 only**, demand **2400**, penetrations **0.2, 0.4**.
- Population **textbook**, warmup **60 s**, controlled horizon **120 s**.
- At most **2 workers concurrently**. Three sequential collector invocations,
  each running its two penetration jobs, consume the entire 6-episode budget.
- Retain normal TTC evaluation. Use `--no-physics-log` to match the launcher;
  `episode.npz` and action timelines are still written.

Template for each invocation (as executed; see the smoke record above):

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

### Capacity calculation populated from the smoke (measured)

The smoke schedule (60+120) scales both phases by five to the full schedule
(300+600). Using the slower collector wall time per controller
(`full_episode_seconds[c] = 5 * max(smoke_collector_wall_seconds[c])`):

```text
native            5 * 69.43 = 347.15 s
dreamer (u50)     5 * 41.37 = 206.85 s
mappo (legacy)    5 * 31.51 = 157.55 s
```

The smoke measured only native, Dreamer and legacy MAPPO at one demand, with
2 workers on 22 laptop CPUs; baseline costs (damping/PI/constant), congested
late-window traffic and 46-worker box contention remain unmeasured, so keep a
contention/traffic allowance `f >= 1` and evaluate:

```text
box_seconds ≈ 13 * sum(full_episode_seconds[c] for c in seven_controllers) * f
```

For the four unmeasured controllers (damping, PI, both maxaccel), use the
slower measured learned controller (Dreamer, 206.85 s) provisionally:
`sum = 347.15 + 206.85 + 157.55 + 4*206.85 = 1538.95 s`, so
`box_seconds ≈ 13 * 1538.95 * f = 20006 * f`.
Reserve **20% of eight hours** for startup, tail jobs, report generation and
transfer: usable compute is 23,040 s. The measured projection therefore fits
the reserve-adjusted eight-hour window for **f <= 1.15** (f <= 1.44 without
reserve). The uniform per-controller threshold of 253.19 s (23,040/91) is met
by Dreamer/MAPPO estimates but exceeded by native's 347.15 s: the sum-based
projection above is the operative check, and native's slower waves are offset
by faster policy waves. Replace provisional estimates with box telemetry as
soon as the first controller blocks report.

Budget thresholds for reference (not measurements): a uniform effective
full-episode time of 253.19 s (23,040/91) with reserve, 316.48 s without;
equivalently `t × f <= 50.64 s` on a fivefold-scaled smoke collector time.

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
- Real SUMO smoke executed this session: 6/6 episodes complete, 0 collisions,
  0 teleports, 0 dropped-CAV steps; per-episode wall times, acceleration ranges
  and lane-command counts recorded above.
- Smoke comparison report generated at
  `runs/active/evalv5_smoke/comparison/report.md` (6 accepted episodes).

Record the delivery commit with `git log -1 --oneline` after committing this plan.
