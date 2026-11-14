# J4 WallTokenError diagnosis — the 18 walls_pilot_chunk4 runs FEAT1 could not featurize

Branch `zcode/feat1j4fix` (from `zcode/feat1run4` ae7550c), 2026-10-01.
Scope: the 18 `walls_pilot_chunk4/*__w-*` arm runs absent from the FEAT1 v4
shard tree `D:/rd-staging/wmn1_feat/shards_v2/walls_pilot_chunk4` (read-only
reference; 19 absent arm dirs minus the 1 registered smoke run
`w8697m2700p40t240__w-mid-early`, `is_smoke=True`, never featurized).
Raw data root: `E:/research-dense/runs/campaign_walls_pilot_chunk4` (read-only).
Splits re-decided from the chunk-4 plan before any file was opened:
CALIBRATION = w8658m2600p10t360__w-lo-early, w8692m2600p40t300__w-lo-late;
TRAIN = the other 16. No HOLDOUT/chunk-5 run is involved (seeds 8657-8706
are below the sealed chunk-5 block 8707-8736).

Method (zero SUMO): per run, wall episodes from `wall_events.jsonl`; one
streaming pass over `run/physics_steps.jsonl` reading ONLY the mechanics
fields `time_s`, `vehicles` (keys + `lane_id`), `cav_ids`, `departed_ids`;
for each wall member the per-tick timeline was compared with the wall's
visible window `[request_s, terminal_s]`. No evaluate_summary /
suffix_summary / dispatcher_rows / insertions file was opened.

## Result: two failure mechanisms, no data bug

Every run has exactly one wall (`*:wall1`), two CAV members, and every
member is a designated CAV (`cav_ids`) at every tick it is in the network.
No genuine non-CAV member id exists in any of the 18 member lists.

### Mechanism A — road-end departure precedes the recorded abort (15 runs)

The wall driver records `safety_abort` (`reason=member_left_network`,
`original_reason=member_frame_missing`) 0.5-1.5 s AFTER a member has
physically left the network (detection latency). At the ticks in between
the wall is still visible (holding) while the member is already gone from
the physics row -> `walls_at` raises at a NON-terminal tick. The old
`corpus/j4_wall_fix.py` wrapper tolerated missing members only AT the
terminal tick, which is exactly why these 15 failed. Consistent with
amendment 2 (road-end relabel; both members last seen on the exit edge).

### Mechanism B — present-but-unrostered member in dense worlds (3 runs)

`w8669m2700p10t360__w-mid-early`, `w8669m2700p10t360__w-mid-late`
(terminal = `released` 620.0 / 560.0) and `w8704m2700p40t480__w-mid-late`
(terminal = `released` 667.0): the members NEVER leave the network and are
CAV-reported throughout, but these dense runs exceed the featurizer's fixed
128-slot pool (`n_dropped_concurrent` = 18282 / 7287 / 6458), so from the
request tick onward the members repeatedly have no slot -> missing from
both `positions` and the roster at active ticks -> same WallTokenError.
(The run-level drop counter is pre-existing frozen-featurizer behaviour,
recorded in shard `meta.n_dropped_slot_steps`; only these 3 runs drop a
WALL MEMBER at active wall ticks.)

## Per-run findings

First failing tick = first tick where the wrapper's roster check fires
(physics time; featurizer ticks are `k = (t - t0_s) / 0.5` after the
world's `t0_s`).

### Mechanism A (15 runs) — absent member = left the network

| run | split | request | assembled | terminal (abort) | member | last in network | first absent tick |
|---|---|---|---|---|---|---|---|
| w8658m2600p10t360__w-lo-early | CALIBRATION | 360.0 | 380.0 | 487.0 | mainline.217 | 486.0 | 486.5 |
| | | | | | mainline.218 | 486.5 | 487.0 (= terminal) |
| w8664m2600p10t360__w-lo-early | TRAIN | 415.0 | 435.0 | 503.0 | mainline.226 | 502.0 | 502.5 |
| | | | | | mainline.205 | 508.0 | (present at abort; tolerated at terminal) |
| w8667m2700p10t420__w-lo-early | TRAIN | 428.0 | 465.0 | 520.0 | mainline.249 | 519.0 | 519.5 |
| | | | | | mainline.250 | 519.5 | 520.0 (= terminal) |
| w8670m2700p10t300__w-lo-late | TRAIN | 360.0 | 401.0 | 497.0 | mainline.229 | 496.0 | 496.5 |
| | | | | | mainline.232 | 496.5 | 497.0 (= terminal) |
| w8682m2400p20t240__w-lo-late | TRAIN | 300.0 | 316.0 | 399.0 | mainline.151 | 398.0 | 398.5 |
| | | | | | mainline.152 | 398.0 | 398.5 |
| w8684m2700p20t360__w-lo-early | TRAIN | 360.0 | 364.0 | 480.0 | mainline.211 | 479.0 | 479.5 |
| | | | | | mainline.212 | 486.0 | (present at abort) |
| w8686m2600p20t240__w-mid-late | TRAIN | 300.0 | 309.0 | 433.0 | mainline.185 | 432.0 | 432.5 |
| | | | | | mainline.186 | 432.5 | 433.0 (= terminal) |
| w8692m2600p40t300__w-lo-late | CALIBRATION | 360.0 | 374.0 | 462.0 | mainline.203 | 461.0 | 461.5 |
| | | | | | mainline.204 | 462.0 | (present at abort) |
| w8694m2600p40t240__w-lo-late | TRAIN | 300.0 | 313.0 | 488.0 | mainline.200 | 487.0 | 487.5 |
| | | | | | mainline.201 | 487.5 | 488.0 (= terminal) |
| w8697m2700p40t240__w-lo-late | TRAIN | 300.0 | 314.0 | 405.0 | mainline.163 | 404.0 | 404.5 |
| | | | | | mainline.164 | 404.5 | 405.0 (= terminal) |
| w8698m2600p40t300__w-lo-late | TRAIN | 360.0 | 364.0 | 482.0 | mainline.210 | 481.0 | 481.5 |
| | | | | | mainline.211 | 481.0 | 481.5 |
| w8700m2600p40t420__w-mid-late | TRAIN | 480.0 | 489.0 | 626.0 | mainline.314 | 625.0 | 625.5 |
| | | | | | mainline.315 | 627.0 | (present at abort) |
| w8703m2600p40t540__w-lo-early | TRAIN | 540.0 | 554.0 | 686.0 | mainline.334 | 685.0 | 685.5 |
| | | | | | mainline.333 | 685.5 | 686.0 (= terminal) |
| w8705m2700p40t300__w-lo-early | TRAIN | 300.0 | 304.0 | 390.0 | mainline.147 | 389.0 | 389.5 |
| | | | | | mainline.148 | 389.0 | 389.5 |
| w8705m2700p40t300__w-lo-late | TRAIN | 360.0 | 364.0 | 446.0 | mainline.196 | 445.0 | 445.5 |
| | | | | | mainline.197 | 445.0 | 445.5 |

All aborts carry `reason=member_left_network`,
`original_reason=member_frame_missing`, `handback=native`. Members are
absent from the row's `vehicles` at the failing tick while having been
CAV-reported at every tick they were present. Categories ruled out: no
internal `:`-lane report gap at any failing tick (junction ticks of members
are isolated early transits, always reported); no non-CAV member id.

### Mechanism B (3 runs) — member in the network, unrostered at active ticks

| run | split | request | terminal | members | first unrostered tick | slot-pool drops (n_dropped_concurrent) |
|---|---|---|---|---|---|---|
| w8669m2700p10t360__w-mid-early | TRAIN | 478.0 | released 620.0 | mainline.344, mainline.347 | 478.0 (= request tick) | 18282 |
| w8669m2700p10t360__w-mid-late | TRAIN | 478.0 | released 560.0 | mainline.344, mainline.347 | 478.0 (= request tick) | 7287 |
| w8704m2700p40t480__w-mid-late | TRAIN | 540.0 | released 667.0 | mainline.377, mainline.378 | 540.0 (= request tick) | 6458 |

At e.g. t=540.0 (w8704) both members are in the row's `vehicles` on normal
lanes AND in that row's `cav_ids`, but >128 concurrent vehicles leave them
without a featurizer slot -> absent from `positions`/roster. No departure,
no junction gap, no non-CAV id. (Members are present in the network at
every active tick; their ids appear in the cumulative CAV report, e.g. in
the row's `cav_ids` at the spot-checked tick and in the featurizer's
`all_cav_ids` union.)

## Fix (corpus-owned wrapper only; frozen v1 files untouched)

`corpus/j4_wall_fix.py` keeps the unconditional terminal-tick tolerance and,
at other visible ticks, tolerates a roster-missing member only with physics
evidence from one lazy per-run streaming pass (members only):

- member absent from the network at t (mechanism A) -> tolerated; the wall
  keeps its recorded event timeline (holding -> aborted flash at the
  recorded terminal tick) while the departed member contributes no
  positions, exactly like the existing absent-at-t degradation (amendment-2
  road-end case; downstream relabel unchanged);
- member present in the network but CAV-reported somewhere in the run
  (mechanism B / report-gap family) -> tolerated; last reported member
  state carried (no new geometry that tick);
- member present in the network and never CAV-reported anywhere -> genuine
  non-CAV member (data bug) -> still fails loudly;
- no physics evidence at all -> only the terminal tick tolerates (previous
  behaviour, keeps the synthetic corpus tests' contract).

Raw evidence: `J4_WALLTOKEN_DIAGNOSIS_RAW.json` (this directory), produced by
`j4diag/diagnose_j4_walltoken_failures.py` (read-only over the runs).

## Re-featurization of the 18 runs (fix in place)

`python -m corpus featurize-all --inventory _tmp_j4fix_refeat_inv.parquet
--data-root E:/research-dense --out E:/rd-wt/feat1j4fix/_tmp_j4fix_refeat
--workers 1 --idea-fast` (cwd `output/wd01/wm_n1`, `WD_FORBID_D=1`,
1 worker / BelowNormal, output on E: only, D: untouched):

```
featurize-all: runs=18 written=18 skipped=0 failed=0 workers=1
```

| # | run | split | result |
|---|---|---|---|
| 1 | w8658m2600p10t360__w-lo-early | CALIBRATION | ok (written, n_ticks=1201) |
| 2 | w8664m2600p10t360__w-lo-early | TRAIN | ok |
| 3 | w8667m2700p10t420__w-lo-early | TRAIN | ok |
| 4 | w8669m2700p10t360__w-mid-early | TRAIN | ok |
| 5 | w8669m2700p10t360__w-mid-late | TRAIN | ok |
| 6 | w8670m2700p10t300__w-lo-late | TRAIN | ok |
| 7 | w8682m2400p20t240__w-lo-late | TRAIN | ok |
| 8 | w8684m2700p20t360__w-lo-early | TRAIN | ok |
| 9 | w8686m2600p20t240__w-mid-late | TRAIN | ok |
| 10 | w8692m2600p40t300__w-lo-late | CALIBRATION | ok |
| 11 | w8694m2600p40t240__w-lo-late | TRAIN | ok |
| 12 | w8697m2700p40t240__w-lo-late | TRAIN | ok |
| 13 | w8698m2600p40t300__w-lo-late | TRAIN | ok |
| 14 | w8700m2600p40t420__w-mid-late | TRAIN | ok |
| 15 | w8703m2600p40t540__w-lo-early | TRAIN | ok |
| 16 | w8704m2700p40t480__w-mid-late | TRAIN | ok |
| 17 | w8705m2700p40t300__w-lo-early | TRAIN | ok |
| 18 | w8705m2700p40t300__w-lo-late | TRAIN | ok |

No run remains in error. Verification of the 18 fresh shards:
`features.cli verify --data-root` -> rc=0, ok=true, n_runs=18;
`corpus.idea_a verify_idea_a` -> `{'ok': True, 'n_runs': 18, 'errors': []}`.

## Byte-identity regression (already-finished J4 wall runs)

The two TRAIN J4 wall runs featurized by the OLD terminal-tick-only wrapper
(`w8660m2700p10t300__w-lo-late`, `w8661m2600p10t240__w-mid-early`) were
re-featurized with the NEW wrapper into
`E:/rd-wt/feat1j4fix/_tmp_j4fix_byteid` (`--idea-fast`, workers=1) and every
file sha256-compared against the live
`D:/rd-staging/wmn1_feat/shards_v2` (READ-ONLY):

```
BYTE-IDENTITY: 38/42 file shas equal vs live shards_v2 (read-only)
```

The 4 non-equal files are the two receipts per run (`DONE.json`,
`IDEA_A_DONE.json`), whose ONLY differing fields are `shard_dir`
(E: tmp output path) and `written_utc` (wall-clock): every recorded array
sha and all 19 data files per run (12 v1 arrays + slot_vid + meta +
outcomes + 5 Idea A arrays + input physics) are byte-identical. The new
wrapper does not change any already-featurized J4 shard.

