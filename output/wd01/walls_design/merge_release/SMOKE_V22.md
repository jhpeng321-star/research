# SMOKE_V22.md — declared mechanics smoke fixture, wall law v2.2 'merge-release'

2026-10-01. Owner-approved laptop smoke: **3 runs x 300 sim-s = 900 sim-s
total**; the RECONSTRUCTOR runs it (the proposer never runs SUMO). The
machine-readable fixture is `SMOKE_V22.json` (same directory). Smoke
runs are mechanics fixtures, **never data**: they are never read as
outcomes and never tune anything.

## 1. What this smoke proves (mechanics, on the patched code)

For each of the three runs, on the REAL frozen dispatcher + patched
v2.2 law, from the restored mode-27 saved state:

- MC1 the wall ASSEMBLES upstream of the merge (all members still on
  entry edge 1191554636 before the release tick);
- MC2 `merge_release` is logged exactly once, as a RELEASE kind, at
  `s_road >= 2644.2` (release point 2644.2 = the merge start), BEFORE
  any member reaches the merge edge (no member-lanes row shows edge
  1487606179/1487606178 before the release tick; the lead member is at
  most one control tick of travel past the point at release);
- MC3 ZERO `member_left_network` anywhere (the J4 road-end failure
  mode is structurally gone; any safety_abort in a smoke run FAILS);
- MC4 the registered hold draw (240 s) is CAPPED at the merge start and
  the cap is logged (`hold_cap` row: `requested_hold_s` 240 vs effective
  < 240, `capped_at_merge: true`, `cap_point_s: 2644.2`; branch_record
  `hold_capped_at_merge: true`, `merge_release_s` set, outcome
  `merge_released`);
- MC5 the receipt contract holds (`exit_status complete`,
  `load_state_calls == 1`, `horizon_mode smoke_cap`, 600 physics rows).

## 2. Deployment (reconstructor, in a no-campaign window)

1. Apply the four patches (LF, unified diffs vs the live v2.1 files,
   shas in `PATCH_SHAS.json`; from the research root, e.g.
   `git apply E:/rd-wt/wallv22/output/wd01/walls_design/merge_release/patches/common.py.diff`
   — the diff headers target `output/campaign_walls_v1/*.py` and
   `traffic_shepherd/*.py`; the LF sha256 of each patched file must
   equal the `post` entry in `PATCH_SHAS.json`):
   - `patches/common.py.diff` → `output/campaign_walls_v1/common.py`
   - `patches/wall_request.py.diff` → `output/campaign_walls_v1/wall_request.py`
   - `patches/wall_driver.py.diff` → `output/campaign_walls_v1/wall_driver.py`
   - `patches/wall_group.py.diff` → `traffic_shepherd/wall_group.py`
   (`sensing.py` needs NO change — its frames already carry the lane id
   and lane-local position the edge-aware `s_road` reads.)
2. Finalize the J4B v2.2 registration (write-once) at
   `output/campaign_walls_v1/results_plan/WALLS_PILOT_REGISTRATION_V22.md`
   from `REGISTRATION_WALL_V22.md` + the launch blocks of
   `REGISTRATION_J4B_DRAFT.md`. The J4 registration file stays frozen.
3. `runs/campaign_walls_v22_smoke` must NOT exist (write-once).

## 3. Selection (a priori, J4 geometry only, no outcomes)

TRAIN worlds only; registered hold draw 240 s (so the MERGE-START CAP —
not the draw — ends the hold); a-priori proposer gate
`D_min = 70 * v_wall` admitted with positive margin; predicted
merge-start arrival `assembled + (2644.2 - lead_s_road_assembled)/v_wall`
fits the 300 sim-s horizon with >= 100 s slack. Source: the J4 analysis
geometry rows (`j4_merge_analysis.json`, TRAIN only; request/assembly/
hold records — no outcome field was opened).

| run (world / arm) | p | v_wall (J4) | asm lead s_road | D_min | margin | predicted arrival | hold draw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| w8694m2600p40t240 / W-lo-early | 0.4 | 15.87 m/s | 1219.4 m | 1110.9 m | +313.9 m | t0+93.8 s | 240 s |
| w8688m2700p20t420 / W-mid-late | 0.2 | 26.31 m/s | 669.4 m | 1841.7 m | +133.1 m | t0+146.1 s | 240 s |
| w8661m2600p10t240 / W-mid-early | 0.1 | 22.52 m/s | 358.1 m | 1576.4 m | +709.7 m | t0+175.5 s | 240 s |

## 4. Exact launcher command lines (cwd = `E:/research-dense`)

Run the three commands SEQUENTIALLY (laptop smoke; one fresh SUMO
process per run inside the child):

```text
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe output/campaign_walls_v1/run_wall_branch.py _exec_wall_branch --run w8694m2600p40t240__w-lo-early --plan output/campaign_walls_v1/results_plan/WORLD_PLAN.json --runs-root runs/campaign_walls_v22_smoke --registration output/campaign_walls_v1/results_plan/WALLS_PILOT_REGISTRATION_V22.md --bases-root E:/research-dense/runs/campaign_stage1_chunk4 --horizon-ticks 300

C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe output/campaign_walls_v1/run_wall_branch.py _exec_wall_branch --run w8688m2700p20t420__w-mid-late --plan output/campaign_walls_v1/results_plan/WORLD_PLAN.json --runs-root runs/campaign_walls_v22_smoke --registration output/campaign_walls_v1/results_plan/WALLS_PILOT_REGISTRATION_V22.md --bases-root E:/research-dense/runs/campaign_stage1_chunk4 --horizon-ticks 300

C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe output/campaign_walls_v1/run_wall_branch.py _exec_wall_branch --run w8661m2600p10t240__w-mid-early --plan output/campaign_walls_v1/results_plan/WORLD_PLAN.json --runs-root runs/campaign_walls_v22_smoke --registration output/campaign_walls_v1/results_plan/WALLS_PILOT_REGISTRATION_V22.md --bases-root E:/research-dense/runs/campaign_stage1_chunk4 --horizon-ticks 300
```

`--horizon-ticks 300` caps each branch at 300 sim-s; NO
`--hold-override-s` is passed (the registered 240 s draws stay as
drawn — the merge-start cap is what must end the hold; the override is
a smoke-only flag this fixture deliberately does not use). The child
refuses an existing run dir (write-once; a re-smoke uses a NEW runs
root and is registered as such).

## 5. Pass rule

ALL of MC1–MC5 for EACH of the three runs (checks specified field-by-
field in `SMOKE_V22.json`). Any FAIL stops the fixture; the J4B launch
gate (registration + package + dependency digests) binds only on a
recorded PASS.
