# WALLFIX: missing wall members in the 72-state WM check

Branch `c4/v10r5wall` from `c4/v10int` `500ce758`. Inference only: laptop CPU, 3 workers,
no GPU, no SUMO, no training. TRAIN-val/wall-regen states only; no evaluation seeds and
no sealed/holdout paths were touched. The original `wall_same.json` was read-only; all
rebuilt/sidecar files are under `_out/`.

## 1. Cause and counts (all 144 windows)

`traffic_shepherd.c4.wallregen_states` writes the recorded futures of the **t0 roster
only** (`_record`: `roster = sorted(tl.present(j0)); recorded = tl.future(j0, roster)`).
A wall member that has not spawned at t0 is therefore not in `initial_state`, not in the
`wall`/`native` recorded futures, and not in `history`. With entries off it never exists
in the WM rollout.

Counts over `E:/agent-work/c4-v6merge/_ref/states_h60/wall_same.json`:

| quantity | count |
|---|---|
| windows | 144 |
| windows with `wall_timeline.members` | 141 (3 `not_feasible` windows have none) |
| member slots | 282 |
| members present in the t0 roster | 124 |
| members absent from t0 and from every recorded future | **158** |
| windows with >= 1 absent member | 80 |
| windows with all members absent | 78 |
| windows with a complete wall | 61 |

All 158 absent members have an id **above the highest mainline id in the t0 roster**
(they have not spawned yet). None is below that id, none appears in `history`, and none
appears anywhere in the recorded arms. The 158 slots split by outcome as 126
`assembled_released`, 30 `aborted_in_hold`, 2 `aborted_in_assembly`. All 80 windows with
a missing member have `wall_phase_at_t0 = "none"`; the 61 complete windows are 37
`assembly`, 17 `hold`, 7 `none`.

The affected blow-up states in `BLOWUP_DIAG.md` section 3/R1 reproduce exactly:

| state | members | missing | request rel s | release rel s | outcome |
|---|---|---|---|---|---|
| 16 | mainline.161,160 | 2/2 | 40 | 101 | assembled_released |
| 18 | mainline.161,160 | 2/2 | 40 | 161 | assembled_released |
| 74 | mainline.246,241 | 1/2 (246) | 8 | 102 | assembled_released |
| 84 | mainline.344,347 | 2/2 | 86 | 219 | assembled_released |
| 86 | mainline.344,347 | 2/2 | 86 | 219 | assembled_released |
| 96 | mainline.292,291 | 2/2 | 50 | 292 | assembled_released |
| 98 | mainline.292,291 | 2/2 | 50 | 292 | assembled_released |
| 102 | mainline.169,168 | 1/2 (169) | 7 | 248 | assembled_released |

## 2. Fix

* `wms.eval.rollout_inputs.wall_member_teacher` extracts a member's recorded
  `(s, lane, v)` path from `state["wall_member_paths"]` and builds the same
  `teacher / present / junction` arrays used for t0 CAVs. It starts at the member's first
  recorded on-road row, carries through junction transits, and stops at a real missing
  row. Members already in the t0 roster are left to the normal CAV teacher.
* `wms.eval.rollout_inputs.WallMemberInsert` inserts each missing member into a reserved
  slot on the step before its first recorded appearance. It follows the existing entry
  process interface (`insert(state) -> mask`), so the model's normal `cav_teacher` path
  then forces the recorded trajectory. There are no hand-set behaviour rules.
* `wms.eval.predictor.RolloutFlags.wall_members` (default **off**) pads the state with
  those slots (present=False, is_cav=True), extends `teacher / present_seq /
  junction_seq / driver_history`, and attaches the insertion process. The t0 roster and
  all metrics are unchanged.
* `tools/v8wm3_check.py`: `--wall-members {off,insert}` (default off) and
  `--wall-member-paths SIDECAR.json`. Off is the frozen comparison path; insert is only
  active for windows present in the sidecar.
* `tools/wall_member_paths.py`: rebuilds the missing paths from the original wallregen
  sim (`world.json`, `base.npz`, `arm_*.npz`) into a sidecar, without modifying the
  states file. It reads the member rows directly from each tape step; the exporter's
  `future()` helper cannot be used because it prunes the roster to t0.

Unit test: `tests/unit/test_wall_members.py` (3 tests) covers first-appearance
extraction, insertion into the reserved slot, switch-off invariance, and an end-to-end
predictor rollout where the member appears only at its recorded first row and follows
the teacher path.

## 3. Round-2 72-state check

`E:/Temp/dc_laptop/ck_round2/wm_s_stage240.pt`, `wall_same.json` every second window
(72 states), recorded `wall` arm, CAVs teacher-forced, entries off, recorded SUMO exits,
120 s, lane seed 0, 3 workers. Off/on are the same command except for the switch and the
sidecar.

| variant | stuck in lane 2 | blow-ups | HDV ds60 | HDV ds120 | median abs dep. err s | early>2 s / late>2 s |
|---|---|---|---|---|---|---|
| off | 50 of 368 (4 SUMO-stuck) | 9/72 | 29.0 | 46.0 | 17.0 | 160 / 121 |
| insert | 50 of 368 (4 SUMO-stuck) | 9/72 | 29.0 | 46.0 | 16.8 | 160 / 119 |

Both variants cover all 72 states with 0 errors. A committed-code spot rerun of state `0`
(`--limit 1 --every 2`, 1 worker, 14-15 s per variant) reproduces the table exactly:
off `35.9 / 2220.6`, insert `33.3 / 1807.4`, 0 blow-ups both.

Only the windows with usable member paths changed:

| state | wall | off ds60/ds120 | insert ds60/ds120 | blow-up |
|---|---|---|---|---|
| 0 | 2/2 missing | 35.9 / 2220.6 | 33.3 / 1807.4 | n -> n |
| 2 | 2/2 missing | 35.9 / 2220.6 | 33.3 / 1807.4 | n -> n |

The requested blow-up states and the control states did **not** change, because their
worlds' member paths are not available (section 4):

| state | wall | ds60/ds120 off = insert | blow-up |
|---|---|---|---|
| 16 | 2/2 missing | 51.1 / 142.6 | Y |
| 18 | 2/2 missing | 51.1 / 142.6 | Y |
| 74 | 1/2 missing | 46.7 / 144.6 | n |
| 84 | 2/2 missing | 54.2 / 311.2 | Y |
| 86 | 2/2 missing | 54.2 / 311.2 | Y |
| 96 | 2/2 missing | 59.6 / 138.9 | Y |
| 98 | 2/2 missing | 59.6 / 138.9 | Y |
| 102 | 1/2 missing | 78.6 / 169.9 | Y |
| 40 (control) | complete | 14.2 / 2.9 | n |
| 82 (control) | complete | 20.8 / 16.8 | n |
| 114 (control) | complete | 10.4 / 15.7 | n |
| 132 (control) | complete | 15.1 / 15.5 | n |
| 50 (control) | complete | 21.5 / 20.8 | n |

Wall-body association in the off run: 16 states with an incomplete wall contain 7 of the
9 blow-ups; the 56 complete-wall states contain 2 (`90`, `106`). The two missing-member
states with available paths (`0`, `2`) are not blow-ups, so the insertion mechanism is
exercised but this sample does not test the blow-up recovery.

## 4. Member-path source and sidecar (cpuN/gpuR copies)

The paths do **not** come from `wall_same.json` (it keeps only the t0 roster). They come
from the original wallregen tape of the same run: per world,
`out/sim/worlds/w<world>/world.json` (arm registry + metadata), `base.npz` (pre-branch
history) and `arm_<arm>.npz` (the 0.5 s tape: `time_s, s, lane, v, a, mask, vid`). The
timeline is `Timeline(base, arm_npz, registry, is_cav)` from
`traffic_shepherd.c4.wallregen_states`; `tools/wall_member_paths.py` reads member rows
from those tape steps directly (`present(j)` / `row(j, vid)`), bypassing the exporter's
`future()` roster pruning. The recorded futures in `wall_same.json` stay untouched.

The sidecar is keyed by state `key` (world + window/arm), so the gpuR and cpuN copies of
`wall_same.json` produce the same sidecar. Build it on the host that still has the 36
wallregen worlds (here only 3 sample worlds survive, hence the 4-window sidecar):

```bash
# gpuR copy of the states (known path) and its sibling sim directory
export SET=/root/autodl-tmp/c4/v5/runs/v5_wallregen_20261014/out/states_h60full/wall_same.json
export SIM=/root/autodl-tmp/c4/v5/runs/v5_wallregen_20261014/out/sim
python tools/wall_member_paths.py --set "$SET" --sim "$SIM" \
    --out _out/wall_members_h60full.json --horizon-s 120

# cpuN copy: same states layout under the cpuN base ~/c4 (substitute the actual copy)
# export SET=$HOME/c4/runs/v5_wallregen_20261014/out/states_h60full/wall_same.json
# export SIM=$HOME/c4/runs/v5_wallregen_20261014/out/sim
# python tools/wall_member_paths.py --set "$SET" --sim "$SIM" \
#     --out _out/wall_members_h60full.json --horizon-s 120
```

The sidecar contains `{"schema":"wall_member_paths/1", "set":..., "sim":...,
"worlds_available":..., "windows_with_members":..., "member_paths":...,
"windows":{<key>:{"members":[{"id","time_s","s","lane","v","present"}, ...]}}}`.
Then rerun the check (copy the sidecar next to the checkpoint on cpuN if needed):

```powershell
python tools/v8wm3_check.py --var base=E:/Temp/dc_laptop/ck_round2/wm_s_stage240.pt `
    --workers 3 --entries off --wall-members insert `
    --wall-member-paths _out/wall_members_h60full.json --out _out/r5wall/round2_on_full.pkl
```

Why the current sample stops at 2 states: the original sim was pruned on this host. Only
three sample worlds remain in `E:/agent-work/c4-wallhist/_ref/sim/worlds`: `w962000`,
`w962006`, `w962009`. The sidecar therefore has 4 usable windows (`0`-`3`, 8 member
paths); only states `0` and `2` are in the 72-state subset. The target worlds
`w962004` (16/18), `w962018` (74), `w962021` (84/86), `w962024` (96/98) and `w962025`
(102) are absent, so their member trajectories cannot be reconstructed from the state
file (it contains no post-t0 entrants). Smallest next action: copy the full 36-world
`out/sim` (or just those five world directories) to this host and run the commands
above.

## 5. Checks

```powershell
pytest tests/unit/test_wall_members.py tests/unit/eval/test_eval_units.py `
       tests/unit/test_v9entry_eval.py tests/unit/test_wallhist.py `
       tests/unit/test_wallregen.py -q     # 35 passed, 2 skipped (sample sim absent in repo)
ruff check src/wms/eval/predictor.py src/wms/eval/rollout_inputs.py `
           tools/v8wm3_check.py tools/wm_eval_common.py `
           tools/wall_member_paths.py tests/unit/test_wall_members.py   # clean
```
