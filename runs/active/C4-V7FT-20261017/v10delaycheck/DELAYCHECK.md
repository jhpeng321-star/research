# A6 DELAY-ACCURACY CHECK — WM checkpoints and the loop rollback rule

Branch `c4/v10delaycheck` (from `c4/v10int` `e80d323b`). Tool:
`scripts/c4/wm_delay_check.py`; tests: `tests/unit/test_wm_delay_check.py`. Loop integration:
`scripts/c4/loop_v10.py` (`delay_check` stage + A6 rollback rule) and
`configs/c4_dreamer_loop_v10loop.json` (`delay_check` key, new `rollback` keys). This report lives
outside the code worktree because `runs/` is git-ignored; it is force-added on the branch.

## 1. What the check measures

### 1.1 Delay LEVEL (WM vs SUMO, on-road, by region)

`delay = Σ_t Σ_v present(v) · relu(1 − v/31.29)` sampled on a **1 s grid** (`t = t0, t0+1, …`),
binned by region, for both the recorded SUMO arm and the WM rollout of the same start. This is
the `onroad_diag` definition (`runs/active/C4-V7FT-20261017/onroad_diag/ONROAD_DIAG.md`): free
speed `31.29 m/s` (`wms.control.contract.SPEED_LIMIT_MPS`), on-road `s < 553.33 m`
(`wms.physics.batched.MODEL_FRAME_LANE_LENGTHS` / `wms.control.belief.DEFAULT_LANE_EXTENTS`).

Regions (RSU zones, `wms.control.contract`), `region_of(s, lane)` in the tool:

| region | lanes | s range (m) |
|---|---|---|
| `up_near` | mainline 0/1 | `[-800.0, 0.0)` (`RSU_MAINLINE_S_M`) |
| `ramp` | ramp/accel lane 2 | `[-378.66, 75.44)` (`RSU_RAMP_S_M`) |
| `merge` | mainline 0/1 | `[0.0, 75.44)` (`MERGE_S0_M`..`MERGE_S1_M`) |
| `downstream` | mainline 0/1 | `[75.44, 553.33)` |

`total = up_near + ramp + merge + downstream`. `ramp_queue` is reported separately: stopped
(`v < 0.5`) vehicle-seconds of the **entrance backlog** (queued before the ramp start or an
`__entry_` slot). With `--entries queue` the WM rollout is seeded with the recorded entrance
backlog (`RolloutFlags(entries=True, entries_v2=True, entry_queue_state=True)`, c4/v9entry), so
the ramp/source queue is carried through the rollout. Horizons reported: 60/120/300 s. WM delay
is the **mean over `--k` rollout samples** (default `k=8`).

### 1.2 PAIRED DELAY DIFFERENCE (recorded genrand_branch pairs)

`data/genrand_branch` branches each saved SUMO state into arms sharing one `pair_key`
(`<world>@t<branch time>`): the base/native reference arm and action arms (`SEED:j`), all recorded
in SUMO from the **same t0 slots/kinematics** (`wms.data.c4_pairs.PairedBranchSource`). For each
pair the tool computes

* `sumo_delta = delay_level(action, recorded) − delay_level(base, recorded)`,
* `wm_delta   = delay_level(action, WM)        − delay_level(base, WM)`,

and across pairs reports **sign agreement**, **Spearman rank correlation** and **mean absolute
error** of `wm_delta` vs `sumo_delta` per horizon. WM arm deltas are `k`-sample averaged. The
reference arm is `native`/`B`/`…_native`, else the lexicographically first arm.

### 1.3 Reported (non-gating) diagnostic

Per-vehicle `|ds|` at 10/30 s for HDVs within `SENSOR_RANGE_M = 100 m` of a CAV at t0
(the old rollback term). Reported only; it never gates the rollback.

## 2. Deterministic selection

48 starts and 48 pairs, evenly spaced across the sorted keys
`(world, key, source_arm, t0_s)` / `(pair_key, ref_arm, action_arm, t0_s)`. The same input always
yields the same records. `--save-selection` / `--save-pairs-selection` write the selection once
(`c4_wm_delay_check/selection/1`); later runs read it via `--selection` / `--pairs-selection`.
TRAIN-val data only; evaluation seeds 964000–964199 are never touched; any path carrying
`sealed/holdout/heldout/cal_split/chunk5/j4b/inc1` is refused.

## 3. Laptop smoke (this worktree, no GPU, no SUMO)

`v5m` (md5 `d736db86`) and `data/genrand_branch` are **not in this workspace**, so the laptop
smoke uses the in-tree CPU-checkpoint `tests/golden/data/c1/wm_s.pt` and a 2-start / 2-pair
synthetic TRAIN-val-like dataset (10 mainline + 2 ramp vehicles + a persistent merge queue; arms
`native`/`wall` and base `B`/action `SEED:0`). It exercises the full plumbing (real WM rollout →
region delay → paired delta) at reduced scale, `k=2`, `workers=3` (≤ 3, shared laptop):

```sh
PYTHONPATH=src python scripts/c4/wm_delay_check.py \
  --wm tests/golden/data/c1/wm_s.pt \
  --states <smoke>/starts.json --pairs <smoke>/pairs.json \
  --horizons 60,120,300 --k 2 --device cpu --workers 3 --entries off \
  --out <smoke>/result.json --md <smoke>/result.md
```

Output (WM `wm_s.pt`):

| horizon | WM total | SUMO total | log ratio | sign agree | Spearman | MAE |
|---|---|---|---|---|---|---|
| 60 | 130.3311 | 228.3058 | −0.5594 | 0.0000 | n/a | 41.0007 |
| 120 | 130.3311 | 348.3058 | −0.9825 | 0.0000 | n/a | 41.0007 |
| 300 | 130.3311 | 708.3058 | −1.6927 | 0.0000 | n/a | 41.0007 |

Rollback block: `delay_level_log_ratio = −0.9825` (120 s), `pair_sign_agreement = 0.0` (60 s),
`near_cav_abs_ds_10/30 = n/a` (no CAV in the synthetic fleet). The C1 checkpoint is an old
`full_struct` model (no `lane_learnable`, no entries), so the absolute numbers are not the A6
result — they only confirm the tool runs and produces finite per-region / paired statistics.
`Spearman = n/a` because only two pairs are selected in the smoke (needs ≥ 2 with non-zero
variance; it is `n/a` in the smoke by construction).

## 4. Exact CPU-box command (coordinator)

48-core CPU box, `~/c4`, `~/miniconda3/bin/python`, no GPU, `k=8`, `workers=44`. Sync once:
`scp <new_wm> cpuN:~/c4/delay_check/wm_s_best.pt` and copy the fixed starts / genrand_branch
records to `~/c4/delay_check/`.

```sh
cd ~/c4
OMP_NUM_THREADS=1 ~/miniconda3/bin/python scripts/c4/wm_delay_check.py \
  --wm ~/c4/delay_check/wm_s_best.pt \
  --states ~/c4/delay_check/fixed_train_val_states.json \
  --pairs ~/c4/delay_check/genrand_branch_states.json \
  --selection ~/c4/delay_check/delay_states_selection.json \
  --save-selection ~/c4/delay_check/delay_states_selection.json \
  --pairs-selection ~/c4/delay_check/delay_pairs_selection.json \
  --save-pairs-selection ~/c4/delay_check/delay_pairs_selection.json \
  --horizons 60,120,300 --k 8 --device cpu --workers 44 --entries queue \
  --sumo-pairs ~/c4/loop_v10/iter_001/check/pairs.json \
  --out ~/c4/loop_v10/iter_001/check/delay_check.json \
  --md  ~/c4/loop_v10/iter_001/check/delay_check.md
```

For the v5m baseline (`wm_v5m/wm_s_best_ap.pt`, md5 `d736db86`) point `--wm` at the synced copy;
for the iteration-0 WM use `runs/c4_wm_v10_iter000/wm_s_best.pt` (+ its `physics.json`,
`lane_learnable=true`). The fixed starts are the same 48 as
`scripts/c4/wm_fidelity_v10.py --select 48 --save-selection`; the pair records come from the
converted `genrand_branch` source (`paired=true`).

## 5. Loop rollback integration

New config block `delay_check` (own key) and a `delay_check` stage in
`loop_v10.plan_iteration` (host `cpuN`, runs `wm_delay_check.py` per iteration after `sumo_check`,
before `rollback`). The `rollback` stage now runs
`loop_v10.py decide-rollback --config configs/c4_dreamer_loop_v10loop.json --prev <prev delay_check.json> --cur <check>/delay_check.json`.

**Replaced rollback WM term** (`hdv.abs_ds.60`, removed) by (A6):

* revert if the **paired-delta sign agreement** drops by `> 0.05` vs the previous iteration
  (`rollback.pair_sign_drop_max = 0.05`, metric = 60 s sign agreement),
* revert if the **120 s delay-level `|log(WM/SUMO)|`** worsens by `> 0.1` vs the previous
  iteration (`rollback.delay_log_ratio_worsen_max = 0.1`),
* the SUMO total-D term is kept (`sumo_total_D_max_worsen = 0.0`).

The per-vehicle `|ds|` at 10/30 s near CAVs is kept as a **reported** field
(`decision["reported"]`, `near_cav_abs_ds_10/30`) and never gates. `decide_rollback` reads the
metrics from the delay-check JSON's `rollback` block (legacy flat dicts still work). On revert the
next policy-update count is halved (`halve_factor 0.5`, floor `min_policy_updates 50`).

## 6. Tests

`tests/unit/test_wm_delay_check.py` (16): region bounds, 1 s-grid delay sums, entrance queue,
near-CAV `|ds|`, sign agreement / Spearman (with ties) / MAE / log ratio, deterministic selection
+ round trip, `group_pairs` (per-arm merge and multi-arm record), rollback block, restricted
paths, CLI parser (repeatable `--wm`, defaults), markdown render.
`tests/unit/test_c4_loop_v10.py`: rollback rule (each A6 term reverts on its own, reads the
`rollback` block, halving/floor) and the `delay_check` stage (cpuN, `wm_delay_check.py`,
`--horizons 60,120,300 --k 8 --entries queue --save-selection`, rollback consumes
`delay_check.json`). All 45 tests in the three touched suites pass; `plan` and `dry-run` execute
the full stage list with 0 external commands.

## 7. Files and commits (branch `c4/v10delaycheck`)

* `scripts/c4/wm_delay_check.py` (new) — the delay-accuracy tool.
* `tests/unit/test_wm_delay_check.py` (new) — its tests.
* `scripts/c4/loop_v10.py` — `delay_check` stage + A6 `decide_rollback`.
* `configs/c4_dreamer_loop_v10loop.json` — `delay_check` key + new `rollback` keys.
* `tests/unit/test_c4_loop_v10.py` — updated rollback/stage tests.
* `8e18f4cf` — A6 delay-check tool + tests + loop rollback rule.
* (this report) — force-added `runs/active/.../v10delaycheck/DELAYCHECK.md`.
