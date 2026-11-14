# C1 evaluation harness — status

**Status: C1 v2 sets REBUILT (2026-10-06) — `_out/c1_states_v2/` holds the clean
`unseen_u_same_world.json` (140 states / 140 worlds), `unseen_u_new_world.json`
(21 states / 21 worlds) and `seen.json` (157 states / 26 worlds), plus `MANIFEST.json`
(counts, world lists, sha256) and `SEPARATION.json`. The v2 guard passes:
`u_same ∩ u_new = 0`, `seen ∩ u_same = 0`, and 0 training-window key/run leak in any
set. Full-corpus scoring is still NOT executed here; the 6-state E2E proof below stands.**

## C1 v2 clean sets (2026-10-06)

The first pass leaked: `seen_val` shared 21 worlds with `unseen_pi_fleet161` and 4 with
`unseen_pi_mixed_sample`, and 18 mixed-sample records were TRAIN runs. The v2 rebuild
applies explicit rules (zero simulation; only the small metadata JSONs are read, and the
materialised files are streamed once for the manifest) via
`wms_eval/c1sets.py` + the new `materialize` outputs.

| set | rule | states | worlds |
|---|---|---|---|
| `unseen_u_same_world` | PI states whose world also has non-PI arms in `windows_train_noPI.json` — the paired "same situation, new controller" case | 140 | 140 |
| `unseen_u_new_world` | PI states whose world has **no** training window at all (world-grouped validation worlds) | 21 | 21 |
| `seen` | held-out PI-free validation states disjoint from every U-same world | 157 of 189 requested | 26 |

Rules enforced by `c1sets.unseen_split` / `c1sets.seen_selection` and checked by
`c1sets.verify` (fail-closed):

* every PI record whose `run`/`key` appears in `windows_train_noPI.json` or
  `windows_val.json` is dropped (0 dropped in this corpus);
* U-same and U-new worlds are disjoint;
* no U-same / U-new / seen record reuses a training-window `key` or `run`;
* `seen` is disjoint from every U-same world (0 overlap).

`seen` is the held-out ordinary (PI-free) validation reference; its 26 worlds include the
21 U-new worlds, whose own PI-free arms are the same-world paired reference for that
split (`seen ∩ u_new = 21`, reported, not refused). The strict subset of validation
worlds disjoint from *both* unseen subsets contributes only 5 states after the known
origin-0 `base` snapshots are dropped (1-vehicle degenerate), so reaching the ≥100 target
uses the paired U-new validation worlds — the only held-out ordinary states available
(all others are training windows or CALIBRATION data).

```powershell
cd E:/rd-wt/ws-c1eval/output/wd01/wm_n1
# materialise (one state at a time; packs mmap_mode='r'; RSS < 3 GB)
& $py -m wms_eval materialize --pack pack_v3=D:/rd-staging/wmn1_feat/pack_v3 ... `
  --pi-states E:/rd-wt/ws-p5/_out/lpo/pi_eval_states.json `
  --windows-val E:/rd-wt/ws-p5/_out/lpo/windows_val.json `
  --train-windows E:/rd-wt/ws-p5/_out/lpo/windows_train_noPI.json `
  --unseen-u-same E:/rd-wt/ws-c1eval/_out/c1_states_v2/unseen_u_same_world.json `
  --unseen-u-new  E:/rd-wt/ws-c1eval/_out/c1_states_v2/unseen_u_new_world.json `
  --seen-v2       E:/rd-wt/ws-c1eval/_out/c1_states_v2/seen.json --horizon-s 300
# guard + manifest
& $py -m wms_eval c1sets verify --u-same ... --u-new ... --seen ... --train-windows ...
& $py -m wms_eval c1sets manifest --set ... --source ... --out .../MANIFEST.json --verify ...
```

Primary metric: the unseen/seen ratio per model on **U-same** (world-disjoint reference)
and on **U-new** (same-world paired reference) separately, so neither mixture hides the
other. The superseded `_out/c1_states/*.json` sets are left untouched (git-ignored,
regenerable) and are no longer valid inputs.

The one remaining gap from the previous C1 harness pass was a pack->state
materialiser (`windows_val.json` / `pi_eval_states.json` are metadata only). It now
exists as `output/wd01/wm_n1/wms_eval/materialize.py` and produced:

| set | states | file | size |
|---|---|---|---|
| `unseen_pi_fleet161` (primary fleet-PI) | 161 | `_out/c1_states/unseen_pi_fleet161.json` | 473 MB |
| `unseen_pi_mixed_sample` (stratified remaining PI) | 200 | `_out/c1_states/unseen_pi_mixed_sample.json` | 441 MB |
| `seen_val` (non-PI `windows_val` worlds) | 157 of 189 | `_out/c1_states/seen_val.json` | 278 MB |

The state JSONs are large, regenerable products (git-ignored); the builder, tests and
the 6-state proof are committed.

No SUMO, GPU, SSH, CAL/HOLDOUT/chunk-5/J4B/INC1/sealed data was opened. All packs and
the WM-S/A11 checkpoints were read read-only (`mmap_mode='r'`); the WM-S checkpoint was
copied inside the workspace only because the loader is workspace-confined.

## What the materialiser does

For every metadata window `(pack, world, run, arm, t0_s)` it emits one self-contained
C1 state record `{initial_state, recorded{arm -> frames to +300 s}}`:

* `pack_v3` `manifest.json`/`INDEX.json` give the run `tick_start/tick_end` and the
  item `tick` (the snapshot tick), read exactly like `train.dataset.MemmapCorpus`;
* the t0 vehicle identities come from the verified identity sidecar
  (`identities.npy` / `global_positions.npy` / `vehicle_map.json`);
* `veh_kin` (`s_down`, lane one-hot, `v`, `a`, length), `veh_mask`, `veh_step[:,7]`
  (`is_cav`) and `target_mask` reconstruct the position/speed/lane future;
* the SAME cumulative `target_mask` product the trainer uses (`train.idea_a.window`)
  keeps a slot's t0 vehicle through the window and never re-labels a reused slot;
* the `native` sibling run (if present in the same pack) is attached as the paired
  reference arm.

Memory: packs stay memory-mapped, exactly one window's tick slice is touched, records
are streamed to disk one at a time (one state ≈ 3 MB); RSS stays far below 6 GB.

`pure_physics` (the A11 adapter) normally reconstructs its decision row from the raw
run (`state_row` + `base_run_dir`), which a pack state does not have. `models.py`
`FrozenA11Model._live_from_state` now assembles the identical 7-column kinematics view
from the materialised `initial_state` (counter history defaults to zeros — documented
approximation; every other channel exact).

### Build command (exact)

```powershell
py = C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe
cd E:/rd-wt/ws-c1eval/output/wd01/wm_n1
& $py -m wms_eval materialize `
  --pack pack_v3=D:/rd-staging/wmn1_feat/pack_v3 `
  --pack pack_dspeed=E:/rd-wt/dspeed-data/_out/pack_dspeed `
  --pack pack_damping=E:/rd-wt/damping-snap/_out/pack_damping `
  --identities pack_v3=E:/tmp/wmn1_train_runs_20261002/sidecars3_20261003T0005CN/identities `
  --identities pack_dspeed=E:/rd-wt/a9/_out/identities_dspeed `
  --identities pack_damping=E:/rd-wt/damping-snap/_out/identities_damping `
  --pi-states E:/rd-wt/ws-p5/_out/lpo/pi_eval_states.json `
  --windows-val E:/rd-wt/ws-p5/_out/lpo/windows_val.json `
  --unseen-pi  E:/rd-wt/ws-c1eval/_out/c1_states/unseen_pi_fleet161.json `
  --unseen-mixed E:/rd-wt/ws-c1eval/_out/c1_states/unseen_pi_mixed_sample.json `
  --seen-val   E:/rd-wt/ws-c1eval/_out/c1_states/seen_val.json
```

`seen_val` keeps 157/189: the 32 dropped windows are `pack_v3` `base`/`native_base`
arms featurised from origin 0 (the known t0-label misalignment, owner decision (b)),
whose snapshot holds 1-2 vehicles; they are refused rather than scored at the wrong
time. Rebuild an E2E subset with `--limit N --stride K` (deterministic).

## 6-state end-to-end proof (executed)

```powershell
cd E:/rd-wt/ws-c1eval/output/wd01/wm_n1
& $py -m wms_eval c1 `
  --models keep_speed= --models pure_physics= `
  --models wms=E:/rd-wt/ws-c1eval/_out/c1_ckpt/wm_s.pt `
  --unseen E:/rd-wt/ws-c1eval/_out/c1_states/e2e_unseen6.json `
  --seen   E:/rd-wt/ws-c1eval/_out/c1_states/e2e_seen6.json `
  --out E:/rd-wt/ws-c1eval/_out/c1_proof/E2E-WMS-20261006 `
  --workers 1 --horizon 300 --metric hdv_ds@60 --bootstrap 64
```

Result (exit 0; `C1.json` / `C1.md` / `RESULTS.json` in the out dir), median HDV
`|ds|` in metres, baseline arm `native`:

| model | set | 10 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|---|
| keep_speed | unseen | 3.13 | 20.48 | 64.98 | 1519.6 |
| keep_speed | seen | 2.67 | 17.59 | 57.88 | 1572.9 |
| pure_physics | unseen | 7.67 | 28.46 | 65.06 | 1112.1 |
| pure_physics | seen | 7.19 | 28.87 | 70.25 | 726.0 |
| wms | unseen | 2.44 | 9.85 | 33.82 | 146.1 |
| wms | seen | 2.88 | 10.69 | 24.13 | 144.5 |

Unseen/seen ratio (CI over worlds, n_boot=64): keep_speed 1.12 [0.58, 2.43],
pure_physics 0.93 [0.60, 1.72], wms 1.40 [0.94, 3.04]. The WM-S checkpoint is the
best long-horizon model; the >1 unseen/seen ratio is the C1 signal to test at scale.
The 6-state `seen` windows are non-PI `native` arms, so their `pi` recorded arm is
absent (reported as a per-arm error, not used); the headline ratio uses the shared
`native` baseline on both sets.

## Full-run command and estimated CPU time

```powershell
cd E:/rd-wt/ws-c1eval/output/wd01/wm_n1
& $py -m wms_eval c1 `
  --models keep_speed= --models pure_physics= `
  --models wms=E:/rd-wt/ws-c1eval/_out/c1_ckpt/wm_s.pt `
  --unseen E:/rd-wt/ws-c1eval/_out/c1_states/unseen_pi_fleet161.json `
  --seen   E:/rd-wt/ws-c1eval/_out/c1_states/seen_val.json `
  --out E:/rd-wt/ws-c1eval/runs/active/C1-WMS-<date>/ `
  --workers 4 --horizon 300 --metric hdv_ds@60 --bootstrap 1000
# secondary: swap --unseen for unseen_pi_mixed_sample.json
```

Measured on this host, one process, torch 4 threads, ~100 vehicles, 300 s horizon:
WM-S ≈ 16 s per arm-rollout, `pure_physics` ≈ 8 s, `keep_speed` ≈ 0.1 s.
318 states (161 unseen + 157 seen) × 2 arms: WM-S ≈ 2.8 h serial / ≈ 0.7 h at 4
workers; `pure_physics` ≈ 1.4 h / ≈ 0.35 h; `keep_speed` negligible. **Full three-model
run ≈ 1.1 h at 4 workers** (one worker per pack state, one state at a time). Add the
`unseen_pi_mixed_sample` set (200 states, one arm) for another ≈ 0.4 h. The
`blackbox_id` / `blackbox_traj` variants (paired structured-vs-black-box) are still not
merged, so that comparison is unavailable.

## Seen/unseen and training separation (fail-closed)

`wms_eval.c1` now refuses a run when the seen and unseen state sets share a world
id, or when an unseen state reuses a training-window `key`/`run`
(`--train-windows PATH`). The guard runs after the sets load and before any
output directory is created, so an invalid comparison never produces a report;
the observed counts are returned and written to `C1.json` (`separation`).

Streaming check `_out/c1_overlap_check.py` (zero simulation; each state JSON is
scanned in 8 MB chunks, never `json.load`):

| unseen | seen | shared worlds | train key overlap | train run overlap |
|---|---|---|---|---|
| `unseen_pi_fleet161.json` (161) | `seen_val.json` (157) | **21** | 0 / 1076 | 0 / 1076 |
| `unseen_pi_mixed_sample.json` (200) | `seen_val.json` (157) | **4** | 0 / 1076 | **18 / 1076** |

The expected count is **0**; these are **not 0**, so the C1 full run is currently
refused by the new guard:

- `seen_val` shares whole worlds with both unseen sets; the unseen/seen ratio and
  its bootstrap interval assume independent world sets, so these pairings
  invalidate the headline ratio.
- `unseen_pi_mixed_sample` contains 18 `__native` runs that are TRAIN windows in
  `windows_train_noPI.json` (e.g. `w8483m2700p02t240__native`), i.e. ordinary
  training states, not unseen. `unseen_pi_fleet161` has zero key/run leak.
- The 6-state `e2e_unseen6.json` / `e2e_seen6.json` proof pair is already
  world-disjoint (0 overlap); the checked-in pytest fixture was not and is fixed
  to use disjoint `u*` / `s*` worlds.

Remediation (executed 2026-10-06, see "C1 v2 clean sets" above): the unseen PI
states were split into U-same-world (140) and U-new-world (21), the mixed
fixed/switching sample was dropped, and `seen` was rebuilt disjoint from every
U-same world. `c1sets.verify` fails closed on any U-same contamination or
training-window leak; the remaining `seen ∩ u_new = 21` is the intended
same-world paired reference for the U-new split, not training leakage.

## Test evidence

```
tests/test_wms_eval_c1.py
# 12 passed (4 new: overlap counts/refusal unit test, unseen-train key/run leak
#            refusal, CLI refuses overlapping worlds, CLI refuses unseen state
#            in training windows; the end-to-end fixture is now world-disjoint)
tests/test_wms_eval_materialize.py + c1/metrics/seen/confinement/wms_plugin
# 27 passed (5 new: synthetic-pack id binding + target_mask slot tracking,
#            degenerate-snapshot fail-closed, selection rules, chunk-prefix resolution,
#            streaming writer)
ruff check <new/changed modules>   # All checks passed
```

## Memory discipline (owner rule)

`materialize` opens every `.npy` with `mmap_mode='r'`, reads one window's
`[tick:tick+H+1]` slice, and writes one record at a time; no whole pack or all-runs
load. `c1` reads one state at a time; each of the ≤4 workers holds one small model and
one state. All Python resident memory stayed far below the 6 GB ceiling.
