# REAL-DATA STEPS — execution record (2026-10-09)

CPU only; no training, no SUMO. Authorized roots: `E:/rd-wt/ws-wallc1`
(wms/wallc1), `E:/rd-wt/ws-brakefix` (wms/brakefix), permitted TRAIN material
under `E:/research-dense`. CAL/HOLDOUT/sealed/chunk-5/J4B/INC1 were never opened.
Interpreter `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`. Every python
process stayed far below 6 GB (largest sampled peak 324 MB). Packs opened only as
`mmap_mode='r'`; one window at a time.

## Part 1 — WALL-C1 LPWO (branch wms/wallc1)

### Staging (coordinator, workspace-local)
- `_out/wallc1_inputs/canonical_exclusions.json` + `canonical_guard.py` copied
  from `E:/research-dense/protocols/holdout_exclusions.json` and
  `tools/holdout_guard.py`.
  - spec sha256 `04337b7ade64f40f1e8184fbac9082aae36cb5d3862ec813d66a535a77d210e6`
  - tool sha256 `45ca391838cbce22305808747dd38ab3c4b34c1204a98f190ee8c8a9c17d8299`
  - 13 excluded worlds; 0 guard refusals; workspace-only enforcement on.
- pack INDEX + manifest copied for `pack_v3` (from `D:/rd-staging/wmn1_feat/pack_v3`,
  byte-identical to the `E:` junction copy), `pack_dspeed`, `pack_damping`.
- per-run TRAIN metadata (`switch_schedule.json`, `branch_record.json`,
  `wall_events.jsonl`) copied for 316 of the 589 distinct pack_v3 runs in the LPO
  lists (632 files); the remaining 273 runs have no recoverable schedule
  metadata. See `_out/WALL_C1_STAGING.json`.

### Split build (`python -m lpwo`, metadata only, fail-closed)
`_out/lpwo_verified_20261009/BUILD_RECEIPT.json` (validate: **ok**, 0 problems).

| quantity | value |
|---|---:|
| windows before (train) | **1307** |
| windows before (val) | **189** |
| windows after (train) | **140** |
| windows after (val) | **21** |
| removed windows | **1335** (train 1167, val 168) |
| excluded run keys | **2636** (1564 inherited PI + 1072 added wall) |

Policy counts before → after:

| split | policy | before | after |
|---|---|---:|---:|
| train | damping | 358 | 0 |
| train | native | 254 | **140** |
| train | trajectory | 560 | 0 |
| train | wall | 135 | 0 |
| val | damping | 52 | 0 |
| val | native | 33 | **21** |
| val | trajectory | 84 | 0 |
| val | wall | 20 | 0 |

Removed by reason: `unverifiable_schedule_missing` **1180**,
`wall_engaged` **130**, `wall_owned_not_assembled` **16**,
`wall_label_not_engaged` **9**. Fail-closed schedule removals include every
`pack_dspeed`/`pack_damping` window (their INDEX rows carry no `root_rel`) and
every pack_v3 run without a recoverable schedule. Result: **native-only** lists.
The wall-corpus total is 155 runs; 140 train / 21 val retained worlds.

### Pilot bindings
`_out/wallc1_pilot_bindings.json` — **155** explicit
`campaign_walls_pilot_chunk4` wall bindings (freshly guarded; no discovery).

### Test-set selection sizes (metadata level; pack arrays required to materialise)
`_out/WALL_C1_SELECTION.json`:

| set | candidates | notes |
|---|---:|---|
| `wall_same` (removed train-origin in retained worlds) | 1089 | 332 pack_v3 / 560 pack_dspeed / 197 pack_damping |
| `wall_same` with a verified timeline | **135** | engaged 114, owned-not-assembled 13, label-not-engaged 8 |
| `wall_pilot` requested | 155 | assembled_released 82, aborted_in_hold 48, aborted_in_assembly 16, not_feasible 9 |
| `wall_pilot` **engaged** | **130** | 114 of them overlap retained training worlds (also wall_same) |
| `seen_controls` candidates | **39** | pack_v3 |

### Materialisation: BLOCKED (environmental, not governance)
The guarded materialiser (`lpwo.materialize_wall`) requires every source under
`WORKSPACE = E:/rd-wt/ws-wallc1`. The selected windows need all three packs:
`pack_v3` (~50 GB) + `pack_dspeed` (~12 GB) + `pack_damping` (~5 GB) ≈ **67 GB**.
- `E:` (workspace volume) has **6.19 GB free** → a real copy cannot fit.
- The pack sources live on `D:` (`E:/rd-wt/ws-cl/_ext/pack_v3` is a **junction**
  to `D:/rd-staging/wmn1_feat/pack_v3`), so NTFS hard-link staging fails
  cross-volume (`WinError 17`, "cannot move file to a different disk drive").
- Directory junctions/`resolve()`-following paths are forbidden by the guard
  (they resolve outside `WORKSPACE`), and the READY note forbids junction
  bypasses. No `resolve()`-safe way exists to expose the packs under `WORKSPACE`.

Consequently **0 wall_same / 0 wall_pilot / 0 seen_controls state files were
materialised**, and no big state file exists to hash. Staging scripts
(`_out/stage_wallc1_inputs.py`, `_out/stage_wallc1_packs.py`) and
`_out/analyze_wallc1_selection.py` are preserved.

**Smallest user action to finish:** place the three approved packs (or their
arrays hard-linked on the workspace volume) under
`E:/rd-wt/ws-wallc1/_out/wallc1_inputs/{pack_v3,pack_dspeed,pack_damping}` with
the matching identity sidecars, then re-run the materialize command in
`_out/WALL_C1_READY.md` (new `--out-dir`). Alternatively authorize a
guard-allowed root on the volume holding the packs.

## Part 2 — BRAKEFIX real audit (branch wms/brakefix)

Root cause of `real_audit_01` `ValueError: array metadata mismatch: veh_kin`: the
audit string-equated the manifest `dtype` token; `pack_v3` spells it `"float16"`
while `pack_dspeed`/`pack_damping` spell it `"<class 'numpy.float16'>"`. The
trainer's `MemmapCorpus._array` never compares the token. Fixed by normalising
via `np.dtype` (`_manifest_dtype`), fail-closed on unknown tokens. Regression test
added. Commit **`8cca2f3`** "Fix brakefix audit real-pack dtype-token validation"
(2 files, +57/-1).

Tests: `pytest test_wms_brakefix_audit.py test_wms_brakefix.py` → **72 passed**;
ruff clean.

Audit re-run tag `real_audit_02` → `_out/BRAKEFIX_REAL_AUDIT_02.json`, status
passed (rc 0, 156 s, peak working set 324 MB). 60 windows
(pack_v3 24 / pack_dspeed 25 / pack_damping 11), **563,649** HDV steps.
Acceleration-channel alignment: `channel_vs_speed_fd_correlation = 0.99992`,
mean |difference| 0.00742 m/s² (max 0.0593); the module still labels the
destination-channel alignment *assumed*, not verified.

Recorded HDV braking shares and reachability (state-dependent interval share of
braking steps; global envelope in brackets):

| setting | lower envelope m/s² | share a<-1.5 | share a<-3 |
|---|---:|---:|---:|
| residual / no cap | -9.0 | 0.9952 | 0.9939 |
| **residual / cap ±1.5 (run-3)** | -9.0 | **0.6260** | **0.6297** |
| driver_params / defaults (physics decel clamp) | -4.2 | 0.0 [0.9478] | 0.0 [0.7981] |
| hybrid / cap 1.5 | -5.7 | 0.6696 | 0.7558 |
| **hybrid / asym +1.5 / -3.0 (proposed)** | -7.2 | **0.8402** | **0.8622** |
| hybrid / +1.5/-4.5 | -8.7 | 0.8918 | 0.8924 |

Recorded braking distribution: `a<-1.5` 4.52 %, `a<-2` 3.12 %, `a<-3` 1.17 %,
`a<-3.5` 0.62 %, `a<-4` 0.31 %, `a<-4.5` 0.037 %; min recorded decel
-5.3477 m/s². **Projection is not measured** by this audit (it constrains
post-step speed/lane/position, not raw acceleration) → reported `null`, not
inferred. Negative-correction deficit shares (e.g. cap 3, a<-3) are recorded in
the JSON. No checkpoint/driver inference and no recorded-outcome skill claim.

## Integrity notes
- Guard enforced TRAIN-only, workspace-only sources, 13 excluded worlds and the
  CAL/HOLDOUT/sealed/chunk-5/J4B/INC1 denylist; refusal count 0 (no forbidden
  identity was even requested).
- Source packs and run directories were read-only; nothing under
  `E:/research-dense/runs/` was modified.
- Committed on `wms/wallc1`: the small LPWO lists/receipts, pilot bindings,
  staging and selection receipts (see `output/wd01/wm_n1/lpwo/real_20261009/`).
