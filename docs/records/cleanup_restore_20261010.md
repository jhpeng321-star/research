# Stage 1 clean-up and partial restore — 2026-10-10

Analysis/maintenance record (no simulation, no training).

## What ran (owner, from `tmp/morning/MORNING_CLEANUP.ps1`)
| run (journal, UTC) | scope | removed | freed |
|---|---|---|---|
| 05:42–06:07 `-Stage 1 -Execute` (deep hash) | backup-verified items | 467 | E: 116.0 GB, D: 60.9 GB, C: 10 MB |
| 06:08–06:22 `-Stage 1 -Execute` | finished worktrees in `E:/rd-wt` | 95 | E: 14.1 GB |

Every item was re-verified against its Drive copy (size + MD5) right before removal; journal:
`tmp/morning/journal.log`. Held (not removed): 94 worktrees with never-uploaded sensitive-named files,
protected/live worktrees, keep-local data, dirty or unregistered worktrees.

## Defect found afterwards
The reference scan matched only absolute paths (`X:\...`). It missed relative paths (`output/x/...`),
dotted imports (`from output.x import ...`) and the ledger (`protocols/` was not scanned). Stage 1
therefore removed 351 untracked `output/` folders, of which:
- 36 are still used by active code or tests (e.g. `traffic_shepherd.mechanics` imports
  `output.controller_verification_v8/v10`; planner tests import `output.planner_v1`): 40 test modules
  failed to collect;
- 237 hold files cited by `protocols/experiment_history.json`: `validate` went from 11 to 1095 hard errors.

## Restore (from `gdrive:research-archive/repo-evidence/output`, `rclone check` 0 differences)
- 36 code folders restored whole (8 MB of archives).
- For the other 237 folders, only the 905 ledger-cited members, plus 4 small cited folders restored whole
  (`oe57_w6_review_v1`, `worldmodel_v3_horizon`, `worldmodel_v3_wall_assembly`, `worldmodel_v3_wall_history`).
  The full folders remain on Drive.
- Nothing was overwritten (each target was absent). Staging: `tmp/restore_20261010/`.

## Checks after the restore
- `pytest --co`: 2065 collected, 0 errors (was 40 errors). Full suite: 1860 passed, 202 skipped; the only
  remaining failures are the two zero-hard-error ledger tests (baseline, below). The 4
  `test_evidence_regeneration_v1` errors were fixed by the follow-up restore below (24 passed in the rerun).
- Stage 1 had also removed 11 individual items inside `output/evidence_regeneration_v1` (the folder itself
  is held because it contains protected-name entries). The 10 small ones were restored
  (`regenerate.py`, `CLOSEOUT.md`, `restore_receipts/`, `PLAN*.json`, ...). `dataset_v2_retry1` (1.9 GB) stays on Drive.
- `experiment_history.py validate`: 11 hard errors, the pre-switch baseline: WM-03 `worldmodel_dataset_v1`
  (8, MIG-04) and WM-32/36/37 `numbers_source` in `D:/rd-wt/wm32|wm36|wm37` (3, worktrees removed before
  2026-10-04). The two ledger tests that assert zero hard errors therefore still fail, as before the switch.
- `verify-sources`: 2177 values performed, 0 mismatches, 0 failures (19 skipped = MIG-04 deletions).
- `python -m traffic_shepherd inspect` OK; `wms preflight` passed for `r3_packs_only` and `r3`
  using the new `data_cache/` registry paths.

## Fix to the clean-up tool
`tmp/morning/morning_cleanup.template.ps1` now also scans relative and dotted references, resolved
against the tree root. `gen_cleanup.py` adds `protocols/` to the scanned directories. A report-only
check confirmed the restored folders are now HELD as REFERENCED.

## Stage 2 (copy only, run 06:23–06:41 UTC)
14 keep-local items (75 GiB) copied into `data_cache/` with SHA-256 per file: pack_v3, pack_dspeed,
pack_damping, switching pack, eval states (v2, h60, h60_real), C1FAST checkpoint, lpwo_v3,
sidecars, identities and overlays. `configs/data_registry.toml` now points at `data_cache/`.
