# Switch-over record: E:/research-dense now runs the clean line (2026-10-10)

Executed from `tmp/morning/SWITCH_OVER.md` sections 2-5 (owner-approved, including the guard rename
to `tools/data_guard.py`). Section 6 (data cache, worktree removal) was NOT run: the owner runs deletions.
Nothing was deleted, reset, cleaned, stashed or force-checked-out. `sealed/` (about 62 GB, untracked) was never
staged and was not opened.

## Result

| item | value |
|---|---|
| main tree `E:\research-dense` | branch `wms/clean-final` @ `a9cb11a`, `git status --short` is empty (ignored data only; `/sealed/` now ignored) |
| integration worktree `E:\rd-wt\ws-clean-int` | detached at `1f9a9e8` (the same line without the last commit); branch `wms/clean-int` untouched at `5c304a7` |
| checkpoint of the old dirty tree | branch `checkpoint-pre-clean-20261010` @ `78c8f52` (42 tracked files + 139 untracked files, none under `sealed/`) |
| unit tests | worktree 508 passed / 1 skipped (17m50s); main tree after switch 508 passed / 1 skipped (15m57s); skip = `tomllib` missing on Python 3.10 |
| ruff | `ruff check src tests tools` clean in both trees |
| `python -m traffic_shepherd inspect` | exit 0, model sha 762fde07... equals the expected sha, network and scenario template found |
| `python -m wms --help` | exit 0 (train, preflight, eval, score, fetch) |

Commits on `wms/clean-final` (oldest first): `78c8f52` checkpoint (on the checkpoint branch), `01f00ec` guard rename,
`48018a6` merge `--no-ff`, `1f9a9e8` ignore `/sealed/`, `a9cb11a` ignore the two local `archive/pre_circles` junctions,
and the commit that adds this record.

## Open item (needs an owner decision)

`tools/experiment_history.py validate` in the switched main tree reports **17 hard errors; the pre-switch baseline of the same
tree was 11**. The 6 extra errors are path-resolution only, not missing evidence:

* `WM-S-TRAINSETUP-20261009` and `WALL-C1-PREP-20261009` (clean-line entries) cite `_out/TRAINSETUP_READY.md`,
  `_out/TRAINSETUP_REGISTRATION.json`, `_out/WALL_C1_READY.md`, `_out/lpwo/SUPERSESSION.md`; these now live under
  `archive/wms_era/records/_out/` (listed in `archive/wms_era/PATH_MAP.json`).
* `CV-18` and `CV-20` cite `tests/test_cv18_runner.py`, `tests/test_cv20_runner.py`; relocated by the clean line,
  listed in `archive/tests_dead/PATH_MAP.json`.

The tool's `relocated()` only reads `archive/circles_era/PATH_MAP.json`. Diagnostic (no repo change):
`tmp/switch_run/validate_allmaps.py` runs the unmodified tool with every `archive/*/PATH_MAP.json` loaded and returns exactly
the 11 baseline errors (the file lists are identical, `diff` empty). Proposed fix, NOT applied (it would edit the ledger tool,
the ledger is untouched): in `relocated()` iterate `glob(archive/*/PATH_MAP.json)` instead of the single circles_era path.
The 11 baseline errors are the MIG-04 deletions (WM-03 `output/worldmodel_dataset_v1/*`) and three `D:/rd-wt/...` sources
(WM-32, WM-36, WM-37); they are unchanged.

`verify-sources` in the switched main tree: 33 sources, complete=21, deleted=12, 0 mismatches (identical to the pre-switch
baseline). `summary --check`: CURRENT (ledger sha 7289a3aa4735...).
In the data-less integration worktree the same tools report `source file missing` (validate 1566 hard errors, verify-sources
14 failures); this is environmental (no `runs/`, no local `output/` data there) and is why the main-tree runs are the real check.

## Commands run (in order)

All git commands were `git -C <tree>`; Python is `C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe` (3.10.20).

1. Process check (Win32_Process command lines) for `upload_driver`, `process.ps1`, `research-dense\_out`, `rd-wt`: no match other
   than my own shells. Also present and left alone: `mem_log.ps1`, `mem_guard.ps1` (tmp/overnight), the owner's claude.exe, three idle
   `python.exe -` interpreters, and a report-only `MORNING_CLEANUP.ps1 -Stage 1` (no `-Execute`, read-only; it held `journal.log`,
   so I waited for it to finish before Stage 3). Re-checked before step 3 and before step 5: still none.
2. `powershell -File tmp\morning\MORNING_CLEANUP.ps1 -Stage 3` (read-only): matches SWITCH_OVER section 0 (37 files edited on
   both sides, 28 identical, 9 real conflicts; 29 untracked-also-on-clean, 1 differs: `tools/holdout_guard.py`; 46 ignored files
   overwritable, 21 differ). It rewrote `switch_*.txt` lists (untracked list now 139 incl. `MORNING_REPORT.md`).
   Copy-aside of the 21 ignored files to `tmp/morning/pre_switch_overwrite_backup/` with `cp -p`; `cmp` of all 21: identical.
3. `git switch -c checkpoint-pre-clean-20261010`; `git add -u`; for each of the 139 lines of
   `git ls-files --others --exclude-standard | grep -v '^sealed/'` a single `git add -- <file>` (no wildcards, no `-A`);
   check: status lines other than staged A/M/D/R and `?? sealed/` = none; staged paths under `sealed/` = 0;
   `git commit` -> `78c8f52` (139 A, 41 M, 1 D); `git ls-tree -r HEAD | grep ^sealed/` = empty.
   Size review of the untracked list: only `archive/pre_circles/manifest.json` (53,264,098 bytes, the AGENTS.md checksum manifest)
   exceeds 1 MB; next largest is a 792 KB pptx. Nothing unknown and large was added.
4. In `E:\rd-wt\ws-clean-int` (clean at start): `git switch -c wms/clean-final`; `git mv tools/holdout_guard.py tools/data_guard.py`;
   `tests/unit/data/test_data_layer.py`: `import holdout_guard as hg` -> `import data_guard as hg` and a docstring naming the new path
   (the module docstring had no path to change); `docs/ARCHITECTURE_V4.md` and `ARCHITECTURE_V4_CHANGES.md`: 7 references to
   WP-C's tool renamed to `tools/data_guard.py` (INTEGRATION_REPORT.md left as the historical collision record); commit `01f00ec`.
   `git merge --no-ff --no-commit checkpoint-pre-clean-20261010`: 5 textual conflicts (below), 4 of the 9 predicted conflicts merged
   automatically (EXPERIMENT_HISTORY.md, ROUTE.md, experiment_history.json, test_mixed_assignment.py) and were checked.
   Resolutions applied with small Python scripts; commit `48018a6`. `.gitignore` += `/sealed/` -> `1f9a9e8`.
5. Verification in the worktree: `PYTHONPATH=E:\rd-wt\ws-clean-int\src`, `TMP/TEMP` and `--basetemp` under the worktree (`tmp_verify/`,
   git-ignored): `pytest tests/unit -q` 508 passed / 1 skipped; `ruff check src tests tools` clean; `summary --check` after
   `summary --write`: CURRENT; validate / verify-sources: environmental failures described above. `verify-sources` rewrites
   `protocols/source_verification_report.json` each run, so the committed copy was saved first and restored byte-for-byte after
   every run (worktree and main tree); `git status` was clean afterwards each time.
6. Switch: `git -C E:\rd-wt\ws-clean-int switch --detach`; `git -C E:\research-dense switch wms/clean-final` (no refusal, no
   untracked-overwrite error). `git status --short` showed one entry, `archive/pre_circles/output/`: a local Windows junction
   to `archive/pre_circles/reports/output` (the old `.gitignore` ignored it; the clean line's does not). Added
   `/archive/pre_circles/output/` and `/archive/pre_circles/runs/` to `.gitignore` (commit `a9cb11a`); status is now empty.
   `git diff --ignore-cr-at-eol --stat`: empty. The 21 backed-up ignored files now hold the tracked versions (all 21 differ
   from the backup, as expected; backup intact).
7. In the main tree with `PYTHONPATH=E:/research-dense/src`: `python -m traffic_shepherd inspect`, `python -m wms --help`,
   `pytest tests/unit -q` (508 / 1), `ruff check src tests tools`, `validate`, `verify-sources`, `summary --check` as reported above.

## Conflict resolutions

| file | how |
|---|---|
| `README.md` (1 hunk) | main-tree `## Current status (2026-10-09)` heading and bullets kept; the clean line's six integration/readiness bullets (WM-S second integration ... WM-S2 integrated) kept verbatim as `### Integration and readiness records (clean line, 2026-10-06 to 2026-10-09)`; links to `docs/WM_CHANGE_LOG.md` and `MORNING_REPORT.md` added to the top banner; the `tools/` row of the layout table lists both `data_guard.py` and `holdout_guard.py`. |
| `docs/INDEX.md` (1 hunk) | main-tree INDEX (header, current/superseded tables, historical blocks) kept; the clean line's `WPF-layout` block re-inserted under the banner; row for `../MORNING_REPORT.md` added. |
| `docs/OPEN_QUESTIONS.md` (1 hunk, both sides appended at EOF) | union: `WALL-C1-PREP-20261009` block, then the main tree's WD-01/WM-72/PL-EXP1b/WM-73/owner-decision blocks. |
| `docs/RESULTS_SUMMARY.md` (1 hunk, only the ledger sha line) | regenerated with `experiment_history.py summary --write` from the merged ledger (sha 7289a3aa...); the diff against the main tree's copy is that one line. |
| `protocols/source_verification_report.json` (19 hunks) | main-tree report (generated 2026-10-01T19:10:35, 21 complete / 12 deleted / 0 mismatches) taken wholesale; the clean line's 2026-10-01T07:39 report only recorded sources "missing" because it was produced in a data-less worktree. The earlier version stays in git history. |
| `protocols/experiment_history.json` (auto) | verified as a union: 330 entries = 280 base + 2 clean (`WALL-C1-PREP-20261009`, `WM-S-TRAINSETUP-20261009`) + 48 main; no id lost, no duplicate, no entry modified by both sides; `campaigns` 73, `runs` 475, `correction_log` 47, `migration_log` 11 equal the main tree's. |
| `docs/EXPERIMENT_HISTORY.md`, `docs/ROUTE.md` (auto) | both sides' additions present (ROUTE +86 lines vs the clean line, +27 vs the main tree); a `MORNING_REPORT.md` line was added next to the existing WM_CHANGE_LOG line in ROUTE. |
| `tests/test_mixed_assignment.py` (auto) | result equals the main-tree file; it already contains the clean line's only edit (`law_version: wall-v2.1-closure`) plus the added cases. |
| `tools/holdout_guard.py` | no conflict after the rename: the planner/reconstructor tool keeps this name, WP-C's `DataGuard` is `tools/data_guard.py`. |
| `AGENTS.md`, `tools/experiment_history.py` | main-tree versions (only the main tree changed them). |

## Disclosure

While locating references to the old guard name I ran one recursive `grep -rn holdout_guard .` over the whole integration worktree.
It necessarily opened tracked files whose names contain `inc1`, `holdout` or `cal` (for example under `output/wd01/inc1/`,
`_out/lpwo_v3/`, `archive/wms_era/records/_out/`). I read only the first 2 KB of its output (which showed archive and
`output/wd01/LEDGER_DRAFT_*` matches); the rest was saved by the harness and not read. No file under `sealed/` or any CAL/HOLDOUT/chunk-5
data path was opened, and later searches used `git grep` with explicit pathspecs.

## Rollback

* Main tree back to the old dirty-tree state (everything that was uncommitted is in `78c8f52`):
  `git -C E:\research-dense switch checkpoint-pre-clean-20261010`
  (the old `.gitignore` then ignores the junctions again; `sealed/` is untracked and not ignored on that branch, so never `git add -A` there).
  The 21 ignored `output/...` files that the switch replaced are preserved in `tmp\morning\pre_switch_overwrite_backup\` (copy them back by name).
* Integration worktree back to its branch: `git -C E:\rd-wt\ws-clean-int switch wms/clean-int` (that branch was never modified).
* Abandon the merged line entirely: `git -C E:\rd-wt\ws-clean-int branch -D wms/clean-final` after switching the main tree away from it.
* `output/`, `runs/`, `sealed/`, `tmp/`, `data_cache/` were never touched by git in this procedure.

## Not done (owner)

Section 6: Stage 2 data cache copy, registry re-pointing, `-RemoveOld`, removal of `E:\rd-wt\ws-clean-int` and Stage 1 `-Execute`.
Note for section 6: `git worktree list` still shows 220 registered worktrees.
