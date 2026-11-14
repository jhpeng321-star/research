# Clean-up tool (`tools/dataops/cleanup/`)

Retires local copies that are verified on Google Drive: backed-up data directories, the `E:\rd-wt\*` git
worktrees, and the keep-local hot data that moved into `data_cache/`.  It is **report-only by default**; nothing is
deleted, moved or copied without `-Execute` (typed confirmation `DELETE` / `COPY`).  Moved here on 2026-10-10 from
`tmp/morning/` (which is left intact); outputs now live in the git-ignored `data_ops/cleanup_out/`.

| file | role |
|---|---|
| `gen_cleanup.py` | reads the receipts, writes `data_ops/cleanup_out/{MORNING_CLEANUP.ps1, cleanup_manifest.json, gen_state.json}` (no research file is touched) |
| `morning_cleanup.template.ps1` | the PowerShell 5.1 script; `gen_cleanup.py` embeds the manifest into it |
| `verify_reorg3_retirable.py` | read-only check before `E:\rd-wt\reorg3` is removed by hand (which files are on Drive / copied into the repo / residue; `--bundle-dir` writes `git bundle`s of its 12 nested repos) |
| `README.md` | this file |

Runbook for the final run: `docs/records/final_cleanup_runbook_20261010.md` (copy: `data_ops/cleanup_out/FINAL_RUN.md`).

## Regenerate

```powershell
C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe -I E:\research-dense\tools\dataops\cleanup\gen_cleanup.py
```

* receipts are read from `data_ops/receipts/{dataops,driveops,wtclean}/` (tracked); the heavy generated parts they point
  to (member manifests, upload lists, `E:\rd-wt\reorg3\...`) are read from `--heavy-root` (default `E:\rd-wt\reorg3`);
* `--reorg3-receipts` reads the receipts from `E:\rd-wt\reorg3` instead (comparison with the original location);
  `--receipts-root PATH`, `--out PATH`, `--protect NAME` (repeatable), `--carry-from MANIFEST`, `--pending`, `--summary`.
* **worktree candidates = every `E:\rd-wt\<name>` that has an `ok` (or `commit_failed` with clean exits) receipt in
  `wtclean/receipts/`**; `inventory.json` is no longer used.  The protected list (`PROTECTED_WORKTREES` in
  `gen_cleanup.py`, `--protect`) is empty: the owner decided that all worktrees retire.  Receipts whose file name is
  sensitive-named (`j4bfix.json`) are never opened by the generator; their worktree entry is carried over from the
  previous manifest (`CARRY_WORKTREES`).
* directories under `E:\rd-wt` without a receipt are listed as `NO_RECEIPT` (never removed).

## Run

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File E:\research-dense\data_ops\cleanup_out\MORNING_CLEANUP.ps1 -SelfTest
powershell -NoProfile -ExecutionPolicy Bypass -File E:\research-dense\data_ops\cleanup_out\MORNING_CLEANUP.ps1 -Stage 1          # report
powershell -NoProfile -ExecutionPolicy Bypass -File E:\research-dense\data_ops\cleanup_out\MORNING_CLEANUP.ps1 -Stage 1 -Execute # acts
```

Every decision (also in report mode) is appended to `data_ops/cleanup_out/journal.log`; reports are
`report_stage<N>_<stamp>.json`; removed paths go to `removed_paths.jsonl`, and after an `-Execute` Stage 1 the script runs
`tools/offload_map.py update <journal>` so the experiment ledger reports "offloaded to Drive" instead of "missing".

* **Stage 1** removes items and worktrees whose Drive copy is re-verified live (size + MD5) - details below.
* **Stage 2** copies the keep-local hot data into `data_cache/` (per-file SHA-256); `-RemoveOld` removes the old copy only
  if the cache copy re-verifies, a Drive copy verifies (`dir_md5`, `tar_parts`, `archive_cover`, `dir_listing`,
  `git_tracked`, or `wt_tar` = "every file is a member of the verified worktree tar.zst and the cache SHA-256 equals the
  archived SHA-256") and no active code references the old path.
* **Stage 3** is obsolete (the single-work-tree switch is done) and only prints a notice.

## How a worktree is retired (Stage 1)

All of these must hold, the first failing check is the printed reason (status):

1. receipt `ok`; not protected; not in use by a running process (command-line match); no active **code** reference
   (records - `protocols/*.json`, `docs/records/*`, `tools/data_registry.json` - only cite; the ledger resolves offloaded
   paths through `protocols/offloaded_paths.json`); not a keep-local source (Stage 2 `-RemoveOld` first).
2. registered git worktree; branch exists and HEAD is contained in it (detached HEAD: contained in some ref); no uncommitted
   tracked change **unless preserved**: the tracked state equals the receipt's `preserveRef`/`commitSha`, or every dirty file
   is archived verbatim (`__wtclean_tracked_dirty__/`) in the verified tar.zst and unchanged.
3. every untracked/ignored file is in the uploaded list (Drive archive/objects re-verified live by size + MD5) - files
   below a junction are not the worktree's and are ignored (git with `core.symlinks=false` descends into junctions).
4. sensitive-named untracked files (`holdout|heldout|cal_split|chunk-?5|j4b|inc1|sealed|cal` token) must already be in
   `sealed\relocated_20261010\` (`tmp/consolidate/MOVE_TO_SEALED.ps1`); if the receipt says some were skipped and none is
   left, each recorded name must have a `MOVED` row in `sealed\relocated_20261010\MOVE_MANIFEST.csv`
   (`SENSITIVE_UNACCOUNTED` otherwise).  Worktrees whose own root is sensitive-named (`j4bfix`) - or any worktree with
   `-RelocateSensitive` - get their leftover untracked sensitive-named files moved there by this script itself right before
   the removal (opaque move: SHA-256 before/after, no overwrite, `MOVE_MANIFEST.csv` row + journal line).
5. plain directories (not git worktrees) with a nested `.git` are held (`NESTED_GIT`) unless `-AllowNestedGit`.

**Junctions / symlinks** (`Unlink-AllLinks`): immediately before `git worktree remove --force`, a fresh no-follow walk
lists *every* reparse point in the worktree; each is unlinked **link only** (`RemoveDirectory` for directory
junctions/symlinks, `DeleteFile` for file symlinks - nothing below a link is touched), journaled
(`UNLINKED <link> ... target <T>; target verified present afterwards | target was already absent (dangling link)`), and
a target that existed right before the unlink must still exist right after it.  A final walk must find no reparse
point, otherwise the worktree is not removed.

**Order** (`Resolve-WorktreeOrder`): a worktree whose junctions point into another worktree is removed *before* that
tree (`ws-s2 -> ws-p2 -> damping-snap/dspeed-data/a9`); a worktree that stays and links into an eligible tree holds it
back (`LINKED_FROM_HELD`); cycles are held (`LINK_CYCLE`).  `reorg3` (it holds the lists/manifests every check reads)
is held until `-RetireReorg` and then removed last, only if `E:\rd-wt` holds nothing else.

## Report-only simulations

`-AssumeSealedMoved <SENSITIVE_MOVE_PLAN.csv>` treats the sensitive-named files that the plan covers as already moved;
`-AssumeStage2bDone` treats the keep-local old copies as already removed.  Both refuse `-Execute`.
`-ExplainRefs <path>` lists every code reference / record citation that names a path.  `-Only a,b` / `-Skip a,b` filter by
substring.  `-SelfTest` exercises the primitives (walker, no-follow delete, junction unlink, dependency order, sealed
relocation, tracked-dirty rule, Stage 2 copy/verify/remove) on a scratch folder under `data_ops/cleanup_out`.

## Flags worth knowing (Stage 1)

| flag | effect |
|---|---|
| `-RelocateSensitive` | leftover sensitive-named untracked files of a retiring worktree are moved (not deleted) to `sealed\relocated_20261010\` instead of blocking it |
| `-AllowNestedGit` | releases plain directories with a nested `.git` (their history is not backed up) |
| `-ScanSensitiveNamedFiles` | the reference scan also reads sensitive-named code files (default: skipped, so agents never open them) |
| `-RetireReorg` | allows `E:\rd-wt\reorg3` (last, only if `E:\rd-wt` holds nothing else, and only if its own checks pass - see `verify_reorg3_retirable.py`) |
| `-AllowSensitiveDup`, `-RemoveAroundProtected`, `-ReleaseProtectedNames`, `-IncludeBackupOnly` | older partial-release switches, see the script header |
| `-Only` / `-Skip` | substring filters on item id / path / worktree name (`-Skip 'e:\research-dense\'` leaves the in-repo items alone) |

## Known quirks

* PowerShell variables are case-insensitive: never assign a local `$m` (it shadows the manifest `$M`); the template uses `$mf`.
* A report-only Stage 1 takes about 10 minutes (live Drive listing + one `git ls-files` per worktree).
* The generated script is ~0.8 MB because the manifest is embedded; Windows PowerShell 5.1 `ConvertFrom-Json` handles it
  (the generator warns above 1.9 MB).
