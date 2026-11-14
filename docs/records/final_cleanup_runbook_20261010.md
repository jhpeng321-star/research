# Final clean-up run - owner commands in order (prepared 2026-10-10)

Tool: `tools/dataops/cleanup/` (see its `README.md`).  Generated script: `data_ops/cleanup_out/MORNING_CLEANUP.ps1` (report-only unless `-Execute`).
Tracked copy of this file: `docs/records/final_cleanup_runbook_20261010.md`.  Reports behind the numbers (all report-only, nothing was executed):
`REPORT_stage2_removeold_20261010.txt` (22:17), `REPORT_stage1_today_no_steps_1_2_20261010.txt` (22:30, state of today), `REPORT_stage1_projected_after_steps_1_2_20261010.txt`
(22:37, with the simulation flags `-AssumeSealedMoved` = step 1 done and `-AssumeStage2bDone` = step 2 done), journal of those runs: `journal_agent_report_runs_20261010.log`
(a fresh `journal.log` starts with your first run; `tools\offload_map.py` reads only `EXEC DELETED` rows with a `gdrive:` location).
**j4bfix** (a registered worktree whose receipt and list are sensitive-named) was not evaluated by the agent; your Stage 1 run evaluates it, unlinks its 4 junctions
and moves its untracked sensitive-named files to `sealed\relocated_20261010\E\rd-wt\j4bfix\` first.

Every command is PowerShell, run from `E:\research-dense`:

```powershell
$PY = 'C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe'
$CL = 'E:\research-dense\data_ops\cleanup_out\MORNING_CLEANUP.ps1'      # run as:  powershell -NoProfile -ExecutionPolicy Bypass -File $CL <flags>
```
Sizes are GiB.

## 0. Pre-flight

1. No agent session, uploader or training job may use `E:\rd-wt\*` (a running process whose command line names a worktree makes the script hold it: `IN_USE`).
2. Regenerate and self-test:
   ```powershell
   & $PY -I tools\dataops\cleanup\gen_cleanup.py            # prints: Stage 1 items 599, worktrees (receipts, dir present) 139 (carried 1), dirs without receipt 0
   powershell -NoProfile -ExecutionPolicy Bypass -File $CL -SelfTest      # expect: SELF TEST: 45 passed, 0 failed
   ```
   Verified by the two printed lines; the self-test works on a scratch folder under `data_ops\cleanup_out` and touches no research file.

## 1. Move sensitive-named files into `sealed\` (frees nothing; protects data)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File E:\research-dense\tmp\consolidate\MOVE_TO_SEALED.ps1            # report: plan rows 1528, to move 1525 (7.68 GiB, 23 cross-drive), held 3
powershell -NoProfile -ExecutionPolicy Bypass -File E:\research-dense\tmp\consolidate\MOVE_TO_SEALED.ps1 -Execute   # type MOVE
```
* Frees nothing (same-volume renames on E:; the 23 D: files are copied, SHA-256 checked, then the D: original is removed).
* Held by the script itself: 2 `IN_USE` data_cache files and the worktree root `E:\rd-wt\j4bfix` (step 3 handles it).
* Verified by: `DONE: moved 1525, failed 0`; `Select-String E:\research-dense\sealed\relocated_20261010\MOVE_JOURNAL.tsv -Pattern FAILED` finds nothing; `sealed\relocated_20261010\MOVE_MANIFEST.csv`
  has 1525 `MOVED` rows (step 3 reads this file as evidence that a worktree's recorded sensitive-named files really went to `sealed\`; it never opens the files).
* Known gap: the move plan has no `heldout` rule, so 441 pytest-output files named `*HELDOUT*` stay behind in `ws-clean-c` (61), `ws-clean-int` (130), `ws-main2` (244), `ws-r3-evalstates` (2),
  `ws-r3-score` (4).  Step 3 runs with `-RelocateSensitive`, which moves leftovers like these into `sealed\relocated_20261010\` itself (opaque, SHA-256 before/after, never overwrites).

## 2. Stage 2 - old pack copies (the registry already points at `data_cache\`)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $CL -Stage 2 -RemoveOld                  # report: read the "WOULD remove old copy" lines
powershell -NoProfile -ExecutionPolicy Bypass -File $CL -Stage 2 -RemoveOld -Execute         # type COPY, then REMOVE-OLD once per item
```
* Frees (report 22:17): **D: 48.80** (`D:\rd-staging\wmn1_feat\pack_v3`) and **E: 18.35** (`pack_dspeed` 11.88, `pack_damping` 4.85, `wallc1_states_h60_real` 0.66, `wallc1_states_h60` 0.48,
  `wallc1_states_v2` 0.46, `sidecars_packv3_v2` 0.02, overlays/identities ~0.01).  Nothing is copied: all 14 cache copies are already `CACHED`.
* Drive proof per item, re-checked live: `pack_v3` = the `training_pack_v3_k192` tar.zst parts match `SHA256SUMS` and all 19 archived members' SHA-256 equal the cache copy; packs/sidecars/overlays = per-file size+MD5
  under `worktrees/<wt>/_out/...`; the three `wallc1_states_*` = every file is a member of its worktree's verified tar.zst and the cache SHA-256 equals the archived SHA-256 (`wt_tar` proof).
* Held: `r3_results` (7.13 GiB, `runs\active\WM-S-R3-20261010\box\gpuA\wmn1_switching_train`): no Drive archive of `runs/active/WM-S-R3-20261010` yet; `ckpt_c1fast_dir` and `lpwo_v3` are `keep` by design.
* Verified by: `journal.log` rows `REMOVED_OLD`; every `data_cache\<name>\CACHE_RECEIPT.json` has `complete: true`; the printed freed GiB per item.

## 3. Stage 1 - items and ALL worktrees

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $CL -Stage 1 -RelocateSensitive -AllowNestedGit            # report (6-10 min: live Drive listing, one git scan per worktree)
powershell -NoProfile -ExecutionPolicy Bypass -File $CL -Stage 1 -Execute -RelocateSensitive -AllowNestedGit   # type DELETE; each item is re-verified on Drive right before its removal
```
Flags: `-RelocateSensitive` (leftover sensitive-named untracked files of a retiring worktree go to `sealed\` instead of blocking it; always the case for `j4bfix`), `-AllowNestedGit` (releases `bundle3`, whose nested
`rented_bundle\.git` history is regenerable).  Optional: `-ScanSensitiveNamedFiles` (also scan `*holdout*`-style code files for path references; skipped by default so that agents never open them - with it `pl34` is held,
because `tests\planner\test_holdout_guard.py` names `E:\rd-wt\pl34\output`); `-Skip 'e:\research-dense\'` leaves the in-repo `runs\` / `output\` items alone.

Projected result after steps 1-2 (report 22:37; items below 50 MB are in the totals):

| location | removed | GiB | notes |
|---|---:|---:|---|
| `E:\rd-wt` worktrees (137; `j4bfix` is evaluated only in your run) | 137 | 60.74 | 18.35 of it is the old packs already freed in step 2, so step 3 frees ~42.4 here; 12 worktrees hold 168 junctions/symlinks that are unlinked first (link only) |
| `D:\rd-staging` | 10 | 56.84 | `shards_v3\chunk1-4`, `walls_pilot_chunk4`, `truearms_chunk2/3`, `chunk1_fix27` (51.2), `raw_chunk1_p40` 4.55, `cpu_payload` 1.12 |
| `E:\research-dense\runs` (in repo) | 1 | 66.68 | `campaign_stage1_chunk3` (Drive-only, your decision 2) |
| `E:\research-dense\output` (in repo) | 9 | 14.59 | `evidence_regeneration_v1` 14.5 (its holdout-named folder is moved in step 1) |
| `E:\rented_bundle_dspeed`, `E:\research-github` | 2 | 0.19 | |
| **Stage 1 total** | **159** | **199.06** | steps 2+3 together free about **247.9** (199.06 + 48.80 for `pack_v3` on D:; the 18.35 on E: are inside the worktree figure) |

Free space after steps 2+3 (from 90.5 on D: and 73.0 on E: today): about **D: 196**, **E: 215**.
For comparison, Stage 1 as it stands today (no step 1/2): 16 items (122.4) + 21 worktrees (6.6) eligible; 109 worktrees wait for step 1 (`SENSITIVE_SKIPPED`), 6 for step 2 (`KEEP_LOCAL`), `bundle3` for `-AllowNestedGit`.

How a worktree is retired, immediately before `git worktree remove --force` (git on this machine has `core.symlinks=false` and would otherwise walk into a junction and delete the target's data):
a fresh no-follow walk lists every reparse point -> each is unlinked **link only** (`RemoveDirectory` / `DeleteFile`) and journaled with its target -> a target that existed right before the unlink must still exist right after
-> a final walk must find none.  Order (dependents first): `ws-s2` -> `ws-p2` and `ws-p2data` -> `damping-snap`, `dspeed-data`, `a9`; `pl30` (112 links into `runs\campaign_stage1_chunk1_fix27` / `..._verify27`), `ws-cl`, `ws-eval`,
`ws-torchphys`, `ws-p4`, `ws-r3-jam`, `ws-main`, `ws-main2`, `ws-clean-int` need no ordering.  A worktree that stays and links into an eligible one holds it back (`LINKED_FROM_HELD`).
`j4bfix`: 4 junctions into `output\campaign_pilot_v1`, `controller_verification_v2`, `_v4`, `_v7` are unlinked; ~20 untracked sensitive-named files go to `sealed\relocated_20261010\E\rd-wt\j4bfix\...`; the other 296 files are re-verified on Drive.

Verified by (after the `-Execute`):
```powershell
$j = 'E:\research-dense\data_ops\cleanup_out\journal.log'
(Select-String $j -Pattern "`tDELETED`t").Count                                                 # items + worktrees removed (expect 159 + j4bfix)
Select-String $j -Pattern "`tERROR`t|CHANGED_AT_DELETE|SKIP_AT_DELETE|HELD_LAST|LINKED_FROM"    # should be empty, or explained in the console output
Select-String $j -Pattern "`tUNLINKED`t" | Where-Object { $_ -notmatch 'verified present afterwards|already absent' }   # must be empty
Test-Path E:\research-dense\runs\campaign_stage1_chunk1_fix27                                   # junction targets untouched (True if it existed before)
git -C E:\research-dense worktree list
```
The script then runs `tools\offload_map.py update` (appends the removed paths to `protocols\offloaded_paths.json`; commit that file).  If a worktree ends in `ERROR: git worktree remove failed` (a process holds a handle),
close the process and run `git -C E:\research-dense worktree remove --force <path>` by hand: its junctions are already unlinked and everything in it is on Drive / in git.

## 4. Prune git's worktree registry

```powershell
git -C E:\research-dense worktree prune
git -C E:\research-dense worktree list        # expect: E:/research-dense + the 3 C:\Users\jpeng\.codex\worktrees\... entries (reorg3 is not a worktree)
```
Branches are never deleted (all `wms/*`, `wmn1/*`, `wip-preserve/*` refs stay in the object store).

## 5. Manual leftovers (the script does not touch these)

| item | size | why not automatic | exact commands / verification |
|---|---:|---|---|
| `E:\rd-wt\reorg3` | 0.11 | holds the lists/manifests every check reads (so it goes last); 981 files are not in its tar.zst (897 are copied into the repo, 37 are residue, the rest junk / sensitive-named); 12 nested git repos whose history is not on Drive | `& $PY -I tools\dataops\cleanup\verify_reorg3_retirable.py --bundle-dir E:\tmp\reorg3_nested_git_bundles` (prints the 37 residue files - mostly `layout\_in` / `reviewer\_in` review inputs, logs, `.gitattributes`, and `dataops\receipts\output_state_calibration_v1.json`, which was not copied to `data_ops\receipts` because of its name; writes the 12 bundles) -> `rclone copy E:\tmp\reorg3_nested_git_bundles gdrive:research-archive/repo-evidence/reorg3_nested_git_bundles` -> keep what you want from the residue -> `cmd /c dir /AL /S E:\rd-wt\reorg3` (no junctions) -> `Remove-Item -LiteralPath E:\rd-wt\reorg3 -Recurse -Force` |
| `E:\rd-wt$w\_ext` | 0 | stale junction -> `E:\rd-wt\ws-p2\_ext` (dangling after step 3) | `cmd /c rmdir 'E:\rd-wt$w\_ext'` (link only), then `Remove-Item -LiteralPath 'E:\rd-wt$w'` |
| `E:\tmp` | 0.35 | not in the clean-up allow-list; 84k files, 9 junctions, 99 sensitive-named; `E:\tmp\wmn1_train_runs_20261002\...\identities` is the target of `ws-p2`'s `_ext\identities` junction | after step 3: review, `cmd /c rmdir` each junction, delete |
| `E:\research-dense-backups` | 2.03 | referenced by `tools\experiment_history.py:101` (a path-remap root, not a data consumer) | keep, or add `experiment_history` to `IGNORE_REF_FILES` in `gen_cleanup.py`, regenerate, re-run step 3 (the 15 ledger-cited files are then kept) |
| `D:\rd-staging\wmn1_feat` rest | ~0.02 | `shards_v3` top-level `ALLOCATION.json`, `INDEX.json`, `PLAN_MAP.json`, `SHA256SUMS`, `opened_paths.json` (cited individually by `tools\data_registry.json`); 4 `labels_*` sidecars (`BACKUP_ONLY`; `-IncludeBackupOnly`) | keep, or delete by hand |
| `C:\Users\jpeng\.codex\worktrees\wmn1-*` (3 worktrees of this repo) | 0.57 | no wtclean receipt (their untracked content is not on Drive); tracked state clean, commits are on `codex/wmn1-eval-bridge` / `codex/wmn1-j4-identity` | `git -C <wt> status --short --ignored`, then `git worktree remove <wt>` (no `--force`) if you are satisfied |
| `r3_results` duplicate | 7.13 | no Drive archive of `runs\active\WM-S-R3-20261010` yet | upload it, re-run step 2 |
| in-repo `runs\` / `output\` items still referenced | ~54 | e.g. `runs\campaign_stage1_chunk4` is cited by code (`REFERENCED`) | your call |

Close-out: `& $PY tools\experiment_history.py validate` (baseline 11, nothing new), `verify-sources` (0 mismatches), commit `protocols/offloaded_paths.json`.

## Decisions for you

1. `-AllowNestedGit` (bundle3) and the reorg3 nested histories (bundle them first, step 5).
2. Scan sensitive-named code files for path references or not (`-ScanSensitiveNamedFiles`; then `pl34` is held).
3. Whether the in-repo removals `runs\campaign_stage1_chunk3` (66.68) and `output\evidence_regeneration_v1` (14.5) belong in this run (`-Skip 'e:\research-dense\'` excludes them).
4. `E:\research-dense-backups` (2 GiB) and the 3 Codex worktrees.
5. The move plan lacks a `heldout` rule (step 1 note); `-RelocateSensitive` covers it - say if you prefer to extend and re-run the plan instead.
