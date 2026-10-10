# Worktree backup summary (wtclean)

Generated 2026-10-10 09:22:21 by tools/summarize.py (read-only over inventory/receipts).

## Counts (inventory snapshot of 224 worktree-like dirs; 13 active excluded)

| state | worktrees |
|---|---|
| verified, tar.zst (this takeover) | 72 |
| verified, no untracked content to back up | 0 |
| verified, per-file mirror (earlier uploader, status ok) | 134 |
| per-file mirror ok but commit step reported failure (upload+check exit 0; accepted by template) | 5 |
| failed / not passed | 0 |
| pending (no receipt) | 0 |
| eligible total | 211 |

tar.zst volume: 61052 members, 36.60 GiB source -> 8.53 GiB archives (4.3x); upload 0.56 MB/s per archive (sum of upload seconds 16475 s). 0.88 MB/s aggregate archive bytes over the 173 min between first and last tar receipt (72 archives)

Pending/failed workload (inventory sizes): 0 worktrees, 0 files, 0.00 GiB.

## Parked failed tar attempts (earlier receipt kept)

- rc05: 1 attempt(s); last: error local verify failed ok_tar_t=True ok_sha=False listed=361 seen=361 bad=[]

## Skips recorded in tar.zst receipts

- sensitive-named files skipped (names only, never opened/uploaded): 45 worktrees, 300 files total
- worktrees with junctions/symlinks (not followed): pl30(112), ws-eval(7), ws-main(6)
- tracked-dirty sensitive-named paths left uncommitted: none

## Active / not touched

- inventory active: reorg3, ws-cl, ws-main2, ws-p2, ws-r3base, ws-r3-evalstates, ws-r3-hist, ws-r3-int, ws-r3-jam, ws-r3-resid, ws-r3-score, ws-r3-stage, ws-wallc1
- not in inventory (created later, treated ACTIVE): ws-attn, ws-clean, ws-clean-a, ws-clean-b, ws-clean-c, ws-clean-d, ws-clean-e, ws-clean-f, ws-clean-int, ws-golden, ws-r3-evalstates2, ws-r3-evalstates3, ws-r3-score2

## Schema notes for E:/research-dense/tmp/morning/gen_cleanup.py

- tar receipts keep every legacy field (name,path,class,manual,branch,headBefore,headType,dirtyBefore,committed,commitSha,preserveRef,dest,listFile,uploadCount,uploadBytes,upload{ran,exit,log},check{ran,exit,result,log},junctions,skippedSensitiveCount/Sample,skippedJunctionCount,status,error,timestamp).
- added: method='tar.zst', archive{remote,size,md5,sha256,manifest_*,member_count,member_bytes,verified{..}}, memberCount, memberBytes, skippedSensitive (full list), junctionsFound, nestedRepoDirs.
- dest = gdrive:research-archive/worktrees/<name>; files on Drive: <name>.tar.zst and <name>.manifest.json (member path,size,sha256). listFile = lists/<name>.tar.txt (archived member paths).
- The template's per-file Drive proof (dest/<relpath> size+md5) does NOT apply to method='tar.zst' receipts: it must check the archive (size+md5 in receipt.archive) instead; see tools/patch_proposal/*.patch (unapplied proposal).

## Takeover notes (static, written by the 06:15-09:30 takeover)

- Method: tools/pack_wt.py = commit tracked changes (non-sensitive paths) -> git ls-files --others [+ --ignored] (junction/symlink/sensitive filtered) -> tar(PAX)|zstd -T0 -10 with per-member sha256 -> local verify (zstd -dc | tar -t == manifest; streamed sha256 == manifest) -> rclone copy (archive + manifest) -> rclone md5sum + lsjson size check -> receipt. tools/run_queue.py runs <=3 in parallel; tools/verify_remote.py is an independent Drive round-trip (rclone cat | zstd -dc | sha256 per member vs manifest).
- Upload chunk size: 256M for archives <= 64 MB, 32M above (accurate progress for the stall watchdog); a Clash upstream node sometimes pins a connection to ~0.03-0.4 MB/s, so a watchdog kills OUR rclone and retries on a new connection.
- The earlier per-file uploader (powershell tools/process.ps1, PID 40236, started 23:00) was NOT killed; it keeps walking the same list. It collided on a few items (wm53, pl21, rc05, ws-review2, ws-swgen, ws-ranking, ws-eval, j4bfix...): when it finishes after a tar.zst receipt it overwrites it with a legacy per-file receipt (both are valid backups on Drive). Prior receipts are copied to receipts_prev/, failed tar attempts are parked in receipts_failed/.
- Keep-local per-file mirrors (generator dir_md5 proofs): done for sidecars_packv3_v2, identities_dspeed, identities_damping, labels_overlay_dspeed, labels_overlay_damping (tools/mirror_keeplocal.py). The two big packs (pack_dspeed 12 GB, pack_damping 5 GB) were started with tools/mirror_pack.py but STOPPED at ~9%/14% to give bandwidth to gpu1; they are inside the verified dspeed-data/damping-snap tar.zst archives. Re-run: python -I tools/mirror_pack.py dspeed-data _out/pack_dspeed 7 ; python -I tools/mirror_pack.py damping-snap _out/pack_damping 5.
- tools/git_precheck.py (read-only) found, among OLD per-file receipts: pl53 and pl57 have tracked changes again after their backup (still in use?); mcpclaude/mcpdrive/mcpopencode branches do not exist in the main repo (other repo).
