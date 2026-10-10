# DELETE_SUMMARY.md - owner-facing backup & local-deletion summary

Generated 2026-10-09 by the `driveops` agent (workspace `E:\rd-wt\reorg3\driveops`).
Drive tool: `C:\Users\jpeng\miniconda3\bin\rclone.exe`, remote `gdrive`
(`rclone.conf` was never opened). All Drive targets below use the CURRENT
post-reorganisation paths from `MOVE_WITHIN_DRIVE.json`
(`gdrive:research-archive/...`). Report-only; **nothing was deleted locally,
nothing was deleted on Drive**.

## How to run the owner-run deletion

Report-only (safe, default; prints what would be removed and 0 bytes deleted):

```
powershell -NoProfile -ExecutionPolicy Bypass -File E:\rd-wt\reorg3\driveops\DELETE_VERIFIED_LOCAL.ps1
```

To actually delete the eligible items (re-verifies local + Drive first, journals
every step, prints the true bytes removed on the final line):

```
powershell -NoProfile -ExecutionPolicy Bypass -File E:\rd-wt\reorg3\driveops\DELETE_VERIFIED_LOCAL.ps1 -Execute
```

The script reads `receipts\*.json` (schema `driveops_receipt/1`, `check = passed`,
class `DELETE-CANDIDATE`/`DRIVE-ONLY`), rejects wildcards, refuses excluded or
link-bearing trees, re-checks local MD5 against `receipts\md5\*.md5`, re-checks
the Drive copy with `rclone check --one-way --checksum` (or `md5sum`+size for
loose files), appends to `journal\delete_<utcstamp>.jsonl`, and removes without
following reparse points.

## Freeable now (report-only preview)

| Drive | Eligible items | Bytes | GiB |
|------:|---:|---:|---:|
| D: | 17 (see table) | 203,958,220 | 0.190 |
| E: | 0 | 0 | 0 |

Current free: `C:` 67.83 GiB, `D:` 33.67 GiB, `E:` 42.85 GiB.

| Local path | Class | Bytes | Drive copy (current path) |
|---|---|---:|---|
| `D:\rd-staging\box_push_j4b_bases` | DELETE-CANDIDATE | 8,184,711 | research-archive/data/staging/rd-staging/box_push_j4b_bases |
| `D:\rd-staging\wmn1_feat\code_ship_02a2d8c.tar` | DRIVE-ONLY | 7,946,240 | research-archive/data/staging/rd-staging/code_ship |
| `D:\rd-staging\wmn1_feat\code_ship_aec0d8d.tar` | DRIVE-ONLY | 7,946,240 | research-archive/data/staging/rd-staging/code_ship |
| `D:\rd-staging\wmn1_feat\code_ship_d4f716b.tar` | DRIVE-ONLY | 7,946,240 | research-archive/data/staging/rd-staging/code_ship |
| `D:\rd-staging\wmn1_feat\logs` | DELETE-CANDIDATE | 3,379 | research-archive/data/staging/rd-staging/featurization_logs/logs |
| `D:\rd-staging\wmn1_feat\logs_v2` | DELETE-CANDIDATE | 681 | research-archive/data/staging/rd-staging/featurization_logs/logs_v2 |
| `D:\rd-staging\wmn1_feat\logs_v3` | DELETE-CANDIDATE | 109,047 | research-archive/data/staging/rd-staging/featurization_logs/logs_v3 |
| `D:\rd-staging\wmn1_feat\logs_v4` | DELETE-CANDIDATE | 418,057 | research-archive/data/staging/rd-staging/featurization_logs/logs_v4 |
| `D:\rd-staging\wmn1_feat\logs_pack_v3` | DELETE-CANDIDATE | 1,807 | research-archive/data/staging/rd-staging/featurization_logs/logs_pack_v3 |
| `D:\rd-staging\wmn1_feat\logs_v3k192` | DELETE-CANDIDATE | 451,441 | research-archive/data/staging/rd-staging/featurization_logs/logs_v3k192 |
| `D:\rd-staging\wmn1_feat\pack_v2c_meta` | DELETE-CANDIDATE | 3,763,835 | research-archive/data/staging/rd-staging/pack_v2c_meta |
| `D:\rd-staging\plexp1d_v3_root` | DELETE-CANDIDATE | 868,462 | research-archive/data/staging/rd-staging/plexp1d_v3/root |
| `D:\rd-staging\plexp1d_v3_slot.tar` | DELETE-CANDIDATE | 33,269,760 | research-archive/data/staging/rd-staging/plexp1d_v3/slots |
| `D:\rd-staging\plexp1d_v3_slot_attempt2.tar` | DELETE-CANDIDATE | 33,259,520 | research-archive/data/staging/rd-staging/plexp1d_v3/slots |
| `D:\rd-staging\plexp1d_v3_slot_attempt3.tar` | DELETE-CANDIDATE | 33,259,520 | research-archive/data/staging/rd-staging/plexp1d_v3/slots |
| `D:\rd-staging\plexp1d_v3_slot_attempt4.tar` | DELETE-CANDIDATE | 33,259,520 | research-archive/data/staging/rd-staging/plexp1d_v3/slots |
| `D:\rd-staging\plexp1d_v3_slot_attempt5.tar` | DELETE-CANDIDATE | 33,269,760 | research-archive/data/staging/rd-staging/plexp1d_v3/slots |

## What stays, and why

Backed up but deliberately **not** offered for deletion:

- `E:\rented_bundle_dspeed` (ARCHIVE, receipt passed) - keep the only copy.
- `E:\research-github` (ARCHIVE, receipt passed) - keep the only copy.
- `E:\rented_bundle`, `E:\glm`, `E:\glm-worker`, `E:\migration_logs`
  (ARCHIVE, receipts passed) - keep.
- `D:\rd-staging\wmn1_feat\labels_*` (KEEP-ACTIVE-backup-only, receipt passed) -
  backup only; LIVE-referenced, never delete.

Backed up (receipt passed) but blocked from this script by the brief's rules:

- `E:\research-dense\output\partB_bundles` - **excluded**: inside the live repo
  `E:\research-dense`.
- `E:\research-dense-c-migrated` - **referenced by active code/config** (2 files:
  `E:\research-dense\tmp\disk_c_sonnet_20261004\move_c_research.ps1`,
  `E:\research-dense\tmp\disk_cleanup_20261005\part1\tools\build_deliverables.ps1`).
- `D:\rd-staging\wmn1_feat\shards` (v1) - **referenced by active code/config**
  (15 files, e.g. `ws-r3-stage\output\wd01\wm_n1\trainstage_tools\*`).
- `D:\rd-staging\wmn1_feat\logs_v5`, `logs_v2c`, `quarantine_v2` - **referenced by
  active code/config** (55 / 22 / 11 files). The other 6 `logs*` dirs are
  eligible; these three stay pending reference cleanup.

Excluded by the brief regardless of receipts: live packs
`D:\rd-staging\wmn1_feat\pack_v3`, `pack_dspeed`, `pack_damping` and the real
data dirs; live worktrees `E:\rd-wt\ws-*`; the repo `E:\research-dense`;
`E:\rd-wt\reorg3`; and any `sealed` / CAL / HOLDOUT / chunk-5 / INC1 content.
`DELETE_VERIFIED_LOCAL.ps1` hard-codes all of these refusals.

## Failed / not-backed-up items

No receipt in `receipts\` is failing. STEP 1 re-uploads and re-verification:

- `rented_bundle_dspeed` - first `rclone check` failed (incomplete copy); re-run
  now `check=passed (2561 files, 175,120,615 bytes)`.
- `research-github` - first `rclone copy` timed out at 10,800 s with only
  3,422/5,408 objects; resumed copy completed, then `check=passed (5408 files,
  33,515,736 bytes)`.

Not yet backed up (no passing `driveops_receipt/1` in this folder):

- `D:\rd-staging\wmn1_feat\shards_v2\{chunk3,chunk4,truearms_chunk2,truearms_chunk3,walls_pilot_chunk4}`
  (~21 GiB) - Drive copies are TRUNCATED (2026-10-06 record); too large to
  re-upload in this window. Keep local.
- `D:\rd-staging\wmn1_feat\{pack_v3,shards_v3}` + `D:\rd-staging\cpu_payload`
  (~101 GiB) - KEEP-ACTIVE/live-referenced; not candidates.
- The older 2026-10-06 Drive-verified D: archives (`upload_v2`, `upload_v2c`,
  `raw_chunk1_p40`, and 4 `shards_v2` subfolders) live under the **old**
  `research-dense-cold` layout; they were not re-verified against the current
  `research-archive` paths here, so they have no receipt in this folder.
- `E:\research-dense\tmp\*` DELETE-CANDIDATE scratch (~0.38 GiB) - excluded repo;
  regenerable, not worth a Drive copy. Same for `E:\rd-wt\reorg2_*` /
  `reorg_inv_*` scout scratch (~0.09 GiB) and `E:\rd-release\venv-test` (~0.9 GiB).
- `C:\Users\jpeng\AppData\Local\Temp\claude` (~4.6 GiB) and `pip-unpack-*`
  (~0.13 GiB) - regenerable agent/pip temp; delete locally without a backup.

## Notes / caveats

- The script never deletes a remote Drive object and never deletes an item whose
  Drive copy does not currently re-verify.
- The eligible bytes above are the sum of the receipt-verified local sizes; the
  `-Execute` run prints the bytes **actually** removed.
- Report-only journals produced while validating this script are in `journal\`.
