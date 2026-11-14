**Coordinator read-only update (2026-10-02):** later logs supersede the historical stopped statuses below. Batches2-5 have local mounted-Drive copy/cache hash receipts (last12:46); batch6 is partial through shards_v2 chunk2 at13:50. No uploader was running at15:18. Server-side checksums and independent download verification remain unestablished. The batch1 backup copy log predates its rebuilt archive and needs reconciling. No new upload/deletion is authorized or performed by this update. See `tmp/zcode_review/COORDINATOR_CLOUD_STRUCTURE_REVIEW_20261002.json`.

# Cloud offload plan (Google Drive): uploads first, deletions only on the owner's direct OK

2026-10-02, organize session. Owner goal (relayed): all data uploaded to the cloud and deleted locally, so it can later be used from rented RunPod GPUs.

- **Cloud root:** `G:/我的云端硬盘/research-dense-cold/` (Google Drive for desktop; its cache is `D:/Google_drive_cache`).
- **Batch 1 is already there:** the repo bundle, a worktree snapshot, the backups, and the chunk-1 p40 and dataset_v1 train archives, 3.5 GB. See `tmp/gdrive_stage_b1`.

**No local deletion happens under this plan without the owner's approval given directly in the organize session.**

## (a) Inventory, local need, offload order

Sizes are unique bytes (hard links counted once), measured 2026-10-02.

| tree | size | needed locally in the next ~5 days? | order |
|---|---|---|---|
| **KEEP LOCAL (until told otherwise)** | | | |
| repo code, active worktrees (E:/rd-wt/*, incl. v3own, v3arch, gpu1, wd01, pl*, j4bfix, wmn1integC) | ≈26 GB | yes | never offloaded by this plan |
| D:/rd-staging/wmn1_feat/{pack_v3 50.0, upload_v3 7.1, shards_v3 52.5} | 109.6 GB | yes: training until verified | after training is verified (owner) |
| D:/rd-staging/cpu_payload | 1.2 GB | yes: CPU box | — |
| runs/campaign_walls_j4b (J4B raw, held-out (i) featurization) | 10.3 GB | yes | after the (i) eval |
| runs/active | 39.0 GB | yes: live | — |
| runs/campaign_walls_pilot_chunk4 | 11.3 GB | probably (J4) | ask |
| runs/campaign_stage1_chunk2 / chunk3 / chunk4 (+ chunk2/3_truearms) | 69.8 / 68.3 / 53.1 (+ 8.1 / 8.2) GB | likely: re-featurization, INC1, PL-WM1 sources | ask the WM/planner sessions which they read |
| runs/plexp1*_oracle_v1 | 3.6 GB | maybe: PL-WM1 | ask |
| **COLD: upload now** | | | |
| runs/_archives/{campaign_stage1_chunk5, campaign_jpulse}.tar.zst (already lossless-archived; chunk5 SEALED, bytes only) | 3.8 GB | no | 1 |
| runs/campaign_stage1_chunk1 (remaining: 70 bases + non-p40 branches + 14 HOLDOUT-world branches, bytes only) | 47.2 GB | no (pre-fix speedMode 31; superseded by fix27 for p40) | 2 |
| runs/campaign_stage1_chunk1_fix27 + campaign_stage1_verify27 | 25.1 + 2.5 GB | no | 3 |
| runs/{fresh_validation_v1_r3 3.8, regenerated_v1 3.7, regenerated_v1_retry1 0.3, campaign_pilot_v1_r3/r4 0.8, campaign_rep_pilot_v1 0.15, smoke dirs} | ≈8.8 GB | no | 4 |
| output/worldmodel_dataset_v2 (= evidence_regeneration_v1, hard-linked) | 23.0 GB | no (old WM corpus) | 5 |
| output/worldmodel_dataset_v1 (data/holdout 6.0 GB, bytes only; train already archived) | 6.3 GB | no | 5 |
| D:/rd-staging/wmn1_feat/{shards_v2 35.5, upload_v2 6.5, upload_v2c 6.6, raw_chunk1_p40 4.7, quarantine 0.1} | 53.4 GB | no (superseded by v3) | 6 |
| C: leftovers (Temp: pytest-of-jpeng 7.1, opencode 3.4) | 10.5 GB | scratch | not uploaded (regenerable); delete separately |

COLD total ≈ 168 GB raw. Run data compresses about 10× (measured: chunk1 p40 21.6→0.70, jpulse 14.8→1.49, chunk5 23.4→2.33 GB), so roughly **15–25 GB to upload**.

## (b) Method

1. **Archive per tree:** `tar | zstd -6 -T2` into a local staging file. Sizes:
   - chunk1 rest: about 2–5 GB, one archive per tree;
   - the D: shards: already binary or compressed, so expect little gain; split them per sub-folder.

   Each archive gets a manifest with:
   - a sha256 per member, recorded while packing;
   - the archive's sha256 and **MD5** (Drive stores an MD5 server-side, see (c)).

   Sealed and HOLDOUT data: bytes are hashed and packed only, never parsed or inspected. The originals stay untouched.
2. **Copy to G:** in **≤ 10 GB batches** into `research-dense-cold/batch<N>_<date>/`.
   - Before each batch: D: must keep at least 15 GB free after the batch lands in the Drive cache.
   - After each batch: wait until the Drive app has finished uploading (see (c)) before starting the next.
3. **Integrity on G:** right after copying, re-read the hashes through G:. This only verifies the bytes in the local Drive cache, so it is not cloud proof.

## (c) Verification before ANY local deletion

The local Drive cache can serve reads even when the cloud copy is missing or incomplete, so cache reads are never accepted as proof. Required, in order:

1. **Drive app idle.** The tray status shows "All files up to date", with no pending uploads; the owner checks this. In addition, machine-wide upload traffic stays near 0 for 10 minutes (`Get-NetAdapterStatistics` sent-byte deltas, logged).
2. **Server-side checksum.** This is the decisive check. The owner signs in with rclone once (`rclone config`, Drive remote; the token stays in the owner's rclone config, never in project files). Then:
   - `rclone md5sum gdrive:research-dense-cold/batch<N>_<date>/` prints the MD5s **Google computed on its servers**;
   - they must equal the local MD5s recorded in each manifest.

   This proves the cloud copy itself, without any download.
3. **Spot download.** Download one archive per batch from the cloud with `rclone copy` into a temp folder off the Drive cache, and check its sha256 and a full `zstd -t` integrity test.
4. Only after 1–3: the owner approves the deletion **directly in the organize session**. Deletion is per tree, after a final listing and a check for directory junctions; a pointer note stays at each original path.

The browser pane is not signed in to Google, and no sign-in was attempted. So tonight's status is "uploaded, local cache verified, cloud checksums pending owner sign-in".

## (d) Ledger handling

- **Pointer note at each offloaded path:** `<tree>.OFFLOADED.md`, with the cloud path, the archive sha256 and MD5, the manifest, and restore instructions.
- **Ledger entry:** one MIG-06-style `migration_log` entry per batch ("offloaded to cloud": tree, archive, shas, cloud path, verification evidence), appended with the exact JSON round-trip formatting.
- **Close-out after each batch:** run `summary --write`, then `summary --check`, `validate` and `verify-sources`, which must stay at 0 hard errors and 0 mismatches.
- **Ledger-cited source paths:** `verify-sources` hashes some registered source files (29 sources, 11 recorded as deleted). Any tree containing a registered source path must stay local, or be recorded in `protocols/deleted_evidence.json` with its cloud location before deletion, following the MIG-04/05 precedent. This check runs per tree before the owner is asked to approve its deletion.

## (e) Time estimate

Measured speed: see `tmp/cloud_offload/UPLOAD_SPEED.log` (filled tonight from the network sent-byte deltas during batch 2). The estimate is: COLD compressed (~15–25 GB) ÷ the measured MB/s. At 2–5 MB/s (a typical home uplink) that is about 1–3.5 hours for the run data. The D:/rd-staging v2 shards (≈53 GB, barely compressible) take about 3–7.5 more hours.

## (f) Later retrieval (RunPod)

- The owner configures rclone with their own Google sign-in on the pod (`rclone config` → drive; or `rclone authorize` on a local machine and paste the token into the pod's rclone config). No tokens are kept in project files.
- Fetch: `rclone copy gdrive:research-dense-cold/batch<N>_<date>/<tree>.tar.zst /workspace/data/ --progress`.
- Unpack: `zstd -d -c <tree>.tar.zst | tar -x -C /workspace/data/`.
- Check: compare against the manifest sha256 with `python - <manifest>` (script in the batch folder).
- For training, prefer copying the training packs (`pack_v3` / `upload_v3`) to the pod's volume once and keeping them there across runs.

## Status 2026-10-02 ~03:30: uploads NOT started

The organize session's permission check refused the copy to Google Drive as data exfiltration, because the instruction was relayed and not given by the owner in that session. Nothing was uploaded by the organize session.

To proceed, the owner either:
- approves the uploads directly in the organize session (or adds a permission rule); or
- runs these commands. They upload only: originals stay untouched, and each batch waits for Drive to finish uploading.

```
cd E:/research-dense/tmp/cloud_offload
python cloud_upload.py batch2_20261002 copy:E:/research-dense/runs/_archives/campaign_stage1_chunk5.tar.zst copy:E:/research-dense/runs/_archives/campaign_jpulse.tar.zst archive:E:/research-dense/runs/campaign_stage1_chunk1
python cloud_upload.py batch3_20261002 archive:E:/research-dense/runs/campaign_stage1_chunk1_fix27 archive:E:/research-dense/runs/campaign_stage1_verify27
python cloud_upload.py batch4_20261002 archive:E:/research-dense/runs/fresh_validation_v1_r3 archive:E:/research-dense/runs/regenerated_v1 archive:E:/research-dense/runs/regenerated_v1_retry1 archive:E:/research-dense/runs/campaign_pilot_v1_r3 archive:E:/research-dense/runs/campaign_pilot_v1_r4 archive:E:/research-dense/runs/campaign_rep_pilot_v1
python cloud_upload.py batch5_20261002 archive:E:/research-dense/output/worldmodel_dataset_v2 archive:E:/research-dense/output/worldmodel_dataset_v1
python cloud_upload.py batch6_20261002 archive:D:/rd-staging/wmn1_feat/upload_v2 archive:D:/rd-staging/wmn1_feat/upload_v2c archive:D:/rd-staging/wmn1_feat/raw_chunk1_p40 archive:D:/rd-staging/wmn1_feat/shards_v2
```

Logs are written to `tmp/cloud_offload/UPLOAD_LOG.txt` and `UPLOAD_SPEED.log`, with one `<batch>.json` per batch. Then run the verification in (c) (rclone md5sum after your sign-in) before any deletion request.

## Status 2026-10-02 11:05: batch 2 partially uploaded, then stopped

Owner approved the uploads directly in the organize session (2026-10-02).
- `batch2_20261002/` on G: holds `campaign_stage1_chunk5.tar.zst` (md5 3212293f577efc4c9880e4b9f2c7e9c3) and `campaign_jpulse.tar.zst` (md5 15c752bdc1e5d2b0444bf58ca0c05642). The G: cache hash is OK for both; the cloud md5 is not yet checked.
- The first run then crashed on chunk1 because `os.path.isjunction` needs Python 3.12. The uploader now has a reparse-point fallback, the per-folder naming `archive:<path>@<name>` / `toplevel:<dir>@<name>`, and a size guard that aborts on any archive over 10 GB.
- The resume (`tmp/cloud_offload/run_chain.sh`, from batch-2 chunk1 onward, with shards_v2 split per sub-folder) was refused by the session's auto-mode permission check. It needs a Bash permission rule, or the owner runs `bash E:/research-dense/tmp/cloud_offload/run_chain.sh` directly.

No local data was deleted.

## Coordinator follow-up: unexplained partial batch6 stop (2026-10-02)

Read-only inspection found GoogleDriveFS still running, no uploader/python/bash process, no batch6 completion receipt. chain_stdout.log contains the earlier 11:00 Python isjunction crash; current cloud_upload.py replaces it with is_reparse and later batches resumed, so that old exception does not establish the latest stop cause. Current upload helper still unlinks staging archives and uses overwrite compression; do not restart unchanged under current no-deletion rule. Coordinator has not restarted uploads or deleted anything. Remote Google-server checksums/independent spot-download proof remain missing.

## Direct owner-approved resume (2026-10-02, coordinator)

Owner directly in the executing chat: "I approve. please start the upload". Allocation CLOUD-RESUME-20261002-01 was registered before execution. The retained staged chunk3 archive passed opaque zstd integrity testing and was copied to a new batch6_resume_20261002T0758Z folder; cache SHA256 and MD5 match. Machine-wide outgoing traffic was 2.8 MiB/s during a 15 s sample. Drive server checksums and independent download remain pending. All source/staging files are retained. Remaining cold folders wait for independent source review of a fresh-output, no-deletion helper in E:/rd-wt/cloudresume. Earlier no-authorization/no-restart status paragraphs are historical. No deletion or account authentication is authorized by this resume.

Owner-approved remaining queue started 2026-10-02 16:23 China time: reviewed preserve-only helper `a05f3236`, PID6020. Four remaining cold folders and the rebuilt opaque backup are queued; initial ten-minute quiet wait observed.11 standalone synthetic checks plus a tiny real zstd/Drive-cache copy passed. All originals/staging retained; server proof pending. Allocation `CLOUD-RESUME-20261002-01`, remaining receipts `tmp/cloud_offload/resume_remaining_20261002T0826Z/receipts.jsonl`.


## Coordinator local completion and archive guide (2026-10-02)

The preserve-only remaining uploader completed at 08:59:44 UTC (16:59 China). Five remaining archives total 4,315,870,968 bytes; with the earlier resumed chunk3, six new archives total 5,587,422,331 bytes. Mounted-cache SHA256/MD5 checks and the final ten-minute quiet heuristic passed. This is local completion, not Google-server checksum or independent-download proof. All originals and staging files remain. Completion receipt: `tmp/cloud_offload/resume_remaining_20261002T0826Z/completed.json`.

Owner-requested organization added two fresh root indexes to the signed-in Drive mount: `00_START_HERE_20261002.txt` and `00_ARCHIVE_CATALOG_20261002.json`. The catalog lists 63 visible outer files using names/sizes and already-known receipt hashes; it opens no archive members or outcome data. Local copies and receipt: `tmp/cloud_offload/archive_index_20261002/`. Existing archives were not moved or deleted. The proposed shared local Drive MCP remains under source review; it cannot itself establish Google-server verification or repair the ChatGPT plugin installation.

2026-10-02 Google server metadata verification: all 63 cataloged outer files match names and sizes, including the six resumed archives (5,587,422,331 bytes), and both root operator indexes are present. Receipt: `tmp/cloud_offload/DRIVE_SERVER_METADATA_20261002.json`. The connected Drive MCP omits requested MD5/SHA256 fields, so server content hashes remain unverified. No archive content was downloaded; all originals and staging remain retained.

Owner-requested Drive organization completed under CLOUD-ORGANIZE-20261002-01: six top-level categories, 24 folders, 63 existing files verified after metadata-only moves, nine original batch/probe folders and two first indexes preserved under upload records, two current root guides uploaded. Current guide: https://drive.google.com/file/d/1MEJfC2Y25_NB3n1PjTTBdk3f7mCog06U/view?usp=drivesdk. Receipt: `tmp/cloud_offload/CLOUD_ORGANIZE_20261002_01_VERIFIED.json`. Historical receipt strings remain unchanged; the R2 catalog records old/new locations and stable file IDs. Current member-1 release is still local and unapproved. No archive reads, replacements or deletions.
