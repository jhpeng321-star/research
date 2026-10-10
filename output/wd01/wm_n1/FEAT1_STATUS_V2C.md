# FEAT1_STATUS_V2C — COMPLETE (2026-10-01)

Corpus pack_v2c built and verified on branch `zcode/feat1v2c` (from
`zcode/feat1j4fix` @ `e100fa6`). Full detail: `FEAT1_REPORT_V2C.md`.

## Status: COMPLETE — no resume needed

| step | result |
|---|---|
| 1. featurize the 18 J4 wall runs into shards_v2 (`--idea-fast`, workers 6, absence asserted first) | `written=18 skipped=0 failed=0` |
| 2. count assertion | `2476 - 13 = 2463`; verified dirs 2463/2463; INDEX 2463 (`D:/rd-staging/wmn1_feat/logs_v2c/assertion_v2c.json`) |
| 2b. idea_a verify (shard tree) | `ok: true, n_runs: 2463` |
| 3. pack_v2c + corpus verify | 2463 runs, 3,190,623 ticks, 35,309,431,491 bytes; `ok: true` |
| 3b. manifest | `D:/rd-staging/wmn1_feat/upload_manifest_v2c.json` (2153 TRAIN / 310 CAL, excluded 0) |
| 4. upload_v2c build + local verify | 4 parts (~2 GiB), `ok: true`, 18 members byte-identical |
| 5. slot-pool census | `SLOT_POOL_CENSUS.md` + `.json` (21,368,334 dropped vehicle-ticks, 6.00% of rostered) |
| 6. report | `FEAT1_REPORT_V2C.md` (committed) |

## Artifacts (D:/rd-staging/wmn1_feat — approved root only)

- `shards_v2/` (+18 wall-run shards; PLAN_MAP 2463; INDEX 2463)
- `pack_v2c/` (NEW), `upload_manifest_v2c.json`, `upload_v2c/` (NEW:
  4 parts + SHA256SUMS + PARTS.json + UNPACK.md)
- `logs_v2c/` (subset parquet, snapshots, assertion receipt)

Untouched: `shards/` (old tree), `pack_v2`, `upload_v2` (observed
interrupted rebuild by another process ~14:19–14:21 local — not ours,
not touched), `D:/research-dense`.

## Resume commands (only if something above is ever redone)

```powershell
cd E:\rd-wt\feat1v2c\output\wd01\wm_n1
$env:WD_FORBID_D="1"
& C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -X utf8 -m corpus featurize-all --inventory D:/rd-staging/wmn1_feat/logs_v2c/sub_j4_18.parquet --data-root E:/research-dense --out D:/rd-staging/wmn1_feat/shards_v2 --workers 6 --idea-fast --min-free-ram-gb 8 --min-free-virtual-gb 10 --allow-d-output-root D:/rd-staging/wmn1_feat
& C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -X utf8 feat1v2c_tools/verify_count_v2c.py
& C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -X utf8 -m corpus pack --data-root D:/rd-staging/wmn1_feat/shards_v2 --out D:/rd-staging/wmn1_feat/pack_v2c --workers 6 --allow-d-output-root D:/rd-staging/wmn1_feat
& C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -X utf8 -m corpus manifest --pack D:/rd-staging/wmn1_feat/pack_v2c --out D:/rd-staging/wmn1_feat/upload_manifest_v2c.json --allow-d-output-root D:/rd-staging/wmn1_feat
& C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -X utf8 feat1v2c_tools/upload_build_v2c.py --pack D:/rd-staging/wmn1_feat/pack_v2c --out D:/rd-staging/wmn1_feat/upload_v2c --manifest D:/rd-staging/wmn1_feat/upload_manifest_v2c.json
& C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -X utf8 feat1v2c_tools/upload_verify_v2c.py --pack D:/rd-staging/wmn1_feat/pack_v2c --out D:/rd-staging/wmn1_feat/upload_v2c --manifest D:/rd-staging/wmn1_feat/upload_manifest_v2c.json
& C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -X utf8 feat1v2c_tools/slot_pool_census.py
```

## Known non-blockers

- Ledger `validate` has 1560 pre-existing hard errors (MIG-04-deleted
  `output/...` paths) and `verify-sources` 11 pre-existing source-missing
  failures — pre-date v2c; no new entries added (integrating agent owns
  the shared ledger close-out).
- `upload_manifest_v2c.json`: CLI stdout sha (LF body, `aa36eca1…`) vs
  on-disk-bytes sha (`b2efda91…`, recorded in PARTS.json) — pre-existing
  `corpus.manifest.write_manifest` newline quirk, identical convention to
  upload_v2.
