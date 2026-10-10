# FEAT3 REPORT V3 — A5 K=192 STAGE 2 (pack_v3 → verify → manifest → upload_v3 → smoke)

Executed 2026-10-01 on branch `zcode/feat3b` at `e0dac62`, cwd
`output/wd01/wm_n1`, per `KFIX_192_RUNBOOK.md` §3 onward. Python
`C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`, `WD_FORBID_D=1` with
the single owner-approved D: root `D:/rd-staging/wmn1_feat`, zero SUMO,
BELOW_NORMAL priority, `--force` never used, no v2/v2c artifact touched.

## 1. Pre-flight (step 0)

- D: free before start: **95.8 GB** (≥ 65 GB gate); shards_v3 present 51.18 GiB
  (54,955,538,467 bytes).
- **2463 / 2463 `DONE.json`** under `shards_v3`.
- Stage-1 success line (logs_v3k192/featurize_stdout.log):
  `featurize-all: runs=2463 written=2463 skipped=0 failed=0 workers=10`.
- `shards_v3/ALLOCATION.json`: `format "v3"`, `k_max 192`, workers 10
  (recorded override), `idea_fast true`,
  `source_override raw_chunk1_p40/SOURCE_OVERRIDE_MAP.json (73 entries)`.

## 2. Per-shard verify (step 1)

`census_kmax_tools/shard_verify_census_v3.py` (new; ruff-clean):

- `features.writer.verify_index` (DONE.json output-sha verification, the
  featurizer's own resume check) over the shard tree:
  **`[verify] ok=True n_runs=2463 errors=0 wall=142.3s`** → **2463/2463 ok**.
- Meta self-description: **all 2463 shards** carry
  `schema wm_n1_features_format_v3/1`, `format "v3"`, `k_max 192`.
- Idea A channels over the shard tree:
  `python -m corpus.idea_a verify --data-root .../shards_v3` →
  **`{"ok": true, "n_runs": 2463, "errors": []}`**.

## 3. SLOT_POOL_CENSUS_V3 (step 1b)

`SLOT_POOL_CENSUS_V3.md` / `.json` (wm_n1; counts only, from shard
`meta.json` drop counters + `veh_mask` occupancy at K_max=192):

| metric | value |
|---|---|
| runs | 2463 (runs with drops: 63) |
| dropped vehicle-ticks @K=192 | **134,708** |
| rostered vehicle-ticks | 377,385,090 |
| measured residual drop share @K=192 | **0.0357%** |
| ticks with a full 192-slot pool | 10,301 (upper-bound indicator set) |

**A5 comparison:** census residual estimate (pooled, 300-run sample,
`KCENSUS_RESULT.md`) = 0.1399% (57,157 / 40,855,639). Measured full-population
residual 0.0357% is **below** the estimate (delta −0.1042 pp): the A5 rule
estimate was conservative; K=192 removes ≥ 99.96% of the K=128 slot drops
(v2c census: 21,368,334 dropped veh-ticks, 6.00%).

## 4. Pack (`pack_v3`, step 2) — detached PID 32000, BelowNormal

```
python -m corpus pack --data-root D:/rd-staging/wmn1_feat/shards_v3 \
  --out D:/rd-staging/wmn1_feat/pack_v3 --workers 6 \
  --allow-d-output-root D:/rd-staging/wmn1_feat
```
logs_pack_v3/pack_stdout.log (stderr empty), wall ≈ 11m20s (22:21:47→22:33:07):
```json
{
 "schema": "wm_n1_corpus_pack/2",
 "n_runs": 2463,
 "n_ticks": 3190623,
 "n_decision": 49260,
 "totals": {"n_runs": 2463, "n_items": 2463, "n_ticks": 3190623,
            "n_decision": 49260, "bytes": 52399623957}
}
```
pack_v3 on disk: 52,401,636,301 bytes (15 packed arrays + `INDEX.json` +
`manifest.json` + `outcomes.parquet` + `SHA256SUMS`).

## 5. Verify ×2 + idea_a (step 3) — both `ok:true`, idempotent

`python -m corpus verify --pack D:/rd-staging/wmn1_feat/pack_v3
--allow-d-output-root D:/rd-staging/wmn1_feat` — pass 1 AND pass 2 (identical):
```json
{"ok": true, "n_runs": 2463, "n_items": 2463, "n_arrays": 15,
 "k_max": 192, "format": "v3", "errors": []}
```

## 6. Upload manifest (step 4)

`python -m corpus manifest --pack .../pack_v3 --out
D:/rd-staging/wmn1_feat/upload_manifest_v3.json --allow-d-output-root ...`:

- schema `wm_n1_upload_manifest/1`, n_runs 2463, **files 19**, **refused 0**,
  splits **2153 TRAIN / 310 CALIBRATION**; zero HOLDOUT/excluded/chunk-5
  entries (split-before-open refusal code; sealed paths never opened).
- On-disk sha256: `3910598990cd821b021a57788743e8a4afb407be1bd054af46ebfe529413ca1d`
  (recorded in `PARTS.json`/`UNPACK.md`; the CLI stdout prints the LF-body
  sha `55675c46…` — the known `write_manifest` newline quirk, same as v2/v2c).

## 7. upload_v3 (step 5) — v2c builder, detached PID 38116, BelowNormal

The runbook's `upload_v2_build.py` auto-locates `upload_manifest.json`, which
in `wmn1_feat/` is the frozen **pack_v2** manifest (n_runs 2445) — a wrong
anchor for v3. Used instead the v2c builder with the explicit manifest
("exactly like upload_v2c" per brief; same deterministic GNU-tar/zstd/parts
contract):
```
python -X utf8 feat1v2c_tools/upload_build_v2c.py \
  --pack D:/rd-staging/wmn1_feat/pack_v3 \
  --out D:/rd-staging/wmn1_feat/upload_v3 \
  --manifest D:/rd-staging/wmn1_feat/upload_manifest_v3.json
```
wall ≈ 14m12s (23:01:19→23:15:31), zstd v1.5.7 level 6 `-T6`, 2 GiB parts,
18 members sorted by name (pack minus `SHA256SUMS`), mtime=0 uid=gid=0 mode
0644; **no stray empty parts** (4/4 parts listed in SHA256SUMS+PARTS.json).
tar stream 7,472,878,351 B; ratio 0.1426 vs 52,401,634,855 member bytes.

| part | bytes | sha256 |
|---|---|---|
| pack_v3.tar.zst.part000 | 2147483648 | `f3c7108a199613149b78516d309c8af1070ef093049dc094cfdc939c3a6d16f0` |
| pack_v3.tar.zst.part001 | 2147483648 | `e112bdffce2a189f316bff7c57de43e9a7cce63c6f754d50456f3a6858be7949` |
| pack_v3.tar.zst.part002 | 2147483648 | `964422be0e8f155de7694119f0204b3f787b3a4cbff81834ca2cf1767a987e07` |
| pack_v3.tar.zst.part003 | 1030427407 | `0323f68ec9886dd47c90900f670dfc3a2c5767fad4db6912048906c7a7270f22` |

Integrity anchors:
- pack_v3 `SHA256SUMS` sha256: `faa13b0b764580844c51a63d160a20beb02fa74d77fe71a5b3cd9f106abd1411`
- upload `SHA256SUMS` sha256: `67d0ff766549869b359b31468e4191822a65c5f97145f4593eace9a8d372a716`
- `PARTS.json` sha256: `4c83e34d95a59592ddbb255d109a33adde0c16800617cf5bce12932cbe735a2d`
- `UNPACK.md` sha256: `ac7a499ea83e9808b1b8143082f13f3d7f123e545ac164ece475dd7db9a21604`
- upload_manifest_v3.json sha256: `3910598990cd821b021a57788743e8a4afb407be1bd054af46ebfe529413ca1d`

Local verification (`feat1v2c_tools/upload_verify_v2c.py`: every part re-hashed
vs `SHA256SUMS`+`PARTS.json`; anchors re-checked; parts concatenated →
`zstd -dc` → tar stream, per-member payload sha256 vs pack `INDEX.json`):
```json
{"ok": true, "parts": 4, "part_bytes_total": 7472878351, "members": 18,
 "expected_members": 18, "member_bytes_total": 52401634855,
 "pack_member_bytes": 52401634855, "errors": []}
```

## 8. Real-path Idea A smoke on pack_v3 (step 6)

`smoke_idea_a_v3.py` (new; ruff-clean) — the `train.idea_a` box CLI path
(same parser/construction as `train.idea_a.main`, TRAIN split only, pack
opened read-only memmap; all writes on E:) with the single smoke knob
`epochs=1, steps_per_epoch=2` (the box CLI exposes no `--steps-per-epoch`;
the config field is the supported mechanism). Out dir
`output/wd01/wm_n1/smoke_idea_a_v3_m1/` (untracked; not committed).

- **k_max resolved: 192** (pack manifest v3; TRAIN items 2153).
- member 1, seed 101, **2 optimizer steps** (losses 1.98591 → 1.09770),
  wall 7.4 s; `TRAIN_RECEIPT.json` + `member_1.pt` + sidecar written.
- Active losses — all 5 lambda weights 1.0 (`residual_nll`, `rollout`,
  `arrival`, `correction_crps`, `response_contrast`).
- Terms computed **on real pack_v3 batches** during the 2 fit steps:
  `arrival`, `residual_nll`; on a **real pack sibling pair** (treatment +
  native reference from the pack): `arrival`, `residual_nll`,
  `response_contrast`.
- **All 5 A3.2 terms active and finite** with the REAL P2/P4/P5 bundle
  (`HdvResidualNet` / `IdeaAArrivalHead` / `CorrectionHead`) — the project's
  all-5 standard (G4): `[arrival, correction_crps, residual_nll,
  response_contrast, rollout]`, finite=True.
- Structural note (frozen code, `batch_from_corpus_items` docstring): the
  v1 packed-corpus batch has no `teacher` trajectory or `summary_raw`, so
  `rollout`/`correction_crps` are unavailable on corpus batches **by
  design**; they are exercised by the canonical-window path above. This is
  pack-format-inherent, not a pack_v3 defect.

## 9. 73-run archive source disclosure (stage 1, carried into pack_v3)

73 chunk1 p40 runs whose raw `physics_steps.jsonl` were missing on disk were
re-sourced from the owner-approved lossless archive
(`raw_chunk1_p40/`, commit 8e6f67f). Evidence (logs_v3k192_extract_console.log):

```
archive sha256 OK: 4166f207ffd134222ed2602a2da3aacceece075c6984de4b3e45aebe20a00826
plan: 73 runs, 292 featurizer-read members (from 639 archived files)
precheck: 73/73 archive physics sha == pack_v2c INDEX input_sha256
extract: wrote 292 members (4.89 GB), 0 already-verified skips
verify: 73/73 physics_steps.jsonl == pack_v2c INDEX input_sha256
```
Recorded in `shards_v3/ALLOCATION.json` (`n_source_override: 73`);
`SOURCE_OVERRIDE_MAP.json` carries the 73 run keys. Everything else
featurized from `E:/research-dense` directly.

## 10. Timing / resources

| step | wall |
|---|---|
| shard verify (2463 DONE sha) | 142.3 s |
| census veh_mask scan (2463) | 10.5 s |
| corpus.idea_a verify (shards) | ok (hash pass over Idea A channels) |
| pack_v3 | ≈ 11m20s |
| corpus verify ×2 | ok (each a full re-hash pass) |
| manifest | < 1 min |
| upload_v3 build | ≈ 14m12s |
| upload_v3 local verify | ok (part re-hash + streaming member check) |
| smoke (2 steps + checks) | 7.4 s fit (+ bundle/window checks) |

D: free after all stages: 40.04 GB. Peak jobs: pack PID 32000, build PID
38116, both BelowNormal, ≤ 6 threads. One aborted launch (own PID 27008,
broken stdout pipe) left a 0-byte `part000` that was removed before the real
build; no other retries, no overwrites, no `--force`.

## 11. Acceptance criteria — status

- [x] 2463/2463 shard verify ok
- [x] pack_v3 n_runs 2463, k_max 192, format v3, schema wm_n1_corpus_pack/2
- [x] verify ×2 `ok:true` (+ `corpus.idea_a verify` `ok:true` on shards)
- [x] upload_manifest_v3.json (TRAIN/CAL only, refused 0) + upload_v3 parts +
      SHA256SUMS + PARTS.json locally verified (stream zstd -dc | tar, 18/18)
- [x] pack_v3 real-path smoke: 2 steps, TRAIN only, k_max 192, all 5 loss
      terms active (see §8 for the corpus-path structural note)
- [x] SLOT_POOL_CENSUS_V3 (.md/.json), residual 0.0357% vs A5 estimate 0.14%
- [x] FEAT3_REPORT_V3.md committed on zcode/feat3b
