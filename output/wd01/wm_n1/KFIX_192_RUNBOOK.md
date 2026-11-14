# KFIX_192_RUNBOOK — A5 option K=192 re-featurization (PREPARED, NOT RUN)

**Status:** prepared 2026-10-01, branch `zcode/kfix192`, amendment 5 §4 option
"K = 192" (`REGISTRATION_WM_N1_AMENDMENT_5_KMAX_RULE.md`).  Every command
below is **staged only** — nothing here has been executed and nothing has been
written to `D:`.  Execute only after the owner authorizes the A5 K=192 rule
result (§5 of the amendment).  If the A5 census rule picks **B (cap at 128)**
instead, discard this prep: the code is format-parameterised and the default
v2 path is untouched (byte-reproducible), so B needs no code change.

Conventions for the executing task: python
`C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`, cwd
`E:\rd-wt\kfix192\output\wd01\wm_n1`, `WD_FORBID_D=1` with the single
owner-approved D: root `D:/rd-staging/wmn1_feat` (never `D:/research-dense`),
zero SUMO, BELOW_NORMAL priority, `--idea-fast` (byte-identical fast Idea A
baseline), `--workers 6` (the hard maximum; no override), no `--force`,
nothing on the v2/v2c side (`shards_v2`, `pack_v2`, `pack_v2c`, `upload_v2`,
`upload_v2c`) is read for writing, touched, overwritten or deleted.

## 0. Disk estimate (provision before starting)

Anchors: FEAT1 v2 pack = 35.31 GiB measured (`FEAT1_REPORT_V2C.md`), K-scaling
per `KMAX_OPTIONS.md` §1 (K-dependent share 96.8 %, scales 192/128 = 1.5;
K-independent 3.2 %).

| target (on `D:/rd-staging/wmn1_feat/`) | estimate |
|---|---|
| `shards_v3/` (2463 runs, K=192, uncompressed) | ~52 GB |
| `pack_v3/` (packed arrays + outcomes + manifests) | **~52 GB** (52.05 GB per KMAX_OPTIONS §1) |
| `upload_v3/` (zstd parts, v2 ratio ≈ 0.196) | ~10.5 GB |
| **total staging** | **~115 GB → provision ≥ 130 GB free on D:** |

Free-space gate: refuse to start a stage below its estimate + 10 %.
Runtime anchor: full re-featurization ≈ 5 h at 6 workers (~30 worker-h,
`KMAX_OPTIONS.md` §5); pack ≈ 1-2 h; verify ≈ 15-30 min/pass; upload build
≈ 25-45 min (v2: 1211 s).

## 1. Pre-flight (read-only, no D: writes)

```bat
cd /d E:\rd-wt\kfix192\output\wd01\wm_n1
set PY=C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe
set WD_FORBID_D=1

:: code state: K=192 format prep is committed on zcode/kfix192
git log --oneline -1

:: tests that must be green before spending the allocation
%PY% -m pytest -q features/tests corpus/tests train/tests bundle/tests evalh/tests combine/tests model/tests
%PY% -m preflight all --data-root E:/research-dense

:: free-space check (>= 130 GB on D:)
:: estimate mode at K=192 (reads only the inventory + plans, no run files)
%PY% -m features estimate --chunk chunk4 --format v3
```

Expected: all suites pass, preflight 6/6 PASS, estimate prints
`format=v3 k_max=192`.

## 2. Re-featurize (featurize -> `shards_v3`)

```bat
%PY% -m corpus featurize-all ^
  --inventory inventory/inventory.parquet ^
  --data-root E:/research-dense ^
  --out D:/rd-staging/wmn1_feat/shards_v3 ^
  --workers 6 --idea-fast ^
  --min-free-ram-gb 8 --min-free-virtual-gb 10 ^
  --format v3 ^
  --allow-d-output-root D:/rd-staging/wmn1_feat
```

* Resumable by shard sha (completed shards skip); a crash = re-run the same
  command (never `--force`).
* `ALLOCATION.json` records `format: "v3"`, `k_max: 192`, workers, memory
  gates and the per-run list.
* Success line: `featurize-all: runs=2463 written=2463 skipped=0 failed=0`
  (plus `PLAN_MAP.json` + `opened_paths.json` clean).
* Per-shard `meta.json` self-describes: schema
  `wm_n1_features_format_v3/1`, `"format": "v3"`, `"k_max": 192`.

## 3. Pack (`pack_v3`)

```bat
%PY% -m corpus pack ^
  --data-root D:/rd-staging/wmn1_feat/shards_v3 ^
  --out D:/rd-staging/wmn1_feat/pack_v3 ^
  --workers 6 ^
  --allow-d-output-root D:/rd-staging/wmn1_feat
```

Expected stdout: schema `wm_n1_corpus_pack/2`, `n_runs 2463`,
`n_ticks ~3.19e6`, `k_max 192`, totals.bytes ≈ 52e9.  `manifest.json` +
`INDEX.json` carry `"format": "v3"`; a mixed-K tree is refused here.

## 4. Verify ×2 (both must be `ok: true`)

```bat
:: pass 1 — full verify (arrays, sha256, manifest, K contract, source shards)
%PY% -m corpus verify --pack D:/rd-staging/wmn1_feat/pack_v3 ^
  --allow-d-output-root D:/rd-staging/wmn1_feat

:: pass 2 — re-run from disk (idempotence; same verdict expected)
%PY% -m corpus verify --pack D:/rd-staging/wmn1_feat/pack_v3 ^
  --allow-d-output-root D:/rd-staging/wmn1_feat

:: Idea A channel receipts over the shard tree
%PY% -m corpus.idea_a verify --data-root D:/rd-staging/wmn1_feat/shards_v3 ^
  --allow-d-output-root D:/rd-staging/wmn1_feat
```

Expected each pass: `{"ok": true, "n_runs": 2463, "n_items": 2463,
"n_arrays": 15, "k_max": 192, "format": "v3", "errors": []}`; idea_a verify
`ok: true, n_runs: 2463`.

## 5. Upload manifest + `upload_v3`

```bat
%PY% -m corpus manifest ^
  --pack D:/rd-staging/wmn1_feat/pack_v3 ^
  --out D:/rd-staging/wmn1_feat/upload_manifest_v3.json ^
  --allow-d-output-root D:/rd-staging/wmn1_feat

%PY% upload_v2_build.py build ^
  --pack D:/rd-staging/wmn1_feat/pack_v3 ^
  --out D:/rd-staging/wmn1_feat/upload_v3 ^
  --d-override

%PY% upload_v2_build.py verify ^
  --pack D:/rd-staging/wmn1_feat/pack_v3 ^
  --out D:/rd-staging/wmn1_feat/upload_v3 ^
  --d-override
```

* The upload builder derives the archive name from the pack directory
  (`pack_v3.tar.zst.partNNN`, 2 GiB parts) — no v2 artifact is touched.
* Record in the completion report: parts table (bytes + sha256 each),
  `pack_v3/SHA256SUMS` sha256, `upload_manifest_v3.json` on-disk sha256,
  build/verify walls, and the `[done] N/N members byte-verified` line.
* Box-side procedure = the generated `upload_v3/UNPACK.md`.

## 6. Post-fix gates (after upload; amendment 5 §4)

* Slot census on `shards_v3` reporting the residual drop share
  (the A5 §4 "post-fix slot census").
* Envcheck + timing gate on the GPU box **against pack_v3 at K=192**:
  `python -m bundle.envcheck --format v3` and the real-member gate with
  `make_config(format="v3")` (`n_vehicles=192`); member 1 + held-out tests
  train on pack_v3 with `IdeaATrainConfig(k_max=192)`.

## 7. Non-negotiables

* v2/v2c artifacts are frozen evidence: read-only, never overwritten,
  never deleted; `shards_v3/pack_v3/upload_v3` are NEW directories.
* HOLDOUT, chunk-5 (seeds 8707-8736) and excluded worlds are refused by the
  same split-before-open guards as v2; never bypass them.
* No featurization/pack/upload runs from a session that has not registered
  the allocation (ALLOCATION.json) and the free-space check.
