# UPLOAD_V2 — resumable compressed upload bundle of pack_v2

Date: 2026-10-01. Executor: GLM (zcode/feat1zip), task "FEAT1 corpus UPLOAD as
compressed resumable chunks". Gate waited on (2-min poll, ~14 min actual):
`FEAT1_REPORT.json` (zcode/feat1run5) with `verify.corpus_verify.ok=true`
(2445 runs, 15 arrays, 0 errors), `verify.idea_a_verify.ok=true` (2445 runs,
0 errors), and `upload_manifest.json` present at
`D:/rd-staging/wmn1_feat/upload_manifest.json`.

## What was built

`D:/rd-staging/wmn1_feat/upload_v2/` (7 files, nothing pre-existing overwritten):

| file | bytes |
|---|---|
| pack_v2.tar.zst.part000 | 2147483648 |
| pack_v2.tar.zst.part001 | 2147483648 |
| pack_v2.tar.zst.part002 | 2147483648 |
| pack_v2.tar.zst.part003 | 419337747 |
| SHA256SUMS | 360 |
| PARTS.json | 1501 |
| UNPACK.md | 1646 |

Format: deterministic GNU tar (members sorted by name, mtime=0, uid=gid=0,
mode 0644) -> zstd v1.5.7 level 6, `-T6` -> fixed 2 GiB parts.
Tooling: python `zstandard` absent, 7-Zip absent, zstd CLI present
(`C:/Program Files/MSYS2/mingw64/bin/zstd.exe`) -> tarfile + zstd CLI per the
brief's fallback order. gzip fallback not needed. CPU BelowNormal; compression
threads 6. WD_FORBID_D=1 semantics kept; exempt root registered as
`D:/rd-staging/wmn1_feat` via `corpus._util.activate_d_output_override`
(`--d-override`); pack_v2/shards_v2/shards untouched (read-only).

## Sizes / ratio / time

- pack (18 files listed in pack_v2/SHA256SUMS): 35,071,202,460 B (32.66 GiB)
- compressed parts total: 6,861,788,691 B (6.39 GiB)
- ratio: 0.1957 (5.11x)
- build wall: 1211.5 s (second run; first run 524.3 s, D: contention differs)
- verify wall: 223.3 s

## Integrity anchors (recorded in PARTS.json, cross-checked vs FEAT1_REPORT.json)

- pack_v2/SHA256SUMS sha256: `da7381dd7e4dab31375d9338f2cfb6e63b23b9fb58049ad882339a2af1b9f5dc`
- upload_manifest.json sha256: `a1a5a871b44a7145c3f976d85b48b860639b07cc80f1470d0ea0d12a74496db9`

## Local verification (real stdout excerpts)

Pre-build: all 18 pack files hashed, each `[preverify] ok <name>`.
Post-build (both builds; parts identical):

```
[verify] re-hashing 4 parts ...
[verify] part ok pack_v2.tar.zst.part000
[verify] part ok pack_v2.tar.zst.part001
[verify] part ok pack_v2.tar.zst.part002
[verify] part ok pack_v2.tar.zst.part003
[verify] member ok INDEX.json t+12.9s
... 18 lines ...
[done] 18/18 members byte-verified, tar=35071202460 B, parts=6861788691 B, wall=223.3s
```

Method: parts re-hashed from disk; then full streaming decompress
(`zstd -d` over the concatenated part stream, no temp files) with per-member
content sha256 compared against pack_v2/SHA256SUMS (E: free space 32.5 GB is
insufficient for a full extract, so the sanctioned streaming path was used).
Member count and decompressed tar byte total (35,071,202,460) match the pack
exactly. Negative test: a corrupted copy of part001 is flagged
(`part hash FAIL ... != ...`), proving per-part resumable re-download logic.

Determinism: two independent builds produced byte-identical parts
(all 4 sha256 equal). One earlier build attempt left a stray empty
`part004` + a wrong UNPACK.md `cat` example (script bugs, fixed in
`upload_v2_build.py`); those artifacts were removed and the bundle rebuilt;
current bundle byte-identical to the good run.

## Box-side procedure

See `upload_v2/UNPACK.md`: `sha256sum -c SHA256SUMS` (re-download only
mismatching parts) -> `cat pack_v2.tar.zst.part* > pack_v2.tar.zst` ->
`tar --zstd -xf pack_v2.tar.zst` -> `python -m corpus verify --pack pack_v2`.

Tool: `upload_v2_build.py` (this dir): `build --pack <pack> --out <dir>`
/ `verify --pack <pack> --out <dir>`; self-tested on a synthetic pack
(rotation, exact-multiple boundary, determinism, corruption detection).
