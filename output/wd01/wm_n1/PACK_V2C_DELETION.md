# Deletion record: D:/rd-staging/wmn1_feat/pack_v2c

- **Deleted:** 2026-10-01 ~20:10 local, by the world model session, on the
  owner's explicit approval in chat ("I approve deleting
  D:/rd-staging/wmn1_feat/pack_v2c now (upload_v2c kept as its compressed
  copy)").
- **What it was:** pack_v2c, 2,463 runs, K = 128, 19 files, ~33 GB. It is
  superseded by pack_v3 (A5, K = 192).
- **Still preserved:**
  - `D:/rd-staging/wmn1_feat/upload_v2c/` (4 zstd parts, PARTS.json,
    UNPACK.md);
  - its SHA256SUMS, sha256 `7717fa2c…`;
  - the manifest `upload_manifest_v2c.json` (`b2efda91…`).
  - pack_v2c can be restored byte-for-byte from these with UNPACK.md.
- **Metadata kept** in `D:/rd-staging/wmn1_feat/pack_v2c_meta/`, copied and
  sha-verified before the deletion:

  | file | sha256 |
  | --- | --- |
  | INDEX.json | `550a29ac…` |
  | manifest.json | `f4fbab21…` |
  | SHA256SUMS | `53c27b37…` |

  Tools that read `pack_v2c/INDEX.json` (the census reconcile, the
  SOURCE_OVERRIDE sha checks, INC1 zero-shot) must use
  `pack_v2c_meta/INDEX.json`.
- **Pre-checks:**
  - no process referenced pack_v2c;
  - no reparse points inside;
  - upload_v2c was present.
- **Effect:** D: free went from 94 GB to 127 GB.
