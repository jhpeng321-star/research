# Owner summary

## What the end state looks like

One persistent checkout at `E:/research-dense`, with code, docs, tests, small run receipts, and frozen-path mapping tracked in Git. Large datasets, packs, checkpoints, and bulky run outputs live under `gdrive:research-archive/{backups,code,data,datasets,models,records,repo-evidence,packs,staging,worktrees}`. Local `data_cache/` is ignored, capped by default at 20 GiB, and populated only by explicit verified fetches. No move, upload, or deletion is performed by this package.

## Sequence and rough timeline

1. **Design/tool review (now; no network):** owner reviews this plan, decides restricted-data policy and cache budget, and approves which exact data item is first.
2. **Inventory + manifests (owner environment; hours to days):** build no-follow link/reference and writer inventory, then exact member/MD5 manifests. Time depends on dataset count and sizes.
3. **Copy + verify (per item; duration unknown):** upload an explicit stable snapshot to a new Drive path; independently verify complete members, bytes, MD5, and a restore sample. Keep all originals.
4. **Consumer cutover (separate idle window per live item):** configure registry/path mapping and verify actual readers. No live cross-volume move. Owner retains source until validated.
5. **Optional cleanup (later, owner-run):** exact per-file review and owner-executed deletion only after revalidation. Repository-root consolidation is not the same as archival upload.

## Decisions needed (4)

1. For each restricted class—sealed, CAL, HOLDOUT, chunk-5, INC1, and J4B—choose local-only retention or an explicitly access-controlled restricted Drive destination. No data content was examined.
2. Confirm the default 20 GiB cache budget, and whether it should be increased for the owner’s machine/workloads.
3. Approve the Drive taxonomy and who may access each category, especially `repo-evidence/`, `models/`, and `packs/`.
4. Select the first ordinary, non-restricted dataset for inventory/manifest registration and separately confirm the no-writer window before any consumer cutover. Backup verification alone is not deletion authorization.
