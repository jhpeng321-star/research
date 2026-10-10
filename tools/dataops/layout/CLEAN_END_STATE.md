# Clean end state: one work tree, Drive-backed data

**Design only.** Nothing in this proposal moves, renames, uploads, or deletes current files. Paths and sizes below are a target contract, not a fresh inventory of `E:/research-dense` or Drive. The supplied repo snapshot is dirty; preserve every existing change and the existing `E:/research-dense` identity until the owner approves and performs each separate step.

## Target local layout

The single persistent project root is `E:/research-dense`. It contains tracked source, tests, protocols, concise records, and documentation; `.git` and normal Git metadata; and one ignored, disposable, re-fetchable `data_cache/`. Keep the established project subtrees (`traffic_shepherd/`, `baselines/`, `assets/`, `scenarios/`, `archive/`, `protocols/`, `tests/`, `tools/`, `docs/`, `runs/`) where code and frozen references currently expect them. Do not flatten/rename them as part of this migration.

```text
E:/research-dense/
  .git/                         Git metadata
  AGENTS.md README.md           project rules and entry point (tracked)
  traffic_shepherd/ baselines/ assets/ scenarios/ archive/ (tracked code/config/frozen small assets)
  docs/ protocols/ tools/ tests/ (tracked)
  runs/<id>/                    small registration, command, log-index, receipt, summary (tracked)
  data_cache/                   ignored; fetched data only, never source of truth
  <at most one task worktree at a time, outside this root while needed>
```

**Cache budget:** default hard planning budget 20 GiB across `data_cache/`; fetch must refuse a new item when its registered byte size would exceed the configured budget (default can be overridden by an explicit CLI option/configuration). Owner may raise it after checking free space. Cache contents are expendable only because each item is registered, verified, and re-fetchable; cache eviction is an explicit owner command/process, not part of fetch or reorganization. Keep no second permanent checkout. Agent task worktrees are transient, created for a specific task, and removed by the owner only after merge and reference checks.

Tool interface from the repository root: `python tools/fetch_data.py list`, `status [name]`, `verify [name]`, or `get <name>`. `get` invokes `rclone copy` only when explicitly requested, uses the configured `rclone` executable, and has no delete/cleanup verb. `--cache-dir`, `--registry`, `--rclone`, and `--budget-bytes` are available for controlled invocation. The registry record shape is `{ "drive_path": "gdrive:research-archive/datasets/<object>", "size_bytes": 123, "md5_manifest": {"relative/file": "<32 hex MD5>"}, "local_path": "data_cache/<logical-name>" }`; only add an entry after its complete manifest and restore destination are confirmed.

`data_cache/` contains only explicitly fetched registry items, each under its registry `local_path`. `.gitignore` should ignore `/data_cache/` at the root and must not broaden to hide source, receipts, or arbitrary `runs/` content. Add that ignore rule only as a separately reviewed implementation step in the target repository; this design workspace does not contain or modify that repository.

## Drive layout

All validated data and backup copies target `gdrive:research-archive/`:

```text
research-archive/
  backups/       verified point-in-time backup sets and their manifests/receipts
  code/          optional source snapshots; Git remains canonical for code
  data/          generated and intermediate research data
  datasets/      immutable source datasets/packs and member manifests
  models/        checkpoints and model artifacts
  records/       durable experiment records, receipts, and recovery manifests
  repo-evidence/ frozen evidence bundles and path/member manifests
  packs/         pack-formatted corpora and their sidecars/manifests
  staging/       explicitly temporary transfer staging; never canonical storage
  worktrees/     optional archived task-worktree snapshots after owner review
```

Each backup/offload is a new, uniquely named object set with a versioned receipt binding the exact source snapshot, relative member paths, byte sizes, MD5 values, destination, check/copy exit statuses, and time. Upload plus independent complete-member verification is a recovery copy, **not** deletion permission. A check without common per-member hashes is insufficient. Revalidate source and destination immediately before any later owner-authorized deletion.

## Placement of current repo-root content

These are target dispositions by role. Resolve each concrete path with a fresh inventory and reference scan before any owner action; the supplied `REPO_ROOT_LS.txt` is a snapshot, not proof of current content or completeness.

| Current path/class | End-state placement |
|---|---|
| `AGENTS.md`, `README.md`, `pyproject.toml`, requirements, `.gitignore` | Stay at root in Git. Preserve current worktree bytes/changes. |
| `traffic_shepherd/`, `baselines/`, `scenarios/`, source-like portions of `output/` | Stay in Git at existing paths unless a separately reviewed source migration proves all imports/references and history. |
| `assets/` | Small code-required, frozen assets stay local and versioned/checksummed as appropriate. Large re-fetchable assets get a registry entry and a path mapping; never silently replace live paths. |
| `docs/`, `protocols/`, `tools/`, `tests/` | Stay in Git. Preserve historical and user-authored records; no broad cleanup. |
| `runs/<id>/` | Keep small registration, command/provenance, status, summary, and artifact index/receipt in Git. Put large logs, arrays, checkpoints, trajectories, and generated packs in the Drive category selected by type; fetch via registry for a consumer. Keep the local path resolvable while any reader depends on it. |
| large generated files under `output/`, `runs/`, or other roots | Registry-managed Drive object plus ignored local cache, only after per-item manifest and verified restore contract exist. Never classify by filename, age, failed-run status, or “backup exists” alone. |
| root `sealed/`, CAL, HOLDOUT, chunk-5, INC1, J4B and related restricted material | **Owner decision required** for each class/item: retain local-only or use an approved restricted Drive folder/access policy. No contents are inspected by this task; no entry is pre-populated in the registry. |
| old D:/ staging and any linked worktree/data roots | No implied removal or migration. Migrate only a named item with known writers, reverse references, size, hash manifest, capacity and rollback plan. D:/ and E:/ cannot be treated as one-volume rename operations. |

## Frozen evidence path preservation

Create and version a `PATH_MAP.json` at the repository root (or a repository-approved stable location) before any path changes. It maps each historical absolute/relative source path to (a) its still-present local path, (b) a compatibility alias/junction whose exact target is recorded, or (c) a logical registry key plus the local `data_cache/` resolution path and restore instructions. Keep every original frozen manifest and report byte-for-byte unchanged. Do not change their recorded paths to make a migration appear complete. Mapping entries are additive, checksummed where relevant, and tested by resolving all inventoried frozen references. A Drive URL alone does not resolve a Windows path. Do not create aliases until references, exact targets, ancestors, and link topology are inventoried without following links.

## Ordered, staged, reversible procedure and gates

Every stage is a proposal until the owner explicitly starts it. Record before/after receipts outside the candidate deletion set. Stop on uncertainty; no script has permission to perform deletion.

1. **Freeze scope and inventory.** Preserve dirty Git changes. Inventory the exact repository and data paths, Git worktree registrations, running readers/writers, open handles where available, file counts/bytes/hashes, and all inbound/outbound links. Enumerate reparse points iteratively using metadata only; identify a link before descending and never follow it. Include ancestors and nested links. Unknown/inaccessible state blocks the item. Do not inspect restricted contents.
2. **Classify and map.** Owner classifies each exact item as code, small record, re-fetchable cache, retained unique evidence, or restricted. Build the additive `PATH_MAP.json`, registry entries, and exact source-to-destination/member manifest. Keep unknown/only-known-copy data. Check all project/frozen references resolve without rewriting manifests.
3. **Back up to Drive (copy only).** Require a stable snapshot or quiescent source; measure available Drive/local capacity and reserve; upload one explicit item at a time to a new destination. No archive expansion into a local intermediate. Record a durable intent and result receipt. Upload failures stop the batch.
4. **Verify restore readiness.** Independently compare complete source/destination relative member sets, sizes, and per-member MD5s; require all command exit codes successful and source hash recheck unchanged. Then perform a restore-to-temporary-location test for at least one representative item/type and verify its manifest. The original remains untouched. A verified upload never authorizes deletion.
5. **Prepare consumers and cache use.** Add a complete registry manifest and fetch workflow; test code/data readers against the cache at expected paths. If compatibility aliases are proposed, ensure every destination is on the same volume where required, exact link targets are known, and each old path resolves. Cross-volume live data stays in place while any training/evaluation reader is active; do not cut over or move it under a live reader.
6. **Owner cutover decision.** Owner confirms the exact item, source snapshot, restored copy, full reference graph, idle window, capacity, and rollback. For cross-volume changes, copy-verify first, stop all writers, switch consumers/aliases as a separately journaled operation, run path/integrity checks, and retain original source until owner confirms the new path is stable. A dirty repository stays in place; no repo-root move is included in this plan.
7. **Owner-only deletion, if explicitly approved.** Present a per-file manifest and impact/restore receipt. The owner performs every deletion manually/explicitly after a fresh no-follow link/reference audit and immediate source/destination revalidation. Never recursively delete a tree with any link, unknown child, protected path, or incomplete inventory. Retain the mapping, receipts, source manifests, and recovery evidence permanently. No automated cleanup is supplied by this task.
8. **Close out.** Verify clean Git status only after the owner has reviewed/merged their changes; validate each old evidence path through the map; test cache fetch/status/verify; update records. Remove a task worktree only after merge and an owner no-follow/reference review.

**Hard stops:** no `git clean/reset/stash/checkout`; no broad deletion; no junction-following; no Drive credentials/config access by these tools; no network operation unless the owner separately runs the data tool; no sealed/restricted inspection; no cross-volume movement of data being read by training/evaluation; and no deletion by this procedure without the owner's direct action.
