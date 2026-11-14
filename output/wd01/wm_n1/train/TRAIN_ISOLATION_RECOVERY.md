# TRAIN ISOLATION RECOVERY — DRAFT (TESTS PENDING)

> **STATUS: DRAFT / TESTS PENDING GUARD.**
> Code-only safety repair. Per the coordinator brief (2026-10-02), NO test,
> pytest, loader import, or CLI execution has been run and NO real pack,
> outcome, HOLDOUT/CAL/sealed chunk-5/J4B/INC1 bytes were opened. All test
> selections below await the reviewed guard + exact test list in a later
> follow-up. Synthetic-only contract tests; `WD_FORBID_D=1` required.

Branch: `zcode/trainisolate` (draft commits only; no merge to main/checkpoint).
Executor: GLM-5.3-Flash (ZCode execution agent, per owner direction).

## Scope

Resume of the preserved TRAIN-label-isolation draft (`train/dataset.py`,
`train/label_isolation.py`, `train/tests/test_label_isolation.py`) after a
concrete review, fixing fail-open paths without running anything.

## Follow-up draft (2026-10-02, code-only — eligible-pool sampler + INDEX audit)

Coordinator compact follow-up on draft 2cfbe4a. Still NO test/pytest/CLI
execution and no real data touched; synthetic-only draft tests added.

9. **Eligible-pool `ensure_pair` fix** (`train/dataset.py`): the batch
   completeness check inside `ensure_pair` considered ANY completed semantic
   sibling pair, so a batch holding only an unsupported (out-of-pool) pair
   bypassed a filtered eligible `pairs` pool entirely. `batch_has_pair`
   gained an optional `pairs` argument: WITH a pool, only a pair FROM the
   pool counts as complete (partner map, both directions); WITHOUT a pool
   the historical semantic check is kept. `ensure_pair` now uses the pool
   check; the appended partner(s) still come only from the deterministically
   cycled `pairs[cursor % len(pairs)]`, append-only, cursor advancing once
   per completion. `pairs=None` keeps the upstream contract (callers already
   compute the default pool); an explicitly EMPTY pool invents nothing.
   Primary items stay exactly-once per epoch; `epoch_batches` /
   `PairAwareBatchSampler` parity is preserved for filtered pools (and the
   in-process path passed through `epoch_batches` in `train/idea_a.py` is
   unchanged — that file was NOT modified).
10. **Single-split TRAIN manifest vs a PRESENT INDEX** (`train/dataset.py`
    + new `train/label_isolation.audit_pack_identity`): the former claim
    "a wholly single-split TRAIN manifest proves no excluded outcome rows"
    ignored a contradicting pack INDEX. Now, when an INDEX is PRESENT, a
    metadata-only audit (no parquet, no shard tree, runs BEFORE any label
    read) reconciles it with the manifest: every INDEX run must be a usable
    record (recognized split, no chunk-5 spelling, `shard_dir` matching the
    `<batch>/<world>__<arm>` identity, non-negative `n_outcome_rows`),
    every manifest item must match exactly one run by identity with an
    agreeing split and `t0_s`, and every run must be matched by an item.
    A non-TRAIN or unmatched run (or split/`t0_s` drift) makes the wholly
    TRAIN pack REFUSE (`CorpusRefusal`) before any label read. A wholly
    TRAIN manifest WITHOUT an INDEX keeps the legacy mixed-table read (no
    ban on legacy/synthetic fixtures with explicit nonempty TRAIN items);
    MIXED manifests still always require usable INDEX identity (unchanged
    refusals and messages).
11. **Empty manifest refuses labels**: a pack manifest with no items is
    refused at load (`CorpusRefusal`, TRAIN and CALIBRATION alike) — an
    empty manifest proves no training-only table and previously fell into
    the `splits <= {"TRAIN"}` branch, allowing the packed-table read.
12. **Prior guards kept**: file containment, fresh-output, portable shard
    identity, failure-receipt preservation — untouched; raw pack bytes and
    frozen math untouched.

### New draft tests (NOT run)

`train/tests/test_eligible_sampler_recovery.py` (NEW, owned):
`test_unsupported_semantic_pair_no_longer_bypasses_eligible_pool`,
`test_out_of_pool_pair_gets_only_the_missing_member_appended`,
`test_in_pool_complete_pair_returned_unchanged`,
`test_empty_pool_never_invents_and_none_defers_to_caller`,
`test_filtered_pool_partner_choice_is_deterministic_and_cycling`,
`test_epoch_batches_filtered_pool_keeps_primary_coverage_and_parity`.

`train/tests/test_label_isolation.py` (additions):
`test_train_manifest_contradicted_by_index_fails_closed` (cal_run /
foreign_run / dropped_run), `test_train_manifest_index_nontrain_foreign_run_fails_closed`,
`test_wholly_train_real_pack_without_index_keeps_legacy_read`,
`test_empty_manifest_pack_refuses_labels`,
`test_audit_pack_identity_agrees_on_real_pack_and_refuses_empty`.
The pre-existing `test_wholly_train_real_pack_keeps_original_behavior`
doubles as the agreeing-INDEX audit regression.

## Follow-up draft (2026-10-02, code-only — portable recorded source identity)

Coordinator compact follow-up on f748d0e. Still NO test/pytest/CLI
execution and NO import of the owned module; no real data touched; static
`py_compile` + scoped `ruff` only. Source-only fixtures added.

13. **Portable lexical recorded-source parsing**
    (`train/label_isolation.py`): the coordinator finding —
    `_shard_rel_logical` parsed the recorded `shard_dir` with the HOST
    `pathlib.Path`, so on Linux a recorded Windows absolute
    `D:/rd-staging/...` path parses as RELATIVE with >3 components and a
    valid portable label sidecar refuses. New `_recorded_parts` helper:
    recorded paths are now parsed LEXICALLY and OS-independently —
    `PureWindowsPath` recognizes a Windows drive (`D:/...`, `D:\...`) or
    UNC (`//server/share/...`) prefix on ANY host OS; everything else
    parses as `PurePosixPath`. The parse NEVER opens, creates or resolves
    the recorded source. `_shard_rel_logical` semantics otherwise
    unchanged: a relative `shard_dir` must be exactly
    `<chunk>/<world>/<arm>`; an absolute one must END in the identity
    triple; the returned logical identity is the same pure-POSIX
    `chunk/world/arm` string, so the canonical Windows
    `D:/rd-staging/wmn1_feat/shards_v3/<batch>/<world>/<arm>` and its
    POSIX `/mirror/.../<batch>/<world>/<arm>` twin yield IDENTICAL
    normalized identities (receipt `shard_rel` / `shard_dir_recorded`
    reconciliation keeps working cross-OS — e.g. a sidecar built on
    Windows now loads on the GPU box). NEW strict refusals (fail-closed,
    no weakening): empty spellings, drive-relative spellings (a Windows
    drive without a root, `D:` / `D:chunk4/...`) and any `..` traversal
    component ANYWHERE in a recorded path (previously a `..` hidden in the
    middle of a native absolute path could slip past identity matching;
    now refused outright). Exact 3 logical segments, identity match,
    conservative split/chunk spellings and the metadata audit order are
    unchanged.
14. **Foreign-flavor absolute under a local shards root**
    (`_shard_rel_of`): a recorded absolute path of the HOST's own flavor
    must still resolve inside the (already resolved) shards root exactly
    as recorded — that containment check is untouched. A recorded
    absolute of the OTHER flavor (Windows `D:/...` validated on the POSIX
    GPU box, or a POSIX `/...` path on Windows) cannot be
    containment-compared locally; it is addressed by its tail logical
    identity under the LOCAL shards root, where all per-component
    literal/symlink, 3-segment and escape checks still run, and the
    recorded old source is never opened or resolved. Relative recorded
    identities take the same joined path as before.
15. **Portable load untouched where already safe**: `load_train_outcomes`
    still takes only pack + sidecar, still revalidates everything
    metadata-only against the re-derived plan BEFORE the sidecar parquet
    is opened, and still never opens the receipt's recorded
    `shard_dir_recorded` / old `D:` sources; `plan_sidecar(...,
    shards_root=None)` remains pure metadata. `audit_pack_identity` still
    runs the metadata audit BEFORE any label read (dataset call site
    unchanged; malformed recorded paths refuse there too).

### New source-only fixtures (NOT run)

`train/tests/test_label_isolation.py` (additions):
`test_recorded_windows_source_identity_is_portable_across_hosts`
(canonical `D:/rd-staging/wmn1_feat/shards_v3/<batch>/<world>/<arm>`,
backslash spelling, UNC, and POSIX `/...` — the normalized logical
identity matches across flavors),
`test_malformed_recorded_source_paths_are_refused` (empty, `.`, short and
long relative, relative traversal, drive-relative `D:` and
`D:chunk4/...`, too-short absolute, absolute traversal, wrong-world
identity — all refused),
`test_windows_recorded_pack_plans_portably_without_opening_source`
(Windows-recorded INDEX: the metadata audit passes BEFORE labels and a
`shards_root=None` plan derives every identity with zero parquet opens),
`test_foreign_flavor_recorded_source_uses_local_shards_mirror_only`
(foreign-flavor recorded absolutes plus a local shards root: plan
identity holds and a build reads ONLY the local mirror — the recorded old
source never opens). These extend the pending whole-file pytest selection
below; nothing was executed.

## Follow-up draft (2026-10-02, code-only — deletion-free isolation fixtures)

Coordinator compact follow-up on 141f018 (reviewed/integrated). Root AST
safety audit found five filesystem-destruction sites in
`train/tests/test_label_isolation.py`; per owner policy NO test file may
destroy filesystem entries, including synthetic fixture bytes. All five
sites are now CONSTRUCTIVE: fixtures build initially-missing objects or
fresh synthetic variant trees, and every pre-existing fixture file is
preserved. Production modules and all other tests untouched. Still NO
test/pytest/CLI execution and no real data touched; static `py_compile` +
scoped `ruff` only.

16. **`_symlinks_supported` probe**: the dangling `symlink_probe` symlink
    is left in place inside the test's private `tmp_path` (pytest owns
    that tree); support detection is unchanged (creation failure ⇒ skip).
17. **Symlinked shard outcomes refusal**: instead of swapping the shared
    fixture's real shard file, the test builds a fresh synthetic mirror
    of the shards tree in which `chunk4/w1/base/outcomes.parquet` IS a
    symlink from the start (target: a real outside outcomes file copied
    byte-identically; all other mirror entries are byte-identical
    copies). `build_sidecar` runs against the ORIGINAL pack plus the
    mirror; the guard must refuse BEFORE any read (`spy == []`), and a
    new assertion pins the original shard as a preserved regular file.
18. **Mixed pack with missing INDEX**: a fresh variant pack directory is
    constructed that NEVER contains `INDEX.json` (new helper
    `_pack_variant_without_index` copies every pack file byte-identically
    EXCEPT `INDEX.json`); the built mixed pack — INDEX included — stays
    preserved. Same refusals before any label read (`mixed-split corpus
    pack`, `corpus[0]` + `outcome_target(0)`, `spy == []`).
19. **Missing receipt**: a fresh `sidecar_without_receipt` directory holds
    a byte-identical COPY of the sidecar parquet and never a receipt; the
    built sidecar (receipt included) stays preserved. Load must refuse
    with "missing" before opening the parquet (`spy == []`).
20. **Wholly-TRAIN pack without INDEX (legacy read)**: uses
    `_pack_variant_without_index` so the INDEX-less variant is born
    without INDEX; the full pack (with INDEX) stays preserved. Same
    legacy mixed-table assertions (`len == 3`, spy exactly the variant
    pack's `outcomes.parquet`).

### Fixture note (pending nodes, NOT run)

The five rewritten nodes keep their names and assertion strength:
`test_build_refuses_symlinked_output_path` (probe),
`test_build_refuses_symlinked_shard_outcomes_file`,
`test_mixed_pack_missing_index_fails_closed`,
`test_missing_receipt_fails_closed`,
`test_wholly_train_real_pack_without_index_keeps_legacy_read`. They remain
part of the whole-file pending selection below (`WD_FORBID_D=1`); nothing
was executed. A token scan of the test file for every prohibited
filesystem-destruction spelling (delete/move/replace family) returned zero
matches; the suite now performs no filesystem destruction at all.

## Follow-up draft (2026-10-02, source-only — synthetic test rename)

Coordinator compact follow-up on d9d0b28, after the root's first guarded
whole-file attempt at `test_label_isolation.py` retained 21 failed /
15 passed / 34 errors. Root identified two distinct causes; the first (the
dynamic frozen partA_v2g source not being pinned — the root is fixing the
external manifest after source review) is root-owned and NOT addressed here.
The second distinct cause is owned here: pytest derives the per-test
`tmp_path` directory basename from the test function name, so
`test_cal_split_keeps_authorized_mixed_read` created fresh directories
named `test_cal_split_keeps_authorized_mixed_read0/...`, and the reviewed
root guard protects the prohibited `test_cal` prefix even inside fresh
synthetic directories.

21. **Synthetic test rename** (`train/tests/test_label_isolation.py`):
    ONLY the synthetic test function
    `test_cal_split_keeps_authorized_mixed_read` is renamed to
    `test_synthetic_cal_keeps_authorized_mixed_read`; its body,
    assertions and `mixed_env, monkeypatch` fixtures are byte-identical
    (the tmp_path basename no longer starts with the prohibited prefix).
    Guard and production modules untouched; no other name, selection or
    assertion changed. Still NO test/pytest/CLI execution and NO real
    data touched; static `py_compile` + scoped `ruff` only.

## Review fixes applied (code-only)

1. **Mixed manifest TRAIN can no longer silently open the mixed table**
   (`train/dataset.py::_mixed_split_identity_pack`): a MIXED-split manifest
   now REQUIRES identity-bearing INDEX metadata. A missing `INDEX.json`, an
   INDEX without a usable `runs` list, or any run with incomplete identity
   (`shard_dir` / `n_outcome_rows` blank) raises `CorpusRefusal` BEFORE any
   label read. The former "manifest-only synthetic/legacy mixed fallback"
   exception is removed.
2. **Single-split legacy read is conditional**: only a wholly single-split
   TRAIN manifest (or empty items) keeps the original packed-table read,
   because that metadata proves no excluded outcome rows exist.
3. **Portable sidecar validation** (`train/label_isolation.py`): every run's
   logical `<chunk>/<world>/<arm>` identity is now ALWAYS derived from the
   recorded `shard_dir` and enforced — with `shards_root=None` (dataset /
   GPU-box path) the receipt's `shard_rel` AND `shard_dir_recorded` are
   reconciled against the pack-derived identity; the previous
   "skip when None" behavior is gone. Relative shard dirs must be exactly
   three components; absolute ones must END in the identity triple, so `..`
   traversal spellings can never match.
4. **File-level symlink/escape guards**: every `outcomes.parquet` we read or
   hash — TRAIN source shards (build + `--rehash-sources`) and the sidecar
   parquet — is guarded as a regular, non-symlink file whose whole path
   chain is literal inside the resolved shards root / sidecar dir, BEFORE
   `pd.read_parquet` or hashing.
5. **Sidecar output containment**: `build_sidecar` refuses output that
   exists, is a symlink, travels through any symlink component, or resolves
   inside the pack or source-shard trees. Fresh-only, never overwrite.
6. **Failure receipts preserved**: `_write_json_atomic` no longer
   `os.unlink`s its temp file on failure; the `.tmp` remains as the failure
   receipt.
7. **Exactness**: the receipt now records the sidecar's exact
   columns/dtypes; load re-verifies them, plus row-id coverage, per-run
   identities, pack manifest/INDEX shas and sidecar sha — all pack-metadata
   tamper checks run BEFORE the sidecar parquet is opened.
8. **Conservative spellings**: only `TRAIN`/`CALIBRATION` item/run splits are
   recognized (unknown spellings refuse the pack/plan); every chunk-5
   spelling variant (`chunk-5`, `chunk_5`, `Chunk 5`, ...) is treated as the
   sealed chunk, and a `chunk`-prefixed label without a digit is refused as
   unknown (real `chunk1..chunk4`, `chunk3-fix27` style labels still pass).

## Fixture updates (draft tests only)

- `test_manifest_only_mixed_pack_fails_closed_without_identity` replaces the
  former unsafe mixed fallback test (mixed manifest-only pack now refuses).
- `test_manifest_only_train_only_pack_keeps_original_behavior` pins the
  legitimate single-split legacy behavior.
- New tests: missing/empty/incomplete INDEX identity (parametrized),
  portable no-`shards_root` validation (plan + receipt reconciliation of
  `shard_rel`/`shard_dir_recorded`), recorded shard identity mismatch and
  traversal refusal, unknown split spelling, chunk-5 spelling variants,
  output-inside-pack/shards refusal, symlinked output path and symlinked
  shard outcomes file refusal (skip-guarded where the OS forbids symlinks),
  receipt dtype exactness, and failure-temp-receipt preservation.

## Pending test list (NOT run — awaiting coordinator guard + selection)

`train/tests/test_label_isolation.py` (whole file, synthetic only,
including the portable recorded-source-identity fixtures above), with
`WD_FORBID_D=1`, e.g.:

- `python -m pytest output/wd01/wm_n1/train/tests/test_label_isolation.py -q`
- `python -m pytest output/wd01/wm_n1/train/tests/test_eligible_sampler_recovery.py -q`

Sampler regression nodes (coordinator's call, also pending):
`train/tests/test_dataset.py::test_ensure_pair_makes_every_batch_complete_and_deterministic`,
`::test_contrast_pairs_exclude_switching_native_reference`,
`::test_ensure_pair_makes_every_batch_complete_and_deterministic` parity in
`test_m1fix2.py` (non-owned; reported for selection only).

Exact selections and any broader regression set (e.g. `test_dataset.py`,
`test_train_interface.py`) are the coordinator's call; non-owned tests were
NOT edited. NOTE for the reviewer: the dataset is now strictly fail-closed
on mixed manifest TRAIN reads, so any non-owned test that opens a MIXED
pack as TRAIN without `WM_N1_TRAIN_OUTCOMES` would now (correctly) refuse.

## Constraints honored

- No git reset/clean/checkout/stash-drop/push; no merges; nothing deleted
  from the workspace; dirty user work preserved.
- No real HOLDOUT/CAL/CAL-split/sealed chunk-5/J4B/INC1 outcome reads; no
  `CAL_SPLIT_A4.json` contents; no SUMO; no network; no approvals touched.
- Owned files only: `train/dataset.py`, `train/label_isolation.py`,
  `train/tests/test_label_isolation.py`, this report.
- Syntax verified via `python -m py_compile` only (no imports executed).

## Open items for the coordinator

1. Supply the reviewed guard + exact test selections, then run pending
   tests with `WD_FORBID_D=1` (env: `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`).
2. Decide whether non-owned fixtures that open mixed packs as TRAIN need a
   sidecar-based update (owned by their piece, not this draft).
3. GPU-box portability: `WM_N1_TRAIN_OUTCOMES` names a self-contained
   sidecar dir; validation without the shard tree is covered by the
   portable logical-identity checks above.
