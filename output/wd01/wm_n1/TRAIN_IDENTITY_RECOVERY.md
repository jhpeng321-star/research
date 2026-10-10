# WM-N1-RECOVERY-01: TRAIN vehicle identity sidecar draft

Status: **DRAFT TESTS PENDING GUARD**. This implementation task has read source
code and instructions only. Zero real-data reads, project imports, CLI calls,
tests, training steps, or simulations were performed. The integrating coordinator
owns the shared recovery ledger and final close-out. No scientific conclusion or
frozen loss definition is changed by this draft.

## Source-derived defect and repair

`corpus/pack.py` omits `slot_index` and `slot_vid` from `TRAIN_ARRAYS`.
`features/FORMAT.md` specifies reusable concurrent slots: `slot_index[t,k]`
indexes a run-local, first-appearance-ordered `slot_vid` string table. Equal slot
columns or equal local integer indices across arms therefore do not establish
vehicle identity. `features/writer.py` records hashes for both identity files in
`DONE.json` (the array hash key is `slot_index`, without `.npy`).

The new independent `train/identity_sidecar.py` uses only original TRAIN shard
identity files. It leaves the frozen pack bytes unchanged. It does not import the
dataset, feature writer, model, or label-isolation module.

## Interface for the consumer owner

- `build_identity_sidecar(pack_dir, shards_root, out_dir) -> receipt`: requires an
  exclusively fresh output directory outside pack/source trees, with no symbolic
  links or Windows reparse points in any path component. Standard frozen/archive
  directories are refused. Source/output authorization still belongs to the
  separately reviewed access guard; name checks alone cannot discover every
  externally frozen directory.
- `load_identity_sidecar(pack_dir, sidecar_dir) -> IdentitySidecar`: needs pack
  metadata and the sidecar only. Original shard roots and historical absolute
  paths are never opened on this path. Receipt content cannot select filenames.
- `accessor.ids_for_item(original_manifest_global_position) -> int64[K]`: a
  read-only view backed by immutable bytes, with `-1` for absent slots. A
  TRAIN-subset ordinal is **not** the original manifest position. Missing positions
  fail. Explicit pickle state preserves read-only behavior in DataLoader workers.
  Consumers must translate pair order to their TRAIN item and then to that item's
  original global manifest position before calling the accessor.
- `accessor.receipt_sha256`: SHA256 of the raw receipt file, for binding in the run
  record. Record this externally before transfer/execution.
- Proposed consumer environment variable: `WM_N1_TRAIN_IDENTITIES`, pointing to
  the sidecar directory. This draft intentionally does **not** edit consumers or
  automatically read that variable.

The loader requires a canonical pack directory whose complete path contains no
symlinks or reparse points. A GPU deployment alias such as `$WORK/pack` is not a
valid direct loader argument. The coordinator must independently validate that
specific alias and approve its canonical target before a consumer resolves it;
then pass that canonical directory to the loader. Record the submitted alias and
approved target separately. Raw manifest/INDEX SHA bindings still identify the
pack on either machine. This module never silently follows unknown aliases, and
the source/output path policy remains strict.

Canonical IDs enumerate sorted `(chunk, world, actual_vehicle_id_string)` tuples
present in selected item snapshots. Different arms in the same chunk/world share
an ID only when their actual vehicle-ID strings agree, independently of local
slot-index numbers. IDs are stable within this bound pack/sidecar, not promised
stable after adding/removing a corpus. They are exclusively provenance/loss
alignment metadata, never learned network input. The snapshot row is selected at
`manifest_item.tick - INDEX_run.tick_start`; identities do not stand in for a
full future trajectory alignment.

The paired HDV contrast consumer must intersect these verified actual identities
and gather the corresponding potentially different slot columns before computing
the registered pinball term. Real headline fitting must fail without verified
identities. Missing identities must never silently disable that term or change
its scientific definition. Synthetic tests can explicitly supply fake identities.
These consumer changes are owned by a separate worker.

## Binding and refusal behavior

The full manifest/INDEX reconciliation completes before any shard identity is
opened. Exact TRAIN selection is established using a one-to-one run/item mapping;
known CALIBRATION records are excluded. Unknown splits, chunks or batch/chunk
relationships, missing INDEX/run keys, duplicate identities, split disagreements,
invalid tick/decision/outcome ranges, inconsistent widths and forbidden content
fail closed. The conservative batch/chunk allowlist is copied from the source
`corpus/featurize_all.py` mapping; unsupported future batches require review.

For selected TRAIN runs only, the builder opens `DONE.json`, verifies its identity
and both identity-file hashes, then opens `slot_vid.json` and `slot_index.npy`.
It rejects missing/duplicate string maps, non-int32 slot arrays, wrong widths or
tick counts, out-of-range identity indices and duplicate vehicles within a tick.
`DONE.n_slots` counts columns occupied at any tick, as defined by
`features/featurizer.py`; it is not the length of `slot_vid`. It must be an integer
in `[0,K]` and equal `sum(any(slot_index >= 0, axis=0))` after loading. A run's
distinct vehicle table can exceed K when columns are reused. Independent source
review caught the initial draft's incorrect equality against vehicle-table length;
this revision corrects it and the synthetic fixture's metadata construction.
`-1` follows the source format's absence convention. No outcomes, packed tensors,
physics, original source-path targets, or excluded shard files are opened.

Output files are fixed: `identities.npy` `[n_TRAIN_items,K]`,
`global_positions.npy` `[n_TRAIN_items]`, `vehicle_map.json`, and
`TRAIN_IDENTITY_RECEIPT.json`. The receipt binds raw manifest/INDEX hashes, exact
selection, source DONE/identity hashes, namespaces, file hashes and vehicle count.
Loading checks metadata, receipt digest, fixed filenames and all file hashes
before parsing arrays. Hashing reads opaque bytes; it does not parse arrays.
Receipt self-digests detect corruption, not malicious coordinated bundle rewrites;
external receipt SHA binding is required for provenance authentication.

Recorded absolute shard paths are parsed lexically in either Windows or POSIX
syntax and reduced to a validated logical `<batch-or-chunk>/<world>/<arm>` suffix.
Only this suffix is resolved beneath the explicitly supplied shard root. Every
actual file and ancestor is checked before open/hash. Paths through junctions or
symlinks are refused even when they resolve in-bounds. Filesystem mutation races
are not claimed secure against a concurrent hostile writer; keep reviewed source
trees stable during an authorized build.

The builder preserves `BUILD_STARTED.json`, partial outputs and a
`BUILD_FAILED.json` on a caught failure. It never removes files, replaces previous
sidecars, or retries in-place. A new attempt needs a new output directory.

## Exact future synthetic test nodes (not executed)

File prefix for every node below:
`output/wd01/wm_n1/train/tests/test_identity_sidecar.py::`

- `test_swapped_slots_and_same_vid_different_local_index`
- `test_reused_slot_different_vid_is_not_a_match`
- `test_selected_tick_uses_current_identity`
- `test_vid_table_larger_than_slot_width_with_reused_columns`
- `test_invalid_occupied_column_count_refused[-1]`
- `test_invalid_occupied_column_count_refused[4]`
- `test_invalid_occupied_column_count_refused[0]`
- `test_invalid_occupied_column_count_refused[1]`
- `test_invalid_occupied_column_count_refused[3]`
- `test_readonly_pickle_and_portable_load`
- `test_exact_train_selection_before_source_opens`
- `test_missing_identity_file_fails_and_preserves_failure[DONE.json]`
- `test_missing_identity_file_fails_and_preserves_failure[slot_index.npy]`
- `test_missing_identity_file_fails_and_preserves_failure[slot_vid.json]`
- `test_invalid_identity_mapping_fails[missing_map]`
- `test_invalid_identity_mapping_fails[duplicate_map]`
- `test_invalid_identity_mapping_fails[duplicate_tick]`
- `test_invalid_identity_mapping_fails[width]`
- `test_invalid_metadata_fails_before_array_load[hash]`
- `test_invalid_metadata_fails_before_array_load[done_identity]`
- `test_invalid_metadata_fails_before_array_load[missing_hash]`
- `test_invalid_metadata_fails_before_array_load[tick]`
- `test_invalid_metadata_fails_before_array_load[run_key]`
- `test_invalid_metadata_fails_before_array_load[split]`
- `test_invalid_metadata_fails_before_array_load[mixed_split]`
- `test_invalid_metadata_fails_before_array_load[chunk]`
- `test_invalid_metadata_fails_before_array_load[missing_runs]`
- `test_invalid_metadata_fails_before_array_load[duplicate_run]`
- `test_missing_index_refused_before_source_access`
- `test_unrecognized_chunk_refused_before_any_source[chunk5]`
- `test_unrecognized_chunk_refused_before_any_source[Chunk-5]`
- `test_unrecognized_chunk_refused_before_any_source[chunk_5]`
- `test_unrecognized_chunk_refused_before_any_source[chunk]`
- `test_unrecognized_chunk_refused_before_any_source[unknown]`
- `test_unrecognized_chunk_refused_before_any_source[chunk6]`
- `test_receipt_tamper_fails_before_array_read[split]`
- `test_receipt_tamper_fails_before_array_read[sources]`
- `test_receipt_tamper_fails_before_array_read[files]`
- `test_receipt_tamper_fails_before_array_read[pack]`
- `test_array_tamper_fails_before_array_read`
- `test_pack_metadata_drift_fails_before_array_read`
- `test_identity_file_symlink_escape`
- `test_output_symlink_escape_and_freshness`
- `test_recomputed_receipt_cannot_supply_paths`
- `test_recomputed_receipt_unknown_metadata_refused`

These fixtures use only fresh temporary files. One fake CALIBRATION metadata item
is interleaved between two TRAIN items, and has no source files. An explicit path
sentinel refuses any attempt to touch that excluded shard. Portable-load fixtures
retain historical `Z:/old/source/...` strings only as metadata. Symlink tests may
skip if host permissions do not permit creating links; a skip is not evidence
that link rejection was exercised.

Pending: independently reviewed access guard, draft test execution, targeted code
acceptance, consumer identity gathering, and coordinator-owned recovery close-out.
No real pack build/load or scientific training is authorized by this document.
