# DAMPING-SNAP — final report

Branch: `wmn1/damping-snap` (base `956f75f` = A9).
Owner-approved fix of the origin-0 snapshot misalignment for the 263 TRAIN
`base` / `native_base` items excluded by owner decision (b).

## What was built (all under `_out/`, TRAIN only, read-only campaign inputs)

| artifact | size | sha256 |
|---|---|---|
| `_out/pack_damping` (263 items, 315,863 ticks, v3/K192) | 5.21 GB | manifest `63310f5bd1d0c01e2588a77ad110658da5385fee991ee5c55f9cba6953c9db72`, INDEX `abc84f38e777d4ded1959a51c7f34b68c07e23590ed97978ef90677451727512` |
| `_out/shards_damping` (receipts + slot_index/DONE for identity) | 248 MB | per-run ALIGN/DONE |
| `_out/labels_overlay_damping/overlay_outcomes.parquet` (263 groups, 18,147 rows) | 1.27 MB | `8f672ab5720a895a33cdf928891bae64ef8595d535b29f1c5c8aa6444b58426a` |
| `_out/identities_damping` (`wm_n1_train_identity_sidecar/1`, 21,463 vehicles) | 1.79 MB | receipt-not-file `26c78813a075cde6c8e7ecb9982067c1b8e4733d93eedc2f38d58d5abe25957f`; identities `93dbdf35b05516ad7370727cd19f2a56f3869152e6d524281fb1e0713cbe9d72` |

- Featurizer option: `features.featurizer.build_shard(..., snapshot_at_t0=False)`
  (default OFF, frozen bytes). ON sets `t_origin = item t0` for a full-run
  `base`/`native_base` log; first physics row `t0+0.5 s`, `decision_t[0]==t0`;
  pre-`t0` departure history is still read from the same log for the counter
  channel. Verified `t_origin=240`, first present tick => `t=240.5`, decision
  grid `240,270,...,810`.
- Pack builder: `dspeed.build_damping` (streams the same pack path as
  `dspeed.build_pack`); item metadata `policy` (`base->damping`,
  `native_base->native`) + additive marker `damping_snap=True`, zero packed
  outcomes.
- Overlay: `dspeed.build_damping_overlay` from the existing
  `labels_cov_20261004T072826Z/base_train_outcomes.parquet` (no relabeling);
  per item asserts `t_origin==t0`, `first_physics==t0+0.5`,
  `decision_t[0]==t0`, a label row stamped exactly at `t==t0`, and the frozen
  label `policy` class. All 263 accepted, 0 refused.
- Identity sidecar built with `train.identity_sidecar.build_identity_sidecar`
  over `pack_damping` + `shards_damping`; loads and returns an `int64[192]`
  row per item.
- Train wiring: `train.dataset.is_outcome_supervision_excluded` now returns
  `False` for marked `damping_snap` items (pack_v3 origin-0 copies stay
  excluded); `train.idea_a` gains `--extra-identities` (repeatable, positional
  with `--extra-corpus`), a per-pack `_manifest_entries`/`_pos_entry` map, and
  `_extra_identity_sidecar` (pack-local position dispatch). Default OFF.
- CPU check `dspeed.cpu_check_damping`: 2 pair-dense batches over
  `pack_v3 + pack_damping`; batch 0 has 3 damping items / 2 damping-native
  pairs; total 263 extra items, 322 damping-native pairs, 161 pairs involving
  an extra item. Materialized `w8411m2700p02t240__base`: `damping_snap=True`,
  `excluded=False`, outcome mask non-empty (5 taus), `label_t0_rows=5`.

## Tests
- New synthetic: `features/tests/test_snapshot_at_t0.py`,
  `dspeed/tests/test_damping_items.py` (item filter / marker / overlay
  accept + fail-closed misalignment), `dspeed/tests/test_extra_corpus_default_off.py`
  (`--extra-identities` default-off + repeatable).
- `features/tests dspeed/tests`: 123 passed, 1 skipped (dspeed pack absent).
- Related train tests (cli seed, label t0 exclusion, identity sidecar, pair
  dense, label isolation): 153 passed, 4 skipped.
- `ruff check` on changed/new files: clean.

## Pre-existing failures (confirmed at base `956f75f`, unrelated to this change)
- `train/tests/test_dataset.py` (6) and `train/tests/test_contrastfix_recovery.py` (10),
  `test_idea_a_train_interface.py` (2), `test_train_cli.py`, `test_trainer.py` (3),
  `test_cross_piece.py`, `test_format_k192_train.py` (2): synthetic fixtures omit
  `DONE.json`/`INDEX.json` that the committed `label_isolation.py`/`dataset.py`
  require. Reproduced verbatim in a detached-HEAD worktree.

## Known limitation / caveat
For the 63 worlds carrying both `base` and `native_base` re-aligned copies, the
composite pair group also holds the pack_v3 copies, and `_reference_of` picks
the pack_v3 `native_base` reference first, so the extra `base` item pairs
cross-pack (extra treatment vs zero-supervised pack_v3 reference). All 63 extra
`native_base` items are therefore seen as treatments rather than references.
`any-extra` pairs exist (161); the re-aligned outcomes are correctly attached
to the extra items. Fixing reference selection to prefer same-pack siblings
would change the shared A9 pairing used by the dspeed trajectory path and was
out of this task's scope.

E: free after build: ~77 GB (> 12 GB floor).
