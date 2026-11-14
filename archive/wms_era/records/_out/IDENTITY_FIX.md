# IDENTITY_FIX — A9 composite-corpus identity sidecar (URGENT)

Branch: `wmn1/a9` @ `956f75f` (code `output/wd01/wm_n1`). Status: DONE (dry run green).

## Symptom
CPU dry run of `train.idea_a` with
`--corpus D:/rd-staging/wmn1_feat/pack_v3 --extra-corpus E:/rd-wt/dspeed-data/_out/pack_dspeed --extra-overlay ...`
failed in `IdeaACorpus._resolve_global_positions`:
`pack_v3 item 'w8481m2700p02t480'/'arm0' does not map to a unique global manifest position`.

## Root cause
`IdeaACorpus.__init__` indexes `self._manifest_pos` from `self._v1.manifest["items"]`,
but the A9 `ConcatCorpus` dataset items carry OFFSET-SHIFTED `tick`/`plan_row`.
The dataset->manifest lookup `_item_position_key` therefore never matched an
extra-pack item, and the resolver held a single sidecar loaded against
`pack_v3`, so extra-pack items would have queried the wrong sidecar anyway.
Two further composite gaps were then exposed and fixed: `ConcatArray` lacked
tuple indexing (rollout slice `arr[t0:t1, ..., 0]`) and `ConcatCorpus.__getitem__`
dropped the composite provenance (`_pack`/`_manifest_gpos`) needed by the
resolver. A fourth: each sidecar's canonical IDs are PACK-LOCAL, so integer
equality across packs was not a physical vehicle identity.

## Fix
1. `dspeed/identity.py` (new): builds a TRAIN identity sidecar for the dspeed
   pack with the pack_v3 schema/conventions
   (`identities.npy`/`global_positions.npy`/`vehicle_map.json` +
   `TRAIN_IDENTITY_RECEIPT.json`). Vehicle IDs are reconstructed with the EXACT
   frozen `features.featurizer.parse_run` slot assignment over each run's
   `physics_steps.jsonl.gz` (read-only), and cross-checked against the shard
   `slot_vid.json` / `meta.json` / `ALIGN.json` and the pack INDEX
   `input_sha256`. A symmetric strict loader keeps every fail-closed check.
2. Composite resolver (`train/idea_a.py` + `dspeed/composite.py`): each item
   carries its own pack index + ORIGINAL per-pack manifest position
   (`_pack`/`_manifest_gpos`); pack_v3 items use the pack_v3 sidecar, extra-pack
   items use their own sidecar (`--extra-identities` /
   `WM_N1_EXTRA_TRAIN_IDENTITIES`, default OFF). In composite mode rows are
   re-projected into ONE shared canonical index space built from the union of
   the packs' `(chunk, world, vehicle_id)` maps, so a trajectory item and its
   pack_v3 native sibling align on real vehicle IDs.
3. `IdentitySidecar` now retains `vehicle_map` (metadata only, never a net
   input). `ConcatArray` handles tuple indexing; `ConcatCorpus` records
   `_manifest_gpos`.
4. Synthetic tests + real 2-step CPU dry run.

## Artifacts (E:/rd-wt/a9/_out/)
- `identities_dspeed/` — dspeed sidecar, 644 items, 16,635 canonical vehicles.
  - `TRAIN_IDENTITY_RECEIPT.json` sha256 `28c27e049b5a79bcfbf9a2469ed5b575a5be0d3630b857101c5c15fda4fd71b7`
  - `identities.npy` sha256 `3abed33c632d81ec5e02ff5564ee30814ee589e318e836adfcbda499c3ceb4fb`
  - `global_positions.npy` sha256 `4e93cf0b802d23c8e26e1944bc7ebaf63aec0cb2807bafb83b65d92b61c7184b`
  - `vehicle_map.json` sha256 `f9bcaa1d403c32ebf0df1d80502aa56f51657c117942eff9dbe53fd916292155`
- pack_v3 sidecar receipt sha256 `46097207587f8c55cebc3dba35a5ccc96661f00118ec51cb7490e5be68f2f3d2`
  (E:/tmp/wmn1_train_runs_20261002/sidecars3_20261003T0005CN/identities).
- Cross-pack check: world `w8481m2700p02t480` @t0=480 dspeed `arm0` row and the
  pack_v3 `native` row share 138/138 real vehicle IDs.

## Exact GPU command additions (identity only)
```
--extra-identities E:/rd-wt/a9/_out/identities_dspeed
```
env: `WM_N1_EXTRA_TRAIN_IDENTITIES=E:/rd-wt/a9/_out/identities_dspeed`
(existing `WM_N1_TRAIN_IDENTITIES` for pack_v3, `WM_N1_TRAIN_OUTCOMES` unchanged)
Default OFF: with no `--extra-corpus` there is no extra identity path.

## Dry run (green, exact brief command)
`_out/dry2` (2 steps, CPU, K=192, pairs_per_batch 4, paired-effect plan-aware):
- step1 total=33.2043 arrival=0.8305 correction_crps=30.24 paired_effect=0.1072
  residual_nll=1.398 response_contrast=0.3162 rollout=0.3106
  contrast[pairs=4 out=4/20 hdv=4/328, hdv_id_aligned=true]
- step2 total=17.5222 arrival=0.7668 correction_crps=14.11 paired_effect=0.6266
  residual_nll=0.8446 response_contrast=0.8469 rollout=0.3246
  contrast[pairs=4 out=4/20 hdv=4/327, hdv_id_aligned=true]
- label_t0: asserted=50, violations=0 on both steps.

## Tests
- `dspeed/tests/test_identity_sidecar_physics.py` (6 synthetic pass).
- `dspeed/tests/test_composite_identity.py` (4 synthetic pass).
- `dspeed/tests/test_composite.py` tuple indexing (3 pass).
- Full `train/tests`: 43 failed, 406 passed — identical to the pre-existing
  43-failure baseline (no new failures). `dspeed/tests`: 26 passed, 1 skipped.
- `ruff check` clean on all changed files.
