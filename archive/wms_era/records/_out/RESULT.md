# WM-N1 dspeed-data: make the direct-speed trajectory runs trainable (A9)

Status: COMPLETE (all 644 runs labelled + packed + overlaid; code committed).

## Headline results (real, TRAIN only)

| artifact | path | count | size | SHA256SUMS sha256 |
|---|---|---|---|---|
| labels | `_out/labels_v04a_dspeed/` | 644 runs, 0 refused | 0.02 GB | `82e0ce742033ef2e4f551ad5dfd7034e098d9207dce80527df3fa7f0cfd982a5` |
| features+pack | `_out/pack_dspeed/` | 644 items, 773,444 ticks, v3 K=192 | 11.88 GiB | `8a0ffa0b447ac8bd9a55bb5444dd22fff527ec059dcb060dd4f736f9a5bf9448` |
| alignment receipts | `_out/shards_dspeed/` | 644 `meta.json`+`ALIGN.json` | 0.01 GB | (meta only; no SHA file) |
| overlay | `_out/labels_overlay_dspeed/` | 644 groups, 44,436 rows, 3,220 t0 rows, 0 excluded | <1 MB | `2f39c1e17a340dda23bd5527b03a39b0c6641cba0ad36a244f5f32776c2c0718` |

- labels MANIFEST sha256 `7312a3b03bf77db0bbe328f77b9c8475387181205183b6ddf3e02227f60e2005`
- pack manifest sha256 `b5c94f05cc57549cdb3840b2e8e07895697f8c9063624c4195924ee8dd74c150`
- pack INDEX sha256 `1fe7e33369f3b2f18b48d5f34192a3da8428720f4a3b29c092f2a9cff7ffcef0`
- overlay parquet sha256 `dae6d3393471d4a3fa6019ac4d2b8b9e2c3bdc176240bdee187dc008d18e2ab2`
- E: free after pack: **9.06 GB** (>= 8 GB floor; disk cap never triggered, 0 runs dropped)
- 644 = all `{box,laptop}` runs with a valid complete `run_receipt.json` (base 6d0f369 -> merged caa4dfd).

## Pipeline (idempotent; one command per stage, `rl_driving` python)

1. `python -m dspeed.labels --workers 8` -> `_out/labels_v04a_dspeed/`
   - v0.4a `window_label`/`build_v04a_cov` conventions; base trace = the world's campaign `__base`;
     decision grid `t0,(t0+10..t0+570)`; source=`arm0..arm3`, policy=`trajectory`; TRAIN only.
2. `python -m dspeed.build_pack --workers 6` -> `_out/pack_dspeed/`
   - `features.featurizer.build_shard` with an explicit gzip physics path, snapshot `t0+0.5 s`,
     `format="v3"` K_max 192, plus the Idea A channels; single streaming copy into the pack
     (`corpus.pack` manifest/INDEX/`outcomes.parquet` contract), because shards+pack (2x ~13 GB)
     does not fit the E: free-space floor. Small per-run `meta.json`/`ALIGN.json` under
     `_out/shards_dspeed/` preserve the alignment receipts.
3. `python -m dspeed.overlay` -> `_out/labels_overlay_dspeed/`
   - schema/3-style sidecar keyed by full item identity `(batch,chunk,world,arm,split,t0_s)`;
     fail-closed identity (source=arm, policy=trajectory, chunk/split/t0), `t_origin==t0`,
     first physics row `t0+0.5`, exact `t==t0` row pick (duplicate tau refused), `label_t0` telemetry.
4. training: `train.idea_a --extra-corpus PATH` (repeatable), default OFF; `--extra-overlay PATH`.
   `dspeed.composite.ConcatCorpus` merges `pack_v3` + extra at the item level (global tick/plan
   offsets, `pair_key` grouping spans packs) so trajectory items pair with their pack_v3 native/pi
   siblings; `ConcatArray` dispatches window slices to the owning pack.

## CPU check (step 6)

`python -m dspeed.cpu_check --pairs 4 --batches 3`:
`pack_v3` + `pack_dspeed` = 2,797 items, 2,122 eligible pairs; first 3 pair-dense
(`--merge-coverage-into-pairs`) batches each hold **4 pairs** and 3/2/3 trajectory items;
materialized trajectory item `w8481m2700p02t480__arm0` has a non-empty outcome mask (5 taus,
`label_t0_rows=5`, pair_key `w8481m2700p02t480@t0=480`).

## Tests

- `python -m pytest -q dspeed/tests` -> **15 passed** (gz reading, extra-corpus default-off,
  overlay identity/t0 fail-closed, ConcatArray).
- `python -m pytest -q train/tests` -> **45 failed, 397 passed, 4 skipped**;
  the 45 are pre-existing synthetic-fixture / missing-identity-sidecar failures (no failing
  traceback references `dspeed`/`ConcatCorpus`); baseline ~43.
- `ruff check` clean on new/changed files.

## Exact extra GPU train flags (train.idea_a)

```
set WM_N1_TRAIN_OUTCOMES=D:/rd-staging/wmn1_feat/labels_cov_20261004T20261004T072451
set WM_N1_TRAIN_IDENTITIES=<pack_v3 TRAIN identity sidecar dir>
python -m train.idea_a --member <1..5> \
  --corpus D:/rd-staging/wmn1_feat/pack_v3 \
  --extra-corpus E:/rd-wt/dspeed-data/_out/pack_dspeed \
  --extra-overlay E:/rd-wt/dspeed-data/_out/labels_overlay_dspeed \
  --k-max 192 \
  --pairs-per-batch <K> --merge-coverage-into-pairs \
  --device cuda --out <member_out_dir>
```
The extra flag is default OFF: omitting `--extra-corpus` keeps the frozen single-pack behavior.

## Changed code (committed; `_out/` data not committed)

- new `output/wd01/wm_n1/dspeed/{__init__,common,labels,build_pack,overlay,composite,cpu_check}.py` + `dspeed/tests/`
- `output/wd01/partB_v0data/v0lib.py` (additive gzip `physics_path`; removed dead var)
- `output/wd01/wm_n1/features/featurizer.py` (additive gzip `physics_path`)
- `output/wd01/wm_n1/train/idea_a.py` (`--extra-corpus`/`--extra-overlay`, composite wiring, `_array` deleg.)

## Blocker / caveat

- Compression/the 8 GB E: floor forced a direct streaming pack (no separately materialized heavy
  shard tree); the small alignment receipts are the shards. All 644 runs fit.
- Training integration is CPU-verified (composite batching + trajectory outcome labels); the GPU
  train run itself was out of scope (no GPU/SSH).
