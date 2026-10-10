# FEAT1 v5 report — WM-N1 featurization complete

Generated 2026-10-01T05:57:02Z on branch `zcode/feat1run5` (workspace `E:/rd-wt/feat1run5`), main-authorized finish without a process pool.

## Corpus

- verified shards: **2445** (assertion 2476 inventory - 13 non-featurized - 18 v5 exclusions = 2445 == 2445 verified shards)
- ticks: **3,169,005** | decision rows: 48,900 | k_max: 128
- bytes: pack 35,069,206,977 (35.07 GB, all 15 arrays + outcomes) | shard v1 receipts 19,307,474,191 (12 v1 arrays + json/meta/outcomes; the 5 Idea A arrays are receipted separately: 17,469,120,360 B)
- splits: {'CALIBRATION': 308, 'TRAIN': 2137}
- by batch: {'chunk1': 436, 'chunk1_fix27': 84, 'chunk2': 448, 'chunk3': 448, 'chunk4': 648, 'truearms_chunk2': 112, 'truearms_chunk3': 112, 'walls_pilot_chunk4': 157}
- by policy: {'damping': 240, 'fixed': 504, 'switching': 1104, 'native': 256, 'pi': 184, 'wall': 157}

## v5 final run (6 x sequential, no pool)

- launched 2026-10-01T04:32:59Z -> finished 2026-10-01T05:00:09Z (wall 27.2 min; subset sizes [103, 102, 102, 102, 102, 102])
- PIDs [11156, 36048, 34024, 30000, 14396, 32252], --workers 1 each (in-process sequential path `corpus/featurize_all.py:_run_batch`), --idea-fast, BELOW_NORMAL, force never used, no shard overwritten
- prelaunch gates: free_ram>=8 OK=True, free_virtual>=10 OK=True, D:>=40 OK=True

## J4 amendment-2 relabel

- 146 wall-arm runs flagged; 59 road-end tokens relabelled (IDEA_A_DONE receipts)

## Non-featurized runs (13, never readable)

- `chunk1/w8427m2600p05t360__switching2` (TRAIN): is_complete=False receipt_status=None
- `chunk1/w8466m2700p40t300__switching3` (CALIBRATION): is_complete=False receipt_status=None
- `chunk1/w8467m2600p40t360__switching2` (TRAIN): is_complete=False receipt_status=None
- `chunk1/w8469m2400p40t420__native` (TRAIN): is_complete=False receipt_status=None
- `chunk1/w8469m2400p40t420__switching1` (TRAIN): is_complete=False receipt_status=None
- `chunk1/w8471m2700p40t540__pi` (TRAIN): is_complete=False receipt_status=None
- `chunk1/w8471m2700p40t540__switching2` (TRAIN): is_complete=False receipt_status=None
- `chunk1/w8473m2700p40t540__native` (TRAIN): is_complete=False receipt_status=None
- `chunk1/w8473m2700p40t540__switching1` (TRAIN): is_complete=False receipt_status=None
- `chunk1/w8473m2700p40t540__switching2` (TRAIN): is_complete=False receipt_status=None
- `chunk1/w8474m2600p40t300__switching1` (TRAIN): is_complete=False receipt_status=None
- `chunk1/w8476m2600p40t480__switching3` (TRAIN): is_complete=False receipt_status=None
- `walls_pilot_chunk4/w8697m2700p40t240__w-mid-early` (TRAIN): is_smoke=True

## Excluded runs (18, errored twice: subset run + alone retry)

- `walls_pilot_chunk4/w8658m2600p10t360__w-lo-early`: WallTokenError: wall w8658m2600p10t360__w-lo-early:wall1: non-CAV member(s) ['mainline.217'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8664m2600p10t360__w-lo-early`: WallTokenError: wall w8664m2600p10t360__w-lo-early:wall1: non-CAV member(s) ['mainline.226'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8667m2700p10t420__w-lo-early`: WallTokenError: wall w8667m2700p10t420__w-lo-early:wall1: non-CAV member(s) ['mainline.249'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8669m2700p10t360__w-mid-early`: WallTokenError: wall w8669m2700p10t360__w-mid-early:wall1: non-CAV member(s) ['mainline.344', 'mainline.347'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8669m2700p10t360__w-mid-late`: WallTokenError: wall w8669m2700p10t360__w-mid-late:wall1: non-CAV member(s) ['mainline.344', 'mainline.347'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8670m2700p10t300__w-lo-late`: WallTokenError: wall w8670m2700p10t300__w-lo-late:wall1: non-CAV member(s) ['mainline.229'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8682m2400p20t240__w-lo-late`: WallTokenError: wall w8682m2400p20t240__w-lo-late:wall1: non-CAV member(s) ['mainline.152', 'mainline.151'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8684m2700p20t360__w-lo-early`: WallTokenError: wall w8684m2700p20t360__w-lo-early:wall1: non-CAV member(s) ['mainline.211'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8686m2600p20t240__w-mid-late`: WallTokenError: wall w8686m2600p20t240__w-mid-late:wall1: non-CAV member(s) ['mainline.185'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8692m2600p40t300__w-lo-late`: WallTokenError: wall w8692m2600p40t300__w-lo-late:wall1: non-CAV member(s) ['mainline.203'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8694m2600p40t240__w-lo-late`: WallTokenError: wall w8694m2600p40t240__w-lo-late:wall1: non-CAV member(s) ['mainline.200'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8697m2700p40t240__w-lo-late`: WallTokenError: wall w8697m2700p40t240__w-lo-late:wall1: non-CAV member(s) ['mainline.163'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8698m2600p40t300__w-lo-late`: WallTokenError: wall w8698m2600p40t300__w-lo-late:wall1: non-CAV member(s) ['mainline.211', 'mainline.210'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8700m2600p40t420__w-mid-late`: WallTokenError: wall w8700m2600p40t420__w-mid-late:wall1: non-CAV member(s) ['mainline.314'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8703m2600p40t540__w-lo-early`: WallTokenError: wall w8703m2600p40t540__w-lo-early:wall1: non-CAV member(s) ['mainline.334'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8704m2700p40t480__w-mid-late`: WallTokenError: wall w8704m2700p40t480__w-mid-late:wall1: non-CAV member(s) ['mainline.377', 'mainline.378'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8705m2700p40t300__w-lo-early`: WallTokenError: wall w8705m2700p40t300__w-lo-early:wall1: non-CAV member(s) ['mainline.147', 'mainline.148'] -- wall features are CAV-self-report only (no HDV input)
- `walls_pilot_chunk4/w8705m2700p40t300__w-lo-late`: WallTokenError: wall w8705m2700p40t300__w-lo-late:wall1: non-CAV member(s) ['mainline.196', 'mainline.197'] -- wall features are CAV-self-report only (no HDV input)

## Quarantine ledger (MOVE only, never delete/overwrite)

- v2 `20260930T224632Z`: moved 0, kept complete 366, kept Idea-A-pending 6
- v3 `2026-10-01T01:40:45Z`: scanned 1825, ok 1824, moved 1: `chunk4/w8659m2700p10t420/base`
- v3 `2026-10-01T03:05:52Z`: scanned 1834, ok 1830, moved 4: `chunk4/w8659m2700p10t420/base`, `chunk4/w8659m2700p10t420/native`, `chunk4/w8659m2700p10t420/native_base`, `chunk4/w8659m2700p10t420/pi`
- v4 `2026-10-01T04:30:46Z`: scanned 1859, ok 1850, moved 9 (all partial in-flight dirs of chunk4/w8660m2700p10t300 + walls_pilot w8660):
  - `chunk4/w8660m2700p10t300/native`
  - `chunk4/w8660m2700p10t300/native_base`
  - `chunk4/w8660m2700p10t300/pi`
  - `chunk4/w8660m2700p10t300/switching1`
  - `chunk4/w8660m2700p10t300/switching2`
  - `chunk4/w8660m2700p10t300/switching3`
  - `walls_pilot_chunk4/w8660m2700p10t300/w-lo-early`
  - `walls_pilot_chunk4/w8660m2700p10t300/w-lo-late`
  - `walls_pilot_chunk4/w8660m2700p10t300/w-mid-early`

## Verify + manifest

```json
{
 "ok": true,
 "n_runs": 2445,
 "n_items": 2445,
 "n_arrays": 15,
 "errors": []
}
```
```json
{
 "ok": true,
 "n_runs": 2445,
 "errors": []
}
```
- manifest `D:/rd-staging/wmn1_feat/upload_manifest.json` sha256 `a1a5a871b44a7145c3f976d85b48b860639b07cc80f1470d0ea0d12a74496db9` (2445 runs, splits {'CALIBRATION': 308, 'TRAIN': 2137}, refused [])
- `pack_v2/SHA256SUMS` sha256 `da7381dd7e4dab31375d9338f2cfb6e63b23b9fb58049ad882339a2af1b9f5dc`

## Disclosure chain

- 1. first attempt (policy-keyed shard dirs) ABORTED: replicated-arm + cross-batch key collisions (fixed1/fixed2, switching1..3; chunk2/3 vs truearms) -> shard key redefined to batch + run-dir arm suffix, policy demoted to metadata (commit bf5d81a); superseded shards v1 tree kept at D:/rd-staging/wmn1_feat/shards (untouched)
- 2. inventory refresh df30b17 -> restart 6 workers, then owner-approved 6->12 restart (commit 4a7c343 + bcf1ddc, logs_v3, launcher 14980 / featurizer 32340)
- 3. 09:28 crash #1: BrokenProcessPool after a worker raised wall_tokens_v04.WallTokenError the parent could not unpickle (ModuleNotFoundError: wall_tokens_v04 in the executor result reader); 1825/2463 done; parent sys.path fix + per-run [start]/[end]/[error] logging (commit 573a02a)
- 4. ~10:53-11:01 crash #2, SAME signature: the first fix added only the wm_n1/wd01 roots but wall_tokens_v04 lives in output/wd01/partB_v0data; featurize_all parent now imports features.actions_adapter too + per-run exclusion-and-continue (commit ae7550c); 1-worker repro: 9/10 in-flight written, w8658m2600p10t360__w-lo-early reproducibly raises WallTokenError
- 5. 11:15 v4b relaunch (10 workers, featurizer PID 1132): the pool STALLED (workers idle 67 min); the integrator stopped PID 1132 + its children WITH MAIN'S AUTHORIZATION (no run was corrupted: shards are per-run dirs with receipt-gated writes)
- 6. final finish WITHOUT a pool: quarantine v4 (1859 scanned / 1850 verified / 9 moved), 613 remaining runs split into 6 DISJOINT subset parquets, 6 independent detached featurize-all processes --workers 1 (sequential in-process path, no pool), --idea-fast, BELOW_NORMAL, per-run [start]/[end]/[error] logs + PID files (logs_v5); 595 written, 18 errored, all 18 re-run ALONE once, all 18 errored again -> excluded; count assertion 2476-13-18=2445 == verified 2445; pack_v2 + corpus verify ok:true + corpus.idea_a verify ok:true + upload manifest
- 7. --idea-fast byte-identity evidence: corpus tests fast==slow (48 seeded scenes, commit b1d9598); v4-resume 3/3 TRAIN runs 61 sha comparisons (logs_v4/byte_identity_resume_20261001.txt); v5 runs-after-first-1824: 3/3 v5-written TRAIN runs (incl. J4 wall arm w8706m2400p40t360__w-mid-late) re-derived through the SLOW reference path, 61 sha comparisons ALL equal (logs_v5/byte_identity_v5_slowcheck.txt)

## Artifacts

- shards: `D:/rd-staging/wmn1_feat/shards_v2`
- pack: `D:/rd-staging/wmn1_feat/pack_v2`
- manifest: `D:/rd-staging/wmn1_feat/upload_manifest.json`
- logs_v5: `D:/rd-staging/wmn1_feat/logs_v5`
- quarantine_v2: `D:/rd-staging/wmn1_feat/quarantine_v2/20260930T224632Z`
- quarantine_v3: `D:/rd-staging/wmn1_feat/quarantine_v3`
- quarantine_v4: `D:/rd-staging/wmn1_feat/quarantine_v4/20261001T042854Z`
