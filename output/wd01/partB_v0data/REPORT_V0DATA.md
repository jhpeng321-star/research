# WD-01 Part B round 0 - REPORT: V0 native-value dataset v0.2 (partB_v0data)

Generated: 2026-09-28T06:55:42.747930+00:00
Repo: `E:\rd-wt\wd01r` branch `zcode/wd01r` HEAD `849b199fdff3`

## Objective

Consolidated V0 (native-policy value) dataset v0.2 = chunk 1 (WM-44, pre-fix fleet, copied from the wd01n worktree with every
file sha256-verified against the old MANIFEST before and after the
copy) + chunk 2 (WM-47, post-fix fleet, newly built with the
chunk-agnostic builder from the chunk-2 campaign logs).  ZERO
SIMULATION: read-only over the campaign run logs; HOLDOUT worlds
never opened (in-code guard; holdout_accessed=false).

## Deliverables

- `v0lib.py` - streaming trace builder + prefix-sum contract-delay
  labels (equivalence to part4 delay library tested), entry
  waiting, window exposure, insertion derivation
- `featurize_v0.py` - V0 featurizer on the part4/partB skeleton
- `build_v0data.py` - CHUNK-AGNOSTIC resumable per-world builder
  (--chunk; worlds in worlds/<chunk>/; cav_speed_mode resolved
  receipt -> observed -> per-chunk config; pre_fix/p40 flags;
  HOLDOUT refused in code)

v0.2 additions:

- `copy_chunk1_v02.py` - verified chunk-1 migration (hash before +
  after, done markers validated, HOLDOUT file guard)
- `verify_v0data.py` - re-hashes EVERY MANIFEST-listed file;
  PASS/FAIL counts (results/verify_v0data.json)
- `sanity_v02.py` - chunk-2 row contract + speed-mode fallback
  census + TRAIN/native sanity table (results/sanity_v02.json)
- `backup_v02.py` - dataset backup + SHA256SUMS + verification
- `td_report.py`, `make_data_card.py`, `make_report.py`,
  `tests/`, `MANIFEST.json`, `DATA_CARD_V0.md`,
  `worlds/<chunk>/` npz + parquet per source (git-ignored)

## Commands

```powershell
python copy_chunk1_v02.py      # verified chunk-1 copy (done)
python build_v0data.py --chunk chunk2 `
    --plan D:/research-dense/output/campaign_stage1_v2/results_plan_chunk2/WORLD_PLAN.json `
    --runs D:/research-dense/runs/campaign_stage1_chunk2 `
    --default-speed-mode 27
python verify_v0data.py        # re-hash every MANIFEST file
python td_report.py            # TD identity report
python sanity_v02.py           # chunk-2 contract + sanity table
python make_data_card.py       # DATA_CARD_V0.md
python make_report.py          # this file
python backup_v02.py           # backup + SHA256SUMS + verify
python -m pytest tests -q
```

## Build result

- sources: 222 across chunks ['chunk1', 'chunk2']; decision times 15700; label rows 57854; TD tuples 78500
- [chunk1] sources: 110; plan: `D:\research-dense\output\campaign_stage1_chunk1\results_plan_wm44\WORLD_PLAN.json` (sha 9c196a2c5123...); runs root: `D:\research-dense\runs\campaign_stage1_chunk1`; default speed mode: 31; speed-mode resolution: {'observed': 110}; HOLDOUT listed: 14; incomplete skipped: 2
- [chunk2] sources: 112; plan: `D:\research-dense\output\campaign_stage1_v2\results_plan_chunk2\WORLD_PLAN.json` (sha 75fefc75aecc...); runs root: `D:\research-dense\runs\campaign_stage1_chunk2`; default speed mode: 27; speed-mode resolution: {'config': 112}; HOLDOUT listed: 14; incomplete skipped: 0
- wall-clock chunk-2 build: 653 s single process below-normal priority
- per-source median build: 5.3 s
- TD consistency: 42154 tuples, 0 violations, worst 5.82e-11 veh·s
- file verification: PASS=666 FAIL=0

- chunk-2 contract: 68624 parquet rows checked, 0 violations of chunk_id=='chunk2' / cav_speed_mode==27 / pre_fix==False
- chunk-2 speed-mode fallback (config default 27) used on 112 of 112 sources (receipts record null; physics rows carry mixed 31/27 pre/post-restore values)

Sanity table (TRAIN, native source; total delay label, veh·s per demand; descriptive only, NOT interpreted):

| demand | tau | c1 mean | c1 median | c2 mean | c2 median |
| --- | --- | --- | --- | --- | --- |
| 2000 | 30 | 424.1 | 393.6 | 418.9 | 315.3 |
| 2000 | 300 | 4227.5 | 4271.5 | 4188.6 | 2783.9 |
| 2400 | 30 | 1029.6 | 1016.7 | 1287.4 | 1355.8 |
| 2400 | 300 | 10119.9 | 9751.7 | 12573.1 | 13824.4 |
| 2600 | 30 | 1748.6 | 1707.2 | 1610.3 | 1567.6 |
| 2600 | 300 | 17348.1 | 18432.7 | 15857.2 | 16074.6 |
| 2700 | 30 | 2361.4 | 2227.2 | 2255.4 | 2226.4 |
| 2700 | 300 | 23570.4 | 22638.6 | 22464.0 | 21992.1 |

## Verification (pytest)

The suite covers: label computation vs the part4 delay library
(synthetic windows exact to 1e-9; real windows to 1e-6),
TD-tuple consistency on ALL built sources, chunk/fix-flag
columns on every row (per-chunk fleet modes: chunk1 -> {31},
chunk2 -> {27}), resumability, featurizer shapes and the
no-vType guarantee, entry-waiting truncation semantics, HOLDOUT
and incomplete-receipt handling, and the in-code HOLDOUT guard.
Final pytest run of this session: see the session report.

## Known limitations / notes

- The base source is the campaign incumbent arm (fleet-uniform
  damping), included per DESIGN §4 Step 0; `policy`/`source`
  columns let training filter to native-only.  Post-t0 base and
  native share the CRN realization - treat cross-source rows as
  correlated.
- Two fleets in one file: chunk 1 pre-fix (31), chunk 2 post-fix
  (27); filter on pre_fix / chunk_id before training.
- Chunk-1 native receipts are missing for w8469m2400p40t420 and
  w8473m2700p40t540 (base only) - inherited from the chunk-1
  build; re-run the chunk-1 builder if their branch receipts
  appear.
- Branch insertions.jsonl logs are truncated for some worlds;
  realized insertions are derived from the physics departed_ids
  (cross-checked exact against the logs where present).

## Git log

```
849b199 wd01: gate v2 amendment 2 (TTC<3 -> G-2, owner (B)), FROZEN registration WD-01-C2L1 (native Part A v1, chunk-2 CAL look #1), replicate PILOT registration WD-01-REP-P
485be28 wd01: freeze native Part A v1 (working model, gate NOT met), structural LC design note, Part B V0 training plan (docs only)
3a7519b wd01 part3q fit iter1 (folds f1-f4, 91 windows): joint 4.0435 -> 3.6241 (winners: lc_scale_disc 0.0014 -28% disc term, merge_early_prob 0.15, mprob/tau/pcoop/cd at base); MERGE-COUNT LEVERS REJECTED BY THE JOINT LOSS -- cooldown 16 cuts lcM30 term -29% but ttc3 +0.64 (pred over-exposes queue TTC, bias -1.76) => measured trade frontier; members 1-4 = p3o bases + softmax draws
ac63e95 wd01 part3q: merge-event/mevents files + freeze script 'in' key syntax fix
5aa5253 wd01 part3q: diagnosis + baseline k-fold gate (base = p3o iter2 candidate on the 17 TRAIN roots) -- merge LC +57.1%@H30/+30.7%@H60 pooled (folds +33..+118%), pred merges EARLY vs truth (median 7.5 vs 15.0 s @H30; dist-to-end median 26.5 vs 18.1 m); truth lc_s frame fixed to the accel lane (p3k convention); M2 causes: 34910 checks, 5045 accepted, thinning rej 2408; cross-fitted k-fold gate: H30 (c) FAIL on ttc3 0.790 (dphys 0.824), H60 PASS; 595 base rolls + L0 committed
```
