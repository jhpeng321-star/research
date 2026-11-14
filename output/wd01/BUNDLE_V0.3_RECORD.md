# Part B V0 plain bundle v0.3: build record

Built 2026-09-29 by the B1 continuation session (`make_v0_bundle.py`, one
command, zero simulation).  Supersedes the interim `v0.3-pre` bundle
(2026-09-28 16:22 UTC), which predated the chunk1_fix27 stage and the
`--value-model` flag.

## Artifact

- **Bundle:** E:/research-dense/output/partB_bundles/partB_v0_bundle_v0.3.zip
  - 69,237,931 bytes;
  - sha256 33cdaf0644259e85f98301ca9fb0172d05e55a400dbff9784299a2fcf1461f0a;
  - 940 entries, each listed with its sha256 in BUNDLE_MANIFEST.json;
  - `value_model: "net"` (new reserved `--value-model {net,gbm}` flag;
    pass-through to README/BUNDLE_MANIFEST only).
- **Code:** zcode/wd01 @ 8d98208 (dirty worktree): wdpaths.py, partB_v0/
  and part4/partB/, LF endings, no __pycache__.  Model/loss/eval math is
  untouched (B3 owns the v0.3 trainer in zcode/wd01v2).
- **Data:** the V0 dataset v0.3 at
  E:/research-dense/output/wd01_partB_v0data_v03 (MANIFEST sha256 at build
  time recorded in `partB_v0_bundle_v0.3.record.json`).  914 data files
  verified (0 mismatches) before zipping; 22 superseded chunk-1 p40
  sources' files omitted with their MANIFEST entries kept; HOLDOUT worlds
  never opened (0 entries, 44 listed across chunks).

## Data state at build time (results/build_v03_last.json)

- v0.2 copy: 889 files, sha256-verified before AND after, 0 mismatches.
- chunk1: 110 sources (22 p40 TRAIN/CAL marked superseded_by='chunk1_fix27',
  files kept; trainer default --exclude-superseded drops them).
- chunk2: 112 sources.  chunk3: 92/98 sources (10 worlds pending: their
  native RUN_RECEIPT.json not landed; chunk-3 SUMO still running).
- chunk1_fix27 (NEW): the 12 TRAIN/CAL p40 worlds' BASE sources built from
  the verify27 mode-27 bases (E:/research-dense/runs/campaign_stage1_verify27,
  receipts record cav_speed_mode=27; *.lock_refused_* dirs ignored); the
  native branches (E:/research-dense/runs/campaign_stage1_chunk1_fix27, 98
  runs = 7 arms x 14 worlds) had not started, so all 12 worlds are pending
  their native sources and remain in missing_fix27_worlds.  The 2 HOLDOUT
  worlds (w8463/w8464) are listed but never opened.  Fix27 plan verified
  against the chunk-1 WM-44 plan (id/split/t0/p40, 12 non-HOLDOUT worlds)
  from the byte-identical reviewed copy
  E:/rd-wt/wm56/output/campaign_stage1_v3/results_plan_fix27/WORLD_PLAN.json
  (sha256 0cfaf0cfb931e8fbbfef78f56f1a7db5374190188af44db4474b7620eeda585b).
- Verification on the root: verify_v0data PASS (978/978 file hashes),
  td_report worst violation ~1e-11 (0 violations), DATA_CARD_V0.md +
  REPORT_V0DATA.md regenerated (85,462 label rows, 326 sources).

## Re-run when more runs land (idempotent, list-pending)

```
python build_v03.py \
  --fix27-plan E:/rd-wt/wm56/output/campaign_stage1_v3/results_plan_fix27/WORLD_PLAN.json \
  --fix27-runs E:/research-dense/runs/campaign_stage1_chunk1_fix27 \
  --fix27-bases-root E:/research-dense/runs/campaign_stage1_verify27
```

(pick up the campaign's own
E:/research-dense/output/campaign_stage1_v3/results_plan_fix27/WORLD_PLAN.json
once it exists).  Then rebuild the bundle:

```
python make_v0_bundle.py --data-root E:/research-dense/output/wd01_partB_v0data_v03 \
  --code-tree E:/rd-wt/wd01u/output/wd01 --value-model net --tag v0.3 --smoke-test
```

## Smoke self-test

Clean unzip, `python -m partB_v0 train --smoke --device cpu --data ../data`
(5 members): rc=0, DONE printed, data verified, OUTPUT_MANIFEST 13/13
re-hash OK, results.zip 34,481,671 bytes / 15 entries with no ckpt/
(sha256 975451b6060fc8c81fb129ee9374257f85d6cfe680a3303314e757388223b778).

## Known limits

- The numpy non-writable warning and the v0.3 trainer requirements
  (per-member budget, no partial members, concurrent ensemble, OOB
  selection) are B3 work in zcode/wd01v2, deliberately untouched here.
- chunk3 and chunk1_fix27 natives pending at build time: rebuild both the
  dataset and the bundle when the campaigns finish (commands above).

## Integration note (2026-09-29, 06:30, world model)

After merging B3a (3caa21a), the test suites give:

- partB_v0: 60 passed;
- the D: hook suite: 18 passed;
- partB_v0data: 32 passed, 4 skipped, **1 failed**. The failure is
  `test_make_v0_bundle.py::test_bundle_smoke_self_test[ts]`.

Cause: B3a's eval baselines (ridge/GBM) call a train/test split. B1's tiny
synthetic bundle fixture has n_samples = 1, so the split raises "the
resulting train set will be empty".

Fix (queued for the B3a follow-up): the baselines must degrade gracefully on
tiny data (skip, with a note in the report, below a minimum row count).
Real data is unaffected.
