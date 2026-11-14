# Part B V0 plain bundle v0.2: build record

Built 2026-09-28 at about 22:00 local by the world model session, directly
(an owner-authorised exception to the ZCode-only rule, relayed by main
discussion).

## Artifact

- **Bundle:** E:/research-dense/output/partB_bundles/partB_v0_bundle_v0.2.zip
  - 50,907,926 bytes;
  - sha256 d295e629082a51b27d336ef2167e61e718a4b6fc53ea55b5f96c73a8f17cee1a;
  - 915 files, each listed with its sha256 in BUNDLE_MANIFEST.json.
- **Code:** zcode/wd01t commit 0e44f23, the reviewed B2 code: wdpaths.py,
  partB_v0/ and part4/partB/. Line endings were converted to LF.
- **Hand patch:** code/partB_v0/outputs.py only. The diff is below.
- **Data:** the V0 dataset v0.2 at E:/research-dense/output/wd01_partB_v0data
  (MANIFEST sha256 f7fd64116b7c3f0024c5168fe829eb58988a31816cb28e14ebca30f366cc5dfc).
  All 666 data files were verified at copy time.
- **Staging:** E:/research-dense/output/partB_bundles/staging_v0.2
  (README_PLAIN.md and requirements_plain.txt).

## Verification

1. `sha256sum -c` on the zip: OK.
2. Clean unzip into E:/research-dense/tmp/bundle_v02_test_2202, then
   `--smoke --device cpu --data ../data` from code/:
   - rc = 0;
   - 666 data files verified and 222 sources selected, with no HOLDOUT;
   - OUTPUT_MANIFEST has 13 entries, all re-hash OK;
   - results.zip is 32.9 MiB, contains no ckpt/, and its sha matches.
3. Re-running the same command on the completed OUT_DIR:
   - every member logs "resumed at step 6 … already at step 6 >= planned 6;
     skipping", so nothing is retrained;
   - then it re-evaluates and prints DONE (rc = 0).

## Known limits

These are fixed in B2's continuation ea1bae89 and will ship in v0.3:

- there is no automatic OOM batch halving (the README tells the owner to add
  `--batch-size 256`);
- there is a numpy non-writable warning;
- the probe's T4 memory line is an analytic estimate.

## Patch (0e44f23 → bundle)

```diff
106a107,111
>         # v0.2 bundle patch: logs/ are still being appended when the
>         # manifest is written, and ckpt/ (resume-only optimizer state) is
>         # not shipped in results.zip -> neither is hashed here.
>         if rel.startswith(("logs/", "ckpt/")):
>             continue
125a131,134
>                 continue
>             # v0.2 bundle patch: ckpt/ is resume-only (optimizer state,
>             # .prev rotations); it stays in OUT_DIR but is not downloaded.
>             if p.relative_to(out).as_posix().startswith("ckpt/"):
```
