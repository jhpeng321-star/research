# Authoritative17:03 continuation receipt — 2026-11-01 +08

## Active work (never duplicate)

- gpuA remote supervisor473053/start889857083/boot
  a81555d7-176e-4ed8-bc7d-896fc89cc914.
- gpuR remote supervisor924719/start4214495421/boot
  d19e6898-efd0-4288-a901-d63b21f11163.
- Local sole collector25240/create_filetime134379970282244431,
  runs/rb2int_finish_v3. First poll16:57:08:12/216complete world jobs; cache reused.
- Local cache-only publisher44988/create_filetime134379973214244749,
  runs/rb2int_interim_v3. First loop12/216,4matched v5/v4worlds; no dispatch/transfer.
  REUSED_PUBLICATION.json binds corrected16:57receiptSHA
  5b30d8041f8559d2f1ae47c67d4d30c0260fec9a03c115d0c2357621db7a1bd5
  and markdownSHA10a4acfeb2577026ba7e3b2eecfa839ad93572152ee834dcd56cd7ddfc75e0e4.

Remote read-only16:55/56status/resource receipts are FINAL_*_STATUS_INTERIM_1656 and
INTERIM_*_RESOURCES_165*. Both supervisors healthy/no failures; GPUs99% utilized.
No process signals, no inference/source/recipe change and no remote redispatch.

## Published evidence / schedule

- First interim5512f402; corrected CI publication04885264. Two worlds969436/969590
  (one1900/.05cell), allfivearms/K8. v5−v4 delay CRPS+205.6/+2333.2/+11427.9veh-s
  at120/300/600; HDV60−8.7m. Small available slice; no model selection/switching.
- Road-only short direction accuracy45/40/40% for both models at30/60/90. Total
  team accuracy bounds0..100% on this slice: no pair resolves without short source
  waiting. Do not use the road proxy to authorize cooperation/planner switching.
- Next scored-cell refresh~19:57; reserved8 immediately once all eight v5/v4 K8
  worlds exist. Refresh each view~3h, always reconstructing scored strata.
- Final frozen four-model panel/gate/switching dueNov2~10:00; not yet complete.
  After216cache jobs, sole collector's exact progress-report guard STOP yields
  final scoring/publish ownership to the cache-only publisher. Incomplete deadline
  remains explicitly incomplete. Read INTERIM_CONTINUATION and numerical followups.

## Completed transitions (do not restart)

CollectorV2/9660, publisherV1/24320 exited exact source guards; recovery waiter30228
validated both creation times/exits then launched successors once. Its COMPLETE
receipt is runs/rb2int_ci_resume_v1/COMPLETE.json. PublisherV2/25688 finished the
corrected analysis but stopped on simultaneous collector staging. Its exact
commit-only STOP, completed receipts and report were retained; manual04885264
committed them. PublisherV3 checks that STOP/exit and source lineage before reuse.
Commit-window serialization36703851 avoids the shared-index startup collision;
foreign non-collector staging still stops safely. No successful analysis rerun.

Verification:36broad CI/analysis tests PASS9.01s;12latest targeted continuation
tests PASS2.22s (overlapping;38distinct checks total); Ruff PASS. Numerical-zero
variance becomes NA rather than fake equal-endpoint precision; other point
estimates/CI design unchanged. This report is frozen engineering evaluation,
not another fitting/selection round. S-WMTEACH shared READY note updated.
