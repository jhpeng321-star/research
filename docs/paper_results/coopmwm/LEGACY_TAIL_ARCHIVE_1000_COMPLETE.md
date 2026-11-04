# Legacy-tail SHA archive complete

At unix1793758318.01 the registered worker completed normally:
**476,840,470 bytes / 240 files / 4 SHA-certified batches**. Its owner has exited;
this is completion, not a restart trigger. Compact identity/metrics/digests remain
in warm1/ARCHIVED_OLD_TRAIN_SUMMARIES; full originals are retained on gpuA in
`/root/cpuq_archive_20261103/legacy_tail1000`.

Completion cpuQ free space was6,549,172,224bytes (6.10GiB). A later independent
probe at unix1793759097.97 reports7,198,883,840bytes (6.70GiB); subsequent changes
are not attributed to this archive. No SUMO, inference or budget charge.

All four source-removal certificates sum exactly to the plan's240files and
476,840,470bytes; the manifest, remote proofs, source identities and per-file
fresh FD/reference/SHA checks were retained. Both regular movers stayed running.

Evidence: `LEGACY_TAIL_ARCHIVE_1000_STATUS_1793759099215170000.json`.
Never redispatch this completed worker.
