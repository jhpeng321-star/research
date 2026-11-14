# ABORTED attempt 2 — WD-01-C4L1 (2026-09-30 ~10:39-10:47 local)

Preserved, never used for any number. Nothing from this attempt feeds
the C4L1 evidence.

## What happened

- Launch 1 (attempt 1): crashed BEFORE any row — params assertion
  compared `param_set_id()` (raw-bytes sha) to the registered LF-sha.
  Receipt: `c4l1_run_attempt1_aborted_paramsassert.out` (committed).
  The registered "crash before any row -> ONE disclosed restart" was
  used for attempt 2.
- Launch 2 (attempt 2, this shard): the shell wrapper's 20-minute
  timeout KILLED the process externally (not an eval crash) after world
  1 completed and world 2 was partial. Cause of the slowness: the
  laptop was saturated by a CONCURRENT other-session job (PB-RESCORE1
  `build_truearms_labels.py --workers 4`, PIDs 36616 + 4 spawn
  children, started 10:10/10:16; not touched). Per the brief: wait,
  don't oversubscribe.
- State at kill: `shard_0.jsonl` = 467 raw rows (w8621 complete = 280
  rows, w8622 partial = 187 rows). NO aggregation ran. NO metric value
  was printed, read, or seen — only per-world row counts. "No re-run
  after seeing any number" is satisfied.

## Why the rows are NOT usable (mechanical, outcome-independent)

`eval_a2.window_rows` keeps H when `w + H <= log.t_end`, and the
chunk-4 `__native_base` logs run to t0 + 600 s, so attempt 2 scored
windows up to `t0 + 600` (e.g. H120 from k = 1..16; 280 rows/world =
210 main + 70 guards_on). The FROZEN registration (REGISTRATION_C4L1.md
section 4) requires `s + H <= t0 + 540` (H120 k = 1..14; 192 main + 64
guards = 256 rows/world). This is a code-vs-registration mismatch that
the partial run exposed; it is provable from the row counts alone and
is independent of any outcome. Continuing attempt 2 would have baked an
unregistered window rule into the look.

## Disposition

- Files: `shard_0.jsonl` (467 lines), `shard_0.opened.json`.
- Disposition: aborted_protocol_deviation; superseded by the attempt-3
  continuous pass in `results/c4l1/` (fresh shards, registered
  `t0 + 540` horizon, disclosed in REPORT_C4L1.md and the ledger).
