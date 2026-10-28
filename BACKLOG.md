# S-BACKLOG-1 (2026-10-28, zero SUMO, branch c4/backlog)

**Verdict: on the evidence completed so far (partial), the long-credit CQR does NOT qualify at 120, 180 or 300 s. Keep zero credit beyond the 90 s trusted horizon and probes off for cycle 4.**

## Done and verified
1. **Converter fixed** (`wms.data.c4_episodes`): exact interval entrance-waiting `entry_backlog` for paired-branch tapes, requests resolved via the base episode, reconciled to the tape's J ledger (fail closed). 7 tests plus `test_looplong` (49) pass.
2. **Cycles 0-2 re-exported** from existing tapes (60 pairs, 30 worlds; originals kept). All pre-existing arrays byte-identical (1680 files); tape shas and J match the returns receipts. **All 60 pairs reconcile to the authoritative returns**, so `MissingCostClocks` no longer fires. Data: cpuP `~/backlog/reexport/pairs_c0*.a1.backlog`.
3. **E3 set**: 72 registered calibration/validation worlds x {native, ours_c1, coopoff_c1} = 216 episodes, 216/216 reconcile to the authoritative 120/300 s clocks. MAPPO excluded (no loadable actor); fit worlds never read. cpuP/cpuQ `~/backlog/stage/e3cal.*`.

## Qualification (WM2 b8797242, K=8, each cycle's own actor)
Rows: 43 loop pairs at 300 s (c00 8/8, c01 19/26, c02 16/26), 17 more at 120 s only, 23 E3 pairs at 120 s.

| horizon | fit/cal/val worlds | coverage (>=0.8) | ranking mean / lower95 (>0.5) |
|---|---|---|---|
| 120 s | 4/9/17 | 0.71 | 0.65 / 0.41 |
| 180 s | 1/8/16 | 0.75 | 0.58 / 0.33 |
| 300 s | 1/8/16 | 0.875 | 0.33 / 0.08 |

Loop+E3 calibration: +6 cal, +7 val worlds at 120 s only, no E3 ranking evidence yet; same verdict.

## Caveats
- Only 4 loop worlds hash to `fit`; 180/300 s rest on one.
- CPU cost 1-6 h per 300 s pair under shared load; 6 workers died. 120/180 s rows of 300 s pairs are prefixes of one rollout; a check on one pair showed different draws (sd 109 vs 31, n=8): equivalence unproven.
- `probe_cqr_*.json` have finite corrections but come from partial rows and have no validation: do not enable probes.

## Artifacts
cpuP `~/backlog/stage/CQR_PARTIAL` plus `STAGE_MANIFEST.json` (hashes); copy on gpuA `.../restart20261026/backlog_stage/` (outside arm). Niced workers still run (`~/backlog/run/pids*.txt`).
Finish on a GPU: `cqr_rows.py --device cuda`, then `cqr_fit.py`.
