# WMFIXMERGE resume receipt 2 — 2026-10-31 07:12 (frozen session continuation)

Previous session froze ~06:56 after committing 1303d4a2 (DRAFT launch-ready).
This session resumes gate execution. No finished work redone; no duplicate launches.

## Discovered state (measured, not assumed)
- gpuA zero-head panel PID 100400 ALIVE (restarted 06:22, --memory-fraction 0.35,
  code label 71a986d7_resume_batch11): batches 0-15 complete (0-10 from earlier
  runs, 11-15 current), batch 16 in flight at 07:12. ~625 s/batch; single-track
  completion of groups 17-35 projected ~10:45. Out: validation_zero_head/.
- gpuA co-residents: candB pilot 102300 (2.7 GB) + candB future workers
  108853/108854 (2x ~5.9 GB, transient — the batch-11 OOM pattern). Free ~17.6 GB
  but my sanctioned slice is <= 12 GB (addendum 03:55 item 4).
- gpuR is NOT free despite addendum 07:20 ("B-L stopped"): candB relaunched a lean
  WMFAST2 pilot at 06:52 (PID 481173, supervised by 481170 with a launcher — it may
  relaunch). ~11.6 GiB used, only ~12.5 GiB free vs panel peak 12,098 MiB tensors
  + context: does not fit; squeezing in risks OOMing their pilot. NOT attempted.
- gpuB unreachable (connection closed).

## Decision (safety-first, addendum intent preserved)
1. Keep gpuA panel single-track; do NOT add a second 12 GB co-resident process
   (exceeds sanctioned slice; would risk the candB pilot's transient workers).
2. Conditional shard: if the gpuR candB pilot exits, stays out >= 5 min with
   >= 13.5 GB free, launch groups 26-35 there from a hash-verified staged tree
   (identical code/inputs/CRN), then copy batches into gpuA validation_zero_head/.
   Never races gpuA's sequential cursor (it reaches 26 no earlier than ~09:40;
   copies land 26 first at ~+11 min after launch).
3. Gate chain after 36/36: zero_head_compare -> WMFIX_RETARGET.md (three-way
   WM2 / zero-head-entry vs WMREQUAL3 numbers, threshold-free rule) ->
   harness_contract (residual_remains per decision) -> real feasibility
   (--device cuda, <= 12 GB slice) -> REVIEW.json (MAIN) -> LAUNCH_READY final.
4. Feasibility runs only in the residual-REMAINS branch (by design).

## Own PIDs this session
gpuA: none launched yet (monitoring only). Planned shard PIDs will be recorded
in the shard LAUNCH.json on the host that runs them.
