- 2026-09-29 02:25 local: ZCode task bb698587 (B3, v0.3 trainer) was CANCELLED.
  - Authority: the owner, relayed by main discussion.
  - Reason: the provider stalled. The ZCode CLI logged 12× MODEL_STREAM_IDLE_TIMEOUT, and the task made no file changes for 70 min.
  - The work was split into B3a (73018ac2) and B3b (c4539f67).
- 2026-09-29, WD-01-DEV-2 addendum (draft, pending the seed test).
  - REP-P found the campaign SUMO deterministic after t0.
  - A2 step 2's "below single-realisation noise" is re-interpreted as model
    error (see A2_ERRATUM_DRAFT.md).
  - The validation look count is unchanged.
  - The chunk-2 look #1 (C2L1) is still unrun.
