# DECIDE → INT current integration routing — 2026-11-01 ~05:55 +08

Implementation is already delivered and integrated through `e8302db3`:
paired expected-delay picker, owner0.8 confidence gate, breakdown/strict-pair/
selected-switch reliability, interval coverage, CRPS and climatological Brier.
No new simulator switch or picker change is needed from this resume.

## Which publication to include

- `E:/agent-work/c4-rb-decide/HANDOFF.md` has the moving latest-refresh pointer.
  At reconciliation it points to `refresh_20261031_214355Z/INDEX.md`:
  WM2=20; FULL-v1/A2=6 each; FULL-v3=5; B2=3; C2=unavailable complete K8 worlds.
  Use each `matched_MODEL/INDEX.md` for its exact-world WM2 comparison.
- The existing DECIDE collector PID4984 is positively identified and live until
  09:00+08,1200s interval. Its claim is
  `E:/agent-work/rb-shared/DECIDE_REFRESH_0900_v2.claim.json`.
  It will commit newer publications and update the marked handoff block.
- FULL-v4/source22cc32f6 remains under INT's already-running noon collector.
  Use that collector's local saved shards/credibility reports. DECIDE's six-source
  collector does not cover FULL-v4; no second collector has been launched.

## FULL-v4 API compatibility

The generic `tools.wmdecide credibility` CLI already accepts any identity-valid
local evaluator shard directories. Its remote snapshot-root allowlist does not
restrict local scoring. For a new completed source use the existing CLI with
`--allow-partial --worlds ID ...` and the same explicit IDs on WM2. Output must
be NEW; complete K8 is required for every requested world. Reuse INT's finished
reports instead of scoring twice. `tools.wmdecide.publish --matched` exports
verified source reports without scoring and rejects different world sets.

Sparse/singleton-stratum CIs remain unavailable, not zero-width evidence of
superiority. Keep breakdown, all-pair preference, **selected-switch** reliability
and coverage together; improved absolute-delay CRPS does not qualify the0.8 rule.
EVAL/HDV60 adoption gates remain authoritative.

Resume receipt and test history are in
`docs/paper_results/wmdecide/RESUME_0555.md` and
`tools/wmdecide/VERIFICATION.md`. This handoff is documentation-only; all prior
jobs, claims, saved reports, terminal receipts and implementation are preserved.
