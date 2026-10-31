# S-COOPFREE resume receipt

Branch `c4/coopfree`; preserve every committed/uncommitted file and all remote roots.
Read `docs/paper_results/COOPFREE.md`, `COOPFREE_DESIGN.md`, and `tools/coopfree/evidence/` first.
Explicit owner approval is in the shared REGISTRATION.md at the 23:58 COOPFREE budget line.

## Live owned run
- Host cpuQ; immutable `/home/ubuntu/coopfree20261101_build5`.
- Pipeline PID339173, Linux start_ticks47835664; nice19, one CPU thread.
- `python -m tools.coopfree.ops status` is read-only and validates that exact process identity.
- `DISPATCH_BUILD5.json`, `CANARY_BUILD5.json`, `STATUS_SCREEN_PROGRESS.json` are local receipts.
- First baseline/repeat are bitwise identical, each1140actual=charged; latest snapshot12540closed
  actual/41040reserved, no STOP. No optimizer update had completed at that snapshot.
- Do not invoke launch again: the pipeline owns screens, repeated baseline gates, zero-update
  physical canary, whole156-window update(s), checkpointing, u1/final dev, and discovery.
- It reads public peer COOPBANDIT snapshots before updates; a missing final comparator causes
  a sleeping wait after training, not a fake comparator or extra SUMO dispatch.

## Files and recovery
Remote `CHARGES.jsonl`, per-job START/RESULT/CLOSE, `PLAN_NNN.json`, `UPDATE_NNN.json`,
`policy_uNNN.pt`, `DEV_NNN.json`, `DISCOVERY_NNN.json` are authoritative.
`STOP.json` is durable; do not clear it or retry incomplete charged physics automatically.
`BUDGET_STOP.json` is a clean pre-batch stop: no group/sample-count truncation.
No physics was run in build1..4. Build2's CPU schema-test failure and build4's pre-Popen
binary-banner preflight rejection are retained. Build5 CPU18/18PASS/0skips/0sim-s;
additional existing-dependency pickle/clustering CPU QA also passed with unchanged binding.
Existing third-party dependency links point to `/home/ubuntu/coopbandit20261031_cpu1/deps`;
scientific runtime/checkpoint and25 bound Python sources are unchanged.
The bind CLI's repeated three-host audit timed out before writing any binding; build5 instead
reused the already completed three exact exposure receipts, transparently recorded locally.
Future collection should copy only explicit public JSON receipts/checkpoints/traces, never scenarios.
Commit collected milestones separately; no push or changes to peer PIDs.
