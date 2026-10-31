# S-COOPFREE resume receipt

Branch `c4/coopfree`; preserve every committed/uncommitted file and all remote roots.
Read `docs/paper_results/COOPFREE.md`, `COOPFREE_DESIGN.md`, and `tools/coopfree/evidence/` first.
Explicit owner approval is in the shared REGISTRATION.md at the 23:58 COOPFREE budget line.

## Live owned run
- Host cpuQ; immutable `/home/ubuntu/coopfree20261101_build5`.
- Pipeline PID339173, Linux start_ticks47835664; nice19, one CPU thread.
- Batch acceleration receipt: sidecar PID436993/start48005147, eight nice19 CPU workers.
  It temporarily pauses the original supervisor (stateT); do not launch another pipeline.
  Sidecar owns1082 disjoint unstarted arms; it excluded the original in-flight
  `u001_w013_solo1` and resumes the exact original supervisor after draining its allocation.
  Read `COOPFREE_OPERATIONS.md` and `BATCH_*` receipts before intervening.
- `python -m tools.coopfree.ops status` is read-only and validates that exact process identity.
- `DISPATCH_BUILD5.json`, `CANARY_BUILD5.json`, `STATUS_SCREEN_PROGRESS.json` are local receipts.
- Fresh-session identity check: PID339173/start47835664 still alive, no STOP.
- `STATUS_RESUME_1.json`: all24 screens/all12 repeats/zero-update canary complete;
  50,820closed actual/1,216,740precharged; first156-window batch running, no update yet.
  Precharge includes the entire joint+all-member-solo batch; it is not completed physics.
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

## OWNER00:44 checkpoint consumer
Publish each completed `policy_uNNN.pt` with its `UPDATE_NNN.json` and SHA256 for
S-COOPPLAN (`E:/agent-work/c4-coopplan`). At the recovery snapshot there was no learned
checkpoint: u1 requires all156 windows/all member solos before its optimizer step.
Never present the zero-initialized proposal distribution as a learned checkpoint.
For proposal encoding, pass the actual MAPPO **pre-safety** accelerations as anchors,
matching `execution.inject`; see the consumer guide, not post-safety commands.
The singleton read-only publisher polls every120s for12h; after verified receipts appear,
it updates this handoff/report and commits only its publication/documentation paths.
Its dispatch/log receipt is `tools/coopfree/evidence/PUBLISHER_DISPATCH_<PID>.json`.
Initial publisher8556 was stopped by exact owned PID after its successful first
publication to add complete paired-dev/safety/discovery reporting; its receipt/log
are retained. This observer-only maintenance does not pause/restart any SUMO worker.
Replacement publisher PID17964 is live; its receipt gives the exact operational log.
It waits through active physics and publishes updates, paired CIs/safety, and detector/
clustering/C_G summaries automatically, then exits on COMPLETE or durable STOP.

<!-- COOPFREE PUBLICATIONS BEGIN -->
## Automatically published proposal checkpoints
Consumer guide: `docs/paper_results/COOPFREE_PROPOSALS.md`.
Public artifacts: `tools/coopfree/publications/`; verify `MANIFEST.json` hashes.
Frozen remote root: `/home/ubuntu/coopfree20261101_build5` (cpuQ).
No learned checkpoint yet; the full156-window/all-member-solo first update is running.
Latest milestone: 136 complete jobs; 1216740 charged/146760.0 closed-actual sim-s.
Training-window credits describe exploration; they are not paired dev-policy performance.
The owned supervisor is paused while eight same-host workers drain only disjoint precharged arms; the sidecar resumes it for the optimizer step (see COOPFREE_OPERATIONS.md).
<!-- COOPFREE PUBLICATIONS END -->
