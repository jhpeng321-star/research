# S-COOPFREE resume receipt

## OWNER02:21 fresh resume
Production615088/start48394231 and workers615093-615096 verified live/nice19;
publisher15284/creation2026-11-01T02:05:24 was verified, then identity-checked/retired
for the observer-only amendment. Replacement publisher5020 is live for12h;
receipt `tools/coopfree/evidence/HOLD_PUBLISHER_5020.json`. Never duplicate production/publisher.
TRAIN reserved1,649,760/1,650,000:240sim-s remain. Corrected u1 can finish; updates3/6/9
require prospective owner budget/phase amendment. Preserve all historical charges;
do not count optimizer replay as fresh batches. See RESUME_OWNER0221.json.
Read-only learning-check/pairwise learned-vs-HOLD observer extension CPU6/6PASS/SUMO0;
its deployment receipt supersedes old publisher15284; physics supervisor unchanged.

## OWNER01:48 HOLD-credit migration (fresh-session priority)
The inherited u1 arms use MAPPO counterfactuals. Any inherited `policy_u001.pt`
is a **legacy MAPPO-credit checkpoint**, not the requested HOLD-credit learner.
Legacy local publisher17964 was identity-checked and stopped before this migration.
One-shot cpuQ boundary gate591782/start48339232 waits for the existing batch to drain,
then writes YIELD and parks ONLY exact owned supervisor339173, after its current arm
closes and before new legacy dev physics. All existing rollouts/checkpoints are retained.
The original supervisor has now been retired after closing every plant; YIELD was
renamed to its retained receipt. Corrected production is RUNNING on cpuQ,
PID615088/start48394231, four nice19/single-thread workers. Do not invoke launch again.
See `python -m tools.coopfree_hold_ops status`; original branch physics is not repeated.
New exact HOLD baselines can reuse the joint trials for objective2D_G. Existing solos
leave nonexploring members on MAPPO: their C_G is NOT exact all-member-HOLD C_G.
Full replacement of those solos exceeds the remaining2M cap; exact HOLD-credit
diagnostics cover66 complete windows; all156 windows/24worlds drive objective2D_G.
New455tasks=156exact HOLD baselines(161,820sim-s)+299exact replacement solos.
Totalprecharge1,690,800sim-s at dispatch; corrected optimizer/dev/discovery pending.
Corrected proposals publish under `tools/coopfree/publications/hold/`, NOT legacy exports.
New observerPID15284 is live for12h; receipt `HOLD_PUBLISHER_15284.json` records
its exact command/log. It publishes corrected milestones and commits only its
publication/documentation paths; it never dispatches physics or retries a failed arm.
The observer also copies relocated tapes to durable
`E:/Temp/opencode/coopfree20261101_observable_archive`; first64 files SHA-verified.

Branch `c4/coopfree`; preserve every committed/uncommitted file and all remote roots.
Read `docs/paper_results/COOPFREE.md`, `COOPFREE_DESIGN.md`, and `tools/coopfree/evidence/` first.
Explicit owner approval is in the shared REGISTRATION.md at the 23:58 COOPFREE budget line.

## Inherited run receipts (historical; superseded by HOLD continuation above)
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
Replacement publisher PID17964 was live before this migration; now retired by exact
identity-checked PID. Its receipt gives the retained original operational log.
It waits through active physics and publishes updates, paired CIs/safety, and detector/
clustering/C_G summaries automatically, then exits on COMPLETE or durable STOP.

<!-- COOPFREE PUBLICATIONS BEGIN -->
## Corrected HOLD-credit production
Public artifacts: `tools/coopfree/publications/hold/`; verify its `MANIFEST.json`.
Legacy MAPPO-credit checkpoint is retained remotely and excluded from corrected learning.
Owned cpuQ supervisor PID615088/start48394231; alive=True; four nice19 single-thread workers.
New HOLD physics: 260/455 complete tasks; 156 baselines; exact solo diagnostics for 66/156 windows.
Corrected learned checkpoint pending; no zero-update or legacy checkpoint is advertised as corrected.
### OWNER02:21 learning self-check
- Completed corrected fresh updates: []; next scheduled check u3.
- Status: awaiting first corrected update.
- TRAIN remaining: 240 sim-s; one fresh >=150-window joint-only lower bound: 135000 sim-s (HOLD/solos extra).
- **Updates3/6/9 are budget-blocked. Prospective owner phase/budget amendment required; no replay or automatic dispatch.**
- DEV learned-minus-HOLD J600/J900: pending both complete paired panels; no marginal-CI subtraction.
Budget: 1690800 precharged / 1488840 closed-actual sim-s; hard cap2,000,000; no refunds/retries.
TRAIN exploration credit is not paired dev-policy performance.
<!-- COOPFREE PUBLICATIONS END -->
