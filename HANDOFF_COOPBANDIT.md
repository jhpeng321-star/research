# S-COOPBANDIT -> COOPPLAN / RB-INT / COOPBEHAV

First exact partial EXAM is published at
`E:/agent-work/c4-coopbandit/docs/paper_results/coopbandit/partial135/exam-partial/`.
135 complete-option states / 28 TRAIN worlds; exploratory completion order.
Use `MANIFEST.json` to verify all files. Full 300-state panel is still running.
No dev worlds enter this export. Split by `world_id`, never by decision state.

## Predictable contract
- `states.jsonl`: state_id, world_id, t0, observable feature dict, observed roster,
  exact baseline prefix digest. No demand/true penetration/type/driver features.
- `labels.csv`: state_id x O0..O4 x 120/300/600; D = fleet+source J(O0)-J(option).
  Positive D saves delay. `available` is required; unavailable choices are exact no-ops.
- `group_credit.csv`: measured joint/individual D and C_G = D_G - sum(D_i).
  Positive C_G does not imply positive joint D.
- `worlds.json` includes the prospective 60-world selection even in a partial export.
  Only states actually present in states.jsonl belong to the partial EXAM.

## Reusable option/rerun API (this worktree)
`tools/coopbandit/protocol.py`: observable(rows, now, setting), group(rows),
option(option_id, rows, setting, members=None). Joint nearest-two membership
and option availability derive only from observations. Settings 0..3 are
low30/low60/high30/high60; speed delta3/6 or gap tau2/3 as registered.
O1=L2, O2=L3 cross-lane, O3=L4 receiving-lane/ramp, O4=L1.
`tools/coopbandit/backend.py` reuses the original leverprobe pre-safety injector;
`runner.branch` shows full-rerun-prefix equality and exact metric subtraction.
Solos retain JOINT option/target context; do not recompute a singleton wall target.
Frozen leverprobe sources are in `E:/agent-work/c4-leverprobe/tools/leverprobe/`.
Frozen runtime E3 b3c031ad; MAPPO169 d9042cc5; SUMO1.27.1. Never weaken safety.
Python3.12 hosts need scoped `metric_compat.py` plus a full checkpoint bitwise gate;
do not replace the equality gate with tolerance. Runtime sources remain unchanged.

Raw 1 Hz observable/action tapes stay on originating hosts, under explicit jobs:
- cpuQ `/home/ubuntu/coopbandit20261031_v3_labels/jobs/<branch>/observable.jsonl.gz`
- gpuA `/root/autodl-tmp/coopbandit20261031_labels_v4/jobs/<branch>/observable.jsonl.gz`
- gpuR `/root/autodl-tmp/coopbandit20261031_labels_r_v5/jobs/<branch>/observable.jsonl.gz`
Baseline tapes: cpuQ `/home/ubuntu/coopbandit20261031_v3/jobs/s<seed>_screen/`.
Branch ID: s<seed>_t<clock>_O<n>[_solo0|_solo1]. Source receipts bind tape hashes.
Current root/allocation: `tools/coopbandit/evidence/RECOVERY_PLAN_V5.json`.
Only read explicit public observable/action/RESULT/CLOSE paths; all task restrictions apply.

Registered learner remains cautious: 0 OOF interventions on partial data despite
16/135 helpful states (best option D up to +6360 veh-s). Use the exact labels
to test WM sign/rank or drive the separately budgeted SUMO-oracle planner;
do not represent this as demonstrated learned cooperation.
