# S-COOPBANDIT -> COOPPLAN / RB-INT / COOPBEHAV

## Complete panel now ready
`docs/paper_results/coopbandit/full/exam/`: all300states/60TRAINworlds/12cells,
4,500 labels and240 credit rows; original SHA manifest and documented schema.
Full data ledger and completion receipt are in `full/`. See
`docs/paper_results/coopbandit/EXAM_SCHEMA.md` for settings/sentinels/split rules.
45/300states have a helpful option; oracle588.48veh-s/state (95%CI295.05..961.64).
Registered learned selector and companion diagnostics selectO0 throughout OOF CV.

**Useful exact group interaction for planner/WM testing:** s981057_t420/O3,
setting3 (tau3s/T60): D_G+755.39, solos-5203.75/-1481.94, C_G+7441.08veh-s.
This is the sole available measured case with BOTH D_G>0 and C_G>0; selected
TRAIN exploration, not live learned success. Its complete observable context,
exact credit and explicit raw tape paths are in `full/COOPERATIVE_CASE.md`.

Fresh12world live test COMPLETE: no options executed despite44/180eligible
decision opportunities; learned/MAPPO episode clocks bitwise equal, J600/J900
differences/CIs zero. Baseline-t300 restricted oracle had48unavailable no-ops;
it says nothing about later eligible decision opportunities. No executed-event
credit exists. Full report `docs/paper_results/COOPBANDIT.md`; live paired/safety/
descriptive COOPBEHAV receipts `docs/paper_results/coopbandit/live/report/`.
Final actual1,567,326/conservativecharged1,569,840sim-s; all jobs finished.

## Preserved first milestone

First exact partial EXAM is published at
`E:/agent-work/c4-coopbandit/docs/paper_results/coopbandit/partial135/exam-partial/`.
135 complete-option states / 28 TRAIN worlds; exploratory completion order.
Use `MANIFEST.json` to verify all files. The full 300-state panel is now complete above.
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

At the preserved partial milestone, the registered learner made 0 OOF interventions
despite 16/135 helpful states (best option D up to +6360 veh-s). The full-panel
learner also made 0 OOF interventions despite 45/300 helpful states. Use the exact labels
to test WM sign/rank or drive the separately budgeted SUMO-oracle planner;
do not represent this as demonstrated learned cooperation.

## Fresh-session verification — 2026-11-02
Reconciled clean branch `c4/coopbandit` at `e3dea23b` with the shared registration
completion line, DATA_COMPLETE/LIVE_COMPLETE and the live summary. All 11 files
bound by the full EXAM and live-report SHA256 manifests match their original bytes.
Counts verified: 300 unique states, 60 worlds, 4,500 labels, 240 credit rows.
The 1,452 successful-data ledger charges sum to 1,489,200 sim-s; including retained
failed attempts, diagnostics and live receipts gives 1,567,326 actual / 1,569,840
conservative charged sim-s. No new simulations or dispatches were started on resume.

Repeat owner-requested resume on 2026-11-02 started from clean `ca8b8d5d`.
Rechecked the shared COMPLETE registration and both completion receipts, then
independently verified all 11 manifest hashes and the 300 unique states / 60 worlds /
4,500 labels / 240 credit rows / 1,452 data-charge rows. All checks PASS.
The task is complete; this resume started no simulations or dispatches.

Fresh owner-requested resume on 2026-11-03 started from clean `f70f2c46`.
Reconciled the shared COMPLETE registration, DATA_COMPLETE/LIVE_COMPLETE and live
summary. All 11 original-byte EXAM/live-report SHA256 hashes PASS; independently
counted 300 unique states / 60 TRAIN worlds / 4,500 labels / 240 credit rows.
Verified 1,452 unique successful-data charges and 72 unique live charges;
including retained failed attempts/diagnostics gives 1,567,326 actual / 1,569,840
conservative charged sim-s, within the 1,600,000 cap. Main report remains 59/70 lines.
The first local verification encountered Windows' default GBK decoding on report
text; explicit UTF-8 verification passed without changing any artifact bytes.
All four steps were already complete; this resume started no simulations or
dispatches and preserved the existing completed results and handoffs.

Fresh owner-requested resume on 2026-11-03 started from clean `037dba23`.
Read git history/status, shared COMPLETE registration, both completion receipts,
and the final report/live summary before taking action. Independently hashed all
11 EXAM/live-report manifest files; every SHA256 matches. Completion and budget
remain as recorded above. No simulations or dispatches were started; all existing
artifacts and the 59-line final report are preserved.

Fresh owner-requested resume on 2026-11-03 started from clean `dc1a0ef6`.
Shared COMPLETE registration and DATA_COMPLETE/LIVE_COMPLETE confirm all four
steps already finished. Independent verification PASS: all 11 original-byte
manifest hashes, 300 unique states / 60 worlds / 4,500 labels / 240 credit rows,
1,452 unique data charges / 72 unique live charges, and the 59/70-line report.
Ledger totals remain 1,567,326 actual / 1,569,840 conservative charged sim-s;
data/live/global ceilings all PASS. No simulations or dispatches were started.

Fresh owner-requested resume on 2026-11-03 started from clean `18c017da`.
Read history/status, shared COMPLETE registration and completion receipts first.
Independent verification PASS: all 11 manifest hashes; 300 unique states / 60
worlds / 4,500 labels / 240 credit rows; 1,452 unique data and 72 unique live
charges; 59-line report. Actual 1,567,326 / charged 1,569,840 sim-s satisfy all
ceilings. All four steps remain complete; no simulations or dispatches started.

Fresh owner-requested resume on 2026-11-03 started from clean `e79309b1`.
Shared COMPLETE registration and both completion receipts were read first.
Independent verification PASS: all 11 EXAM/live-report manifest hashes, 300 unique
states / 60 worlds / 4,500 labels / 240 credit rows, and 1,452 unique data / 72
unique live charges. Reconciled actual 1,567,326 / conservative charged 1,569,840
sim-s; data/live/global ceilings PASS; final report remains 59/70 lines.
Existing completed work preserved; no simulations or dispatches started.
