# S-COOPBANDIT — prospective design, 2026-10-31
Owner approval: shared REGISTRATION.md S-COOPBANDIT 23:38; actual cap 1,600,000 sim-s.
Question: can observable traffic conditions predict when a cooperative option saves delay?
Primary labels are exact configured-SUMO effects, not uncertainty-free observable predictions.
## Frozen sources and safety
Reuse tools/leverprobe backend, intervention, observation and bitwise-fingerprint contracts.
Frozen E3 SOURCE_MANIFEST b3c031adb6bc2b5c2bca160bea308f83ba5567b97c1cd0e853eddcd8367a8193.
Speed-only MAPPO169 d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca.
SUMO1.27.1, textbook arrivals/drivers, current noise unchanged, native lanes, original safe_accel.
Targets enter BEFORE original safety; log requested/raw/issued/physical and override counts.
Full reruns from time0 only; no saved-state continuation. Every changed branch must match
the baseline physical/observable/commands/metrics/counters/MAPPO-memory prefix BITWISE.
Before interventions, one 900s baseline repeat per cell must match through t900 exactly.
## Worlds, sampling and options
Fresh TRAIN reserve981000–981079: 80 candidate worlds, exact cell map tools/coopbandit/protocol.py.
Cells: shares .05/.10/.20/.40 × demands1900/2400/2700. First4 per ordinary cell;
12 per .20/.40×2400/2700 cell. Both seed guards receive ONLY exact seed/demand pairs.
Audit before registration: cpuQ210524/gpuR359340 safe names, no hits; exact registry ranges clean.
No forbidden paths/truth/type-assignment files read; filename traversal prunes them in advance.
Each candidate follows MAPPO after native warmup300 through t1140. Record decision states
at300/360/420/480/540, all600s outcomes and observable-only snapshots; no outcome-based selection.
For high cells rank worlds by number of these states satisfying lever-probe Q75 CAV-count
AND density plus observed ramp AND adjacent-upstream pair; ties by seed. Choose top5 worlds
and earliest4 remaining (ordinary/unselected-by-rule mix, NOT guaranteed matched ordinary).
For other8 cells choose first3 worlds. Retain all5 times/world: 60 worlds/300 states.
Report actual dense/ordinary coverage, no relaxation/replacement or efficacy-based stopping.
Group k=2 nearest observable upstream mainline CAVs within[-100,0]m, fixed at decision.
O0=MAPPO; O1=L2 metering; O2=L3 wall; O3=L4 observed-ramp gap opening; O4=L1 gap hold.
O2 available only if selected members span lanes0/1; O3 only both in receiving lane1
and ramp observed in[-100,75.44]; unavailable/small groups execute O0, explicitly labelled.
All group members share chosen family. L1/L4 tau2/3s; L2/L3 speed decrement3/6m/s;
T30/60s. Balanced incomplete grid: setting=(candidate repeat+decision index)%4;
setting0/1/2/3=(low30/low60/high30/high60). One setting per state, all4 families.
Solo reruns preserve JOINT membership/target context, intervene on only that member.
Solo subset: first2 selected worlds in EACH high cell, all5 clocks (40states), O2/O3×2solos.
## Exact precharged budgets (all prefixes, failed physics and repeats count)
DATA: screening80×1140=91,200; repeats12×900=10,800;
group options60×4×(900+960+1020+1080+1140)=1,224,000;
O2/O3 solos8×4×5100=163,200; TOTAL1,489,200 <=1,500,000.
LIVE: reserve99,360 <=100,000; total planned1,588,560; unallocated11,440, no automatic retries.
Each attempt O_EXCL owned marker, hash-bound source/design/approval, durable precharge BEFORE reset.
Host cpuQ initially one nice19 CPU; measure first successful runs before scaling up to24 workers,
load admission<=24, disk>=2GiB, memory>=8GiB; yield to existing experiments, own PIDs only.
gpuR optional spare CPUs only after equivalent hash/guard gates and >=2GiB disk; no GPU training.
## Labels, features, traces and offline learning
D_h=J_h(O0)-J_h(option), h120/300/600 after decision, J=fleet speed-deficit+source wait.
C_G=D_group-sum(D_solo); preserve components/safety/TTC<3/hard braking/override share.
Observable features: sensed lane positions/speeds/gaps/density/queue counts, ramp presence,
CAV telemetry/group geometry, time and assigned option setting; NEVER seed/cell/demand/true share/types.
Store full observed decision roster and1Hz full-observable traces/action audits for COOPBEHAV;
lengths are public vehicle geometry, not driver truth. Trace limitations/observable support explicit.
Model per option: sklearn GradientBoostingRegressor,100trees,depth2,min_leaf8,Huber,random_state0.
Five world-grouped outer folds; inner4fold world-grouped signed residual calibration.
Confidence radius=finite-sample95% quantile of per-world MAX overprediction across times/options;
option chosen by max predicted D only if predicted D-radius>0, else O0. No behavioural threshold.
Outer-test labels never calibrate their policy. Full-data model uses grouped5fold calibration.
Report OOF regret vs max(0,true D), oracle helpful fraction, importance and partial dependence;
world-bootstrap2000 seed20261031, empirical oversampled mixture primary, cells descriptive.
## Live dev test and export
Fresh dev981080–981091, one/all12cells, never fitting/calibration.12×2×1200=28,800 for learned/O0,
native warmup300 then900scored; RSU decides t300..1140 every60s, setting=decision_index%4.
Paired J600/J900 fromt300, whole-world percentile95% bootstrap2000, safety and action counts.
Restricted per-decision oracle at baseline t300 only:12×4×900=43,200 (setting0, horizon600).
This is an exact single-decision upper bound, NOT an oracle closed-loop episode.
Executed-option C_G on first8worlds' first non-O0 event at t<=540: reference+2solo full reruns
<=8×3×1140=27,360; later choices frozen to factual schedule/targets; joint factual tail reused.
No qualifying event yields NA/unused budget. Never claim C_G for unmeasured groups.
Export EXAM v1: states.jsonl (observable roster/features), labels.csv (long option/horizon),
group_credit.csv, world-group splits, provenance SHA manifest, schema README; WM predicts sign/rank.
Deliver COOPBANDIT.md<=70lines, CSV/PNG, registration progress/outcome line, commit each step/no push.
