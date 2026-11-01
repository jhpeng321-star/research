# S-COOPFREE — OWNER15:11 continuation, 2026-11-01
Status: +4M approved; preserve immutable build5 design/binding/STOP and corrected u001.
Purpose: discover cooperation through continuous actions; no option/move library or shaped rewards.

## Plant and actions
Frozen E3 b3c031ad / SUMO 1.27.1, speed-only MAPPO169 d9042cc5; native lanes/noise unchanged.
RSU geometry selects ALL observed CAVs in [-100,75.44] m, including ramp/any lane.
These are existing merge-zone sensor boundaries, not learned-behaviour thresholds.
Each member chooses absolute target speed and positive time-gap, held W=10/20/30 s.
Speed/gap tracking emits acceleration BEFORE original safe_accel; env.step retains CommandOwnership.
Exiting the zone immediately restores MAPPO; entrants join at the next group decision.
Own telemetry, sensed nearest front/back per lane and RSU lane counts/speed/position means only.
Shared per-CAV MLP plus pooled RSU broadcast (4 dimensions, zero-initialised output).
Zero mean-head init anchors speed/gap to the current MAPPO raw command, not a hand-coded move.
An absolute held target cannot exactly reproduce a changing MAPPO trajectory for W seconds.
Test first-command parity where representable, saturation error, and exact off-zone MAPPO passthrough;
do not call first-command parity a full-rollout equivalence. Physical canary measures zero-update drift.

## Search and credit
Gaussian residuals in speed / log-gap coordinates; sigma=(4.5 m/s,0.5 log-seconds).
Scale reference is LEVERPROBE delta3/6 m/s and tau2/3 s, not its action patterns.
Per-window rho~Uniform[0,0.95]; sqrt(rho)*shared + sqrt(1-rho)*individual normals.
Score-function gradient uses the EXACT correlated joint Gaussian density, not independent marginals.
Joint and every member-solo use identical sampled absolute targets, roster, context and W.
Full deterministic reruns from world start, bitwise complete-prefix/observable/command/memory gate.
J=road speed-deficit + source wait; D_G=J_zeroHOLD-J_joint; D_i=J_zeroHOLD-J_solo_i.
C_G=D_G-sum(D_i), at 120/300/600 s; primary objective sum(D_i)+D_G+C_G=2D_G.
w_ind=w_coop=w_group=1 from update1; team D_G is shared with every member.
Use joint score * objective, so equal weights cannot reward making solos worse at fixed D_G.
Report member shared credits D_i+D_G+C_G; never sum these to redefine the objective.
One on-policy Adam step/batch (lr1e-3, common fixed reward units 10000 veh-s); no reward clipping.
Target156 windows/update over >=12 distinct TRAIN worlds; prioritise >=2-CAV windows.
Whole-zone groups never truncated; preflight whole batch or STOP. Exact solos on random39/156 windows.
Matched baselines/screens reused, prefixes still rerun for every intervention; no saved-state APIs.
Start with score function: objective2DG requires joint+HOLD; solos measure cooperation only.
If six-update DG/CG slopes both nonpositive or execution<=50%, register phase fix before next batch:
add output-bias population ES with conditional Gaussian body score; same156 charged windows.
Log per update: D_G/sumD_i/C_G distributions, fraction C_G>0, issued/executed change share,
executed-vs-issued share, safety override share, sigma and full sample/world counts.
Motion change compares paired physical speed/position tapes during W (float tolerance1e-6).
If <=50% explored windows change physical motion, multiply next sigma by sqrt(2), at most twice;
this is an execution-identifiability rule, not a traffic-behaviour rule; log before/after evidence.

## States, accounting and evaluation
Retain audited TRAIN982000–982023/DEV982024–982047; no new worlds or completed physics replays.
No restricted blocks/truth/assignments; exact registry/range and safe-filename audits on admitted hosts.
TRAIN24 worlds: all12 cells; four .2/.4 x2400/2700 cells have4 each, other8 have1 each.
Screen clocks300/360/420/480/540; 2:1 upper observable-density/CAV-count+ramp vs ordinary mix.
Quantiles derived from TRAIN screens; no hand-set behavioural cutoffs; keep ordinary states.
Adopt signed observable COOPBANDIT partial findings only via hash-bound TRAIN-only evidence.
Exact total6,000,000 sim-s: screen27,360+repeat13,680+TRAIN5,494,000+DEV288,000;
discovery132,000+UNALLOCATED44,960. Extension4M=TRAIN3,844,000+DEV115,200+discovery40,800.
Retain all1,952,433 existing charges; no refunds/retries. DEV adds4*24*1200, reusing controls.
Every bootstrap/prefix/failed attempt charged before stepping; unfinished charges block resume.
Continue from corrected u001 at u002; up to25 total, whole-batch cap determines actual count.
DEV fixed all12 cells x2worlds; MAPPO, learned, frozen COOPBANDIT when available; J600/J900
paired world-bootstrap95%CIs/safety vs MAPPO/bandit/HOLD at u3/u6/u9/final, at most4 new panels.
Run existing COOPBEHAV detectors on full physical traces; cluster action/motion summaries separately.
Report paired-supported discovered patterns, unknown clusters, and exact learned-behaviour C_G.
cpuQ four nice19/single-thread workers; load<=12/memory>=8GiB/disk>=2GiB, below WMPLAN/C1SWITCH2.
Unique supervisor/charge locks; own-PID receipts; no signals to peers, no duplicate dispatches.
Approval binds this design, source/runtime/checkpoint hashes, seed audits and CPU receipt before launch.

## Continuation details (separate root/source certificate; no original STOP clearing)
Zero-HOLD cache key=world/t0/observable context/absolute mean targets/W; full-prefix gate unchanged.
Fresh noise every batch; W uniform/exogenous; sample39/156 solo indices before observing outcomes.
Self-check every3 updates; pause for registered fix authority if stalled byu6; preserve optimizer/charges.
Fix sigma multiplier=max(1,sqrt(1/measured_execution_share)); ES perturbation scale equals new sigma.
Fix doubles signed full-EXAM observable log-priority+ordinary support; no move patterns/actions/rewards.
Publish every checkpoint/hash/UPDATE and checks to HANDOFF/COOPFREE for COOPPLAN/WMPLAN; nice19 observer.
