# S-COOPFREE — prospective build, 2026-11-01
Status: CPU BUILD ONLY; SUMO requires MAIN's explicit `S-COOPFREE BUDGET APPROVED` line.
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
J=road speed-deficit + source wait; D_G=J_MAPPO-J_joint; D_i=J_MAPPO-J_solo_i.
C_G=D_G-sum(D_i), at 120/300/600 s; primary objective sum(D_i)+D_G+C_G=2D_G.
w_ind=w_coop=w_group=1 from update1; team D_G is shared with every member.
Use joint score * objective, so equal weights cannot reward making solos worse at fixed D_G.
Report member shared credits D_i+D_G+C_G; never sum these to redefine the objective.
One on-policy Adam step/batch (lr1e-3, common fixed reward units 10000 veh-s); no reward clipping.
Target156 windows/update over >=12 distinct TRAIN worlds; prioritise >=2-CAV windows.
Whole-zone groups are never truncated to make a batch affordable; preflight whole-batch cost or STOP.
Matched baselines/screens reused, prefixes still rerun for every intervention; no saved-state APIs.
Start with score function: one joint+all solos supplies credit and a gradient without extra rollouts.
ES/population search considered for valleys; compare gradient SNR per charged sim-s before adding
any population trials; no unmeasured efficiency claim or automatic extra physics budget.
Log per update: D_G/sumD_i/C_G distributions, fraction C_G>0, issued/executed change share,
executed-vs-issued share, safety override share, sigma and full sample/world counts.
Motion change compares paired physical speed/position tapes during W (float tolerance1e-6).
If <=50% explored windows change physical motion, multiply next sigma by sqrt(2), at most twice;
this is an execution-identifiability rule, not a traffic-behaviour rule; log before/after evidence.

## States, accounting and evaluation
Proposed fresh TRAIN982000–982023; DEV982024–982047; both guards + exposure audit before SUMO.
No restricted blocks/truth/assignments; exact registry/range and safe-filename audits on admitted hosts.
TRAIN24 worlds: all12 cells; four .2/.4 x2400/2700 cells have4 each, other8 have1 each.
Screen clocks300/360/420/480/540; 2:1 upper observable-density/CAV-count+ramp vs ordinary mix.
Quantiles derived from TRAIN screens; no hand-set behavioural cutoffs; keep ordinary states.
Adopt signed observable COOPBANDIT partial findings only via hash-bound TRAIN-only evidence.
Exact reservation2,000,000 sim-s: screens24*1140=27,360; repeats12*1140=13,680;
TRAIN1,650,000; DEV2checks*24worlds*3arms*1200=172,800;
discovery counterfactuals80*1140=91,200; UNALLOCATED44,960 (no automatic use/retries/refunds).
Every bootstrap/prefix/failed attempt charged before stepping; unfinished charges block resume.
TRAIN may fit only one156-window update: report actual count, never silently shrink batches/groups.
DEV fixed all12 cells x2worlds; MAPPO, learned, frozen COOPBANDIT when available; J600/J900
paired world-bootstrap95%CIs and collisions/teleports/TTC/braking; checkpoints and dev after u1/final.
Run existing COOPBEHAV detectors on full physical traces; cluster action/motion summaries separately.
Report paired-supported discovered patterns, unknown clusters, and exact learned-behaviour C_G.
Hosts cpuQ/gpuR/gpuA CPU-only nice19/single-thread, initially one worker, spare-load admission.
Unique supervisor/charge locks; own-PID receipts; no signals to peers, no duplicate dispatches.
Approval binds this design, source/runtime/checkpoint hashes, seed audits and CPU receipt before launch.
