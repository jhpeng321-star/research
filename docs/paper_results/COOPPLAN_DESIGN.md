# S-COOPPLAN — prospectively frozen perfect-model cooperation experiment
Date: 2026-11-01. Branch: c4/coopplan; parent c4/coopbandit.
Owner direction: build cooperation usable by the rebuilt world model.
Approval: MAIN registration 00:45, `S-COOPPLAN BUDGET APPROVED`, cap 800,000 actual sim-s.
Status at freeze: BUILD + CPU tests complete; no COOPPLAN SUMO executed.
## Predictor contract
`predict(observable_state, candidate_actions, horizon, K)` returns action→paired outcomes.
Each outcome contains draw ID, team delay, BIC breakdown (None if unscorable), member delay and safety.
Member effects are paired MAPPO-minus-action on-road member delay; not latent driver properties.
Public input: sensed positions/speeds/lanes/identities/CAV flags, own commands and public history.
Full-state replay is private to SUMO-ORACLE, indexed by an opaque replay key, never a planner feature.
SUMO-ORACLE uses deterministic complete reruns, K=1; no state-load approximation or new noise.
Every rerun replays ALL executed decisions before t0 and bitwise-gates physical/controller prefix and roster.
WM-PHYS is an explicit callback stub for RB-INT's identity-frozen rollout + complete future builder.
It builds K hidden-driver/arrival draws ONCE, deep-copies the SAME draw into every action rollout.
Missing WM callbacks fail explicitly; S-RB-DECIDE Future/decide API has an optional parity check.
No WM efficacy/calibration claim is made by the interface or empirical probability alone.
## Planner and intervention freeze
At absolute t=300,360,...,960 (12 decisions), evaluate H=120 s and execute at most a 60 s option.
After t=1020, MAPPO alone through t=1200; report J600 at t900 and J900 at t1200, warmup excluded.
Cadence60 is the least-cost allowed cadence; H120 is within SUMOCORE's credible short-horizon regime.
Measured inherited throughput: 2040 sim-s /48.9293 wall-s =41.69 sim-s/wall-s on one nice19 cpuQ CPU.
Worst decision cost 6×(960+120)=6480 sim-s (~155 s serial); this is an offline ideal-model experiment.
Synchronous simulated clocks pause during prediction; no real-time deployability claim.
Candidate O0=MAPPO, O1 coordinated slow, O2 cross-lane wall, O3 yield gap, O4 longer gap.
Reuse COOPBANDIT k=2 nearest sensed upstream CAVs, geometry availability, magnitudes and safety boundary.
Grid setting cycles 0,1,2,3 by decision index, independent of world/outcome; prune unavailable groups.
Reserve a sixth slot for ONE deterministic frozen COOPFREE learned speed/gap proposal (10/20/30 s).
FREE requires an identity-bound available checkpoint before dispatch; absent at freeze → disabled entire panel.
Select minimum expected delay among actions with strict paired P(Jaction<JMAPPO)>=0.8 AND lower mean delay.
Ties keep MAPPO; exact K1 reduces to strict improvement; no hand-set traffic/behaviour thresholds.
Save membership, complete executed plan/targets, all candidate summaries, member effects and prefix identity.
## Fresh fixed development demonstration — never fitting
Reserve 983100–983111 inclusive, one world each of .05/.10/.20/.40 ×1900/2400/2700, in that order.
Exact seed×cell guard admission and registry-range + safe-filename exposure audits on all three hosts required.
No world/outcome replacement, panel tuning, learned proposal fitting, or outcome-based stopping.
Compare SUMO-oracle planner vs pure frozen speed-only MAPPO169, identical world/arrivals/noise.
J is road speed-deficit + source-wait veh-s; lower is better; differences are MAPPO minus planner.
Report paired J600/J900 and 95% whole-world percentile bootstrap CIs (10,000 resamples; RNG20261101).
Safety: collisions, teleports, TTC<3 exposure, hard braking, safety overrides; failures retained, incomplete ≠ success.
Sample first executed nondefault two-member decision per world, regardless of sign/outcome, at H120.
Reuse its O0/joint predictions; add exactly two member-alone reruns with frozen joint targets.
`C_G=(J0-JG)-sum_i(J0-J_i)`; report all signs, mean/world-bootstrap CI and sampled/unsampled coverage.
Run unchanged S-COOPBEHAV yield-to-merge/gap-creation/cross-lane-hold detectors on observable tapes.
Use same-prefix O0 reference for sampled events; observational event counts alone are not causal proof.
BIC: unchanged LRDIAG data-driven detector, 60 s retained prehistory + half-open 10 s upstream speed bins.
## EXACT conservative budget — full prefixes counted, no saved-state discounts
Per episode decisions×candidates×(prefix+horizon)=6×sum_{t=300:60:960}(t+120)=54,000 sim-s.
All lookaheads: 12×54,000=648,000; unavailable/FREE-absent candidates reduce actual charge only.
Paired episodes: 12×2×(300 warmup+900 scored)=28,800.
Member-alone sample: <=12×2×(960+120)=25,920; O0/joint already included above.
Mechanics: one full1200 MAPPO repeat + two identical (300+120) branches =2,040.
TOTAL=648,000+28,800+25,920+2,040=704,760 <=800,000; 95,240 UNALLOCATED.
Per-world upper allocation=58,560; disjoint host allocations must sum with mechanics to704,760.
Reductions are cadence60, H120, two-member groups, one FREE slot and unavailable-option pruning.
## Execution and stop contract
Before physics: bind design/source/checkpoint/SUMO1.27.1 hashes, approval, exact allocations and CPU receipts.
Mechanics uses first panel world; reuse its valid reference, then require full baseline-repeat and branch parity.
Durable precharge before every new process; actual warmups/failures count; no automatic retries/refunds/top-ups.
Exclusive IDs/locks and immutable receipts; stop own admissions on error/YIELD; own PIDs only.
cpuQ/gpuA/gpuR CPUs, nice19, CUDA hidden, one thread; <=4 world workers/host, oracle runs replace idle live CPU.
Admit only with >=8GiB free RAM/>=2GiB free disk; yield to load/resource limits and shared jobs.
No prohibited paths/truth/assignments/seeds, new venv, training, deletion, push or peer process actions.
Extension: MAIN01:28; see COOPPLAN_EXTENSION_DESIGN.md (separate immutable binding, original unchanged).
H300 continuation: MAIN02:18/02:21; see COOPPLAN_H300_DESIGN.md (18confirmation +6wall, new binding).
