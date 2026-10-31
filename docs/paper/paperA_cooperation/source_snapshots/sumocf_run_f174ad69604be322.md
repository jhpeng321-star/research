# S-SUMOCF run receipts

## MAIN17:17 HOLD / artifact recovery (current status supersedes history)
- Clean inherited7a5e2da8 preserved. DEV baseline48/48 finished; u01 simulations
  had started but **no optimizer update** completed. Existing STOP drained only
  owned work. Active samples finished, pending cancelled, reservations retained.
- Saved-state zero-noise check on969203/t465 and969204/t596 **FAILS** despite
  identical issued actions: first physical step differs; false DG+1233.339747
  and-1555.783397veh-s. These reproduce the dominant original probe values.
  All original SNR/CG numbers below are invalidated as learning evidence.
  ARTIFACT_CHECK.md and evidence/ARTIFACT_RESTORE_FAILURE.json preserve details.
- Corrected persistent offsets use original Torch decoder/safety/memory. Initial
  14CPU tests PASS; full-rerun resumption/world-budget regression brings15PASS2.72s.
  Full same-seed prefix reruns replace save/load for every future branch.
- Full-rerun zero-noise/small-large gate PASS: both states have every zero-noise
  D/CG/M=0 and identical full trajectories. Larger noise increases realized
  response on969203; on969204 SUMO erases both interventions physically (true0).
  Artifact40976sim-s including failed gate. Corrected probe running; HOLD persists.
  Combined failed+fallback gate/probe precharge hard100000 within TRAIN8M.
  No learned checkpoint or cooperation/MAPPO advantage is claimed.

### Completed fixed DEV baseline (S-COOPEVAL bootstrap,24worlds)
- MAPPO minus CTRL0, negative better: J600 **-905.06 [-1904.59,+8.85]veh-s**;
  J900 **+560.82 [-864.84,+2035.35]veh-s**. Both95%CIs include0; ranking flips
  by the mean, so no horizon-robust baseline win is established.
- MAPPO/CTRL0 collisions0/0,teleports0/0; TTC<3 exposure12461/12767.5veh-s;
  hard-braking454/486.5veh-s. Full900s-interval safety totals, not paired effects.
- All57600DEVsim-s actual=charged; original receipt
  tools/sumocf/evidence/DEV_BASELINE.json retained, never re-dispatched.

## Fresh-session recovery and amended CPU gate
- Preserved previous uncommitted trainer draft in794956ff. Prior designbf8c1e20
  superseded by cooperation-first amendment3b006b59 and subsequent implementation
  clarification. No inherited SUMOCF running job or simulation charge found.
- CPU amended suite:7 PASS in1.43s on cpuQ's existing environment, zero SUMO.
  Receipt tools/sumocf/evidence/CPU_V2.json. Zero-update policy actions match the
  exact MAPPO export bitwise; general credit, joint scores and snapshots checked.
- New active entry point: tools.sumocf.supervisor {canary,probe,train}; old draft
  train.py/worker.py are preserved in history and are not launch entry points.
- Stable trace root gpuR /root/sumocf20261031/traces; raw primary credits
  and immutable per-update checkpoint/receipt paths are under that run root.
- No measured cooperation or policy gain yet. Canary and fixed SNR probe precede
  training; the learner remains blocked on their receipts and LAUNCH.json binding.
- Final CPU reruns:7 PASS cpuQ3.64s and gpuR2.16s; gpuR frozen source/export
  verification PASS. cpuQ priority workload grew to17+ ES workers plus Track3B's
  24-worker screen. Primary host switched to gpuR CPUs (25quota,load~10), up to12
  nice19 one-thread workers with load admission. No cpuQ SUMO dispatched.

## SUMO canaries (before SNR probe)
- PASS: live-trunk versus RNG saved-state identical-action branch has bitwise
  J600 AND full physical-trajectory SHA identity. Saved-state engine selected;
  full-prefix rerun fallback was not needed.
- PASS: zero-update direct policy versus frozen MAPPO arm J600 bitwise,
  J600=83841.10161736357 veh-s on TRAIN969624(.4,2700).
- Charged=actual2400sim-s inside8M TRAIN cap; dev0. Wall80.156s overall;
  standalone900sim-s MAPPO arm26.045s,124401.96sim-s/worker-hour.
- Receipt tools/sumocf/evidence/CANARY_V2.json. Host load admission reduced its
  capacity as peers arrived (load19.25 at finish); no peer process modified.

## Fixed probe continuation: idle admission correction
-60/162 jobs completed; completed actual189670sim-s. Next12 prefix reservations
 5534sim-s are unstarted. Original global-load gate paused at hostload32..44.
- Direct measurement: cgroup7.02/25 cores,0 throttled periods. Host-global load
 includes other containers. Admission now measures cgroup+exact OWN child CPU,
 reserves3cores, maximum15workers. Receipt IDLE_GATE_REPLACEMENT.json records
 exact old owned PID502147/start4206064994 and zero live children before SIGTERM.
- No completed sample rerun, no started attempt retried, no reservation refunded.
 The probe's numerical oracle remains the original version throughout the grid.
- CPU9PASS in2.92s: includes a full optimizer/export/world-CI dry step on completed
 probe receipts with dispatch mocked. Temporary actor is not production init.
- Pre-learner budget clarification: cost-only1..8 windows/world, six fresh worlds
 per update, up to31 updates. Dev600+900 panel checks plus PRIMARY fixed-pair
 cooperation diagnostics on3worlds/check fit295200sim-s in6checks.
- Queue utilisation correction:12-job batch barriers left most cores idle behind
  each long four-member sample. Stopped ONLY owned scheduler504712 after exact
  starttime validation; its two live workers505969/505977 continued untouched and
  completed.84probe samples now complete; charged=actual277538 including canaries,
  no live old workers, no data rerun. Continue remaining78 in one15-worker queue,
  preserving every exact job/seed/draw/reservation. CPU9PASS2.28s. Receipt
  QUEUE_GATE_REPLACEMENT.json retains the scheduler/worker metadata.

## SNR grid complete (before learner)
-162/162 fixed samples; charged=actual606309TRAINsim-s including original canaries;
 DEV0. No completed sample rerun. Full27-row table in SUMOCF_DESIGN.md and
 tools/sumocf/evidence/SNR_TABLE.csv. Eligible-world counts and undefined SDs exposed.
- Original selector incorrectly treated crossfit CV means as DG; preserved original
 receipt and deterministically corrected to the owner's raw-DG formula. Correct
 winner:grid26(all-zone,60s,nominalrho1), rawSNR.07247238, four eligible worlds.
 DGmean-82.785/SD1142.293; CGmean-1769.401/SD4634.392/sharepositive.5. This is weak
 signal and does not establish beneficial cooperation. CV worsens variance:beta0.
- Original/corrected receipts retained; no sample additions or exclusions. CPU10PASS
 in2.41s includes selector regression, policy-tail update/export, and credit-schema
 compatibility with S-COOPEVAL. V3 canaries precede learner launch.

## V3 gates / launch binding
- V3 PASS: zero-update arm, live-vs-restored branch, and ALL deployed-mean DEV
  subsets reproduce MAPPO J600 and physical trajectory bitwise; CG0. Independent
  actual-step accounting reconciles5700additionalsim-s. Gates total612009TRAIN,
  DEV0; all canaries8100inside12000 allocation. Receipt CANARY_V3.json.
- CPU10PASS; selected cost chooses5windows/world,6fresh worlds/update,31max;
  expected165720sim-s/update and5,749,329totalTRAIN including gates. Runtime
  estimate~420wall-s/update at15workers before load/straggler waits.
- Launch-ready authority: tools/sumocf/LAUNCH_READY.md and evidence/LAUNCH.json,
  source87ec0192, design d7585ec9, corrected probe f92a51a1. Learner launch follows
  this committed binding and registration, under original owner approval.

## Training launched autonomously
- gpuR owned supervisor PID517331/startticks4206463446, detached nice19,
  `launch.py train`; receipt tools/sumocf/evidence/TRAIN_LAUNCH_PID.json.
- Launch binding commit61045b84 (scientific source87ec0192), no push.
- First dispatch: fixed DEV MAPPO+CTRL0 baseline at600+900s,48episodes x1200s,
  precharged57600DEVsim-s. TRAIN612009alreadycharged; hardcaps unchanged.
- Monitoring through first3optimizer updates and first DEV check(u3) is active.
