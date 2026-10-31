# S-SUMOCF run receipts

## FINAL — STOPPED under MAIN's 2026-10-31 19:52 decision

**The backup did not establish beneficial deployed cooperation or a J600 win
over speed-only MAPPO.** u12 did not improve on u3; MAIN stopped the run early.
This is an unsuccessful backup experiment, not evidence of a cooperative teacher
or an oracle upper-bound gain. Historical continuation notes below are superseded.

### Paired DEV results: fixed 24 worlds, learned minus comparator

Full fleet+source delay, veh-s/world; negative is better. Brackets are 95%
world-bootstrap CIs (2,000 draws, seed0; existing S-COOPEVAL convention).

| Checkpoint | J600 vs MAPPO | J900 vs MAPPO | J600 vs CTRL0 | J900 vs CTRL0 |
|---|---:|---:|---:|---:|
| u3 | +508.3 [-343.5,+1328.3] | -633.4 [-2141.1,+866.1] | -396.8 [-1358.1,+613.9] | -72.5 [-1559.4,+1592.4] |
| u7 | +857.9 [+118.4,+1657.3] | -465.0 [-1869.1,+974.4] | -47.2 [-1134.0,+935.1] | +95.9 [-1609.0,+1620.6] |
| u12 | +548.5 [-397.8,+1520.6] | -584.5 [-2079.7,+1002.3] | -356.6 [-1402.3,+752.1] | -23.7 [-1629.3,+1620.9] |

u7 is significantly worse than MAPPO at600s. All other displayed CIs include0.
MAPPO point-estimate ranking flips600->900 at every check; no horizon-robust win.
u13–u15 were already completed at fresh-session recovery and are preserved but
have no DEV evaluation. No additional DEV was launched for finalisation.

### PRIMARY: deployed cooperation and mechanisms

- u3: C_G=-0.002589,95%CI[-0.005178,0],SD0.003661,share>0=0/2;
  D_G=873.0370, sumD_i=873.0396, pooledM_i=436.5172veh-s;1missing group.
- u7: C_G=+0.001270,95%CI[+0.000854,+0.001686],SD0.000588,share>0=2/2;
  D_G=+0.000899, sumD_i=-0.000371, pooledM_i=+0.001085veh-s;1missing group.
  This tiny two-world contrast does not demonstrate useful cooperation or a
  network improvement; the corresponding whole-panel J600 is worse than MAPPO.
- u12: C_G=0,95%CI[0,0],SD0,share>0=0/3; D_G=sumD_i=-0.000234,
  pooledM_i=-0.000117veh-s;0missing groups. No deployed non-additive gain.
- These are fixed-pair checkpoint-vs-MAPPO-mean diagnostics (three planned
  states/check), not a census of every group. Low-power coverage is explicit.
  Sampled TRAIN C_G occasionally positive is distinct from deployed evidence:
  u15 worldmean+3.99[-199.03,+206.84],rowSD539.29,share>0=.355;
  rowD_G=-354.37,rowsumD_i=-384.95,pooledM_i=-161.90veh-s.
- Existing S-COOPBEHAV u3 paired report:0supported yields/gap creations/cross-lane
  holds/wave damping. Descriptive full-fleet counts0/32/4/60 do not establish a
  learned mechanism. Its immutable source/index is in COOPBEHAV_DEV_U03_RECEIPT.json;
  no new detector/audit was run here. u3 sampled actions differed from same-state
  MAPPO (issued-safe RMS0.13121m/s2; checkpoint-mean RMS0.04404m/s2), which proves
  a control change, not beneficial cooperation. Detailed first3 diagnostics below.

### Safety

All three learned DEV panels and both comparators have0collisions/0teleports;
no collision-increase world, including the registered cooperation checks.
Full900s TTC<3 totals u3/u7/u12=11718.5/11890.5/12257.5veh-s vs MAPPO12461
and CTRL012767.5. Paired TTC differences vs MAPPO are-30.94[-79.17,+16.53],
-23.77[-83.34,+33.17],-8.48[-63.34,+47.40]veh-s/world. Hard-braking totals
u3/u7/u12=411.5/387.5/405vs MAPPO454,CTRL0486.5veh-s. Completed optimizer
updates1–15 also report0collisions/0teleports. These are observed checks, not a
claim about untested checkpoints13–15 on DEV.

### Final stop, budget and preservation

- STOP+HOLD acknowledged at15completed updates;u16retains22started samples
  (3already complete+19active drained;93,635sim-s total),14pending cancelled.
  **No u16optimizer and no optimizer after STOP.**
  Exact supervisor537871/start4206880816 exited; all893recorded worker identities
  inactive, queue0running/0pending, supervisor lock free. No signals used.
  The existing drain raises its expected assertion and overwrites STOP with
  `u2_recovery_failure`; FINAL_STOP_ACK preserves the actual owner-stop authority,
  and FINISHED.json plus FINAL_STOP_VERIFIED.json establish final scientific STOP.
- TRAIN **3,194,716 actual / 3,205,151 charged / 8,000,000 cap**;10,435unused
  reserved sim-s retained without refund (4,014old +6,421final cancellations).
  DEV **184,049 actual=charged / 300,000 cap**. Remaining charged headroom
  TRAIN4,794,849 and DEV115,951 is unspent. All gates, invalidated probe work,
  full rerun prefixes and final drain are included; artifact72,881<100,000.
- Updates1–15:2,309,304sim-s and5,253.15summed update wall-s (mean153,953.6sim-s
  and350.21wall-s/update; DEV/gates excluded). Finalisation itself used0SUMO.
- Reconciliation PASS: tools/sumocf/evidence/final/FINAL_RECONCILIATION.json,
  SHA256`232902f468d293274e539d2527952309dae85f672b8832b6d4d36c4a59cc9523`.
  Unique precharges, actual ledgers/legacy receipts, prefixes, caps, checkpoints,
  DEV pairing and exact process exit verified. Original invalid saved-state
  credits remain invalidated and retained; no retry or data deletion.
- All15policy+Adam checkpoint pairs hash-verified on E: in
  `outputs/sumocf_checkpoints/` and preserved remotely with zero checkpoint.
  Latestu15policySHA`e4b51f64fba54249792237131c2a10a6cbc5d1286c386c2c8dc9374b2d398d8a`;
  last-evaluatedu12SHA`c8b69d163adf6d6903b5356f2eacd565f9ca246566c6c88fca3b4b9814bbdb2d`.
  Full ledgers and u7–u15/u7DEV/u12DEV receipts are committed locally. Stable
  physical traces remain gpuR `/root/sumocf20261031/traces/` (canonical symlinks
  into owned storage retained). Sleeping read-only watcher exits on next poll;
  it cannot dispatch work. gpuR training CPUs released; no live continuation.

## History — first3 updates + first DEV check; continuation subsequently stopped

**No beneficial deployed cooperation or MAPPO superiority established.** Current
exact owned trainer537871/start4206880816 and read-only30min watcher539826/
4206919873 remain live. CPU22PASS; full-rerun artifact gate passed. Source recovery
binding7f97b55b, u2receipt commit2517637b. No simulation or baseline repeated.

Final liveness check:6updates completed, u7running; trainer/watcher exactalive.
TRAINcharged1,769,132/8M, DEV98054/300k. Immutable u4/u5/u6 receipts mirrored
locally. Their sampled TRAIN CG world means/95%CIs are-1916.72[-5096.60,+.000040],
+20.09[-81.52,+141.79],+5.89[+.000272,+16.60]. These are exploration contrasts,
not deployed-policy evidence; u6DGmean-603.35veh-s despite positive interaction,
so positive CG alone would not imply beneficial traffic control. No collisions
or teleports in these updates. Next paired DEV is the registered u7check.

### Update3
-279.55wall-s,128724sim-s;14eligible windows in3worlds,22missing groups.
  CG worldmean-76.16,95%world-clusterCI[-239.27,+10.79], rowmean-46.65/SD193.95,
  sharepositive.357 (CI0–.667). DGrowmean-55.74,sumDi-9.09,pooledMi-51.20veh-s.
-0collisions/teleports; group/B TTC<3 exposures6817/6780.5veh-s, hard-braking
  203.5/208veh-s. Issued-minus-MAPPO RMS.13121m/s², mean-.02554;
  checkpoint-mean-minus-MAPPO RMS.04404/mean-.008680 (same observed states).
- PolicySHAe675eab1153c88a7a85e9be92ce28dbb838c63938cc0287eea3c3c642e1ff56f,
  AdamSHA8167ea5da1d71427f7be479614897d5544b2b87bf27d4858789c16d834c1b872.
  Verified local durable copies: `outputs/sumocf_checkpoints/` (E:; binaries
  intentionally outside source commits), plus immutable remote checkpoints.

### Paired fixed24-world development panel (learned minus comparator; negative better)
- J600 vs MAPPO **+508.26 [-343.45,+1328.32]veh-s**; vs CTRL0
  **-396.80 [-1358.06,+613.89]veh-s** (95%world-bootstrap CIs).
- J900 vs MAPPO **-633.36 [-2141.14,+866.12]veh-s**; vs CTRL0
  **-72.54 [-1559.41,+1592.35]veh-s**. MAPPO mean ranking **flips**600->900;
  CTRL0 mean ranking holds, but all four CIs include0.
- Deployed checkpoint-versus-MAPPO pair diagnostics:2eligible worlds/1missing,
  CGmean**-.00258899veh-s**,CI[-.00517797,0],SD.00366138,sharepositive0.
  DGmean+873.037,sumDi+873.040,pooledMi+436.517veh-s. Nearly all the group
  effect is a solo effect; no beneficial non-additive interaction is established.
- Learned/MAPPO/CTRL0 collision and teleport totals all0. Full900s TTC<3 totals
  11718.5/12461/12767.5veh-s. Paired learned-minus-MAPPO meanTTC=-30.94
  [-79.17,+16.53]veh-s/world; hard-braking=-1.77[-4.52,+1.08]. No dev collision
  increase; registered continuation allowed. FirstDEV cumulativeactual=charged98054.

### Cooperation behaviours (existing S-COOPBEHAV reporter)
Canonical deployed DEV index: COOPBEHAV_dev_u03_fullfleet_INDEX.json, SHA
bc782dd4...; source report/counters in evidence/COOPBEHAV_DEV_U03_RECEIPT.json.
Across two full observed-fleet crops:0yield candidates,32gap openings,4cross-lane
holds,60attenuation episodes,2geometrically fitting ramp slots. **All four paired
supported counts are0** for the intervened groups. Descriptive fleet patterns do
not demonstrate learned cooperation. An initial categorical scope tag disabled
three detectors; metadata was corrected after validating actual length/neighbour
fields, reusing unchanged trace/action/credit data. The old report remains history.

### Reconciliation and live handoff
evidence/FIRST3_RECONCILIATION.json PASS verifies unique precharges, caps, source
seed ranges, every corrected branch's complete prefix+600s cost, immutable
policy/Adam hashes, first3receipts and first24-worldDEV. First3cost424492sim-s;
at audit TRAINcharged1,530,198/8M andcompletedactual1,490,204 (difference includes
inflight work+4014retained cancelled prefix seconds), DEV98054/300kactual=charged.
u4had finished and u5was running during that snapshot. Root/second owned mount
free2.73/4.12GB;1GBguards retained, no old research data moved/deleted.
Continue under original stops to30updates/caps; checksu7/12/20/30 remain.
Stable traces/indices and every update's checkpoint+Adam+credit receipts persist
on gpuR; watcher emits receipts/WATCH_epoch.json every1800s until owned trainer
stops. See tools/sumocf/HANDOFF.md; never duplicate a live launch or old jobs.

## Corrected update2 committed from completed simulations
- Recovery binding7f97b55b; exact owned continuation537871/start4206880816.
  u2policySHA4af8f04d1b204d943a01aa29a23f8548f8d0eb5a54471f54c7c61a53e4ff1063;
  exact u1Adam restored and advanced once. New SUMO for recovery0; all36result
  SHAs verified,9mask rows repaired, original physics/credit JSON unchanged.
-382.13wall-s including382.027original simulation wall-s,152209sim-s;
  21eligible windows in4worlds,15missing. PRIMARY sampled CG world mean-117.93,
  95%CI[-353.80,-.00000229], rowmean-134.78/SD428.05, share>0=.429. Four clusters
  do not support a robust cooperation conclusion; point estimate is adverse.
- DG rowmean+242.38,sumDi+377.17,pooledMi+53.80veh-s. Collisions/teleports0.
  Group/B TTC<3 exposures9873.5/10159.5veh-s; hard-braking262.5/285.5veh-s.
- Member issued-safe minus MAPPO RMS.17644m/s², mean-.02275, meanabs.03465;
  checkpoint mean minus MAPPO RMS.01909/mean-.004287. These same-state action
  differences document a policy change, not evidence of beneficial cooperation.
- u3 automatically follows; same seeds/caps/dev panel, no baseline rerun.

## u2 mechanical stop and prospective CPU-only continuation
- All36u2simulations finished (152209sim-s) but no u2 optimizer/checkpoint was
  applied. The mean assertion caught an exported mask precision mismatch at
  s75.44000244140625: actor float32 includes the merge endpoint, Python-float
  exported mask excluded it. Nonzero RSU branch created~.0024latent disagreement.
- CPU diagnosis on all21eligible samples: deriving the exact actor mask from
  observable t0 positions restores the original tolerance, max error1.91e-6.
  Physics/actions/credits unchanged; original results retained, no SUMO repeated.
-22CPU tests PASS4.70s, including actual36-receipt u2 dry update from savedu1
  policy/Adam, step1->2, endpoint regression and compact trace bit-roundtrip.
- Resume after committed/prospectively registered U2_RECOVERY_READY binding;
  TRAIN1,091,559charged/1,087,545actual, DEV57,600actual=charged. No cap/seed change.
- Future trace exports compact redundant prefix telemetry and source float32
  decimals, use gzip9 and a second owned storage mount through canonical symlinks.
  Measured28.8–47.6% sizes preserve all float32 bits; no old research data touched.

## Corrected update1 completed
-353.40wall-s,143559actualsim-s (36windows,20eligible in5worlds;16missing groups
  disclosed, never zero-filled). Checkpoint SHA373359ef6e2ea8c0fba8a6a2b0e2b07924beb753d379579b7e75860a50ade480;
  optimizer and raw credit receipts saved remotely. u2 automatically started.
- PRIMARY sampled TRAIN CG: world-balanced mean+338.62veh-s,95%world-clusterCI
  [-.455,+1016.30], row mean+84.50/SD378.81, share>0=.35 (shareCI.174–.526).
  Five clusters and uneven eligible windows give weak uncertainty; this is noisy
  exploration credit, not evidence the deployed mean policy cooperates.
- DG row mean+51.51, sumDi-32.99, pooledMi+68.01veh-s;0collisions/teleports.
  Group/B TTC<3 full600s exposures9810/9803.5veh-s, hard-braking334.5/344veh-s.
- Group-member issued-safe minus same-state MAPPO: RMS.08710m/s², meanabs.01444,
  negative25.66%/positive31.92%/zero42.42%. Checkpoint-mean minus MAPPO exactly0
  in these u1 samples (correctly sampled from zero-update policy).
- Stable COOPBEHAV_u01_INDEX.json exported for the existing behaviour watcher.
  Gap/yield/wall detection remains S-COOPBEHAV-owned; see TRACE_HANDOFF.md for
  invalidation of old probe credits and issued-versus-realized distinctions.
- Existing S-COOPBEHAV report for this **u1-only index**, hash4dab3c42..., finds
  whole-fleet4yield candidates/381openings/31cross-lane holds/781attenuation
  episodes; intervened-group reference support is0yield/1opening/0hold/9damping.
  These are sampled-policy descriptive mechanisms, not deployed improvement.
  Do not use its independently auto-scanned mixed historical index, which can
  include invalid original probes. Producer-specific immutable indices are canonical.
- Read-only30min watcher530977/start4206723808 alive, sourceSHAa3f13c41...;
  writes durable WATCH_epoch.json receipts and exits when the exact owned trainer
  finishes/stops. It never dispatches SUMO or signals any process.

## Corrected autonomous training launched (artifact PASS)
- Binding commit96b4c63f; exact owned gpuR supervisor529813/start4206707994,
  detached nice19, CUDAhidden/one numerical thread; own HOLD archived after hash
  verification. No baseline or zero checkpoint repeated. PID receipt
  tools/sumocf/evidence/ARTIFACT_TRAIN_LAUNCH_PID.json.
- Original17:39 MAIN framing respected: SUMOCF is the CPU backup/teacher, WM
  route is Paper-A primary. Cooperation remains this trainer's primary metric.
- First corrected u01 dispatch: fresh969212onward,36windows, quota admission19
  at the first check; peer~2.23cores/no throttling/3.90GB free, no peer changes.
  Every branch uses full same-seed rerun; every prefix/branch precharged.
-18CPU tests PASS/2.52s; zero actor/RSU reconstruction bitwise checked before
  launch. Pair60s,sigma.3,rho uniformly0/.5/1, six windows/world,30max updates.
  Expected191430sim-s/update; TRAINexpected6,538,691/hard8M; DEVupper291600/hard300k.
- Monitor first3updates and u3paired24-world J600/J900+cooperation DEV check.

## MAIN17:17 HOLD / artifact recovery (completed; history)
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
