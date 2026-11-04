# S-COOPMWM — live checks

### Nov4 13:43 UTC+8 — A16/B9/R8; practice handover registered for u10
- 13:49 follow-up: R-u9's24episodes are complete, zero failures, learner active;
  checkpoint/update receipt not yet sealed. cpuQ~6.006GiBfree; both movers live.
  A single cpuQ SSH timeout recovered on a read-only retry; no job restarted.
- **B9 and R8 learning mechanics/composite safety pass.** Both completed five
  full imitation passes. B9 KL0.0449162<0.045, eight cooperative examples,
  action MAE1.229555->1.210766m/s²; first-pass loss-change95%CI
  [-0.060318,-0.039057]. Intent loss2.000171->1.843499 fell this update.
  R8 KL0.0402559, three cooperative examples; its first-pass CI includes0
  without significant harm. First-minibatch PPO KL0 for both.
- On24paired TRAIN worlds, collisions/teleports0 for both. Critical-total
  B9-A9=-16.604CI[-41.418,8.459]; R8-A8=-30.875CI[-56.458,-5.519].
  These checks do not establish learned cooperation; the completed u6
  isolated-group diagnostic remains the current cooperation evidence.
- The existing operator registered practice starts FIRST at **u10 for both
  B and R** (fraction0.25, repeat cap2). Source8b918fcdb/Linux313PASS,
  runtime1d67a685/gate11PASS/launch3preflightREADY are registered. B r2 has
  exited at the intended u9 boundary; R9 SUMO is running. Monitor4 still
  actively covers current updates but binds the older runtime; the operator
  was notified to hand it over with r3. r3 adoption/u8 test dispatch remain
  operator-owned and are not claimed complete here.
- Independent authority reconciliation verifies all registered transfers in
  runtime1d67a685: bank+840,000 from sumo verification, bank+100,000 from
  legacy train, practice+345,600 from paired_branch to v2v_train. Phase sum
  and cap17,471,840; charged13,847,120 including811,640retained once. All
  balances nonnegative at the probe. Future lesson shortfall13,200 remains
  reported; u12 fits. No allocation changed by this continuation.
- Archive batch37 failed with SSH connection-closed255, retained sources,
  and recovered naturally in fresh batch38. Extension total3,509,309,275
  bytes certified; batch39 transferring. cpuQ6,457,327,616bytes free
  (~6.01GiB), only14,876,672bytes above6GiB floor; both movers remain live.
  CPU-S landing9GiB guard and cpuQ collector/backfill stops remain intact.
- Evidence:`coopmwm/RESUME_ADOPTION_20261104_1330.md`,
  `coopmwm/PRACTICE_HANDOVER_READINESS_20261104_1340.md`, and
  `coopmwm/HANDOVER_READINESS_1335_1793770985476960500.json`.

### Nov4 12:57 UTC+8 — B-u8 complete; A14/B8/R7
- **B8 learning mechanics and composite safety pass.** Four full imitation
  passes of five requested, KL0.039161<0.045, eight within-seed cooperative
  examples, no imitation failure; raw action MAE1.242276->1.223864m/s².
  First-pass loss-change95%CI[-0.061287,-0.039790]; first-minibatch PPO KL0.
  Bank intent loss1.954416->2.249387 increased (report-only).
- On24paired TRAIN worlds, collisions/teleports0; critical-total B-A=-24.854
  CI[-43.938,-5.688], hard-brake=-1.688CI[-5.146,1.667]. R7 also passes:
  five full passes, KL0.042762, three cooperative examples; critical-total
  R-A=-18.417CI[-44.854,7.354], collisions/teleports0. R raw action error
  rose slightly despite lower AWR loss. These are conditional TRAIN checks;
  positive learned cooperation remains unestablished by the u6 isolated test.
- Existing training owners and both archive movers are live. Practice operator
  has prepared registered runtime2d73ce02/src8; current r2 children retain their
  prior source. Monitor4 actively covers B8/R7 but still binds06e88ecd; the
  operator was notified to include monitoring in the boundary handover.
- Charged13,670,900/cap17,471,840, including811,640retained exactly once;
  live-bank phases nonnegative. Bank-verification and practice transfers are
  separately registered in bank-binding/runtime authorities; both must survive
  reconciliation. Future lesson-phase gap13,200remains reported; u12 fits.
- cpuQ~6.54GiBfree, MemAvailable~9.63GiB; extension mover3.127GBcertified.
  One SSH connection timeout recovered on a bounded read-only retry. Existing
  CPU-S checking/landing and practice/effective-imitation work remain delegated.
  Evidence:`coopmwm/B8_R7_HEALTH_20261104_1257.md` and
  `coopmwm/PUBLIC_PROGRESS_1793768209687489000.json`.

### Nov4 12:41 UTC+8 — A14/B7/R6; live phase authority reconciled
- **B7 learning mechanics and composite safety pass.** Three full imitation
  passes of five requested, KL0.044938<0.045, eight within-seed cooperative
  examples; action MAE1.258998->1.237346m/s². First-pass loss-change95%CI
  [-0.073663,-0.047613]. Bank intent loss2.254510->2.380863 rose (report-only).
  On24paired worlds: collisions/teleports0; critical-total B-A=-8.5
  CI[-31.439,13.959], hard-brake=-0.208CI[-3.313,2.854].
- Existing r2 supervisors, A and runtime-bound monitor4 remain live. A14 completed
  at12:39:09. B8/R7 had not claimed jobs at the preceding readiness probe;
  no failed episode or new STOP was observed.
- **Accounting reporting repair verified:** phase balances now use the registered
  live B_REPLAY_BINDINGa91884b6, which includes MAIN's840,000sim-s bank transfer.
  The frozen runtime's older phase table would falsely flag bank overspend.
  Actual phases are nonnegative;13,080,120charged/cap17,471,840 at the budget probe,
  including811,640retained superseded exactly once. MAIN/practice operator was
  notified that this retained expense is spent, not available practice funding.
  MAIN's registry13:10corrects the donor to`paired_branch`; the delegated practice
  deployment remains separate. Future lesson-phase shortfall13,200still needs
  an allocation decision.
- cpuQ7,512,530,944bytes free (~7.00GiB); both archive movers live, unchanged
  protected-input scope. CPU-S checking/practice changes remain delegate-owned.
  Evidence:`coopmwm/RESUME_ADOPTION_20261104_1229.md`,
  `coopmwm/ACCOUNTING_COHERENCE_20261104_1235.md`, and
  `coopmwm/RESOURCE_READINESS_1130_1793767132124120800.json`.
  Latest state proof:`coopmwm/RESUME_LIVE_0815_1793767218358734200.json`.

### Nov4 12:23 UTC+8 — A13/B6/R6; future diagnostic phase gap identified
R6 completed five full imitation passes, KL0.038233<0.045 and no imitation
failure. AWR18.1665->17.7739 fell; unweighted action MAE0.453507->0.455639
rose. Intent0.506662->0.447447 fell. Its24world composite safety check passes,
zero collisions/teleports, critical-total R-A=-11.375CI[-26.833,3.938]. Existing
supervisors and runtime-bound monitor4 remain live; B-u7 SUMO was observed.

The full12-entry lesson slate costs8,760sim-s/check. Fixed futureu12/u18/u24
need26,280; `v2v_lessons` has13,080remaining: **13,200sim-s forward phase
shortfall**, reported to MAIN for reallocation within the existing cap.
No allocation changed; current balances remain nonnegative and u12 fits.
Budget12,691,610charged/cap17,471,840. cpuQ~6.14GiBfree; both movers active,
with539MB newly eligible afterR6completion. Proof:
`coopmwm/DIAGNOSTIC_BUDGET_FORECAST_1793766186218814600.json` and
`coopmwm/PUBLIC_PROGRESS_1793766228801723500.json`.

### Nov4 12:20 UTC+8 — B-u6 frozen lessons repaired, 12/12 graded
The additive `LESSON_CHECK_u006_B_MAIN1202_REPAIRED.json` completes all five
previously missing entries and adopts the original seven unchanged. **Mean
delay improvement+161.929veh-s,95%CI[-335.497,+580.651],8/12positive**; the
interval includes zero. Exactly3,480sim-s in five shared-ledger rows verified.
Original incomplete report, checkpoint and runtime hashes are unchanged.
Worker2854683 completed normally; never redispatch. This supersedes the
incomplete7/12 lesson summary below, while preserving that original receipt.
The isolated-group result remains as reported below: positive learned
cooperation is not established. Proof:`coopmwm/U6_LESSON_REPAIR_COMPLETE_MAIN1202.md`.

## Nov4 12:15 UTC+8 — B-u6 complete; isolated check does not show learned cooperation
- **A13/B6/R5 complete**, with the existing r2 training supervisors and bound
  monitor4 running. B6 completed **3 full imitation passes of 5 requested**,
  no imitation failure, KL0.043269<0.045, eight within-seed cooperative examples.
  Action MAE1.277327->1.256317m/s²; first-pass AWR loss-change95%CI
  [-0.066063,-0.042092]. Bank intent loss2.230793->2.417256 rose; report-only.
- **B6 composite safety passes**,24paired TRAIN worlds: zero collisions and
  teleports; critical-total B-A=-7.354CI[-25.292,10.063], hard-brake
  -0.354CI[-3.667,2.875]. R5 safety/mechanics also pass. No active run STOP.
- **The registered isolated-synergy check is complete on seven state rows.**
  Group-only cooperation gain (all group new minus best solo new), veh-s:
  986052:-0.0082;986066:-0.0468;986068:+0.0002;986003:0;
  u8/986026:-514.8703;u7/986036:-160.0504;u18/986047:-0.0560.
  These point results do not establish positive learned cooperation. In the
  six-CAV986066 state, background-only new-policy saving is+2,807.324veh-s;
  it must not be attributed to coordinated improvement of the isolated group.
  State-level point results are conditional TRAIN diagnostics, not final-panel
  evidence or a fresh-seed generalization test.
- Original B6 lesson report is **7/12 graded**, mean delta-32.456veh-s,
  95%CI[-779.163,603.191],4/7positive. Five missing pending envelopes were
  traced to the old certified archive batch, restored655,103bytes with exact
  SHA proofs, and functionally loaded5/5. Both mover scopes exclude replay_bank;
  an additive binding protects710input paths, including665bank files.
- MAIN12:02 now explicitly requests completion of the five missing lessons.
  Registered **3,480sim-s** from existing `v2v_lessons` headroom, fixed B6
  checkpoint, seven existing rows adopted. A single durable worker2854683 is
  running; its additive `_MAIN1202_REPAIRED` output will preserve the original.
  No policy update, duplicate training or rerun of the seven graded lessons.
- cpuQ~6.38GiB free; both archive movers live. Budget12,615,640sim-s charged
  including retained superseded, cap17,471,840; all phases nonnegative.
  CPU-S checking remains separately delegated; cpuQ backfill/collector STOPs
  remain. Current-V2V collection wiring is still the previously reported blocker.
- Evidence:`coopmwm/PUBLIC_PROGRESS_1793765713183406300.json`,
  `coopmwm/U6_PENDING_RESTORE_COMPLETE_1205.md`, and
  `coopmwm/U6_LESSON_REPAIR_DISPATCH_MAIN1202.json`.

## Nov4 11:46 UTC+8 — A11/B5/R4; memory admission recovered
- **B5 completed five full imitation passes**, eight within-seed cooperative
  examples, no imitation failure and KL0.042415<0.045. Action MAE decreased
  1.297296->1.276354m/s²; first-pass AWR loss-change95%CI[-0.067176,-0.043125].
  First-minibatch PPO KL0.0. Bank intent loss rose2.2574->3.0443, report-only.
- **Composite safety passes**,24paired TRAIN worlds, zero collisions/teleports;
  critical-event delta B-A=-15.0833CI[-39.4599,7.8547]. B-u6 lesson and isolated
  policy synergy remain pending; imitation improvement alone is not cooperation
  evidence.
- A verified resource blocker was cleared: two superseded T-stopped owners plus
  twelve parked children still held4,405,244KiB RSS. Following registration,
  five local and five Linux checks, all14were retired with individual identity
  checks. MemAvailable7.61->11.81GiB cleared the unchanged8GiB admission guard.
  No files were deleted and all old charges remain. Existing A-u12/B-u6/R-u5
  episode workers subsequently advanced (12/2/11episodes complete at11:46).
- Current training supervisors, runtime-bound monitor4 and both movers are live.
  cpuQ~6.99GiB disk free, still below the9GiB backfill-resume target. Backfill
  and collector resource stops are retained. Latest budget12,104,980sim-s charged,
  including811,640superseded, against17,471,840registered cap; no negative phases.
- Receipts:`coopmwm/RESUME_ADOPTION_20261104_1130.md`,
  `coopmwm/PARKED_RETIREMENT_1140_COMPLETE.md`,
  `coopmwm/RESOURCE_READINESS_1130_1793763983848115600.json`.

## Nov4 11:10 UTC+8 — A11/B4/R4; recovered B/R updates pass mechanics and safety
- Existing r2 supervisors and runtime-bound monitor4 are live. **B4 and R4 each
  completed five imitation passes**, with KL0.042823/0.035858 below0.045 and no
  imitation failure. B used8 within-seed cooperative examples; R used3.
- B action MAE1.325578->1.301208m/s²; AWR41.1000->38.8359. Its first-pass
  loss-change95%CI[-0.03473,-0.02270] is below0. R action MAE0.455657->0.451148;
  AWR19.3963->18.5896. R first-pass CI includes0 and does not trigger a pause.
- **Composite safety passes for both**,24 paired TRAIN worlds,0collisions and
  teleports. Critical-total B-A+1.6875 CI[-24.5625,29.6464]; R-A-3.125
  CI[-22.875,14.1875]. B intent loss2.2928->3.8845 increased; R0.4942->0.3818
  decreased. Intent trends remain report-only. B6 viability checks remain pending.
- cpuQ~7.02GiB free; both archive movers live. Backfill correctly disk-paused
  below9GiB. Completed tail/slim migrations were adopted without redispatch.
  Full live evidence: `coopmwm/RESUME_ADOPTION_20261104_1110.md`.
- 11:15 resource/accounting follow-up: cpuQ~7.27GiB; extension archive1.581GB
  certified across19batches. The remaining40.9MB eligible pool cannot cover the
  1.860GB gap to9GiB. Live budget12,018,580sim-s charged including811,640retained
  superseded charges, against the later owner-approved17,471,840cap; all phases
  nonnegative, no allocation change. `coopmwm/PUBLIC_PROGRESS_1793762075776629700.json`.

## Nov4 10:54 UTC+8 — TRAINING RUNNING r2; approved storage work complete
- **A10/B3/R3 completed.** The delegated r2 supervisor and B/R runners are live;
  B u4 episodes have run and its learner is active. R retains its registered
  lag-1 schedule. Resume receipt timestamp is10:45:30+08. Monitor4 is live and
  bound to runtime06e88ecd; the old monitor3 is retired.
- Owner10:40 approved the learning-pause release. The source7/a7ae6d9a gate8
  actually **PASSes**, all six offline checks true and no reasons; Linux262/262
  PASS was independently read. Qualified recovery code is adopted in merge
  `6dd5cd1b`, preserving storage work; both monitor conflicts match the deployed
  source exactly. **17local integration checks PASS**. No duplicate job dispatched.
- MAIN10:12 slim migration is **complete and independently verified**:
  **447branches/894files/14remote SHA proofs;936,070,977gross bytes archived,
  4,209,419local SLIM bytes,931,861,558net bytes recovered**. Zero errors/partials.
  Full originals remain on gpuA; retained SLIM seals, result/payload identities,
  current rejected-only eligibility and window totals all verify. Earlier legacy
  tail move separately completed476,840,470bytes. Never redispatch either worker.
- cpuQ had7.22GiB free at the10:47 verification,7.06GiB at10:54 as training writes
  continue. **The10GiB target remains unmet**; no broader source scope or lower
  guard was used. Backfill's9GiB guard and MAIN's collector STOP stay in place.
  The new CPU-S setup/parity is separately delegated.
- B-u6 lesson/isolated-synergy evidence remains pending. Sources and proofs:
  `coopmwm/SLIM_MIGRATION_MAIN1012_V4_COMPLETE.md`,
  `coopmwm/SLIM_MIGRATION_MAIN1012_V4_VERIFIED_COMPLETE.json`, and
  `coopmwm/RESUME_LIVE_0815_1793760835619768600.json`.

## Nov4 10:31 UTC+8 — A9/B3/R3; approved storage moves progress
- Direct public-receipt/process probe shows **A9/B3/R3**, with A's u10 episodes
  running. B/R remain at the mechanics pause. The recovery operator reports
  src7/a7ae6d9a Linux262/262PASS and prospective runtime06e88ecd, but its release
  write was denied and referred to MAIN. Monitor3 was gracefully retired during
  that handover; no competing release, gate or monitor was dispatched here.
- The approved legacy-tail SHA archive is **complete:476,840,470bytes/240files/
  4certified batches**. Full originals remain on gpuA, compact metrics locally;
  both regular movers stayed running. Completed work is not redispatched.
- MAIN10:12's rejected-tape migration is qualified (**23Linux checks**, six local
  proof-boundary tests), registered and dispatched once. The real dry-run admits
  **447explicitly rejected branches/894files/936,070,977bytes**, zero errors.
  Payloads go directly to gpuA over the live mover transport, with remote byte,
  payload and canonical result-digest proof before local SLIM/source removal.
  First progress proof:57branches/128,702,937gross bytes moved,512,456bytes of
  local SLIM summaries; two batches remotely verified. Owner2600683 is live.
- cpuQ free7,229,759,488bytes (~6.73GiB). The complete eligible gross slim pool
  cannot reach10GiB: the dry-run's optimistic remaining gap is2,603,024,575bytes,
  before SLIM overhead/new writes. The9GiB backfill guard and MAIN's collector
  STOP stay in place. Fresh-V2V collection wiring remains previously reported.
- Actual B-u6 lesson/isolated-synergy evidence is still pending. No new SUMO,
  inference, budget charge or training recipe change by this storage work.
  Receipts: `coopmwm/LEGACY_TAIL_ARCHIVE_1000_COMPLETE.md`,
  `coopmwm/SLIM_MIGRATION_MAIN1012_V4_RUNNING.md`, and
  `coopmwm/RESUME_LIVE_0815_1793759473150996300.json`.

## Nov4 ~09:30 UTC+8 — legacy stop sealed; delegated recovery not yet qualified
- **A7/B3/R3.** B/R mechanics recovery remains delegated. Candidate src5/
  9ddb8d48 has a non-PASS Linux receipt (262 tests, zero simulation), so the
  last qualified src4/runtime4a8890cd is retained; no competing launch here.
- Latest completed B3/R3 composite safety passes with zero collisions/teleports.
  Both completed five imitation passes; the old R3 first-pass criterion caused
  the preserved mechanics pause. B-u6 lesson/isolated synergy remains pending.
- Explicit MAIN09:10 `STOPPED_PERMANENT` receipt sealed; both legacy checkpoint
  files SHA-verified. The approved V2V archive has certified1,035,251,575bytes;
  cpuQ~6.35GiB, backfill correctly paused. Fresh current-V2V collection wiring
  remains blocked and reported. `coopmwm/CONTINUATION_STATUS_20261104_0930.md`.

## Nov4 ~09:15 UTC+8 — continuation adopted; existing owners preserved
- Direct live check: **A7/B3/R3**. A continues; the B/R mechanics release and
  statistical pass-1 refinement are with MAIN's delegated launch operator.
- The requested permanent stop of old non-V2V B/R is already complete: actual
  durable endpoints B21/R20, checkpoint hashes kept, all five old owned PIDs
  absent. Shared bank and collection services remain live.
- Both archive workers and runtime-bound monitor3 are live. cpuQ free ~6.43GiB;
  backfill is correctly disk-paused (38 decisions complete). Fresh V2V collection
  still needs its previously reported capture/WM-intent-actor qualification.
- Full brief read; completed warm-memory work adopted. No new simulation or
  duplicate recovery dispatched. `coopmwm/RESUME_ADOPTION_20261104_0915.md`.

## Nov4 ~09:00 UTC+8 — registered mechanics pause at R u3
- **A6/B3/R3.** R3's first imitation-pass AWR loss rose19.347578->19.349314,
  so `pass1_effective=false` correctly triggered the registered B/R pause.
  Both workers exited through the health guard; A continues.
- R nevertheless completed five full passes, ended at lower AWR19.334106,
  reduced L1 error, stayed within KL0.000155<0.045, and passed composite safety.
  No parser, nonfinite, coverage, physics or accounting fault is established.
- The pause is preserved and diagnosis reported to MAIN under the registered
  procedure. No learner activation or criterion change was made. B-u6 remains
  pending. Details: `coopmwm/R3_LEARNING_MECHANICS_PAUSE.md`.
- Archive/monitor workers remain live; cpuQ6.54GiB at the last probe. Backfill
  remains below its9GiB resume threshold.

## Nov4 ~08:50 UTC+8 — B recovery progresses; protected diagnostic archive complete
- **A6/B3/R2**; replacement B/R supervisors observed live under the delegated
  launch3 resume. B's u3 reused24closed/precharged episodes, without replay.
- B3: five full imitation passes, effective pass1, KL0.001632, eight cooperative
  examples; action MAE1.33136->1.32716m/s². Bank intent loss2.2652->2.3780
  again increased and remains report-only. Composite safety passes: no
  collisions/teleports; critical-event delta-29.77CI[-54.21,-9.85] vs A.
- Identical learning monitor3 is qualified, registered and live on the new
  runtime authority. Previous monitor exited gracefully; rules are unchanged.
- Additional **289,878,263bytes/255files/3SHA-certified batches** archived from
  superseded diagnostic branches. cpuQ6.28GiB at completion;9GiB backfill target
  remains unmet. Existing rolling movers continue.
- B-u6 lesson/isolated-synergy evidence is pending. The resumed u3's309s duration
  reused episodes and is not a fresh-update throughput estimate.
- Evidence: `coopmwm/V2V_RECOVERY_ADOPTED_0850.md`.

## Nov4 ~08:15 UTC+8 — fresh-session adoption, delegated recovery retained
- Direct live check: **A5/B2/R2**. A continues; B/R checkpoint003 recovery
  remains with MAIN's separate operator. Existing work and owners adopted.
- Learning monitor remains healthy; last B/R updates completed five full passes
  and passed composite safety. B's increasing bank intent loss remains reported.
- Protected V2V archive has certified **499,403,300bytes /216files**; no more
  currently eligible files. Both archive workers remain live. cpuQ has6.49GiB
  free, about2.70GB below the9GiB backfill-resume target before new writes.
- Evidence: `coopmwm/RESUME_ADOPTION_20261104_0815.md`. No new physics or
  duplicate recovery dispatched by this adoption.

## Nov4 ~07:45 UTC+8 — V2V u3 namespace fault delegated; disk scope extension live
- **07:55 update:** the entire eligible initial pool is archived: **499,403,300
  bytes /216 files /4 SHA-certified batches**. Worker remains live; zero eligible
  files remain. Free space ~6.60GiB, still 2.58GB below the9GiB target.
- A reached u4. B/R remain durably at u2 after B's u3 checkpoint save collided
  with pre-launch legacy-reference symlinks. MAIN07:20 assigned recovery to a
  separate operator. This continuation performed no competing repair or restart.
- R u2 now confirms five full imitation passes on three cooperative examples,
  effective first pass, KL 0.00007318; bank intent loss 0.60790 -> 0.59379.
  B's last complete u2 remains healthy, with the reported rising intent-loss trend.
- Registered protected V2V archive extension is live, using unchanged qualified
  SHA-before-removal machinery. Original regular mover remains running. First
  two batches **285,407,020 bytes / 132 files** are certified moved; compact
  metrics remain local. cpuQ free ~6.41 GiB; the initial eligible pool falls short by
  ~2.57 GB to reach 9 GiB. Backfill remains correctly disk-paused.
- Durable per-update learning monitor2 is live, with six local and six Linux
  checks. It enforces registered learning-mechanics pauses and reports intent
  trends. The existing safety guard remains active.
- Collection config routing repair passes 50 Linux checks, but activation still
  needs V2V capture/WM-intent-actor wiring and per-slot numerical qualification.
  No new collection is claimed from an unchanged supervisor restart.
- The budget observer now reads the additive authority: 11,630,140 charged of
  17,471,840 sim-s, including retained superseded charges; bank phase 597,790
  remaining at that snapshot. No new physics charges from monitoring/archive work.
  Evidence: `coopmwm/V2V_CONTINUATION_0745.md` and its named public receipts.

## Nov4 ~06:45 UTC+8 — V2V training adopted and progressing
- **V2V_STARTED 06:29:32**, root `coopmwm20261104_v2v_coop3`; watcher and
  all three arms verified by boot/start/command. Latest A2/B2/R1, no health pause.
- B u2: **five complete imitation passes**, effective first pass, KL 0.001412,
  eight cooperative examples; action MAE 1.335388 -> 1.331062 m/s².
  Bank intent loss rose 2.400348 -> 2.535167; monitor the trend.
- Composite safety B2 passes: collisions/teleports zero; critical-event delta
  +2.0833 [−20.3964,+25.0219] per paired world. R1 passes with zero deltas.
- Adopted qualified source/evidence `e3a56a76` preserving archive commits and
  existing work. Option-2 recovery coordinator superseded/never run.
- Regular archive mover live; disk ~6.79 GiB. Existing backfill/collection retained.
  Evidence: `coopmwm/RESUME_ADOPTION_20261104_0645.md` and its live probe.

## Current continuation — diagnostic reported; primary launch delegated
- Superseded all_admitted gate:55/55 exact branches graded,0prefix/physics
  faults; diagnostic **FAIL** on group cooperation (986066 bounds-0.020857/
  -0.039186veh-s, practically near-zero driving changes). No actor was launched
  from it. Report:`coopmwm/COOPFIX_GATE_ALL_ADMITTED_DIAGNOSTIC.json`.
- Current cooperative_only gate/launch belongs to MAIN's c4/launch-coop operator.
  Under the latest owner rule, learning mechanics/hard safety/complete data gate
  launch, while cooperation is reported. No duplicate primary job here.
- Completed SHA archives recovered2,547,341,551bytes, all statuses/summaries kept.
  An additional2,871,389,536-byte old B/Ru4..18 output pass is active, retaining
  compact metrics locally and full originals on gpuA. Last proof:271,363,949bytes
  moved/free8,500,752,384bytes. Backfill is correctly disk-paused pending9GiB.
- Live owners, proofs and next checks:`_sol/COOPMWM_STORAGE_AND_HANDOVER_LIVE.md`.
  No sealed outcomes or prohibited truth files were opened.

## Nov4 00:21 UTC+8 — final integrated intent/per-seed source passes Linux
- READY integration merged as`762bdc7d`; the exact immutable I3 source passes
  **994 CPU checks,0 skips,138.534s,zero SUMO**. Both prior metadata-harness
  failures now pass. The full I3 run is complete and must not be repeated.
- Intent sharing, own-seed bank examples, all8 verification, the end-to-end
  update and guarded launch interfaces are included. This is engineering
  qualification; real-bank cooperation and the exact-commit audit remain gates.
- Actual gate and V2V launch stay held. Prospective runtime/sources inputs are
  prepared; MAIN's delegated bank preview and collection are not duplicated.
- Evidence:`coopmwm/FINAL_LINUX_QUALIFIED_I3.md`,
  `coopmwm/V2V_LINUX_CPU_RESULT_INTENT_I3.json`.

## Fresh-session adoption — Nov3 23:08 UTC+8
- 23:41: restored702 archived candidate specs for88B decisions, exact SHA before
  and after copy, no overwrite/deletion/search replay. Independent proof702/702.
  Fresh all8 cost quote now has zero missing specs for either arm. Seed2-only
  minimum1,546,510sim-s still exceeds bank-phase room by1,409,330; no transfer.
  Evidence:`coopmwm/SPEC_RESTORE_AND_QUOTE_2341.md`. Latest delegated collection
  restart is respected; no competing builder/collector was launched.
- Durable legacy state remains A48/B21/R20; all four intentionally held old
  B/R arm/grader owners are absent. No restart or duplicate dispatch.
- Saved searches continue to answer: two B requests remain assigned, zero R
  requests pending. Existing four-slot services are retained.
- cpuQ free6,215,000,064bytes is227,450,880bytes below6GiB; MemAvailable
  22,143,500kB. Final integration remains WIP/READY absent, and the available
  audit is BLOCKING for an older commit. V2V and the real-bank gate stay held.
- Evidence:`coopmwm/RESUME_ADOPTION_20261103_2308.md` and its two live receipts.
  Zero new SUMO/inference/charges/process control/deletion by this adoption.

## Fresh-session adoption — Nov3 22:31 UTC+8
- Read-only guard reporting repaired and verified at22:45: memory-recovery
  owners411152/411153 are correctly reported exited and B/R paused=true under
  the registered owner-specific hold. Shared-root health remains unchanged.
  17 CPU tests plus2subtests pass. Disk has recovered to6,095,859,712bytes,
  still346,591,232bytes below the6GiB V2V gate. No process was restarted.
  Evidence:`coopmwm/WARM_STATE_SCOPED_HOLD_REPAIR_2245.json`.
- Completed warm-memory qualification/restarts and all existing work are adopted.
  Actual durable legacy checkpoints remain **A48/B21/R20**; the four old B/R
  arm/grader owners are absent under their registered intentional resource hold.
- The saved search queue is advancing: a B answer completed at1793716163.88;
  four B requests remain assigned, with no pending R centre requests. Existing
  services are retained, without duplicate dispatches or inference replay.
- Charged10,202,670sim-s; registered cap17,471,840 under the later explicit
  addenda. The bank-confirmation phase has137,180sim-s remaining, so the expanded
  both-seed verification allowance must be re-quoted and registered before use.
- `LAUNCH_FINAL_READY.json` is absent and the available audit targets the older
  f3caf85f code with blockers. V2V and the real-bank gate stay held. Disk is about
  3.88GiB, below the6GiB launch gate; independent cleanup remains MAIN-owned.
- Evidence:`coopmwm/memory_RECOVERY_STATUS_1793716212369358100.json`,
  `coopmwm/SEARCH_CURRENT_AUDIT_1793716279288040900.json`. Zero new simulation,
  inference, process control or deletion by this continuation.

## Latest continuation — resource pause and audit corrections
- MAIN21:58/22:10 now requires per-seed own-tape bank examples, same-seed
  cooperation gate, all-eight-candidate verification on both seeds, intent for
  all three arms, final exact-commit audit/Linux PASS, resources, and reviewed
  cooperation-scored search enabled with `--alternate`. V2V remains held.
- Old reference arms stopped at their actual idle checkpoint boundary **B21/R20**,
  using registered `PAUSE_RESOURCE_MAIN2158B.json`. Four boot/start/argv-verified
  owned arm/grader PIDs acknowledged the pause and exited. No episode was replayed
  or checkpoint rolled back. The supervisor's expected child-exit alert is kept.
- MAIN22:03 scopes this hold to the exited old owners: the original resource
  pause bytes are retained in control metadata, allowing required shared-root
  gate/solo/search tooling to pass its unchanged health checks. Old B/R remain
  held. Registration16802967; no safety guard was relaxed.
- Available memory rose from about14.22GB to22.76GB. Disk was3,783,544,832bytes
  after the pause; MAIN's separate cleanup agent owns further reference checks.
- Audit report is now available and has blockers for its f3caf85f subject.
  Live commits17533a50/f59d0f66 add actual launch-gate dispatch wiring (12 CPU
  checks), isolate intent AWR from PPO, and require the detached intent head.
  Per-seed/gate/interface fixes are being integrated on `c4/launch-final`.
- Cooperation-search review fixes are merged but inactive;20 local CPU checks
  pass. The earlier903-check Linux result remains evidence for its original pin,
  not qualification of subsequent changes. No V2V launch or real-bank gate run.
- Evidence:`coopmwm/LEGACY_RESOURCE_PAUSE_REGISTRATION_2158B.json`,
  `coopmwm/LEGACY_RESOURCE_PAUSE_2158B.json`,
  `coopmwm/LEGACY_ONLY_RESOURCE_HOLD_2203.json`,
  `coopmwm/memory_RECOVERY_STATUS_1793714710968443000.json`.

## Nov3 ~22:00 UTC+8 — reviewed intent integration passes Linux qualification
- Reviewed intent and exact-cooperation-gate branches merged in commits
  `689a5c0f` and `b819c3d2`. Pinned integration `f3caf85f` passes **903 Linux CPU
  checks, zero failures/skips,107.265s,zero simulation** in an immutable cpuQ root.
- This includes causal t-1 intent sharing, zero-init loading, separate full-pass
  imitation, boot_v2 certification, and the paired exact-gate CPU suite. Frozen
  pre-intent git fixtures verify option-off identity inside the no-physics guard.
- Independent audit report is pending. The real-bank exact gate and V2V launch
  remain held; Linux success alone is not a cooperation or launch result.
- Legacy checkpoints are B20/R19 at the live state check, with no health STOP.
  Existing collection/graders/observers are retained. No new SUMO or WM search
  was dispatched by this continuation. Cap17,471,840 and prior charges retained.
- Evidence:`coopmwm/V2V_LINUX_CPU_DISPATCH_INTENT_I1.json`,
  `coopmwm/V2V_LINUX_CPU_RESULT_INTENT_I1.json`; source
  `/home/ubuntu/coopmwm20261103_intentcpu_i1`.

## Nov3 ~18:15 UTC+8 — fix-first V2V HOLD; storage recovered
- All new MAIN addenda through18:00b/line872 read. Replacement V2V resource
  watcher19184 was identity-verified idle/nochildren/nojobs and **HELD** before
  disk recovery. No V2V physics, smoke or PPO update occurred. The historical
  launcher refuses. Intent-int/BANK_STAT_V2/offline+exact gate/launch3 are now
  required; the two-seed Student-t reader is superseded and will not launch.
- Full-pass imitation remains qualified as MAIN17:55 directs:778LinuxPASS;
  actual saved-bank scratch rehearsal completes5full passes,31present-member
  pairs/155visits, error1.493048->1.484244m/s²,separateKL.004488. This is
  mechanical qualification, not the pending cooperative/intent/exact fix gate.
- MAIN17:31 opaque archive COMPLETE:2files/1,250,473,414bytes, SHA proofs before
  removal; gpuR6.62GBfree at completion. e3final skipped as live-referenced.
- MAIN17:52 old generated scenario cleanup COMPLETE:23,319plain leaves,
  **4,875,992,378bytes**, under registered/reference-audited exact scope. No
  scenario/truth/vType content read; checkpoints/logs/live sources untouched.
  Full receipt SHA55217b51ea2f9113b1a67cd48b2722f1fb70b1bcef40217f9c72e2acd670acce.
  A reporting-only exec-variable collision was recovered by moving the exact
  manifest-matched metadata receipt into its control root; deletion not repeated.
- cpuQfree7,362,523,136bytes (~6.86GiB), still below10GiBmaintenance target.
  Existing rolling mover remains adopted. V2V stays held despite passing6GiB.
- Current-policy collection advances: B110decisions/52confirmed withu14data;
  R96/64withu15data;0pendingconfirmation. GPU samples94%/96%. No duplicate
  collection restart is justified. Original nonV2V training remains unchanged.
- Evidence:`coopmwm/V2V_FIX_FIRST_HOLD_1800.md`,
  `coopmwm/SCENARIO_DELETION_RECEIPT_1752.json`,
  `coopmwm/SCOPED_ARCHIVE_DISPATCH_1731.json`.

## Nov 3 17:20 UTC+8 — durable rolling storage and 6 GiB V2V auto-admission
- **17:24:31 update:** initial rolling pool complete, **18 certified batches /
  2,308,751,466bytes**. Fresh scan has zero eligible files; cpuQ free4,194,525,184
  bytes/gpuR free8,142,802,944. Launch shortfall**2,247,925,760bytes** before future
  writes. Mover and resource watcher remain live/no failures; V2V still waits.
  Evidence:`coopmwm/ROLLING_ARCHIVE_EVIDENCE_1793697872664154100.json`.
- MAIN16:47 is applied append-only. Rolling mover **4116186** is live on cpuQ;
  six Linux safety tests pass. SHA manifest, rsync, gpuR `sha256sum`, durable
  verification proof and fresh source SHA/inode/mtime/FD/reference checks precede
  each source removal. Maintenance target8GiB, gpuR hard floor4GiB, cycle15min;
  only authorised completed artifacts older than30min. Original archive adopted.
- At17:19:33, **16 rolling batches / 2,121,435,263 bytes** are certified moved.
  cpuQ free4,042,629,120bytes; gpuR free8,304,652,288. Only183,854,995bytes remain
  eligible at the scan: at least**2,215,966,829bytes additional capacity** is still
  needed for6GiB before concurrent writes. Bank inputs/checkpoints/future final
  fingerprint and discovery inputs remain protected. Old A tapes on `/dev/shm`
  cannot recover disk. MAIN has the exact storage blocker receipt.
- V2V watcher **4125157** is durable and waiting on the corrected **6GiB** gate,
  with live-mover/health checks. Two Linux gate tests pass; original769-check
  numerical release and real-bank preflight are adopted unchanged. No V2V
  physics/smoke/start is claimed. The watcher automatically launches all three
  registered update1 smokes, then exact continuation, once the gate is satisfied.
- Original non-V2V owners remain healthy at **B16/R15**, A48complete, no active
  STOP/health pause. Latest charge8,562,480sim-s/global17,471,840; storage/resource
  work adds zero simulation/inference. New registration SHAs: mover`cf49a514…`,
  resource gate`0a0fd436…`. Evidence:`coopmwm/ROLLING_ARCHIVE_EVIDENCE_1793697575319448000.json`,
  `coopmwm/V2V_RESOURCE_WATCH_DISPATCH_1647.json`.

MAIN14:50 archive first pass is qualified/registered:4 Linux CPU safety checks
pass; source mismatch,remote mismatch,open-FD and forbidden/scenario scope gates
are tested. Privileged FD scan finds0 eligible open files; admitted-bank/lesson
references are protected. Independent audit validates11 existing chosen-branch
trees and11 pending envelopes,with0 manifest conflicts.
ManifestSHA`f5c9fde4c3fddc40b24a746024f46c767f8b4c6022d9a5b7a9a5e0a6fccd1ed0`
contains7391 files/5,048,777,074bytes;55 protected and2922 sensitive/scenario
files are excluded. Registration`51f152c9…` precedes transfer/deletion by this
worker. The other session's529,080,134-byte interim-observable move is adopted;
no interim file appears in this new manifest. Only verified rsync batches may
unlink exact matching sources. This safe pool alone is below the10GiB-free
target; additional eligibility auditing will continue. No simulation/inference.

## Nov3 ~15:35 UTC+8 — V2V Linux qualification passes; disk margin consumed
- Merged V2V release passes **764 CPU checks**, zero failures/skips,86.636s,
  zero simulation/inference. One real-SUMO rehearsal is deselected by the
  inherited CPU selection; this is not a physics/launch qualification.
  Evidence:`coopmwm/V2V_LINUX_CPU_RESULT_V2.json`,SHA
  `ab514b5440fcd3f2820a2713573cccbe5fa2277842611019d470311bfbb40eda`.
- B-u12 finished15:23:03; the original B-u13/R-u12 owners progressed without
  replacement. Disk free then fell to1,823,289,344bytes: the earlier archive
  cleared admission temporarily, not the requested10GiB sustained margin.
- A second narrow archive safely declined before transfer/deletion because it
  overlaps another session's prepared5.05GB manifest. Adopt/coordinate that
  plan; no competing transfer started. See`ARCHIVE_LABEL_OVERLAP_DECLINED.md`.
- Current banks remain B12admitted/5cooperative,R13/11,0pendingconfirmations.
  These selected-bank counts are not evidence of learned policy cooperation.
  Live-bank receipt`LIVE_BANK_NULL_DIVERSITY_1793691302441970300.json` preserves
  null/diversity denominators and failed-list counts separately.
- No new physics dispatch from this session. V2V runtime/rehearsal gates remain.
  The opening6M cover-note concern is reconciled under the user's explicit
  latest-addendum precedence and already-registered17,471,840 cap; no new authority.

## Nov3 15:16 UTC+8 — disk admission recovered with verified archive
- **288 completed interim observable tapes / 529,080,134 bytes** rsynced to
  gpuR and independently SHA256-verified before source removal. Privileged FD
  and public binding/config checks pass; all RESULTs remain on cpuQ.
- Free bytes **1,994,756,096 → 2,524,499,968** clear the unchanged2GiB guard.
  The10GiB target remains unmet; other candidate roots have live/bank dependencies.
- Six archive-safety CPU tests pass. Zero new simulation/inference/reservation;
  existing training/GPU owners retained. Failed attempts and completed transfer
  preserved/adopted. Evidence: `coopmwm/ARCHIVE_MOVE_interim008_observables.json`,
  manifestSHA`b69f1c5c86d9f2fedc5dd317cbd68de76bd5508b46c7121ee0e2bfb1d46ef86a`,
  `coopmwm/DISK_RECOVERY_1450_COMPLETED.md`.

## Nov3 — exact MAIN14:06 three-arm V2V budget registered; disk admission blocked
- Reduced schedule fits **all3 V2V arms without dropping a test or diagnostic**:
  training2,073,600;u8/u24 episodes345,600 each;B lessons21,840;isolated checks
  185,200 = **2,971,840sim-s extra**, fresh references conservatively included.
  Registered global cap **17,471,840sim-s**, below the17.5M ceiling. No V2V
  final paired branches. B lessons+isolated6/12/18/24,R isolated12/24,A no extra.
- Current admitted cooperative slate includes all5 B and11 R states, including
  states whose later u11 diagnostics were refused for disk. Their original
  verification metadata/SHA remains available, so no failed state is removed
  from the cost quote. Future bank growth still needs phase room before a branch.
- QuoteSHA`f72346e46440c2b3e1e8c210211ed39ad43e1c59f6a76a6341f1669bf302a950`;
  cap registrationSHA`67320dae20a51b1ba71c874f686ab9c8c29a1eae25128c06488814e71ae91d7f`.
  Old phase allocations/charges remain; new V2V phases have0 spent. At14:49:59,
  global charges **7,709,790**,remaining9,762,050; every phase nonnegative.
- Runner gates/schedule and cost fallbacks pass10 local CPU checks; integrated
  Linux qualification remains pending. No V2V physics/reservation was dispatched.
- cpuQ has~1.9GiB free, below the2GiB admission guard. Existing B/R supervisors
  are alive at completedu11, no STOP or health pause, but next-wave admission is
  stalled. Nine original R-u11 cooperation states have retained resource-admission
  failures,not imputed/replayed. MAIN14:50 now authorises a precisely scoped,
  SHA-gated move of eligible closed artifacts to gpuR; target10GiB free.
- Evidence:`coopmwm/V2V_LAUNCH_QUOTE_1406_LIVE.json`,
  `coopmwm/V2V_CAP_REGISTRATION_1406.json`,
  `coopmwm/COMPOSITE_PHASE_BALANCES_1793688601087990800.json`.

## Nov3 14:21:56 UTC+8 — four slots and both builders healthy; live-bank niche evidence
- All4 NICHENULL3 workers are live/8 exact startup gates each/no failure. GPU
  snapshots14:09 show94%/96% utilisation. `LS_NULLFIX_CONFIG.json` is active;
  the joint niche/nullfix rollout is complete, so no gate/worker is relaunched.
- Existing `NULL_DIVERSITY_BY_CONFIG.json` covers legacy LS labels only. Current
  B/R bank verifications are now reported separately in
  `coopmwm/LIVE_BANK_NULL_DIVERSITY_BY_CONFIG.json`, preserving that old scope.
  Every current verification digest and all count/denominator invariants validate.
- Current B niches1:8 graded lists,7 complete eight-candidate lists,**0/7 exact
  null**; mean pairwise canonical-perturbation RMS distance **0.215238m/s²**,
  mean occupied niches1,any-member net-deceleration share0.578125.
  Niches4:5 lists,4 complete,**0/4 exact null**; distance **0.253278m/s²**,
  occupied niches4,deceleration share0.15. One partial list per config is excluded
  from null-rate denominators. These small cohorts are descriptive, not a causal
  estimate of the configuration's effect. Full command mirror remains off.
- R random-unscored bank:64 graded lists,54 complete,**10/54 exact null(18.52%)**;
  10 partial lists excluded. R shortlist diversity is unrecorded here and stays
  null rather than being inferred from B. Two additional B snapshots were skipped
  as inactionable before verification. Retained failed/ungraded verifications
  (B3/R9) are excluded and separately counted.
- Banks: B101 valid decisions/49 first-seed positives/49 confirmed/12 admitted/
  5 cooperative; R64/41/41/13/11. Both have0 pending confirmations at this snapshot.
- Logging-only hourly reporter is registeredSHA
  `64217ca2fedfe53862f087dbeb7b9483d23116cbef5fbe1fca6d73618edc7b60`,
  actual saved-TRAIN-read validation passes; zero SUMO/inference. Commit precedes
  its distinct read-only observer dispatch. It never owns simulation or search jobs.
  Evidence:`coopmwm/LIVE_BANK_NULL_DIVERSITY_1793686918953534400.json`.

MAIN13:43/13:45 supersedes the initial V2V start/launch assumptions: all active
arms start from **the same original MAPPO u0**,zero-init V2V,updates1..24.
Runner preparation is updated;7 local CPU tests pass(0.006s),zero simulation.
Automatic launch is conditionally authorised after integration READY,rehearsal
(or the registered >60min fallback plus real all-active-arm smokes),Linux PASS
and budget registration. The +3M ceiling/17.5M total guard and A/B-only fallback
are fail-closed. Quote24213236's conditional A/B reuse scenario2,921,760sim-s
is adopted; no cap/allocation change has occurred. New V2V u3 weights must never
be mistaken for an old bank reference when qualifying baseline reuse. Historical
own-u3 preparation receipts remain preserved. Evidence:
`coopmwm/V2V_RUNNER_COMMON_U0_PREPARATION_1343.json`.

## Nov3 13:59 UTC+8 — non-V2V u11 checks; V2V runner preparation
- B/R-u11 completed13:48:29/13:53:51. Existing watcher3305066 is healthy,
  no alerts/dead children; original pacing to24, runtime ETA~6.4h.
- B's8-state lesson saves **+609.555[-78.492,+1450.336]veh-s**,5/8 positive,
  fixed-entry action error0.592863m/s². Its bank error0.893650→0.895515 increases
  slightly during imitation; the receipt explains the registered KL stop prevented
  further fitting. Ten steps/1280 of1650 pair visits,0 full passes,KL0.065677.
- R's single-state lesson saves **+1573.780veh-s**, no CI; fixed error0.421133.
  Bank error0.622130→0.616824;1 full+partial second pass,19 steps/2312 visits,
  KL0.051399 stop. These remain conditional own-bank diagnostics.
- Composite safety passes24 paired worlds each, collision/teleport deltas0:
  B critical-13.167[-38.730,+9.584],hard-1.167[-3.604,+1.480];
  R critical-14.063[-43.376,+15.042],hard+1.104[-1.979,+4.208].
- A/B/R-v2v runner hooks are prepared with4 local CPU checks passing. Separate
  roots/job namespace,ownu3 checkpoints,zero-init component gate,endpoints24,
  inherited1800s pacing,R lag-1,shared authoritative accounting. No V2V dispatch.
  Full Linux release qualification awaits the existing integrator's READY.
- Budget helper/quote16729f67/2fadf9ac is adopted: fixed3,196,800sim-s before
  diagnostics/growth. Explicit A slate,phase funding and reporting scope remain
  admission gates; retaining legacy dev16 would add86,400sim-s. No cap is raised
  or allocation spent by this preparation. Evidence:
  `coopmwm/V2V_RUNNER_PREPARATION_1354.json`,
  `coopmwm/B_REPLAY_START_WATCH_STATUS_1793685565867810700.json`.

## Nov3 13:47:56 UTC+8 — fixed-u8 interim result (public per MAIN13:35)
- All96 paired worlds (32 focus), fixed A/B/R8; MAPPO/native reused. V2 adds
  B-native without rerunning simulation. Full result: `coopmwm/INTERIM_U8_RESULT.json`,
  SHA`f4f357e377fe5c5385460af42ef3fb717dcb2760f9233f91e7d9e95f21c74ef2`.
- Full-panel treatment-minus-comparator delay, veh-s,95% paired world-bootstrapCI:
  - B-A: J600 **-208.070[-835.088,+428.595]**; J900
    **-532.630[-1641.897,+560.383]**,0.543% reduction; J900 Holm p0.666333.
  - B-MAPPO: J600 **-558.379[-1184.419,+52.760]**; J900
    **-762.266[-1822.305,+287.144]**,0.776% reduction; J900 Holm p0.318768.
  - B-R: J600 **-398.038[-979.476,+175.839]**; J900
    **-900.813[-1789.251,+3.098]**,0.915% reduction; J900 Holm p0.102790.
  - B-native: J600 **-1313.881[-2203.660,-435.751]**,2.356% reduction,
    Holm p0.005599; J900 **-2406.308[-3927.485,-952.685]**,2.408% reduction,
    Holm p0.004400. Only this full-panel comparator is resolved at the stated level.
- Focus J900: B-A **-440.173[-2615.379,+1736.107]**; B-MAPPO
  **-880.858[-2903.912,+1212.533]**; B-R **-152.000[-1661.045,+1381.973]**;
  B-native **-538.002[-3137.097,+1863.053]**. No focus contrast resolves.
- Collision/teleport differences are0 for every reported comparison. Full-panel
  J900 B-native TTC<3 vehicle-seconds **-39.474[-77.448,-2.180]**, hard brake
  **-4.854[-7.297,-2.458]**; individual safety CIs are descriptive/unadjusted.
  Corresponding B-MAPPO differences are-13.807[-43.531,+15.641] and
  -1.391[-2.860,+0.125]. Per-cell and full/focus safety results are in the JSON.
- These are **interim episode-level** results, not the final paired-branch tests
  and not evidence of learned group cooperation. Bootstrap10000/RNG20261101;
  alpha.05 Holm over{all,focus} separately per comparator and J endpoint, with
  no across-endpoint/interim-final multiplicity claim. V1 results are preserved.
  No training/search/checkpoint change is conditioned on the interim; current
  non-V2V B/R continue to fixedu24. V2V follows the separately approved preparation.
- V2 registration`3ca57ba9…`/commit`a3f93c2a` preceded analysis. Owner3647689
  completed/no failure, zero SUMO/inference. Delivery receipt
  `coopmwm/INTERIM_U8_ANALYSIS_DELIVERED_1335_V2.json`.

MAIN13:35 authorises public interim results and adds B-native. Additive V2 uses
the same completed episodes/qualified bootstrap and retains all V1 artifacts.
Four Linux CPU tests pass(0.121s), zero simulation/inference. Registration
`3ca57ba9c925f179df5e1798972eac954ddc7791f6c365c15730a65c2b130c8b`
precedes V2 decoding. B-R/B-native two-scope Holm families are added separately
per endpoint; original B-A/B-MAPPO families retain their values. V2V work is
preparation-only per MAIN13:40; current non-V2V training continues.

## Nov3 13:40 UTC+8 — fixed-u8 analysis delivered to MAIN; durable training healthy
- The authorised interim analysis completed **13:38:57**, all96 paired worlds,
  fixed A/B/R8 plus reused MAPPO/native. All480 episode identities and stored
  outcome fingerprints validate. No SUMO/inference was added. Owner3625742 exited
  with a completion certificate/no failure; adopt completion and never rerun.
- MAIN-only delivery is `E:/agent-work/_sol/COOPMWM_INTERIM_U8_MAIN.md` with
  adjacent JSON. Public resultSHA
  **`b7046c883cc60532587343830026ace78de9ba6e7527db7ff6f43e07fedb7450`**;
  manifestSHA`5ff0fd41fafdf86cb5553849713812239722e1937748d09338dbbecc2a7db74e`.
  Report values are not reproduced here. Final-policy outcomes remain unopened.
- At13:41:00 watcher3305066 and all4 original training/grading children are live,
  no failures/alerts/dead children; B/R remain completedu10 under original pacing
  to fixedu24. Analysis supplied no data to training and changed no live source.
- Budget13:40:49: **7,459,360/14,500,000sim-s charged**,7,040,640remaining;
  bank extension1,723,900remaining. Train806,400 exactly funds28 remaining arm
  updates. Every phase is nonnegative; interim charge345,600 equals its complete
  A/B/R episode quote. SnapshotSHA`236f9c0d8c46d77ad29d0a119f288a25fde9fa3e06a0b12174a6b45a6a66701d`.
- Evidence: `coopmwm/INTERIM_U8_ANALYSIS_DELIVERED_1235.json`,
  `coopmwm/INTERIM_U8_ANALYSIS_STATUS_1793684395479193800.json`,
  `coopmwm/B_REPLAY_START_WATCH_STATUS_1793684462261363500.json`,
  `coopmwm/COMPOSITE_PHASE_BALANCES_1793684451329583600.json`.

MAIN-only interim analysis is live under owned cpuQ3625742/start69850280 from
13:37:05+08. Registration`99552af1…`/commit`c2ee5bf5` preceded launch; detached,
nice19, no SUMO/inference. At13:38:14 it is running without a failure certificate.
Adopt `coopmwm/INTERIM_U8_ANALYSIS_DISPATCH_1235.json`; never duplicate it.

## Nov3 — completed fixed-u8 interim; MAIN-only analysis registered before decoding
- Public identities verify **96 episodes per A/B/R**, checkpoint8 SHA per arm,
  panelSHA`d74f685e…`; both completion certificates exist/no failure. Reused
  MAPPO/native public completions also validate,480 episode identities total.
- MAIN12:35's completed-interim analysis is registered as
  **`99552af12ce0c52d0d06d1bd1690dc42a9d1078fd5f9d727c8008a8dd54f1e62`**
  before outcome decoding. Four Linux CPU tests PASS (0.107s), zero SUMO/inference:
  incomplete-grid and final-checkpoint exclusion, fingerprint mismatch before
  decode, paired-statistic/focus/duplicate-world fixtures and finite JSON safety.
- A detached analysis will read only the completed interim008 A/B/R episodes
  and reused fixedfinal MAPPO/native episodes through checked paths; no final
  policy outcomes. Existing bootstrap10000/RNG20261101/95%CIs, all96/focus,
  J600/J900 contrasts and existing safety metrics; percent reductions/per-cell
  descriptives. Interim Holm is disclosed per endpoint/comparator over{all,focus},
  with no across-endpoint/interim-final multiplicity claim; final branch families
  retain their registration. MAIN-only delivery; public metadata/hashes only.
- No SUMO reservation, inference or training handover. All result-dependent
  training/search/loss/checkpoint changes remain prohibited. Evidence:
  `coopmwm/INTERIM_U8_ANALYSIS_CPU_1235.json`,
  `coopmwm/INTERIM_U8_ANALYSIS_REGISTRATION_1235.json`,
  `coopmwm/INTERIM_U8_COMPLETION_INTEGRITY_1793683899283639300.json`.

## Nov3 13:25 UTC+8 — both u10 updates complete; all phases remain funded
- R-u10 completed **13:23:57**. Watcher3305066 and all four children remain live,
  no alerts/dead children. Runtime ETA to R-u24 is about7.01h; B-u24 about7.04h
  as of its own last update. Original pacing and fixed endpoints continue.
- R's single frozen lesson saves **+853.695veh-s**, no CI from one entry;
  fixed-entry action error0.364220m/s². It is not a paired B-vs-R performance
  estimate. R imitates **11 of13** available admitted targets, capped by B-u9's11.
  Five passes requested;1 full pass plus part of the second,18 optimizer steps,
  2252 pair visits/1830 present-member targets, KL0.055908 stop. Within-update
  bank action error **0.620863→0.608462m/s²**.
- R10-A10 safety passes on24 paired worlds, collision/teleport deltas0:
  hard brake **+0.021[-3.438,+3.333]**, critical total
  **+18.396[-8.918,+45.813]**. Both arms' positive critical point estimates are
  reported; neither exceeds the registered composite significance predicate.
- Budget13:25:12: **7,323,310/14,500,000sim-s charged**,7,176,690remaining;
  bank extension1,805,760remaining. Train has806,400remaining, exactly28 remaining
  arm-updates×24×1200. Every phase is nonnegative. SnapshotSHA
  `eebd6fd86209415598f9703ab1b154822b9c45898e7e3edf0db32398f066545e`.
- Interim public completions13:25:01:A96/B91/R96; R completion/no-failure is
  adopted, A/B still live. Outcome values remain closed.
- Canonical `coopmwm/LESSON_CHECK_u010_R.json` and
  `coopmwm/R_SAFETY_COMPOSITE_010.json` are published. Evidence:
  `coopmwm/B_REPLAY_START_WATCH_STATUS_1793683486875221800.json`,
  `coopmwm/COMPOSITE_PHASE_BALANCES_1793683513532494000.json`.

## Nov3 13:20:58 UTC+8 — isolated R-u9 complete on all8 frozen states
- R-u9 isolated V2 is complete: **42 new branches /34,080sim-s precharge**,
  all8 states graded, report seals and combined/original identities verified.
  The original+140.267veh-s gain on986036/t300 becomes **-225.481veh-s** when
  background CAVs stay on u3: group-only saving+74.059 versus best solo+299.540
  (ramp.23); background-only saving+570.164. Its original verified-plan gain437.185
  remains a separate plan result.
- On986040/t300 (three-member state), group-only saving+820.868 is below the
  best solo+993.583, isolated cooperation gain **-172.716veh-s**. The other
  isolated gains are -0.003237410(986037/t360),-0.000282770(986065/t760),
  -0.023060614(986001/t510),-0.702182829(986051/t360),
  -0.000000889(986040/t300 two-member state), and **+0.000050677**(986038/t320).
  The one positive sign is retained at its measured magnitude. These results
  do not demonstrate substantive transfer of verified-plan cooperation at u9.
- B and R diagnostics use their own frozen bank states; these are not paired
  B-vs-R estimates. Existing sidecars continue subsequent updates. Canonical
  `coopmwm/POLICY_SYNERGY_u009_R_ISOLATED_V2.json` and
  `coopmwm/POLICY_SYNERGY_u009_R_WITH_ISOLATED_V2.json` preserve the original
  diagnostic alongside the isolated one. Status evidence:
  `coopmwm/POLICY_SYNERGY_ISOLATED_STATUS_1793683258153140100.json`.

## Nov3 13:19 UTC+8 — B-u10 complete; R-u10 collecting
- B-u10 completed **13:18:52**, with runtime ETA about7.04h to fixedu24.
  Watcher3305066 remains live/no alerts/dead children. R-u10 has12 claimed
  episodes at13:19:39 and follows the original lag-1 pacing.
- B's fixed8-entry matched lesson saves **-296.774veh-s**, entry-bootstrap95%CI
  **[-784.598,+96.451]**,2/8 positive. Fixed-entry imitation error0.591195m/s².
  The selected-bank check is noisy across updates and does not establish a gain.
- Bank-only imitation requests5 passes and stops during the first at the existing
  KL bound:7 optimizer steps,896/1650 pair visits,607 present-member targets,
  zero complete passes; final all-bank KL0.047416. Within-update bank action error
  **0.898004→0.895713m/s²**. The enlarged bank makes its raw error incomparable
  with u9's different target set. Critic steps and plan-ratio computations are0.
- B10-A10 safety passes on24 paired TRAIN worlds: collision/teleport deltas0;
  hard braking **-0.083[-3.479,+3.125]**, critical total
  **+5.750[-26.729,+39.396]** per episode. The positive critical point estimate
  remains disclosed; the registered nonsignificance predicate passes.
- R-u8 interim has completed **96/96**, completion certificate present/no failure;
  its owner exited normally and is adopted as complete. A/B owner3442884 remains
  live, public completions A96/B78 at13:19:29. Outcome values remain closed.
- Canonical `coopmwm/LESSON_CHECK_u010_B.json` and
  `coopmwm/B_SAFETY_COMPOSITE_010.json` are published. Full metadata:
  `coopmwm/B_REPLAY_START_WATCH_STATUS_1793683180855652300.json`,
  `coopmwm/INTERIM_U8_PUBLIC_STATUS_1793683171122168300.json`.

Operational check13:16+08: both GPUs' two existing NICHENULL3 workers are live,
eight gates qualified per slot, no failure; utilisation snapshots93%/94%.
B-u10's first12 episodes are advancing (observed t1005–1139/1200 at13:13:09);
R remains at completedu9 under lag-1 pacing. No STOP/active B/R pause. Interim
public completions13:13:46:A96/B62/R84, no failures. Local integrity validation
passes all4 u9 report seals, combined/original identity, all20 isolated branches
and phase arithmetic, with zero new SUMO/inference. Artifact
`coopmwm/U9_RECEIPT_INTEGRITY_1793682964164781700.json`. Corrected status also counts
R-u9's5 separately stored fresh-u3 references:38 new branches, not33. Original
R-u9 diagnostic has8 graded states; its+140.267veh-s gain on986036/t300 retains
the original new-policy background, so the isolated variant is needed to interpret it.

## Nov3 13:10 UTC+8 — complete isolated B-u9 check; current banks keep growing
- All four original cooperative B-u9 states are now graded in the isolated V2
  diagnostic: 20 new branches,14,760sim-s precharge, no failed or imputed state.
  With every non-group CAV on u3, group-only saving / best-solo saving /
  isolated cooperation gain are, in veh-s:
  - 986048/t300: **+0.270240 / +0.270273 / -0.000032718**.
  - 986052/t300: **+0.022878 / +0.023592 / -0.000713562**.
  - 986066/t300: **-0.051186 / +0.018387 / -0.069574**.
  - 986003/t390: **0 / 0 / 0**.
- Background-only savings on the same states are respectively **+564.694,
  -379.788,+4989.651,+2071.233veh-s**. Original verified-plan cooperation gains
  remain +958.678,+0.000496,+4430.099,+894.879. This complete conditional check
  does **not demonstrate learned group synergy at u9**. Background effects are
  measured by their own branches; group and background effects need not add.
- Independent isolated owner3531845 remains live; R-u9 is in progress. Original
  diagnostic3443527 and training3305066/children continue with unchanged settings.
- At13:10:31 the banks contain B **96 decisions /46 first-seed positives /
  12 admitted /5 cooperative** and R **46 /31 /12 /10**. Conditional admitted-plan
  mean gains are B **+725.069[126.613,1516.417]**, R
  **+573.599[181.437,1039.873]veh-s**, decision-bootstrap95%CIs. These are separate
  selected-bank summaries, not a paired B-vs-R effect or policy-performance result.
  The B bank now includes one admitted u7-policy state; the u9 diagnostic retains
  its original frozen four-state list rather than adding later admissions.
- Budget13:08:35: **7,073,420/14,500,000sim-s charged**,7,426,580remaining;
  bank extension1,889,900remaining; all phase balances nonnegative. SHA
  `4a7cf8124826c2eb9dcda1cc8c7421f0731be9c0bd07edc97b5cc2c464c55486`.
- Both interim owners are live/no failure. Public completions at13:07:37 are
  A96/B45/R68; only public metadata was observed. Canonical isolated and combined
  V2 reports are mirrored; no training, inference or SUMO dispatch was duplicated.
- Evidence: `coopmwm/RESUME_BANK_ISOLATED_1793682632904292600.json`,
  `coopmwm/POLICY_SYNERGY_u009_B_ISOLATED_V2.json`,
  `coopmwm/COMPOSITE_PHASE_BALANCES_1793682516626446700.json`.
- Receipt-integrity correction: the earlier public status counter counted16
  group/all/solo branches but omitted4 background-only branches stored separately
  in each entry. The charge already includes all20. Local status counting now
  also includes separately stored fresh-u3 reference branches. Original raw
  reports and earlier status receipts are preserved.

Interim metadata update13:00:41+08: **A96/B26/R48** episode hash receipts complete,
both owners live/no failures; outcome values remain closed. V2 read-only observer7456
is creation-time/argv verified live and continues the120s receipt mirror.

## Nov3 12:59 UTC+8 — both u9 checks published; first isolated state is graded
- B-u9 completed12:48:41, R-u9 at12:53:56. Watcher3305066 and its children
  remain healthy, no alerts/dead children; fixedu24 and original pacing continue.
- B's fixed8-entry matched lesson saves **+874.800veh-s**, entry-bootstrap95%CI
  **[-135.269,+2245.181]**,4/8 positive, fixed-entry action error0.592962m/s².
  Its11 admitted bank targets run8 imitation optimizer steps/1024 pair visits,
  zero full passes before minibatch KL stops; within-update error0.829575→0.826316.
- R's fixed1-entry lesson saves **+1750.802veh-s**, no CI from one entry, fixed
  error0.297508. Its10 admitted targets complete1 full pass plus part of the second,
  17 steps/2080 pair visits; within-update error0.637467→0.617574. Both arms keep
  the requested5-pass rule with the registered KL early stop.
- u9 safety passes for both arms on24 paired worlds, collision/teleport deltas0:
  B critical total **+0.104[-20.792,+21.834]**, hard braking+0.354[-3.146,+3.792];
  R critical total **+6.604[-17.771,+30.710]**, hard braking-0.146[-3.875,+3.167].
- First isolated V2 state986048/t300 is fully graded:4 new branches,2880sim-s,
  exact prefix/CRN checks passed. Group-only-new-policy saving **+0.270240veh-s**;
  solo ramp.20 **-0.000059318**, solo ramp.22 **+0.270273**; isolated cooperation
  gain **-0.000032718veh-s**. Background-only saving **+564.693983veh-s**, while
  the original verified plan's cooperation gain was+958.678083. This state does
  not show learned group synergy. Remaining states are in flight; no extrapolation
  to the final panel is made from these conditional TRAIN-bank diagnostics.
- Budget12:59: **6953330/14500000sim-s charged**,7546670remaining,
  bank extension1951370remaining; all phases nonnegative. SHA`6c4da9c2aa2701cc…`.
  Canonical u9 lessons/safety are published. Interim outcome values remain closed.

## Nov3 12:57 UTC+8 — isolated accounting guard fixed; interim continues
- First isolated B-u9 attempts collided with the original diagnostic's charge
  identity. The duplicate guard rejected all4 **before any new reservation or
  physics**. Original failed receipts/alerts remain; the verified idle isolated
  watcher alone was retired. Training and the original diagnostic were unaffected.
- V2 has a distinct isolated charge namespace and output suffix. It passes
  **48 Linux CPU checks**, zero skips,1.428s, zero SUMO, including the real charge
  adapter's identity separation and both output namespaces. RegisteredSHA
  `097e90690ac079f2…`/commit`909b0139` precede dispatch **12:56:55**, owner3531845.
- New read-only mirror7456/create1793681824.706049 is live; it publishes isolated
  and joint`WITH_ISOLATED_V2` receipts plus public interim counts every120s.
  Original observer44248 remains untouched; all previous receipts are retained.
- Interim at12:57:18: **A96/B17/R38** episode hash receipts complete. Both interim
  owners remain live, no failure; outcome values remain closed.

## Nov3 12:50 UTC+8 — isolated cooperation diagnostic is durable from u9
- MAIN12:26 is registered as SHA`efd1fefae30313b1…`; immutable`isosynergycpu1`
  passes **45 Linux CPU checks**, zero skips,1.471s, zero SUMO. Commit`a6698e15`
  precedes dispatch **12:47:54**, cpuQ owner3511227/start69555223.
- From u9, both arms retain the original cooperation test and add group-only
  all/solo-new-policy tests with every background CAV on u3, including arrivals.
  A separate background-only test runs non-group cars new/group cars u3. All
  use the original frozen state list, prefix, window and CRN. The exact sealed u3
  reference is adopted; the all-new lesson is never mistaken for an isolated test.
- Original diagnostic3443527 and training supervisors continue. Combined reports
  use`POLICY_SYNERGY_uNNN_ARM_WITH_ISOLATED.json`; original reports stay immutable.
  Complete new branch sets are precharged to the existing bank-extension phase.
- At12:49:57, B-u9 is complete/R-u8 remains last complete; watcher is live with no
  alerts/dead children. Read-only budget12:48:55: **6888240/14500000sim-s charged**,
  7611760remaining, bank extension1981660remaining; all phases nonnegative.
  BudgetSHA`372f9f6203ffa2f6…`. Interim outcomes remain closed.

## Nov3 12:44 UTC+8 — fixed-u8 interim running, including the requested R supplement
- Existing A8/B8 interim was already running under3442884 since12:18:08. Its
  immutable192-episode manifest is retained. At12:44:07,77 A episode hash receipts
  are complete; B follows in the original manifest order.
- MAIN12:35's R8 supplement is now live under3499917/start69527524, independently
  of training and the existing A/B process. First2 R episode hash receipts exist,
  no failure. All96 R episodes are precharged115200sim-s to`interim_u8_test`;
  combined A/B/R quote345600 remains within the registered restricted allocation.
- Immutable`interimrcpu2`:22 Linux CPU checks PASS, zero skips,0.861s, zero SUMO.
  AmendmentSHA`f377f10f7f414783…` and commit`ba37ebea` preceded dispatch. V1's
  source-identity assertion failed before accounting/reservation/physics; its
  receipts remain preserved. V2 validates the original training-source hashes.
- Status is metadata-only; outcome values remain closed. Training owners and
  original update pacing continue. Artifact`INTERIM_U8_PUBLIC_STATUS_1793681049101241300.json`.

## Nov3 12:27 UTC+8 — both arms at u8; future diagnostic publication is durable
- B/R-u8 complete; watcher3305066 and all children live with no alerts/dead children.
  Original minimum1800s pacing continues to fixedu24. R-u8 has8 admitted targets,
  7 cooperative; its AWR completes3 full passes before KL0.045123 stops it, action
  error **0.689350→0.666460m/s²**. R's fixed1-entry lesson saves **+199.140veh-s**,
  no CI from one state, fixed-entry action error0.242092m/s².
- R8-A8 composite safety passes: critical total **+17.688[-6.876,+43.813]**,
  hard braking **+0.0625[-3.354,+3.333]**,24 paired worlds and collision/teleport
  deltas0. Positive critical point estimates for both arms remain disclosed.
- Policy-synergy sidecar3443527 continues R-u8 and every later update8..24.
  Detached read-only local observer **44248/create1793679983.3196542** is verified
  live and has completed its first receipt mirror. It publishes future canonical
  reports every120s without owning training, inference or SUMO jobs.
- Latest budget **6,527,020/14,500,000sim-s charged**,7,972,980remaining;
  bank extension2,039,970remaining, every phase nonnegative. Artifact
  `COMPOSITE_PHASE_BALANCES_1793680041357999500.json`, SHA`382cce69e605d154…`.
  Interim charges are present in the ledger; no sealed outcome access occurred.
- MAIN notification:`E:/agent-work/_sol/COOPMWM_POLICY_SYNERGY_U8_1227.md`.

## Nov3 12:23:02 UTC+8 — B-u8 policy-cooperation test complete on all4 states
- `coopmwm/POLICY_SYNERGY_u008_B.json` is complete: every then-admitted
  cooperative B bank state is graded, with no failed/imputed branch. Three
  all-new lesson branches were adopted;13 new branches were run, precharged
  **9,630sim-s** in the existing bank-extension phase.
- Policy cooperation gains, with original plan gains alongside (veh-s):
  - 986048/t300: **+0.000008189**, plan+958.678.
  - 986052/t300: **−0.003890589**, plan+0.000495724.
  - 986066/t300: **−168.935077**, plan+4430.099.
  - 986003/t390: **0**, plan+894.879.
- Thus the first u8 check does **not demonstrate meaningful transfer of the
  bank plans' multi-car synergy into the policy**. The tiny positive numerical
  sign on986048 is retained, without a new behavioural threshold. This is a
  conditional TRAIN-bank diagnostic, not the registered final comparison.
- The same sidecar has begun R-u8. Training remains under its original owners
  and pacing; no search/loss/checkpoint change is conditioned on these numbers.
  A read-only local observer is prepared to mirror future reports every120s.

## Nov3 12:19:57 UTC+8 — diagnostic live; first real prefix/CRN checks pass
- Registered code commit`668f04fd` preceded sidecar dispatch **12:18:26+08**.
  cpuQ owner **3443527/start69378435** remains live under detached supervision;
  adopt `POLICY_SYNERGY_DISPATCH_1205.json`. Training owners are untouched.
- First2 B-u8 states have complete results:4 new solo SUMO branches are graded,
  with exact prefix checks and CRN audit passing;2 all-new matched lessons were
  SHA-adopted without simulation. At986048/t300 policy cooperation gain is
  **+0.000008189veh-s** versus the original plan's+958.678; at986052/t300 it is
  **-0.003890589veh-s** versus the plan's+0.000495724. These near-zero values
  do not establish learned multi-car cooperation. Remaining states are in flight.
- **B-u8 completed12:18:08**, R-u7 remains the last completed R update. B-u8 fixed
  8-entry lesson saving **+276.946veh-s**, entry-bootstrap95%CI
  **[-517.476,+1342.929]**,5/8 positive; fixed-entry action error0.601449m/s².
- B-u8 imitates11 admitted targets/1620 pairs. It runs13 optimizer steps/1620
  pair visits before the registered KL stop;0 full passes completed, final
  KL0.050201. Within-update bank error **0.845886→0.833554m/s²**.
- B8-A8 safety: critical total **+13.771[-12.876,+44.563]**, hard braking
  **+1.438[-3.250,+6.146]**,24 paired worlds; collision/teleport deltas0. The
  composite check passes. Watcher3305066 has no alerts or dead children.
- Evidence:`B_REPLAY_START_WATCH_STATUS_1793679557665986900.json` and
  `POLICY_SYNERGY_STATUS_1793679598403885800.json`; no sealed outcome access.

## Nov3 12:17 UTC+8 — policy-cooperation diagnostic qualified and registered
- MAIN12:05 is registered as amendment SHA
  `8d01d928339efc54212ae9f47b1bd4a2a38e938cfe48816d510fa916a118d976`.
  Immutable `policysynergycpu2`: **37 Linux CPU checks PASS, zero skips,
  1.134s, zero SUMO**. Tests cover member-specific policy swaps with shared noise,
  exact synergy algebra, snapshot/window/checkpoint identity, atomic complete-set
  precharge, later-policy states' fresh u3 comparator, and no interrupted-job replay.
- A separate durable diagnostic will measure every then-admitted cooperative bank
  state after each B/R update8..24. It compares all group members using the new
  policy with each member alone using it; the other group members use u3. Non-group
  cars use the new policy as in the matched lesson check. The common captured t0
  action is retained in every branch; the intervention starts at t0+1, as in lessons.
- Sealed exact-checkpoint/state/window all-new lesson branches are adopted, never
  rerun. Other branches are precharged as a complete set in `bank_verification_ext`.
  Original plan cooperation gain is reported alongside policy cooperation gain.
  Failures remain incomplete and alerted; no missing branch is imputed.
- Read-only budget12:16:21: **6,388,610/14,500,000sim-s charged**,
  **8,111,390remaining**; bank extension has **2,079,620remaining**. Every phase is
  nonnegative. Artifact `COMPOSITE_PHASE_BALANCES_1793679382157456200.json`,
  SHA`a33f9fbf037d8366…`. No new simulation or training handover at qualification.

## Nov3 11:54:33 UTC+8 — B/R-u7 complete; conditional learning checks continue
- Both u7 update receipts exist; watcher3305066 and all four children remain live,
  with no alerts/dead children. Original minimum1800s update pacing remains active.
- B-u7 bank grows to11 admitted targets/1620 pair records. It requests5 passes,
  starts the first and takes4 optimizer steps/512 pair visits before the
  **minibatch-local KL stop**. Its final all-bank mean KL0.038714 is a different
  aggregate from the triggering minibatch. Action error0.846535→0.846204m/s²;
  the growing bank makes this raw error incomparable across updates.
- R-u7 has3 admitted targets/450 pairs and completes all5 passes,2250 pair visits,
  KL0.007434; action error0.753600→0.747355m/s². Actual target counts satisfy the
  registered admitted-target cap, with no B data supplied to R.
- Fixed8-entry B lesson: **+445.012veh-s**, entry-bootstrapCI
  **[-78.696,+1025.124]**,6/8 positive; mean fixed-entry action error0.614341m/s²
  (u4/u5/u6:0.646729/0.641686/0.627479). R's fixed1-entry lesson:+3736.900veh-s,
  no CI; that entry's action error0.195964m/s². These remain conditional bank checks.
- u7 B-A hard-brake delta **-1.354[-4.250,+1.458]**, critical total
  **-13.813[-41.188,+19.417]**; R-A hard braking **+0.083[-3.521,+3.229]**,
  critical total **-13.000[-41.084,+15.729]**,24 paired worlds each. All collision/
  teleport differences0; both composite checks pass.
- Canonical u7 lesson/safety receipts are published in `coopmwm/`; full current
  status:`B_REPLAY_START_WATCH_STATUS_1793678074743887200.json`. The remote run
  continues to fixedu24 and the pre-registered hash-sealed evaluation stages.

## Nov3 11:20:36 UTC+8 — TRAINING RUNNING; B/R-u6 complete with full-bank exposure
- Registered release/commit`709a2f4f` preceded actual restart **11:11:32+08**.
  Durable cpuQ watcher **3305066/start68977029**, B**3305070**, R**3305071**,
  graders**3305068/3305069** are verified alive, no child failures. Both update6
  receipts exist. The24 closed B-u6 episodes were reused without simulation or
  recharge; R-u6 collected fresh registered episodes.
- B requested5 full bank-only AWR passes: **2 complete, third started**,2236 pair
  visits/2013 present-member targets/18 optimizer steps. The sole early-stop reason
  is the existing pre-update KL bound (final0.05557 after the triggering step,
  threshold0.045). Bank action error **0.689699→0.678985m/s²**; before the whole
  update it was0.694563. Imitation has0 critic steps and0 plan-ratio computations.
- R completed **all5 passes**,600 pair visits/60 present-member targets/5 steps,
  KL0.003071. Bank action error **0.211841→0.211027m/s²**. It imitated1 admitted
  target within the actual B-target fairness cap; rejected lists do not consume it.
- B-u6 fixed8-entry matched-window lesson: mean absolute saving **+413.050veh-s**,
  entry-bootstrap95%CI **[+1.992,+956.951]**, **6/8 positive (75%)**, mean per-entry
  action error0.627479m/s². This is the conditional frozen TRAIN-bank learning
  check. R's frozen lesson currently has1 entry: saving+1207.756veh-s, no CI from
  one entry; its mean action error is0.182858m/s².
- Both actual u6 safety checks pass the same composite rule. B6-A6 critical total
  **+7.208[-19.855,+33.021]**; R6-A6 **-8.792[-35.626,+15.792]**,24 paired worlds
  each; all collision/teleport differences0. Hard-brake deltas remain separately
  reported: B+3.625[+0.708,+6.521], R+0.750[-2.042,+3.542]veh-s/episode.
- R bank at11:24:16: **6 exact-graded decisions/4 first-seed positives/4 confirmed/
  1 admitted**,1 cooperative under the registered group-vs-best-solo criterion.
  That entry's first-seed group gain3037.483, best solo593.902, difference2443.580
  veh-s; one admitted entry gives no population CI. Original identity declines
  remain byte-preserved with seven SHA-verified copies; no WM inference was rerun.
- MAIN10:56's requested **`coopmwm/SAFETY_RELEASE_u6.json`** is published with
  localSHA`ff8f4462e197138c…` and remoteSHA`96b61cfebea384ecb…`; it references the
  active853bb0b6 certificate and preserves the original pause. Same guard runs
  before every subsequent B/R PPO update. Fixed30-minute minimum pacing and final
  checkpoints24 remain; sealed outputs remain unopened.
- Published artifacts: `LESSON_CHECK_u006_B.json`, `LESSON_CHECK_u006_R.json`,
  `B_SAFETY_COMPOSITE_006.json`, `R_SAFETY_COMPOSITE_006.json` in `coopmwm/`.
- Locked budget snapshot11:41:25: **6,081,770/14,500,000sim-s charged**,
  8,418,230remaining; every phase nonnegative. Train has1,036,800remaining,
  exactly36 fresh arm-updates×24×1200 after u6. Per-phase artifact
  `COMPOSITE_PHASE_BALANCES_1793677286478811300.json`, SHA`a9deb6e7210a6909…`.
- At11:45:04, B-u7's24 episodes are already closed; its guard/update is processing.
  R-u7 waits under the registered lag-1 rule. Watcher remains alive, no alerts or
  dead children, and no sealed outcome access.

## Nov3 11:02 UTC+8 — MAIN09:57 diagnosis passes; SHA-bound release registered
- Full read-only artifact **`coopmwm/SAFETY_DIAGNOSIS_u6.json`** is published.
  B6/A6 and last-completed R5/A5 each contain24 paired, unique TRAIN worlds.
  Collision and teleport differences are0 in every pair.
- B6-A6 total safety-critical events: **+7.208/world**,95%CI
  **[-19.855,+33.021]**; hard-brake component+3.625[+0.708,+6.521], TTC<1.5
  component+3.583[-23.417,+28.917]. The positive point estimate is retained;
  total critical harm is not statistically higher under MAIN's fixed rule.
- R5-A5 total:+0.083[-21.104,+20.605]. Both comparisons satisfy the registered
  release predicate. R6 has not run and must pass the same composite guard before
  its PPO update. The original pause remains on disk; only its exactSHA is released.
- A hard-braking means for u5/u6/u7 are17.354/15.979/16.125veh-s; SDs
  10.185/7.128/8.881. Panel-mean changes are-1.375/+0.146, but panels have no
  common worlds, so these are descriptive and are not paired variability CIs.
- Largest extra-braking worlds:986046(+18),986070(+16.5),986040(+9.5veh-s).
  Across2160 inherited10s windows, continuous deceleration variation versus
  positive issued-CAV-acceleration differences have Pearson r=-0.278. This is a
  descriptive proxy, not an event locator, known-move detector or causal test.
- B5-A5 hard braking:-0.542[-3.083,+2.000]; B6-A6:+3.625[+0.708,+6.521].
  Bank action error changes0.646729→0.641686m/s². Saved metrics show3960 bank-pair
  visits across705/706 mixed PPO batches, rather than only four pair visits; those
  losses were diluted among PPO records. Episode data do not isolate an
  imitation-only braking effect. Five dedicated post-PPO passes remain the fixed
  owner-approved correction, with pre-update KL stopping and before/after errors.
- Immutable `compositereleasecpu1`: **727 Linux CPU checks PASS, zero skips,
  84.104s,zero SUMO**, plus16 local observer/payload checks. Registered release
  binding SHA **`853bb0b6d3afcbc71b7f5ee7c028ebf971d196fe0380d7f977fb702c55d08dbe`**
  verifies diagnosis/rule/pause hashes and leaves every other/new pause active.
  Future B/R updates use the same paired composite check; new live-safety evidence
  cannot overwrite or hide behind the released original.
- Prepared continuation starts from actualcheckpoint5, applying u6 once with the
  new loss;24 completed B-u6 episodes are reused without simulation or recharge.
  Fresh PPO remaining quote1,065,600sim-s fits the existing train phase. R's cap is
  corrected to actual admitted B imitation targets, applied after confirmation;
  rejected lists no longer consume the cap. This is before any R bank grading.
  Dispatch follows the committed binding; no training-start claim at registration.

## Nov3 10:27:06 UTC+8 — corrected fixed-u4 lesson complete, all8 entries graded
- The reporting-only correction **adopted2 valid branches and ran only6 missing
  matched-window branches** under the fresh registered `lesson_u004matched_v3`
  charge/work namespace. All8 entries are graded; no completed branch or episode
  was rerun. Original/V1/V2 outputs and charges remain retained.
- Primary fixed-bank paired saving `J_policy_u3 - J_new_policy`: **+110.659veh-s**,
  95% entry-bootstrapCI **[-252.400,+511.493]**; **5/8 entries improved (62.5%)**.
  This is a conditional diagnostic on the frozen8 admitted TRAIN entries, not a
  population/dev/test policy-effect estimate; the interval includes zero.
- Replication-spread-qualified secondary recapture:7 eligible entries, median1.318,
  CI[-0.147,1,633,604.461]. The raw ratios/very wide secondary interval are retained;
  primary reporting uses absolute veh-s as MAIN09:06 requested.
- New artifact:`coopmwm/LESSON_CHECK_u004_B_MATCHED_WINDOW_V3.json`.
  Receipt:`coopmwm/B_LESSON_CORRECTION_V3_STATUS_1793673746905980500.json`.
  Correction release:8 targeted Linux CPU checks PASS/0skips,0.772s,zero SUMO;
  registeredSHA`7b1b585cc3524fe2…`,4260sim-s upper in existing extension phase.
- Training remains health-paused at checkpoint5. MAIN09:57's composite safety
  diagnosis is registered before its numbers (SHA`1870328f0245c967…`), passes10
  targeted Linux CPU checks, and is running read-only under durable owner3231042.
  Its result, not this lesson diagnostic, determines eligibility for MAIN's release.

## Nov3 09:55:36 UTC+8 — registered SAFETY pause before B-u6 PPO
- The fixed hard-braking guard paused B/R. Across24 unique matched TRAIN worlds,
  B-u6 minus A-u6 is **+0.00402778 hard-brake vehicle-seconds/control-second**,
  paired-world95%CI **[+0.000787037,+0.00724537]**. Equivalently+3.625veh-s/900s,
  CI[+0.708333,+6.520833]. Registered10000 world-bootstrap/RNG20261101.
- Read-only checks confirm identical A/B world descriptors for all24 pairs and
  zero executed-plan steps in B-u6. No pairing/wiring fault was found. B has24
  completed u6 episodes; R has0. **Both durable checkpoints remainu5; u6 PPO was
  not applied.** The pause and all receipts/charges are preserved. Training remains
  paused under the registered rule pending MAIN's scientific decision.
- Pause SHA256 `9e65c7f83ac0bf7961a826fe072be82aa81056d9f25b1a1b93392c03399da238`.
  Evidence:`coopmwm/SAFETY_PAUSE_PROBE_1793672206176824700.json`.
- MAIN09:36 exposure correction is CPU-qualified: **720 Linux checks PASS,
  zero skips,84.077s,zero SUMO**. It adds at least5 full post-PPO AWR passes,
  frozen-pre-update0.045 KL stopping, full member coverage and before/after error.
  The centre transport-identity repair is qualified too. Neither is deployed while
  the health pause is active. The draft u7 activation assumption must be reconciled
  with the now-confirmed unapplied u6 boundary before any permitted restart.
- The separate u4 diagnostic correction preserved2 valid branches. Six new
  branches were blocked before physics by an old charge-kind collision; a fresh
  registered correction namespace with a dedicated regression is being qualified.
  The partial2-entry aggregate is not the complete8-entry learning result.
- Total charged5,806,820/14,500,000 at10:16:44, including retained failures.
  No sealed outcome was opened. MAIN report:`_sol/COOPMWM_SAFETY_PAUSE_U6_1017.md`.

## Nov3 09:37:08 UTC+8 — qualified continuation from both u5 checkpoints
- B/R-u5 completed. At a verified idle boundary, both checkpoint5 hashes and
  no-u6-job/no-physics-child conditions passed. Registered amendment
  `de152a1b76837b9b…`/commit`2ab155f5` preceded the handover. Policy/Adam/RNG
  resume exactly from u5; future first update6 and fixed final24 remain.
- New durable watcher**3088220/start68410564**, B**3088230**, R**3088231**,
  R grader**3088228**; B grader**2979630** was adopted. All alive/no child failure.
  Original30-minute pacing is preserved using u5 claim timestamps. The saved-R
  collector3066675 and GPU/LS services were not interrupted.
- New `B_REPLAY_CONTINUED_LESSONWINDOW.json` preserves the original start receipt.
  The qualified source applies independent per-arm admission and original common
  windows/new primary delta metrics to future lesson checks.
- The separate u4 correction exited before any new branch with a missing `copy`
  import in its valid-branch adoption path. Its error/failed supplement are retained;
  the small reporting-only fault is being repaired with an adoption regression.
  No corrected-u4 learning result is claimed, and the4260 sim-s correction upper
  bound has not been spent. This path is not used by training's future lesson calls.
- Evidence: `coopmwm/B_REPLAY_START_WATCH_STATUS_1793669838696991200.json`,
  `coopmwm/B_LESSON_CORRECTION_STATUS_1793670076564752700.json`.

## Nov3 09:28:03 UTC+8 — R collector live; MAIN09:06 recovery qualified
- R-u4 had no bank because B filled the shared eight-pending admission cap.
  Qualified repair gives B and R independent equal bounded quotas; R keeps the
  registered imitation cap. Existing LS B receipts supply count/K matching metadata
  only; no B actions, search scores or library are supplied to R.
- Durable saved-state collector **3066675/start68356119**, registration SHA256
  `3fbaeb310468c897439a7f180e751a29b538c970a90809cb050b9062de8eea8d`.
  Immutable `replaylesson_cpu3`: **713 Linux CPU checks PASS, zero skips,
  83.796s, zero SUMO**. Real R986045/t360 preflight regenerates every saved own
  action and torch RNG bitwise before publishing a snapshot.
- At09:29:09: **8 recovered R snapshots**,0 centre answers,0 verified decisions,
  0 admitted R entries. These are honestly tagged policyu3, recovered from saved
  R-u4 observable tapes; no episode rerun or extra episode charge. The existing
  watcher has bridged16 total B/R snapshots. Same centre-only +8 random candidates,
  exact verification/replication/safety/synergy; branches retain existing phase caps.
- Source2's real preflight caught a compressed RESULT-marker reader mismatch before
  dispatch. Source3 uses the lossless reader and adds its regression; all earlier
  qualifications/failures are retained. No inference/physics attempt was repeated.
- B-u5 began on the registered30-minute pacing. Its24 episodes have finished;
  R-u5 waits for the completed B-u5 receipt. An attempted u4 boundary freeze
  refused before any signals because u5 had already started. The qualified trainer
  handover will use the next fully idle boundary, exact checkpoint5/Adam/RNG.
- MAIN09:06 lesson reporting is qualified: primary paired absolute saving in veh-s
  with bootstrapCI and positive share; ratio median/CI only when its denominator
  exceeds per-entry replication spread. The common-window correction is also
  qualified; original u4/u5 raw ratios remain excluded from learning evidence.
- Evidence: `coopmwm/R_BANK_COLLECT_DISPATCH_0906.json`,
  `coopmwm/R_BANK_COLLECT_STATUS_1793669350716185400.json`.

## Nov3 09:02 UTC+8 — B-u4 completed; lesson-window diagnostic fault isolated
- B-u4's own-policy PPO and bank-imitation update completed;24 fresh episodes
  finished. The update receipt reports324.571s wall time. R-u4 has24 RESULT files;
  supervisors and graders remain alive with no health STOP.
- Read-only probe found **6/8 B-u4 lesson branches use shorter windows than their
  original bank reference**. `lesson_check` rebuilt a policy branch from only the
  selected candidate, shrinking the original slate's common-max-duration window.
  Its very large recapture ratios are therefore **invalid learning evidence**.
  Original lesson receipts/charges remain retained. This affects the diagnostic,
  not PPO rewards, bank admission or AWR training targets. A registered engineering
  correction is being prepared for the next idle update boundary.
- Evidence: `coopmwm/LESSON_WINDOW_PROBE_1793667760930470800.json`.
- Read-only `warm_watch state` now prefers actual replay arm/grader owners over
  closed historical runs, retaining PID boot/start validation and never hiding an
  exited replay child.14 CPU checks PASS (0.263s); live cpuQ payload PASS. Registered
  observation amendment `e726efe260fb4063…`; no runtime/physics/PID change.

## Nov3 08:52:44 UTC+8 — B_REPLAY_STARTED; MAIN08:35 decoupling live
- Append-only V4 amendment SHA256
  `b967137614d30c9ab358f77670198e1784496deadee2c38a9b6eb76666bbd8bc`
  was registered and committed (`54be3ba4`) before launch. Immutable
  `replaystartcpu5`: **707 Linux CPU checks PASS, zero skips,83.264s, zero SUMO**.
- Start requires **8 admitted entries + qualified replay release + reconciled
  budget + health OK**. GPU niche/nullfix rollout and collector activation proceed
  independently. The actual gate receipt passes all four requirements.
- Remote `B_REPLAY_STARTED.json` was written at **unix1793667164.8449004**.
  Durable watcher **2979442/start68143223**; arm owners **B2979632/R2979633**;
  graders **2979630/2979631**. Boot/start/argv verified alive, no child failures.
  Only the verified idle previous watcher was retired; the original bank builder
  exited gracefully before confirmation ownership moved to the new graders.
- At08:54:40, B-u4 has24 fresh `replay0735` episode claims,12 RESULT files and
  12 ongoing episodes observed at t721–978/1200s. R's supervisor waits under the
  registered lag-1 rule. Both optimizer endpoints remainu3 at this observation.
- Collection continues on existing services; qualified it2_K2 requests remain in
  use until independent joint collector activation enables niches1/4 alternation.
  Frozen-u3 admissions hand over later at a no-active-grade boundary. Existing
  sequential rollout/activation/observer owners10148/39660/40668 remain alive.
- Bank at handover:75 usable decisions,36 confirmations,**8 admitted/3 cooperative**.
  Charged5,633,000 sim-s at08:53:18, including retained superseded/failure charges;
  approved cap14,500,000. CPU disk16.26GB. No policy-learning result yet.
- Evidence: `coopmwm/B_REPLAY_START_WATCH_DISPATCH_0835_V4.json` and
  `coopmwm/B_REPLAY_START_WATCH_STATUS_1793667281338361200.json`.
  `replay_start_ops status` now reports the new child identities and fresh TRAIN
  episode progress; historical `warm_watch` arm-owner precedence predates replay.

## Nov3 08:34 UTC+8 — rolling-controller stale-owner fault fixed
- V5 controller40408 exited after starting gpuR qualifier766170: it attempted
  admission using the retired worker's cached pre-handover READY status. Its
  log and original dispatch remain retained. The GPU qualifier continued running.
- Engineering commit `5295b6ed` refreshes both host inventories after a handover
  before endpoint admission. Live RPC identity checks remain mandatory.
  **13 targeted CPU checks PASS (0.36s)**, including the retired-owner race.
- Registered amendment SHA256
  `818e14acaf073715d1603518c665da71db4a453b430567af1bdf8c54515c175c`;
  V6 controller **10148/create1793665996.30793** is verified alive with expected
  argv and has adopted the existing qualifier. Activation39660 and observer40668
  are also verified alive. No gate or search was repeated.
- gpuR766170 now has **2/8 gates**, aliveR/no failure; its peer623287 is READY8
  and serving. Bank:73 usable decisions/35 confirmed/**8 admitted /3 cooperative**.
  B/R remainu3, no STOP; charged5,582,380 sim-s, cpuQ free16.46GB.

## Nov3 08:30:25 UTC+8 — MAIN threshold amendment live; bank gate satisfied
- MAIN08:16/08:25 lowers the automatic-start requirement from10 to **8 admitted B
  entries**. Append-only V3 binding SHA256
  `26535926cb96ab7fc828901097cba958b8de19decfe39825545c9d886d8c43b0`
  was registered and committed (`37723cb8`) before watcher handover.
- Immutable `replaystartcpu4`: **706 Linux CPU checks PASS, zero skips,
  85.010s, zero SUMO**. Only the verified idle V2 watcher was retired; original
  bindings, source and receipts remain on disk.
- New durable cpuQ owner **2926173/start68009586** is aliveS. Actual status sees
  **8 admitted /3 cooperative /threshold8**. Remaining gates: four joint-release
  GPU slots admitted, then qualified collector activation at a no-active-grade
  boundary. The existing sequential rollout and activation owners continue.
- B/R remainu3; no `B_REPLAY_STARTED.json` yet. Evidence:
  `coopmwm/B_REPLAY_START_WATCH_DISPATCH_0816_V3.json` and
  `coopmwm/B_REPLAY_START_WATCH_STATUS_1793665826404307800.json`.

## Nov3 08:12:32 UTC+8 — fresh-session adoption; sequential release advances
- Adopted the existing remote start watcher **2855195/start67831802**, bank
  builder **2388293/start66646196**, rollout40408, activation39660 and observer40668.
  Boot/start or Windows creation identity plus argv checked; no replacement dispatch.
- gpuA's second slot261418 completed all eight gates. The existing controller
  has moved to draining gpuR's first slot; both original gpuR workers remain
  READY and serving. Completed qualifications and outstanding searches are retained.
- Bank at unix1793664752: **69 usable decisions /34 confirmed /8 admitted /
  3 cooperative**, zero pending confirmations. The remote watcher correctly waits
  for ten admissions, four joint-release slots and collector activation. B/R remainu3;
  no `B_REPLAY_STARTED.json` and no new learning/viability claim.
- Latest MAIN06:27/07:06/07:35 authority and the14.5M append-only budget are adopted.
  Warm-memory qualification and the706-check Linux release are completed prior work.
  This resume adds no reservation, SUMO, optimizer application or process signal.
- Evidence: `coopmwm/RESUME_20261103_0808.md`,
  `coopmwm/B_REPLAY_START_WATCH_STATUS_1793664659936188300.json`,
  `coopmwm/BANK_BUILDER_STATUS_1793664754706524400.json`.

## Nov3 08:00:47 UTC+8 — corrected automatic watcher V2 live
- **2855195/start67831802** is aliveS on cpuQ, control
  `coopmwm20261102_warm1/replay_start_0735_v2`; adopt its dispatch receipt.
  Only the idle V1 watcher was identity/argv/no-children checked and retired.
- V2 amendment SHA256
  `929b1ed89b947fd59387a594b2dc721a89825c4fc5a9d7d514cee17f79ab8e18`
  is registered and committed before dispatch. Original bindings/receipts retained.
- Real-u3 read-only runtime preflight passes for both checkpoint hashes, restored
  Adam/torch generators (5056-byte state), warm history and plant bounds. The B
  wrapper preserves fresh job IDs and same-episode nonblocking snapshot capture.
  Zero optimizer applications, children, reservations, CUDA or SUMO.
- First V2 status: **8 admitted /3 cooperative**, waiting ten admissions,
  four joint-release GPU slots and collector activation; no training-start claim.
  Latest bank probe has67 usable decisions/34 confirmed/zero pending. Existing
  builder, GPU workers, sequential rollout and observer remain active.

## Nov3 08:00 UTC+8 — current-policy R centre propagation corrected before activation
- Pre-activation inspection found the inherited bank collector forwarded actor
  weights only for B; R's centre-only request consequently lacked its current R
  policy. Both now carry their own actor state. R still receives no B weights,
  learned covariance/library, or WM candidate scores. A dedicated regression
  checks R's policy identity and centre-only request contract.
- New immutable `replaystartcpu3`: **706 Linux CPU checks PASS, zero skips,
  83.720s, zero SUMO**. No GPU numerical code or serving worker was modified.
- A new additive V2 binding and idle-watcher handover are prepared; original
  registration, source, receipts and V1 dispatch remain retained. B/R remainu3;
  no training outcome was used for this engineering correction.

## Nov3 07:54 UTC+8 — AUTOMATIC B-REPLAY START WATCHER LIVE
- Registered locally/canonically and on cpuQ before dispatch: amendment SHA256
  `2b41922c9c5c6d95c20b34d0125860d324987da8507f962bd690b90437b563fa`.
- Remote durable owner **2838584/start67791704**, boot identity checked;
  `coopmwm20261102_warm1/replay_start_0735`. Detached session, SIGHUP ignored,
  nice19, singleton lock,120s polling. Its first status at unix1793663647 is aliveS.
- Gate status: **8/10 admitted B entries,3 cooperative**; qualified collector
  activation and remaining joint-release GPU slots pending. Existing rollout
  controller40408, activation waiter39660 and observer40668 are verified alive
  and retained. This watcher does not duplicate their jobs/gates.
- B/R stillu3; no `B_REPLAY_STARTED.json` yet. The remote watcher owns the future
  handover, update4 start, continued updates through24 and registered fixed-u8
  hash-sealed episode precompute. No extra reservation or SUMO at dispatch.

## Nov3 07:53 UTC+8 — durable automatic B-REPLAY start qualified
- New isolated `coopmwm20261103_replaystartcpu2`: **705 Linux CPU checks PASS,
  zero failures/skips,82.780s, zero SUMO**. The first snapshot's missing-parent
  bridge failure is retained; corrected shared-queue publication creates its
  metadata parents. Existing GPU numerical files/gates are untouched.
- The watcher polls every120s and requires ten admitted B entries, all four
  joint-release GPU owners/eight exact gates, and completed collector activation.
  It uses fresh `replay0735` episode IDs and restores policy/Adam/RNG from u3.
- Bank snapshots are captured nonblocking during the same24 own-policy PPO
  episodes; additional collection episode charge is0. Shared LS broker retained;
  R uses centre-only inference and random candidates, never WM candidate scoring.
  Frozen-u3 admissions hand over at a no-active-grade boundary; saved labels drain.
- Confirmation ownership transfers gracefully; whole-decision precharges and
  no-replay claims remain enforced. Lesson checks run every update. Fixed-u8
  interim quote is at most345,600 sim-s in its restricted900,000 phase, hash-only
  while training continues. A/B/R final checkpoints and existing final owner retained.
- Current live preflight: B/R remainu3, no health STOP; R's old u3 pause certificate
  is preserved by SHA-checked relocation at the eventual launch boundary. Arrival
  phase end1200s comes from the existing model-validity binding. Registration and
  remote dispatch follow; **no training-start claim** at qualification.

## Nov3 07:29 UTC+8 — qualified latest budget semantics; rolling handover continues
- Latest frozen replay source `coopmwm20261103_replayreleasecpu6` passes
  **696 Linux CPU checks, zero failures/skips,82.961s, zero SUMO**. It loads
  the authorized07:06 transfer before older budgets and suppresses the superseded
  B/R every4-update checks; registered per-update lesson checks remain.
- The next GPU-slot handover exposed per-file symlinks into the first slot's
  package. V4 retained its failure after retiring only the already-drained
  second owner. New atomic destination-entry installation preserves old bytes,
  link metadata and hashes, and cannot write through an alias into a serving
  peer. Only original-binding or exact incoming CPU7-qualified bytes are accepted.
- Amendment `1182388c8925820a…`: controller **40408/create1793661876.0802872**
  adopts completed drains and gates. Second gpuA qualifier **261418/start904211550**
  is alive with no failure; first slot237704 remains READY8 and has completed a
  genuine clamp-enabled search. All numerical search sources still come from
  immutable CPU7; remaining slots require all8 fresh gates, one qualifier globally.
- Nineteen targeted metadata/budget/install checks PASS (0.39s). New local
  activation waiter **39660/create1793662027.9718082** replaces only verified32220
  and explicitly carries the approved14.5M budget into the future collector
  configuration. It still waits for four READY slots and a no-active-grade boundary.
- At unix1793662173, bank builder2388293 is alive: **65 usable decisions,
  33 confirmed positives,8 admitted /3 cooperative**, zero pending. Mean admitted
  replicated advantage533.302 veh-s, decision-bootstrap95% CI[54.101,1373.853];
  these are conditional bank descriptives. B/R remainu3; no policy-learning or
  training-start claim. Existing collection, confirmations and observer continue.

## Nov3 07:19 UTC+8 — MAIN07:06 train-phase transfer registered
- Latest addendum authorizes144,000 sim-s **checks→train**. All of it fits the
  unspent checks balance; sumo_verification contribution is0. Total remains14.5M.
- New additive `B_REPLAY_PHASE_TRANSFER_0706.json` is registered canonically and
  on cpuQ, SHA256 `b6c1b8bf12d228e5c923efc5d02e0803f0870a8c123a479d0fadd07bea4b5c28`.
  Train allocation3,008,400 /charged1,798,800 /remaining**1,209,600**, exactly
  covering B/R u4-u24 PPO. Checks allocation288,000; all existing charges retained.
- Under B-REPLAY, lesson checks replace the old B/R learning checks and use
  bank_verification_ext; interim uses interim_u8_test. Restricted allocations and
  original bindings are retained. Six budget-contract tests PASS (0.12s).
- No reservation/physics/optimizer in this amendment. Full-run PPO funding is
  resolved; bank start threshold remains8/10 at the last validated read.

## Nov3 07:14 UTC+8 — full Linux B-REPLAY qualification and real bank-read PASS
- Immutable `coopmwm20261103_replayreleasecpu5`: **694 CPU checks PASS, zero
  failures/skips,83.664s, zero SUMO** (642 tests plus52 subtests; the already
  receipted seven-scenario integration rehearsals are adopted, not repeated).
- Earlier cpu2 pre-test indentation failure and cpu3/cpu4 failed suites remain
  on disk. Fixes preserve historical acceleration/resume quotes; the later bank
  budget is loaded through authorized additive bindings. Tests no longer leak
  the recovery runtime into later tests, and operator compile tests isolate their
  OS/git fixtures. cpuQ's older WM library is supplemented only in this isolated
  qualification snapshot with SHA-bound Python source from pinned FULL-v4.
- Two real integration faults fixed before deployment: the trainer must read the
  SHA-linked amended safety confirmations, and recovered LS labels must resolve
  branch tapes under their original `label_attempts/<generation>/branches` root.
  The original labels/grades/confirmations remain unchanged; no branch rerun.
- Real TRAIN-only read check, unix1793661278:63 usable decisions,32 first-seed
  positives, **8 admitted /3 cooperative**, **990 complete imitation samples**.
  Every admitted tape passes its sealed result digest and committed-step checks.
  Cooperative sample share54.55%; no duplication needed for this bank.
- Read check uses diagnostic unit weights only; it writes no temperature,
  applies no optimizer and reserves/runs no SUMO. B/R remainu3; ten admitted
  entries are still required. Phase shortfall144,000 remains reported to MAIN.

## Nov3 07:12:10 UTC+8 — frozen rollout source retained during replay integration
- V3 correctly refused the newly merged local `sumo_verify.py` hash before
  altering the drained second GPU slot. Completed first-slot gates and all
  serving workers/requests are preserved; no inference or qualification replay.
- The deployment loader now reads numerical files/network from the original
  immutable CPU7 snapshot and verifies its manifest plus the registered hashes.
  Evolving replay-integration files cannot enter the separate rolling release.
- Seventeen local metadata/registration checks PASS (0.27s). Amendment
  `1fb0b448cfd3d008…` precedes replacement controller **24476/create1793661130.675497**.
  It continues the existing drains one slot at a time; all untouched slots still
  require all8 fresh gates. Existing activation waiter/observer retained.

## Nov3 07:01:51 UTC+8 — first slot READY; metadata collision recovered
- First joint-release worker237704 passed all8 checks and became READY at
  unix1793659944.3873818. Its completed checks are adopted; no repeated gate.
- Rollout V2 exited after successful admission because its local receipt name
  collided with the observer's same-second admission receipt. GPU/search and
  collection owners continued. Original log and receipt remain intact.
- Serialized remote endpoint admissions plus per-writer publishing names and
  nanosecond local receipt names remove the collision. Nine targeted CPU tests
  PASS (0.35s), including two same-second admissions with both receipts retained.
- Engineering amendment `0a62515b26ffe14d…` registers replacement controller
  **10452/create1793660511.9570189**. It adopts the qualified first slot and
  continues only the registered sequential drained-boundary rollout. Existing
  activation waiter and observer are retained; zero SUMO/PPO dispatch.

## Nov3 06:56 UTC+8 — latest cap reconciled; exact phase shortage reported
- MAIN06:27 supersedes the stale fresh6M guard with **14,500,000 sim-s**.
  Append-only phase reconciliation is registered locally/canonically and on cpuQ,
  SHA256 `b8d8f5bd0696da31ea15f04e4ca9e27a327712c667282347e93fc9fed56c68fc`.
- Locked ledger snapshot: **5,430,440 charged /9,069,560 remaining** in total,
  including all retained superseded/failure charges. The full per-phase balances
  are in `coopmwm/B_REPLAY_PHASE_RECONCILIATION_0627.json`.
- Train has1,065,600 remaining; B/R u4-u24 PPO requires1,209,600. The **144,000
  sim-s train-phase shortfall** is reported to MAIN before any transfer or restart.
  Restricted verification/interim/final allocations remain protected. No new
  reservation or physics occurs in reconciliation; existing live immutable
  bindings await their qualified handover.
- Delegated READY atd51051a9 is integrated by3cc4d0ab. Its seven completed
  rehearsals are adopted; full Linux unit qualification is running separately.
  Bank remains7 admitted/3 cooperative at the last completed probe; B/R stayu3.

## Nov3 06:40 UTC+8 — durable checkpoint while remaining gates run
- Worker237704 is alive with5 check receipts (4 SHA-adopted,1 fresh), no failure;
  gpuA100%,29,203MiB. Existing controller12200, activation waiter32220 and
  observer40668 have verified creation identities and remain alive.
- Collection has58 completed searches/58 usable lists. Bank:28 confirmed,
  **7 admitted /3 cooperative**, zero pending; builder continues on every new
  positive label. Latest baseline exact-delay null rate15/58 (25.86%); post-release
  niche cohorts remain unobserved. Source-bound GPU/collector activation is pending.
- Full delegated replay READY remains absent, with no duplicate qualification
  dispatched. B/R stayu3; the explicit fresh-cap restart budget gate remains
  unresolved. Latest checkpoint and receipts are committed for safe continuation.

## Nov3 06:36 UTC+8 — late dependency failure recovered without repeating exact checks
- Worker220668 completed all4 committed-background/warm-single-batch checks with
  exact-zero errors, then exited on the lazy `resource_config` service import.
  Controller V1 stopped and retained the fault; other3 serving slots continued.
- CPU7: **273 Linux PASS, zero skips,53.814s, zero SUMO**. Complete dependency
  package now also includes the same qualified `resource_config` and
  `query_errors`; actual GPU-environment smoke executes their lazy imports and
  resource-clock loader before dispatch, with CUDA uninitialized.
- Amendment `9796569a299a83cb…` adopts the4 completed checks only after exact
  receipt SHA and complete unchanged numerical-manifest verification. New worker
  **237704/start903906207** runs4 fresh cold/warm full-fast checks; all8 checks
  remain required before admission. Untouched slots still run all8 fresh checks.
  Original failed logs, gate results and bindings remain intact.
- At unix1793658969: new worker alive/4 adopted checks/no failure, gpuA99%,
  27,429MiB. V2 rollout controller12200 and activation waiter32220 are live;
  observer40668 publishes quality/variation evidence. Seventeen local metadata
  tests PASS (0.30s), including changed-source/receipt/nonzero-error rejection.
- Intermediate clamp-enabled searches without a recorded niche configuration
  have their own reporting cohort; they cannot contaminate the pre-release
  baseline or the registered niches1-vs4 comparison. Collector gate and B/R PPO
  restart are still pending; the fresh-cap budget blocker remains published.

## Nov3 06:22 UTC+8 — fresh-cap restart budget gate
- Read-only append-only ledger: **5,391,140 sim-s charged**, including retained
  superseded work; **608,860** remain under the fresh resume's6M guard.
- B/R u3→u24 require42 updates ×24 episodes ×1200s = **1,209,600 sim-s for PPO
  episodes alone**. Minimum shortfall600,740 excludes further bank verification,
  confirmation/synergy, lesson checks and any unreserved learning checks.
- `B_REPLAY_BUDGET_GATE_20261103_0622.json` preserves the quote, with zero new
  reservation/physics. MAIN alert is published in `_sol`. Historical14.5M brief
  authority remains recorded; it has not silently overridden the fresh6M guard.
  Training restart requires authority/phase reconciliation as well as the existing
  ten-admission and delegated release qualifications. Guarded registered collection
  and confirmations continue; no B/R PPO update is applied by this quote.

## Nov3 06:10 UTC+8 — supervised release completion and replay readiness guard
- Sequential rollout controller37004 adopts the live qualifier and will drain,
  qualify and admit each remaining slot only after its predecessor is READY.
  First new K4/H300 committed-background gate passes with exact0 cost error;
  the K8/H420 gate is running. Other3 slots continue serving; no gate replay.
- Activation waiter16784 is separately locked and waits for all4 fresh-gate
  workers. The prepared collector handover freezes only its receipt-verified
  metadata parent, checks for active grade children/incomplete verification
  claims, and resumes it unchanged if either exists. New configuration preserves
  all original episodes/claims/precharges and the6M operational guard.
- Observer14132 replaces only verified7664; it republishes amended
  `BANK_QUALITY_B.json` every30min and null/diversity cohorts hourly. All parent
  bindings/failed generations are retained; collector activation is not claimed.
- Full delegated replay READY is still absent; its existing worker35968 was
  verified alive, so no duplicate integration/rehearsal was launched. A latent
  indentation fault in the guarded Linux-qualification payload was fixed with
  a compile-path regression:3 CPU tests PASS (0.22s), no remote dispatch/SUMO.

## Nov3 06:03 UTC+8 — bank refresh and explicit null/diversity cohorts
- Bank owner2388293 remains alive: **25 confirmed /6 admitted /3 cooperative**,
  zero pending/incomplete. Old-rule disclosure:4 admitted/12 safety rejected.
  Mean admitted replicated advantage711.063 veh-s, decision-bootstrap95% CI
  [72.149,1806.164]; conditional bank descriptives, not policy-learning evidence.
- Saved-label-only report `NULL_DIVERSITY_BY_CONFIG.json`: before joint release,
  **14/53 complete eight-candidate lists are exact delay-null (26.42%)**. Neither
  niches1 nor niches4 has post-release labels yet, so their null/diversity values
  are explicitly unobserved. Historical labels lack diversity measurements.
- Partial/slate-skipped decisions are counted separately and cannot dilute the
  complete-eight null denominator. No trajectory-equivalence claim follows from
  exact delay equality. Command mirror remains OFF after its retained failed gate.
- Twelve targeted reporting/ownership/rollout tests PASS (0.53s). New observer
 7664 replaces only creation/argv-verified8472 and publishes the per-config report
  hourly. Existing bank, collection and GPU owners are retained. B/R remainu3.

## Nov3 05:49:23 UTC+8 — complete dependency smoke and fresh worker generation
- PATH2 also exited before gates: `ls_fast_worker` had existed only in the old
  overlay and was missing from the new package. Both failed owners/logs remain.
- Frozen CPU6 full release suite: **271 Linux PASS, zero skips,54.868s, zero
  SUMO**. Package now includes the same qualified it2_K2 wrapper and geometry
  dependencies; a complete import/install smoke in the actual GPU environment
  passes before dispatch, with CUDA still uninitialized.
- Amendment `7c10945a23493424…` precedes new qualifier **220668/start903637810**.
  A metadata generation parameter writes new owner/gate/failure names and a new
  effective binding with actual hashes. All8 fresh gates are still required;
  one qualifier globally, other3 slots continue serving.
- Read-only collection probe:53 successful searches/52 usable lists, existing
  broker2472252 and supervisor1834956 live; no PPO update. Delegated full replay
  READY remains absent; bank start prerequisite remains5 of10 admissions.

## Nov3 05:45:44 UTC+8 — first qualifier dependency packaging recovered
- Original qualifier213896 exited before any gate: the WM-only GPU package has
  no `traffic_shepherd` package. Adding frozen WM source paths alone did not
  resolve the dependency; that CPU-only probe failed before dispatch.
- Added only two package initializers and the already pinned pure geometry
  source, verified against an immutable dependency manifest before import.
  Actual GPU-environment import/geometry probe PASS with CUDA uninitialized;
  five targeted local metadata/launch tests PASS (1.03s), zero SUMO.
- Amendment `7f9db7082967b2f0…` preceded replacement219214/start903615905.
  All8 fresh gates remain required; original failed dispatch/log and bindings
  are retained. Other3 slots serve; no inference replay or PPO restart.
- Bank builder remains live: all24 positive labels confirmed, **5 admitted /
  3 cooperative**, zero pending/incomplete; original safety rule would admit3.
  Ten confirmed admissions remain required for B-REPLAY.

## Nov3 05:32:02 UTC+8 — first joint-release GPU qualification started
- New source-qualified metadata broker2472252 adopts the original requests and
  assignments. Only its old metadata owner1941060 was retired; no inference or
  SUMO job was restarted. Per-owner drains preserve historical drain receipts.
- gpuA incident1 finished/delivered its outstanding search before110917 retired.
  **New owned qualifier213896/start903533639** is running all8 fresh startup
  gates. Original numerical files were backed up with exact hashes; the original
  binding is retained and the new effective binding records current hashes.
  `LS_NICHENULL_DISPATCH_gpuA_incident1.json` is durable; transport timed out
  after printing its completed dispatch, so the existing owner is adopted.
- Other3 slots and the amended bank builder continue. No second qualifier or
  live clamp/collector activation is claimed before gate completion. Ten local
  metadata/ownership checks pass (4.04s); observer refresh is metadata-only.

## Nov3 05:16 UTC+8 — joint niche/nullfix deployment source qualified
- `nichenullcpu5`: **271 Linux CPU checks PASS, zero skips,54.795s, zero sim-s**.
  Includes the landed niche+nullfix search, pinned road-network fallback, and the
  LS adapter that applies the actionable gate before loading a saved snapshot
  or running SUMO. Inactionable slates have an explicit non-grade receipt;
  partial slates record availableK while preserving the raw8 search candidates.
- The original CPU failures remain recorded: cross-BLAS static golden, a
  globally bound test-network fixture, and a missing labels-parent directory in
  the new skip-path test. Corrected checks retain exact same-runtime pre-change
  byte comparisons, without widening numerical tolerances.
- Amendment `f653a9fa55abf56c…` registers a one-slot-at-a-time rollout with
  all8 fresh source-bound GPU gates, per-owner append-only broker drains,
  hash-verified original-source backups and a new effective binding. gpuR batch
  width1 bounds memory; horizons/futures/population stay registered. Four local
  metadata/payload tests PASS (4.77s).
- This is qualification/registration only: the collector gate and search clamp
  are **not yet live**. Existing four-slot it2_K2 service and the amended bank
  builder remain active while the first GPU handover is prepared.

## Nov3 04:51:37 UTC+8 — amended safety revalidation and required solos complete
- Durable replacement **2388293/start66646196** started04:43:04 after the old
  builder's graceful exit. `BANK_BUILDER_DISPATCH_0426.json` pins its source.
  All original confirmation receipts remain intact; no replication/extended
  branch was repeated. Existing conditional precharges funded the missing solos.
- Current bank: **17 confirmed /5 admitted /3 cooperative**, zero incomplete,
  **0 pending safety revalidations**. Original-rule disclosure:17 confirmed,
  3 admitted,8 safety rejections. New-rule overlapping rejections: replication7,
  later harm3,safety5. All replication/later-harm passes have diagnostic solos,
  including safety-rejected entries; full rows are in `BANK_QUALITY_B.json`.
- Required986016 diagnostic: group saving **+522.037019**, best solo
  **+560.037316** (ramp.26), cooperation gain **−38.000297 veh-s**. It is now
  admitted under the authorized combined-safety rule, **not cooperative**.
  Combined event deltas are−17 (replication) and−89 (extended); collision and
  teleport deltas remain0. The original rejection remains disclosed.
- Mean replicated saving among5 admitted entries: **853.276 veh-s**,
  decision-bootstrap95% CI[86.605,2096.231]. Mean cooperation gain over those
  entries1070.153, CI[−15.205,2842.195]. Conditional bank descriptives only;
  ten-admission restart gate remains unmet and B/R stayu3.
- Collector nullfix/clamp **not yet live**. Combined niche/nullfix source has
  completed267 Linux CPU checks (zero skips,55.300s,zero SUMO); source-bound
  per-slot GPU qualification/deployment remains required. All4 existing it2_K2
  slots continue serving; mirror remains OFF.

## Nov3 04:41 UTC+8 — MAIN04:26 safety amendment qualified and registered
- New shared B/R bank rule: no collision/teleport increase, and no increase in
  hard-brake+trace TTC<1.5 event totals in either confirmation window. Old
  confirmations remain immutable; additive SHA-linked verdicts retain both rules.
- Frozen `bankcpu2`: **68 Linux CPU PASS, zero skips,23.486s, zero sim-s**.
  Amendment `6b39e1ff55df45bf…` is registered before revalidation.
- All16 current positive decisions already have their full conditional-solo
  charges: **0 new sim-s reservation** for revalidation/diagnostics. Charged
  snapshot5,197,130 remains below the6M operational guard. Existing replication,
  extended-window and completed-solo results will be reused; unused precharged
  solos run for every replication/later-harm pass, including986016.
- Old bank owner receives a graceful boundary STOP only. The new owner may
  start only after that process has exited; collection/GPU/PPO owners retained.
  The original first14 summary above remains disclosed under the old rule.

## Nov3 04:29:53 UTC+8 — FIRST14 BANK CONFIRMATIONS COMPLETE
- `BANK_BUILDER_STATUS_1793651394415117200.json` preserves the remote
  `FIRST_BATCH_COMPLETE.json`: all14 original positive decisions confirmed,
  **3 admitted /11 rejected**, zero incomplete confirmations. The builder2332488
  remains alive and is processing newly arriving positive labels.
- Mean replicated advantage among the3 admitted entries: **169.935 veh-s**,
  95% decision-bootstrap CI **[0.0101,509.667]**. Rejection counts overlap:
  replication5, later harm2, safety7. These are conditional bank-admission
  descriptives; they are not B-vs-A or policy-learning evidence.
- All3 admitted multi-member entries received solo twins. **2 cooperative** by
  the registered strict `group advantage > best solo advantage` rule:
  - TRAIN986048,2 members: group **+958.677725**, best solo **−0.000358**,
    cooperation gain **+958.678083 veh-s**; replication/extended/safety pass.
  - TRAIN986052,2 members: group+0.013823, best solo+0.013327,
    cooperation gain+0.000496 veh-s (near zero, retained without a new threshold).
  - TRAIN986019,4 members: group+0.252190, best solo+0.263592,
    cooperation gain−0.011402 veh-s; admitted but not cooperative.
- Mean cooperation gain over all3 admitted entries:319.556 veh-s,
  95% decision-bootstrap CI[−0.0114,958.6781]. `BANK_QUALITY_B.json` contains
  the complete synergy table. Initial whole-decision precharge87,840 sim-s is
  retained, including unused conditional solos; no refunds or search reruns.
- Ten admitted entries are still required for the registered B-REPLAY start.
  B/R remainu3; the full delegated release remains pending.

## Nov3 04:21 UTC+8 — first real bank confirmation complete
- Owner2332488 remains live. The first existing positive decision (TRAIN986016,
  t300,D60) completed replication and extended-window branches with prefix
  checks: **+343.46 veh-s** on the second seed and **+5380.33 veh-s** extended.
- It is **not admitted**: both comparisons record3 more hard-braking events than
  their own-policy branch. Collision/teleport deltas are0; trace TTC<1.5 counts
  decrease. The pre-registered no-safety-regression rule is applied unchanged.
  Conditional solo branches are not run for this rejected entry; all5760
  precharged sim-s remain charged.
- `BANK_QUALITY_B.json` currently reports1 confirmed/0 admitted/13 pending.
  This proves the real confirmation path is working; no cooperation/viability
  claim follows from it. The remaining original batch continues under the same
  durable owner.

## Nov3 04:19:56 UTC+8 — offline bank builder LIVE
- Remote durable owner **2332488/start66507374**, control directory
  `coopmwm20261102_lscollect1/bank_builder_0406`, adopted by
  `BANK_BUILDER_DISPATCH_0406.json`. First status verifies the owner alive with
  branch startup in its log; admission and synergy results are not yet complete.
- Existing rolling controller finished the authorized configuration handover:
  **all four it2_K2 workers READY at04:10:38** (unix1793650238). The completed
  source/parity receipts are retained in `IT2_K2_FOUR_WORKERS_READY.json`.
  No repeated GPU qualification or training restart for this publication.

## Nov3 04:19 UTC+8 — offline bank confirmation qualified and registered
- MAIN04:06 authorizes confirmation now without the full B-REPLAY release.
  Frozen `coopmwm20261103_bankcpu1` pins delegated revision4f090b76 and its
  confirmation/synergy path: **59 Linux CPU PASS, zero skips,22.650s, zero SUMO**.
- Fourteen existing positive B/u3 TRAIN decisions quote **87,840 sim-s upper
  bound**, including replication, extended-window and all conditional member-solo
  pairs. Charged snapshot5,055,590 plus that bound is **5,143,430 <6,000,000**.
  Amendment `685e691af95df8c3…` is registered before physics.
- The isolated adapter reserves each complete decision atomically in
  `bank_verification_ext`; unneeded conditional-solo charges remain charged.
  Confirmation/synergy decisions use the delegated code, with a checked
  precharge hook. Existing collection/GPU workers and PPO checkpoints are retained.
  Dispatch and real admission evidence follow separately.

## Nov3 04:02:49 UTC+8 — first completed it2_K2 searches verified
- `coopmwm/IT2_K2_LIVE_TIMING_1793649769525748300.json` reads four completed,
  hash-pinned forecasts. Every receipt records **iterations2/screenK2/H300,
  full population/finalK8/H_final**, and all8 candidates were returned. Warm
  memory is retained; command mirror is OFF. This establishes live use of the
  authorized variant, not completion of the remaining worker rollout.
- Completed-query mean wall: gpuA/n2 **1019.7s** (n=2), gpuR/n2 **1014.3s**
  (n=1), gpuR/n6 **2910.0s** (n=1). No completed n4 query in this variant sample.
  Four completions over1.126h since the first completed-query start give
  **3.55 completions/h** across the rollout window, including gaps and unfinished
  work. These small, configuration-separated samples are not a steady-state
  throughput estimate or evidence of a2.47x live speedup.
- Read-only reporter has4 CPU tests PASS (0.06s), zero new WM searches/SUMO.
  It controls no processes and changes no numerical sources or live settings.

## Nov3 03:59 UTC+8 — fresh resume adopts live recovery
- Existing controller40716 and observer8472 are alive; no duplicate dispatch.
  gpuA's second successor151635 completed all8 gates at unix1793649461.
  The controller has already started the final gpuR successor623287; only that
  slot is qualifying while the other three remain READY.
- Read-only cpuQ check:39 successful searches/39 usable verified lists,
  7 terminal declines; broker1941060 and grading1834956 alive, disk20.1GB free.
- Delegated B-REPLAY replacement15000 is active; READY remains pending.
  B/R stayu3 until ten confirmed bank admissions and the qualified release.
  Original qualifications/recovered labels retained; zero new SUMO this step.

## Nov3 03:45 UTC+8 — durable rolling handover observer restored
- Registered metadata-only amendment `75405d44…`; new local controller40716
  adopts the existing active qualifier and completed first-tier gates. It writes
  `LS_FAST_ROLL_V5.log`; the original transport failure remains recorded.
- Added a tested scheduling guard: **at most one slot qualifies across both
  GPUs**. After the current gpuA successor is READY, the controller completes the
  remaining gpuR successor from its existing drain receipt. Six targeted local
  CPU tests pass (0.12 s). This controller launches no PPO or SUMO jobs.
- Existing observer8472 continues qualified admission/publication; collection and
  grading retain their owners. Combined niche/nullfix release and delegated
  B-REPLAY release remain pending; no duplicate delegated work was launched.

## Nov3 03:41 UTC+8 — all saved failed labels recovered
- `LS_SAVED_REGRADE_COMPLETE_0319.json`: **10/10 saved-search retries complete,
  90/90 branches graded**, all registered prefix checks passed. **62,640 actual
  sim-s** against **64,800 precharged**, with no refunds. All original failure
  receipts/charges retained; **zero new WM searches**.
- Recovery owner2225116 exited normally after writing COMPLETE. Existing collection
  and grading continue. Bank snapshot: **38 usable lists /13 selected positives /
  0 confirmed admissions**; B/R remainu3 pending the qualified confirmation release.
- Resumed the interrupted rolling configuration handover by adopting completed
  first-tier gates (gpuA110917, gpuR588039 both READY) and the existing second-tier
  drain. Only gpuA's second slot is currently qualifying (151635); gpuR's existing
  second worker stays intact until that gate completes, per one-slot-at-a-time rule.
- Niche/nullfix are awaiting their combined release and one joint requalification.
  The command mirror remains off after the recorded failed gate.

## Nov3 03:34 UTC+8 — real grading recovered; saved retries underway
- Frozen `lspathcpu1`: **229 Linux CPU PASS**, zero skips, **43.132 s**, zero
  sim-s. Three local selection/duplicate-prevention tests also pass.
- Amendment `06de06f4…` registers ten saved prephysics failures as new `SUMOPATH1`
  attempts. **64,800 sim-s** precharged in `sumo_verification`; total charged
  **5,037,230 < 6,000,000**. Old failure receipts and charges remain intact.
- New durable regrading-only owner **2225116/start66231393** processes lists
  serially, preserving all existing collection/broker/GPU owners.
- **First complete retry passed**: world986048, nine SUMO branches, **6,480 actual
  sim-s**, completed unix1793648075. The real policy-prefix smoke passed through
  t300 and was reused in that list (720 s, no extra smoke charge). No WM search
  was repeated. The remaining saved retries continue under the same owner.
- B/R remainu3. Bank selection positives still require confirmation/admission
  before the registered B-REPLAY restart; this is infrastructure recovery evidence.

## Nov3 03:28:32 UTC+8 — original scenario dependency restored
- Registered amendment `4fde4697…` before restoring the missing template opaquely
  from original revision `65cc714e`. Git blob matches exactly:
  `9bbbe41c79eb4df4903be80753001a1b960b4199`; SHA256
  `7b060d35fe1cf66ca5a01bfd7f2ad6be7f4172cba4154c0a5d67943c6eb1a65c`.
- Initial Windows archive failed the digest gate because of line-ending conversion;
  that artifact is preserved. The LF archive passed; no git configuration changed.
- No template contents were inspected, no process signals or restarts, zero new
  physics. Saved failed labels retain their receipts and charges; append-only
  regrading is the next step, with no new WM search.

## Nov3 03:24 UTC+8 — deleted grading dependency identified
- Adopted all work through `1a36cd4e` and the existing uncommitted changes.
- Full missing-path traceback identifies
  `/home/ubuntu/aq_code/code_r13e_65cc714e/scenarios/idm/driver_routes.rou.xml`.
  This is the scenario-template dependency in the deleted `aq_code` directory.
  The diagnostic refuses existing assignment-file contents and runs zero physics.
- SUMO binary/checkpoint are present; frozen runtime manifest has no missing
  allowed paths. Grading owner1834956 passes boot/start identity checks and remains
  live. At unix1793647454 there are27 usable lists and no inflight grading jobs.
- Next is an opaque git-archive restoration with digest verification, preserving
  all failed labels and charges, followed by append-only saved-search regrading.
  Search collection remains under the existing owners; B/R remainu3.

## Nov3 ~02:35 UTC+8 — qualification-memory recovery; completed checks retained
- gpuR576918 passed K2/H300 and committed K4/H300, then exhausted memory while
  re-capturing a full1600-slot H420 reference trace beside its live peer. The
  failed qualifier and completed gates are retained; existing inference was not
  retried. The active earlier gpuA qualifier is adopted without interruption.
- Amendment `0f6f5efa…`: use the inherited **SHA-verified adoption of four
  unchanged committed/warm-single-batch checks**, including K8/H420; each new
  successor still runs a **fresh K2/H300 check plus four fresh cold/warm
  full-vs-fast checks**. All source hashes and zero-error requirements remain.
  The search itself retains its horizons, K, population and acceleration plans.
- `lsfastcpu5`: **231 CPU PASS**, zero skips, **41.268 s**, zero sim-s.
  Controller33124 and observer8472 continue the registered rolling handover.
  This supersedes the earlier redundant-eight-fresh-check admission wording.
- Current bank27 lists /9 selected positives /0 confirmed admissions. The current
  delegated B-REPLAY worker6104 is active. Niche READY has landed at5df3e853;
  its code will be integrated and qualified before any niche activation.

## Nov3 02:10 UTC+8 — it2_K2 parameter-size gates PASS on both GPUs
- `coopmwm/IT2_K2_VARIANT_GPU_QUALIFIED.json` adopts both completed gates;
  no repeated qualification or SUMO. **K2/H300 full/fast trajectory and cost
  error0.0**, identical CRN draw IDs and candidate rankings; warm batch/single
  cost error0.0 on both hosts.
- Gate wall times: **272.589 s gpuA**, **264.914 s gpuR**. These are qualification
  timings, not plans/hour. Search horizons and final K8/H_final remain unchanged.
- Both replacement owners are alive; their eight fresh inherited startup gates
  are still running. The configuration is not yet serving searches. Controller
  17620 and observer34592 continue the registered handover; other slots/grading
  continue. Activation will be reported only after admission and a real receipt.

## Nov3 ~02:03 UTC+8 — registered it2_K2 GPU qualification in progress
- Frozen `lsfastcpu3`: **231 Linux CPU PASS**, zero skips, **40.538 s**, zero
  sim-s; local ownership/payload/config tests **14 PASS**. Registration V4 SHA
  `a3de7cf464bb5f23566c705e03230c54fa237e16d0a46766adae07ffe5f29b78`.
- Existing workers cannot reload search settings. The registered rolling
  handover drains/delivers the original search first, then verifies actual
  process exit and lock release. Failed startup attempts are retained; their
  recovery-flag and shutdown-lock races occurred before GPU qualification.
- New gpuA **110917** and gpuR **576918** are running qualification. They must
  pass exact **K2/H300** full/fast and warm batch/single checks plus all **eight
  fresh inherited gates**. The other two slots continue searching. Samples:
  gpuA86%, gpuR97%. No activation or throughput improvement is claimed yet.
- Detached rolling controller **17620** sequences the remaining two slots only
  after the first pair is READY; observer **34592** publishes and admits gated
  owners. Broker1941060 and grading1834956 continue. No SUMO job was launched
  for these qualification steps; fresh-message6M spending guard remains.
- Latest bank receipt: **25** real nine-branch lists, **8** selected positives,
  **0 confirmed / 0 admitted**. The ten-confirmed-entry restart gate is unmet.
  Delegated B-REPLAY and niche READY receipts are absent; B/R remainu3.

## Nov3 01:20 UTC+8 — latest priority adopted; mirror remains failed
- Four GPU slots and grading owner1834956 are live; disk13.9GB and unblocked.
  Read-only probes show21 usable nine-branch lists/five selected positives.
- The command-mirror gate is already complete and failed. CPU/CUDA traces are
  exact, but absolute member-motion errors are0.859924m/0.381060mps and
  0.078796m/0.055090mps; genuine half-step background observations are absent.
  `coopmwm/COMMAND_MIRROR_MAIN0116_VERDICT.md` preserves the exact failure.
  No rerun of unchanged work or activation is justified by those receipts.
- Latest restart prerequisite is10 **confirmed/admitted** positive B entries,
  superseding20. Five selected positives are not five confirmed entries. The
  delegated HARDEN_INT_READY receipt remains absent; B/R stayu3.
- Preparing the authorized it2_K2 configuration with unchanged H300 screen and
  K8/H_final. Current workers cache their settings, so a source-bound boundary
  handover and K2 batching qualification are required before the switch.

## Nov3 00:59 UTC+8 — first focus admissions verified
- The first three admissions after the handover are TRAIN worlds **986044,
  986045, 986046**, all focus-stratum states at t300. Existing queued requests
  continue unchanged. `FOCUS_ALLOCATION_LIVE_1793638755.json` records the check.
- Both GPU hosts retain two READY workers with eight completed gates each; fresh
  utilization samples are gpuA66% / gpuR95%. Collection has no failure receipt.
- Read-only accounting snapshot: **4,827,170 sim-s charged**, leaving **1,172,830**
  under the conservative fresh-message6M guard. The brief's larger cap is not
  silently activated. Final training/confirmation spending still needs its quote.
- B-REPLAY's delegated READY receipt is still absent. Existing collection/grading
  and observer V6 remain live; B/R stay at u3 until restart prerequisites pass.

## Nov3 00:47 UTC+8 — registered focus allocation deployed; collection continues
- Corrected an outcome-blind queue-ordering omission: the bank collector now repays
  its existing focus-admission deficit, then assigns ≥50% to the registered focus
  demand/share cells while available. Coverage remains new-world/time/group order
  within each stratum. No candidate state is deleted or excluded by this ordering.
- Frozen `coopmwm20261103_lsfocuscpu1`: **226 CPU PASS**, zero skips,
  **40.244 s**, zero sim-s. Amendment SHA
  `2ad71f6c1f88db2107ad381505a10218aedb4dbee264031ee9c5d11b8d002575`.
- At **00:47:01**, verified idle grading boundary and settled answers before
  replacing metadata supervisor1728233 with **1834956/start65229920**. All completed
  episodes, queued searches, labels, smoke and charges are adopted. Four GPU
  owners and the broker continue; no completed simulation or inference is replayed.
- Observer V6 **39872** replaces only identity-verified local V5; three ownership
  guard CPU tests pass. First live poll confirms **four READY slots**, both CPU
  owners alive, **15 real searches / 15 usable lists / 5 exact positives**.
- Focus ordering applies to the next free admission; the existing eight in-flight
  requests finish unchanged. B/R remain u3 pending the 20-positive prerequisite
  and delegated B-REPLAY release. Command mirror remains off after its failed gate.

## Nov3 00:41 UTC+8 — command-mirror gate adopted; activation remains blocked
- Rechecked source hashes against the completed 48-test CPU qualification and
  adopted the existing GPU receipt; zero new inference or SUMO calls.
- CPU/CUDA callback traces agree exactly, but absolute observable member-motion
  errors remain **0.859924 m / 0.381060 m/s** (986051) and
  **0.078796 m / 0.055090 m/s** (986070). Both old/prototype paths have the same
  errors and lack genuine half-step observations. This gate **failed**; the mirror
  remains off under the registered rule. Repeating unchanged code adds no evidence.
- `coopmwm/COMMAND_MIRROR_ADOPTED_1793637654.json` pins the original gate SHA
  and current numerical hashes. The inherited read-only status helper is preserved.
- Grading has caught up to **14 complete lists / 5 exact positives**; null-delay
  lists are **3/14** (all eight candidate delays exactly equal policy, no tolerance).
  `BANK_STATUS_1793637666.json` records these flags. These are bank-selection
  diagnostics, not a training-viability result; B/R remain u3.
- The delegated harden-int worker is still active; its READY receipt is absent.
  Its running CPU rehearsals are adopted rather than duplicated.

## Nov3 00:39 UTC+8 — disk blocker cleared; existing grading resumed
- Read all **578** brief lines; adopted `520fdf4b` and existing uncommitted work.
- cpuQ free space is **14,236,557,312 bytes**, above the requested 10GiB target.
  Approved archival sources were checked by existence only. No archive, deletion,
  replacement supervisor or duplicate dispatch was needed in this session.
- Existing grading owner **1728233** passes boot/start/argv verification and reports
  `disk_blocked=false`. **Six** complete real nine-branch labels are now present;
  three have strictly positive exact grades. Current grading threads are active.
- Evidence: `coopmwm/LS_LABEL_STATUS_1793637526.json`. Collection's cached counters
  lag the complete label files. **A48/B3/R3** remain unchanged; bank qualification
  and the delegated B-REPLAY release are still prerequisites for training.

## Nov3 00:32 UTC+8 — grading stall diagnosed as disk admission wait
- Adopted the existing supervisor **1728233**, verified boot/start/command line.
  Both grading futures are sleeping in the registered pre-claim disk wait;
  `grading_inflight=2` does **not** mean two SUMO jobs are running.
- Free space **2,105,589,760 bytes**, **41,893,888 bytes below** the existing
  2GiB admission requirement. The supervisor rechecks every10seconds and resumes
  automatically when space is available. Restarting would not resolve this blocker.
- **Five usable nine-branch lists / two positives** remain; A48/B3/R3 unchanged.
  Existing episodes, searches and labels are adopted. Storage expansion or explicit
  additional SHA-gated archive authority is required to resume grading.
- Read-only status now publishes identity, thread waits and disk-blocked state.
  Evidence: `coopmwm/LS_LABEL_STATUS_1793637150.json`. No new simulation or dispatch.

## Disk-safe grading wait deployed; four GPU slots qualified — Nov3 00:07 UTC+8
- Frozen `lslabelcpu3`: **224 CPU PASS**, zero skips, **40.944 s**, zero sim-s.
  Amendment `073040972ba2f0576cd4403a733141b5c7d26c44b530639789510e04c717ee97`.
- Metadata/grading supervisor **1728233/start64944056** adopts every completed
  label, smoke and claim. New work waits **before a verification claim** when
  the existing2GiB disk guard fails, then resumes automatically when space exists.
- Actual wait receipt: `LABEL_DISK_WAIT_SUMOENV1.json`, free **2,136,055,808 B**,
  required **2,147,483,648 B**. No new branch claim, physics, failed label or charge
  occurs while waiting. Storage expansion/additional archive authority is needed.
- **Five usable k8 lists**, **two exact-positive candidates** (+522.0370 and
  +0.25219veh-s). Three original missing-binary attempts remain void/no-outcome
  with charges retained. A sixth WM search is complete and queued for grading.
- All **four** GPU owners now pass eight fresh **EXACT0** gates and are admitted:
  gpuA7429/7430; gpuR500795/494955. Existing processes are adopted without another
  dispatch. Observer V5 **7152** continues receipt publication.
- **A48/B3/R3**. The qualified B-REPLAY release is not ready and the bank has
  **2/20** positive entries. No PPO restart or policy-improvement conclusion.

## Real SUMO labels flowing — 23:53 UTC+8
- **Real policy-branch smoke PASS** (world986020), including deterministic prefix
  through t300. All nine branches completed; that list was rejected and consumed
  **5,940 actual sim-s**, within its precharge.
- **Four usable k8 lists** now exist on four TRAIN worlds. The three earlier
  missing-binary labels were re-graded from saved candidates without WM replay;
  their original charges and void/no-outcome receipts remain intact.
- One list (world986016) has an exact-positive selected candidate,
  **+522.037 veh-s**; the other three are rejected/tied. **1/20** positive entries
  toward the prerequisite; this is selected bank evidence, not policy improvement.
- Read-only observer V3 **14156** publishes `BANK_STATUS.json` every30min and
  hourly `COLLECT_THROUGHPUT.json`. At23:48, the outage/startup-inclusive window
  gives **4.11 searches/h** and **4.11 verified lists/h**; no steady-state ETA.
- cpuQ has **2,155,585,536 bytes free**, only8,101,888 above its registered2GiB
  guard. Further storage requires expansion or explicit SHA-gated archive authority.
  No additional deletion. B/R remainu3; B-REPLAY release readiness is still absent.

## MAIN23:32 — SUMO label launcher recovered; real smoke pending
- Two genuine full-pool searches completed on gpuA in **1,131.52 s** and
  **839.24 s**. A third completed before the handover. Their original label
  attempts failed because the collector omitted the installed SUMO environment;
  all branches failed, with **zero actual sim-s**. These are void/no-outcome
  attempts; files and charges remain retained.
- Frozen `lslabelcpu2`: **223 CPU PASS**, zero skips, **40.235 s**, zero sim-s.
  Amendment SHA `161f2ba5c38e0ba786dda0d11973b1e0df7e020690c1dcda2f5d23431db0fb47`.
- **23:44:32 UTC+8:** only the verified metadata/grading supervisor was replaced
  by **1693668/start64854995**. Explicit `SUMO_HOME` and PATH resolve existing
  **SUMO 1.27.1**. Broker and GPU owners continue without restarts.
- Recovery uses saved snapshots/top-eight lists, never another WM search. Failed
  originals remain immutable; new `SUMOENV1` branches are precharged. A real
  policy branch must pass the recorded prefix before candidate branches launch;
  it is reused in the nine-branch list, adding no extra simulation.
- New reservations enforce the fresh-message **6,000,000 sim-s** limit,
  including retained superseded charges, conservatively pending clarification
  of the brief's later 14.5M authority. No refunds or reserve bypass.
- First real smoke/complete label still pending at unix1793634328. **A48/B3/R3**;
  no training restart and no viability claim.

## MAIN22:32 restricted budget amendment; qualification packaging follow-up
- **23:01:53 UTC+8:** gpuR secondary replacement **494955** passed all eight fresh
  **EXACT0** gates and was admitted as `gpuR@resource1651`. The durable broker
  can now serve collection independently of local SSH observation drops.
- Cap **14,500,000 sim-s**, amendment **8571ab38…**, registered and bound on cpuQ.
  The extra **2,600,000** belongs only to `bank_verification_ext` (B/R k8+policy
  bank grades and confirmation branches). Existing phase caps and live collection
  budget are unchanged; no extra allocation has been spent.
- `COLLECT_THROUGHPUT.json` reports successful searches, verified lists, exact
  positives and accounted sim-s per hour. Burn means **full precharges, including
  retained failures**, not an inferred actual execution rate. Startup GPU samples
  are labelled as such; utilization is not evidence of finished searches.
- gpuR secondary491843 completed four fresh gates, then failed because the legacy
  spool lacked the metadata-only `resource_config` import required by its overlay.
  That CPU-qualified dependency is now SHA-bound on both hosts (**4f8265da…**).
  Other three qualifiers continued; replacement **494955** runs fresh-generation
  gates. No interrupted inference, episode or physics job was replayed.
- Observer V2 **25264** recognizes this replacement and publishes hourly throughput.
  The prior observer alone was identity-verified and terminated. Transport-only
  banner/dispatch timeout handling passes **13 local CPU checks (1.50 s)**.
- Qualification remains in progress; the first genuine search and exact label
  are still pending. B/R remain u3.

## MAIN21:27/21:29 — arrival-phase recovery deployed
- All four recovery workers exited on the same fitted-arrival-bound error. The
  observable-arrival artifact ends at **1200 s**; an interrupted request at t860
  plus H420 exceeded that phase. Original failure receipts and claims are retained.
- Registered amendment **f89fe3bf…** derives admission from the model metadata:
  **t0 ≤ 780 s**, with unchanged **K4/H300 screen, K8/H420 maximum final**.
  No look-ahead reduction. Late snapshots remain stored; exclusions by 100 s
  time bucket: 700→255, 800→2443, 900→233.
- Frozen `lsvalidcpu3`: **218 CPU PASS, zero skips, 42.672 s, zero sim-s**.
  Per-query validation/arrival-bound/OOM errors decline with tracebacks; named
  safety/physics/accounting/parity invariants remain fatal.
- All **72 completed episodes / 14,534 snapshots** adopted. Seven invalid or
  interrupted admissions are terminal declines, never inference retries; their
  charges remain. One valid unassigned request is retained.
- CPU broker/supervisor **1485244/1485245** are durable. New GPU owners
  **7429/7430 (gpuA), 491842/491843 (gpuR)** are running **eight fresh gates each**
  before admission. Original numerical files and bindings remain unchanged;
  the error-boundary overlay is separately SHA-registered.
- At unix1793629401: **0 actual finished searches / 0 gold lists**. The legacy
  raw response count of seven consists of terminal declines. The new observer
  distinguishes those counts and will publish the first genuine full-pool response.
- **A48/B3/R3 retained; no u4 PPO update.** B-REPLAY remains inactive pending its
  delegated integration and the registered bank-entry prerequisite.

## LS-COLLECT engineering recovery — frozen B-u3, training paused
- Fresh session read **all 485 brief lines**, adopted commit `341e9417` and prior
  uncommitted work. The latest MAIN18:58/19:12/19:56 collection order is active.
- **72/72 TRAIN collection episodes completed**, **14,534 observable snapshots**
  retained. Completed episodes are adopted without another episode simulation.
- The original CPU supervisor exited without a failure receipt; retaining full
  history/policy payloads in its queue is a suspected memory cause, not a proven
  diagnosis. Recovery retains only queue metadata in RAM and loads payloads on demand.
- Four original GPU handovers drained their work and terminated old workers, then
  failed an exact command-line assertion during process exit. Recovery treats an
  empty command line with the same boot/start identity as exiting, never a signal
  target; changed identities still fail. Original claims and receipts remain intact.
- Frozen `coopmwm20261102_lsrecoverycpu1`: **205 CPU PASS**, **0 skips**,
  **42.859 s**, **0 sim-s**. The top-eight label wrapper preserves the original
  eight finalist WM calls and keeps negative predictions, per MAIN19:56.
- Registered recovery SHA
  `f4ea0e3d3c70d66e779e8ad718636336160edaeb85c5decf543f2c8cf752111b`.
  CPU broker/supervisor **1297398/1297399** are durable. Four replacement GPU
  owners **986989/986990** and **474268/474269** are qualifying with four fresh
  exact gates plus four source-identical adopted gates apiece; no READY claim yet.
- At unix1793623759: all episodes adopted; **8 precharged searches**, **0 finished
  collection searches / 0 gold lists**. Initial admitted worlds are 986004/986005;
  subsequent admissions rotate new worlds before repeats, then middle/late states.
- Registered collection envelope and **11,900,000 sim-s cap** are unchanged;
  restricted interim/k8-training phases are unused. **B/R remain u3, no u4 update**.
  Command mirror remains off after its failed gate; learned training awaits MAIN.

## LS-COLLECT registered before collection physics
- Registration SHA `edacc2b85b04b8c122c5827077be940e6582fe021f18c4b6094b7df507fac7aa`.
- **201 CPU PASS**, zero skips, **42.334 s**, zero simulation; frozen release
  `coopmwm20261102_lsreleasecpu2`. Harden-3 is integrated with MAIN18:54
  breadth-first scheduling and compatibility with existing withdrawal markers.
- Frozen B-u3 collection spans the **72 registered TRAIN worlds**. Episodes
  snapshot eligible groups without waiting and execute policy actions only.
- Registered envelope: **86,400 train sim-s** plus up to **2,332,800 verification
  sim-s** (216 lists × eight original finalists + policy, worst-case full prefixes).
  Each complete list is precharged before search; all failures retain charges.
- Cap **11,900,000** includes the new restricted 900,000 interim phase. Collection
  uses existing train/verification allocations. Remaining training will need a
  fresh quote before MAIN authorizes learned-search resume.
- Four existing GPU slots are adopted; a logging-only profile-capture handover
  drains their current claims and requires fresh startup parity before serving.
  Search arithmetic, horizons, population, iterations and Thompson choice stay fixed.

## MAIN19:12 latest-order pause confirmed
- B/R remain at durable **u3**, with **no u4 checkpoint or PPO update**.
- The superseded FAST1844 wave is closed: **10 B + 12 R claimed attempts**,
  including eight completed R results preserved. None may be replayed.
- Only dispatch/boot/start/argv-verified B/R owners and descendants were signalled;
  the post-pause audit finds **zero remaining owned descendants**. GPU searches,
  broker, grader and final coordinator were retained.
- Closed-wave charges: **26,400 sim-s**, retained; total charged **4,490,930 sim-s**
  at unix1793618754. No new simulation during this pause. Registration and
  `LS_COLLECT_PAUSED_1912.json` preserve hashes and the process tree.
- Next is non-blocking frozen-B-u3 LS-COLLECT under MAIN18:58/19:12. The latest
  cap is **11,900,000 sim-s**, including the restricted interim allocation.
  Training resumes only on MAIN's learned-search resume addendum.

## MAIN18:38 adopted — B/R paused for LS-COLLECT
- Durable B/R checkpoints remain **u3**; **no u4 PPO update**. The ten stalled
  `warmdns1651_u004_B_*` attempts are closed permanently, with all charges kept.
- `LS_COLLECT_PAUSED.json` records checkpoint hashes and the exact verified
  process tree. GPU searches, completed responses and full-pool receipts remain
  preserved. No new SUMO was launched during the pause.
- Next is the registered frozen-B-u3, non-blocking TRAIN collection; training
  resumes only on MAIN's subsequent addendum. Latest owner cap is **11,000,000
  sim-s**. The command-mirror gate has already failed and remains off.

## Both GPU clocks migrated; collection status — 18:10 UTC+8
- Both primary GPU services passed four fresh **EXACT0** full/fast cold/warm
  gates plus four source/SHA-adopted gates. Original bindings and startup receipts
  remain intact. Queue service resumed **17:48:57 UTC+8**, deadline Nov5 23:59.
- Broker `verifybeta13`: **155 CPU PASS**, zero skips, **35.240s**, zero physics;
  owner **927733/start62785248**. A dead legacy worker's assigned plan query is
  terminally declined on CPU so its episode continues on policy. Its inference
  claim remains unreplayed; diagnostic values are never fabricated.
- The known unsupported post-hoc-family label bug also remains in legacy gpuR
  sources. Fixed-source empty replacement spools are qualifying on both hosts;
  eight fresh exact gates are required before separate hot-add. The R replacement
  drains the already-running old second-worker query before taking its memory.
  Existing primary services continue. No search/horizon/learning change.
- Current B-u4: **14/24 complete**; ten fresh-ID episodes at sim t300–390 wait
  for answers. Oldest unresolved request ages **3,624–3,635s**, including the
  migration/queue interval and still censored. **R-u4 has zero claims**, waiting
  for B-u4's final matching quota. No finite steady-state u4/u24 ETA is justified
  yet; no B-u4–6 viability claim. Receipt `EPISODE_PROGRESS_1793614233.json`.
- Read-only progress now resolves additive bindings and fresh-ID overrides;
  **22 local CPU checks pass**. Registered charge remains **4,450,130 sim-s**
  at the latest check. Fixed final outputs remain closed.

## Command-mirror CUDA gate FAILED — MAIN16:39 disposition
- `COMMAND_MIRROR_GPU_GATE_1651.json`: actual WM callback CPU/CUDA trace parity
  **exact0**, but absolute member-vs-existing-SUMO trajectory errors are nonzero.
- World986051: maximum position **0.8599243164 m**, speed **0.3810596466 m/s**.
  World986070: **0.0787963867 m**, **0.0550899506 m/s**. The prototype produces
  the same errors as the old chain on these two null fixtures.
- Background half-steps are law-reconstructed from observable 1s endpoints, with
  roster/lane ambiguities retained. This is a failed necessary observable gate,
  **not a full-road/true-half-step SUMO parity pass**. No new SUMO was run.
  **Command mirror remains OFF for BOTH B and R**, per the owner's failure rule.

## Final resource handover completed — 17:24:48 UTC+8
- Frozen `finalendpoint5`: **90 CPU PASS**, zero skips, **23.806s**, zero physics.
  Regression verifies opaque result bytes are hashed without JSON decoding.
- All **480 A24v2** episode/branch completions and **673 original** opaque receipts
  are adopted. Old waiting owner689787 was identity-verified with no physics
  children and replaced by **852974/start62576671**. No completed job was rerun.
- Additive config/suffix uses the authorized Nov5 resource clock and current
  phase caps. Final outcomes remain closed until fixed B24/R24 endpoints.

## MAIN16:51 DNS recovery — durable supervisors restarted 17:09:20 UTC+8
- The transient DNS V4 STOP closed the ten remaining B-u4 attempts without a
  result. Fourteen completed episodes are retained and adopted; ten replacement
  attempts use fresh `warmdns1651_u004_B_*` IDs on the same registered TRAIN
  worlds and original noise wave. Every original charge and receipt is retained.
- Frozen `verifybeta9`: **153 CPU PASS**, zero skips, **36.388 s**, zero physics.
  Live preflight restored u3 for both arms and checked the original binding SHAs.
  The extension is a new `RESOURCE_EXTENSION_1651.json`; originals are unchanged.
- SSH metadata transport now retries DNS/connection/reset/timeout/exit255 errors
  at most six times over 300 s, rechecking owner/gate receipts before identical
  delivery. Code/parity/physics faults remain fatal. Only the exact diagnosed
  DNS-stop SHA is released; a fresh V5 STOP remains fatal.
- Registered **12,000 sim-s** transfer: discovery150,200→138,200 and
  train2,840,400→2,852,400. Latest ordered cap11,000,000 includes a separately
  restricted **1,500,000** k8-verification phase, still inactive pending activation.
- New detached owners at unix1793610560: **B821013/R821014/broker821011/grader821012**.
  Effective verified guide beta=1. R waits for B-u4's final matching quota. Initial
  logs are empty. The registration SSH timed out after remote completion; all
  additive receipts were verified and adopted before launch, without repeating it.
- New CPU runtime reads the extended deadline **2026-11-05 23:59 +08**. Existing
  GPU processes and final coordinator still cache the original deadline and need
  a qualified handover. Command mirror remains off pending its full trajectory/GPU
  gate; learned search/k8 verification remain inactive. **B-u4–6 viability pending.**

## MAIN16:07 verified-guidance correction — durable boundary adoption
- Warm memory, fast-path parity, MAPPO-u0 restarts, library and the original
  five-branch SUMO verification smoke are completed work and retained.
- **Effective guide beta=1 for B and R under exact verification** is registered.
  The adaptive exact-grade trust curve remains reporting-only. The Controller
  preserves the verifier's effective beta/query probability and consumes exactly
  one original beta/pad draw; exact-positive plans alone execute and enter AWR.
  Count-starvation, safety, harmful-plan and learning checks remain active;
  reporting-only trust reaching zero cannot itself starve verified guidance.
- Frozen numerical `verifybeta2`: **131 CPU PASS**, zero skips/physics,34.463s.
  Includes both arms at tracked beta0, exact-positive acceptance/negative rejection,
  actual episode-loop/five-branch continuation and grade reuse. The initial
  qualification failure is retained (missing fake-world seed and optional branch
  behaviour metadata); corrected before release.
- **R736732** restored u3 Adam/RNG/trust, configured fixed guide beta1 and waiting
  for B-u4's final matching quota; **no R-u4 claim yet** at unix1793608384.
  R-u2/u3 had zero executed plans because legacy beta was0; disclosed deviation.
- **B598899 parent deliberately T**, children continue existing u4 SUMO/GPU work.
  Registered detached controller **748938/start62256954** waits for all24 current
  replacement RESULT+CLOSE receipts, then adopts u3 plus this collected wave and
  applies u4 exactly once. At unix1793608384: **24 claimed/14 complete**, ten open.
  Existing B-u4 episodes retain pre-correction trust; no claimed episode is replayed.
  Fixed beta applies to B's subsequent fresh decisions; this timing is disclosed.
- Controller-only `verifybeta5`: **133 CPU PASS**, zero skips/physics,36.649s.
  Prior controller failures are preserved: Linux empty-argv exit race after the
  child-free R parent was killed, then a missing deadline key before B handover.
  Partial R handover was adopted without re-signalling or redispatching R.
  Current V3 controller is live with an empty log; numerical source2 is unchanged.
- Same 9,500,000 cap/MAIN15:56 allocation; no additional handover physics charged.
  Original **18:07:02 UTC+8** resource deadline retained, not implicitly extended.
  Final outcomes remain closed. **B-u4–6 viability is still pending.**

## Approved final-only recovery live — 16:04:22 UTC+8
- MAIN15:56 allocation registered as `758ad384…`, commit `ca2a1022`:
  **375,800 sim-s** transferred from remaining separate exact grading
  (431,160→55,360); final-test allocation **691,200**, paired-branch **1,297,400**.
  Total stays **9,500,000**; training/discovery/verification allocations retained.
  This uses the remaining freed extended-B grading allocation; B-u25–48 training
  had already been released and is not counted a second time.
- The source3 checkpoint repair is retained. A final-only accounting adapter was
  needed because it otherwise loads the old immutable training binding's phase
  caps. Frozen `finalendpoint4`: **89 CPU PASS**, zero skips, **23.55 s**, zero
  SUMO/inference, commit `f037d749`. Training bindings and supervisors retained.
- New detached final coordinator **689787/start62093997** is live. Its held
  predecessor was never resumed: all386 invalid jobs were closed, no physics
  descendants remained, and only its identity-verified idle pool and parent
  were terminated. All charges and original673 opaque receipts remain retained.
- At unix1793606718: **23 corrected A24v2 jobs complete**. Published checkpoint
  metadata verifies actual **A-u24 SHA**; no failure or outcome opening. Final
  analysis remains gated on B-u24 and R-u24. No final outcome values were read.
- B/R collection, four GPU slots, exact grader and observe-only publisher continue.
  Original resource deadline **18:07:02 UTC+8** still needs registered continuation.

## Final-only engineering hold — spawned-worker checkpoint authority
- **Final coordinator625637 is deliberately frozen**, before any outcome opening.
  B/R training, four GPU workers, exact grader and observer continue.
- Code inspection found that spawned final workers import the original u48
  budget; they do not inherit the coordinator's in-memory u24 override. The
  A24-labelled specs lacked explicit checkpoint authority, so they loaded A48.
  This invalidates those attempted primary jobs; no outcome values were read.
- **386 reserved jobs / 375,800 sim-s** completed and are quarantined with every
  original charge retained. The original 673 opaque precompute receipts remain
  intact. No invalid job is reused as an A24 result.
- Registered incident `078cf357…`; frozen corrected `finalendpoint3` passes
  **88 CPU tests**, zero skips, **23.115 s**, zero physics. Each serialized policy
  spec now carries `checkpoint_update`; missing authority fails closed, workers
  load the exact checkpoint and publish its SHA. The regression simulates a
  fresh u48 worker import consuming a serialized u24 spec. Fresh A24v2 IDs and
  separate continuation markers preserve every prior artifact.
- **Replacement final physics awaits MAIN allocation approval:** 375,800 exceeds
  the delegated 50,000 limit. Proposed within the existing9.5M cap: exact-grade
  allocation431,160→55,360; final-test +115,200; paired-branch +260,600. No
  transfer has been applied. See `_sol/COOPMWM_FINAL_SPAWN_BUDGET_ALERT.md`.
- Current B-u4 replacement schedule at unix1793605168: **24 claimed /14 closed**;
  ten windows t300–390 await WM responses (oldest unresolved ages950–962s,
  censored). R-u4 waits for B's final quota. Four search slots assigned,38queued;
  gpuA sample95% utilisation/45,083MiB/171.92W. No finite update ETA yet.
- MAIN14:59 clarification: disk recovery used cache cleanup and SHA-verified
  in-place gzip; the relay archive was stopped. Earlier relay attribution is
  superseded. This continuation performed no deletion. Original GPU/service
  deadline18:07:02 remains unchanged; resource continuation requested.

## Fixed-u24 final hooks migrated; opaque precompute retained
- Frozen `finalendpoint1`: **87 CPU PASS**, zero skips, **22.584 s**, zero
  SUMO/inference during qualification. Registered amendment `0b877c48…`.
- Detached coordinator **625637**, start61917851, replaces only the two verified
  waiting final/discovery owners. All **673** original MAPPO/native/A48/reuse
  receipts remain adopted and opaque. The sole old child was Python's owned
  resource tracker; no episode/branch worker was active and no child was signalled.
- Frozen A24 episodes/branches use fresh `fixedfinal_A24_…` / `fixedbranch_A24_…`
  identities inside the already registered 460,800 sim-s allocation. Primary
  comparisons use A24/B24/R24; A48 episode/branch results are separately labelled
  secondary. No checkpoint selection, panel change or duplicate precompute.
- B24/R24 completion still gates outcome opening and registered analysis, then
  detector-first B24 n+2 twins under the remaining discovery allocation.
  At unix1793604918: coordinator live, no failure, **outcomes still closed**.
- Observe-only V11 **11640** restored after V10 was verified absent. **21 CPU
  checks pass**; first publication recovered eight immutable TRAIN receipts.
  Existing training/GPU owners and original resource deadline retained.

## SUMO VERIFICATION LIVE — common update 4
- Actual B/R continuation starts **15:23:20 / 15:23:31 UTC+8**; durable live
  publication **15:25:29**. Frozen `verifycpu12`: **105 CPU PASS**, zero skips,
  28.041 s. Both arms adopt their u3 checkpoints, optimizer/RNG and trust state.
- B takes up to four WM candidates; R takes four random candidates. Trust is
  drawn first. Exact SUMO picks a strict improvement over policy, and its grade
  is reused for trust/imitation. R's existing beta is **0** after its three
  legacy exact grades, so its trust draws currently select policy; not reset.
- Detached owners: **B598899/R599343/grader598089**. Broker545008 and all four
  qualified GPU workers retained. Both supervisors have fixed **u24** endpoints.
  Latest state has no active STOP/health pause.
- Activation preflight found all 24 old B-u4 attempts had warmed to t300 after
  disk admission recovered, then stopped on the pre-existing V3. No RESULTs or
  control outcomes exist. All original charges remain. Registered amendment
  `e84d10e6…` funds **28,800 sim-s** same-cell TRAIN replacements from discovery
  (179,000→150,200, above the 120,000 floor), within the unchanged 9.5M cap.
- The transient SSH closure did not cancel the remote binder: its completed
  receipt was adopted before dispatch. No duplicate binder or episode replay.
- Final primary A24/B24/R24 hook migration and B-u4–6 viability remain pending.

## Real five-branch verification smoke passed — 15:13:08 UTC+8
- Pre-registered first eligible B-u3 TRAIN decision: world986000 at t470.
  **All five exact branches pass prefix and CRN checks.** SUMO rejected all four
  candidates; the live policy continuation matches the policy branch at all
  **11 fingerprints, t470–480**. Original smoke cost **3,730 sim-s**.
- The first helper wrongly required an accepted candidate to pass. Registered
  engineering amendment `4dc2f2f6…` makes the gate outcome-independent: compare
  the accepted candidate branch, or the policy branch on rejection. No repeated
  SUMO, inference or charge; the original completed receipts were rechecked.
- Frozen `verifycpu11`: **105 CPU PASS**, zero skips, **28.153 s**, zero physics.
  Original `verifycpu10` and the failed helper log remain preserved. Smoke SHA
  `ca3a470c04a5ced54e473973b78c66906636847e1d4588a9bd26589fa3241342`.
- Owner-approved **9.5M sim-s** allocation registered (`5f73c1e3…`): 2.8M
  verification phase; separate exact grading 431,160, retaining all spent grades.
  Variable acceptance can exhaust a phase: accounting blockers remain binding.
- R-u2 applied with **0/12 requested matches**, fully reported; R-u3 collecting.
  **SUMO VERIFICATION LIVE is not yet claimed:** common-u4 activation is next.

## MAIN14:47 recovery — completed R episodes adopted; matching is best effort
- Frozen `matchcpu1`: **125 CPU PASS**, zero skips, **21.715 s**, zero SUMO/GPU
  inference. Amendment `d54fbc440e931a9eec1e192e98fc1c4dab8d69d69fa634045fd41d72641c2967`
  registered before continuation.
- All **24 R-u2 replacement episodes** completed before the assertion. They are
  adopted with the original R-u1 optimizer/RNG; no completed episode is rerun.
- Matching now prefers the same regime and nearest eligible group size within
  ±1, retains the ticket duration, and records missing/extra strata explicitly.
  A shortfall cannot stop the update. R remains an unscored random proposer.
- Exact SHA/authority release covers only the diagnosed V3 matching assertion;
  fresh V4 faults and all health pauses remain fatal. Original receipts retained.
- Restored detached cpuQ owners: grader **544875**, broker **545008**, R **545912**.
  First logs are empty and no fresh STOP at unix1793602747. R resumes through u3
  to reach the common u4 verification boundary. B remains at durable u3.
- B's old supervisor naturally exited on V3 at **14:57:15 UTC+8**. Its 24 B-u4
  jobs have no claims and retain their existing precharges. No signal was sent.
- Disk recovered to **3,404,877,824 bytes** after MAIN's relay archive; the
  existing 2 GiB guard is retained. This continuation performed no archive deletion.
- **Real verification smoke and matched activation remain pending.** Latest
  owner cap is 9.5M sim-s; verification-phase allocation still needs registration.

## MAIN14:24 recovery — both GPUs serving all four slots
- At **14:28:10 UTC+8**, the existing gpuA replacement **759743** was admitted
  as `gpuA@incident1`. All eight startup gates were already **EXACT0**.
- The admission fault was visibility: the legacy restricted RPC returned only
  six gates. The original CPU26-qualified `hotadd_rpc` exposes both additional
  committed-plan receipts; its source SHA was checked before installation.
  Four focused local CPU tests pass. Fresh restricted credentials remain on cpuQ.
- Broker receipt unix1793600899 confirms **four admitted slots, four assigned
  queries, no unassigned queue**, with failed `gpuA@0` still disabled. Existing
  workers, queries, numerical source and original deadline were retained.
- No qualification or episode was repeated. Admission SHA:
  `4aaa6fdd665bf4c873f3cd5dd8dd1ff1506703f1a1103118fb0ac3612e23777f`.
- Next prescribed steps: finish existing B-u3/R-u2, real five-branch verification
  smoke, then matched activation. The latest owner cap **9,500,000 sim-s** in the
  complete brief supersedes earlier resume figures; old cap-block notes are historical.

### B-u3 complete; cpuQ disk blocks the next cohort
- B-u3 durable update timestamp **14:27:12 UTC+8**: **10 executed plans**,
  34 query records, 26 accepted forecasts. Exact-grading lag mean **10.51 s**,
  max **19.06 s**, with no pending/running/failed grades at update publication.
- Recorded collection-inclusive wall time **19,446.06 s** includes the incident
  and prolonged replacement-admission outage. Its linear ETA is not a forecast
  of the restored four-worker service.
- At unix1793601294, R-u2's registered replacement schedule has **17/24 completed**,
  four active episodes at sim t350–660, three still unclaimed. Existing owners
  continue. B-u4 has **zero claims** and is waiting on disk admission.
- cpuQ free disk at unix1793601042: **2,069,700,608 bytes**, below the existing
  **2 GiB** guard. The only approved SHA-gated archive was already completed and
  its source removed; its receipt is retained. Additional disk capacity or explicit
  additional own-source archive/removal authority is needed. Alert sent through
  canonical registration and `_sol/COOPMWM_MAIN1424_DISK_ALERT.md`.
- Real SUMO verification smoke and matched-boundary activation remain pending.

## 11:18 continuation — verifier grade reuse CPU-qualified
- Original warm-memory qualification, MAPPO-u0 starts, same-arm/seed library and
  remote owners adopted. At unix1793589133: **A48/B2/R1**, B/R/broker/grader
  live, no active STOP or pause receipts; charged **3,507,050sim-s**.
- **100 CPU PASS, zero skips,27.928785s, zero SUMO/GPU inference**, frozen
  `verifycpu9`, commit `afbb1082`. Replay now retains pre-verification beta and
  R-sampler draws even after own-policy/rejected decisions. Legacy draws unchanged.
- The actual CPU episode/controller loop passes five-branch verification and
  chosen-branch/live fingerprints through the next query window. Exact source
  grades import into trust/AWR with no new branch/charge; concurrent importers
  adopt one receipt. A truncated plan instead queues its executed-portion grade.
- Reused grades are explicitly **decision-local exact verification** on the
  common max-candidate-duration+120 window, with unknown future proposals absent;
  they are not relabelled as later marginal grades of the completed live episode.
- **Not live:** no real smoke, common B/R boundary switch or endpoint-supervisor
  migration. Fresh6M versus canonical9.5M cap clarification remains outstanding
  before new verification spending. Original live training and observers retained.
- Qualification detail/failures: `coopmwm/VERIFIER_CONTINUATION_CPU_20261102.md`.
  No new viability claim; B-u4–6 learning checks remain pending.

## MAIN10:30 continuation — verifier CPU-qualified; current owners retained
- Full322-line brief read; original warm-memory qualification and MAPPO-u0
  starts adopted. Recovery B/R/broker/grader remain live, no active STOP/pause.
  At unix1793587436: **A48/B2/R1**, charged **3,507,050sim-s**.
- gpuA replacement759743 is now **READY, all eight retained gates EXACT0**;
  no qualification was repeated. This is worker readiness, not an admission claim.
- Landed verifier8ff759fd integrated. Pre-verification trust now draws once;
  own-policy draws launch no branches, verified winners execute without a second
  draw. Durable decision claims block duplicate charge/physics and preserve
  interrupted claims. Explicit run-root accounting installed.
- **82CPU PASS,0skips,0SUMO/inference,22.574s**, frozen `verifycpu2`.
  First failed harness receipt retained; no tests skipped. Integration details:
  `coopmwm/SUMO_VERIFY_INTEGRATION_CPU_20261102.md`. **Not activated**: real smoke,
  exact-grade reuse, matched B/R update boundary and final-hook migration pending.
- **Budget clarification required before new verification physics:** fresh
  resume says6M, supplied later addenda and canonical owner approvals say9.5M.
  Current adopted recovery allocation7,028,840 is bound under7.5M. No new
  SUMO/GPU dispatch or PID control was performed by this continuation.
- Observe-onlyV10 **PID9636**, startedunix1793587515.739, original18:07:02
  deadline retained;18CPU observer/state/publication checks PASS. First live
  poll/publication succeeded. It cannot launch training or inference.
- Corrected `episode_progress --current` to use the incident replacement
  schedule and exclude retired query keys. Earlier snapshot1793587542 showed
  retired original jobs; retain it as superseded observation, not current waits.
  Three actual-payload CPU tests PASS. Corrected snapshot1793587611:
  B-u3 **24claimed/14closed**, ten active episodes att300–470, oldest request
  ages1129–1139s; R-u2 **8claimed/4closed**, four active att410–570, request
  ages114–1136s. Queue waits are censored; no post-recovery update ETA yet.
- Latest published B exact marginal evidence remains15plans,mean+145.2veh-s,
  95%CI[−60.80,+487.82]. **Viability is not established**; B-u4–6 pending.

## MAIN10:22 fresh recovery — B/R continued from durable checkpoints
- Complete316-line brief and canonical registration adopted. Latest owner cap
  **9,500,000sim-s**; primary final checkpoints **A24/B24/R24**, A48 secondary.
  Both B and R receive exact SUMO candidate selection at the same update index
  once the delegated verifier is integrated and qualified.
- Recovered incident source **118 CPU PASS**, zero skips/SUMO/inference,
 21.287s. Existing recovery allocation **7,028,840sim-s** is registered under
  its7.5M binding;179,000discovery reserve after the explicit21,000transfer.
  Verification allocation under the higher cap is a separate pending registration.
- Recovery binding completed remotely after the observer SSH timed out; its
  four durable binding/release/config receipts were verified before launch.
  No duplicate binder or episode dispatch. Completed episodes and original
  B-u2/R-u1 optimizer/RNG/library state adopted; only21registered interrupted
  attempts replaced, failed charges retained,10unclaimed R precharges adopted.
- **TRAINING CONTINUED B/R at10:27:22/10:27:30UTC+8** (unix1793586442/6450).
  Remote detached owners: B4146203/start60072070, R4146817/start60072842,
  broker4145985/start60071018, grader4146093/start60071516. All four live,
  empty logs and no fresh STOP at unix1793586457. Frozen `recoverycpu1`.
- gpuA replacement759743 remains in its already-running final startup gate;
  seven retained exact-zero gates observed. Other three qualified workers retained.
  Original failed inference is never replayed. Warm qualification/original restart
  retained; command-mirror OFF pending genuine trajectory parity.
- B-u4–6 viability remains pending; latest published exact marginal evidence
  **15plans,+145.2veh-s,95%CI[−60.80,+487.82]** does not establish viability.

## MAIN09:44 incident — labelling fixed; physics recovery budget blocked
- The gpuA primary failed at unix1793583692 on the legacy2/3-member family
  validator, called **after** a4–6-member winner had been selected. Broker STOP
  at1793583695 then interrupted B-u3/R-u2 and the exact grader. A48 is complete.
- The unsupported-size labeller now returns `new`; other label failures return
  `unlabelled` plus traceback. The actual4/5/6-member runtime paths pass CPU tests.
  Source1 **97PASS**, source2 **98PASS**, zero skips/SUMO/inference. Source2 also
  records worker-scoped restart requests without globally stopping healthy slots;
  safety/physics/parity/accounting failures remain fatal. Per-query code faults
  decline with traceback; partial compute counts are unknown, never imputed zero.
- Registered amendment **452d096b…**. Replacement gpuA primary **759743**, start
  **896453815**, launched once at unix1793584324 (**09:52:04UTC+8**) in isolated
  `incident1`. Fresh eight exact-zero startup gates are running; READY/admission
  is not yet claimed. Other three GPU workers remain live/qualified. The original
  failed inference claim is retained and has not been replayed. Source2 broker
  refinements are CPU-qualified, not deployed; GPU worker runs frozen source1.
- Read-only census1793584192: B-u3 **14 completed/10 interrupted**; R-u2
  **3 completed/11 interrupted/10 never claimed**. Original B2/R1 checkpoints and
  all completed episodes remain. Interrupted jobs have no RESULT or durable SUMO
  checkpoint. Their original25,200sim-s charge stays; replacements need another
  **25,200sim-s**, against only4,960 unallocated. Full quote **7,520,240** exceeds
  owner cap7,500,000 by **20,240sim-s**. MAIN allocation decision requested in
  canonical registration and `_sol/COOPMWM_MAIN0944_BUDGET_ALERT.md`; no reserve
  cut, refund or replacement physics has been silently authorised.
- **B/R remain stopped** pending qualification, registered recovery allocation
  and deployment. Charged3,481,850 unchanged; original18:07:02 GPU deadline
  retained. B-u4–6 viability remains pending; latest exact marginal CI includes0.

## 09:34 continuation — owners adopted; parallel-wait reporting corrected
- Fresh session read the complete298-line brief and recovered receipts. At
  unix1793583292: **A48/B2/R1**, current B/R/broker/exact-grader live, no active
  STOP/pause/failure. The existing local observe-only V9 and telemetry V3 remain
  live. Original warm qualification/restart and all assigned work were adopted.
- B-u3 has24 episodes claimed/14 closed; ten quota windows submitted33 parallel
  TRAIN searches. R-u2 has7 claimed/3 closed. All four existing GPU workers are
  ready with eight retained exact-zero startup gates each; no gates were rerun.
  Individual GPU samples: gpuA83–84%, gpuR89–95%. No post-fan-out update has
  completed yet, so the old B-u2 ETA is not a new-rate forecast.
- **Reporting correction**: `episode_progress` had summed concurrent query ages,
  producing13,098s for an approximately1,637s episode wait. It now reports the
  oldest unresolved request age as `current_query_wait_s`, with separate pending
  count and aggregate request-age effort. At unix1793583125 the ten blocked B
  episodes are at sim t300–380 with oldest pending ages1725–1736s. Waits include
  admission/queue time and remain censored. Two actual-payload CPU tests pass;
  real remote observation passes. Amendment and old receipts retained; commit
  `dd1e00d9`. This is an observation-only correction.
- Exact grader at unix1793583156:18 receipts across both arms, no pending/running
  grades, smoke passed. All12 repaired registered n+2 exact diagnostic receipts
  000–011 have now been published alongside the retained failed attempts. These
  diagnostics use their registered MAPPO background; marginal training grades
  preserve other plans/current-policy background and are a different contrast.
- Latest B exact marginal evidence remains **15 plans, mean+145.2veh-s,
  95% world-cluster CI[−60.80,+487.82]**. **Viability is not established**;
  the B-u4–6 curves/package and full WM diagnostic comparison remain pending.
- Charged3,481,850sim-s; allocation7,495,040/cap7,500,000,4,960 unallocated.
  Existing remote training and publication supervisors continue under the
  registered guards; original GPU deadline18:07:02UTC+8 retained.

## TRAINING CONTINUED — parallel-group fan-out is live
- **B continuation started2026-11-02 09:01:19UTC+8**, owned supervisor
  **3946079/start59555725**, immutable `tailcpu3`. All24 original B-u2 results
  were adopted and the exact u1 checkpoint/Adam/RNG restored; no original episode
  or completed update was rerun. Original warm qualification/start are retained.
- **B-u2 completed with12 executed plans. B-u3 is running**: at unix1793581947,
  all24 episodes claimed/14 closed,10 quota windows submitted33 training searches.
  All four registered GPU slots are assigned. One utilisation sample at
  unix179358149x gives gpuA75%,gpuR96%; this is not a steady-state rate claim.
  **R-u2 runs concurrently**,6 claimed/2 closed, using original matching/no library.
- Current broker3893211 retained all assigned original forecasts. The one-time
  old-wave drain automatically ended at `TAIL_HANDOVER_COMPLETE.json` before
  B-u3; ordinary quota-aware admission is restored. Old supervisor/transition
  owners exited; current B/R/broker/grader are live, no active STOP/pause.
- Same-B/seed20261101 **winning library now contains7 exact-positive entries**.
  Warm memory, K4/H300, K8/Hfinal, full search/top-k/Thompson, exact-only learning,
  horizons and budget remain as registered. Source3 CPU141PASS; broker48PASS.
- B-u2 collection-inclusive wall time **9838.6425s** (2h44m), next beta0.7509695;
  exact-grade lag mean11.756/max17.956s, no pending/running/failed grades at update.
  The printed125.716h remaining estimate extrapolates the old pre-fan-out u2;
  post-amendment timing awaits a complete new update. No completion ETA is invented.
- **Query metadata correction** (`coopmwm/TAIL_QUERY_CENSUS_002.json`,3 CPU tests
  plus real read-only census):122 bound queries =22 completed WM inferences +100
  resolved capacity declines,15 accepted forecasts,0 pending responses/unknowns.
  Raw routing93 counts legacy visible query routes; queries hidden by continuing
  plan routes explain why it is not a WM-call census. Raw `gpu_busy.pending_queries`
  100 incorrectly counts resolved zero-compute declines as missing broker-completion
  receipts. Use this supplement; original update/grade data are retained. Completed
  inference-interval fractions themselves remain labelled, not GPU hardware util.
- Latest published exact marginal evidence: **15 plans,mean+145.2veh-s,95% CI
  [−60.80,+487.82]** (`B_EXACT_VIABILITY_99cf50a5…`). **Viability is not established**;
  B-u4–6 learning curves and the registered package remain pending. The existing
  isolated diagnostic repair is completing the same registered sample alongside B.
- Observe-only **20516/V9** replaced only verified local1372/V8; **23 CPU tests
  PASS**, current TAIL owner/failure schemas loaded. Telemetry24140/V3 remains live.
  Allocation7,495,040/cap7,500,000sim-s,4,960 spare; charged3,481,850 at snapshot.
  Original GPU deadline18:07:02UTC+8 remains. Depth diagnostics await spare GPU
  capacity; the unqualified command-mirror prototype stays disabled.

### Tail release continuation — real checkpoint preflight passed
- Real CPU preflight caught a continuation-only fault **before any new B launch**:
  `restore_progress` read compressed `RESULT.json` completion markers as raw
  episodes. Only the owned transition controller was held; existing SUMO/GPU,
  broker, grader and R continued. The original checkpoint and all results remain.
- Fixed reader to use the existing digest-verified lossless `accel_storage.read`.
  Immutable **tailcpu3:141 CPU PASS**, zero skips,27.091s, zero SUMO/GPU jobs;
  adds a real compressed-result fixture and adoption of the already-frozen wave.
  Repaired transition **3907544/start59459454** is live since unix1793580316;
  old held controller3874249 was identity-verified and terminated.
- Real preflight at unix1793580384 passes: exact u1 checkpoint SHA6606191e940…,
  exact generator state,17 Adam parameter states,1226 cumulative diagnostic
  records,3 prior diagnostic selections, and **two same-B/seed library entries**.
  Restored-state SHA1853bfb4d653… matches the prior successful read. The earlier
  preflight's `library_entries=1` counted the library container key; its state
  equality was correct and the reporting field is now corrected to2.
- The one-time quota-complete admission drain is live under broker
  **3893211/start59424477**, registered V2, **48 CPU PASS**. Its first launcher
  refused before signalling because the rootless `serve` entry point has no root
  argv; exact argv + boot/start + source PYTHONPATH now qualify its owned identity.
  Original broker/config/receipts retained; all assigned inference claims adopted.
- At unix1793580332: **A48/B1/R1**, B-u2 all24 claimed/22 closed, quota10 executed
  plus2 in-flight. The two original GPU-R searches remain incomplete and owned;
  no cancellation or inference replay. Transition/broker logs empty, active STOP
  null. The controller waits for both episodes, adopts all24 results and continues
  the exact checkpoint. Ordinary admission restores before new B-u3 fan-out.
- Latest published exact marginal evidence remains11 B plans/10 worlds,
  +224.8358veh-s with95% world-cluster CI[−35.3982,+596.7440]. **Viability is not
  established**; the B-u4–6 package is pending. Allocation/deadline unchanged.

## MAIN08:18 — quota-tail concurrent group admission qualified and registered
- Recovery adopted completed **WARM MEMORY QUALIFIED**, the existing B/R restart,
  and exact-positive same-B/seed library; no qualification or training was redone.
- Immutable `tailcpu2`: **139 CPU tests PASS**, zero skips,27.055s, zero SUMO/GPU
  launches. Includes actual frozen MAPPO parity, own-step PPO/exact-only grade
  learning, exact CRN draws, concurrency reservations and safe continuation tests.
  Initial `tailcpu1` fixture failures are retained. Local generated observer and
  publisher payload suites: **15 PASS**; real guard state passes/no active STOP.
- Amendment **e9af582e914d8a44…** registered before deployment: admitted windows
  while the execution quota is unmet publish searches for **all eligible groups
  before waiting**. Results resolve in inherited spatial order, first accepted;
  one plan starts per control tick. All other forecasts are logged as searched,
  not executed. One Bernoulli/window and one possible beta draw preserve the
  existing exact-grader CRN contract. One batch reserves one executable window.
- Each query keeps the same warm history, policy centre, search population,
  K4/H300 screen, K8/Hfinal, top-k/Thompson and exact-only learning. R continues
  the existing count/regime/size/duration matching. After executed quota10 the
  original single-group nonblocking admission applies.
- Detached transition **3874249/start59377221** dispatched once at
  unix1793579494. It pauses only original B supervisor3551873, while all current
  SUMO/GPU searches, broker3706352, grader3427015 and R3551874 continue. At
  unix1793579500: **A48/B1/R1; B-u2 22/24 closed**. It waits for the two original
  episodes, adopts all24 results, and resumes from exact u1 checkpoint/Adam/RNG,
  trust/library and diagnostic-selection state. New fan-out applies to the next
  wave. Never redispatch the transition or those already-owned searches.
- New update receipts include collection wall time from original claim times and
  per-slot completed-inference busy fractions (pending work censored). No new
  simulation allocation: **7,495,040/7,500,000sim-s;4,960 unallocated**. Original
  GPU deadline remains18:07:02UTC+8. B-u4–6 viability is still pending.
- Follow-up at unix1793579645: the two original episodes remain t350/t340 and
  keep opening new free-slot queries after quota10. A one-time old-wave boundary
  drain is CPU-qualified (**45 PASS**,0.862s,zero SUMO), pending registration:
  finish every assigned query, decline only new unassigned old-wave requests
  after executed quota10, and restore ordinary admission automatically before
  new B-u3 work. This creates a finite deployment boundary without cancelling
  existing physics/searches. Per-slot R-centre fairness remains unchanged.
- Observation-only telemetryV3 **24140** replaced only verified localV2 **28752**
  at unix1793579639, loading the previously qualified bounded-git helper;12 CPU
  tests pass. Existing learning-check observer1372 remains live.

## 08:12 continuation — Windows publication failure repaired
- Fresh session adopted A48/B1/R1 and every existing remote owner; no training,
  search or physics job was restarted. At unix1793578141 B-u2 has22/24 episodes
  closed; both remaining worlds have advanced to new, already-owned searches.
  R-u2 awaits B-u2's final matching quota. No finite completion ETA is asserted.
- Local observer46108 exited at unix1793578179 with Windows error206. Its log
  identifies an unbounded `git diff --stat` argument list: immutable receipt names
  accumulated across polls. Remote B/R, broker, grading and GPU work survived.
- Repair `0fc421c3`: intersect intended paths with changed/untracked filenames,
  bound diff command lines to16000 characters, and stage literal NUL-delimited
  pathspecs through stdin. Existing operator-staged work still defers publication.
  **Eight CPU tests PASS**, including1500 long receipt names and an actual
  observe-only run that cannot launch training; real Git stdin dry-run passes.
- Registered `coopmwm/EXACT_OBSERVER_COMMAND_LENGTH_RECOVERY.json`; replacement
  **observe-only1372/V8** started2026-11-02 08:12:36UTC+8 after46108 was verified
  absent. Its first real status/publication/commit cycle passed (`b830163f`).
  Existing telemetry28752 remains live. Original GPU deadline is unchanged.
- At unix1793578173:13 current B executed starts; exact grader14 receipts,
  pending/running0, smoke passed. Receipt count includes both arms and is not a
  count of B-qualified effects. Charged3338010sim-s; allocated7495040/cap7500000,
  spare4960 unchanged under the later owner-approved addenda.
- Warm qualification, original restart, exact-positive same-arm/seed library and
  covariance integration are adopted, not rerun. The command-mirror prototype is
  still disabled: completed CPU callback checks do not establish absolute
  trajectory/GPU parity. B-u4–6 viability remains pending; no new scientific claim.

## Fresh continuation — independent command audit and live evidence
- Read-only guard/status compatibility now includes the **current fairness broker
  and exact grader** with newest-dispatch precedence and boot/start identity.
  A dead newest owner is reported as dead rather than hidden by an older live
  owner. **12 CPU tests PASS**, including actual generated transport payloads;
  real cpuQ verifies broker3706352/grader3427015 live, active STOPnull. Existing
  local observers and all remote jobs continue; receipt
  `coopmwm/WARM_STATE_SERVICE_OWNERS_REPAIR.json`.
- Existing A48/B1/R1 training owners and observers adopted; no redispatch. At
  unix1793576973 B-u2 has22/24 closed episodes; R-u2 waits for its final B quota.
  No active STOP/pause; grader queue pending/running0 at unix1793576976.
- Latest published exact report at unix1793576889 contains **11 B plans across
  10 TRAIN worlds**, mean **+224.8358veh-s**, world-cluster95% CI
  **[−35.3982,+596.7440]**. Five positive grades. B-u2 physical nulls are3/8
  qualified grades; all zeros and losses retained. **Viability is not established.**
  Full rank/type/group breakdown is in
  `coopmwm/warm_live/B_EXACT_VIABILITY_f1d3fd1e73331c00e1ce9476a7ec8f4933728b3aed82117e089eb608441244ff.json`.
- Independent actual-callback CPU check propagates the named groups continuously
  and derives intermediate background speed from observable endpoints using the
  frozen semi-implicit Euler law. **Three new CPU tests PASS**, also on the GPU-A
  host with subprocess launches prohibited; runtime0.715s, zero SUMO/GPU jobs.
  Both named plan-versus-own-policy trajectories remain **exactly identical**,
  with both original and prototype chains. The prototype does not reduce absolute
  errors (position/speed maxima0.859924m/0.381060m/s and0.078796m/0.055090m/s).
  Receipt `coopmwm/COMMAND_CHAIN_KINEMATIC_CALLBACK_PROBE.json` discloses unresolved
  roster/lane changes and interpolation fallbacks. Law-derived rows are not genuine
  sensed half-step records. **Full trajectory/GPU qualification remains unpassed;
  mirror stays disabled.** No acceptance, horizon, training or budget change.

## MAIN07:18 clarification and command-chain audit
- **R-u1 completed** after the centre-fairness repair. A48/B1/R1 at
  unix1793576017; B-u2 has20/24 closed episodes and four still collecting.
  Original B/R and grader owners continue with no active STOP or health pause.
- **Per-update physical-null census is live** under report-only owner3746050/
  start59055744, immutable `nullreport1`, amendment `ae657cfe…`, **9 CPU tests
  PASS/zero skips/zero SUMO**. Only the old receipt-owned evidence reporter was
  replaced; it never changes training, exact grades, trust or library inputs.
  Initial report is `warm_live/B_EXACT_VIABILITY_a4449fdbeb2d139a7389dd860a754e6bc074192b912825009c097b0cf9b5384e.json`.
- At unix1793576283: **9 qualified B grades across8 TRAIN worlds**, mean
  **+275.2106veh-s**, whole-world95% CI **[−44.2595,+705.3209]**. Four positive
  grades; the interval includes zero. **No general viability conclusion.**
- B-u1:0/3 physical nulls; B-u2: **3/6 exact-graded plans are confirmed physical
  nulls**. All three exact-zero tapes were checked over their complete420s
  windows; identical sensed rosters/motion/rewards. The newly checked third case
  is world986002/t380, also `origin=centre`, predicted versus own-policy **0.0**,
  versus MAPPO **+113.1611veh-s**. Counts use graded plans as denominator; pending
  plans are not imputed. All zeros/losses remain in the exact evidence.
- Post-safety mirror prototype: **48 CPU tests PASS/zero skips/zero SUMO** in
  immutable `commandcpu3`, including exact float64 agreement with the real
  `CommandOwnership` writer on the four named members and speed-bound/stop
  fixtures. It mirrors one-second `direct_target` and `slowDown` interpolation
  before the inherited half-step speedMode27 cap. **Default remains disabled.**
- CPU replay of the **actual frozen WM CAV callback** on both named observable
  tapes gives **identical plan-versus-own-policy member motion on every active
  half-step callback**, for both original and prototype chains. This independently
  corroborates the already-zero WM own-policy contrast. No GPU process changed.
  The tapes contain only integer-second neighbours: the interpolation-based
  check has absolute speed residuals up to0.208864/0.055090m/s against SUMO,
  unchanged by the prototype. This is **not an exact full-trajectory parity
  pass** and therefore does not authorize a numerical hot-swap. Receipt
  `COMMAND_CHAIN_NULL_CALLBACK_PROBE.json` retains every error and the stated
  limited observation contract. Earlier CPU1/2 harness failures (missing test
  environment/newer GPU runtime) are retained; no physics/inference retries.
- Fresh-session inspection of the **original K8/H420 forecasts** shows both named
  physical-null plans predicted **0.0 saving versus own-policy continuation** and
  `centre_open_loop_gap=0.0`. Their +164.1375/+26.2245veh-s predictions compare
  against whole-road MAPPO. Every returned top-k plan has zero predicted saving
  versus own-policy in these two searches. Exact zero marginal SUMO grades agree
  with that relevant contrast; the larger numbers are not marginal plan errors.
- The registered later MAIN22:39 top-k rule permits mean-positive versus MAPPO;
  it does not require strict improvement versus own-policy. That scientific
  acceptance rule is retained. The distinction has been recorded in the central
  registration log for MAIN. No nulls, losses or quota executions are removed.
- Independently auditing the exact final command chain: the inherited WM already
  applies `alwayscmd.inferred_cap` (SUMO IDM speedMode27/allowed-speed ceiling),
  while plant `CommandOwnership.direct_target` and the one-second `slowDown`
  target need explicit parity checks. No numerical hot-swap is claimed yet.
  `COMMAND_CHAIN_SOURCE_{gpuA,cpuQ}.json`, `COMMAND_CHAIN_LAWS_gpuA.json` and
  `COMMAND_CHAIN_NULL_FIXTURES.json` pin public runtime code/observable TRAIN
  receipts only; collection launches neither SUMO nor WM inference.
- Recovered pipeline-starvation repair was already CPU-qualified and registered.
  It is now deployed as broker3706352/start58959212 at unix1793575314: original
  four assigned searches adopted, stable per-slot centre/plan interleaving at
  boundaries. B/R/grader and GPU owners retained. No horizon/search/budget change.

## MAIN06:58 DIAGNOSIS — two exact-zero grades are physical nulls
- Read-only diagnosis at unix1793574387, receipt
  `coopmwm/ZERO_GRADE_DIAGNOSIS_986051_986070.json`; no new SUMO or WM calls.
  Both are B-u2, t300, two-member nominal300s plans, exact window300–720s.
- **Both selected `origin=centre`**, with all generic perturbation knots exactly0
  and **predicted saving versus current policy0**. Predicted savings versus MAPPO
  were164.1375 and26.2245veh-s. These are policy-centre executions, not evidence
  that a new cooperative perturbation was discovered. The registered top-k rule
  permits mean-positive versus MAPPO; the two contrasts are explicitly distinct.
- Independent original and without-plan tapes have **identical sensed vehicle
  rosters, positions, speeds and lanes across all420 recorded seconds**, identical
  observable fingerprints and every per-second reward. Whole-window totals
  independently agree:56395.37730018104 and49187.82654237546veh-s, respectively.
- The without-plan controller executed **zero plan steps**. Requests really differ:
  max raw-command changes per member are0.21710/0.69096m/s² (986051) and
  0.13606/0.22765m/s² (986070). Some post-safety commands differ too, yet these
  did not produce observable motion or reward changes before the members left.
  Members are present for only2/4s and1/3s after start, respectively; a nominal
  long commitment therefore did not imply a long physical intervention here.
- **Verdict: genuine exact physical-null effects; no grading artifact found.**
  Prefix/CRN checks pass. Zeros remain registered exact trust evidence and enter
  neither positive-only imitation nor the winning-plan library. No outcome-based
  filtering, behavioural threshold, retry or search change was introduced.
- Flags/rate at unix1793574641: **2 confirmed physical-null plans /6 exact-graded
  B plans (33.3%)**, a case-driven descriptive fraction. Both still count as
  executed. The trace's acceleration field is **before CommandOwnership**;
  final ownership speed-bound clipping can erase its command differences.
  Strict final-issued-action equality is therefore not established by these
  tapes. `ZERO_GRADE_NOOP_CLASSIFICATION.json` distinguishes physical-null,
  raw-request equality and pre-ownership command equality instead of treating
  them as interchangeable no-op definitions. No observations were filtered out.

### MAIN guard state compatibility repaired
- `warm_watch state` now resolves current A/V3 and B/R QUOTA owners with boot/PID
  start identity, uses the existing registered health guard for active STOPs,
  exposes health pauses/current supervisor failures, and retains the released old
  SSH STOP separately as historical evidence. **11 CPU tests PASS**; real cpuQ
  state has A48/B1/R0, original B/R alive, active STOPnull, no health pause.
  Registered receipt `coopmwm/WARM_STATE_CURRENT_OWNERS_REPAIR.json`.

## Current continuation snapshot — 06:49:19UTC+8
- A48 complete; B-u2 and R-u1 are collecting concurrently, **24/24 jobs claimed
  for each**. B-u2 has two confirmed plan starts plus eight unresolved potential
  executions toward Nexec10. No active health/engineering STOP at this snapshot.
- Winning-plan seeding is verified in the durable B-u1 receipt: **two library
  entries**, both exact-positive grades from B/seed20261101; covariance buckets
  ordinary|3 and ordinary|4. **R has no search library.**
- Observe-only local46108 (V7) is live with successful repeated status/publication
  cycles; telemetry28752 is retained. Detached training, grading and diagnostic
  owners remain live. All three repaired exact-synergy results below are published.
- Allocated **7,495,040/7,500,000sim-s**,4,960 unallocated; latest charged
  3,331,280 includes retained failures and registered diagnostic repair attempts.
  Original GPU deadline **Nov2 18:07:02UTC+8** is retained. The B-u4–6 viability
  package and a quota-compliant full-run timing estimate are still pending.

## First three complete exact-synergy diagnostics
- Isolated diagnostic owner3589919 has completed the same registered first three
  n+2 groups after the departed-roster repair. Successful original branches are
  adopted; all failed-attempt charges remain. Full receipts000/001/002 are now
  published under `coopmwm/warm_live/ACCEL_EXACT_PLAN_00*.json`.
- **H300 C_G**, in registered sample order: **−868.1767, +0.3667, −2015.2881veh-s**.
  H600: **−6459.3298, +5.8709, −3877.9212veh-s**. These first three diagnostics
  do **not** show consistent positive synergy. No result was dropped or imputed.
- These E2 contrasts use the registered MAPPO background; the exact marginal
  grades above compare against own-policy continuation while retaining other
  recorded plans. Their different contrasts must not be conflated. WM paired
  predictions/variance are queued at diagnostic priority and do not gate PPO.
- The same qualified replay guard is now bound to the still-waiting final
  discovery executor: owner3604109, immutable `finaldiscovery2`, **84 CPU tests
  PASS/zero skips/22.73s/zero SUMO**, amendment `a62810d6…`. Detector/sampling,
  budget and fixed-endpoint/outcome-opening gates are unchanged. Hash-only status
  confirms all673 precompute artifacts remain closed; no final values were read.

### Publication recovery — timing metadata is not an update
- Our added `B_UPDATE_001_TIMING.json` matched an older broad status glob and
  stopped local observer41116 at unix1793571681 on `int('TIMING')`. Remote
  training, grading and diagnostic workers continued; the receipt is retained.
- Numeric update indices now require exactly three digits. **Six local CPU tests
  PASS**, including execution of the actual generated status payload with timing
  metadata present; real remote status and one-shot publication also pass.
  QUOTA owners and Q1 failures are now visible. Adopted-u1 partial-stopwatch ETA
  is suppressed in new formatting. Observation-only V7 restart is next.

### Departed-roster diagnostic repair qualified and registered
- Isolated `diagrepaircpu1`: **242 CPU tests PASS, zero skips,38.96s, zero SUMO**.
  Only recorded counterfactual replay at/after the registered branch time may
  ignore a vehicle absent from current command IDs. Live control and the common
  prefix remain strict; common-roster requests are unchanged. Regression tests
  reproduce the departed-member failure and verify the guarded replay.
- Amendment `af8b16e5…` keeps the **same first12 registered random executed-plan
  sample**. It waits for the original failed pool to settle, adopts every completed
  original branch after world/prefix/all-horizon checks, then permits one explicitly
  registered repaired attempt for each missing branch under a fresh job ID.
  Any further fault stops this diagnostic sidecar; no completed branch is rerun.
- All old charges remain. The entire missing branch set must fit and be charged
  together inside the existing200,000 discovery phase before physics. Full task
  allocation remains7,495,040/7,500,000. No grading/final allocation is borrowed.
  Running training/search/exact grades/trust/imitation/library are unchanged.
- The deferred final-discovery executor needs this same qualified replay guard
  bound before its future physics; its current waiting owner is retained.

## B-u1 complete; quota-aware B-u2 and pipelined R-u1 live
- **B-u1 completed2026-11-02 06:19:04UTC+8.** The retained24-episode wave was
  adopted once. Its registered engineering deviation is3 executed plan-starts
  against Nexec10; no extra episodes. All24 B-u2 jobs are now claimed; ten
  quota-reserved queries were observed, four assigned and six queued. R-u1 is
  collecting independently (six claims at the timing snapshot).
- **Eight actual WM searches, three accepted/executed plans.** The171 logged
  query attempts also include explicit zero-compute capacity/boundary declines.
  Search sizes included2/3/4/6 members; K4/H300 and K8/Hfinal were retained.
  `B_QUERY_VOLUME_001.json` keeps every completed WM call and its effort/timing.
- Three qualified exact marginal grades across **two TRAIN worlds**: two positive,
  one negative. Mean **+862.8525veh-s**; registered world-cluster bootstrap95% CI
  **[+6.6664,+1290.9455]veh-s**. With only two worlds this is very sparse,
  selection-conditional evidence; it does not establish population performance.
  Rank1: two positive grades; rank3: one negative grade. All are ordinary-regime
  MPPI plans, tagged“new” by the existing matcher; a missing matching prototype
  alone is not proof of novel behaviour. Full type/regime/rank/group breakdown:
  `warm_live/B_EXACT_VIABILITY_08bc9c74435ce8b6fd1e10f5a29cfa890d4303ed5c1860dd689915bb0612c110.json`.
- Exact-only trust next beta **0.75097**. Only the two exact-positive plans enter
  imitation/library; the losing plan is retained as trust evidence and is not
  imitated. Uniform AWR weights remain in effect before20 grades. Grade lag
  mean12.18s/max15.69s, no pending/orphan/failed grades at the update snapshot.
  Up to two plans were concurrently active; one overlap re-check, no truncation.
- **Timing correction:** full elapsed from current collection dispatch through
  u1 completion is **3047.69s (50m48s)**. The adopted-wave stopwatch is93.53s;
  its automatic1.22h final ETA is **invalid** because it omits collection.
  `B_UPDATE_001_TIMING.json` preserves both and marks the raw ETA invalid. A
  full-run ETA awaits a quota-compliant update; the original GPU deadline remains
  Nov2 18:07:02UTC+8.
- Continuation owners B3551873/R3551874 started06:17:29UTC+8 from immutable
  `quotacpu4`; broker3514272/grader3427015 and all original GPUs are retained.
  Old transition failure is historical, followed by the registered repair and
  `MAIN0548_HANDOVER_COMPLETE.json`; no active health/engineering STOP.
- Separate n+2 diagnostic twins000/001 failed on a departed recorded vehicle
  (`KeyError mainline.158/.137`). Their charges/failures are retained. This is
  the diagnostic open-loop replay path, not the exact marginal grader; the latter
  passed its prefix/CRN gates. Isolated diagnostic repair is being qualified.

### DIAGNOSIS — completed-wave parent-exit race repaired
- All24 current B-u1 episodes closed once; no queries remained. The handover
  watcher stopped at unix1793571197 because Linux cleared a verified exiting
  supervisor's command line before its `/proc` entry disappeared. No B optimizer
  update or new B/R supervisor had been launched. All results/failure receipts kept.
- Qualified immutable `quotacpu4`: **234 CPU tests PASS, zero skips,39.22s,
  zero SUMO**. Same boot/start-tick identity remains mandatory; an empty command
  line is polled as“exiting” without another signal. Wrong nonempty commands stop.
  Real complete-result-adoption/checkpoint0 Adam/RNG preflight also passes.
- Registered engineering amendment `5693f169…`; original GPU owners, broker and
  grader retained. No traffic rerun, new physics charge or search-design change.

## First qualified exact TRAIN plan grade — descriptive only
- Existing detached reporter published
  `warm_live/B_EXACT_VIABILITY_dccf60491b28433e4500845a95addb7fd0debe0792e7b71f6bef9c65e00b8d5a.json`.
  World986025, B-u1, t320: a3-member,180s ordinary-regime MPPI plan,
  predicted rank1, post-hoc label“new” (no prototype on these members).
- Exact marginal SUMO saving **+6.6664veh-s** over executed duration+120s;
  qualified deterministic prefix/CRN, first successful real grade reused as smoke.
  The corresponding WM prediction was **+240.9659veh-s** at its registered final
  horizon. These are different registered estimands/horizons and are both shown.
- **One plan, one world: CI unavailable.** This is selection-conditional exact
  evidence for that execution, not a B-vs-A learning result or a general viability
  claim. Losing/unresolved/ungraded plans remain subject to the registered rules.
  No sealed outcomes are involved. Further exact evidence publishes automatically.

## MAIN05:48 quota-aware admission — qualified and registered
- **DEPLOYED:** quota broker3514272/start58480523 and detached handover
  supervisor3514691/start58480999 are alive/no errors at unix1793570536.
  Original GPU searches and grader3427015 continue. The watcher automatically
  adopts the closed B-u1 wave, then starts B/R once with the qualified runtime;
  its receipts are exposed by `tools.coopmwm.quota_ops status`.
- At unix1793570600, one formerly waiting episode had resumed SUMO to t648;
  three admitted queries remained pending. Four completed current-release queries
  measured769.08–1194.88s across2/3/4-member groups. One plan start, zero exact
  grades: these are small inference samples, not steady-state throughput or
  evidence of beneficial SUMO cooperation. A finite u1 ETA is not yet established.
- Full recovery adopted the existing owners and completed warm-memory evidence.
  A is at fixed u48. At unix1793570244,20/24 current B episodes were closed;
  the other four had admitted queries at t300/310/320/320. R had no claims.
- **231 CPU tests PASS, zero skips,37.49s, zero SUMO**, immutable
  `coopmwm20261102_quotacpu3`. Tests include24-way serialized quota reservation,
  rejected/trust-rejected query release, no double-counting a confirmed start,
  exclusion of superseded starts, unchanged GPU request identity, existing-claim
  adoption, boundary drain, identity-checked parent controls and one-use dispatch.
- Registered plan `981e496c…`, runtime `bfef6422…`: queue eligible queries while
  executed plus potentially executable in-flight queries is below10; thereafter
  use free-slot-only admission. Probability and queue predicate are recorded.
  R(u) consumes B(u)'s final matching quota independently of B(u+1).
- Only the verified B/R supervisor parents were frozen before optimization.
  Their SUMO children and GPU searches continue. Safe migration finishes every
  already assigned search once, declines further unassigned old-wave queries,
  adopts all24 completed B results and preserves checkpoint0 Adam/RNG. MAIN's
  permitted u1 shortfall will be explicit; no extra traffic episodes are added.
- The failed CPU1 missing-directory receipt is retained; CPU2 passed227 tests.
  Allocation remains **7,495,040 /7,500,000 sim-s**,4,960 unallocated.
  First-update/exact-grade viability evidence is pending at this entry.

## MAIN05:08 nonblocking admission — repaired release live
- Latest05:40:54UTC+8 snapshot:20/24 B episodes closed; the four admitted searches
  are still running, with747.8–750.3s waits. Fresh-release completed searches,
  executed starts and exact grades were all0 at the preceding05:39:12 poll.
  Search-time observations are censored, so no completed-search rate or u1 ETA
  is claimed. Per-episode receipt: `coopmwm/EPISODE_PROGRESS_1793569254.json`.
- Observation restored: local41116 is **observe/publish only**; telemetry28752
  replaces only its receipt-owned predecessor after PID/creation/command checks.
  Six publication/observer tests and four telemetry tests pass. Restart rates
  exclude superseded engineering queries/starts; completed receipts remain intact.
- Opaque final precompute is complete:673jobs/691,800sim-s, MAPPO-reuse gate PASS.
  Outcomes remain closed; no final outcome values or summaries were read. Original
  completion/discovery supervisors wait for fixed A48/B48/R24 endpoints.
- **B/R collection restarted 2026-11-02 05:28:17 UTC+8**, retaining the qualified
  warm-memory/fast-path WM and all four independently qualified GPU processes.
  A has completed its fixed u48. B3432378/R3432791, broker3426521/grader3427015
  are detached and identity-verified; status1793568612 has no active STOP/pause.
- B now launches **24 episodes concurrently**. At1793568595, **20/24 have closed**;
  four are waiting for genuinely admitted searches at simulation t300/300/300/310,
  with current query waits88.8–91.4s. Each episode's progress/wait is recorded in
  `coopmwm/EPISODE_PROGRESS_1793568595.json`; incomplete gzip tails are censored.
- Busy eligible windows execute the policy with query probability0; free capacity
  uses the registered0.95 Bernoulli draw. Simultaneous-client races receive an
  explicit zero-compute capacity decline. Admitted queries wait for their actual
  forecast. The old pre-query capacity-wait loop is removed. K4/H300, K8/Hfinal,
  MPPI population/iterations and Nexec10 are unchanged.
- R is alive but awaits B's final exact matching quota: it has no episode claims.
  Its shared current-policy-centre extraction uses the WM at lower priority;
  its random plans are never WM-scored and have no library/covariance learning.
  Exact matching prevents claiming concurrent R episode collection at this point.
- No completed new B update/exact grades yet. New-release plan acceptance and
  steady-state four-worker throughput are not measured; a finite u1/u48 ETA is
  not established. Per-update wall-time/ETA remains automatic once u1 closes.
  The original service deadline stays Nov2 18:07:02UTC+8.

### DIAGNOSIS — failed handover and charged recovery
- The first admission-broker release read a legacy request after file creation
  but before JSON writing finished. Its engineering STOP fired; **18 attempted
  B episodes became incomplete, six reserved jobs had never been claimed**.
  No B optimizer update had occurred. This was our newly introduced metadata-read
  race, not a WM ranking result. All failures, requests, forecasts and charges
  remain on disk. GPU searches were not restarted or recomputed.
- Fix: new requests publish atomically; a legacy in-progress JSON file gets a
  bounded30s metadata grace, while a persistent malformed request still stops.
  Old assignment/admission receipts are adopted, the first STOP is SHA-released
  under a registered engineering amendment, and any new fault has a fresh STOP_V2.
- **216 CPU tests PASS/zero skips/zero SUMO** cover the full inherited release
  plus admission, partial-request and same-cell recovery tests. A separate launch
  merge error was caught before any new claim/physics; its repair passed
  **19 focused CPU tests** and a real production-binding/u0 Adam/RNG preflight.
- Recovery adopts only the six never-claimed jobs and selects18 different
  registered TRAIN worlds by ascending unused seed within the same cells.
  The exact24-episode,12focus/12other mix remains. **21,600 additional sim-s**
  were registered before collection; no failed world is replayed. Allocation is
  now **7,495,040 /7,500,000**, **4,960 unallocated**; exact-grade/discovery/final
  liabilities are unchanged. Amendments `e70b67aa5e429cdb…` and `a073c07b5daf9904…`.
- Receipts: `MAIN0508_REPAIR_REGISTRATION.json`, `CAPACITY_LAUNCH_PREFLIGHT.json`,
  `WARM_{B,R}_DISPATCH_V6.json`, `CAPACITY_REPAIR_STATUS_1793568520.json`.

## Early exact-plan viability publication armed
- Detached cpuQ reporter3357311 is live, immutable `viabilityreport1`; registered
  amendment `1c66899ddbce10f2cbb99c2117aa3686b528a64a4351b6367028dd23af8da0bb`.
  **5 CPU tests PASS / zero skips / 0.066s / zero SUMO**. It reads only qualified
  B/seed20261101 TRAIN grades, verifies receipt/plan SHA and prefix/CRN gates,
  keeps losing plans, excludes unresolved/ungraded plans without imputation.
- As exact grades arrive, the existing observer publishes the overall exact
  marginal saving plus full by-type/regime/predicted-rank/new-vs-known/group-size
  and update results in immutable `B_EXACT_VIABILITY_<hash>.json` receipts.
  Inference uses10,000 whole-world cluster resamples; with one world the CI is
  unavailable. This report is observational and never enters training or search.
- Status1793566596: reporter alive, no errors, no exact-evidence report yet.
  Exact-C_G twins and B-vs-A TRAIN checks continue under their existing schedules.

## Fresh-resume live snapshot — MAIN04:23 continuation
- A has completed u46; latest registered TRAIN check u44 gives J900 A−MAPPO
  **−952.147 veh-s**, 95% world-bootstrap CI **[−3413.962,1424.270]**, point
  delay reduction1.05%. The interval includes zero; this does not establish a gain.
  B/R are still collecting their first update; the B-vs-A curve is not available.
- Progress1793565861: seven completed TRAIN searches, five executed plan starts;
  no completed exact grades/smoke. The pending grades cannot seed imitation,
  trust, covariance or the winning-plan library. No SUMO plan-benefit claim yet.
- gpuR second worker56877 passed **all8EXACT0** and was hot-added automatically
  at unix1793566045.8605 (04:47:25UTC+8). Original workers/searches continue.
  gpuA617099 remains qualifying; the1vs2 capacity comparison awaits new completed
  searches and keeps qualification/boundary overlaps separate.
- Original service deadline is still **2026-11-02 18:07:02 UTC+8**. It has not
  been reset. Full-run resource continuation needs an outcome-blind registered
  resolution before that deadline; the first completed B-update and post-admission
  throughput are still needed for a credible end-to-end forecast.
- Search-depth diagnostics remain deferred under MAIN23:04 while both GPUs serve
  training/qualification. Warm qualification and03:53UTC+8 B/R restart are retained.

## MAIN04:23 frozen-policy precompute — CPU qualified and registered
- Detector-first final continuation is also registered (`bcbbbacb3a73b49cf141d959c8b7a01d17bd221b1528faedf645414b96eb6c94`),
  detached cpuQ3338335. **6 CPU tests PASS / zero skips / 0.064s / zero SUMO**.
  It waits for all fixed endpoints and final outcome opening/analysis, then uses
  unchanged detectors and observable2–6-member components for complete n+2 exact
  twins within the remaining200,000 discovery reserve. No legacy pair-only gate,
  new detector threshold, partial-group spending or outcome-based selection.
- Dedicated local telemetry observer6216 records TRAIN search throughput/GPU use,
  admission and hash-only final status every300s. Qualification-overlap queries
  are reported separately from one-worker/two-worker operation; unfinished
  searches are disclosed as censored. Existing learning-check observer34356 stays
  live. Both retain the original GPU deadline.
- Detached supervisor **cpuQ3316828/start57982222** is running; status1793565552
  confirms STARTED, zero evaluation jobs while waiting for A48, outcomes closed.
- **22 CPU tests PASS / zero skips / 1.148s / zero SUMO** in the existing cpuQ
  environment, immutable `coopmwm20261102_finalcpu1`. Covers fixed endpoints,
  exact panel/phase liabilities, no partial-job retry, hash-only publication,
  complete MAPPO branch-continuation equivalence and registered branch/Holm tests.
  True-native and MAPPO constructor/state-schema preflights also pass.
- Registered amendment `598cd74d59194ce14bf78c92151061d1bc603d674a29afcbcc9de9f61b7c1946`.
  Once A48 completes, precompute MAPPO/native/A48 episodes and A branches, with
  the one-world/one-branch 600s MAPPO reuse check first. **691,800 sim-s** comes
  entirely from existing final/branch/reuse allocations; total remains
  **7,473,440 / 7,500,000**, 26,560 unallocated.
- Up to24 cpuQ workers use **nice19 and Linux SCHED_IDLE**, inherited by SUMO;
  B/R and exact grading retain priority. New submissions wait while grading is
  active or load/memory/storage guards bind. Existing training/GPU owners persist.
- During training only status, job identities and artifact SHA receipts are exposed.
  No outcome summary or analysis is opened until fixed A48/B48/R24 completion.
  The continuation then evaluates the remaining B/R episodes and B branches once,
  followed by the registered paired episode and comparator-specific Holm analyses.
  Detector-first discovery retains its separately registered reserve.

## MAIN04:19 additional search workers — qualification running
- Detached cpuQ admission monitors3287007/3287221 now wait for all eight exact
  gates and hot-add slotsgpuA@1/gpuR@1 to the original broker automatically.
  Restricted private keys stay on cpuQ; connection-only retries are bounded.
  Second-root full-pool exporters618085/57740 are live (logging only).
  During qualification total GPU utilisation rose to92%/99%, memory32.0/16.5GB;
  this is a health observation, not a search-throughput claim.
- Registered source-identical second search workers before dispatch; original
  GPU workers and all in-flight B searches continue. New owned GPU PIDs
  **gpuA617099 / gpuR56877**; roots `coopmwm20261102_second1` on each host.
- **26 CPU tests PASS / zero skips / 0.785s / zero SUMO** on cpuQ, including
  owner identity, all-eight-exact-gate admission, immutable search/deadline,
  restricted RPC, transport, durable inference and broker hot-add tests.
  A local combined-suite attempt had one missing-torch environment error;
  the complete suite passed in the existing qualified remote environment.
- GPU-A new worker cap42%, widths18/10; GPU-R cap34%, widths4/4 to fit alongside
  the existing model. Both run all eight fresh source-bound exact startup gates.
  Neither is admitted to the broker until READY/all8EXACT0; no inherited gates.
- Same registered MPPI population/iterations, K4/H300, K8/Hfinal, warm history,
  Nexec10 and original GPU deadline. No learned proposer activation or new SUMO.
  Throughput comparison is pending live two-worker searches, not inferred from
  GPU utilisation. Receipts `coopmwm/SECOND_{STAGE,REGISTRATION,DISPATCH}_*.json`.

## MAIN04:13 full-pool logging export — deployed without search interruption
- **DIAGNOSIS:** immutable live mppi_runtime already uses RecordingEvaluator and
  embeds complete screened pools in every B forecast. The missing standalone
  proposer_receipts directory caused the apparent missing-pool report.
- Registered logging-only amendment
  `a2a729c1c243b22f2f2ee84ba66e062c72f2da820afa75f3e68c147ec4fb50aa`.
  Detached exporters gpuA614389/gpuR54031 write the embedded receipts into each
  GPU root's proposer_receipts directory, including rejected searches. Running
  model/search sources and in-flight queries are preserved; no learned seeds or
  warm-start activation. Same original GPU deadline; zero new WM/SUMO calls.
- **4 CPU tests PASS on each GPU host**, zero skips/SUMO. First five real recorded
  forecasts export with byte-identical search output (SHA audits): gpuA3 pools,
  gpuR2 pools; complete screen counts120 or240,128–252 candidate records including
  finalists. Export is idempotent; conflicting/incomplete pools stop the exporter.
- Receipts: `coopmwm/POOL_EXPORT_REGISTRATION.json`, `POOL_EXPORT_STATUS_*`.

## Warm restart live — 2026-11-02 03:53:05 UTC+8
- **TRAINING STARTED B/R**, MAPPOu0, immutable exactcpu5 (204CPU PASS/0skips).
  B3209145, R3209416, grader3208762, broker3208897 detached on cpuQ; original
  A2998262 continues (u35 at restart). Both GPU owners have8source-bound EXACT0
  startup gates; prior warm qualification/E2 labels retained without repetition.
- Registered cap7500000/allocation7473440/free26560; exact-only marginal SUMO
  grades, K4H300 always, K8Hfinal and Nexec10. The library/covariance use only
  same-B-seed exact-positive complete plans; R has neither. Proposer logging only,
  activation requires explicit OWNER approval as well as the offline check.
- **First live timing** (EXACT_PROGRESS_1793563243): gpuA 2-CAV query445.607s,
  8.079queries/inference-hour for ONE query, accepted0/executed0. This is not
  steady-state throughput or an executed-plan ETA. No exact grades yet.
- **First executed plan** (EXACT_PROGRESS_1793563560): gpuR 2-CAV query542.437s,
  6.637queries/inference-hour for ONE query, accepted1/executed1. B world986024,
  t300, ramp.29/ramp.32,90s,120screened candidates/3iterations, predictedrank2,
  posthoc-new, beta1. Predicted saving18.754veh-s is a WM estimate only. The exact
  marginal SUMO grade/smoke awaits the completed episode; no benefit claim yet.
- Source/registration/owners: `coopmwm/EXACT_RUNTIME_REGISTRATION.json`,
  `EXACT_TRAINING_STARTED.json`; observerV2 43408 publishes minute-by-minute.

## MAIN03:03 exact-grader integration — CPU qualified
- Final runtime snapshot **exactcpu4:203PASS,0skips,38.11s,zeroSUMO**. Two live-binding
  regressions additionally verify SHA-authorized budget adoption and the exact
  registered `exact_plan_grading` phase. Grader supervision is exclusive and its
  status receipts are append-only. A remains on its original immutable runtime.
- **201 CPU tests PASS, zero skips, 38.25 s, zero SUMO** on cpuQ, immutable
  `/home/ubuntu/coopmwm20261102_exactcpu3`; failed CPU1/CPU2 receipts retained.
- Integrated landed grader526465de: only exact marginal SUMO grades drive plan
  imitation, trust and B's same-seed winning-plan library/covariance. Own-step PPO,
  all-step critic; positive-plan weights uniform until20grades, then frozen
  first20-grade RMS temperature, clip20, whole positive buffer each update.
- **DIAGNOSIS:** concurrent plan execution overwrote eligible-query metadata when
  a new query was declined while an older plan continued. That hid RNG draws and
  broke exact common-random-number replay. Preserve query metadata; both B/R
  independent-oracle tests now pass bitwise. Grader now replays OTHER groups during
  the graded plan, grades truncated executed portions, and selects imitation
  members by plan-execution records. Unlabelled truncated tails do not seed search.
- First successful real B grade will be reused as the one-plan smoke; no grade
  enters learning before its prefix/CRN gates pass. Grading remains asynchronous;
  no automatic physics replay. CPU capacity: B8/R8/grader8 plus original A4.
- Existing concurrent GPU qualifiers596572/36740 are adopted; their source-bound
  background/recheck gates are still running. B/R remain unstarted at this entry.
- Latest approved budget remains **7,473,440 /7,500,000sim-s**,26,560spare.

## Fresh resume — full brief through MAIN02:43
- Original detached A2998262 remains running: **u21 complete, 496/504 jobs
  closed, 1,592,840 sim-s charged**. B/R/broker unstarted; no active pause/STOP.
- Both original GPU spools are alive/READY with all six startup gates EXACT0;
  the already-published warm qualification is retained without rerunning it.
- Exact-only asynchronous grading and concurrent-plan WM backgrounds remain
  pre-launch integration work. Delegated grader implementation is in progress.
  Latest cap7.5M/allocation7,473,440 and fixed H300 screen are adopted.
- Recovery receipt: `coopmwm/RESUME_MAIN0243_20261102.md`; zero new dispatch.

## Latest budget/control registered before B/R outcomes
- **7,473,440 allocated / 7,500,000 cap / 26,560 unallocated**, including a
  1,000,000 sim-s exact-plan-grade reserve. Complete prefix+horizon precharge
  required per grade; failures retained. Budget/transport contracts: **11 PASS**,
  zero SUMO. Budget amendment `4dc4d875e0730e3c5a71f58e7c2cdc1c5c72259c1d77e129ab991f3aabe1c977`.
- Control amendment `cf82885ccdfd6f842fe4b6722f85a4df4406bd274a4d53ab68873a2a610f4cee`
  fixes K4/H300 screen, K8/Hfinal, auto-continue A/B48,R24 under health guards,
  and nonblocking asynchronous exact grading. Ungraded plans cannot enter
  imitation/trust/library/covariance. No B/R physics authorized by this alone.

## Fresh resume — full brief through MAIN02:36
- Original detached A2998262 is running: **u20 complete, 458/480 jobs closed,
  1,564,040 sim-s charged**. B/R/broker unstarted, no pending request or active pause.
- Original GPU spools both READY, all six startup gates each EXACT0. Completed
  qualification is adopted. Delegated exact grader has not landed (clean e6581008).
- **MAIN02:36 revokes the cheaper screen:** K4/H300 always; K8/Hfinal unchanged.
  Auto-continue A/B48,R24 under health guards; asynchronous exact-only plan grades.
  Cap7,500,000/allocation7,473,440. Receipt `coopmwm/RESUME_MAIN0236_20261102.md`.

## Fresh resume — full brief through MAIN02:34
- Original detached A2998262 is running; **u19 complete, 444/444 jobs closed,
  1,520,840 sim-s charged**. B/R/broker unstarted; no pending request or pause.
- Both original detached GPU spools are now **READY: all six startup gates each
  EXACT0**, including warm batch/single and full/fast trajectory parity. Existing
  qualifications are adopted without replay.
- Latest owner scope: cap7,500,000; every-plan exact SUMO grading; asynchronous
  grading never blocks own-action PPO. Auto-continue A/B48,R24 under health guards;
  conditional cheaper screen after first B update if executed-plan rate<20/hour.
- Delegated exact-grader module has not landed; its integration and prospective
  runtime registration remain. Recovery receipt `coopmwm/RESUME_MAIN0234_20261102.md`.

## Fresh resume — full brief through MAIN02:30
- Preserved HEADe6581008 and existing receipts/claims. A's original detached
  owner2998262 is running; **u18 complete, 420/420 jobs closed, 1492040sim-s charged**.
  B/R and broker remain unstarted; no pending request or active pause/STOP.
- Both original GPU workers are alive and have **five startup gates each EXACT0**;
  their final warm full-fast gate continues on the existing processes.
- Owner cap7500000 and exact grades for every executed plan adopted. Delegated
  grader has not landed; integration/qualification and budget registration remain.
- MAIN02:30 full-resource sizing will be based on measured capacity and existing
  worker identities. No duplicate job, new physics or process-control action here.

## Fresh resume — MAIN02:24 exact-grading amendment adopted
- Complete239-line brief read; preserved HEADbdcddfed and recovered work.
  A's existing detached V3 owner is running, u17 complete;396/396 jobs closed,
  1,463,240 sim-s charged. B/R and the remote broker have no dispatch.
- Both original GPU workers remain alive; warm K4/H120 and K8/H300
  batched/single startup checks are EXACT0. Remaining gates continue unchanged.
- Latest owner cap is7,500,000 sim-s. Exact same-state SUMO grades for EVERY
  executed B/R plan supersede critic-based plan imitation/trust/library evidence.
  Implementation, qualification and prospective registration are pending.
- No new physics or inference job was dispatched during recovery.

## A exact-u16 continuation started — detached remote supervision
- A V3 PID2998262/start57133822, cpuQ boot identity verified; dispatch unix1793557060.492315.
  Exact u16 checkpoint/Adam/RNG retained, now collecting u17; fixed final48 unchanged.
- Fresh status1793557149: processS, no failure or active transportSTOP; 360/372 new
  jobs closed, 1,434,440 sim-s conservatively charged. B/R still unstarted.
- Both persistent GPU workers pass their first warm K4/H120 batched/single EXACT0
  gate; remaining startup gates run on the same owned processes. No duplicate search.

## Durable runtime and health guards — CPU qualified, remote gates running
- **160 CPU tests PASS, zero skips, 8.99s, zero SUMO**. Health pauses are integrated
  into live guided episodes, pre-update checks and paired learning checks. A remains
  independent of B/R health pauses. The original transport STOP is SHA-bound and
  preserved; completed A-u16/check/dev/Adam/RNG were verified, all348 new jobs closed.
- Detached GPU workers dispatched once: gpuA588981/start893707080 and
  gpuR29230/start4218346888. Six exact source-bound startup gates precede READY.
  The unchanged registered warm MPPI search and original deadline are retained.
- Real cpuQ→bothGPU restricted RPC route probes pass; task-only private key stays
  on cpuQ. GPU model processes and forecasts now survive local/SSH disconnects.
- Registered runtime amendment `dc9879ebb44596573c8520a8291809903b737795548afa800129fb513b70e10b`.
  Source `/home/ubuntu/coopmwm20261102_durablecpu2`; V3 learner bindings enforce
  MAIN02:01. A resumes exactu16; B/R await full GPU gates and release registration.
- Retained failed durablecpu1 receipt: its sole failure was a stale test asserting
  the superseded6M cap; latest6.5M boundary is tested in durablecpu2. One SSH reset
  while staging was retried once; no physics or inference was repeated.

## MAIN02:01 learner correction — CPU qualified and registered
- **133 CPU tests PASS, zero skips, 8.86s, zero SUMO** on cpuQ, including 7 new
  AWR semantic tests. Committed plan steps are never policy-scored or ratioed;
  PPO uses standard own-action ratios, including rejected/declined query steps.
- The critic trains on all steps. Only completed plans with positive realized
  discounted SUMO advantage become imitation targets; negative/incomplete plans
  contribute zero imitation. AWR weights=min(20,exp(A/T)), coefficient1, no beta.
- Baseline-only temperature **T=4.822766875237093** learner units, frozen from
  RMS MAPPO TRAIN-check(J900−J600)/10000. No A/B/R outcomes used for this scale.
- Registered amendment `0bb2f157cea8d3f5d65d7e7927f5e8460a873d62b8a7bbe64864434e4aed7ef4`.
  A/B seed1 u24 is a fixed secondary point only if later 24-update replications
  are approved. Finals A/B48,R24 and the approved budget remain unchanged.
- Receipt `coopmwm/MAIN0201_AWR_REGISTRATION.json`; runtime binding/restart pending.

## Fresh resume — MAIN02:01 correction adopted before B/R outcomes
- Preserved HEAD73a42ad1 and all recovered edits. A is at u16/exited; 288/288 warm
  jobs closed, zero pending WM requests, 1,405,640 sim-s charged. B/R unstarted.
- Both original qual4 jobs completed: warm K4/H120, K8/H420 full/fast parity and
  empty-history/cold equality all exactly 0.0. Reused group2/4/6 timings are
  559.567/1029.964/898.275 s per query (6.434/3.495/4.008 plans/hour).
- Latest MAIN02:01 requires own-action-only PPO, positive completed-plan SUMO
  advantage-weighted imitation, critic on all steps. Implementation/CPU tests and
  registration precede B/R restart. Durable broker/health/startup gates remain.
- Approved allocation6,473,440/cap6,500,000; no duplicate job or new SUMO in recovery.

## MAIN01:53 cap confirmation adopted — recovery verified
- Owner-approved cap **6,500,000 sim-s** supersedes earlier 6,000,000 resume figures
  (full brief through line225; shared registry lines6011/6053/6054). The approved
  **6,473,440 allocation leaves26,560**; earlier budget-block receipts are superseded.
- Fresh status: A preserved at **u16/exited**, **288/288 warm jobs closed**, no
  pending requests, **1,405,640 sim-s charged**; B/R have no dispatch.
- Both original qual4 jobs remain identity-verified running with warm K4/H120,
  K8/H420 and empty-history/cold **EXACT0** gates. Group4/6 timing is pending.
- Remaining launch work: durable remote service/broker, health pauses and persistent
  worker startup gates. No duplicate qualifier or new SUMO was launched in recovery.

## Warm GPU gates passed; restart budget-blocked
- Both original qual4 jobs now pass **K4/H120 and K8/H420 warm fast-path parity
  with exactly 0.0 cost error**, plus **empty-history == cold, exactly 0.0**.
  Their boot/PID/start identities were reverified. Group4/6 search timing is still
  running; completed 2-member timing remains **559.567 s/query, 6.434 plans/h**.
- No duplicate qualifier, service or SUMO job was launched. The detached spool
  probe completed normally; its 9 CPU tests and lifecycle evidence are retained.
- Fresh receipt-only re-quote confirms **6,473,440 sim-s** for the registered
  full plan, **473,440 above the explicit 6,000,000 resume cap**. The brief's
  later 6,500,000 cap would leave 26,560. No training/test allocation was changed.
  Receipt: `coopmwm/RESUME_CAP6000000_BLOCK_20261102.json`.
- A remains preserved at u16; B/R have not started. Full warm qualification still
  requires the remaining timing and persistent-service startup gates. MAIN01:38
  health pauses and durable-spool broker integration remain pre-launch work.
  **WARM MEMORY QUALIFIED / TRAINING STARTED B/R are not yet claimed.**

## Remote-worker durability engineering — MAIN01:33
- Added a disk-spool input adapter around the unchanged warm GPU service. A
  detached worker can retain its model process, request claim and completed
  forecast across SSH/Windows disconnects. Search/model settings are unchanged.
- **17 local CPU tests PASS** (9 spool lifecycle + 8 existing transport), zero
  skips/SUMO. Remote cpuQ: **9 PASS**, plus a real detached process that processed
  its request after the launching SSH session ended, with stdin disconnected.
  Three identical deliveries produced exactly one inference claim and forecast.
- Conflicting inputs, changed completed requests, interrupted inference and wrong
  result identities stop without replay. Durable receipts:
  `coopmwm/WARM_SPOOL_QUALIFICATION.json` and `WARM_SPOOL_REGISTRATION.json`.
- This qualifies transport lifecycle only. Source-bound GPU startup parity and
  broker wiring are still required before activation. Both original qual4 GPU
  jobs continue; zero new SUMO. A direct cpuQ→gpuA alias probe failed DNS, so
  remote broker connectivity is not assumed or advertised as working.

## Fresh resume — 2026-11-02 ~01:40
- Complete brief read through MAIN01:38. Fresh status verifies A at u16/exited,
  **288/288 warm jobs closed, zero pending requests, 1,405,640 sim-s charged**;
  B/R have no dispatch. Original SSH-failure STOP and all checkpoints are preserved.
- Both original qual4 GPU jobs are identity-verified running, each with causal
  memory, reused cold/warm E2 and H120 parity receipts. H420/group timing remains
  pending; no replacement qualification or new physics was launched.
- Recovered transport qualification is already registered: **8 CPU tests PASS,
  zero skips/SUMO**, bounded connection-only recovery with identity/receipt checks.
  Remote supervision and MAIN01:38 health pauses remain launch requirements.
- Explicit resume cap **6,000,000** conflicts with the registered full allocation
  **6,473,440**. Dispatch is guarded pending cap/scope resolution; inference-only
  engineering continues. Warm qualification and B/R restart are not yet claimed.

## Fresh recovery — 2026-11-02 ~01:32 host time
- A completed update16 and exited at the durable SSH-failure STOP; all288 warm
  jobs are closed, no pending WM requests, **1405640 sim-s conservatively charged**.
  B/R have no dispatch. Existing checkpoints, failures, labels and charges retained.
- Both GPUs are idle; no local warm bridge/observer remains. GPU A's completed
  group2 timing is reused; its group6 geometry failed. GPU R's qual3 passed H420
  exact parity, then failed H300 screening memory admission. Recovered qual4 edits
  had not been staged or dispatched; they remain unqualified engineering work.
- MAIN01:29 permits bounded retries of SSH connection failures only, with identity
  and receipt reverification. This repair is pending; original STOP stays durable.
- Explicit resume cap6000000 versus latest brief6473440/cap6500000 is unresolved.
  New SUMO dispatch remains gated; inference qualification can proceed without SUMO.

## Pre-outcome MPPI/R integration corrections — CPU8 qualified
- **126 CPU tests PASS, zero skips, 10.01s, zero SUMO**. Found and fixed a real
  bookkeeping bug before B/R updates: adding MPPI expanded the type/regime buckets
  from8 to10, while trust still used8-bucket column offsets. MPPI evidence would
  alias own-policy controls. Dimensions/offsets now derive from the actual bucket
  and regime counts; the regression exercises both MPPI regimes and distinct controls.
- R's held-current-command centre proxy would confound the intended B-R comparison.
  It now uses the **same current-policy trajectory extractor as B** (warm K1/H300,
  current R weights), then samples an unscored generic perturbation: **zero candidate
  WM scores/rankings**, no library/covariance learning. Tests enforce both properties.
  The baseline-only helper is scheduled after B-u4's registered pause to preserve
  primary B capacity. This shared nuisance-centre computation is disclosed WM use;
  R's proposer has no WM candidate knowledge.
- Both corrections are registered before any B/R update; CPU8 per-armV2 source
  bindings supersede the unstarted older runtime. A's preserved CPU6 training is
  unaffected. Full allocation6473440/cap6500000 is unchanged.
- Warm2CAV full-medium search: **559.567s/query, 6.434plans/hour** on gpuA;
  centre open-loop vs K4 closed-loop mean cost gap−52.302veh-s (different future
  draws, not a direct execution-parity test). Group4/6 timings remain pending.
- GPU R H420 failed its65% allocator admission despite7.83GiB free physical memory;
  all failed receipts retained. Registered qual3 uses80%, reuses completed E2/H120,
  and has now passed H420 EXACT0 plus empty-history equality. No new SUMO.

## A continuation active; warm qualification progresses
- Final CPU6: **123 PASS, zero skips, 8.87s, zero SUMO**, including nonzero imagined
  actor vs observable plant actor, warmup-history priming, A-u4→u48 resume and
  exact prospective focus/budget/branch-family contracts. Speed2's frozen-WM
  bridge test requires its GPU-host runtime; the retained CPU5 environment failure
  does not substitute for GPU qualification.
- A resumed once, owned PID2767879/start56464482, from exact preservedu4/Adam/RNG.
  Its u5 update has completed. New root `coopmwm20261102_warm1`; A continues to48,
  checks4/dev16; B/R remain qualification-held. All988040prior charges are retained.
- Storage amendment is nondestructive: new unclaimed output paths use an owned
  RAM root (27.22GB free at admission); existing48paths stay in place. No deletion
  or move. Closed observable/results/claim/close files receive digest-verified GPU
  backups; scenario/truth files are excluded from traversal and backup.
- GPU A has passed warm H120/H420 full-vs-fast parity and empty-history equality.
  GPU R's first fixture used nonconnected CAVs and stopped before search; replacement
 11430 reuses both completed E2 measurements and chooses connected groups. Original
  GPU A group2 timing continues; group4/6 timing is pending. Qualification/restart
  announcements remain gated on complete evidence.

## Integrated warm MPPI — qualification running
- `dd06b2aa`: **117 CPU tests PASS, zero skips, 8.66s, zero SUMO**. Real frozen
  MAPPO parity, observable-history guards, landed MPPI/branch modules and extended
  profile/transfer integration are covered. Failed CPU2/3 receipts are retained.
- Fixed legacy integration-test mocks for group-components; the stochastic long-hold
  test now checks reuse/release of a verified long library plan rather than assuming
  a finite search finds a global optimum for every LAPACK basis. Sampling uses a
  unique Cholesky factor for the positive-definite covariance (same distribution).
- One source-bound inference-only qualifier per GPU: gpuA573068, gpuR9694.
  Existing exact-labelled E2 windows00/01 and genuine archived60s observable
  histories are reused. Causal posterior preparation and cold/warm comparisons
  are running; batch exact parity and group2/4/6 timing follow, no new SUMO.
- **Full scope registered before B/R training**, amendment
  `b03f152870331ac6e56f2b692cb5b02682c2bd3d677259dc2b80aece487a9a21`:
  6473440/6500000 allocation, fixed A/B48,R24, TRAIN4/dev16, discovery200000,
  R branches deferred; four branch moments, comparator-specific Holm families.
  Existing COOPBANDIT lines7–23/COOPPLAN73–102 are SHA-pinned for the owner focus
  hypothesis; prospective TRAIN batches contain12focus+12other existing worlds.
  A completedu1..4/Adam/RNG/check is retained; emphasis starts prospectivelyu5.

## Latest resume — MAIN23:54 budget resolution adopted
- Full206-line brief read, including all ordered addenda. MAIN22:58/23:54 supersede
  the earlier6M cap and the23:36 blocker: **6500000 cap**, TRAIN every4/dev every16,
  discovery reserve200000; R paired branches deferred, R episode test retained.
- Fresh `WARM_RECOVERY_1793548586.json`: original supervisor2477255 and all24 held
  B identities remain T; A accelerationu1..4/check retained; **988040sim-s charged**.
- Expected prospective allocation6473440/free26560; exact quote/registration next.
  No new physics or dispatch; warm/MPPI release qualification remains pending.

## 23:36 — resume verified; latest full plan BLOCKED by budget
- Read the complete 201-line brief through MAIN23:30. Recovered runtime/actor/
  library integration is preserved in `3f70f405`; it remains release-unqualified.
- Fresh identity receipt `WARM_RECOVERY_1793547096.json`: original supervisor
  2477255 and all24 held B worker/SUMO identities remain T. A continuation2611888
  has exited after its registered u4 pause; all four acceleration updates and
  the completed u4 TRAIN check survive. **988040sim-s conservatively charged.**
- Latest scope is A48/B48/R24, 24episodes/update,96-world episode test plus
  four branch moments300/500/700/900 with300s continuation and later R branches.
  Original mechanics/calibration/discovery reserves and every failed/superseded
  reservation remain charged. MAPPO branch reuse is assumed to pass.
- `RESUME_BUDGET_BLOCK_1793547402.json` records all prescribed reductions in order:
  checks4/dev8:7089440; checks4/dev16:6859040; checks8/dev16:**6657440sim-s**.
  The last includes the already-completed A-u4 check14400 rather than refunding it.
- Explicit resume cap**6000000**: shortfall**657440sim-s**. Even the brief's
  6500000cap is short157440. Seed2 was already dropped. A failed MAPPO-reuse
  equivalence check would additionally require115200sim-s, not included above.
- **STOP before new physics**, as required by the budget rule. No new dispatch,
  job duplication, process-control action or final-panel outcome access occurred.
  Existing S-WMSPEED2 GPU experiments are preserved. A is at its existing pause;
  B/R stay held. Full-plan registration/restart needs an owner budget/scope decision.
- End-to-end MPPI/warm decision-quality/restart qualification remains incomplete;
  **WARM MEMORY QUALIFIED and a restart time are not claimed**. Pre-start depth
  diagnostic is deferred under MAIN23:04; existing qualification receipts are retained.

## 22:37 — fresh recovery; A continuation preflight failure identified
- Original B/R hold identities remain stopped; all prior work and charges retained.
- A-only continuation PID2576197 exited before acquiring a claim or launching physics:
  inherited global verification rejected the existing bridge `No module named 'torch'` STOP.
- No A acceleration optimizer update exists; conservative charge remains872840sim-s.
  A source-bound, registered release scoped to this exact bridge STOP is being qualified;
  the original STOP remains durable. Both GPUs are currently empty; S-WMSPEED2 has
  independent CPU qualification processes whose jobs are preserved.

## 22:11 — OWNER22:08 warm-memory hold recorded
- Verified cpuQ supervisor2477255/start55545604 and stopped its shared optimizer
  scheduling; separately stopped only its twelve B workers and their SUMO children
  after ancestry/start-time checks. No acceleration optimizer update exists.
- Twelve A episodes are complete and preserved; remaining A reservations can be
  adopted once by an A-only continuation. B/R cold runs are superseded engineering
  work; **872840sim-s conservatively charged**,51reservations, no refunds.
- Registered engineering amendment
  `082dc02b3c443b83a228ae2000be719eb2a1297e1d9cf4cb23b7895c14b1863e`.
  B/R restart from MAPPOu0 requires C1 observable60s history, learned FULL-v4
  per-driver posteriors, CPU/empty-history tests, both-GPU parity and timed
  cold/warm decision comparison on registered exact-labelled diagnostic states.
- Qualification pending; **WARM MEMORY QUALIFIED is not yet claimed**. No new
  physics or sealed-panel access during recovery.

## 21:49 — fastest qualified capacity admitted for early viability
- Fixed4 two-worker searches are slower than one: gpuA4.6604vs5.3373plans/hour;
  gpuR4.7692vs5.1416. All workers reproduce the complete reference search exactly.
- Capacity amendment SHA `780fa80741ef2f573c79ee8eb377e3c4f07c20c41e5d3554b3a6a0ac1b4ad38b`
  admits the faster already-qualified **one wide34/10 worker/GPU**, aggregate
  **16.6654plans/hour /8132.71imagined futures/hour**, same medium search.
- Verified/stopped only the owned inference supervisor/descendant trees; both
  GPUs now have no compute processes. Completed1/2timings retained; incomplete
  four-worker trials marked deferred, not counted as completed measurements.
- Deferred4/8trials resume as new inference-only jobs **after** the u4 PPO pause,
  retaining completed1/2measurements and source-bound full-search reference.
- **CPU37 79PASS/zero skips/2.26s/zero SUMO**; both30-dependency audits match the
  actual completed wide-probe source. Startup single/batch parity is once per
  persistent worker under its final admitted memory cap.
- Final release amendment SHA `4b0709ee6ce986d050aa57a244c45cf3ca89b137adcda6f65a9fc907d48c70b8`.
  Nexec10, R matching, fixedu4pause,4280840allocation/6Mcap and sealed96 unchanged.

## 21:29 — persistent-worker memory admission finalized
- **CPU36 79PASS/zero skips/2.26s/zero SUMO**. Worker allocator fractions are now
  applied after the inherited WM bootstrap (which installs its own8GiB cap), so
  the registered service limit min(.65,.80/N) is actually enforced. CPU regression
  checks ordering; each persistent worker still qualifies K4/K8 parity once.
- Original CPU33 scale jobs continue unchanged and report actual measured peaks
  under the inherited8GiB cap. No running search was cancelled or duplicated.
- Both30-dependency audits PASS: CPU36 forecast/search/WMTEACH/frozen-cost hashes
  are identical to the original CPU33 scale source. Both SSH environments select
  the frozen FULL-v4 default; no v5 switch was made.
- Latest prospective amendment SHA
  `63ac5e1f9248a658a42ab6ce99e2ee82131f1503141c535bcb035bd5232d8b66`.
  Numerical quota10, u4pause, all budgets and sealed-panel rules unchanged.

## 21:14 — random-arm matching bookkeeping qualified before restart
- **CPU35 78PASS/zero skips/2.28s/zero SUMO**. Corrected the random-arm sampling
  denominator: estimate B's query probability over ex-ante eligible, uncommitted
  decisions, excluding the many1s committed-plan records and unavailable windows.
- R returns to the own-policy route with p0 after its matched regime tickets are
  exhausted; empty proposals cannot manufacture extra query atoms or lose PPO data.
- Amended registration SHA `d3d30eeed563b4844121d6e851bd1d23a5fb542647627baa64cb2eb09f0161ad`;
  quota10/update, medium search, budget, fixedu4pause and sealed rules unchanged.
  The old observer was identity-verified/quiesced before any training binding.
- All original smoke/probes/scaling jobs continue/reuse their completed results.
  GPU model/search source and full WMTEACH dependency hashes must match the CPU35
  service binding; the new qualification does not dispatch duplicate GPU work.

## 21:04 — disk admission restored; original A/B/R smoke PASS
- Opaque archive **3119974400bytes** is durable on gpuA at
  `/root/autodl-tmp/archive_cpuQ_20261101/coopmwm20261101_v2.tar`.
  Source-stream and independently rehashed destination SHA256 both
  `5b3b74cf25f6bc3e09f8be0b0c676436f102925292e5112d9dfe5c4add8dc1bc`.
- Only after that exact match, removed the owner-approved literal superseded v2
  source root; no peer directory or PID touched. cpuQ free **5200531456bytes**.
- Existing smoke2280552 then completed once: **A/B/R PASS**,3600charged sim-s,
  zero new MAPPO simulation; all three metrics match the reused baseline and
  final-state fingerprints match. Total conservative charge stays **815240**.
- CPU34 and numeric viability registration are complete. Acceleration optimizer
  updates remain held solely for the registered two-GPU concurrency qualification.

## 21:00 — full-population batch qualification complete on both GPUs
- Original probe3 completed once: K4/H120width34 and K8/H300width10 single-vs-batch
  parity **0.0veh-s on both GPUs**, including nonzero nongroup/afterplan residuals.
- Same96candidate/3iteration/488future medium search now takes gpuA424.835s,
  **8.4739plans/hour,4135.26imagined futures/hour**; gpuR439.479s,
  **8.1915plans/hour,3997.46imagined futures/hour**. Peak allocation13.666GB each.
- Aggregate16.6654plans/hour is a single-wide-worker measurement, not executed
  plan throughput. Registered fixed4 multiworker scaling is running on both hosts;
  its measured best concurrency will determine service sizing. Search unchanged.
- Existing smoke/charges are preserved. The bounded viability observer38380 polls
  every30s and will publish each qualification/check, adopt the original smoke,
  bind/release the registered restart once, and stop after the four-update package.
  Old superseded observer1116 was identity-verified and stopped before replacement.

## 20:57 — OWNER20:35 viability registration; CPU76 PASS
- Preserved all recovered code, checkpoints, jobs and charges. CPU33 74PASS and
  final **CPU34 76PASS/zero skips/2.25s/zero SUMO** qualify the new source snapshot.
- Fixed a pre-training augmented-behaviour logging bug: a committed plan crossing
  an otherwise eligible10s decision tick retained the policy-route label. It now
  remains an executed plan atom with zero policy-route importance mass at **every**
  committed step. The regression covers the tick boundary and all plan steps.
- Registered amendment SHA `4d8a37137b052224cbe81def95d406307d18a9694e25412f0eaa272c705b18f2`:
  **Nexec10 minimum/update**, medium32x3/elite8/K4H120/top8K8H300 search unchanged,
  A/Bu1..4 first then count/regime/duration-matchedRu1..4, durable owner-review pause.
- At the prior measured11.1728queries/hour,10accepted starts has a **0.895h capacity
  floor** per update if acceptance were100%. Actual acceptance/latency is unknown;
  this is not an execution ETA. Both GPUs stay busy beyond the minimum when possible.
- Concurrency supervisors are owned gpuA542616/gpuR977289; each waits for the
  original probe3 to finish, then measures A1/2/4/8 and R1/2/4 workers. The best
  aggregate medium-search capacity is selected without training outcome input.
- Four-update viability maximum450000sim-s includes the already-reserved3600smoke;
  with811640superseded charges, maximum1261640. Full allocation remains
  **4280840/6000000**, free1719160; sealed96-world test awaits the full run.
- Episode tapes now use digest-checked, lossless gzip storage; identical decoded
  optimizer/exact-twin inputs. Real-SUMO diagnostics run asynchronously and do not
  gate optimizer updates. A logs comparable eligible-group SUMO noise without queries.
- Approved opaque archive transfer is in progress (2.38GB at20:57); no extraction
  or scenario-content inspection. Source removal still requires matching SHA256.

## 20:28 — first measured medium-CEM throughput; full-population batches qualifying
- Complete fixed-width4 search: **gpuA635.987s =5.6605plans/hour**;
  **gpuR653.082s =5.5123plans/hour**. Each searches96candidate slots over3CEM
  iterations, K4/H120screen then top8K8/H300; peak allocated2.978/2.981GB.
- Original receipts report2,626/2,558candidate futures/hour (their numerator
  excludes the24screen-comparator futures). Including all comparator futures,
  each query evaluates488real futures; subsequent receipts count all488.
- This capacity is not sufficient for50executed starts/update within the original
  GPU-time quote: even100%acceptance would require~107service-wall-hours for1200
  primary B starts. This is a capacity lower bound, **not a training ETA**.
- Registered probe3 retains the medium search and full1600roster, widens to34
  screening candidates/10finalists per batch, and qualifies both K4/H120 and
  K8/H300 on nonzero own-policy residuals. New owned PIDs gpuA539643/gpuR974432;
  earlier probes are complete, not duplicated. Numeric quota/re-quote still pending.
- CPU32 **72PASS/zero skips/2.14s/zeroSUMO** adds exact/WM profile-reversion algebra,
  MAPPO background matching, conditional-noise accounting and world-cluster scale
  checks. No guidance/training reward uses a WM number.

## 20:18 — exact GPU plan parity PASS; registered smoke waits on disk
- Both GPUs pass full1600-slot/fixed-four-candidate K8 parity with **0.0veh-s
  maximum error**. Large acceleration profiles change futures/costs by up to
  **1,281.163veh-s** on the registered engineering geometry. Compact-forward
  candidate's318.494veh-s parity failure is retained/rejected.
- CPU31 **71PASS / zero skips /2.15s /zero SUMO** adds full1s fake-plant
  orchestration, actual frozen MAPPO command parity, exact Adam/RNG resume,
  singular-mixture likelihood, per-member committed profiles, nongroup/postcommit
  own-policy continuation and duplicate-CEM-ID regression.
- Medium CEM end-to-end throughput remains pending on both original probe2 PIDs;
  no throughput or training ETA is inferred from the parity timing alone.
- Acceleration smoke registered/bound once; owned cpuQ supervisor2280552,
  start54957271, reserves **3,600sim-s for A/B/R**. Total conservative charge
  **815,240sim-s**. No new SUMO: all three jobs remain unclaimed at disk admission.
- cpuQ free2,105,962,496bytes is below unchanged2,147,483,648-byte guard;
  memory54,077,566,976bytes/load<1 are clear. No deletion/guard weakening/retry.
  Receipt `coopmwm/CPUQ_DISK_ADMISSION_BLOCK.json`. Training remains held.

## 19:56 — acceleration CPU and enlarged-panel registration
- CPU29 **65 PASS / zero skips / zero SUMO**, including real frozen MAPPO
  executor byte parity at1s, AR conditional density, committed acceleration
  profiles, Thompson dual-comparator gate and all96worlds through both seed guards.
- Fresh60metadata worlds **990600–990659**, five/cell, augment original36 to
  **96worlds/eight per cell**. Registry+539safe protocol files and all admitted
  host filename audits have **zero exposures**. cpuP remains excluded.
- Panel SHA256 `d74f685ef75a819248bbe7bb0bef89cf0295037ff3b03b334b13ccec75c40bcc`.
  Primary MAPPO/native/A/B once after fixedu24, R later once; no outcome access yet.
- Registered allocation **4,280,840 /6,000,000sim-s**, including all811,640 old
  charges;1,719,160 remains unallocated. Seed2 dropped under owner19:38:
  retaining both replications with required checks/dev/test costs6,239,240.
- Medium quality-first search registered:32population,3CEM iterations,elite8,
  K4/H120screen then top8K8/H300, shared-future-bootstrap Thompson selection.
  Numerical guidance quota and end-to-end plan throughput still await qualification;
  **all arms remain held**. No acceleration training or efficacy result is claimed.

## 19:41 — fresh-session recovery; OWNER19:38 pending qualification
- Owned A supervisor2146725 boot/start identity verified, state **T**.672/678
  jobs closed; six remaining update20 reservations never CLAIMed. All **811,640
  sim-s** retained, including these reservations. No new physics or dispatch.
- Both GPUs have no task process and no occupied compute memory at inventory.
- Latest requested experiment is common1s acceleration-residual A/B/R from MAPPOu0,
  committed joint acceleration plans with Thompson selection,96 sealed test worlds.
  Old set-point sources/results remain superseded; training stays held pending
  corrected parity, CPU qualification, panel exposure audit, exact re-quote and registration.
- Recovery receipt: `coopmwm/RECOVERY_ACCEL_1938.json`.

## OWNER19:34 — acceleration-space restart HOLD (fresh recovery)
- Verified cpuQ supervisor2146725 boot/start identity and SIGSTOPed that owned
  supervisor. A reached update19; eighteen update20 jobs remain incomplete at the
  snapshot and may drain. All A/B set-point results are superseded engineering runs.
- **811,640sim-s charged/678 reserved jobs**, retained in full. B/R remain held;
  no new physics or duplicate dispatch. Receipt `HOLD_ACCEL_1934.json`.
- New common A/B/R policy must use1s acceleration residuals on frozen MAPPO,
  unchanged safe_accel/CommandOwnership; B needs committed group acceleration plans.
- Probe6 trailing-slot trimming FAILED forecast parity: max1912.604veh-s on gpuA;
  both failed receipts retained. No trimmed implementation is admitted to training.
- Restart awaits corrected batch parity, CPU qualification, measured plan capacity,
  and a prospective exact budget/quota registration including superseded A charges.

## 19:14 — A resumed independently; B/R volume qualification remains held
- CPU25 **55PASS/zero skips/1.72s/zero SUMO**, including private multi-window
  forwarding, exact gate-to-closed-episode matching, R regime quotas, p=0 own-route
  density, restart checkpoint isolation, and never-attempted precharge adoption.
- Registered A-only amendment; **5,005,200 allocated/6,000,000 cap**,994,800 free.
  The extra86,400 retains all original B-u1..3 training charges; no refund.
- Owned old supervisor quiesced after all attempted jobs closed. Twelve reserved
  A-dev jobs had never CLAIMed and were adopted exactly once without recharge.
  Owned A-only supervisor2146725 now continues from existing checkpoints; B/R held.
- B/R quota/mixture/matching code is CPU-qualified, but guided release remains gated
  on measured GPU capacity and prospective numeric registration. No efficacy claim.
- Fused1600-slot two-window K8 parity passes both hosts (max0.0000749veh-s), but
  capacity is31.06/28.65queries/hour. gpuR four-windowK8 OOM is retained.
  A960-slot source-capacity candidate changed paired costs by3.506veh-s and was
  **rejected before training**. Next registered probe retains source draws/capacity,
  trims universally absent trailing slots only during the one-step model forward,
  and restores them before inherited insertion/observer. Original deadline retained.

## OWNER18:54 — guidance-volume HOLD (18:55 recovery)
- Verified owned cpuQ supervisor1945256 boot/start and paused it before B update4.
  A has update4; B updates1–3 are superseded engineering runs (all bytes/charges retained).
  R has not started. The shared supervisor pause temporarily also holds A scheduling.
- Charge307,640sim-s/258 reserved jobs at hold;24 already-reserved A dev jobs may drain.
  No new SUMO dispatch. `HOLD_1854.json` records the outstanding job identities.
- Both existing owned GPU services remain (gpuA520375/gpuR954996); no duplicate service.
  The preserved K8 whole-future batch probe passed max cost error0.00065279veh-s,
  peak allocated2.43GB. Multi-window throughput qualification/gating is next.
- Latest18:54 amendment supersedes the earlier continuation: B/R restart at MAPPO u0
  only after registered execution-count gating, measured batch capacity and matched R counts.

## DIAGNOSIS — 2026-11-01 18:47, OWNER18:38 fast path
**No action-injection/roster/reversion bug found; release the engineering hold.**
Window01 `.139` has target-minus-MAPPO speed **+5.70434m/s**, tau−0.000249751s;
`.154` −0.076919m/s, tau−0.202225s. Window00 `ramp.72` −3.53868m/s,
tau+0.279726s; `.214` +0.477607m/s, tau+0.00685912s. Thus targets are not uniformly
tiny. The decoded requests reach the inherited WM callback, with both members
present/CAV. For `.139`, initial bounded raw action2.60000 vs MAPPO2.59707m/s²
becomes **the same safe action−0.265855m/s²**; all traced12s positions/speeds
remain identical despite differing raw requests. This is safety/plant saturation,
not a missing action. Non-requested CAVs run the frozen deterministic MAPPO
bridge, not native/IDM; a real CPU callback test verifies standalone MAPPO equality
and nonzero speed/position response to an unsaturated braking set-point.
Inherited available O1/O3/O4 proposals DO alter `.139` trajectories (maximum12s
position difference8.569/8.228/8.228m); H60 single-draw costs are distinct from
policy/MAPPO (O3/O4 legitimately coincide in this geometry). O2 is unavailable
here and retains its availability mask. Window00 also responds (ramp12s maximum
position difference1.146m), with no inherited upstream option available.
`ACTION_DELTAS_1838.json`, `ACTION_PROBE2_RESULT_00/01.json`, `DIAGNOSIS_1838.json`
retain decoded requests, every0.5s accel/speed/position and before/after-safety
values. **CPU WM regression1PASS/0skips, CUDA disabled, SUMO forbidden.** CPU22's
wrong-runtime fixture failure and first focused fixture failure are preserved;
the qualified WM library fixture passes in `CPU_WM_ACTION_RESPONSE2_1838.json`.
Exact twins and original K8 results are reused; these diagnostic rerolls add0SUMO.
The late exact-SUMO divergence (H60/H120 near0; H300/H600 +1699/+5723veh-s)
is compatible with amplification of small execution differences, but two windows
do not establish an empirical variance reduction or WM accuracy/scale CI.
Conditional WM variance alone cannot support the paper's noise-reduction claim.

### Suggestion-rate diagnosis and prospective release
Original B-u1 used1/900 dense and1/3600 ordinary query probability, before
availability was known: **1 query,0 accepted,0 executed**. Both GPUs' real available
H300 K8+canary calls took213.67/234.79s (aggregate32.18queries/hour), explaining
the throughput constraint. Outcome-independent numeric fit raises the shared
B/R dense rate to**0.0022449262636300934**, ordinary**0.0005612315659075234**
(2.02× historical rates). It uses309dense+219ordinary eligible B-u1 windows and
274.0465s original48-episode span scaled4→12workers; no efficacy-based sampling.
Numeric registration/binding completed; **18:48 same owned supervisor resumed**
(`RELEASE_1838.json`, identity checked, stateS), preserving original u1
optimizer/RNG/checkpoints and existing GPU queues. Exact allocation remains4,918,800/6,000,000sim-s,
1,081,200unallocated; current charge83,200. All final-test/checkpoint rules remain.

## OWNER18:34 — engineering HOLD (fresh-session recovery)
- Verified supervisor PID1945256 boot/start identity; paused it with SIGSTOP at the
  update-2 sampling gate. B/R are held at update1/not-started respectively. The
  interleaved supervisor also pauses A until safe arm-independent continuation exists.
- All70 jobs are closed;83,200sim-s charged. No new SUMO, duplicated job or checkpoint
  replacement. Existing GPU inference/exact receipts are retained for wiring diagnosis.
- `coopmwm/HOLD_1834.json` records the owned pause. No numeric sampling binding will
  release learning before action-response regression and a registered repair amendment.

## 2026-11-01 18:31 — source-prior ABI qualified; primary checkpoints preserved
- CPU21: **50PASS/zero skips/1.71s/zero SUMO**, including non300-clock integer source
  sufficient statistics, real frozen MAPPO parity and per-future proposal parity.
- First availability-qualified H300 probe exposed a pre-rollout integer-count API fault;
  failed dual1 services ended. Both remote inventories verified no surviving task GPU process.
- Fresh immutable dual2 services: gpuA ownedPID520375/gpuR ownedPID954996; original
  GPU deadline retained. They now qualify/timestamp an actual available K8 H300 proposal.
- Source encoding uses rounded FIT-prior rate×current-clock pseudo-counts, NOT measured
  source requests. This current-clock pseudo-exposure is weaker than retaining the
  donor posterior exposure; no causal/full-history source-conditioning claim is made.
- All u1 weights/Adam/RNG, original queue, completed forecasts and exact twins are kept.
  **83,200sim-s charged /70closedjobs**, no physical duplicate; u2 remains gated on
  numeric registered sampling. Allocated4,918,800/cap6,000,000 is unchanged.
- Owned30s publisherPID1116 will publish and commit E2 windows/full agreement CIs,
  every UPDATE, every4-update TRAIN check, dev/final and STOP receipts for up to24h.

## 2026-11-01 18:10 — owner17:55 dual-GPU amendment active
- Registered unchanged **4,918,800 allocated /6,000,000 cap**,1,081,200 unallocated.
- CPU18/19: **47PASS/zero skips/zero SUMO**; finalCPU19 2.87s. Design remains70lines.
- Both immutable FULL-v4 services are ready: gpuA ownedPID516897, gpuR ownedPID951313;
  each has an8GiB cap. Original failed bridge/startup receipts remain untouched.
- Same registered first-window K8 diagnostic is running on both GPUs for cross-host
  parity; next is one H300 proposal timing per host and outcome-independent sampling fit.
- cpuQ will adopt all62finished jobs/74,400charged;12nice19training workers plus2E2
  workers, shared serialized accounting. E2 now runs beside PPO and cannot gate learning.
- A/B/R final checkpoints, primary-first once-only test and remaining seed2 plan unchanged.

### 18:15 — first-update ETA
- Both primaryu1 checkpoints/UPDATE receipts exist; supervisor ownedPID1945256 is alive.
  The62closed jobs were adopted without new SUMO;74,400sim-s remains charged.
- B u1 query1/accepted0/executed0; no learned-cooperation efficacy claim.
- Initial48-episode rollout closes span254.33s with4workers;12worker CPU scaling plus
  PPO/check overhead gives a provisional **1–3h after numeric u2 sampling release**
  for both primaryu24 checkpoints. This is a planning range, not measured end-to-end
  throughput; update after u2 elapsed time and WM queue latency are available.
- u2 cannot launch physics until the outcome-independent sampling fit is numerically
  registered and hash-bound. First dual-GPU E2 forecast is still running; no agreement result yet.

## 2026-11-01 18:01 — fresh recovery and frozen reducer repair
- Recovered all62 closed jobs /74,400sim-s; no live task supervisor or GPU service.
- Prior GPU startup failed because `inspect` resolved an exec-created reducer against
  the producer's shared synthetic filename. Explicit qualified22cc32f6 reducer source
  is now frozen and bytecode-matched before adapting only the horizon tuple.
- CPU17: **44 PASS / zero skips /1.72s / zero SUMO**, including the actual frozen
  reducer, incorrect-source rejection and MAPPO command parity. Failure receipts retained.
- Owner17:55 dual-GPU service/sampling amendment is next; no finished physics repeated.

## 2026-11-01 17:48 — recovery and random-guide amendment
- Registered owner17:42 arm R before any UPDATE/E2 outcome; registration and exact quote
  preserved in `BUDGET_AMENDMENT_1742.json` and commit `bb9d6fdb`.
- Revised allocation **4,918,800 / 6,000,000 sim-s**, unallocated1,081,200; A/B seeds1+2,
  R primary only, fixedu24; seed3 dropped. Primary A/B test still precedes R/replications.
- Recovered owned supervisor1877910; let all **62/62 jobs close** before controlled adoption.
  **74,400 sim-s charged** includes smoke,12TRAIN-check controls and48initial A/B episodes.
- At the E2 forecast wait, verified boot/start identity and no active/queued physics;
  stopped only that supervisor. Original jobs/checkpoints/charge ledger/WM queue survive.
  Receipt `QUIESCE_1742.json`; existing GPU bridge retained, no duplicate dispatch.
- R implementation/CPU qualification underway; no training update or R physics yet.

### 17:53 — arm R CPU qualification PASS
- CPU15/16 both **43 PASS / zero skips / zero SUMO**; final CPU16 took1.66s.
- Covers actual frozen MAPPO parity; uniform available inherited options without WM;
  mixture mass/imitation/SUMO-only reward; per-type/regime trust; safe completed-job reuse;
  fixed R final-test one-use gate, reused primary controls and paired B-R bootstrap.
- Final70-line design SHA `02879dd48ed26b466575891df920c1cbfad9da848a841c26af56c26777466cea`.
- Noise diagnostics report C_G, each D_i, sum D_i and D_G variance from the original K8
  draws. No superseded reward-combination weight is used; no inference is repeated.
- GPUA snapshot17:50:1,359MiB/49,140MiB,32%util; original service waiting on admission,
  WMPLAN priority retained. E2/first-update results and a measured ETA are still pending.
- Exact registered allocation remains4,918,800/6M; `READINESS_1742.json` binds CPU/design.

## 2026-11-01 17:41 — full-episode zero-update smoke PASS
- Owned cpuQ smoke PID1872644 finished; both registered1200s episodes closed.
- A and B metrics exactly equal reused bitwise MAPPO baseline at J600/J900;
  A/B final full-state fingerprints equal. **2,400 sim-s charged**, new MAPPO simulation0.
- Collisions0, teleports0 in both episodes; this is mechanics/parity, not learned performance.
- One-use CPU dispatch, GPU bridge PID41056 and immutable bindings are preserved;
  no training update yet. Primary paired training dispatch follows this receipt.
- E2 agreement/noise is diagnostic only; it will use preselected initial TRAIN windows.

## 17:27 refinement — CPU qualification and prospective registration
- Recovered at17:29: no task runtime/jobs/services on cpuQ/gpuA/gpuR; no duplicate dispatch.
- CPU13: **40 PASS, zero skips, 1.36 s, zero SUMO**; CPU11/12 failed receipts preserved.
- Eight option-type x observable-regime trust buckets use normal-means empirical Bayes:
  between-bucket variance and bootstrap uncertainty determine shrinkage toward pooled evidence.
- Independent random query precedes winner-type selection; execute with that bucket's beta.
  PPO uses the logged query measure, including rejected/declined query atoms; reward remains SUMO-only.
- Observable merge-dense rule projects the union of LEVERPROBE's four recorded upper-quartile
  bands, with ramp and cross-lane CAV pair; no true demand/share input or new behavioural cutoff.
- Trust evidence is a critic-based selected-type residual contrast against own actions in the
  same regime. It is **not** an exact type-specific causal effect; nonquery winner labels are unobserved.
- Cap remains **6,000,000 sim-s**, allocated5,738,400, unallocated261,600. Three paired training
  seeds and all primary/replication final-test liabilities remain protected.
- Source/design hashes are frozen in CPU13; exact amendment: `coopmwm/BUDGET_AMENDMENT_1727.json`.
- CPU14 repeats40PASS/0skips/1.35s; deployment now copies the tested immutable snapshot,
  avoiding concurrently changing peer tooling. Initial v1 staging failure was before binding/SUMO.
- Bound cpuQ `/home/ubuntu/coopmwm20261101_v2` and gpuA `/root/coopmwm20261101_v2`;
  source/cache/runtime/checkpoint hashes verified;4nice19workers, task simulation still0 at binding.
- Next: one full-episode zero-update parity smoke per arm, then primary training/E2.

## 17:12 redesign — implementation CPU-verified before physics
- Arm B now uses WM-guided exploration and **SUMO-only team reward/advantage**.
- FULL-v4 K8 H300 compares inherited O1–O4 set-point proposals with policy/MAPPO.
- Logged random routing permits exact importance correction on the augmented
  behaviour mixture, including deterministic proposal atoms; imitation weight is beta.
- Beta starts at1 and uses a monotone, scale-free world-bootstrap SUMO-evidence rule.
- Primary A/B episodes share an interleaved admitted worker pool; fixed primary u24
  test precedes replications, with MAPPO/native final-panel controls reused.
- CPU9: **36 PASS, zero skips, 1.32 s, zero SUMO**, including actual frozen MAPPO
  parity, proposal-mixture mass, SUMO reward isolation, adaptive trust and leakage checks.
- CPU8's failed syntax/roundoff checks are preserved; fixed before CPU9 and physics.
- E2 agreement/variance remains diagnostic; historical reward-credit gates below are superseded.
- Registration/runtime deployment is next; no task simulation or update dispatched yet.

## 17:07 amendment — registered before physics
- MAIN excludes unreachable cpuP; existing registry/protocol and cpuQ/gpuA/gpuR audits qualify.
- Recovery found no task runtime, training jobs or GPU services on admitted hosts; SUMO remains zero.
- Stopped the superseded metadata-only waiter by verified owned PID46700.
- Primary A/B train first in parallel. Run primary four-arm final panel once at both u24 checkpoints;
  train replications afterwards and evaluate A/B once per seed with MAPPO/native reused.
- Added replication-test liability172,800: allocated **5,738,400 / 6,000,000**, unallocated261,600.
- `BUDGET_AMENDMENT_1707.json` and the shared registry record this prospective amendment.
- Historical entries below describe the prior gate/order; the 17:07 amendment supersedes them.

## Recovery and prospective registration — 2026-11-01
- Preserved recovered work in `0a3df807`; previous CPU5 sources matched 108 staged files.
- No COOPMWM runtime or processes existed at recovery; task SUMO/training updates: zero.
- CPU6: **29 PASS, zero skips, 1.35 s, zero SUMO**, with simulator subprocess creation forbidden.
- Includes real speed-only MAPPO169 `d9042cc5` command parity; WM/exact credit fixture algebra;
  fixed ramp; observable-only/no-leakage checks; dual seed guards and three-seed/final-panel roles.
- Exact prospective plan appended to the shared `REGISTRATION.md` before any outcome.
- Three independent training seeds fit: 24 PPO updates per arm per seed; 4,147,200 training sim-s.
- Total allocated 5,565,600 / 6,000,000 sim-s; 434,400 remains unallocated, not automatically spent.
- Final-test liability **172,800 sim-s precharged**: fresh 989600–989635, 36 worlds, four arms.
- Panel canonical SHA256: `69b8a300eab45fb8208bd4bc66cefd32c6ebd16c71c23b5aac8567d41ee3c182`.
- Final checkpoints fixed to primary-seed update24; additional seeds are replications, never selectors.
- Panel stays unopened for outcomes until all training ends; one complete evaluation, no reruns.
- Registry/protocol audit: 538 safe files, no conflicts. Available execution-host filename audits:
  cpuQ 337,849; gpuA 895,554; gpuR 328,139; all zero exposures.
- Additional cpuP audit could not complete: SSH connection closed. cpuP is not admitted;
  full cross-host exposure qualification is pending resolution before physics.

## Registered credit/noise check
- FULL-v4 `22cc32f6`, original WMTEACH K8 paired-future executor, H60/120/300/600.
- Twelve randomly selected initial TRAIN windows, one per cell, selected before credit outcomes.
- Report D_i/C_G sign agreement, world-bootstrap 95% CIs and learned credit scale immediately.
- Every horizon must pass both sign lower bounds >0.5; primary H300 if admitted.
- No passing horizon stops B. Calibration/weights remain frozen for all three training seeds.
- Compare per-window K8 variance of the same joint-action team-return advantage with paired replay
  credit variance, in veh-s². A fixed critic baseline does not change the conditional variance.
- CRN pairing may cancel shared hidden-driver/arrival noise; this is a hypothesis to measure,
  not an observed noise reduction. Conditional WM variance is not empirical SUMO variance.
- Report actual across-TRAIN PPO advantage variance separately; no noise/performance result yet.

## Next checks
- Immutable final runtime/CPU/source binding and complete exposure admission.
- One full-episode smoke per arm, against the reused bitwise MAPPO cache.
- WM/exact gate, then every-four-update TRAIN-cell checks and first registered dev check.

## CPU7 — final workflow implementation
- **30 PASS, zero skips, 1.28 s, zero SUMO**; receipt `coopmwm/CPU_coopmwm20261101_cpu7.json`.
- Runner now namespaces all three training-seed checkpoint/job/check receipts and fixes
  seed0/update24 for the final test. The final-test gate requires all training completions;
  missing B finals produce an explicit incomplete test, never a fallback checkpoint.
- Native reference reuses the genuine native plant. Final paired bootstrap supports all36 worlds.
- H600 uses the original producer with H600/1600 entrant capacity and four indexed K2 batches
  assembled into complete paired K8. Draw identities/executor/reducer are retained; actual GPU
  memory/forecast parity remains a pre-inference check, not established by structural CPU tests.
- Actual team-only GAE variance is logged separately from conditional WM credit noise.
- cpuP SSH still closes at the second bounded metadata-only probe. No runtime binding,
  owned training/GPU service, smoke, credit agreement or learning update has been dispatched.

## Live publication — 2026-11-01T10:15:00.034770+00:00
- Charged 74,400/6,000,000 sim-s; closed jobs 62/62; pending WM 1.

### S0_A_UPDATE_001.json
- A update1, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.66493e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_001.json`.

### S0_B_UPDATE_001.json
- B update1, seed20261101; SUMO-only reward; queries/accepted/executed 1/0/0; next pooled beta 1.0.
- Per-bucket trust: O1|merge-dense=1; O1|ordinary=1; O2|merge-dense=1; O2|ordinary=1; O3|merge-dense=1; O3|ordinary=1; O4|merge-dense=1; O4|ordinary=1.
- TRAIN GAE variance 4.69666e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_B_UPDATE_001.json`.

## Live publication — 2026-11-01T10:20:31.236333+00:00
- Charged 79,200/6,000,000 sim-s; closed jobs 66/66; pending WM 7.

### CREDIT_WINDOW_00.json
- E2 window world986000 t600: exact quartet complete.
- H120 conditional WM K8 variance: joint-team advantage 801277, paired C_G 0 veh-s²; not empirical SUMO noise.
- H300 conditional WM K8 variance: joint-team advantage 9.8999e+06, paired C_G 0 veh-s²; not empirical SUMO noise.
- H60 conditional WM K8 variance: joint-team advantage 86872.7, paired C_G 0 veh-s²; not empirical SUMO noise.
- H600 conditional WM K8 variance: joint-team advantage 5.84757e+07, paired C_G 0 veh-s²; not empirical SUMO noise.

## Live publication — 2026-11-01T10:29:51.501844+00:00
- Charged 83,200/6,000,000 sim-s; closed jobs 70/70; pending WM 7.

### CREDIT_WINDOW_01.json
- E2 window world986007 t400: exact quartet complete.
- H120 conditional WM K8 variance: joint-team advantage 265240, paired C_G 0 veh-s²; not empirical SUMO noise.
- H120 WM/exact C_G 0/-7.09977e-07; D_i [0.0, -0.001434922218322754]/[-0.07634644157951698, 7.099770300555974e-07]; zero C_G is a tie, not sign evidence.
- H300 conditional WM K8 variance: joint-team advantage 3.5524e+06, paired C_G 0 veh-s²; not empirical SUMO noise.
- H300 WM/exact C_G 0/-7.52552e-07; D_i [0.0, -0.001447141170501709]/[1698.600333714603, 7.525522960349917e-07]; zero C_G is a tie, not sign evidence.
- H60 conditional WM K8 variance: joint-team advantage 31878, paired C_G 0 veh-s²; not empirical SUMO noise.
- H60 WM/exact C_G 0/9.01442e-07; D_i [0.0, -0.0004132091999053955]/[0.11916434977320023, -9.014420356834307e-07]; zero C_G is a tie, not sign evidence.
- H600 conditional WM K8 variance: joint-team advantage 4.016e+07, paired C_G 0 veh-s²; not empirical SUMO noise.
- H600 WM/exact C_G 0/-7.52567e-07; D_i [0.0, -0.001447141170501709]/[5723.486351683328, 7.525668479502201e-07]; zero C_G is a tie, not sign evidence.

## Live publication — 2026-11-01T10:49:25.085948+00:00
- Charged 145,600/6,000,000 sim-s; closed jobs 98/122; pending WM 1.

### CREDIT_WINDOW_02.json
- E2 window world986012 t600: exact quartet complete.
- H120 conditional WM K8 variance: joint-team advantage 1.02526e+06, paired C_G 1.81899e-12 veh-s²; not empirical SUMO noise.
- H120 WM/exact C_G 4.76837e-07/0; D_i [0.0046305060386657715, 0.0]/[-0.008384575048694387, 0.0]; zero C_G is a tie, not sign evidence.
- H300 conditional WM K8 variance: joint-team advantage 1.10728e+07, paired C_G 1.81899e-12 veh-s²; not empirical SUMO noise.
- H300 WM/exact C_G 4.76837e-07/0; D_i [0.004627645015716553, 0.0]/[-0.008384783359360881, 0.0]; zero C_G is a tie, not sign evidence.
- H60 conditional WM K8 variance: joint-team advantage 96105.6, paired C_G 1.81899e-12 veh-s²; not empirical SUMO noise.
- H60 WM/exact C_G 4.76837e-07/0; D_i [0.004109740257263184, 0.0]/[0.00042830735037568957, 0.0]; zero C_G is a tie, not sign evidence.
- H600 conditional WM K8 variance: joint-team advantage 7.85469e+07, paired C_G 1.81899e-12 veh-s²; not empirical SUMO noise.
- H600 WM/exact C_G 4.76837e-07/0; D_i [0.004627645015716553, 0.0]/[6.07445183393429, 0.0]; zero C_G is a tie, not sign evidence.

## Live publication — 2026-11-01T10:50:27.621315+00:00
- Charged 145,600/6,000,000 sim-s; closed jobs 122/122; pending WM 1.

### S0_A_UPDATE_002.json
- A update2, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.70697e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_002.json`.

## Live publication — 2026-11-01T10:51:00.705174+00:00
- Charged 203,200/6,000,000 sim-s; closed jobs 128/170; pending WM 1.

### S0_B_UPDATE_002.json
- B update2, seed20261101; SUMO-only reward; queries/accepted/executed 2/0/0; next pooled beta 1.0.
- Per-bucket trust: O1|merge-dense=1; O1|ordinary=1; O2|merge-dense=1; O2|ordinary=1; O3|merge-dense=1; O3|ordinary=1; O4|merge-dense=1; O4|ordinary=1.
- TRAIN GAE variance 4.77848e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_B_UPDATE_002.json`.

## Live publication — 2026-11-01T10:52:34.609181+00:00
- Charged 203,200/6,000,000 sim-s; closed jobs 170/170; pending WM 1.

### S0_A_UPDATE_003.json
- A update3, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.429e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_003.json`.

## Live publication — 2026-11-01T10:53:06.635788+00:00
- Charged 260,800/6,000,000 sim-s; closed jobs 172/218; pending WM 1.

### S0_B_UPDATE_003.json
- B update3, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 1.0.
- Per-bucket trust: O1|merge-dense=1; O1|ordinary=1; O2|merge-dense=1; O2|ordinary=1; O3|merge-dense=1; O3|ordinary=1; O4|merge-dense=1; O4|ordinary=1.
- TRAIN GAE variance 4.50958e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_B_UPDATE_003.json`.

## Live publication — 2026-11-01T10:54:40.121508+00:00
- Charged 264,440/6,000,000 sim-s; closed jobs 222/222; pending WM 1.

### CREDIT_WINDOW_03.json
- E2 window world986019 t310: exact quartet complete.
- H120 conditional WM K8 variance: joint-team advantage 966381, paired C_G 7.21099e-12 veh-s²; not empirical SUMO noise.
- H120 WM/exact C_G -1.19209e-06/0; D_i [8.344650268554688e-07, 4.76837158203125e-07]/[0.0, 0.0]; zero C_G is a tie, not sign evidence.
- H300 conditional WM K8 variance: joint-team advantage 1.51445e+07, paired C_G 7.21099e-12 veh-s²; not empirical SUMO noise.
- H300 WM/exact C_G -1.19209e-06/0; D_i [8.344650268554688e-07, 4.76837158203125e-07]/[0.0, 0.0]; zero C_G is a tie, not sign evidence.
- H60 conditional WM K8 variance: joint-team advantage 55320.5, paired C_G 7.21099e-12 veh-s²; not empirical SUMO noise.
- H60 WM/exact C_G -1.19209e-06/0; D_i [8.344650268554688e-07, 4.76837158203125e-07]/[0.0, 0.0]; zero C_G is a tie, not sign evidence.
- H600 conditional WM K8 variance: joint-team advantage 1.31309e+08, paired C_G 7.21099e-12 veh-s²; not empirical SUMO noise.
- H600 WM/exact C_G -1.19209e-06/0; D_i [8.344650268554688e-07, 4.76837158203125e-07]/[0.0, 0.0]; zero C_G is a tie, not sign evidence.

## Live publication — 2026-11-01T10:55:12.537480+00:00
- Charged 278,840/6,000,000 sim-s; closed jobs 225/234; pending WM 1.

### S0_A_UPDATE_004.json
- A update4, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 5.03716e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_004.json`.

## Live publication — 2026-11-01T11:15:29.022889+00:00
- Charged 336,440/6,000,000 sim-s; closed jobs 258/282; pending WM 0.

### S0_A_CHECK_004.json
- A_minus_MAPPO: {"bootstrap":"10000 whole-world paired resamples; RNG20261101; negative delay is better","learned_minus":"mappo","metrics":{"600_collisions":{"ci95":[0.0,0.0],"mean":0.0},"600_hard_brake":{"ci95":[-2.7510416666666657,7.5],"mean":2.2083333333333335},"600_team_delay_vs":{"ci95":[-1249.1101470572473,3372.3958383866448],"mean":1203.3873745997926},"600_teleports":{"ci95":[0.0,0.0],"mean":0.0},"600_ttc_below_3_vs":{"ci95":[-11.709375,97.91770833333332],"mean":44.0},"900_collisions":{"ci95":[0.0,0.0],"mean":0.0},"900_hard_brake":{"ci95":[-7.5,4.583333333333333],"mean":-1.875},"900_team_delay_vs":{"ci95":[-1868.5913824377792,4332.582430582843],"mean":1372.7795884038114},"900_teleports":{"ci95":[0.0,0.0],"mean":0.0},"900_ttc_below_3_vs":{"ci95":[-55.21041666666667,109.5],"mean":28.833333333333332}},"worlds":12}

### S0_A_DEV_004.json
- A_minus_MAPPO: {"bootstrap":"10000 whole-world paired resamples; RNG20261101; negative delay is better","learned_minus":"mappo","metrics":{"600_collisions":{"ci95":[0.0,0.0],"mean":0.0},"600_hard_brake":{"ci95":[-4.645833333333333,2.625],"mean":-0.875},"600_team_delay_vs":{"ci95":[-2008.6695155343664,953.7011788286733],"mean":-526.333169436575},"600_teleports":{"ci95":[0.0,0.0],"mean":0.0},"600_ttc_below_3_vs":{"ci95":[-90.14583333333333,9.750520833333326],"mean":-40.354166666666664},"900_collisions":{"ci95":[0.0,0.0],"mean":0.0},"900_hard_brake":{"ci95":[-4.083333333333333,4.416666666666667],"mean":0.2916666666666667},"900_team_delay_vs":{"ci95":[-2525.7345253020353,2240.0123362352256],"mean":-184.73109833748217},"900_teleports":{"ci95":[0.0,0.0],"mean":0.0},"900_ttc_below_3_vs":{"ci95":[-107.08333333333333,23.75104166666665],"mean":-43.104166666666664}},"worlds":24}

## Live publication — 2026-11-01T11:16:31.748572+00:00
- Charged 365,240/6,000,000 sim-s; closed jobs 282/306; pending WM 0.

### S0_A_UPDATE_005.json
- A update5, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.64449e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_005.json`.

## Live publication — 2026-11-01T11:18:05.307340+00:00
- Charged 394,040/6,000,000 sim-s; closed jobs 307/330; pending WM 0.

### S0_A_UPDATE_006.json
- A update6, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.62821e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_006.json`.

## Live publication — 2026-11-01T11:19:08.132519+00:00
- Charged 422,840/6,000,000 sim-s; closed jobs 330/354; pending WM 0.

### S0_A_UPDATE_007.json
- A update7, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.70447e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_007.json`.

## Live publication — 2026-11-01T11:20:10.926285+00:00
- Charged 437,240/6,000,000 sim-s; closed jobs 354/366; pending WM 0.

### S0_A_UPDATE_008.json
- A update8, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.90423e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_008.json`.

## Live publication — 2026-11-01T11:21:13.725511+00:00
- Charged 466,040/6,000,000 sim-s; closed jobs 368/390; pending WM 0.

### S0_A_CHECK_008.json
- A_minus_MAPPO: {"bootstrap":"10000 whole-world paired resamples; RNG20261101; negative delay is better","learned_minus":"mappo","metrics":{"600_collisions":{"ci95":[0.0,0.0],"mean":0.0},"600_hard_brake":{"ci95":[-1.9166666666666667,8.0],"mean":3.2083333333333335},"600_team_delay_vs":{"ci95":[-2134.4230452651373,2514.0648813466746],"mean":366.9245178487299},"600_teleports":{"ci95":[0.0,0.0],"mean":0.0},"600_ttc_below_3_vs":{"ci95":[-39.376041666666666,84.3770833333333],"mean":23.458333333333332},"900_collisions":{"ci95":[0.0,0.0],"mean":0.0},"900_hard_brake":{"ci95":[-6.375,8.0],"mean":1.1666666666666667},"900_team_delay_vs":{"ci95":[-2310.9142790688265,4254.126301910809],"mean":1194.8769365963044},"900_teleports":{"ci95":[0.0,0.0],"mean":0.0},"900_ttc_below_3_vs":{"ci95":[-61.833333333333336,125.25],"mean":36.375}},"worlds":12}

## Live publication — 2026-11-01T11:22:16.980124+00:00
- Charged 494,840/6,000,000 sim-s; closed jobs 390/414; pending WM 0.

### S0_A_UPDATE_009.json
- A update9, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.50355e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_009.json`.

## Live publication — 2026-11-01T11:23:19.724350+00:00
- Charged 523,640/6,000,000 sim-s; closed jobs 414/438; pending WM 0.

### S0_A_UPDATE_010.json
- A update10, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 5.15375e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_010.json`.

## Live publication — 2026-11-01T11:24:56.588634+00:00
- Charged 552,440/6,000,000 sim-s; closed jobs 442/462; pending WM 0.

### S0_A_UPDATE_011.json
- A update11, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 5.05183e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_011.json`.

## Live publication — 2026-11-01T11:25:59.568360+00:00
- Charged 566,840/6,000,000 sim-s; closed jobs 462/474; pending WM 0.

### S0_A_UPDATE_012.json
- A update12, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.69977e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_012.json`.

## Live publication — 2026-11-01T11:26:31.329316+00:00
- Charged 595,640/6,000,000 sim-s; closed jobs 474/498; pending WM 0.

### S0_A_CHECK_012.json
- A_minus_MAPPO: {"bootstrap":"10000 whole-world paired resamples; RNG20261101; negative delay is better","learned_minus":"mappo","metrics":{"600_collisions":{"ci95":[0.0,0.0],"mean":0.0},"600_hard_brake":{"ci95":[-3.2083333333333335,4.583333333333333],"mean":0.375},"600_team_delay_vs":{"ci95":[-661.2338917042277,2949.737879335951],"mean":1282.3524654967944},"600_teleports":{"ci95":[0.0,0.0],"mean":0.0},"600_ttc_below_3_vs":{"ci95":[6.833333333333333,98.54270833333332],"mean":50.083333333333336},"900_collisions":{"ci95":[0.0,0.0],"mean":0.0},"900_hard_brake":{"ci95":[-6.208333333333333,4.6260416666666515],"mean":-1.125},"900_team_delay_vs":{"ci95":[-41.61367904337277,4606.31203827328],"mean":2363.594791228685},"900_teleports":{"ci95":[0.0,0.0],"mean":0.0},"900_ttc_below_3_vs":{"ci95":[-31.46145833333333,117.54166666666667],"mean":42.791666666666664}},"worlds":12}

## Live publication — 2026-11-01T11:28:05.670888+00:00
- Charged 624,440/6,000,000 sim-s; closed jobs 499/522; pending WM 0.

### S0_A_UPDATE_013.json
- A update13, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 5.2456e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_013.json`.

## Live publication — 2026-11-01T11:29:09.162719+00:00
- Charged 653,240/6,000,000 sim-s; closed jobs 522/546; pending WM 0.

### S0_A_UPDATE_014.json
- A update14, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 5.09244e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_014.json`.

## Live publication — 2026-11-01T11:30:12.024242+00:00
- Charged 682,040/6,000,000 sim-s; closed jobs 546/570; pending WM 0.

### S0_A_UPDATE_015.json
- A update15, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.82592e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_015.json`.

## Live publication — 2026-11-01T11:31:47.424047+00:00
- Charged 696,440/6,000,000 sim-s; closed jobs 570/582; pending WM 0.

### S0_A_UPDATE_016.json
- A update16, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 5.15962e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_016.json`.

## Live publication — 2026-11-01T11:32:19.359481+00:00
- Charged 725,240/6,000,000 sim-s; closed jobs 582/606; pending WM 0.

### S0_A_CHECK_016.json
- A_minus_MAPPO: {"bootstrap":"10000 whole-world paired resamples; RNG20261101; negative delay is better","learned_minus":"mappo","metrics":{"600_collisions":{"ci95":[0.0,0.0],"mean":0.0},"600_hard_brake":{"ci95":[0.041666666666666664,9.25],"mean":4.125},"600_team_delay_vs":{"ci95":[-78.65468623970786,3459.082650857894],"mean":1686.0698296149446},"600_teleports":{"ci95":[0.0,0.0],"mean":0.0},"600_ttc_below_3_vs":{"ci95":[-10.626041666666666,101.25],"mean":43.291666666666664},"900_collisions":{"ci95":[0.0,0.0],"mean":0.0},"900_hard_brake":{"ci95":[-3.3333333333333335,7.2093749999999845],"mean":1.25},"900_team_delay_vs":{"ci95":[331.8738344183094,6951.7146298653915],"mean":3346.086848854511},"900_teleports":{"ci95":[0.0,0.0],"mean":0.0},"900_ttc_below_3_vs":{"ci95":[-27.0,99.08437499999998],"mean":29.291666666666668}},"worlds":12}

## Live publication — 2026-11-01T11:33:53.736644+00:00
- Charged 754,040/6,000,000 sim-s; closed jobs 612/630; pending WM 0.

### S0_A_UPDATE_017.json
- A update17, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 5.19562e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_017.json`.

## Live publication — 2026-11-01T11:34:58.261814+00:00
- Charged 782,840/6,000,000 sim-s; closed jobs 631/654; pending WM 0.

### S0_A_UPDATE_018.json
- A update18, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.75876e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_018.json`.

## Live publication — 2026-11-01T11:36:01.920162+00:00
- Charged 811,640/6,000,000 sim-s; closed jobs 654/678; pending WM 0.

### S0_A_UPDATE_019.json
- A update19, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 5.30598e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_019.json`.

## Acceleration viability — 2026-11-01T12:59:38.429998+00:00

### SCALE_gpuA_FIXED4_K4_H120.json
- SCALE_gpuA_FIXED4_K4_H120.json: {"H":120,"K":4,"max_cost_error":0.0,"pass_":true,"width":4}

## Acceleration viability — 2026-11-01T12:59:40.615520+00:00

### SCALE_gpuR_FIXED4_K4_H120.json
- SCALE_gpuR_FIXED4_K4_H120.json: {"H":120,"K":4,"max_cost_error":0.0,"pass_":true,"width":4}

## Acceleration viability — 2026-11-01T13:01:21.640167+00:00

### SCALE_gpuA_FIXED4_K8_H300.json
- SCALE_gpuA_FIXED4_K8_H300.json: {"H":300,"K":8,"max_cost_error":0.0,"pass_":true,"width":4}

## Acceleration viability — 2026-11-01T13:02:35.562746+00:00

### SCALE_gpuR_FIXED4_K8_H300.json
- SCALE_gpuR_FIXED4_K8_H300.json: {"H":300,"K":8,"max_cost_error":0.0,"pass_":true,"width":4}

## Acceleration viability — 2026-11-01T13:03:17.429398+00:00

### ARCHIVE2035_COMPLETE.json
- ARCHIVE2035_COMPLETE.json: {"archive":{"destination":{"bytes":3119974400,"path":"/root/autodl-tmp/archive_cpuQ_20261101/coopmwm20261101_v2.tar","sha256":"5b3b74cf25f6bc3e09f8be0b0c676436f102925292e5112d9dfe5c4add8dc1bc"},"new_sumo_sim_s":0,"source":"/home/ubuntu/coopmwm20261101_v2","source_stream":{"bytes":3119974400,"sha256":"5b3b74cf25f6bc3e09f8be0b0c676436f102925292e5112d9dfe5c4add8dc1bc"},"transport":"opaque tar stream via existing cpuQ/gpuA SSH keys; no extraction/content inspection","unix":1793538181.6069808,"verified":true},"disk_free":5200531456,"new_sumo_sim_s":0,"source_removed":true,"unix":1793538196.7923505}

## Acceleration viability — 2026-11-01T13:04:31.100800+00:00

### SMOKE.json
- SMOKE.json: {"arms":["A","B","R"],"new_MAPPO_sim_s":0,"pass_":true,"sim_s":3600,"timestamp_utc":"2026-11-01T13:04:03.970616+00:00"}

## Acceleration viability — 2026-11-01T13:16:39.800621+00:00

### SCALE_gpuA_CONCURRENCY_1.json
- SCALE_gpuA_CONCURRENCY_1.json: {"final_width":4,"host":"gpuA","imagined_futures":488,"imagined_futures_per_hour":2604.6138264975925,"new_sumo_sim_s":0,"pass_":true,"peak_worker_allocated_bytes":3486505472,"plans_per_hour":5.337323414954083,"screen_width":4,"source_bound_full_search_parity":true,"unix":1793538723.0724921,"wall_s":674.4953828193247,"workers":1}

## Acceleration viability — 2026-11-01T13:16:41.556657+00:00

### SCALE_gpuR_CONCURRENCY_1.json
- SCALE_gpuR_CONCURRENCY_1.json: {"final_width":4,"host":"gpuR","imagined_futures":488,"imagined_futures_per_hour":2509.0930862791824,"new_sumo_sim_s":0,"pass_":true,"peak_worker_allocated_bytes":3431991808,"plans_per_hour":5.141584193195046,"screen_width":4,"source_bound_full_search_parity":true,"unix":1793538828.7716107,"wall_s":700.1733054891229,"workers":1}

## Acceleration viability — 2026-11-01T13:38:28.216313+00:00

### SCALE_gpuA_CONCURRENCY_2.json
- SCALE_gpuA_CONCURRENCY_2.json: {"final_width":4,"host":"gpuA","imagined_futures":488,"imagined_futures_per_hour":2274.278974195577,"new_sumo_sim_s":0,"pass_":true,"peak_worker_allocated_bytes":2970765312,"plans_per_hour":4.660407734007331,"screen_width":4,"source_bound_full_search_parity":true,"unix":1793540275.9209359,"wall_s":1544.9292016793042,"workers":2}

## Acceleration viability — 2026-11-01T13:39:43.755261+00:00

### SCALE_gpuR_CONCURRENCY_2.json
- SCALE_gpuR_CONCURRENCY_2.json: {"final_width":4,"host":"gpuR","imagined_futures":488,"imagined_futures_per_hour":2327.3484840710466,"new_sumo_sim_s":0,"pass_":true,"peak_worker_allocated_bytes":2973430272,"plans_per_hour":4.769156729653784,"screen_width":4,"source_bound_full_search_parity":true,"unix":1793540350.4746027,"wall_s":1509.7008565962315,"workers":2}

## Acceleration viability — 2026-11-01T13:53:29.877407+00:00

### TRAINING_STARTED.json
- TRAINING_STARTED.json: {"N_exec":10,"charged_sim_s":815240,"eta_primary_hours":{"acceptance_unknown":true,"capacity_lower_bound":14.401095344890471},"search":{"elite":8,"final_H":300,"final_K":8,"finalists":8,"iterations":3,"knot_s":10,"max_duration_s":60,"population":32,"screen_H":120,"screen_K":4},"timestamp_utc":"2026-11-01T13:52:59.130144+00:00"}

## Acceleration viability — 2026-11-01T13:53:31.481315+00:00

### VIABILITY2035v4_TRAIN_DISPATCH.json
- VIABILITY2035v4_TRAIN_DISPATCH.json: {"boot_id":"b8bc1fcb-c656-4e1e-a43d-4543118c834e","pid":2477255,"root":"/home/ubuntu/coopmwm20261101_accel1","source":"/home/ubuntu/coopmwm20261101_acceltrain1","start_ticks":"55545604","unix":1793541178.313232}

### VIABILITY2035v4_BRIDGE_DISPATCH.json
- VIABILITY2035v4_BRIDGE_DISPATCH.json: {"pid":8760,"root":"/home/ubuntu/coopmwm20261101_accel1","unix":1793541179.244051}

## Acceleration viability — 2026-11-01T14:12:01.672729+00:00

### STOP.json
- STOP.json: {"error":"No module named 'torch'","source":"owned plan bridge; no retry"}

## Warm live publication — 2026-11-01T17:00:51.040221+00:00
- Conservatively charged 1247240/6500000 sim-s.

### Warm continuation — A_CHECK_008.json
- A_minus_MAPPO: J600 Δ=-848.095 [-2162.494,342.381] veh-s; delay reduction 1.67%; J900 Δ=-1649.653 [-3661.632,97.574] veh-s; delay reduction 1.82%. Negative Δ is better; 95% whole-world bootstrap.
- Reporting-only check, never checkpoint selection. Full safety/per-cell results in the linked immutable receipt.
- Receipt: `coopmwm/warm_live/A_CHECK_008.json`.
<!-- warm-receipt:A_CHECK_008.json:eb8c205599f3cc583a3e406e345b901b9a33a37108489a9dd565f4620b556a0c -->

### Warm continuation — A_CHECK_PRESERVED_004.json
- A_minus_MAPPO: J600 Δ=-773.934 [-1947.182,457.898] veh-s; delay reduction 1.53%; J900 Δ=-510.387 [-3105.952,2048.194] veh-s; delay reduction 0.563%. Negative Δ is better; 95% whole-world bootstrap.
- Reporting-only check, never checkpoint selection. Full safety/per-cell results in the linked immutable receipt.
- Receipt: `coopmwm/warm_live/A_CHECK_PRESERVED_004.json`.
<!-- warm-receipt:A_CHECK_PRESERVED_004.json:b4cbad934935fbda76a371ccdfc1d01a6550d900966c5133625ad2f19129ae13 -->

### Warm continuation — A_UPDATE_001.json
- A u1: queries/accepted/executed 0/0/0; wall 134.1s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_001.json`.
<!-- warm-receipt:A_UPDATE_001.json:24d95608dcdf11b8d8b7c76077244e7a0b7ac61fadf6d1c4f001b629467651f3 -->

### Warm continuation — A_UPDATE_002.json
- A u2: queries/accepted/executed 0/0/0; wall 155.8s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_002.json`.
<!-- warm-receipt:A_UPDATE_002.json:006c8ee1844a0092edd168b1aadd96f16f81cfceee7a9d0ce03a72f52323118d -->

### Warm continuation — A_UPDATE_003.json
- A u3: queries/accepted/executed 0/0/0; wall 152.5s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_003.json`.
<!-- warm-receipt:A_UPDATE_003.json:ed00946e9842cbeebb3c2092eb06e36a9e179435cb410108fb95ea09f1b4ffea -->

### Warm continuation — A_UPDATE_004.json
- A u4: queries/accepted/executed 0/0/0; wall 157.1s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_004.json`.
<!-- warm-receipt:A_UPDATE_004.json:37d336c70f60602d83805dc1f5fc7b8c0af09bb6e886e2acfcec60b76e37e2f5 -->

### Warm continuation — A_UPDATE_005.json
- A u5: queries/accepted/executed 0/0/0; wall 250.7s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_005.json`.
<!-- warm-receipt:A_UPDATE_005.json:8174be3a23eeec1e951365bde6cdd37f2c3dc03d72f18de51b42934bc4f68e10 -->

### Warm continuation — A_UPDATE_006.json
- A u6: queries/accepted/executed 0/0/0; wall 246s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_006.json`.
<!-- warm-receipt:A_UPDATE_006.json:bebd0f521303cdb2645fa1a39ee5abd45b112b9b18de9d5484aad9a78a897ddb -->

### Warm continuation — A_UPDATE_007.json
- A u7: queries/accepted/executed 0/0/0; wall 253.7s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_007.json`.
<!-- warm-receipt:A_UPDATE_007.json:7316866ad0b28601331c125c78a6d267a3ed4cc5602bfe947edc7f254ef2bef2 -->

### Warm continuation — A_UPDATE_008.json
- A u8: queries/accepted/executed 0/0/0; wall 245.5s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_008.json`.
<!-- warm-receipt:A_UPDATE_008.json:d9589c46c3ab8071cec49000a99cfa4ab062bd11d2bfe1f7aabd5e85d410b115 -->

### Warm continuation — A_UPDATE_009.json
- A u9: queries/accepted/executed 0/0/0; wall 251.8s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_009.json`.
<!-- warm-receipt:A_UPDATE_009.json:33e89021e39010f991e3fefeafc7c350dfbc50f603447068d1634b844c074f48 -->

### Warm continuation — A_UPDATE_010.json
- A u10: queries/accepted/executed 0/0/0; wall 249.4s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_010.json`.
<!-- warm-receipt:A_UPDATE_010.json:080a82141559b3b60345d9dc7542879cf497e9f99ca5143b5c044742e2786255 -->

### Warm continuation — A_UPDATE_011.json
- A u11: queries/accepted/executed 0/0/0; wall 253.1s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_011.json`.
<!-- warm-receipt:A_UPDATE_011.json:2e34651f2d23fece5d3fbfbe26836cfed594d6c7d0218621035fce69d38387a2 -->

### Warm continuation — A_UPDATE_012.json
- A u12: queries/accepted/executed 0/0/0; wall 243.3s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_012.json`.
<!-- warm-receipt:A_UPDATE_012.json:22c41d0678b2ed5c5cdc360b6ad56d22e5a44a4ad9162d41a26e116022aa5832 -->

### Warm continuation — A_VIABILITY_005.json
- A u5: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_005.json`.
<!-- warm-receipt:A_VIABILITY_005.json:cd3a18299aa7d3cac4696ec86690593a81a108ccca5c0f6bd89a8091b07a0f05 -->

### Warm continuation — A_VIABILITY_006.json
- A u6: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_006.json`.
<!-- warm-receipt:A_VIABILITY_006.json:c65ede6084ab6a01656299be6dc22433a77496e5671639bc7eb73ccb42c64081 -->

### Warm continuation — A_VIABILITY_007.json
- A u7: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_007.json`.
<!-- warm-receipt:A_VIABILITY_007.json:2fe83674e9d0ca83c05920f8fd4ca1a982a2fb0eb30ddf72566e18cf0c497784 -->

### Warm continuation — A_VIABILITY_008.json
- A u8: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_008.json`.
<!-- warm-receipt:A_VIABILITY_008.json:33b24bc0a55f0ed2f332fad49b5a812c538dddd285b5ff05a26a40d176774303 -->

### Warm continuation — A_VIABILITY_009.json
- A u9: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_009.json`.
<!-- warm-receipt:A_VIABILITY_009.json:23683b33d083a4112bd1f2f6d7facd7e275503b1a42cf33c7d35ca47e8a51986 -->

### Warm continuation — A_VIABILITY_010.json
- A u10: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_010.json`.
<!-- warm-receipt:A_VIABILITY_010.json:7402480142ca38aa6ffd10b7ce36de98e25d65076d9526a45aaae864057386d0 -->

### Warm continuation — A_VIABILITY_011.json
- A u11: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_011.json`.
<!-- warm-receipt:A_VIABILITY_011.json:76386e6fd9d98b244c714e85d92dd86cbaa67ef6ff4151f37dfa078850c4ae41 -->

### Warm continuation — A_VIABILITY_012.json
- A u12: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_012.json`.
<!-- warm-receipt:A_VIABILITY_012.json:c6d86d9ca9a568571a6a745cec2327befae4134addc91acc220545984270a8fc -->

## Warm memory diagnostic — 2026-11-01T17:15:47.304839+00:00
- 2 CAVs, gpuA: **559.567s/query, 6.434plans/hour**, 120 candidates/584 futures; screen `{'H': 300, 'K': 4}`, final `{'H': 300, 'K': 8}`; open-loop centre cost gap -52.302veh-s (future0 centre vs K4 closed-loop comparison).
- cold E2 query times: `[85.88906374387443, 89.05008675158024]`s.
- cold H60: D_i sign `{'sign_agreement': 0.5, 'n': 2, 'ties': 2, 'ci95': [0.0, 1.0], 'wilson_ci95': [0.09453120573423074, 0.9054687942657693]}`; C_G sign `{'sign_agreement': None, 'n': 0, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 1.0]}`.
- cold H120: D_i sign `{'sign_agreement': 0.0, 'n': 2, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 0.6576197724933469]}`; C_G sign `{'sign_agreement': None, 'n': 0, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 1.0]}`.
- cold H300: D_i sign `{'sign_agreement': 0.5, 'n': 2, 'ties': 2, 'ci95': [0.0, 1.0], 'wilson_ci95': [0.09453120573423074, 0.9054687942657693]}`; C_G sign `{'sign_agreement': None, 'n': 0, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 1.0]}`.
- cold H600: D_i sign `{'sign_agreement': 0.5, 'n': 2, 'ties': 2, 'ci95': [0.0, 1.0], 'wilson_ci95': [0.09453120573423074, 0.9054687942657693]}`; C_G sign `{'sign_agreement': None, 'n': 0, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 1.0]}`.
- cold exact-labelled chosen-action savings: `[{'world': 986000, 'horizons': {'120': {'chosen': 'joint', 'exact_labelled_saving': -4.001776687800884e-11, 'predicted_saving': 4.868022978305817}, '300': {'chosen': 'joint', 'exact_labelled_saving': 6.548361852765083e-11, 'predicted_saving': 5.789575934410095}, '60': {'chosen': 'joint', 'exact_labelled_saving': -3.992681740783155e-09, 'predicted_saving': 1.8523305654525757}, '600': {'chosen': 'joint', 'exact_labelled_saving': 6.548361852765083e-11, 'predicted_saving': 86.09206318855286}}}, {'world': 986007, 'horizons': {'120': {'chosen': 'all', 'exact_labelled_saving': 0.0, 'predicted_saving': 0.0}, '300': {'chosen': 'all', 'exact_labelled_saving': 0.0, 'predicted_saving': 0.0}, '60': {'chosen': 'all', 'exact_labelled_saving': 0.0, 'predicted_saving': 0.0}, '600': {'chosen': 'all', 'exact_labelled_saving': 0.0, 'predicted_saving': 0.0}}}]`.
- warm E2 query times: `[86.10405902191997, 84.69402858614922]`s.
- warm H60: D_i sign `{'sign_agreement': 0.5, 'n': 2, 'ties': 2, 'ci95': [0.0, 1.0], 'wilson_ci95': [0.09453120573423074, 0.9054687942657693]}`; C_G sign `{'sign_agreement': None, 'n': 0, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 1.0]}`.
- warm H120: D_i sign `{'sign_agreement': 1.0, 'n': 2, 'ties': 2, 'ci95': [1.0, 1.0], 'wilson_ci95': [0.34238022750665303, 1.0]}`; C_G sign `{'sign_agreement': None, 'n': 0, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 1.0]}`.
- warm H300: D_i sign `{'sign_agreement': 0.5, 'n': 2, 'ties': 2, 'ci95': [0.0, 1.0], 'wilson_ci95': [0.09453120573423074, 0.9054687942657693]}`; C_G sign `{'sign_agreement': None, 'n': 0, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 1.0]}`.
- warm H600: D_i sign `{'sign_agreement': 0.5, 'n': 2, 'ties': 2, 'ci95': [0.0, 1.0], 'wilson_ci95': [0.09453120573423074, 0.9054687942657693]}`; C_G sign `{'sign_agreement': None, 'n': 0, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 1.0]}`.
- warm exact-labelled chosen-action savings: `[{'world': 986000, 'horizons': {'120': {'chosen': 'all', 'exact_labelled_saving': 0.0, 'predicted_saving': 0.0}, '300': {'chosen': 'all', 'exact_labelled_saving': 0.0, 'predicted_saving': 0.0}, '60': {'chosen': 'all', 'exact_labelled_saving': 0.0, 'predicted_saving': 0.0}, '600': {'chosen': 'all', 'exact_labelled_saving': 0.0, 'predicted_saving': 0.0}}}, {'world': 986007, 'horizons': {'120': {'chosen': 'joint', 'exact_labelled_saving': -0.07634644157951698, 'predicted_saving': 1.1913661062717438}, '300': {'chosen': 'joint', 'exact_labelled_saving': 1698.600333714603, 'predicted_saving': 1.368777722120285}, '60': {'chosen': 'joint', 'exact_labelled_saving': 0.11916434977320023, 'predicted_saving': 0.5568383932113647}, '600': {'chosen': 'joint', 'exact_labelled_saving': 5723.486351683328, 'predicted_saving': 1.3687926232814789}}}]`.
- two registered near-parity set-point windows, not novel MPPI plan labels; ties separate, two-world CIs uninformative; no established warm decision-quality improvement. All original exact SUMO labels reused; qualification adds0SUMO.
- Immutable complete receipt: `coopmwm/WARM_MEMORY_DIAGNOSTIC_1ebc1e2bcdb28522.json`.

## Warm live publication — 2026-11-01T17:29:19.610782+00:00
- Conservatively charged 1405640/6500000 sim-s.

### Warm continuation — A_CHECK_012.json
- A_minus_MAPPO: J600 Δ=-11.643 [-1397.644,1337.034] veh-s; delay reduction 0.023%; J900 Δ=847.227 [-1205.672,2963.049] veh-s; delay reduction -0.934%. Negative Δ is better; 95% whole-world bootstrap.
- Reporting-only check, never checkpoint selection. Full safety/per-cell results in the linked immutable receipt.
- Receipt: `coopmwm/warm_live/A_CHECK_012.json`.
<!-- warm-receipt:A_CHECK_012.json:52d09325ea3147d53eaf57e69bc272b8731f0af16f0491d36f8d9f7d9d8ec36b -->

### Warm continuation — A_CHECK_016.json
- A_minus_MAPPO: J600 Δ=-416.815 [-1811.617,918.673] veh-s; delay reduction 0.822%; J900 Δ=-417.853 [-2879.691,1910.328] veh-s; delay reduction 0.461%. Negative Δ is better; 95% whole-world bootstrap.
- Reporting-only check, never checkpoint selection. Full safety/per-cell results in the linked immutable receipt.
- Receipt: `coopmwm/warm_live/A_CHECK_016.json`.
<!-- warm-receipt:A_CHECK_016.json:365cf59c3c1d444837024b8d8577df9b4a302025a78c9ad9d8cc5c2bcea9bda8 -->

### Warm continuation — A_DEV_016.json
- A_minus_MAPPO: J600 Δ=53.955 [-622.870,744.081] veh-s; delay reduction -0.0974%; J900 Δ=804.504 [-533.098,2296.395] veh-s; delay reduction -0.819%. Negative Δ is better; 95% whole-world bootstrap.
- Reporting-only check, never checkpoint selection. Full safety/per-cell results in the linked immutable receipt.
- Receipt: `coopmwm/warm_live/A_DEV_016.json`.
<!-- warm-receipt:A_DEV_016.json:c4ea062098cb65d11b2a485b0998f4f56be9435203f52d2fd64dc06875faa047 -->

### Warm continuation — A_UPDATE_013.json
- A u13: queries/accepted/executed 0/0/0; wall 255.2s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_013.json`.
<!-- warm-receipt:A_UPDATE_013.json:784fa7060ed4a902f0232359099b865b5a59ae7c242bcba78324295f7c3e00f6 -->

### Warm continuation — A_UPDATE_014.json
- A u14: queries/accepted/executed 0/0/0; wall 249.4s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_014.json`.
<!-- warm-receipt:A_UPDATE_014.json:ea13d124960eb7322c2402f767a59b105d5595982b0ce3e4aba028507a7b5f36 -->

### Warm continuation — A_UPDATE_015.json
- A u15: queries/accepted/executed 0/0/0; wall 254.2s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_015.json`.
<!-- warm-receipt:A_UPDATE_015.json:4f35c675667539f2158118919a8d9400e76c04c5e6922e3ccb04397b0e0b72e4 -->

### Warm continuation — A_UPDATE_016.json
- A u16: queries/accepted/executed 0/0/0; wall 249.3s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_016.json`.
<!-- warm-receipt:A_UPDATE_016.json:237a19c2da948a35738bff9afe3e486d0d3d13f6203b53dd45d19a191b0bfb64 -->

### Warm continuation — A_VIABILITY_013.json
- A u13: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_013.json`.
<!-- warm-receipt:A_VIABILITY_013.json:acb908a340073e082f80a223a0fbbef206c257a6538e7bcd6acd3829386206e4 -->

### Warm continuation — A_VIABILITY_014.json
- A u14: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_014.json`.
<!-- warm-receipt:A_VIABILITY_014.json:6268dcdb5dd3688522f804c5043f808ea616c609f6dce07d9d3fe43d92fb1b3f -->

### Warm continuation — A_VIABILITY_015.json
- A u15: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_015.json`.
<!-- warm-receipt:A_VIABILITY_015.json:ddb34bf89c60c3630f382da5bda8436b3f3bae32c3849c6e7f64d4f860e09967 -->

### Warm continuation — A_VIABILITY_016.json
- A u16: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_016.json`.
<!-- warm-receipt:A_VIABILITY_016.json:d623d8870effff5ee581e902166b9907616117327d94a1244f26d359a21c8cec -->

### Warm continuation — STOP.json
- `{"error": "Connection closed by 1.15.42.207 port 22\n", "source": "owned warm bridge; no automatic retry"}`
- Receipt: `coopmwm/warm_live/STOP.json`.
<!-- warm-receipt:STOP.json:a9d41ffc12f5fb372065076f27f18401645865453a3a71678d6ef794419492b0 -->

### Warm continuation — WARM_A_FAILED.json
- `{"error": "ValueError('durable warm STOP before next wave')", "timestamp_utc": "2026-11-01T17:22:37.917523+00:00"}`
- Receipt: `coopmwm/warm_live/WARM_A_FAILED.json`.
<!-- warm-receipt:WARM_A_FAILED.json:6b5d32e58c957eaac717b3af8662087d3027ae48f71271ee3fb5f9e72cf21e8d -->

## WARM MEMORY QUALIFIED — 2026-11-01T18:42:03.622059+00:00
- 2 CAVs, gpuA: **559.567s/query, 6.434plans/hour**, 120 candidates/584 futures; screen `{'H': 300, 'K': 4}`, final `{'H': 300, 'K': 8}`; open-loop centre cost gap -52.302veh-s (future0 centre vs K4 closed-loop comparison).
- 4 CAVs, gpuR: **1029.964s/query, 3.495plans/hour**, 240 candidates/1064 futures; screen `{'H': 300, 'K': 4}`, final `{'H': 360, 'K': 8}`; open-loop centre cost gap 0.134veh-s (future0 centre vs K4 closed-loop comparison).
- 6 CAVs, gpuA: **898.275s/query, 4.008plans/hour**, 360 candidates/1544 futures; screen `{'H': 300, 'K': 4}`, final `{'H': 420, 'K': 8}`; open-loop centre cost gap -0.003veh-s (future0 centre vs K4 closed-loop comparison).
- cold E2 query times: `[85.88906374387443, 89.05008675158024]`s.
- cold H60: D_i sign `{'sign_agreement': 0.5, 'n': 2, 'ties': 2, 'ci95': [0.0, 1.0], 'wilson_ci95': [0.09453120573423074, 0.9054687942657693]}`; C_G sign `{'sign_agreement': None, 'n': 0, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 1.0]}`.
- cold H120: D_i sign `{'sign_agreement': 0.0, 'n': 2, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 0.6576197724933469]}`; C_G sign `{'sign_agreement': None, 'n': 0, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 1.0]}`.
- cold H300: D_i sign `{'sign_agreement': 0.5, 'n': 2, 'ties': 2, 'ci95': [0.0, 1.0], 'wilson_ci95': [0.09453120573423074, 0.9054687942657693]}`; C_G sign `{'sign_agreement': None, 'n': 0, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 1.0]}`.
- cold H600: D_i sign `{'sign_agreement': 0.5, 'n': 2, 'ties': 2, 'ci95': [0.0, 1.0], 'wilson_ci95': [0.09453120573423074, 0.9054687942657693]}`; C_G sign `{'sign_agreement': None, 'n': 0, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 1.0]}`.
- cold exact-labelled chosen-action savings: `[{'world': 986000, 'horizons': {'120': {'chosen': 'joint', 'exact_labelled_saving': -4.001776687800884e-11, 'predicted_saving': 4.868022978305817}, '300': {'chosen': 'joint', 'exact_labelled_saving': 6.548361852765083e-11, 'predicted_saving': 5.789575934410095}, '60': {'chosen': 'joint', 'exact_labelled_saving': -3.992681740783155e-09, 'predicted_saving': 1.8523305654525757}, '600': {'chosen': 'joint', 'exact_labelled_saving': 6.548361852765083e-11, 'predicted_saving': 86.09206318855286}}}, {'world': 986007, 'horizons': {'120': {'chosen': 'all', 'exact_labelled_saving': 0.0, 'predicted_saving': 0.0}, '300': {'chosen': 'all', 'exact_labelled_saving': 0.0, 'predicted_saving': 0.0}, '60': {'chosen': 'all', 'exact_labelled_saving': 0.0, 'predicted_saving': 0.0}, '600': {'chosen': 'all', 'exact_labelled_saving': 0.0, 'predicted_saving': 0.0}}}]`.
- warm E2 query times: `[86.10405902191997, 84.69402858614922]`s.
- warm H60: D_i sign `{'sign_agreement': 0.5, 'n': 2, 'ties': 2, 'ci95': [0.0, 1.0], 'wilson_ci95': [0.09453120573423074, 0.9054687942657693]}`; C_G sign `{'sign_agreement': None, 'n': 0, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 1.0]}`.
- warm H120: D_i sign `{'sign_agreement': 1.0, 'n': 2, 'ties': 2, 'ci95': [1.0, 1.0], 'wilson_ci95': [0.34238022750665303, 1.0]}`; C_G sign `{'sign_agreement': None, 'n': 0, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 1.0]}`.
- warm H300: D_i sign `{'sign_agreement': 0.5, 'n': 2, 'ties': 2, 'ci95': [0.0, 1.0], 'wilson_ci95': [0.09453120573423074, 0.9054687942657693]}`; C_G sign `{'sign_agreement': None, 'n': 0, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 1.0]}`.
- warm H600: D_i sign `{'sign_agreement': 0.5, 'n': 2, 'ties': 2, 'ci95': [0.0, 1.0], 'wilson_ci95': [0.09453120573423074, 0.9054687942657693]}`; C_G sign `{'sign_agreement': None, 'n': 0, 'ties': 2, 'ci95': [0.0, 0.0], 'wilson_ci95': [0.0, 1.0]}`.
- warm exact-labelled chosen-action savings: `[{'world': 986000, 'horizons': {'120': {'chosen': 'all', 'exact_labelled_saving': 0.0, 'predicted_saving': 0.0}, '300': {'chosen': 'all', 'exact_labelled_saving': 0.0, 'predicted_saving': 0.0}, '60': {'chosen': 'all', 'exact_labelled_saving': 0.0, 'predicted_saving': 0.0}, '600': {'chosen': 'all', 'exact_labelled_saving': 0.0, 'predicted_saving': 0.0}}}, {'world': 986007, 'horizons': {'120': {'chosen': 'joint', 'exact_labelled_saving': -0.07634644157951698, 'predicted_saving': 1.1913661062717438}, '300': {'chosen': 'joint', 'exact_labelled_saving': 1698.600333714603, 'predicted_saving': 1.368777722120285}, '60': {'chosen': 'joint', 'exact_labelled_saving': 0.11916434977320023, 'predicted_saving': 0.5568383932113647}, '600': {'chosen': 'joint', 'exact_labelled_saving': 5723.486351683328, 'predicted_saving': 1.3687926232814789}}}]`.
- two registered near-parity set-point windows, not novel MPPI plan labels; ties separate, two-world CIs uninformative; no established warm decision-quality improvement. All original exact SUMO labels reused; qualification adds0SUMO.
- Immutable complete receipt: `coopmwm/WARM_MEMORY_QUALIFIED.json`.

## Warm live publication — 2026-11-01T19:20:10.179755+00:00
- Conservatively charged 1823240/7500000 sim-s.

### Warm continuation — A_CHECK_020.json
- A_minus_MAPPO: J600 Δ=-864.562 [-2221.630,394.301] veh-s; delay reduction 1.71%; J900 Δ=-1287.859 [-3601.466,775.920] veh-s; delay reduction 1.42%. Negative Δ is better; 95% whole-world bootstrap.
- Reporting-only check, never checkpoint selection. Full safety/per-cell results in the linked immutable receipt.
- Receipt: `coopmwm/warm_live/A_CHECK_020.json`.
<!-- warm-receipt:A_CHECK_020.json:dfcb3e906bcc9f894ed4d816136605f5b2fe1709db41ba6c476dee519476c6be -->

### Warm continuation — A_CHECK_024.json
- A_minus_MAPPO: J600 Δ=-692.848 [-1941.644,428.910] veh-s; delay reduction 1.37%; J900 Δ=-826.567 [-2940.192,1241.208] veh-s; delay reduction 0.911%. Negative Δ is better; 95% whole-world bootstrap.
- Reporting-only check, never checkpoint selection. Full safety/per-cell results in the linked immutable receipt.
- Receipt: `coopmwm/warm_live/A_CHECK_024.json`.
<!-- warm-receipt:A_CHECK_024.json:6a4b7b4b1699e96d3b789f7e9c75f6008150e6c1d43428c09436373770450d51 -->

### Warm continuation — A_CHECK_028.json
- A_minus_MAPPO: J600 Δ=-368.845 [-1619.405,837.426] veh-s; delay reduction 0.728%; J900 Δ=-988.170 [-2587.751,564.361] veh-s; delay reduction 1.09%. Negative Δ is better; 95% whole-world bootstrap.
- Reporting-only check, never checkpoint selection. Full safety/per-cell results in the linked immutable receipt.
- Receipt: `coopmwm/warm_live/A_CHECK_028.json`.
<!-- warm-receipt:A_CHECK_028.json:8cd553bddaf3aadaab6587a9b9e736e7ebe3adc8d34a7f3119f78eaf32d6f0d0 -->

### Warm continuation — A_UPDATE_017.json
- A u17: queries/accepted/executed 0/0/0; wall 277.8s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_017.json`.
<!-- warm-receipt:A_UPDATE_017.json:102043a7581dc28fd0bf59de15f92d72cab0b76e9bbae7cd6bef576420f1f353 -->

### Warm continuation — A_UPDATE_018.json
- A u18: queries/accepted/executed 0/0/0; wall 272.9s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_018.json`.
<!-- warm-receipt:A_UPDATE_018.json:b6d4e9787563f39ac42be976b5964c56efc776f51f8c903b151d06317e1dabd8 -->

### Warm continuation — A_UPDATE_019.json
- A u19: queries/accepted/executed 0/0/0; wall 281.4s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_019.json`.
<!-- warm-receipt:A_UPDATE_019.json:e2415732959cba8a576906b1204f22687c32202d70a5e137106a6eea356c9811 -->

### Warm continuation — A_UPDATE_020.json
- A u20: queries/accepted/executed 0/0/0; wall 275.1s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_020.json`.
<!-- warm-receipt:A_UPDATE_020.json:3b6c0a351de5265a86f5442d789b65d0ff3eefe384e9eb286e10f738b5a26602 -->

### Warm continuation — A_UPDATE_021.json
- A u21: queries/accepted/executed 0/0/0; wall 281.3s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_021.json`.
<!-- warm-receipt:A_UPDATE_021.json:c45734c1e5ffbdc8d95ff1ca960e249671bf4ff1a1479dbb1e63f188ac3b64ab -->

### Warm continuation — A_UPDATE_022.json
- A u22: queries/accepted/executed 0/0/0; wall 275s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_022.json`.
<!-- warm-receipt:A_UPDATE_022.json:a0b4b4c202c2fe4f55f3a5daaeaa82948184955fae14d4ec4d4a1f40acb38809 -->

### Warm continuation — A_UPDATE_023.json
- A u23: queries/accepted/executed 0/0/0; wall 280.5s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_023.json`.
<!-- warm-receipt:A_UPDATE_023.json:a09ccd5e1cd165624d1eeaf912f919ffeb19f65875f2fc69d6d230107677afa1 -->

### Warm continuation — A_UPDATE_024.json
- A u24: queries/accepted/executed 0/0/0; wall 270.1s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_024.json`.
<!-- warm-receipt:A_UPDATE_024.json:be33c3f9e330778dc99dc596e2fbd23312e0b8ef7844a1f3872d51df81ac4c91 -->

### Warm continuation — A_UPDATE_025.json
- A u25: queries/accepted/executed 0/0/0; wall 276.8s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_025.json`.
<!-- warm-receipt:A_UPDATE_025.json:c6f059d58167c14f770907597a5d0352bc31d1b945c8fd02313dfea68a5e86b0 -->

### Warm continuation — A_UPDATE_026.json
- A u26: queries/accepted/executed 0/0/0; wall 273.1s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_026.json`.
<!-- warm-receipt:A_UPDATE_026.json:8343a92edbc08269af60efec26d4179e70ab965598153d79ede14ec485078994 -->

### Warm continuation — A_UPDATE_027.json
- A u27: queries/accepted/executed 0/0/0; wall 280.5s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_027.json`.
<!-- warm-receipt:A_UPDATE_027.json:baa111266216da2af1a11b0f6204e8d5a365e9bb2da7940b135f5384bae13a8c -->

### Warm continuation — A_UPDATE_028.json
- A u28: queries/accepted/executed 0/0/0; wall 271.3s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_028.json`.
<!-- warm-receipt:A_UPDATE_028.json:a03577ab1711b81a2891e6a5f48ced3bd1395a1d82237b6374ca0ed633382c2c -->

### Warm continuation — A_VIABILITY_017.json
- A u17: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_017.json`.
<!-- warm-receipt:A_VIABILITY_017.json:13bb4a4889b1929d337db5cdee97036838bcd086b9fb309ba438ab3b63e4942c -->

### Warm continuation — A_VIABILITY_018.json
- A u18: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_018.json`.
<!-- warm-receipt:A_VIABILITY_018.json:587e3d04809143e2a04f89a8992ee36b1aeda4b494eb7820c31de5766513a81c -->

### Warm continuation — A_VIABILITY_019.json
- A u19: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_019.json`.
<!-- warm-receipt:A_VIABILITY_019.json:b53e5da7976de67af6cb7deafdd410aea88758a96e4b8abf92360ba66244a900 -->

### Warm continuation — A_VIABILITY_020.json
- A u20: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_020.json`.
<!-- warm-receipt:A_VIABILITY_020.json:f8efacb8cbb80a4117d6d3035c48b780a69a0915207adac84cdf8d0c481410b2 -->

### Warm continuation — A_VIABILITY_021.json
- A u21: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_021.json`.
<!-- warm-receipt:A_VIABILITY_021.json:cbe2d413f561c7d7fa5a7ae77bc7589da58a6026a432cdf2ab76dabd1bd93484 -->

### Warm continuation — A_VIABILITY_022.json
- A u22: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_022.json`.
<!-- warm-receipt:A_VIABILITY_022.json:65c32d60e07eb854d8185a03f7160b5226d1f1e298fd4df5f145ae5b8d90bf55 -->

### Warm continuation — A_VIABILITY_023.json
- A u23: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_023.json`.
<!-- warm-receipt:A_VIABILITY_023.json:12de8095aebbe7dfec5694f128375eb1b820af6dc454efa0a7ea2b5bd3f57730 -->

### Warm continuation — A_VIABILITY_024.json
- A u24: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_024.json`.
<!-- warm-receipt:A_VIABILITY_024.json:8f99b681be32b0bac81cb2af0d45f93a8585a618e010f2f8bc624775f4865592 -->

### Warm continuation — A_VIABILITY_025.json
- A u25: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_025.json`.
<!-- warm-receipt:A_VIABILITY_025.json:d3d643fb7be6939f8aa9498c46272a4ba86115130e434c8984c85c1479725e2b -->

### Warm continuation — A_VIABILITY_026.json
- A u26: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_026.json`.
<!-- warm-receipt:A_VIABILITY_026.json:44e9609ab8f9c5502cd16a72af1ddcc69f719f8fe78cf94ffb7629fe917f1c12 -->

### Warm continuation — A_VIABILITY_027.json
- A u27: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_027.json`.
<!-- warm-receipt:A_VIABILITY_027.json:5e3afa12e4d7dc1667751c2ecc135c582d863d638fef4ff6889e4b7d16fadc05 -->

### Warm continuation — A_VIABILITY_028.json
- A u28: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_028.json`.
<!-- warm-receipt:A_VIABILITY_028.json:db7e0dc6d54f891576144453aa2ee9ac5ec685444052c1a6947ecd62a3ee91ee -->

### Warm continuation — A_VIABILITY_029.json
- A u29: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_029.json`.
<!-- warm-receipt:A_VIABILITY_029.json:6d322c6817821a7aba7ccd12c96c95798753af31278af29553d811a50d7b4ae0 -->

## Warm live publication — 2026-11-01T19:31:59.561631+00:00
- Conservatively charged 1909640/7500000 sim-s.

### Warm continuation — A_UPDATE_029.json
- A u29: queries/accepted/executed 0/0/0; wall 282.2s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_029.json`.
<!-- warm-receipt:A_UPDATE_029.json:2f3b869815f19d08fce5babdc84885969b000dd2eac588e8a77d7759e4573b4d -->

### Warm continuation — A_UPDATE_030.json
- A u30: queries/accepted/executed 0/0/0; wall 272.5s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_030.json`.
<!-- warm-receipt:A_UPDATE_030.json:18943f8964427e9fded020a3beb8c1744bd7742f35adb85825eff8095eed4ded -->

### Warm continuation — A_UPDATE_031.json
- A u31: queries/accepted/executed 0/0/0; wall 277.4s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_031.json`.
<!-- warm-receipt:A_UPDATE_031.json:4b937c971ddfef2515402a912f249a0b0eb4eeb8a3f37d138f494f9a9ec8274c -->

### Warm continuation — A_VIABILITY_030.json
- A u30: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_030.json`.
<!-- warm-receipt:A_VIABILITY_030.json:19e97f1a35607c5ebd56a3f8ae252f830a4662b8efd348d7c2ec7d4444c64363 -->

### Warm continuation — A_VIABILITY_031.json
- A u31: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_031.json`.
<!-- warm-receipt:A_VIABILITY_031.json:f60e91651700d8e2201d9ca1337c681ccf506c82b6dfd427707c6a826d389c29 -->

## Warm live publication — 2026-11-01T19:34:10.064374+00:00
- Conservatively charged 1909640/7500000 sim-s.

### Warm continuation — A_VIABILITY_032.json
- A u32: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_032.json`.
<!-- warm-receipt:A_VIABILITY_032.json:bc95b4d5348aaf907be171b8f485079d352d64ca03774058a413f5c13ad4eb86 -->

## Warm live publication — 2026-11-01T19:43:07.553121+00:00
- Conservatively charged 1981640/7500000 sim-s.

### Warm continuation — A_CHECK_032.json
- A_minus_MAPPO: J600 Δ=-549.992 [-2046.123,975.998] veh-s; delay reduction 1.09%; J900 Δ=-1327.428 [-3635.620,971.740] veh-s; delay reduction 1.46%. Negative Δ is better; 95% whole-world bootstrap.
- Reporting-only check, never checkpoint selection. Full safety/per-cell results in the linked immutable receipt.
- Receipt: `coopmwm/warm_live/A_CHECK_032.json`.
<!-- warm-receipt:A_CHECK_032.json:131dabdaaa0cc041baa3ef0969ed38142c9cb72af4e82dea08fb682e8d3797df -->

### Warm continuation — A_DEV_032.json
- A_minus_MAPPO: J600 Δ=133.616 [-1000.154,1230.670] veh-s; delay reduction -0.241%; J900 Δ=1007.088 [-1208.128,3045.616] veh-s; delay reduction -1.03%. Negative Δ is better; 95% whole-world bootstrap.
- Reporting-only check, never checkpoint selection. Full safety/per-cell results in the linked immutable receipt.
- Receipt: `coopmwm/warm_live/A_DEV_032.json`.
<!-- warm-receipt:A_DEV_032.json:c0742a14c116a49c94afcefc328e44942fc79009103a864582057cefbc1f998a -->

### Warm continuation — A_UPDATE_032.json
- A u32: queries/accepted/executed 0/0/0; wall 272.6s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_032.json`.
<!-- warm-receipt:A_UPDATE_032.json:567e2158e7ed7b45661f182283314b7d193a8f720744aedc96939edb8c9266da -->

### Warm continuation — A_VIABILITY_033.json
- A u33: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_033.json`.
<!-- warm-receipt:A_VIABILITY_033.json:f46ff03a040041597dbe9c2d9b8c4e388cef640a528a743858735302abc8359e -->

## Warm live publication — 2026-11-01T19:44:12.538520+00:00
- Conservatively charged 2010440/7500000 sim-s.

### Warm continuation — A_UPDATE_033.json
- A u33: queries/accepted/executed 0/0/0; wall 277.5s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_033.json`.
<!-- warm-receipt:A_UPDATE_033.json:c9c13306ed08a627569648d007c774d4367a8056f760e1e4337c985870c553e4 -->

## Warm live publication — 2026-11-01T19:47:28.417678+00:00
- Conservatively charged 2010440/7500000 sim-s.

### Warm continuation — A_VIABILITY_034.json
- A u34: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_034.json`.
<!-- warm-receipt:A_VIABILITY_034.json:3ce300370e626f7692e63f72792aebe45e22fba2c74a43301ade3005bec54ce3 -->

## Warm live publication — 2026-11-01T19:48:33.756570+00:00
- Conservatively charged 2039240/7500000 sim-s.

### Warm continuation — A_UPDATE_034.json
- A u34: queries/accepted/executed 0/0/0; wall 270.7s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_034.json`.
<!-- warm-receipt:A_UPDATE_034.json:cd52931a7a863acd476aa6cb19473c13c0a92a3acb1cdc0ce57cd362d821fbe4 -->

## Warm live publication — 2026-11-01T19:51:50.018016+00:00
- Conservatively charged 2039240/7500000 sim-s.

### Warm continuation — A_VIABILITY_035.json
- A u35: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_035.json`.
<!-- warm-receipt:A_VIABILITY_035.json:1924b86d6d05098e864e0bd2b1e27643c0a3ee76cd4c62ae2e453bb47506b4ec -->

## TRAINING STARTED B/R — 2026-11-01T19:53:05.522558+00:00
- MAPPOu0, immutable exactcpu5; B/R and exact grader/broker detached on cpuQ.
- WARM MEMORY QUALIFIED retained; both concurrent GPU workers pass8source-bound EXACT0 gates.
- K4/H300 screen, K8/Hfinal, Nexec10; no look-ahead reduction. Exact grades only for trust/imitation/library, asynchronous.
- R centre requests retain lower priority until B finishes; no candidate scoring/library for R.
- First-update executed-plan rate/ETA pending. Owners: `coopmwm/EXACT_TRAINING_STARTED.json`.

## Warm live publication — 2026-11-01T19:53:06.574668+00:00
- Conservatively charged 2096840/7500000 sim-s.

### Warm continuation — A_UPDATE_035.json
- A u35: queries/accepted/executed 0/0/0; wall 276.3s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_035.json`.
<!-- warm-receipt:A_UPDATE_035.json:361218349f22807025846448d433d89032aaf576f52596b7268ab99ca4bd1593 -->

## Warm live publication — 2026-11-01T19:56:23.824685+00:00
- Conservatively charged 2096840/7500000 sim-s.

### Warm continuation — A_VIABILITY_036.json
- A u36: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_036.json`.
<!-- warm-receipt:A_VIABILITY_036.json:6a8ffd5ea3b6eef3c8f019707720d874fca4c66dd6979d4b60f133ec0ff50f6e -->

## Warm live publication — 2026-11-01T19:57:29.338947+00:00
- Conservatively charged 2111240/7500000 sim-s.

### Warm continuation — A_UPDATE_036.json
- A u36: queries/accepted/executed 0/0/0; wall 274s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_036.json`.
<!-- warm-receipt:A_UPDATE_036.json:7a3949c4801a0826128440c142bbd5d2eeb2433622d44df87bef2c1ed1049126 -->

## Warm live publication — 2026-11-01T19:59:40.744307+00:00
- Conservatively charged 2140040/7500000 sim-s.

### Warm continuation — A_CHECK_036.json
- A_minus_MAPPO: J600 Δ=-306.217 [-1580.705,815.006] veh-s; delay reduction 0.604%; J900 Δ=-756.662 [-3226.879,1594.933] veh-s; delay reduction 0.834%. Negative Δ is better; 95% whole-world bootstrap.
- Reporting-only check, never checkpoint selection. Full safety/per-cell results in the linked immutable receipt.
- Receipt: `coopmwm/warm_live/A_CHECK_036.json`.
<!-- warm-receipt:A_CHECK_036.json:01e954fedaa7fed16c29a3d5b5c4207cf3d8e84a42a1a0fd2c11e87616634e65 -->

## Warm live publication — 2026-11-01T20:01:51.538925+00:00
- Conservatively charged 2140040/7500000 sim-s.

### Warm continuation — A_VIABILITY_037.json
- A u37: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_037.json`.
<!-- warm-receipt:A_VIABILITY_037.json:bb25af20d5d20aaf5c6ba1b58f18d951b597d23d54856d918f9917adc97f32cb -->

## Warm live publication — 2026-11-01T20:04:02.989872+00:00
- Conservatively charged 2168840/7500000 sim-s.

### Warm continuation — A_UPDATE_037.json
- A u37: queries/accepted/executed 0/0/0; wall 278.9s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_037.json`.
<!-- warm-receipt:A_UPDATE_037.json:84e2bb51d9ddb493b840b5d57a4d1d0ebe0ce597ddcd74a63293ec38485f4cc7 -->

## Warm live publication — 2026-11-01T20:06:14.273186+00:00
- Conservatively charged 2168840/7500000 sim-s.

### Warm continuation — A_VIABILITY_038.json
- A u38: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_038.json`.
<!-- warm-receipt:A_VIABILITY_038.json:f9272573db90d15ec0bc07d27ee5e17c175a5e6e532511031e57363f1a2ec9d9 -->

## Warm live publication — 2026-11-01T20:08:25.108222+00:00
- Conservatively charged 2197640/7500000 sim-s.

### Warm continuation — A_UPDATE_038.json
- A u38: queries/accepted/executed 0/0/0; wall 277.3s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_038.json`.
<!-- warm-receipt:A_UPDATE_038.json:6c7fbab07104782799cb47ee386b967079d3544ac123f075c9a4970203274e9f -->

## Warm live publication — 2026-11-01T20:11:42.096674+00:00
- Conservatively charged 2197640/7500000 sim-s.

### Warm continuation — A_VIABILITY_039.json
- A u39: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_039.json`.
<!-- warm-receipt:A_VIABILITY_039.json:dfae5e0e7265cc5b03b1683572e04fe4a875cc7115cbaf3b216bd4b7f60a1ad8 -->

## Warm live publication — 2026-11-01T20:12:48.085828+00:00
- Conservatively charged 2226440/7500000 sim-s.

### Warm continuation — A_UPDATE_039.json
- A u39: queries/accepted/executed 0/0/0; wall 281.2s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_039.json`.
<!-- warm-receipt:A_UPDATE_039.json:bd6ffa39d7b83f3e96bc2b10855fd6aebdb92d41dfa71163b4b38907781d754d -->

## Warm live publication — 2026-11-01T20:16:01.697981+00:00
- Conservatively charged 2226440/7500000 sim-s.

### Warm continuation — A_VIABILITY_040.json
- A u40: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_040.json`.
<!-- warm-receipt:A_VIABILITY_040.json:85a44dc47d44ab0ad05a2b58c28ba9fa0d2f5ab1dc0d61919f6457ad21e6347b -->

## Warm live publication — 2026-11-01T20:18:14.500393+00:00
- Conservatively charged 2240840/7500000 sim-s.

### Warm continuation — A_UPDATE_040.json
- A u40: queries/accepted/executed 0/0/0; wall 271s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_040.json`.
<!-- warm-receipt:A_UPDATE_040.json:b74d2edb9e1690fb001669b082e50239ce3cd36f93c980a47362b63daff6538b -->

## Warm live publication — 2026-11-01T20:19:19.350717+00:00
- Conservatively charged 2269640/7500000 sim-s.

### Warm continuation — A_CHECK_040.json
- A_minus_MAPPO: J600 Δ=-166.208 [-1606.067,1356.448] veh-s; delay reduction 0.328%; J900 Δ=-631.564 [-2851.264,1587.848] veh-s; delay reduction 0.696%. Negative Δ is better; 95% whole-world bootstrap.
- Reporting-only check, never checkpoint selection. Full safety/per-cell results in the linked immutable receipt.
- Receipt: `coopmwm/warm_live/A_CHECK_040.json`.
<!-- warm-receipt:A_CHECK_040.json:559ad6cbbbc424a3cdc7190b6dc29480b13b39cbf794719c828aa72bb93ea2e4 -->

## Warm live publication — 2026-11-01T20:21:26.507075+00:00
- Conservatively charged 2269640/7500000 sim-s.

### Warm continuation — A_VIABILITY_041.json
- A u41: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_041.json`.
<!-- warm-receipt:A_VIABILITY_041.json:ea1b12c678678bfac66d7cc55569efcd11fffc9cf4857b5afa096ad00c1a032b -->

## Warm live publication — 2026-11-01T20:31:58.229602+00:00
- Conservatively charged 2327240/7500000 sim-s.

### Warm continuation — A_UPDATE_041.json
- A u41: queries/accepted/executed 0/0/0; wall 277.3s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_041.json`.
<!-- warm-receipt:A_UPDATE_041.json:1dda282394530f02a1dfff1c084077e75d5752bfa2702131768d265162f71133 -->

### Warm continuation — A_UPDATE_042.json
- A u42: queries/accepted/executed 0/0/0; wall 269.1s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_042.json`.
<!-- warm-receipt:A_UPDATE_042.json:dc1b2356b24a7dfc1330c199a5b5549fe243890be11017f282001fab0eeac25d -->

### Warm continuation — A_VIABILITY_042.json
- A u42: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_042.json`.
<!-- warm-receipt:A_VIABILITY_042.json:e66ac23dbb764bcfba6d62e4426e2bf9bd958bb109d5e3e38f2454ef44afde0a -->

### Warm continuation — A_VIABILITY_043.json
- A u43: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_043.json`.
<!-- warm-receipt:A_VIABILITY_043.json:4a0aa010e0514563ea325251ce661555f436ed317d2d40c7870ba340bd873bd6 -->

## Warm live publication — 2026-11-01T20:33:04.260454+00:00
- Conservatively charged 2356040/7500000 sim-s.

### Warm continuation — A_UPDATE_043.json
- A u43: queries/accepted/executed 0/0/0; wall 280s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_043.json`.
<!-- warm-receipt:A_UPDATE_043.json:6274b793a761c6465bc1bc4140970963461b9fd2b2ef6bee5d42784ae1e15e63 -->

## Warm live publication — 2026-11-01T20:35:16.019579+00:00
- Conservatively charged 2356040/7500000 sim-s.

### Warm continuation — A_VIABILITY_044.json
- A u44: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_044.json`.
<!-- warm-receipt:A_VIABILITY_044.json:9019195ee5f67d488e1cbfa53ebc666f1c8cfcd9a08b43ed6db1a241e41c096d -->

## Warm live publication — 2026-11-01T20:37:24.400178+00:00
- Conservatively charged 2370440/7500000 sim-s.

### Warm continuation — A_UPDATE_044.json
- A u44: queries/accepted/executed 0/0/0; wall 273.5s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_044.json`.
<!-- warm-receipt:A_UPDATE_044.json:0fb9ff83813c3f8b49350cc24fc8fdf3b33d4ce620d2d55c5f14f51484e3a9a9 -->

## Warm live publication — 2026-11-01T20:38:28.431693+00:00
- Conservatively charged 2399240/7500000 sim-s.

### Warm continuation — A_CHECK_044.json
- A_minus_MAPPO: J600 Δ=-713.607 [-1867.982,279.454] veh-s; delay reduction 1.41%; J900 Δ=-952.147 [-3413.962,1424.270] veh-s; delay reduction 1.05%. Negative Δ is better; 95% whole-world bootstrap.
- Reporting-only check, never checkpoint selection. Full safety/per-cell results in the linked immutable receipt.
- Receipt: `coopmwm/warm_live/A_CHECK_044.json`.
<!-- warm-receipt:A_CHECK_044.json:7b77302c4c00b5f936211286602ef5e309c174de96822cd22381f2914dbcd733 -->

## Warm live publication — 2026-11-01T20:41:41.647424+00:00
- Conservatively charged 2399240/7500000 sim-s.

### Warm continuation — A_VIABILITY_045.json
- A u45: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_045.json`.
<!-- warm-receipt:A_VIABILITY_045.json:c68d90ed05a06eac259c0b59743084190d31df5f54ef9a4bd2496cdf66f73141 -->

## Warm live publication — 2026-11-01T20:43:51.187344+00:00
- Conservatively charged 2428040/7500000 sim-s.

### Warm continuation — A_UPDATE_045.json
- A u45: queries/accepted/executed 0/0/0; wall 276.7s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_045.json`.
<!-- warm-receipt:A_UPDATE_045.json:4c819aa351e41905488277c6721f5a13697aa3cce88af134e341d036e50efc9e -->

## Warm live publication — 2026-11-01T20:46:01.571030+00:00
- Conservatively charged 2428040/7500000 sim-s.

### Warm continuation — A_VIABILITY_046.json
- A u46: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_046.json`.
<!-- warm-receipt:A_VIABILITY_046.json:d7e10980cfe678b1e5f17a96f246252a95ad6f303eeda39ce56c3fd59896f4aa -->

## Warm live publication — 2026-11-01T20:48:10.120956+00:00
- Conservatively charged 2456840/7500000 sim-s.

### Warm continuation — A_UPDATE_046.json
- A u46: queries/accepted/executed 0/0/0; wall 276s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_046.json`.
<!-- warm-receipt:A_UPDATE_046.json:e5cd40c1458b7a737cd1adbbd14629b88b87c3e263322efea32e58d0f2e9e93c -->

## Warm live publication — 2026-11-01T20:50:23.571628+00:00
- Conservatively charged 2456840/7500000 sim-s.

### Warm continuation — A_VIABILITY_047.json
- A u47: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_047.json`.
<!-- warm-receipt:A_VIABILITY_047.json:f77d304a49cc41c4f5ef70e9273590a354353cbcad7a4c51f2511c6216f25239 -->

## Warm live publication — 2026-11-01T20:52:35.564718+00:00
- Conservatively charged 2485640/7500000 sim-s.

### Warm continuation — A_UPDATE_047.json
- A u47: queries/accepted/executed 0/0/0; wall 276.9s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_047.json`.
<!-- warm-receipt:A_UPDATE_047.json:b7fe9109954ab7519294257eea003ded6f477775c556cd39e2cb1df44293d680 -->

## Warm live publication — 2026-11-01T20:55:49.077658+00:00
- Conservatively charged 2485640/7500000 sim-s.

### Warm continuation — A_VIABILITY_048.json
- A u48: executed-plan vs own-action SUMO H300 critic residual NA veh-s, 95% CI `None`; new/known-like plans 0/0.
- Selection-conditional critic contrast, not a same-state causal effect; type/rank/noise details remain in the receipt.
- Receipt: `coopmwm/warm_live/A_VIABILITY_048.json`.
<!-- warm-receipt:A_VIABILITY_048.json:b48f2c1cdc38fdd952f4e30f352150c74c5113af336ad00beca2dc98af048f02 -->

## Warm live publication — 2026-11-01T21:07:17.489483+00:00
- Conservatively charged 2919640/7500000 sim-s.

### Warm continuation — A_CHECK_048.json
- A_minus_MAPPO: J600 Δ=-16.713 [-1461.458,1462.350] veh-s; delay reduction 0.033%; J900 Δ=500.964 [-2369.329,3196.457] veh-s; delay reduction -0.552%. Negative Δ is better; 95% whole-world bootstrap.
- Reporting-only check, never checkpoint selection. Full safety/per-cell results in the linked immutable receipt.
- Receipt: `coopmwm/warm_live/A_CHECK_048.json`.
<!-- warm-receipt:A_CHECK_048.json:8efc4c0ed6ec7fc2935a193b708e77fd3a7f9348b09e1b29fd560c119e0fcd05 -->

### Warm continuation — A_DEV_048.json
- A_minus_MAPPO: J600 Δ=-763.695 [-1999.903,326.983] veh-s; delay reduction 1.38%; J900 Δ=-536.708 [-2002.688,902.116] veh-s; delay reduction 0.547%. Negative Δ is better; 95% whole-world bootstrap.
- Reporting-only check, never checkpoint selection. Full safety/per-cell results in the linked immutable receipt.
- Receipt: `coopmwm/warm_live/A_DEV_048.json`.
<!-- warm-receipt:A_DEV_048.json:77a91f06e148dc591d931dd2105d332877ea36d653257181d2b108179a922771 -->

### Warm continuation — A_UPDATE_048.json
- A u48: queries/accepted/executed 0/0/0; wall 273.4s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Receipt: `coopmwm/warm_live/A_UPDATE_048.json`.
<!-- warm-receipt:A_UPDATE_048.json:a0c4d732b834d5fa039028f1aa6db48ef33ff3826fc940436a97c209847923c5 -->

### Warm continuation — WARM_A_PAUSED.json
- `{"charged_sim_s": 2528840, "full_endpoint": 48, "reason": "A fixedu48 complete; wait for B48 before final panel", "timestamp_utc": "2026-11-01T21:00:22.741918+00:00", "update": 48}`
- Receipt: `coopmwm/warm_live/WARM_A_PAUSED.json`.
<!-- warm-receipt:WARM_A_PAUSED.json:24bdd81f07433591de0aaad97516dbc2697e6f2fa58acba29059fd6d0e61d47d -->

## Warm live publication — 2026-11-01T21:33:33.987705+00:00
- Conservatively charged 3242240/7500000 sim-s.

### Warm continuation — WARM_B_FAILED_V3.json
- `{"error": "ValueError('durable guided engineering STOP')", "timestamp_utc": "2026-11-01T21:18:18.617196+00:00"}`
- Receipt: `coopmwm/warm_live/WARM_B_FAILED_V3.json`.
<!-- warm-receipt:WARM_B_FAILED_V3.json:ba05b928d9a3277eb9baab66d4c2600544cac2f139201c511c5b1be921ed4386 -->

### Warm continuation — WARM_GUIDED_STOP.json
- `{"error": "JSONDecodeError('Expecting value: line 1 column 1 (char 0)')", "source": "capacity broker control", "unix": 1793567885.634367}`
- Receipt: `coopmwm/warm_live/WARM_GUIDED_STOP.json`.
<!-- warm-receipt:WARM_GUIDED_STOP.json:6bb256081946b912414cdfafa30321e2dbcc46a732ff96557ea621d65faa187d -->

### Warm continuation — WARM_R_FAILED_V3.json
- `{"error": "ValueError('durable guided engineering STOP')", "timestamp_utc": "2026-11-01T21:18:06.783912+00:00"}`
- Receipt: `coopmwm/warm_live/WARM_R_FAILED_V3.json`.
<!-- warm-receipt:WARM_R_FAILED_V3.json:e17365eccd47e54e7c26ac5aee1bf8024f0bbb3ace135399d1b681a62c9d3fb8 -->

## Warm live publication — 2026-11-01T22:07:13.978197+00:00
- Conservatively charged 3242860/7500000 sim-s.

### Warm continuation — B_EXACT_VIABILITY_dccf60491b28433e4500845a95addb7fd0debe0792e7b71f6bef9c65e00b8d5a.json
- B u1: exact paired SUMO plan saving mean 6.666 veh-s, 95% world-cluster CI `None`, 1 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_dccf60491b28433e4500845a95addb7fd0debe0792e7b71f6bef9c65e00b8d5a.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_dccf60491b28433e4500845a95addb7fd0debe0792e7b71f6bef9c65e00b8d5a.json:2571b2ee336a7c888cf8223afb8ff229f2b3c8e11f6944d3d6625442cef1b7fc -->

## Warm live publication — 2026-11-01T22:11:35.001477+00:00
- Conservatively charged 3244070/7500000 sim-s.

### Warm continuation — B_EXACT_VIABILITY_08bc9c74435ce8b6fd1e10f5a29cfa890d4303ed5c1860dd689915bb0612c110.json
- B u1: exact paired SUMO plan saving mean 862.9 veh-s, 95% world-cluster CI `[6.666416322528676, 1290.9455121911988]`, 3 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_08bc9c74435ce8b6fd1e10f5a29cfa890d4303ed5c1860dd689915bb0612c110.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_08bc9c74435ce8b6fd1e10f5a29cfa890d4303ed5c1860dd689915bb0612c110.json:7b05fb7de1c2c56958933b5e59f7774ef17f3ac19600d33e483557fc6835ae1d -->

## Warm live publication — 2026-11-01T22:18:03.981180+00:00
- Conservatively charged 3244070/7500000 sim-s.

### Warm continuation — B_VIABILITY_001.json
- B u1: exact paired SUMO plan saving mean 862.9 veh-s, 95% world-cluster CI `[6.666416322528676, 1290.9455121911988]`, 3 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_VIABILITY_001.json`.
<!-- warm-receipt:B_VIABILITY_001.json:6fa161521a1ea9e8ffd58abcb6613082d17183a029a2fec50cdc84184bc4d189 -->

## Warm live publication — 2026-11-01T22:19:13.310595+00:00
- Conservatively charged 3306220/7500000 sim-s.

### Warm continuation — B_UPDATE_001.json
- B u1: queries/accepted/executed 171/3/3; wall 93.53s; next beta 0.751. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 3, "executed": 3, "queries": 3}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Exact-grade queue/lag: `{"failed": 0, "graded": 3, "max_lag_s": 15.68671441078186, "mean_lag_s": 12.17886463801066, "orphaned": 0, "overlap_unresolved": 0, "pending": 0, "running": 0, "stopped": 0, "update": 1}`; final ETA 1.221h.
- Receipt: `coopmwm/warm_live/B_UPDATE_001.json`.
<!-- warm-receipt:B_UPDATE_001.json:3edcfbe3fe2f0fbef8be8aa7ee026a2b33e5d7873ba79ab13994170f0d7b1fa9 -->

## Warm live publication — 2026-11-01T22:20:17.473903+00:00
- Conservatively charged 3316220/7500000 sim-s.

### Warm continuation — ACCEL_EXACT_PLAN_000_FAILED.json
- Exact executed-plan twins: `{}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_000_FAILED.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_000_FAILED.json:b5d3fd1294c58c0ee1e7f4d6275565da004cf49bdbcf29a943b3d265a085fcbb -->

### Warm continuation — ACCEL_EXACT_PLAN_001_FAILED.json
- Exact executed-plan twins: `{}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_001_FAILED.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_001_FAILED.json:ec5c2dace890b0b8e481f7a1b63cb46a25649ddf078c8b23218309ad0af09057 -->

## Warm live publication — 2026-11-01T22:41:51.293380+00:00
- Conservatively charged 3331280/7500000 sim-s.

### Warm continuation — ACCEL_EXACT_PLAN_000.json
- Exact executed-plan twins: `{"120": {"C_G": 53.17882023977927, "D_G": 6.57300294097513, "D_i": [0.10668684939628292, 19.595361023921214, -66.30786517212164]}, "300": {"C_G": -868.1767234589606, "D_G": 1042.1523691190523, "D_i": [0.114493582801515, 1812.8935004434134, 97.32109855179806]}, "60": {"C_G": 6.2103093511809675, "D_G": -3.061505717970249, "D_i": [0.2528922868045811, -1.982097747957141, -7.542609607998656]}, "600": {"C_G": -6459.329807218768, "D_G": 266.37194287207967, "D_i": [0.11983589885494439, 5633.5996205810225, 1091.9822936109704]}}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_000.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_000.json:67fcf65f9aa005bc1b158c7584ae6d13a54e17c62e495ba0a20d98c40852097a -->

### Warm continuation — ACCEL_EXACT_PLAN_001.json
- Exact executed-plan twins: `{"120": {"C_G": 8.554533997084945e-05, "D_G": 3.044738833149495, "D_i": [-0.0772545429172169, -1.8969694792758673e-07, 3.121908020423689]}, "300": {"C_G": 0.36667844047406106, "D_G": 12.062233908927738, "D_i": [-0.44386971215863014, 1.3139204384060577e-05, 12.139412041407923]}, "60": {"C_G": 0.003699239289289835, "D_G": 0.5022742682449461, "D_i": [-0.08090635090547948, -2.3617531041963957e-07, 0.5794816160364462]}, "600": {"C_G": 5.8708897903052275, "D_G": -245.68730165935995, "D_i": [-5.948102965368889, 2.4539200239814818e-05, -245.61011302349652]}}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_001.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_001.json:43930167b84d68eac731232210d827bc6504d085bff42759a66a6829accd9022 -->

### Warm continuation — ACCEL_EXACT_PLAN_002.json
- Exact executed-plan twins: `{"120": {"C_G": 323.14976656404633, "D_G": 51.94878721373425, "D_i": [-178.7850831113292, -61.435964183475335, -30.976000006536196, -0.003932048971364566]}, "300": {"C_G": -2015.288082288851, "D_G": -387.9573838097531, "D_i": [-448.6201439672368, -337.26579334696, 2413.2205122709784, -0.0038764776836615056]}, "60": {"C_G": 4.878723412531144, "D_G": 6.778537182067339, "D_i": [2.52286761295818, -0.9676864238149392, 0.34711848917459065, -0.002485908781636681]}, "600": {"C_G": -3877.9211828140687, "D_G": -5380.062628581938, "D_i": [-2753.352061770296, -3652.751319996758, 4903.965868840714, -0.0039328415296040475]}}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_002.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_002.json:29e8797c3fc3ccbdd75db68e3de8e7fd6aeb559f981e476b77bcb182dec2b73e -->

### Warm continuation — ACCEL_EXACT_PLAN_002_FAILED.json
- Exact executed-plan twins: `{}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_002_FAILED.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_002_FAILED.json:b9852045f9918ee39b77c38791a59eab2abe6ccbd26b6eb1e0d9eb2331a37ca2 -->

### Warm continuation — B_EXACT_VIABILITY_0c40c7b9c10e3da061f60f02d832269ac712fb969ee703f8a080cd5267aac28d.json
- B u2: exact paired SUMO plan saving mean 650.4 veh-s, 95% world-cluster CI `[6.666416322528676, 1290.9455121911988]`, 4 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_0c40c7b9c10e3da061f60f02d832269ac712fb969ee703f8a080cd5267aac28d.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_0c40c7b9c10e3da061f60f02d832269ac712fb969ee703f8a080cd5267aac28d.json:fc63cc1e92cffab2fa092355ad470494a0647fe95103d4d46dd51a5cdfdf8cc8 -->

## Warm live publication — 2026-11-01T22:51:46.862946+00:00
- Conservatively charged 3332720/7500000 sim-s.

### Warm continuation — B_EXACT_VIABILITY_be5d56ed3ddcb8d9ec6857124c703778a2ada7f233594620f930bb630f8c8ec5.json
- B u2: exact paired SUMO plan saving mean 433.6 veh-s, 95% world-cluster CI `[1.333283264505735, 969.8162260524452]`, 6 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_be5d56ed3ddcb8d9ec6857124c703778a2ada7f233594620f930bb630f8c8ec5.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_be5d56ed3ddcb8d9ec6857124c703778a2ada7f233594620f930bb630f8c8ec5.json:a445bfe79082a5fc9328cba29128c0af7a63d2ae134878b84ea173481c7c5715 -->

## Warm live publication — 2026-11-01T23:12:18.761471+00:00
- Conservatively charged 3333180/7500000 sim-s.

### Warm continuation — B_EXACT_VIABILITY_24e9412867374bcf340358ee85286db93608021f0a5d68faba1650d322c6b30a.json
- B u2: exact paired SUMO plan saving mean 371.6 veh-s, 95% world-cluster CI `[1.1145700291757141, 862.0635351249571]`, 7 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_24e9412867374bcf340358ee85286db93608021f0a5d68faba1650d322c6b30a.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_24e9412867374bcf340358ee85286db93608021f0a5d68faba1650d322c6b30a.json:f2efc93c62b39fc3e73075a32d5159746c73813845bb4719e59c0dfd9339d1f4 -->

## Warm live publication — 2026-11-01T23:27:28.364748+00:00
- Conservatively charged 3335130/7500000 sim-s.

### Warm continuation — B_EXACT_VIABILITY_609079e15490a9e0ff58184ddc0b4786e3dfe5f94da6d1f55fe0c23d9df21bf1.json
- B u2: exact paired SUMO plan saving mean 309.6 veh-s, 95% world-cluster CI `[-50.58229688374455, 775.8550812272088]`, 8 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_609079e15490a9e0ff58184ddc0b4786e3dfe5f94da6d1f55fe0c23d9df21bf1.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_609079e15490a9e0ff58184ddc0b4786e3dfe5f94da6d1f55fe0c23d9df21bf1.json:4216b9c68e9d279ddd94aef803f5e0d5e3377b90240cc8b4e6db94922759d7d9 -->

## Warm live publication — 2026-11-01T23:29:36.731538+00:00
- Conservatively charged 3336950/7500000 sim-s.

### Warm continuation — R_VIABILITY_001.json
- R u1: exact paired SUMO plan saving mean -132.4 veh-s, 95% world-cluster CI `[-264.77152655943064, -0.08526310693014238]`, 2 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/R_VIABILITY_001.json`.
<!-- warm-receipt:R_VIABILITY_001.json:8dcb4f2400ed86e8d8eeb6cd9428527e6285a55e6e072d21842e226e106fdecb -->

## Warm live publication — 2026-11-01T23:30:41.094243+00:00
- Conservatively charged 3336950/7500000 sim-s.

### Warm continuation — R_UPDATE_001.json
- R u1: queries/accepted/executed 3/3/3; wall 4370s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 3, "executed": 3, "queries": 3}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Exact-grade queue/lag: `{"failed": 0, "graded": 6, "max_lag_s": 19.90656304359436, "mean_lag_s": 13.826808452606201, "orphaned": 0, "overlap_unresolved": 0, "pending": 0, "running": 0, "stopped": 0, "update": 1}`; final ETA 27.92h.
- Receipt: `coopmwm/warm_live/R_UPDATE_001.json`.
<!-- warm-receipt:R_UPDATE_001.json:8cc72c87a0ad82ec69e37247c90d6a2e783eca948c31572b02cce6bcd3f5a885 -->

## Warm live publication — 2026-11-01T23:31:47.063278+00:00
- Conservatively charged 3336950/7500000 sim-s.

### Warm continuation — B_EXACT_VIABILITY_5628d6582e2b51d75a30e3c7bc6eb5541e450dd3f471eae3ed8f8709c780dc18.json
- B u2: exact paired SUMO plan saving mean 275.2 veh-s, 95% world-cluster CI `[-44.25950977327648, 705.3208916745057]`, 9 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_5628d6582e2b51d75a30e3c7bc6eb5541e450dd3f471eae3ed8f8709c780dc18.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_5628d6582e2b51d75a30e3c7bc6eb5541e450dd3f471eae3ed8f8709c780dc18.json:1ede4a277240e882bae056f32591738d6206ff022648794627f8c480e82918fe -->

## Warm live publication — 2026-11-01T23:38:21.464369+00:00
- Conservatively charged 3336950/7500000 sim-s.

### Warm continuation — B_EXACT_VIABILITY_a4449fdbeb2d139a7389dd860a754e6bc074192b912825009c097b0cf9b5384e.json
- B u2: exact paired SUMO plan saving mean 275.2 veh-s, 95% world-cluster CI `[-44.25950977327648, 705.3208916745057]`, 9 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_a4449fdbeb2d139a7389dd860a754e6bc074192b912825009c097b0cf9b5384e.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_a4449fdbeb2d139a7389dd860a754e6bc074192b912825009c097b0cf9b5384e.json:d507138106acce5e45df61b08c29c3db3ebac98a2d9b5ce09cef38f5eb6bfa63 -->

## Warm live publication — 2026-11-01T23:43:42.489770+00:00
- Conservatively charged 3338010/7500000 sim-s.

### Warm continuation — B_EXACT_VIABILITY_8bc3da306ed1cc000bd35089dc4f8f490185ccd8c21f76b03b3d5c84bff68657.json
- B u2: exact paired SUMO plan saving mean 247.7 veh-s, 95% world-cluster CI `[-38.65403159252047, 646.5537876101174]`, 10 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_8bc3da306ed1cc000bd35089dc4f8f490185ccd8c21f76b03b3d5c84bff68657.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_8bc3da306ed1cc000bd35089dc4f8f490185ccd8c21f76b03b3d5c84bff68657.json:dd8ce1d2d0a70adaf730ce401837a4f076d1b93663e69ea72486bcf6acd9da4f -->

## Warm live publication — 2026-11-01T23:49:06.319561+00:00
- Conservatively charged 3338010/7500000 sim-s.

### Warm continuation — B_EXACT_VIABILITY_f1d3fd1e73331c00e1ce9476a7ec8f4933728b3aed82117e089eb608441244ff.json
- B u2: exact paired SUMO plan saving mean 224.8 veh-s, 95% world-cluster CI `[-35.39819642332041, 596.7440008949391]`, 11 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_f1d3fd1e73331c00e1ce9476a7ec8f4933728b3aed82117e089eb608441244ff.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_f1d3fd1e73331c00e1ce9476a7ec8f4933728b3aed82117e089eb608441244ff.json:88f26b0dcb5598d32cc6d58d149dbef3c02fae0848e0a938fe2e45dfa2b8a924 -->

## Warm live publication — 2026-11-02T00:54:05.428442+00:00
- Conservatively charged 3339010/7500000 sim-s.

### Warm continuation — B_EXACT_VIABILITY_95f658a887c5263e52165505ce17127ceabec465a9bdcfa6ef92f65f594a6559.json
- B u2: exact paired SUMO plan saving mean 190.2 veh-s, 95% world-cluster CI `[-29.51649880329449, 545.5074550050579]`, 13 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_95f658a887c5263e52165505ce17127ceabec465a9bdcfa6ef92f65f594a6559.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_95f658a887c5263e52165505ce17127ceabec465a9bdcfa6ef92f65f594a6559.json:a31edfb168e94833eb3355b3120e3d961a18bba3dd3c580949a1f5474259f681 -->

## Warm live publication — 2026-11-02T01:02:49.123728+00:00
- Conservatively charged 3340270/7500000 sim-s.

### Warm continuation — B_VIABILITY_002.json
- B u2: exact paired SUMO plan saving mean 145.2 veh-s, 95% world-cluster CI `[-59.979113030558544, 479.8055757232458]`, 15 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_VIABILITY_002.json`.
<!-- warm-receipt:B_VIABILITY_002.json:52176300eae65e80b012a3d3d54da4e1a9dbd815f23d2773e7f0a8d38d6de39f -->

## Warm live publication — 2026-11-02T01:03:54.208857+00:00
- Conservatively charged 3414470/7500000 sim-s.

### Warm continuation — ACCEL_EXACT_PLAN_003_FAILED.json
- Exact executed-plan twins: `{}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_003_FAILED.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_003_FAILED.json:ef6452a96db97502e476028361280bd68886cd4222877450b3571a380fc2b33a -->

### Warm continuation — B_UPDATE_002.json
- B u2: queries/accepted/executed 93/15/12; wall 9839s; next beta 0.751. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 15, "executed": 12, "queries": 14}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Exact-grade queue/lag: `{"failed": 0, "graded": 12, "max_lag_s": 17.956080436706543, "mean_lag_s": 11.755984206994375, "orphaned": 0, "overlap_unresolved": 0, "pending": 0, "running": 0, "stopped": 0, "update": 2}`; final ETA 125.7h.
- Receipt: `coopmwm/warm_live/B_UPDATE_002.json`.
<!-- warm-receipt:B_UPDATE_002.json:89ebeceb585f0139702643c5b4216ffe88d469024aa793f6f64d9f05901b0caf -->

## Warm live publication — 2026-11-02T01:05:00.834516+00:00
- Conservatively charged 3426740/7500000 sim-s.

### Warm continuation — ACCEL_EXACT_PLAN_004_FAILED.json
- Exact executed-plan twins: `{}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_004_FAILED.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_004_FAILED.json:007bc8ea9688cc7aba35ab1749353d33600ee6ed7d04b91feb2df295c9a48e8c -->

### Warm continuation — ACCEL_EXACT_PLAN_005_FAILED.json
- Exact executed-plan twins: `{}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_005_FAILED.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_005_FAILED.json:63243f3992fef9b7ee8b06bd1380592e5bbd7ad6ec0112f84d859d24333820c7 -->

### Warm continuation — ACCEL_EXACT_PLAN_006_FAILED.json
- Exact executed-plan twins: `{}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_006_FAILED.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_006_FAILED.json:56eda232125b706f2a7ba6d3da0da4c88d1ced6ffb396888e8f3c33b71eaf579 -->

### Warm continuation — B_EXACT_VIABILITY_99cf50a5c7abc1694be3d527c9ee6cc65d900cea0a353767d86da2594cbdb73c.json
- B u2: exact paired SUMO plan saving mean 145.2 veh-s, 95% world-cluster CI `[-60.80176778292476, 487.8205913035406]`, 15 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_99cf50a5c7abc1694be3d527c9ee6cc65d900cea0a353767d86da2594cbdb73c.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_99cf50a5c7abc1694be3d527c9ee6cc65d900cea0a353767d86da2594cbdb73c.json:d1a48a5473159954368b4347245c2daa1ffbd11e487c3f1d02593d822501a5b1 -->

## Warm live publication — 2026-11-02T01:06:05.898788+00:00
- Conservatively charged 3449020/7500000 sim-s.

### Warm continuation — ACCEL_EXACT_PLAN_003.json
- Exact executed-plan twins: `{"120": {"C_G": -5.606213562714402e-06, "D_G": -5.018876527174143e-05, "D_i": [0.0, 2.053314346994739e-06, -4.2315060454711784e-05, -4.320805601309985e-06, 0.0]}, "300": {"C_G": 0.0001282441990042571, "D_G": -4.351610186859034e-05, "D_i": [0.0, -6.464272883022204e-05, -3.5602570278570056e-05, -7.151500176405534e-05, 0.0]}, "60": {"C_G": 4.793837433680892e-07, "D_G": -4.866109520662576e-05, "D_i": [0.0, -1.959032942977501e-06, -4.6629252210550476e-05, -5.521937964658719e-07, 0.0]}, "600": {"C_G": 0.00022405739582609385, "D_G": 2.9864830139558762e-05, "D_i": [0.0, -0.00012245139805600047, 5.373297608457506e-05, -0.00012547414371510968, 0.0]}}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_003.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_003.json:d55b108ad083bdffe9c080e02f72cdf7c36ab55aa4bbcadbbc2a21439107adc8 -->

### Warm continuation — ACCEL_EXACT_PLAN_004.json
- Exact executed-plan twins: `{"120": {"C_G": 0.13067381548944468, "D_G": 6.178846435998366, "D_i": [0.3218793296973672, 5.7262932908115545]}, "300": {"C_G": 0.05758131807851896, "D_G": 9.010734297089584, "D_i": [0.3218792337229388, 8.631273745288127]}, "60": {"C_G": 0.10886161458029164, "D_G": 3.420459787325399, "D_i": [0.32186853271718974, 2.9897296400279174]}, "600": {"C_G": 0.13743175230410998, "D_G": 9.143711409749812, "D_i": [0.3218792337283958, 8.684400423717307]}}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_004.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_004.json:e5f0cde12eaf3a610044bf5fca11e201d0ef1a7a756bccaff22c9609f1e81bae -->

### Warm continuation — ACCEL_EXACT_PLAN_007_FAILED.json
- Exact executed-plan twins: `{}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_007_FAILED.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_007_FAILED.json:c5e8d1a65b6df060f47d2ec0142e97481b38cb74a8915dbcf05a629c2a18e36a -->

### Warm continuation — ACCEL_EXACT_PLAN_008_FAILED.json
- Exact executed-plan twins: `{}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_008_FAILED.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_008_FAILED.json:db65d973974eae969f4422e3d936957084b097ea37e9f22e578ac370248273dc -->

### Warm continuation — ACCEL_EXACT_PLAN_009_FAILED.json
- Exact executed-plan twins: `{}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_009_FAILED.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_009_FAILED.json:5770e344353ba01ee2b890dff0e2980cafec1076a6c8e8cdfcb2e26ee8ccaea2 -->

## Warm live publication — 2026-11-02T01:07:15.120833+00:00
- Conservatively charged 3462290/7500000 sim-s.

### Warm continuation — ACCEL_EXACT_PLAN_005.json
- Exact executed-plan twins: `{"120": {"C_G": 0.0, "D_G": 0.0, "D_i": [0.0, 0.0]}, "300": {"C_G": 0.0, "D_G": 0.0, "D_i": [0.0, 0.0]}, "60": {"C_G": 0.0, "D_G": 0.0, "D_i": [0.0, 0.0]}, "600": {"C_G": 0.0, "D_G": 0.0, "D_i": [0.0, 0.0]}}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_005.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_005.json:f2a928d688bc5b7369a11b7cc97cb37ddb88429112b6dd880bda10ab1fc84bbc -->

### Warm continuation — ACCEL_EXACT_PLAN_006.json
- Exact executed-plan twins: `{"120": {"C_G": 0.0, "D_G": 0.0, "D_i": [0.0, 0.0]}, "300": {"C_G": 0.0, "D_G": 0.0, "D_i": [0.0, 0.0]}, "60": {"C_G": 0.0, "D_G": 0.0, "D_i": [0.0, 0.0]}, "600": {"C_G": 0.0, "D_G": 0.0, "D_i": [0.0, 0.0]}}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_006.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_006.json:d1251e9dc9a3af439459fc738fb5ed6e093b3007d348f2980fc4671b092cf2aa -->

### Warm continuation — ACCEL_EXACT_PLAN_010_FAILED.json
- Exact executed-plan twins: `{}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_010_FAILED.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_010_FAILED.json:334e98728cf2c7f1dcad3c350e0c728bbea51d50061518cbc1d1c58ab74f0fb4 -->

### Warm continuation — ACCEL_EXACT_PLAN_011_FAILED.json
- Exact executed-plan twins: `{}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_011_FAILED.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_011_FAILED.json:1c2bf23ef13ae3b0c11b696b2e38fe5db6dde810047597169d0ca5ce98d0f097 -->

## Warm live publication — 2026-11-02T01:08:19.506646+00:00
- Conservatively charged 3469570/7500000 sim-s.

### Warm continuation — ACCEL_EXACT_PLAN_007.json
- Exact executed-plan twins: `{"120": {"C_G": -70.79773344534988, "D_G": -82.06016380201527, "D_i": [-4.3038726289523765e-06, -0.09971551787475619, -11.162710534918006]}, "300": {"C_G": -70.84277842292977, "D_G": -82.11204705923774, "D_i": [-4.306133632780984e-06, -0.09970040885127673, -11.16956392132306]}, "60": {"C_G": -62.951096260510894, "D_G": -68.52203141533664, "D_i": [-4.638513928512111e-06, -0.05034652445738175, -5.520583991854437]}, "600": {"C_G": -70.84741214476526, "D_G": -82.11596088526858, "D_i": [-4.306133632780984e-06, -0.09991234323751996, -11.168632091132167]}}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_007.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_007.json:a3e1d31b8814f17d277555b6b15f3ad53bc28d643c414e096673cfdb38806e0f -->

## Warm live publication — 2026-11-02T01:09:18.539981+00:00
- Conservatively charged 3473570/7500000 sim-s.

### Warm continuation — ACCEL_EXACT_PLAN_008.json
- Exact executed-plan twins: `{"120": {"C_G": -0.00020298262097639963, "D_G": -0.032811252434839844, "D_i": [2.7288624551147223e-06, -1.974938641069457e-06, 0.00030393097404157743, 0.004067536177899456, 3.747612936422229e-06, -0.036984238502554945]}, "300": {"C_G": -0.00017034033226082101, "D_G": -0.034062605060171336, "D_i": [2.477740781614557e-05, 2.2219930542632937e-05, 0.0002907063317252323, 0.003958963512559421, 7.822556653991342e-06, -0.03819675446720794]}, "60": {"C_G": 0.0003675223006212036, "D_G": 0.05693955406877649, "D_i": [2.0816778487642296e-06, 5.224700544204097e-06, 0.00015328077734011458, 0.0017959473434530082, 5.503825377672911e-07, 0.054614946886431426]}, "600": {"C_G": -17.07724909040553, "D_G": -12.377576812825282, "D_i": [3.742568152185413, 3.7167922478402033, 8.118825157231186, 0.5129231108730892, 0.006844088624347933, -11.398280479173991]}}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_008.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_008.json:bfb63a6f6767ebf4ca2dbdde28760634ad861ccbeb627644390d3b12de409de4 -->

## Warm live publication — 2026-11-02T01:10:23.502682+00:00
- Conservatively charged 3481850/7500000 sim-s.

### Warm continuation — ACCEL_EXACT_PLAN_009.json
- Exact executed-plan twins: `{"120": {"C_G": 0.0, "D_G": 0.0, "D_i": [0.0, 0.0]}, "300": {"C_G": 0.0, "D_G": 0.0, "D_i": [0.0, 0.0]}, "60": {"C_G": 0.0, "D_G": 0.0, "D_i": [0.0, 0.0]}, "600": {"C_G": 0.0, "D_G": 0.0, "D_i": [0.0, 0.0]}}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_009.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_009.json:4fb44a40f49731a78b52c31a1f181037f9bad5b091ffc22c469352486ef13d5d -->

### Warm continuation — ACCEL_EXACT_PLAN_010.json
- Exact executed-plan twins: `{"120": {"C_G": -0.000137253902721568, "D_G": 2.2410635638625536, "D_i": [2.2403303424471233, 0.0008704753181518754]}, "300": {"C_G": -0.000686486888298532, "D_G": -53.89407006950205, "D_i": [-53.89346310379551, 7.952118176035583e-05]}, "60": {"C_G": 0.0, "D_G": 0.0, "D_i": [0.0, 0.0]}, "600": {"C_G": -0.000686478684656322, "D_G": 1363.1895355773158, "D_i": [1363.1901425434626, 7.951253792271018e-05]}}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_010.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_010.json:63ec92c63617cd1f016471745d7a78a9546a79bd331d2ee0a5efaf0121424b26 -->

## Warm live publication — 2026-11-02T01:11:32.946959+00:00
- Conservatively charged 3481850/7500000 sim-s.

### Warm continuation — ACCEL_EXACT_PLAN_011.json
- Exact executed-plan twins: `{"120": {"C_G": 3.601599019020796e-06, "D_G": 0.0034052084456561715, "D_i": [0.0, 0.0, 0.0034016068466371507]}, "300": {"C_G": 3.407079930184409e-06, "D_G": 0.0035613359286799096, "D_i": [0.0, 0.0, 0.003557928848749725]}, "60": {"C_G": 5.020457592763705e-06, "D_G": 0.006628526091390086, "D_i": [0.0, 0.0, 0.006623505633797322]}, "600": {"C_G": 3.407079930184409e-06, "D_G": 0.003558316548151197, "D_i": [0.0, 0.0, 0.0035549094682210125]}}`.
- Receipt: `coopmwm/warm_live/ACCEL_EXACT_PLAN_011.json`.
<!-- warm-receipt:ACCEL_EXACT_PLAN_011.json:881c18e02701857965e8e1d35d14dd5cd73868a445c358c1abde89e607d76006 -->

## Warm live publication — 2026-11-02T02:45:21.526009+00:00
- Conservatively charged 3507050/7500000 sim-s.

### Warm continuation — WARM_B_FAILED_T1.json
- `{"error": "ValueError('new durable guided engineering STOP')", "timestamp_utc": "2026-11-02T01:41:36.626856+00:00"}`
- Receipt: `coopmwm/warm_live/WARM_B_FAILED_T1.json`.
<!-- warm-receipt:WARM_B_FAILED_T1.json:dba3c8766171161a1344f6334433f8855d915df45f5ecfb680aadec782fa007e -->

### Warm continuation — WARM_GUIDED_STOP_V2.json
- `{"error": "ValueError('GPU failed; no inference retry')", "source": "capacity broker", "unix": 1793583695.5195312}`
- Receipt: `coopmwm/warm_live/WARM_GUIDED_STOP_V2.json`.
<!-- warm-receipt:WARM_GUIDED_STOP_V2.json:75fc7070edf55a16763460b72afd20094720ee369df51b27d6dd8a218ddf6f02 -->

### Warm continuation — WARM_R_FAILED_Q1.json
- `{"error": "ValueError('new durable guided engineering STOP')", "timestamp_utc": "2026-11-02T01:41:45.322047+00:00"}`
- Receipt: `coopmwm/warm_live/WARM_R_FAILED_Q1.json`.
<!-- warm-receipt:WARM_R_FAILED_Q1.json:da0aeaabe12c92d6b41685790c3d97a46fe5a7c2607902fd8051f7e9c0e25245 -->

## Warm live publication — 2026-11-02T06:36:14.116206+00:00
- Conservatively charged 3541050/7500000 sim-s.

### Warm continuation — B_EXACT_VIABILITY_0c8958f6326cb646c9e79c01203898110ddc2b43c690fa291e7776dda5c7ff3e.json
- B u3: exact paired SUMO plan saving mean 136.1 veh-s, 95% world-cluster CI `[-56.946541822816314, 456.72515283199084]`, 16 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_0c8958f6326cb646c9e79c01203898110ddc2b43c690fa291e7776dda5c7ff3e.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_0c8958f6326cb646c9e79c01203898110ddc2b43c690fa291e7776dda5c7ff3e.json:3ca2ad711d6f43be090adc92a8d8ba586b2bf5314788c4db6be57e2442df0e7e -->

### Warm continuation — B_EXACT_VIABILITY_2c4fde9b4d1caff8568aebfcbc7d01052cc337806741799b2c7b50da4aaf2fcf.json
- B u3: exact paired SUMO plan saving mean 130.7 veh-s, 95% world-cluster CI `[-33.564138422912364, 391.64083244841953]`, 20 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_2c4fde9b4d1caff8568aebfcbc7d01052cc337806741799b2c7b50da4aaf2fcf.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_2c4fde9b4d1caff8568aebfcbc7d01052cc337806741799b2c7b50da4aaf2fcf.json:33c5f3ef424dd378f20c791638459b2d469fc6f7b259adacb02e9d4e6a77032e -->

### Warm continuation — B_EXACT_VIABILITY_4170a33317214d9cbd24173b30c7da2507b03967c266de28cc533b05a4dc54df.json
- B u3: exact paired SUMO plan saving mean 108.9 veh-s, 95% world-cluster CI `[-43.85938748656751, 347.3596921264801]`, 22 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_4170a33317214d9cbd24173b30c7da2507b03967c266de28cc533b05a4dc54df.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_4170a33317214d9cbd24173b30c7da2507b03967c266de28cc533b05a4dc54df.json:2cc80066096668f792456196f6dd6fab7af5980f8ee650d435df1c506c5153e9 -->

### Warm continuation — B_EXACT_VIABILITY_42010834abe6a6f57ae77e4389afaf04c0c8c47052775cf64347f55010d8fb74.json
- B u3: exact paired SUMO plan saving mean 96.94 veh-s, 95% world-cluster CI `[-44.60122551151658, 319.7964441086416]`, 24 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_42010834abe6a6f57ae77e4389afaf04c0c8c47052775cf64347f55010d8fb74.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_42010834abe6a6f57ae77e4389afaf04c0c8c47052775cf64347f55010d8fb74.json:5e8a9019f2813a5dfeb85b070010185d5e7837d66ef494e39d5f4e4bfef1c5ab -->

### Warm continuation — B_EXACT_VIABILITY_4eb06ac1d63bf1f522fe30eae36ee93491b5b8a89b59cafa8a67f30f81d92893.json
- B u3: exact paired SUMO plan saving mean 137.4 veh-s, 95% world-cluster CI `[-41.11234860466027, 424.1336676226603]`, 18 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_4eb06ac1d63bf1f522fe30eae36ee93491b5b8a89b59cafa8a67f30f81d92893.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_4eb06ac1d63bf1f522fe30eae36ee93491b5b8a89b59cafa8a67f30f81d92893.json:dc5b53087c204e577f3bfbe18623d8167e8b4bb68b54d0cb101d58e19cc72607 -->

### Warm continuation — B_EXACT_VIABILITY_52feb3bc5373c7e5c2bf34dad70cb784dfd2ee387e2b8b70bb01127cfc300ff4.json
- B u3: exact paired SUMO plan saving mean 137.5 veh-s, 95% world-cluster CI `[-35.693173042451086, 409.1304890657517]`, 19 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_52feb3bc5373c7e5c2bf34dad70cb784dfd2ee387e2b8b70bb01127cfc300ff4.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_52feb3bc5373c7e5c2bf34dad70cb784dfd2ee387e2b8b70bb01127cfc300ff4.json:482a63485e1adb15b2125a8ce7c9aa1e8df166423100a34e54849630f54b4087 -->

### Warm continuation — B_EXACT_VIABILITY_a61d6b15e89623d46fc290a1ed0a240ec1502ecfe7b6df3dbf6ade28f3a01849.json
- B u3: exact paired SUMO plan saving mean 123.4 veh-s, 95% world-cluster CI `[-33.00448622954187, 372.1497608971724]`, 21 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_a61d6b15e89623d46fc290a1ed0a240ec1502ecfe7b6df3dbf6ade28f3a01849.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_a61d6b15e89623d46fc290a1ed0a240ec1502ecfe7b6df3dbf6ade28f3a01849.json:c6bd151efd887adf880e1837a279fb24f497decd64db4b62a530ebd050f8ed69 -->

### Warm continuation — B_EXACT_VIABILITY_c9674081b758a5ed2173956c653eb43d4b2b66ac706beb3a373a72dce8915b53.json
- B u3: exact paired SUMO plan saving mean 109.4 veh-s, 95% world-cluster CI `[-37.67019314200868, 339.57944210475034]`, 23 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_c9674081b758a5ed2173956c653eb43d4b2b66ac706beb3a373a72dce8915b53.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_c9674081b758a5ed2173956c653eb43d4b2b66ac706beb3a373a72dce8915b53.json:ba5ecbc8d85201274facdbee30dd9d56e0ea191756b1724d477ffbc60d18dcc7 -->

### Warm continuation — B_EXACT_VIABILITY_d1cbca5a52a9b461174bb28ec9258818f09c66c595d1478e445caf4cbdbe7b95.json
- B u3: exact paired SUMO plan saving mean 128.1 veh-s, 95% world-cluster CI `[-52.51988902238005, 430.9817550540364]`, 17 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_d1cbca5a52a9b461174bb28ec9258818f09c66c595d1478e445caf4cbdbe7b95.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_d1cbca5a52a9b461174bb28ec9258818f09c66c595d1478e445caf4cbdbe7b95.json:c96a9c93204cec478b7e4af55223db5c42ca51bba5387eb4a346ae9e4d2bbfc1 -->

### Warm continuation — B_EXACT_VIABILITY_fc40713c0db7757fb38449adffbee7c2884fa69a2f2f6fae927d2cc1986b83f3.json
- B u3: exact paired SUMO plan saving mean 93.06 veh-s, 95% world-cluster CI `[-43.75592948832516, 310.2947246887355]`, 25 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_EXACT_VIABILITY_fc40713c0db7757fb38449adffbee7c2884fa69a2f2f6fae927d2cc1986b83f3.json`.
<!-- warm-receipt:B_EXACT_VIABILITY_fc40713c0db7757fb38449adffbee7c2884fa69a2f2f6fae927d2cc1986b83f3.json:60d90fa2c6aecfec7c1e55ea8f5ac5affc3aa023986d4fb7d95d9057c47bdc29 -->

### Warm continuation — B_UPDATE_003.json
- B u3: queries/accepted/executed 34/26/10; wall 1.945e+04s; next beta 0.751. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 26, "executed": 10, "queries": 34}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Exact-grade queue/lag: `{"failed": 0, "graded": 10, "max_lag_s": 19.058584928512573, "mean_lag_s": 10.514461827278136, "orphaned": 0, "overlap_unresolved": 0, "pending": 0, "running": 0, "stopped": 0, "update": 3}`; final ETA 113.4h.
- Receipt: `coopmwm/warm_live/B_UPDATE_003.json`.
<!-- warm-receipt:B_UPDATE_003.json:acec7dd6327cba304dcdb21e4e971e3f0b518a4c87a6bf9e3d97d1301cac463f -->

### Warm continuation — B_VIABILITY_003.json
- B u3: exact paired SUMO plan saving mean 96.94 veh-s, 95% world-cluster CI `[-45.644077496616276, 321.15091081315353]`, 24 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/B_VIABILITY_003.json`.
<!-- warm-receipt:B_VIABILITY_003.json:830efd4034f651202b8db61d2b47ad1bae0b65a36f694c4ebaa9d5ca94f97ca6 -->

## Warm live publication — 2026-11-02T07:29:17.592528+00:00
- Conservatively charged 3601530/7500000 sim-s.

### Warm continuation — R_UPDATE_002.json
- R u2: queries/accepted/executed 114/114/0; wall 91.67s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 114, "executed": 0, "queries": 114}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Exact-grade queue/lag: `{"failed": 0, "graded": 12, "max_lag_s": 17.956080436706543, "mean_lag_s": 11.755984206994375, "orphaned": 0, "overlap_unresolved": 0, "pending": 0, "running": 0, "stopped": 0, "update": 2}`; final ETA 0.5602h.
- Receipt: `coopmwm/warm_live/R_UPDATE_002.json`.
<!-- warm-receipt:R_UPDATE_002.json:14e99afd71c27c362e8c5859f87cffa726b86464b57cb90563af854351d88468 -->

### Warm continuation — R_UPDATE_003.json
- R u3: queries/accepted/executed 13/13/0; wall 533.7s; next beta 0. SUMO-only reward.
- Executions by type/regime: `{"MPPI|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "MPPI|ordinary": {"accepted": 13, "executed": 0, "queries": 13}, "O1|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O1|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O2|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O2|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O3|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O3|ordinary": {"accepted": 0, "executed": 0, "queries": 0}, "O4|merge-dense": {"accepted": 0, "executed": 0, "queries": 0}, "O4|ordinary": {"accepted": 0, "executed": 0, "queries": 0}}`.
- Exact-grade queue/lag: `{"failed": 0, "graded": 10, "max_lag_s": 19.058584928512573, "mean_lag_s": 10.514461827278136, "orphaned": 0, "overlap_unresolved": 0, "pending": 0, "running": 0, "stopped": 0, "update": 3}`; final ETA 3.113h.
- Receipt: `coopmwm/warm_live/R_UPDATE_003.json`.
<!-- warm-receipt:R_UPDATE_003.json:bb314cdb79ce8f6c206408f40dc4989ed9b901d40bcf1a45106bcb72d76dcd4a -->

### Warm continuation — R_VIABILITY_002.json
- R u2: exact paired SUMO plan saving mean -1046 veh-s, 95% world-cluster CI `[-2873.630369603925, -0.08526310693014238]`, 3 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/R_VIABILITY_002.json`.
<!-- warm-receipt:R_VIABILITY_002.json:256b1784d11f1c42b561f1feaf2fb24f31032036734552de1f9f3ef83c97707e -->

### Warm continuation — R_VIABILITY_003.json
- R u3: exact paired SUMO plan saving mean -1046 veh-s, 95% world-cluster CI `[-2873.630369603925, -0.08526310693014238]`, 3 graded plans. Positive saving is better. Ungraded plans excluded.
- Critic diagnostics are labelled separately in the receipt and never grade plans. Receipt: `coopmwm/warm_live/R_VIABILITY_003.json`.
<!-- warm-receipt:R_VIABILITY_003.json:fb04669f66a74de7d3792ac65704599f191b34ed85653adb509abab67dfaf047 -->

### Warm continuation — WARM_B_FAILED_I1.json
- `{"error": "ValueError('new durable guided engineering STOP')", "timestamp_utc": "2026-11-02T06:57:15.276917+00:00"}`
- Receipt: `coopmwm/warm_live/WARM_B_FAILED_I1.json`.
<!-- warm-receipt:WARM_B_FAILED_I1.json:ac490f96553656a328e87434b74860d8c7d3452dcfb22c488e84ab335a0ad9ab -->

### Warm continuation — WARM_GUIDED_STOP_V3.json
- `{"arm": "R", "error": "AssertionError('R count/regime/group-size/duration gate')", "timestamp_utc": "2026-11-02T06:43:03.905193+00:00"}`
- Receipt: `coopmwm/warm_live/WARM_GUIDED_STOP_V3.json`.
<!-- warm-receipt:WARM_GUIDED_STOP_V3.json:d0e9bca2ff5bccfddd740d4be7961ea0886663a423505cdcfc2814d3669149a9 -->

### Warm continuation — WARM_R_FAILED_I1.json
- `{"error": "AssertionError('R count/regime/group-size/duration gate')", "timestamp_utc": "2026-11-02T06:43:03.902197+00:00"}`
- Receipt: `coopmwm/warm_live/WARM_R_FAILED_I1.json`.
<!-- warm-receipt:WARM_R_FAILED_I1.json:e280d2737807b5f6339c1dd87ae3f94fbe57a48fddc64653294f4986ece75a40 -->

### Warm continuation — WARM_R_PAUSED.json
- `{"charged_sim_s": 3572730, "full_endpoint": 24, "reason": "registered checkpoint boundary complete; final evaluation waits for fixed primary endpoints", "timestamp_utc": "2026-11-02T07:09:25.480589+00:00", "update": 3}`
- Receipt: `coopmwm/warm_live/WARM_R_PAUSED.json`.
<!-- warm-receipt:WARM_R_PAUSED.json:a74c3613d6e5eedcb89a64b452a4e44c517b5f8b8d7be0e70a9e053357d26e3e -->
