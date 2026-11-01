# S-COOPMWM — live checks

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
