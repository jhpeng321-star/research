# CURRENT MAIN01:28 PARTIAL ANALYSIS COMPLETE — exit, no compute armed

- MAIN01:28 supersedes the historical complete-panel-only reporting restriction:
  no further SUMO/GPU; compute registered statistics on all-arm-complete worlds.
- Analysis commit `08de06cc`; `docs/paper_results/C1SWITCH.md` now reports PARTIAL
  n5 worlds969112..969116, all8arms,15measured decisions/model/150branches.
- `docs/paper_results/c1switch/partial_results.json`, `partial_worlds.csv`,
  `partial_decisions.csv` preserve every estimate/CI and selected input hashes.
- Primary WM-BB regret60 +5.101695[-9.388681,16.745943]veh-s;
  WM-best-fixed +8.231654[1.692420,18.476727],best=MAPPO169 with draw reselection.
- CPU12PASS/RuffPASS; newSUMO0/GPU0. No full16world efficacy verdict.
- Do not run the historical `finish report` generator over the amended report.
  Receipt-only reproduction is `python -B -m tools.c1switch.partial`.
- Task is complete under MAIN01:28. Do not poll/relaunch/complete missing worlds.

## Historical shared_v10 TERMINAL GPU-BUDGET STOP

- Read `docs/paper_results/C1SWITCH.md`, final design line79,
  `V10_COLLECTION.json`, `GPU_ACTUAL_SEGMENTS.json`, `V10_RETAINED_QA.json`.
- All owned cpuQ supervisor259509/children and gpuR715850 exited; local relays
 30024/32612/28584 absent, terminal closed-lock receipts retained. No retry armed.
- Charged=closed actual383190sim-s.107fully evaluated episodes:
  WM5/16;BB16/16;native/CTRL0/MAPPO/PI/CIRCLES16/16 EACH;random6/6.
 315successful branches,WM56/160committed picks,BB160/160;all preserved.
- Original10episodes+231branches never repeated; superseded WM12450,
  prior abandoned17820 and terminal abandoned960 remain charged, no refund.
- Exact ALL-phase GPU allocation wall14382.225540s/3.995063h; conservative
  admission14396.231360s/3.998953h. Combined4117s alarm ended the shared phase.
- Standard independent full-rerun completion projection500640sim-s exceeds
  original500000cap by640, before any further failures; not a universal lower
  bound for a future validated shared-physics plan. All random episodes retained
  by the trimmed plan are already complete, so dropping them cannot refund cost.
- Retained-data QA PASS:107episodes/315unique branches/all16available-arm
  warmups,exact clocks/schedules/live prefixes/finite metrics/accounting;
  zero collisions/teleports. CPU49cpuQ/50gpuR;local49PASS/4hostskips;ruffPASS.
- Full-panel primary/bootstrap/paired efficacy comparisons NOT EVALUATED.
  This is an operational/budget stop, not a positive/negative model verdict.
- New MAIN authority for GPU continuation AND a validated SUMO-cap recovery
  amendment is needed before any compute. Never re-stage or re-dispatch existing
  roots. Reuse complete PHYSICS, results, forecasts and every historical debit.
- Local terminal mirror `evidence/shared_v10_panel` contains all complete results
  and closed job receipts. `finish collect/report/verify` are receipt-only; do not
  run any old launch command below. Existing model/controller/seed rules unchanged.

## Historical shared_v10 DISPATCHED

- Build31313092/stagea0d078f5; design lines78–79, V10_PLAN/STAGE/CPU/PROGRESS.
- cpuQ `/home/ubuntu/c1switch20261031/shared_v10`, own supervisor259509;
  14 own nice19 single-thread jobs, at most2WM episodes active. Retains completed
  episode PHYSICS even if its evaluation branches are pending; no physics repeats.
- gpuR `/root/c1switch20261031/shared_v10`, ONE owned GPU process715850;
  both frozen adapters share ONE4117s alarm. Started1793462789.657412;
  conservative prior10279.222074s, maximum14396.222074/14400owner GPU-s.
- Local own relay30024 at `evidence/relay_shared_v10a` is REQUIRED; transport
  retries don't repeat physics; terminal GPU STOP also closes paused CPU waiters.
  Initial shared relay32612 was cooperatively closed before GPU allocation.
- Latest command `python -B -m tools.c1switch.sharedops progress` only inspects.
  NEVER stage/launch/relay/main_panel again; no duplicate dispatch. Old v4–v9
  deployments/STOPs remain immutable. gpuA services and v9 original-host services
  are closed. All old successful physics/forecasts imported once.
- Exact499680sim-s/reserve320: Step047010;WM190830(incl superseded12450 and
  abandoned5880);BB184440(incl abandoned11940);random5400;fivefixed14400each.
  Random only969112–969117; last10worlds omitted BEFORE any random outcomes.
- CPU49cpuQ/50gpuR PASS; BOTH original-host rawK8 full60s parity gates retained,
  frozen scientific sources/attestations verified at shared model loading.
- GPU throughput may prevent completion:142WM/10BB uncached picks at staging;
  original-host forecast~78s. Owner cap still enforced; no partial-panel efficacy.
- At terminal run `python -B -m tools.c1switch.finish collect`, then `... report`.
  It refuses live owners/mismatched closed accounting and only bootstraps a fully
  complete registered panel. Write final REGISTRATION line/commit afterwards.
  `finish.py` is local analysis code added after staging; do not mutate live stage.

## Historical MAIN23:25 recovery DISPATCHED

- Latest build/stage `c6443e7a`; read design lines74–77 and `V9_PROGRESS.json`.
- cpuQ recovery `/home/ubuntu/c1switch20261031/recovery_v8`, supervisor211684,
  14 episode workers then 14 independent evaluation branch workers, nice19.
- gpuR `/root/c1switch20261031/recovery_v8`, WM709203/BB709259; BOTH full60s
  anonymous-draw rawK8 original-host parity PASS. cpuQ45/gpuR46CPU tests PASS.
- Local own relay28584 at `evidence/relay_recovery_v8` is required. It targets
  gpuR, not gpuA. Its transport errors are retried without repeating physics.
- `python -B -m tools.c1switch.resumeops progress` inspects current receipts.
  NEVER run stage, relocate, import_receipts, GPU launch or main_panel again.
- Exact499260 sim-s; random last3 worlds969125/126/127 omitted prospectively.
  All16WM/BB/fixed worlds preserved. Reuse10episodes+231branches;11100abandoned
  original sim-s and12450superseded WM remain charged, no refunds.
- SharedGPU prior9718.965539s includes gpuA failed parity57.231276s;
  newWM4470/BB200 partitions imply<=14388.965539s/14400owner cap. Throughput
  at risk (WM original-host parity78.22s); caps remain enforced.
- gpuA's separate recovery_v8 root failed cross-host bitwise parity, both own
  processes exited. Preserve those STOPs, every v4–v7 root and source/ledger.
- New inference uses frozen v7 scientific adapter; original forecasts cached
  only with exact observable history equality and committed pick/prefix checks.
- Physics writes PHYSICS.json; evaluation-only full reruns occur after episode
  choices. Final RESULT.json requires all registered five-way branches.
- Final analysis must account for11100 abandoned +12450 superseded +47010Step0
  separately, and random n13 vs all other arms n16. No partial-panel efficacy.

## Historical MAIN23:04 continuation: v7 STOPPED / BUDGET BLOCKED

- First read `V7_INTERRUPTION_AUDIT.json` and the last four lines of the design.
  All audited remote owned PIDs exited. BB hit its hard3600s GPU allocation cap;
  subsequent STOPs and WM timeouts closed all active SUMO jobs. No restart yet.
- Cumulative227310sim-s charged=closed actual;10complete episodes(9BB/1WM),
  231successful branches,14partial episodes and9partial branches retained.
  Standard independent full-rerun recovery projection501960sim-s exceeds500000;
  BB remaining GPU budget0. Preserve original v7 adapter and every old STOP.
- Local owned relay13556 cooperatively stopped after receipt inspection. Its
  SSH error is recorded but not proven to be the initial cause of the panel stop.
- MAIN23:04 requested same-plan resume, but original BB hard cap is exhausted.
  A new registered budget/protocol decision is required before compute resumes.
- `python -B -m tools.c1switch.recovery` is a CPU-only preflight using the saved
  audit. It reconstructs all608registered runs, preserves every partial debit,
  and writes `V7_RECOVERY_ADMISSION.json` with `dispatch_authorized=false`.
  Twelve CPU tests PASS (`V7_RECOVERY_CPU.json`). This does not lift old STOPs
  or establish a new authorized recovery plan. Check fresh receipts before any
  future implementation. Original frozen v7 scientific sources remain untouched.

## Historical MAIN20:52 main dispatch

- Start with `python -B -m tools.c1switch.continuation progress`; do not stage,
  launch, duplicate, or retry an existing operation. v4/v5/v6 STOPs remain intact.
- Design/build `c1908877`, immutable staging `8a9aa534`, all canary/parity/reuse
  gates committed `bbfd5674`. Source binding
  `dd4edb41d4f6c111a5fd09949721b5f3f592dad44d04f6014c0b118e1fcd8cf3`.
- cpuQ `/home/ubuntu/c1switch20261031/main_v7`: main supervisor3984657;
  initial14BB episode workers969113..969126,112pending,2canary episodes reused.
- gpuR `/root/c1switch20261031/main_v7`: own WM617609 and BB634799 services;
  local owned relay13556 at `evidence/relay_main_v7` is REQUIRED for live RPC.
- Pre-main cumulative84360sim-s charged=actual. Total490860=Step047010+
  WM184950(incl superseded original12450)+BB172500+random/five fixed14400 EACH.
  Remaining406500, reserve9140 unallocated; no refunds/seed replacements.
- Both full60s original-draw rawK8 GPU parity gates PASS. All10 corrected BB
  picks identical: its original episode/15branches reused. Corrected WM-only
  first-world episode/15branches PASS; old WM schedule superseded, retained.
- Paired native warmup exact;30live-prefix/60metric gates PASS;0collisions/
  teleports. Current model/control/observable adapter is the frozen v6 version.
- GPU hard caps WM10800/BB3600s include all prior services/startup/idle. v7 prior
  conservative2557/1664s; allocations started Unix1793451558.0146513 and
  1793453075.1145563, remaining phase8243/1936s. No cap extension authorised.
- Measured WM81.59s/forecast puts complete-panel GPU throughput at risk; MAIN
  informed in REGISTRATION. Hard stops remain active; no partial-panel efficacy.
- Current owned GPU memory observed1012MiB combined. cpuQ <=28SUMO workers
  including branch children, nice19/single-thread; no peer process actions.
- At resume check STOPs, supervisor/child PIDs, RPC relay/error receipts,
  reservations vs closed actual, and results before any action. Current evidence:
  `V7_MAIN_DISPATCH.json`, `V7_PROGRESS.json`, `V7_CANARY_GATE.json`,
  `V7_GPU_PARITY.json`, `V7_PAIRED_WARMUP_GATE.json`, `V7_THROUGHPUT.json`.

## Historical MAIN 19:12 integration continuation

- Resumed clean `c4/c1switch` at `1e521ef3`; Step0 results retained.
- `cpuops inspect`: cpuQ owned PID3525930 absent; COMPLETE16 states and both
  bitwise/headroom gates PASS; gpuR v2 never dispatched, legacy debit930 retained.
- Registered main431400 + Step047010 =478410 sim-s; reserve21590 unallocated.
- gpuR admission probe:22340 MiB free of24564 (a probe, not a reservation).
- Remaining gates: executable WM2/BB checkpoints, causal controller-state
  continuation, cross-host transport, one-world integration canary.
- Inference must use anonymous observable tracking and genuine121-tick history.
  Learned entrant mix is configured but dormant with entries OFF; no source or
  unseen-arrival forecast is claimed. Selection predicts current-belief fleet only.
- PI continuation must transfer the committed raw recurrence (not issued clipped
  speed); MAPPO continuation must transfer its CAV command memory by self ID.
  A changed controller resets these states exactly as the real Handover does.
- Canary will be a retained first main-world episode, not an extra seed/world;
  successful completed work is reused and charged to its registered episode arm.
- No inference/SUMO dispatched by this inspection; no existing jobs duplicated.
