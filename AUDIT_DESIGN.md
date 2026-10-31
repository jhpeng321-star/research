# S-COOPMEASURE — prospective cooperation audit

Authority: owner 2026-10-31 15:45, REGISTRATION.md line5544; branch
`c4/coopmeasure` from `c4/sumocf@bf8c1e20`. Headline: **measured cooperation**.
No positive fleet gain or visually coordinated behaviour substitutes for C_G.

## Estimand, interventions and controllers
- **Binding 17:20 owner amendment:** primary arms are speed-only MAPPO and
  wall-v2.1 mechanism positive control, plus S-SUMOCF's trained policy when an
  immutable checkpoint exists, paired on the same worlds/groups/design. PI,
  CIRCLES and CTRL0 are historical/optional only after primary work and budget.
  The original five-arm specification below is retained as execution history.
- D(S)=J(B)-J(S), C_G=D(G)-sum_i D(ONLY:i), positive beneficial interaction.
  Exact same-state SUMO branches, not additive model approximations.
- B means native SUMO car-following (IDM population), native lane decisions, no
  controller acceleration request. A disabled member receives no command; ownership
  explicitly releases previous control to native (`setSpeed(-1)` inherited path).
  This differs from CTRL0's always-commanded native-acceleration adapter. At the
  end of the intervention window all vehicles are released to native, continuing
  the same branch for the remaining outcome horizon. Entrants are always native.
- Controllers: original speed-only MAPPO iter169 sha256 d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca;
  PI external-U and CIRCLES damping through frozen env fallbacks; CTRL0 via exact
  zero.py24095a10; wall-v2.1-closure through `WallGroupSession`. Preserve frozen
  runtime b3c031ad, original safety caps, speedMode27 and native lanes.
- Wall provenance: docs/INDEX.md and WM_CHANGE_LOG.md read; commit5022d92 merges
  3b2bb094, `traffic_shepherd/wall_group.py`. Closure: front v_wall; rear
  v_wall+min(2,0.1*gap), acceleration clip[-3,1.5], 6m/1m/s assembly tolerance,
  60s timeout, original span/rear-risk/missing-member/lane-coverage aborts.
  Fixed planner request v_wall=12m/s; choose the closest distinct-mainline-lane
  pair with span<=75m inside G, geometric tie-break only, before acting. Other G
  members are inert for wall. Solos retain the frozen *full wall roster/context*
  and mask its commands to ONLY:i; never replace a solo by a refused singleton
  request. Record absent wall coverage and aborts, never replace worlds. This is
  a known coordinated *mechanism* positive control, not an assumed positive C_G.

## Fresh worlds, selection, SNR dependency
- Reserve TRAIN969650-969699. Main969650-969677; canary969699 (.4,2700).
  `tools/coopaudit/design.py` is the exact mapping: six repeat-major worlds in
  each (.2/.4)x(2400/2700) cell, then (.05,2400),(.1,2700),(.2,1900),(.4,1900).
  Disjoint from969200-391,969400-543,969550-599,969600-624,981200-981299;
  never964000-964199,962xxx,977xxx, validation/calibration or sealed data.
- Native prefix t0=300s, one frozen G/world, no outcome-based reselection.
  Only RSU-observable identity/position/lane/junction and CAV eligibility. Zone
  s in[-150,75.44]m (aligned to the binding SUMOCF amendment), nearest |s|
  then identity. Missing |G|<2 stays missing.
- **WAIT_SUMOCF**: adopt the S-SUMOCF selected group size and 10/30/60s window
  from its TRAIN-only SNR table before main dispatch. Prefer the registered
  selected design even if it differs from expectations. All-zone groups are
  supported without an artificial four-member cap. Freeze selected table SHA,
  local-outcome definition and any geometry difference before any main SUMO.
- Reuse COOPMEAS2 saved RNG + fresh-process loadState + Python snapshot,
  identity-keyed requests, masked controller memory/dwell and world bootstrap.
  Bitwise trajectory+metrics repeat canary mandatory. No silent full-rerun fallback.

## Outcomes and analysis
- Team fleet+source delay at300/600s; local cohort = observable vehicles in the
  merge segment s[0,75.44] at a pre-control report during window+120s. Local J is
  the original Measurement tracker's on-road+source delay for that cohort at
  window+120s (exact S-SUMOCF oracle.py lines185-208 convention). Retain the
  fixed-zone integral and approach-cohort diagnostics separately. This definition
  is frozen before any production; the earlier canary-local number is an
  approach-cohort diagnostic, not this production crossing-cohort outcome.
- For every controller and outcome: world mean C_G and95% percentile stratified
  world-bootstrap CI (5000draws, seed20261031, shared resamples), fraction C_G>0,
  D_G and sumD_i, all-world and high-share cohort/cell coverage. Tiny/null gains
  retained, numerical-null resolution1e-6, missing never zero. Safety: collisions,
  teleports, dropped steps, TTC<3s and hard-braking veh-s for B/G/ONLY branches.
- Per-group traces: own-root `jobs/sSEED/traces/CONTROLLER_BRANCH.jsonl.gz`;
  members and observed nearby ramp vehicles (within300m of a member or in zone),
  t0-30 through t0+window+30, 0.5s physics frames, identity/s/lane/v/a, membership,
  executed commands and wall events; no hidden types. Document for S-COOPBEHAV.

## Budget and gates
- Combined **500000 actual sim-s HARD CAP**, including all prefixes, canaries,
  failed attempts and any repeats. Durable fsynced precharge under exclusive
  ledger lock before each attempt; no refunds/retries. No other stream's budget.
- Native-prefix census:28*300=8400sim-s; exact group cardinality determines
  full-panel admission before any main branches. Each production worker reruns
  ONE300s native prefix (8400additional sim-s), checks census state/group hashes,
  then saves/restores it for ALL branches, with no branch-prefix reruns. This
  bounds resident workers to one and avoids serialising live TraCI/ONNX objects. Each
  production world costs300 +600*(2+5*(|G|+1)) for B/Brepeat and five(G+all solos).
  Canary separately bounded at20000sim-s. Main panel must fit remaining cap as
  a complete fixed allocation; otherwise STOP for budget/design amendment,
  never truncate groups/worlds or silently change600s to300s after outcomes.
- Before launch: both frozen seed guards + exposure audit, CPU algebra/group/
  wall wiring tests, runtime/wrapper hashes, same-state/RNG canary, native and
  controlled repeated trajectory identity, wall command/abort evidence.
- gpuR spare CPUs max2 workers nice19, one thread, CUDA hidden; pause new work
  if load>20 or S-SUMOCF priority active (explicit cooperative admission file).
  cpuQ optional max2 only with priority admission. Never signal other PIDs.
  STOP/infrastructure/determinism/lane/safety anomaly blocks new dispatch.
- Commit after design, CPU implementation, canary, and completed run/report.
  Final docs/paper_results/COOPAUDIT.md <=60lines and one registration entry.
  Exposure disclosure: tools/coopaudit/evidence/EXPOSURE_NOTE.md.

## Zero-step infrastructure correction (before any SUMO step)
The first uncharged import-only launch failed on a namespace import. The next
launch precharged300 but the environment constructor refused because SSH's PATH
did not contain the existing SUMO executable. Actual0sim-s; original reservation,
START and traceback retained. One manual setup repair (same969699, new attempt2
identity, newly precharged; no refund) is permitted within the same20000 canary
and500000 total caps. PATH/executable/import checks now precede reservations.
No retry after any simulator step or outcome is authorised. This is an explicit
infrastructure amendment before observations, not a scientific replacement world.

## Step3 verified canary (2026-10-31)
gpuR attempt2 seed969699:2280actual/2580charged including retained300 zero-step
reservation. Native LIVE/B/B_repeat trajectories AND metrics bit-identical;
wall/MAPPO GROUP repeats bit-identical. Wall106 commands, assembly at307s,
native handback at353s when a member exits coverage. No collision/teleport/drop.
Wall C_G180=-154.834 and local C_G=-173.709veh-s on this one world: the
coordinated mechanism is detected but is NOT beneficial synergy here. MAPPO
C_G180/local=0. These are mechanical-canary observations, not panel estimates
or controller selection. Receipt tools/coopaudit/evidence/CANARY_VERIFIED.json.

## Dependency-controlled continuation
`tools/coopaudit/continue_audit.py` takes an exclusive local lock and checks the
requested SUMOCF_DESIGN.md every120s. It requires a published measured-SNR-results
section AND the complete PASS27-row machine receipt, verifies its predeclared
winner, binds source-document/table hashes and source commit, and commits that
binding before dispatch. gpuR is the current authoritative probe host (SUMOCF's
capacity refinement), with cpuQ support only if that registered host changes.
Then it runs the native census, admits only the complete exact-cost panel under
500000 including2580 already charged, executes on one spare nice19 GPU-host CPU,
and exports complete-panel bootstrap CSVs/report. Below1GiB disk, load>20,
YIELD/STOP, a failed process, inconsistent census or cap refusal prevents dispatch.
No retry or truncation is automatic. CPU tests now also verify exact frozen CTRL0
masking and PI/CIRCLES subset/native release. Production local crossing-cohort
scoring is aligned to the SNR oracle; no new controller/physics action path was
introduced after the canary. Pending report is clearly labelled until completion.

## Binding artifact HOLD and recovery (2026-10-31, owner 17:20)
- Local continuation39620 was identity-verified and stopped. The owned gpuR
  YIELD flag and ARTIFACT_HOLD.json block dispatch; owned worker516243/start
  ticks4206436595 was identity-verified and SIGSTOPped while idle between worlds.
  No in-progress world was interrupted; no peer process was signalled.
- Completed work is preserved: canary,28 census prefixes, and main969650's
  original five-arm branches. At pause:21180actual/21480charged sim-s of500000;
  unused setup reservation300 remains charged. Receipt:
  tools/coopaudit/evidence/ARTIFACT_HOLD.json. No jobs are duplicated or retried.
- The prior grid6/two-member/60s binding used the now-disclosed CV selection
  defect. S-SUMOCF's published raw-owner-rule winner is grid26/all-zone/60s/rho1,
  itself under artifact investigation. Neither selection currently authorises
  further audit SUMO. The completed first world is historical, not a completed
  corrected-design observation; its results and charges remain untouched.
- **Release condition:** E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md must explicitly
  report PASS, or a corrected/re-selected design with a passed artifact check.
  Bind that evidence and the current owner-rule selection, revise exact complete
  panel costs, and register/commit before dispatch. Missing/ambiguous/FAIL blocks.
  No unconditional timer release or old-worker resume is authorised. As checked
  at initial recovery the root-level file did not exist. Located subsequently in
  docs/paper_results/ARTIFACT_CHECK.md: explicit HOLD, no resume authorised. Its
  pending gates include zero-noise bitwise identity and a corrected<=100k probe.
- If the selected group changes, regenerate only deterministic groups from the
  existing census's observable positions; keep the original census/state hashes.
  Never rerun finished prefixes to discover new groups or overwrite old jobs.
  A new design namespace is required for non-reusable branches, all newly charged.
  Reserve space for the paired trained-policy arm before admitting the panel;
  absent a checkpoint, report its pending status rather than fabricate a comparator.

## Fail-closed launch implementation (17:20 amendment)
`release.py` checks the local and remote launch paths independently: an active
ARTIFACT_HOLD, missing release, YIELD/STOP, non-PASS evidence, scope mismatch,
missing three-arm budget reservation, changed artifact/design hashes, or absent
registration provenance prevents launch. The artifact document must contain an
explicit PASS status line or artifact-check title, without a conflicting HOLD/FAIL;
release is reviewed and registered, never inferred
from a timer or table title. No release receipt has been issued during this HOLD.
The old table-polling auto-bind/auto-dispatch code is removed. Selection validation
now recomputes abs(raw DG mean)/raw DG SD and rejects a substituted CV statistic.
Primary controller loops use MAPPO/wall only. Budget admission reserves the third
arm's G+solos plus a second300s prefix/world for a checkpoint delivered later.
Historic main job names/census groups cannot be reused silently by the new runner;
an explicit versioned-panel migration after PASS is still required. The paused
  old worker must never be SIGCONTed into the outdated five-arm/two-member loop.

## Corrected full-prefix design v2 (2026-11-01, before new SUMO)
- Upstream `658b09a3` artifact PASS is now published: document SHA
  `4ac2b3e82b61bbdb9e2fbbf788af8311a0e8f4d62b07c88b48868ca31410d6af`,
  PASS receipt `372d33266ef6ebbd7431ead73fd7e392a338a33779323539f82c4bad34ed75ab`.
  Saved-state/loadState is INVALIDATED, including historical audit estimates;
  retain all old data/charges, but never pool those observations with v2.
- Adopt corrected grid7: nearest RSU-observable **pair, 60s**, nominal probe rho.5.
  Table SHA420a4904780844c3a49b8eb4ed8855c0ebf8f0b235d39aa69c2116c4260acfbf.
  Selection was low-power/roundoff-driven, not evidence of raised SNR. No noise
  is used in this deployed-controller audit. This supersedes the larger-group
  expectation and every saved-state/prefix-cost provision above.
- Every B, repeated B, G and ONLY branch starts a NEW SUMO process, replays the
  exact same native300s prefix, verifies tensor-state/request/observable hashes,
  then continues600s. Never save/load state. Same native no-command counterfactual,
  60s intervention, handback to IDM and native lane decisions, original outcomes.
- Primary arms run together: MAPPO, wall-v2.1 and final completed SUMOCF **u15**,
  SHAe4b51f64fba54249792237131c2a10a6cbc5d1286c386c2c8dc9374b2d398d8a.
  Freeze latest completed update before audit outcomes, not best DEV checkpoint.
  Use original Torch decoder/safety/memory plus exact upstream RSU adapter,
  hash-bind its source; no probe offsets/exploration. Mask membership identically.
- Reuse28 completed observable censuses ONLY for frozen selection/identity.
  Exactly21 eligible groups (20/24 high-share),7missing; only3 feasible pair walls.
  Retain all28 worlds/coverage, missing !=zero, no replacements or extra census.
  All-world stratified CI can be undefined with a missing stratum; report that
  explicitly alongside available-case mean/coverage and prespecified cell results.
  Secondary conditional eligible-world CIs resample eligible worlds within their
  represented cells, weighted by eligible-cell counts; report this estimand and
  its21-world (20high-share) denominator separately from the undefined full panel.
- Full panel207900sim-s + corrected mechanics canary14400 (16 full900s reruns)
  + retained21480 = **243780 charged maximum**, below500000. Each full rerun
  precharges900 before its prefix. Canary uses the existing969699 geometric wall
  pair, B/Brepeat, no-op through empty enabled set, three G+solos and G repeats;
  verifies bitwise native/control identity, zero-effect equivalence and wall wiring.
  No failed/completed branch retry, no refund, no truncation.
- Store new data at `/root/autodl-tmp/coopaudit20261101_v2`, linked into the
  original own jobs root as `v2_sSEED`/`v2_canary_s969699`; original ledger remains
  authoritative. Root filesystem has <1GiB; new storage has2.2GB at inventory.
  Gate the actual new storage at>=1GiB before EVERY full rerun; preserve all data.
- Up to6 spare nice19 single-thread/CUDAhidden world workers; per-branch resource
  checks yield on load>20, RAM<8GiB, disk<1GiB, or own YIELD/STOP. SUMOCF priority
  remains binding (its trainer has stopped); no cpuQ/peer process intervention.
  Retire only exact owned stopped516243/start4206436595 with SIGKILL (never resume
  stale code), preserve PID/HOLD history, acquire singleton lock before new work.
- `evidence/CORRECTED_DESIGN.json` binds each census/group/cost and upstream hashes.
  Release only after registration/design commit and CPU tests; corrected canary
  must complete and be committed before the full panel is dispatched.

## Corrected canary VERIFIED (2026-11-01)
Full-prefix16/16 completed:14400actual=charged; cumulative35580actual/35880charged.
All16 native prefix trajectories bitwise identical; B/Brepeat, MAPPO-disabled,
SUMOCF-disabled and all three GROUP repeats match exactly as required. Every
branch has zero collisions/teleports/dropped steps. Wall106commands, assembly307s,
native handback353s. Corrected wall J600 D_G+1616.622358, sumD_i-41.617400,
**C_G+1658.239758veh-s** (solos-18.920193/-22.697207). MAPPO and SUMOCFu15 each
C_G600=0, D_G.000823. This is one geometry-selected mechanics control, separate
from panel inference. No old saved-state C_G is reused. Receipt
`evidence/V2_CANARY_VERIFIED.json` binds every scientific source and raw archive.
