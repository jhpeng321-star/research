# S-RB-INT handoff

Updated 2026-11-01 ~00:49 +08; branch `c4/rb-int`. Owner registration5683/5697/5708/5733/5736.
Full earlier history retained in `docs/paper_results/WMPHYS_INT_HANDOFF_ARCHIVE_0034.md`.

## READY FOR EVAL
- WM-PHYS-A: RUNA claimed; constructor FAILED before any rollout; see Jobs.
- **WM-PHYS-B / RUNB gpuR**: `E:/agent-work/c4-rb-int/docs/paper_results/WM-PHYS-B.runb.json`.
- WM-PHYS-C: NOT READY (same controller-interface prerequisite as A); original
  manifest retained at `docs/paper_results/WM-PHYS-C.runa.json`.
- Exact candidate source commit **b0f66943ec300594fc220d3779fb482c26534b1c**.
- Immutable runtime on BOTH hosts: `/root/rbeval20261031/phys_code_b0f66943ec30`.
- Config: that runtime's `configs/wm_phys.toml`; entrypoint B:
  `tools/rbint/evaluate_B.py`. Uses INT's callback-capable runner, SAME published CLI.
- Smoke on both hosts: `/root/rbeval20261031/SMOKE_b0f66943ec30.json`, SHA256
  `74749fba6e10ca55bcecc6202483157f235c501e62d838491145b19c61e4ca66`.
- B population defaults apply to BOTH existing and incoming drivers; Poisson
  future requests from observable demand/backlog. Identity-bound recipe and sampler.
- RUNA compatibility independently verified on gpuA: loader_module is
  `tools.rbint.evaluate`, which selects the immutable staged complete runner
  BEFORE RUNA's `from run_wm import main`. Resolved path is the staged runtime's
  tools/rbeval/run_wm.py; signature explicitly includes future_builder. Thus
  A/C use the bound compatible runner even with an older EVAL worktree copy.

## What exists
- Ported ONLY existing frozen MAPPO `policy_native_lanes` ownership option
  from c4/speedhead bridge (default OFF). Needed by the unchanged registered
  callback; policy acceleration preserved, WM native lanes delegated.
- FULL recipe/entrypoint now wired (NOT READY until staged smoke): fitted
  shrinkage_v1 joint draws + learned entrance-counter/Gamma-rate/Poisson clocks.
  Reads ONLY hash-bound existing start/history exports; numeric vid_index follows
  recorded identity through packed/recycled slots. Unknown posterior IDs use the
  fitted population; reserved-world conditional-motion refit not yet available.
- FIT50f2cd02 integrated: shrinkage_v1 preferred physical posterior (learned
  strength19.35568, already applied in artifact), UNC bridge and SUMOCORE
  sign diagnosis. Fitted LCv2 remains OFF: still loses its event-rate baseline.
- UNCe038da91 integrated: fitted finite-support posterior adapter, identity-keyed
  existing-driver draws, learned observable-counter arrival family + Gamma rate
  uncertainty. Default-OFF; FULL recipe wiring next. Optional fitted LC left OFF.
- MERGE75a24397 integrated (kernel0090c168): Euler brakeGap with tau, overtake
  time+space checks, unblocked-leader cap, persistent lookahead; ONLY two adapter
  keyword hooks, retaining INT's entrant identity and mainline ownership fixes.
- LANE811cd7e8 merged: finite continuation geometry, stale keep-right gate,
  SUMO numerical epsilon; prior immutable A/B/C runtime remains source-bound.
- CF742922b6, LANE (including right-overtake prevention), MERGE follow-ups,
  UNC1f918969 and FIT11d36f75 integrated. EVAL scorer/finalizer merged.
- A: existing motion encoder EXACT, hidden-driver joint draws + Poisson arrivals.
- B: inherited physics population point, Poisson arrivals; not encoder centre.
- C: motion-only pooled_full_v5 point fit (JSON SHA bound), Poisson arrivals.
  CAV plant/settings retained; adverse fitted LC is NOT adopted.
- All variants: cf_model=sumo_idm, lane_model=lc2013, merge_model=sumo;
  no learned acceleration/lane/merge/courtesy/admission/noise corrections or training.
- Source fixes: EPS=.001, POSITION_EPS=.1, passenger emergencyDecel9;
  HDV Euler positions; no missing-leader 250m phantom/no duplicate ramp stop.
- Entrant admission and newborn motion use EXACT same weighted ordinal-keyed draw.
- `tools/rbint/{recipe,evaluate_A,evaluate_B,evaluate_C,smoke,stage}.py`.
  Full-population ledger/exit/finite audit attached to every candidate task.

## Verification / receipts
- Controller ownership + recipe **18 PASS**; a broader TimelineControllers
  test needs local onnxruntime (absent). Real five-arm staged smoke still required.
- FULL recipe integration **26 CPU tests PASS /24.61s**. Checks packed-vs-recorded
  IDs, prior/posterior distinction, untouched CAV settings, action-paired replay,
  learned source conditioning, changed-start/permutation rejection. Smoke next.
- Existing cpuQ FIT source suite **17 PASS /2.10s**; latest21-test source gate
  still to run in NEW INT staging (local SciPy absent). UNC adapter20PASS above.
- UNC integration: **34 CPU tests PASS /13.33s**; fitted-IDM/arrivals, legacy
  futures and INT recipe/audit. Test fixture now explicitly eval(), matching
  production loader and the new fail-closed deterministic-future guard.
- MERGE integration: **51 CPU tests PASS /16.99s**, including new full-fleet
  diagnostics and combined INT ramp/mainline/correction/source gates.
- Resumed LANE merge gate: **116 CPU tests PASS /33.30s** (all touched
  LC/CF/merge/insertion/hidden-future/INT modules). Combined fixture repaired:
  prior mainline position1000m was beyond553.33m road; now upstream -1000m,
  with60s window for source finite-length keep-right accumulation. No physics
  threshold changed. Broader equivalence attempt timed out; not counted as PASS.
- Full touched-module CPU suite **159 PASS /33.34s**; independent source anchors,
  CF, LC2013, insertion, hidden futures, native lanes, INT correction/encoder guards.
- Additional recipe reconstruction suite **5 PASS /8.29s** (inherited mode resets).
- Default-OFF legacy and WM2 synthetic weights/forward/rollout bit-exact.
- **M1 engineering smoke PASSED**: 6x60s on recorded native FIT969420/969421,
  A/B/C, exact pre-t0 history, no future masks/teacher/arrivals; finite, conserved,
  zero learned residual; temporary hooks reject every parameterized correction.
- `docs/paper_results/WMPHYS_SMOKE_b0f66943.json`: 94 and71 initial cars;
  A exits28/38, B43/40, C32/39; exact final+exit conservation. CPU wall11.68s.
- Prior three immutable stages preserved; two constructor incompatibilities and
  one missing frozen controller asset fixed before any completed rollout/task.

## Jobs / coordination
- A claimed/attempted once by RUNA at01:08; all3 workers failed before rollout:
  frozen MAPPO callback uses policy_native_lanes unsupported by base bridge.
  Original remote FAILED.json/logs preserved, no completed task to repeat.
  New recovery source/variant names only after five-arm interface verification.
- RUNA intake-path repair: its discover() ONLY matches rb-shared/evalclaims
  paths, not INT docs paths. Published semantically identical A/C launch JSONs
  there; existing watcher44796 will claim them. No duplicate watcher/dispatch.
- Fresh-session audit: clean HEAD61e4e036 retained; all prior receipts/stages preserved.
- RUNB already owns B, supervisor gpuR728893 WAITING_RESOURCES; do not redispatch.
- RUNA watcher44796 owns intake of published A/C; no A claim at00:42 poll.
- New ready source increments: LANE811cd7e8, MERGE0090c168, UNC46c33fbe.
- Owner00:44 schedule supersedes05:00 freeze: interim04:00/08:00, freeze FULL
  around09:30, full evaluation09:30–11:30, final report with credibility11:45.
- DECIDE handoff not present at00:42; FIT finite-profile posterior is available,
  learned arrival upgrade pending UNC. Existing C remains pooled-point fit.
- No INT scientific dispatch yet. RUNB owns B after exclusive shared claim.
- EVAL WM2 gpuR workers706825..706828 and finalizer708801 already live; preserve.
- SUMOCORE144/144 complete; all supervisors exited; compact results available.
  Physical validity FAIL (18/144 initial-movement collision flags retained).
- FIT all jobs completed; LANE/MERGE diagnostic outputs integrated, no redispatch.
- Shared helper artifacts consumed; raw truth/assignment/prohibited paths untouched.
- Poll workers about every20min. No peer process action, push, deletion or training.

## Remaining / known gaps
- RUNA must retain the published loader module so the verified runner resolves.
- Matched distributional efficacy, HDV60s non-regression and paired CIs pending.
- LC2013 is highway subset (synchronous messages; incomplete full negotiation).
- Historical lane reproduction and ramp delay are different panels, not efficacy
  evidence for the integrated source-fixed candidate. Sparse8-world cells cannot
  certify full distributional qualification.
- Morning report must explicitly show CRPS120/300/600, breakdown-distance and
  HDV60s differences WITH paired-world CIs under `Switching-eligibility rule`.
