# S-RB-INT handoff

Updated 2026-11-01 ~00:43 +08; branch `c4/rb-int`. Owner registration5683/5697/5708/5733.
Full earlier history retained in `docs/paper_results/WMPHYS_INT_HANDOFF_ARCHIVE_0034.md`.

## READY FOR EVAL
- **WM-PHYS-A / RUNA gpuA**: `E:/agent-work/c4-rb-int/docs/paper_results/WM-PHYS-A.runa.json`.
- **WM-PHYS-B / RUNB gpuR**: `E:/agent-work/c4-rb-int/docs/paper_results/WM-PHYS-B.runb.json`.
- **WM-PHYS-C / RUNA gpuA next**: `E:/agent-work/c4-rb-int/docs/paper_results/WM-PHYS-C.runa.json`.
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
- Fresh-session audit: clean HEAD61e4e036 retained; all prior receipts/stages preserved.
- RUNB already owns B, supervisor gpuR728893 WAITING_RESOURCES; do not redispatch.
- RUNA watcher44796 owns intake of published A/C; no A claim at00:42 poll.
- New ready source increments: LANE811cd7e8, MERGE0090c168, UNC46c33fbe.
- Owner00:40 schedule: integrate continuously, freeze FULL around05:00, retain
  immutable A/B/C, evaluate through07:15 and report with DECIDE credibility07:30.
- DECIDE handoff not present at00:42; FIT finite-profile posterior is available,
  learned arrival upgrade pending UNC. Existing C remains pooled-point fit.
- No INT scientific dispatch yet. RUNB owns B after exclusive shared claim.
- EVAL WM2 gpuR workers706825..706828 and finalizer708801 already live; preserve.
- SUMOCORE production cpuQ203500 already launched; do NOT duplicate.
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
