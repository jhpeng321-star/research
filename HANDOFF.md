# S-RB-INT handoff

Updated 2026-11-01 01:33 +08; branch `c4/rb-int`. Owner registration5683/5697/5708/5733/5736.
Full earlier history retained in `docs/paper_results/WMPHYS_INT_HANDOFF_ARCHIVE_0034.md`.

## READY FOR EVAL
- **WM-PHYS-A2 / RUNA gpuA**: `E:/agent-work/rb-shared/evalclaims/WM-PHYS-A2.runa.json`.
- **WM-PHYS-B2 / RUNB gpuR**: `E:/agent-work/rb-shared/evalclaims/WM-PHYS-B2.runb.json`.
- **WM-PHYS-C2 / RUNA gpuA next**: `E:/agent-work/rb-shared/evalclaims/WM-PHYS-C2.runa.json`.
- NEW ready source **4015b6412cdfa2a94168b7095a59287d0fddcd3b**, staged BOTH hosts
  `/root/rbeval20261031/phys_code_4015b6412cdf`; recipe configs/wm_phys.toml.
  Smoke8x60s PASS, SHA8cfb30c91e6d81c9cb4ef2e1d04b7eb6621c36cfa381388533874828037a90eb.
  Real five-arm600s canary969432/k0 PASS206.03s, FULL settings; controller ABI verified.
- **FULL-v1 INT-owned already running gpuA**: shared exclusive claim; do NOT
  redispatch. Own333463/333464/333465, root `/root/rbeval20261031/int/WM-PHYS-FULL-v1`.
  Monitor335983 (source9f35ba4f) live: exact owned incarnations/18.5GiB guard,
  owner11:30 deadline,04/08 complete-K8-only interim + full frozen scorer once.
  `panel_ops status` is read-only; never repeat canary/continue/monitor dispatch.

## What exists
- MERGEda3607eb integrated: source22.22→31.29 ramp limits/speedFactor, EPS
  constants, physics-native opt-in callback and matched native-relative accel.
  Old learned native head's missing-leader dv gathers unrelated slot0 speed:
  can falsely stall an unled ramp CAV. New callback bypasses that head, no training.
  **59 CPU tests PASS /1CUDA skip**, including INT query parity and old adapters.
  Callback not yet selected by the live4015b641 panel or new recipe.
- NEW default-OFF `PhysicsConfig.sumo_merge_query_fast`: exact one-ego geometry
  per reservation rank (junction_query.py). CPU dense/query geometry+reservation
  bit parity PASS;34tests PASS/1CUDA skip. Synthetic640slot kernel benchmark42.3x
  CPU speedup; recorded FULL60s FIT969420/969421 dense-vs-fast COMPLETE tapes
  bit-exact (positions/speeds/lanes/masks/accel/residual). Small closed-cohort
  rollout speedup only1.00–1.03x; padded/GPU speedup unmeasured. Source5a83518e,
  receipt docs/rbint/QUERY_PARITY_5a83518e.json. NOT in live4015b641 runtime.
- Ported ONLY existing frozen MAPPO `policy_native_lanes` ownership option
  from c4/speedhead bridge (default OFF). Needed by the unchanged registered
  callback; policy acceleration preserved, WM native lanes delegated.
- FULL recipe/entrypoint wired and live: fitted
  shrinkage_v1 joint draws + learned entrance-counter/Gamma-rate/Poisson clocks.
  Reads ONLY hash-bound existing start/history exports; numeric vid_index follows
  recorded identity through packed/recycled slots. Unknown posterior IDs use the
  fitted population; reserved-world conditional-motion refit not yet available.
- FIT50f2cd02 + UNCe038da91 integrated: shrinkage strength19.35568 already
  applied, no double shrinkage. Fitted LCv2 stays OFF (loses event-rate baseline).
- MERGE75a24397 integrated (kernel0090c168): Euler brakeGap with tau, overtake
  time+space checks, unblocked-leader cap, persistent lookahead; ONLY two adapter
  keyword hooks, retaining INT's entrant identity and mainline ownership fixes.
- LANE811cd7e8 merged: finite continuation geometry, stale keep-right gate,
  SUMO numerical epsilon; prior immutable A/B/C runtime remains source-bound.
- CF742922b6 + EVAL scorer/finalizer integrated; earlier branches retained in Git.
- DECIDE16dff9ad merged: tools/wmdecide strict paired0.8 win rule, calibration/
  reliability/coverage reports. **15 CPU tests PASS /4.38s** in INT worktree.
- A: existing motion encoder EXACT, hidden-driver joint draws + Poisson arrivals.
- B: inherited physics population point, Poisson arrivals; not encoder centre.
- C: motion-only pooled_full_v5 point fit (JSON SHA bound), Poisson arrivals.
  CAV plant/settings retained; adverse fitted LC is NOT adopted.
- All variants: cf_model=sumo_idm, lane_model=lc2013, merge_model=sumo;
  no learned acceleration/lane/merge/courtesy/admission/noise corrections or training.
- Source fixes: EPS=.001, POSITION_EPS=.1, passenger emergencyDecel9;
  HDV Euler positions; no missing-leader 250m phantom/no duplicate ramp stop.
- Entrant admission and newborn motion use EXACT same weighted ordinal-keyed draw.
- `tools/rbint/{recipe,evaluate_A,evaluate_B,evaluate_C,evaluate_FULL,panel_ops}.py`.
  Full-population ledger/exit/finite audit attached to every candidate task.

## Verification / receipts
- Latest complete touched-module CPU suite **155 PASS /18.70s**. Exact staged
  FIT test file on cpuQ **17 PASS /1.54s** (SciPy from existing environment).
- New closed-cohort A/B/C/FULL smoke **8x60s PASS**, gpuA CPU source0ec484ab;
  `docs/paper_results/WMPHYS_SMOKE_0ec484ab.json`. All finite/conserved/no corrections.
  FULL driver identity and learned-arrival conditioning constructors also pass.
- Real all5-controller600s canary now PASS; canary task reused, not rerun.
- FULL26PASS/24.61s: packed-vs-recorded IDs, prior/posterior/CAV preservation,
  paired replay, source conditioning, changed-start/permutation rejection.
- Default-OFF legacy and WM2 synthetic weights/forward/rollout bit-exact.

## Jobs / coordination
- Old A/C claimed/attempted once by RUNA; workers failed before rollout:
  frozen MAPPO callback uses policy_native_lanes unsupported by base bridge.
  Original remote FAILED.json/logs preserved, no completed task to repeat.
  All failed sources/logs/claims retained. A2/B2/C2 use NEW immutable source.
- RUNA watcher44796 and RUNB watcher33076 retained. Old B gpuR728893 waiting
  for resources at last audit; prerequisite note requests owner-side cooperative
  stop before any old-source inference. INT never signals peers.
- Owner00:44 schedule supersedes05:00 freeze: interim04:00/08:00, freeze FULL
  around09:30, full evaluation09:30–11:30, final report with credibility11:45.
- DECIDE credibility scorer ready; run on complete K8 outputs, not single futures.
- EVAL WM2 gpuR workers706825..706828 and finalizer708801 already live; preserve.
- SUMOCORE144/144 complete; all supervisors exited; compact results available.
  Physical validity FAIL (18/144 initial-movement collision flags retained).
- FIT jobs complete; LANE/MERGE diagnostics retain source-specific clocks.
- Shared helper artifacts consumed; raw truth/assignment/prohibited paths untouched.
- Poll workers about every20min. No peer process action, push, deletion or training.

## Remaining / known gaps
- RUNA must retain the published loader module so the verified runner resolves.
- Matched distributional efficacy, HDV60s non-regression and paired CIs pending.
- EVAL endpoint gate request: rb-shared/INT_HDV60_GATE_REQUEST.md; bin/clock-only
  completed tasks cannot supply60s HDV positions. Do not infer non-regression.
- LC2013 is highway subset (synchronous messages; incomplete full negotiation).
- Historical lane reproduction and ramp delay are different panels, not efficacy
  evidence for the integrated source-fixed candidate. Sparse8-world cells cannot
  certify full distributional qualification.
- Morning report must explicitly show CRPS120/300/600, breakdown-distance and
  HDV60s differences WITH paired-world CIs under `Switching-eligibility rule`.
