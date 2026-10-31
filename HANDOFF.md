# S-RB-LANE handoff — integration-ready highway subset

**00:10 precision continuation active:** prior commits/results retained. New
per-predicted-motivation diagnostics added; source-backed stale keep-right
request fix implemented (now gated by anticipated lane speeds). Candidate4
verified: precision2.577% vs2.562%, recall34.900% unchanged (255FP removed).
Source through `568d0314`; safe small fix, insufficient overall. Route LENGTH vs remaining-distance audit continues;
see `docs/LC2013_PRECISION_ITERATIONS.md`. Shared helper lcspec now consumed.
Panel3 attribution verified: keep-right 203/27924 (0.727% precision),
64.17% of all false-positive steps. Gate-fix CPU tests23/23 PASS.

Branch `c4/rb-lane`, base `f0db3c82`; source implementation through `127a106f`.
Owner WM-PHYS authorization REGISTRATION.md:5683 (2026-10-31 23:12).

## Exists / switch
- `src/wms/physics/lc2013.py`: SUMO v1_27_1 highway/sub-lane-OFF subset,
  strategic ramp/route, cooperative clearing, speed gain, keep-right, persistent
  fixed-point wishes/lookahead, exact IDM secure gaps, follower-help speed
  messages, right-overtaking prohibition. No learned weights or fitted cutoffs.
- `ModelSpec(lane_model="lc2013", lane_head="off")`, default unchanged.
  `merge_decision="hazard"` (no learned merge module). New adapter; small
  guarded spec/wms edits. WM lanes left0/right1/ramp2, target never ramp.
- Only matching motion-inferred settings map. MOBIL politeness/threshold are
  NOT comparable LC settings: LC style uses SUMO defaults. IDM secure gap:
  `max(0,v*tau+v*(v-vlead)/(2*sqrt(a*b)))`, not existing braking-distance rule.

## Integrator contract
- Build LC spec; do NOT just mutate `.lane_model` on an old first-passage/
  hazard_v5 model retaining conflicting modules. Disable other learned accel/
  merge corrections in WM-PHYS. Factory unchanged; no new checkpoint weights.
- Forward diagnostics use current snapshot. Step recomputes AFTER longitudinal
  motion/CAV execution; fixed CAV lanes preserved, native/delegated masks allowed.
  Geometric conflicts arbitrated front-to-back; unmerged ramp held at lane end.
- `rollout` carries `lc2013_*` across steps/chunks, resets newborns. Direct step
  callers carry `physics.lc2013.STATE_KEYS`. Numeric IDs: `vehicle_id` or `vid`.
  Reset on absence/identity/observed lane change. Messages consumed next step.
- Optional route distances/offset/best-lane queued occupation inputs supported;
  defaults: scenario ramp end plus continuing mainline. No sublane.

## Verified result
- **62 CPU tests PASS**, JUnit `tools/rblane/receipts/CPU_TESTS.xml`.
- FIT969400–969431 only,160episodes/5arms;60s warmup+60s teacher forcing.
  Final exact LC1136/3255=**34.90%** vs paired MOBIL244/3255=**7.50%**;
  mainline32.36% vs0.37%. Historical1.4% has a different panel/contract.
- LC positive steps44333, exact precision2.56%. Compressed runs5159,
  ±2s recall26.45%/precision16.69%, timingMAE0.472s/positionMAE8.242m.
  Improved recall is NOT full parity/distributional qualification.
- `receipts/panel2_results/`, source/aggregate/input/MOBIL-reuse checks PASS
  (`VERIFIED.json`). First candidate+smoke preserved; baseline not rerun.
- Completed GLM targets cross-check:3220 events agree;35 expected300.5s tape
  boundary omissions reconciled, original mismatch retained. lcspec absent.
- All owned cpuQ workers180907/187712/214459 exited. No outstanding jobs.
  Six allowed observable files only; no hidden data/SUMO/GPU/training/signals.

## Known gaps / references
Synchronous messages/independent proposal gaps vs SUMO serial lane changer;
full leader overtake/fall-behind/counterchange reservation and all follower
branches not ported; nonlegacy keep-right follower adjustment, nonzero speed
lookahead and nonhighway/emergency/stochastic overtake-right cases omitted.
Public source read only from NEW empty temp directories, never executed.
Details: `docs/LC2013_SOURCE_NOTES.md`, `docs/paper_results/RB_LANE.md`.
Distributional gate belongs to integrated CF+merge+LC via S-RB-EVAL.
