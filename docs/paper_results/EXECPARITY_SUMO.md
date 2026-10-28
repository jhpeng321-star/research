# S-EXECPARITY real-SUMO verification

## Status and accounting (00:20 law; updated 2026-10-29 01:26 local)

**Superseded status preserved below.** The 00:20 MAIN DECISION
(REGISTRATION.md) replaced the continuous law with the EXACT ALWAYS-COMMANDED
law (`C4_EXECUTION_PARITY=2`, contract `always_commanded_ctrl0/1`): every CAV
commanded every tick through the single CommandOwnership writer (CTRL0 path,
residual kept), executed accel = speedMode-27 safe-speed cap of
(base + residual), accel 1.2; imagination mirrors the identical envelope. This
needs no same-step TraCI hook, so the former blocker is void.

Registered adapted check (same 7,200 sim-s cap, 4 scientific + ≤4 charged infra
retries): (i) `s977000_always0` byte-identical to stored S-CTRLARM CTRL0
`jobs/s977000_ctrl0` (metrics JSON + traj_base.npz + episode.npz +
cumulative JSONs); (ii) C1 `alwaysx` on 977001 (cpuQ), 977002 (cpuP), 977003
(cpuQ) with per-tick ledgers (residual, base law, issued, speed-before,
leader gap/speed); (iii) offline replay via
`tools/execparity/alwayscmd_replay.py` — PASS = pooled imagination executed-accel
RMSE ≤ pooled WM native-head RMSE on the same rows AND canary bit identity;
the gate receipt (schema `execparity_always_commanded_acceleration/1`) is
written only on PASS.

Dispatch state: canary monitor local PID 29400 (`tools/execparity/sumo_operate.py
canary`), remote worker cpuP PID 3410593 nice 19 single-thread, staged source
worker sha `ccbad268…`, adapter sha `19f4f869…`, frozen runtime manifest
`b3c031ad` verified (543 files), inputs SHA-bound. Worker waits for host load
≤ 8 before each allocation; the charge (`S-EXECPARITY ALLOC`) is appended to
the authoritative REGISTRATION.md and ACKed before any SUMO step. Three local
restarts occurred BEFORE any allocation (session freeze + staging assert);
nothing was charged. Allocation then production (both hosts) proceed
automatically under the same guard; production requires the canary's
`bit_identical` from its local COMPLETE event. Offline replay and this report
are updated after the runs. cpuP/cpuQ nice 19, never touching peer jobs.

## Superseded: blocked status of the continuous-law design (23:40–00:20)

Owner approval: authoritative `runs/active/C4-V7FT-20261017/REGISTRATION.md`,
2026-10-28 23:50. Separate maximum **7,200 charged sim-s**, eight900s allocations:
four scientific jobs and at most four charged infrastructure retries. This task
has made **0 allocations, 0 charged sim-s, 0 actual sim-s**. No remote coordinator,
SUMO worker or automatic retry has been launched. The whole cap remains unused.

The required combined execution is not available in the frozen stock-SUMO
backend. CPU fixtures passing does not establish an executable simulator path.
`getSpeedWithoutTraCI()` reads the completed move's cached native speed; it cannot
supply next-step native LC-yield speed before stepping. Applying a tiny residual
to that cached speed retains a finite zero-to-epsilon discontinuity. Source
inspection and installed-TraCI introspection are receipted in
`tools/execparity/{backend_timing,backend_preflight}.json`. A genuine callback
after native CF/LC speed calculation and before external-speed/safety composition
is required. This is a **functional blocker**, not an infrastructure retry.

27 CPU execution tests PASS, including zero/native bit identity in a simulator-free
same-step deployment-writer fixture, positive authority below the ceiling, zero
capped gradients, tiny-residual continuity and rejection of the stock cached API.
No true combined-law realised acceleration or trajectory comparison is reported.

## Frozen approved design for resumption

Preserve the exact E3-final world metadata and frozen source manifest `b3c031ad`,
C1 checkpoint `64f70af4`, deterministic policy settings, textbook population,
300s native warmup and600s intervention. Default-OFF combined flag and its
same-step backend are the only execution treatment. Keep original host shards
(cpuP/cpuQ), nice19 and one numerical thread; at most one task job per host.
Never inspect/signal peer jobs. No unapproved scientific arm may consume retries.

1. **977000, residual exactly zero, continuous path:**900sim-s. Compare the
   complete episode/physics NPZ bytes, initial-state hash, metrics and cumulative
   120/300/600s payloads with the stored native run. Every zero residual must be
   native with no longitudinal override. An existing native canary is not this
   new executor's zero-residual canary.
2. **977001, C1 continuous:**900sim-s.
3. **977002, C1 continuous:**900sim-s.
4. **977003, C1 continuous:**900sim-s.

Jobs2–4 log every0.5s physics step: vehicle identity, timestamp, requested
residual, speed before stepping, native/no-TraCI speed generated **during that
same step** (read afterwards), issued pre-safety acceleration, realised
speed-difference acceleration, accepted/native ownership and backend contract.
Preserve full observable episode states, source/input hashes and process/start
receipts. Never pass the offline next-speed target into the actor.

Offline, replay those exact states/residuals through WM-native-base + residual,
frozen learned IDM cap and matching1.2/emergency-deceleration envelope. Report
per-world and pooled bias, MAE, RMSE, p05/50/95 and extrema, cap/inert fractions,
and the same states' native-head error relative to the completed-step no-TraCI
base. Primary error is imagined executed acceleration minus **actual realised
SUMO acceleration**, not a shared-cap synthetic target. Pass requires byte
identity in job1 AND execution RMSE no larger than the native-head error band.
Report cap/head projection errors separately; do not select a friendlier
denominator, retune the driver artifact or replace RMSE with world-mean RMSE.

Before each individual start, append its unique allocation ID, host, world, arm,
900s charge, cumulative charge, source/flag/input hashes and approval reference
to the authoritative REGISTRATION.md. Persist a start receipt before simulation.
Four jobs cost3,600sim-s; four infrastructure-only retries reserve another3,600.
No retry is charged, refunded or started now. SSH polls at least60s apart,
refusal backoff at least300s. Same-step backend implementation/requalification
is needed before resuming; the current code deliberately cannot dispatch it.

## Offline evidence available now

Ten old native tapes,6,998 sampled stable-CAV rows: shared-inferred-cap
continuous proxy RMSE0.170964m/s² versus raw native-head0.198964. World-mean
proxy0.18316[0.15948,0.20628] versus native head0.20896[0.18524,0.23094]. This is
**CPU sensitivity evidence**, not measured real-SUMO parity: both synthetic
branches share cap error, and the recorded-native feasibility projection itself
has RMSE0.15099. `continuous_results.json` explicitly has
`measured_real_sumo=false` and `deployment_passed=false`; the deploy gate rejects it.

The older switch-path recorded-acceleration acceptance also FAILS its strict
pooled criterion: MAPPO0.200415 exceeds native-head0.198964m/s². Historical C1
cap-only RMSE0.170708 does not waive this failure. All old outputs are retained.

## Native/MAPPO runtime canary

The ported residual-aware BranchController/split_output, observer and explicit
`--population textbook` are CPU-qualified against frozen E3-final:600MAPPO
sampled ticks/6,664CAV rows, commands/lanes bit-identical; native sends no command.
This does not certify closed-loop bytes. Once executable, the approved job1
tests native identity. An additional full MAPPO977000 runtime canary would cost
900sim-s and require separate scientific allocation approval; it is **not** an
infrastructure retry in the fixed8-job design. Compare both NPZ byte hashes and
all frozen metrics before any broader evaluation. No such MAPPO run was launched.
