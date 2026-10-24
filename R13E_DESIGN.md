# S-R13E: emergency safe speed and secure insertion

## Status and provenance

Engineering implementation complete; **offline measurement is complete; deployment still requires the reviewed fixes**.
Merged `c4/v10r13c` tip `eb75010c6a2a7ef66eebc3282e03f50e37d4b202` into
`c4/v10r13e`. The subsequent owner-authorized CPU study on gpuR is recorded in
**R13E_OFFLINE.md** with committed JSON evidence and the appended external preregistration.
No training or GPU operation was launched. The pinned gs is NOT GAPACC-equivalent
(120/256 same-event decisions differ); BASE-1 rejects all three arms on the strict
train-role panel. The mechanism descriptions below document the pinned implementation.

## Mechanisms and constants

Both `physics.emergency_braking` and `physics.secure_gap_veto` default false,
are omitted from default legacy-flat configs, and introduce no network tensors.
Strict checkpoint loading preserves every tensor byte.

Emergency braking caps the IDM baseline with a differentiable ballistic stopping
envelope, permitting deceleration beyond learned comfortable braking. Its emergency
limit is **B = -physics.accel_min_mps2 = 9 m/s²**, the existing plant clamp; its
interval is the existing **dt = 0.5 s**. The standstill clearance is the learned
driver min-gap. The safe next speed solves

`dt*(v+u)/2 + u²/(2B) + dt*u/2 <= gap-s0 + vl²/(2B)`.

The half-tick term bounds the unchanged speed-clipped trapezoidal plant's stopping
sum; a second constraint handles a leader stopping within the tick. Smooth minima
use only `sqrt(dtype epsilon)*B`, a numerical acceleration scale, not a behavioral
margin. Real bumper gaps, real leaders, and the existing ramp-end obstacle are used.
Infeasible initial states receive bounded best-effort braking, not a safety promise.
The integrator and pathwise adjoint remain unchanged.

The two-sided insertion veto is inside `FirstPassageNet.boundary`, evaluated at the
existing post-motion decision geometry. For follower f and leader l:

`required = learned_s0_f + v_f*learned_T_f + max(0, v_f²/(2*learned_b_f) - v_l²/(2*learned_b_l))`.

Check the target follower behind the inserter and the inserter behind its new leader,
gathering actual per-slot settings through neighbour indices. An insufficient gap
sets the rollout boundary to negative infinity; no latent can override it. Exact
body-overlap vetoes and reservations are unchanged. Recorded-replay likelihood stays
ungated. The labeled rollout straight-through surrogate multiplies willingness by
`sigmoid(log(gap/required)/tau)`, reusing the existing learned positive pivot scale;
its FD certificate is for that continuous surrogate, not the hard veto.

This is the documented internal SUMO-style secure-gap form. **It is NOT equivalent to
external GAPACC `gs`, which omits the learned s0 term** (see R13E_OFFLINE.md).
No fixed driver types, forced merges, timers, gap padding, or mass floors
were introduced. Offline diagnostic cutoffs below never enter model decisions.

## Offline study and verification

Empirical arms: **off / emergency / emergency+gs complete** on gpuR, with all three
paired contrasts and 95% state-bootstrap CIs. The original historical panel is
validation-role; a preregistered strict train-role 48-state replacement was used.
The checkpoint remains unchanged. Tables, BASE verdicts and provenance are in R13E_OFFLINE.md.

`tools/r13e_checks/offline.py` prepares 48 unique TRAIN states × 2 seeds × 120 s,
three CRN arms, one to four one-thread CPU workers, Linux nice19. It checks strict checkpoint
identity and TRAIN provenance, emits BASE one-step NPZ evidence, and computes
1,000-draw paired 95% snapshot-cluster CIs with both seeds kept together. Metrics:
overlap pair-steps, TTC<1.5 s, realized a<−3 rows, HDV-inserter/follower braking
within six seconds after insertion, ramp waiting>60 s below 1 m/s, merges/run,
speed-deficit delay by region, and blowups. Definitions/denominators are in the report.
Formal empirical BASE verdicts: all arms reject BASE-1 and pass BASE-2/3/5 on that panel.

CPU verification: R13E suite 160 passed, plus the added baseline-integrity test;
regressions 279 passed, two real-data tests skipped. The Windows WSL Bash lookup
failed once; the same shell test passed with Git Bash. Ruff and Bash syntax passed.
Flag-off identity compares all returned tensors, RNG-derived outputs and state dicts
against actual 13C sources over four model cases, forward/step calls and six-step
rollouts. FD, feasible-state no-collision, veto, replay-parity and gate checks pass.

## Launch plan and risks

Stage A: both flags on, S-R12C FINAL as-is, core head/physical centre only, 30×120 s;
output `runs/c4_wm_v10_round13e_a`. Stage B: A FINAL, exact 13C repaired-courtesy and
residual recipe, core frozen and temporal off, 40×120 s + 60×300 s;
output `runs/c4_wm_v10_round13e_b`.

Future operator: `scripts/round13e_gate.sh --prepare` builds
`repo_r13e_<commit-sha8>` from committed archives, including pinned 13C comparison
sources. Supply complete offline evidence and matching preregistration, then smoke
both stages. Explicit `--launch-a` and subsequent `--launch-b` are separate operations;
B requires A checks and re-smokes from A FINAL. Neither was executed here.
`chk_r13e/{a,b}` uses the 13C paired stop rule: brake share>2×S-R12C or lane-1 MAE
delta>0.1 m/s²; measurement errors stop fail-closed, exact owned PID/start/argv only.
Diagnostics are CPU-only, at most four workers, nice19.

Risks: strict gaps may starve merges; conservative stopping may over-brake; stage-B
residual/courtesy may counteract the baseline envelope. Synthetic baseline safety
does not establish full residual-rollout collision freedom. Resolve through the
registered empirical checks and stage gates, without relaxing physical vetoes.
