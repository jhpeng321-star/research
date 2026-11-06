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

## Review fixes + integration

Owner decisions (2026-10-25, S-R13E-INT), recorded verbatim:

> B1 mid-run/stage-A-end stop rule reference = S-R12C evaluated with the SAME 13E flags on (same invocation/panel); stage A's end check must not block stage B on a flag-only difference; thresholds unchanged (2x hard-brake share, one-step MAE +0.1) relative to that reference.
>
> B2 apply the emergency envelope to the FINAL pre-clamp acceleration (after residual, courtesy and yield terms), gradients bounded; test that a large positive residual cannot exceed it.
>
> B3 secure-gap veto 'gs' must match GAPACC's gs exactly (drop the extra learned s0_f term); finish tools/r13e_checks/gs_equivalence.py and run it on a small event set.
>
> Y merge c4/v10r13y into c4/v10r13e (conflict-free per merge-tree) and ADD the COMBO300 'pass rule' to the yield: a lane-1 follower yields to a blocked lane-end merger only while it is still OUTSIDE the secure gap to that merger (the same learned-parameter secure-gap function as gs, no new constants); once inside it, the yield releases (follower may pass) - this removes the hold deadlock (COMBO300 arms EyG/EgsyG). Yield params remain learned (init documented). Synthetic tests: crawl-beside then release; no deadlock (follower stopped inside the gap does not stay held); no effect without a blocked merger.
>
> Launch config: 13E = E + gs + yield(pass rule) in both stages (stage A core recalibration from S-R12C incl. yield params; stage B 13C recipe with flags on). Also prepare a fallback config 13E_fb = E + yield(pass rule), no gs. GPU-busy guard in scripts/round13e_gate.sh (refuse while another training process holds the GPU; --wait polls).

Implementation:

- **Y (merge).** `c4/v10r13y` (e6e0cdd4) merged conflict-free (merge commit 73fa3200).
  `model.mainline_merge_yield_learnable` stays default-off (no tensors/RNG when off).
- **B3.** New `src/wms/physics/secure_gap.py`: `need = T_back*v_back + relu(v_back²/(2b_back) -
  v_front²/(2b_front))`, pass iff `gap >= need` (GAPACC `install_gate(use_T=True)` arithmetic and
  direct comparison; no s0, no multiplier). `FirstPassageNet.secure_gap_margin` uses it; the
  log-ratio feeds only the straight-through surrogate and its sign is forced to the direct
  decision (no 1-ULP log disagreements). Deliberate difference: invalid/non-finite settings or
  negative speeds fail CLOSED (GAPACC would accept a non-finite need). learned `min_gap` no longer
  enters gs. `tools/r13e_checks/gs_equivalence.py` now asserts identical decisions on the valid
  domain (synthetic: 4,096/4,096 random heterogeneous cases and 25/25 valid edge cases equal in
  fp32 and fp64; 0 log-vs-direct differences); `tools/r13e_checks/gs_events_small.py` runs the
  same comparison on recorded insertions of the first 12 R13E TRAIN-panel states (result below).
- **Y (pass rule).** In `mainline_merge_yield` (learned path only), after nearest-pair selection the
  follower is held only while `secure_gap_ok(gap - s0_f, need(v_f, v_m, T_f, b_f, b_m))`, i.e. its
  clearance beyond its learned standstill gap (the R13Y virtual leader sits s0_f behind the merger
  rear; SUMO compares gap - minGap with the secure gap) still satisfies the SAME gs law. Inside it
  the yield releases (bound = input acceleration) and the follower may pass. At standstill this is
  COMBO300's G rule (hold iff gap >= s0) with the learned s0. Discrete mask, like the pairing; no
  constants. Yield scalars stay learned: init 2 m/s² and 60 m (`merge_yield.log_decel/log_distance`).
  Note: a follower whose speed makes its secure gap longer than its distance passes; yielding is
  therefore confined to slow/near followers (queue zipper), the regime of the COMBO300 deadlock.
- **B2.** `apply_emergency_envelope` (accel.py) is applied ONCE in `WmSModel.forward` to the final
  pre-clamp acceleration (baseline + residual + noise + courtesy + yield cap); the composition uses
  the un-enveloped baseline (`_physics_step(..., emergency_envelope=False)`). `physics_accel_mps2`
  remains the enveloped baseline (CAV exec cap, diagnostics); direct `physics_accel()` = baseline ->
  yield cap -> envelope. d(out)/d(composed accel) ∈ [0, 1] (smooth minimum). Tests: large positive
  residual/noise and courtesy cannot exceed the envelope; envelope applied once.
- **B1.** `round13e_gate.derive_same_flags_reference`: the r12c reference arm is S-R12C (identical
  tensor bytes) with exactly the candidate's `physics.emergency_braking`, `physics.secure_gap_veto`
  and `model.mainline_merge_yield_learnable`; any other physics difference fails closed. Used for
  both the one-step MAE and the rollout hard-brake share in the same invocation/panel; r9 context
  arm stays flags-off. Thresholds unchanged. A flag-only difference can no longer stop A or block B.
- **Configs.** 13E A: E + gs + yield, trainable = R12C core head/centre + `merge_yield.*`
  (lr_scale 1000, same as the centre scalars). 13E B: 13C recipe + flags, `merge_yield` frozen.
  Fallback `c4_wm_v10round13e_fb{,_stage_b}.toml`: identical with `secure_gap_veto=false`
  (`--variant fb`, separate run/check/lock names). `scripts/round13e_gate.sh` refuses smoke/launch
  (exit 5) while `nvidia-smi` lists any compute process; `--wait` polls every 60 s; never signals.

### Integration evidence (2026-10-25, code d5ff39b1; no training)

- Tests (Windows, rl_driving, `--basetemp` in worktree): R13E/R13Y suites 236 passed, 4 Linux-only skipped;
  regressions (r13c_contract, r12c_physics_only, r12_base, r12_checks, r13a, r13f, r10b) 90 passed,
  2 skipped (R12_SLOW). Flags-off bit identity vs the actual 13C tip passes. gpuR smoke pytest: 240 passed.
- gs equivalence: synthetic 4,096/4,096 + 25/25 valid edge cases identical (fp32/fp64); recorded insertions of
  the first 12 TRAIN-panel states: 66 events, 0 decision differences (gpuR `diag_emerg/gs_events_small_d5ff39b1.json`).
- 300 s inference sanity (S-R12C checkpoint, gpuR CPU, 4 workers, nice 19; first 12 R13E TRAIN-panel states,
  1 seed, entries queue; `diag_r13e_sanity/run300`): 11 states complete for both 13E arms (mean [state CI]):

| metric | SUMO | E+gs+yieldG | E+yieldG |
|---|---|---|---|
| lane-1 exact-stop share 120-300 s | .006 | .082 [.043,.129] | .125 [.069,.184] |
| lane-1 discharge veh/min 120-300 s | 15.1 | 15.7 | 15.9 |
| ramp merges 240-300 s (late) | 6.7 | 4.5 | 4.5 |
| ramp merges/min 120-300 s | 6.2 | 4.6 | 6.2 |
| overlap pair-s | 74 | 101 | 105 |
| TTC<1.5 pair-s | 14 | 123 | 137 |
| HDV post-insertion follower hard brake % | 0.0 | 1.5 | 6.0 |
| ramp stuck roster cars | 0 | .36 | .18 |

  Flags-off base on the 7 states where it finished (3 base arms hit the 1,800 s timeout): overlap 2,624 pair-s,
  exact-stop .099, hard-brake 4.9%. No column freeze (COMBO300 Egsy stop share .76) with the pass rule.

Gate fix (65cc714e): the first smoke (d5ff39b1) failed stage A only on the gate's blanket finite check,
triggered by the weight-0 telemetry `merge_pivot_on_diagnostic_nll = inf` (also inf in S-R12C production and
the passed 13C smoke). Keys containing `_on_diagnostic_` are now exempt; all optimized metrics/losses must
still be finite. Re-verdict of the preserved failed-smoke artifacts with the fix: ok. Failed artifacts were moved
to `r13e_superseded_d5ff39b1_gatefinite/` on gpuR and the run was re-registered (REGISTRATION.md addendum).

Sanity final (all 12 states, both 13E arms; base arm 1,800 s timeouts on 4/12 states): exact-stop .085/.117
(SUMO .006); discharge 16.4/16.6 (15.4); merges/min 120-300 s 4.6/6.1 (6.3); late merges 5.3/5.4 (6.7);
overlap pair-s 95/99 (76); TTC<1.5 121/134 (15); follower post-insertion hard brake 1.4%/5.9% (0%); stuck .33/.17.
E+gs+yieldG under-merges mid-horizon (gs starvation risk); E+yieldG matches merge rate with more hard braking.
