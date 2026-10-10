# WD-01 — DESIGN §3 v2: Part B as a planner-policy value function

Design text, 2026-09-26. **This replaces DESIGN_PARTS_A_B.md §3.1 ("plan held to
the horizon").** It implements the owner's Part B decision relayed by main
discussion (2026-09-26). No training is dispatched until this design is reviewed.
Prerequisite: action-conditioned Part A with CAV controller emulators (ablation A2).

## 1. Estimand

A decision at time t0 in state s covers only the CURRENT action a for a
**commit window Δ** (start: Δ = 30 s; test 10/30/60 later). An action is a per-CAV
assignment plus walls. Its consequences are judged over the full horizon T = 600 s
from t0, **assuming the planner keeps re-deciding afterwards** under its own policy π.

Q^π(s, a) = E[ D[t0, t0+Δ](a) + V^π(s'_a, τ = T − Δ) ]

- D[t0, t0+Δ](a) is the contract §4.1 total delay accrued inside the commit
  window. It is computed by **Part A** at its internal 0.5 s step. This matters
  because the 1 s output reads about 9% low (WD-01 p4).
- s'_a is Part A's end-of-window state under action a (one per member m and
  sample k).
- V^π(s, τ) is the expected total delay over the remaining τ seconds, from state s,
  with π in control. It is **finite-horizon and τ-conditioned**, with
  V^π(s, 0) = 0. The contract horizon is truncated at t0 + T, so a τ-free value
  would be wrong.
- The planner's comparison is

  ΔQ(a) = [D_A(a) − D_A(native)] + [V^π(s'_a, τ) − V^π(s'_native, τ)].

  Both sides use the **same V ensemble member m and the same sample k** (common
  random numbers). ΔQ therefore stays an [M, K] paired quantity, and the planner's
  ≥ 80% agreement rule and margin apply unchanged.
- "Native" in ΔQ still means the §5 native reference *for the commit window*.
  After Δ, both branches continue under π, so the comparison is "this action now
  vs native now, then the planner as usual" (option b).

### 1.1 Walls under a commit window

A wall chosen now usually lasts longer than Δ (assembly lock, hold). The action
a at t0 is the *decision to start or continue* a wall. V^π sees the wall's
phase, speed and time in phase in s'_a, and π decides later releases. Assembly
is locked, so π cannot release during assembly. V must therefore take wall-state
tokens as input (§3).

### 1.2 Safety beyond the commit window

The owner's relative safety rule (no increase in TTC<3 exposure or hard braking
vs native) must not end at Δ. V gets auxiliary value heads for the remaining
horizon under π:
- V_exp(s, τ): TTC<3 exposure in veh·s;
- V_brake(s, τ): hard-braking seconds;
- P_coll(s, τ): probability of any collision.

Relative safety is then window difference (Part A) plus future difference (these
heads). The absolute safety gate stays on Part A's window (collision, min TTC ≥ 1.5 s).

## 2. Contract impact (to be agreed with PL-01; would become `wd01_predict/2`)

| item | v1.1 | proposed v2 |
| --- | --- | --- |
| plan semantics | plan held from t0 (Part B assumed held to horizon) | plan = action for the commit window `commit_s` (Δ). Continuation is π, identified by `meta.continuation_policy_id` (V_k version) |
| inputs | — | `commit_s` (default 30) in `predict_many`; `continuation` ∈ {"planner", "native"}. "native" = the old option-a semantics, kept for the perfect-simulator baseline and for V0 |
| delay split | part_a = rollout window, part_b = held-plan remainder | `delay_part_a_s` = D[t0, t0+Δ]; `delay_part_b_s` = V^π(s', T − Δ) |
| safety window | `safety_window_s` = Part A horizon | absolute gate: `safety_window_s` = Δ (Part A). New [M,K] fields: `future_ttc_lt3_exposure_s`, `future_hard_brake_s`, `p_future_collision`, plus their `native_` versions, from the V heads |
| trajectories | Part A window | commit window Δ (the interaction check is unchanged in spirit) |
| perfect simulator | branch with plan held | must implement the same continuation: after Δ, the **planner runs closed-loop inside the branch** (π_k). This is expensive, see §5. `continuation="native"` remains as a cheap perfect-sim mode for V0-level checks |
| meta | — | `commit_s`, `continuation`, `continuation_policy_id`, `value_model_id` + sha256 |

### 2.1 Additions agreed with PL-01 (review of 2026-09-26)

- Δ is PREDICTION semantics ("this action for Δ, then π"), not a dispatch
  dwell. The planner still re-decides every tick (or at `decide_every_s`).
- **Future wall heads** [M,K] per wall_id:
  - `p_assembly_completes`;
  - `expected_assembly_time_s`;
  - `p_assembly_abort`.
  These are needed because assembly often exceeds Δ, so the in-window
  `assembled` flag is often NaN. There is no native_ version. V is trained only
  on data where π respects the assembly lock.
- **Fail-closed for future heads:** `future_cell_valid` / `native_future_cell_valid`
  bool[M,K] are REQUIRED in v2. False means the V output is unusable, and the
  consumer fails that cell closed. NaN never signals failure (same rule as §8.7/§9).
- **Future collision threshold:** the owner's "no predicted collision in ALL
  samples" becomes `p_future_collision ≤ p_max` in every cell. p_max is an
  OWNER decision. The planner config default of 0.01 is pending the owner.
- **Future min-TTC:** a `future_min_ttc_q05_s` head (5th percentile of the
  minimum TTC over the remaining horizon), plus a native_ version, so the 1.5 s
  floor can be applied beyond Δ. Until that head is validated, the floor is
  enforced only within Δ. This is recorded as a **known weakening** of the
  absolute safety rule.
- **Versioning:** v1.1 stays frozen. Validators and conformance are dual-version
  (1.x and 2). The planner supports both.
- **Closed-loop continuation inside perfect-sim branches** uses the LEARNED
  V_k / Part A (a perfect sim inside a perfect sim is infeasible). It uses a live
  pass-through hybrid-state builder at a coarse π cadence (10–30 s). This is owned
  by the planner (PL-04). Most perfect-sim checks use `continuation="native"`;
  closed-loop runs only on a registered subset for regret.

### 2.2 Owner decisions (2026-09-26, via main discussion)

- **Future collision veto:** a plan is vetoed if `p_future_collision > 0.01` in
  ANY [M,K] cell, AND vetoed if its `p_future_collision` exceeds native's in the
  paired cell (relative condition). The relative tolerance is configurable and
  defaults to 0.
- **Temporary weakening accepted:** the 1.5 s min-TTC floor is enforced only
  within the 30 s commit window until `future_min_ttc_q05_s` is validated.
  **GATING ITEM G-1:** that head MUST be validated before any live or
  closed-loop use of the planner outside controlled experiments.
- **Order:** Part A action conditioning (controller emulators, A2) first. Part B
  round 0 (V0, native value) runs on the GPU after the owner's restart.

### 2.3 Owner decisions (2026-09-27)

- A native collision does not veto a safe plan; it is logged.
- The 30 s window must be non-increasing on its own for exposure and hard
  braking, in addition to the summed rule.
- CAVs get emergency braking (SUMO speedMode 27) in future data. Chunk-1
  high-penetration outcomes are **pre-fix**: do not mix them with post-fix
  data without a `cav_braking_fix` flag, and do not use pre-fix
  high-penetration branches to fit CAV controller emulators without
  checking that the braking behaviour matches.

Unchanged: [M,K] pairing, native reference §5 (for the commit window),
fail-closed (§8.7/§9), entrance queues (§8.14), merge_observed (§8.15).

## 3. Model

- **Inputs** at s' (end of Part A):
  - the lane-bin field on the 3 physical lanes (count, mean speed, speed spread);
  - CAV tokens (position, lane, speed, current action, wall id/phase/speed/time in phase);
  - entrance queues;
  - a future-demand summary from the arrival draw (per-minute counts per entrance
    for the remaining τ);
  - τ;
  - the demand regime.
- **Architecture:** as in the original §3.3 (conv over lane bins + transformer over
  CAV/wall tokens), about 1–3 M parameters, GPU-batched over plans × M × K.
- **Heads**, all τ-conditioned:
  - V (delay): symlog + twohot distributional (or quantile), per-regime return
    normalisation;
  - V_exp and V_brake: symlog + twohot;
  - P_coll: Bernoulli.
- **Ensemble:** M = 5, bootstrap over roots. Member m of Part B pairs with member m
  of Part A.

## 4. Training: policy iteration, critic-style

**Step 0, native value V0** (possible now for the native part):
- Data: native trajectories: chunk-1 native branches and bases, and the p2C native
  dataset where long enough.
- Target: finite-horizon multi-step TD(λ).
  - Along a native trajectory, V(s_t, τ) ← D[t, t+Δ] + V̄(s_{t+Δ}, τ − Δ),
    with n-step / λ-returns and a slow target network.
  - Where a trajectory covers t + τ, the full Monte-Carlo return anchors the
    targets.
  - τ is sampled in [0, 600].
- Checks:
  - held-out-root MC return vs V0 (calibration of the twohot distribution);
  - n-step consistency;
  - per-regime error.

**Step 0 amendment (2026-09-27, main discussion; V0 dataset c18cd2b).**
- Chunk-1 "base" runs are the fleet-uniform DAMPING incumbent, not native.
  V0 is therefore trained as a policy-CONDITIONED value
  V(s, τ, policy_id), with policy_id ∈ {native, damping_fleet, …}.
  - The damping rows train the shared state encoder.
  - V0_native = V(·, ·, native) is the planner's reference.
- Native evaluation uses native-source rows only.
- Report per-policy label counts by τ. The native branches give only about 3
  τ = 570 labels each, so long-τ native coverage is thin.
- If native coverage at large τ proves insufficient, write a costed option for
  the owner: native-policy base runs, or longer native branches, in a later
  chunk. Chunk 2 is unchanged.

**Step k** (needs closed-loop runs, a new campaign type):
1. π_k = planner using Part A + V_{k−1} (with its margins, agreement rule and
   safety gates).
2. Run π_k closed loop in SUMO (planner in the loop, real decisions every tick)
   across demand regimes → trajectories + decisions, logged at 0.5 s with the
   campaign fields.
3. Fit V_k by TD(λ) on π_k data (on-policy). Earlier iterations' data may be mixed
   in with a `policy_id` feature, and weighted down.
4. Evaluate decision regret vs the perfect simulator (§6). Stop when regret and
   ||V_k − V_{k−1}|| on held-out states stabilise.
   - Guard against chattering: the planner's margin and ≥ 80% agreement rule stay
     in place.
   - Guard against the optimizer's curse: ensemble spread inflates the margin.

**Later:** imagination rollouts (Part A + π_k inside the model) can add V targets
cheaply, but only after Part A's action conditioning passes its gates. Imagined
targets never replace SUMO data for evaluation.

**Paired-difference supervision (optional, strong):** campaign sibling branches
(action a vs native for Δ, both continued by π_k in SUMO) give direct ΔQ targets.
They check and fine-tune V differences, which is what the planner consumes.

## 5. Data needs

| data | purpose | status / owner |
| --- | --- | --- |
| native bases/branches ≥ t0 + 600 s at 0.5 s | V0 TD and MC anchoring | chunk 1 (WM-36/41); confirm run lengths reach t0 + 600 s |
| p2C native dataset (60–600 s) | extra V0 TD transitions (short τ only) | available |
| action branches (commit Δ, then native) | Part A action conditioning; V0-level ΔQ checks (option a) | campaign plan |
| **closed-loop planner runs (new type)** | V_k on-policy TD per iteration | **new campaign type; budget for the owner** |
| perfect-sim regret evaluation (closed-loop continuation per candidate) | owner priority (1) | **expensive; subsampled** |

Rough budget per policy iteration (for the owner's decision; to refine with
learning curves):
- closed-loop runs: 4 demand levels × 15 roots × (300 s warm-up + 600 s) ≈ 15 sim-h;
- regret evaluation: 50 decision points × 5 candidates × 600 s closed-loop
  continuation ≈ 42 sim-h;
  - cheaper variant: continuation="native" (option a), ≈ the same seconds but
    without planner compute;
  - or a 20-point subset ≈ 17 sim-h.
- Expect 3–5 iterations until regret stabilises.

## 6. Evaluation (owner priorities)

1. **Decision regret**: extra delay of the learned planner's chosen actions vs the
   perfect simulator with the SAME continuation semantics, on held-out roots.
2. **Ranking**: Spearman / top-1 of ΔQ across candidates, and sign accuracy vs native.
3. **Safety recall**: window gate (Part A) plus future heads vs realised
   closed-loop exposure / hard braking / collisions.
4. **Calibration**: M×K coverage of ΔQ and of V vs MC returns; twohot PIT.
5. **Speed**: predictions/s for predict_many with Part A (Δ = 30 s) + V heads, on
   CPU and the RTX 3060.
6. **Policy-iteration diagnostics**: regret and V change per iteration, and
   decision churn.

## 7. Open points for review

- Δ = 30 s vs Part A's native horizon evidence. Part A only needs to be good up to
  Δ now, which relaxes the Part A horizon question.
- Does the perfect-simulator implementation (planner-owned) accept closed-loop
  continuation cost, or does it use option-a continuation for most checks with
  closed-loop continuation on a subset?
- V's dependence on π makes V_k non-stationary across iterations. Every
  prediction must carry `continuation_policy_id`, and ledger entries must pin
  the V_k sha256.
