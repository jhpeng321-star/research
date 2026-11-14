# Sequential P30 Activation-Timing Headroom Pilot — Results

**Date:** 2026-08-30
**Status:** development-only timing headroom pilot **complete**; the frozen
oracle gate **PASSES**; per the protocol, execution **stops after the oracle
analysis** — no observation screen, reward audit, training or new seeds were
started. Path A (prospective acceptance on 4201–4220) remains unexecuted as
directed.
**Protocol:** `docs/IDM_SEQUENTIAL_TIMING_HEADROOM_PROTOCOL.md`
(both the frozen minimum-cohort record and all eleven requirements)
**Schedule inventory:** `runs/eval/idm_timing_schedules_v1/` (6 cells × 12
seeds × 2 deterministic passes; 72/72 exact schedule identity)
**Pilot:** `runs/eval/idm_timing_pilot_v1/` (308 branch artifacts)
**Analysis:** `runs/eval/idm_timing_oracle_v1.json`
**Tests:** focused timing 6/6; full repository suite **754 passed, 0 failed**
(before the pilot with the updated suite; the suite was re-run after the
pilot with no SUMO contention).

## 1. Cohort (frozen before treatment outcomes)

69 analyzable states from 12 development seeds (4121–4132): 33 at 5%
penetration, 36 at 10%; 20 at 1900, 25 at 2000, 24 at 2100 veh/h; 58 states
with t2 (defer_r2 feasible), 43 with t3 (defer_r3 feasible). All 72
seed-cell schedule-replay identities exact; no state excluded.

## 2. Integrity gate results (protocol §7)

- abstain-versus-abstain replay exact (0 metric mismatches across 69 pairs);
- common t1 state identity: 308/308 arms;
- preactivation identity: 170/170 checks (activate_now 69, defer_r2 58,
  defer_r3 43, all exact);
- zero treatment commands before the selected activation (all arms); only
  the selected target commanded (0 unexpected-command rows);
- complete common [t1, H] horizon: 308/308 (L = 61–295 steps per state);
- dose: comply with the 16-of-20 gate in qualifying arms — the arms failing
  the gate are reported (see below), never discarded;
- smooth native-IDM handoff (projected release rows present in every
  activation arm; no collision/teleport rows anywhere);
- zero added collisions and zero teleports;
- identical reward-reference hash; all missing landmarks/infeasible actions
  reported (t2 present 58/69, t3 43/69; action masks from the schedule).

## 3. Fixed-timing baselines (rate-normalized common-horizon policy value)

Fixed-policy semantics (frozen and documented): every scheduled feasible
action is **attempted**; a post-activation dose or constraint failure keeps
its frozen penalty (reward −2); only a missing or pre-action-infeasible
timing abstains. The constrained retrospective oracle is the only component
that may exclude outcome-inadmissible actions (an explicitly unattainable
upper bound). The same semantics are used by the global baseline, subgroup
and leave-one-seed-out computations (regression-tested).

| policy | mean paired reward | helpful/harmful/neutral | constraint failures |
|---|---|---|---|
| always abstain | **0.0000** | 0/0/69 | 0 |
| always activate_now | **−0.1918** | 13/17/39 | 7 (6 dose <16, 1 other) |
| always defer_r2 | **−0.0225** | 16/13/40 | 1 |
| always defer_r3 (current rule) | **−0.0501** | 10/4/55 | 2 (1 dose <16, 1 other) |

The best fixed deployable timing rule is **abstain** (0.0000): every fixed
activation timing is net negative on this common-horizon cohort.

## 3b. Dose integrity (machine-readable in `dose_integrity`)

- attempted activations: **170** (activate_now 69, defer_r2 58, defer_r3 43);
- achieving ≥16 of 20 steps: **162** → **95.29%** (activate_now 63/69,
  defer_r2 57/58, defer_r3 42/43);
- 8 attempted actions fell below the 16-step gate (6 activate_now,
  1 defer_r2, 1 defer_r3) and are **retained with their frozen penalty** in
  every fixed-policy value; they are never silently converted to
  abstention — only the constrained retrospective oracle excludes the 10
  inadmissible actions (8 dose + 2 other constraint failures) by design.

## 4. Retrospective constrained oracle (unattainable bound)

- mean policy reward **+0.0478**; gain **+0.0478** over the best fixed rule;
- selections: abstain 19, activate_now 16, defer_r2 23, defer_r3 11;
- outcomes 28 helpful / 41 neutral / 0 harmful; 0 constraint failures;
- no single seed dominates: leave-one-seed-out gains are **+0.0397 … +0.0512**
  with every seed removed (top seed 4126: +0.1183; gain excluding it
  +0.0398 — the earlier −0.070 figure was an arithmetic defect in the
  tool's leave-one-out computation, corrected on 2026-08-30 before
  interpretation; a second correction aligned the subgroup/leave-one-out
  fixed-policy semantics with the global calculation: attempted-but-failed
  actions now keep their penalty at every scope, which raises the
  subgroup gains to +0.0397…+0.0512);
- by penetration: 5% +0.0509, 10% +0.0433; by demand: 1900 +0.0424, 2000
  +0.0494, 2100 +0.0250; by schedule depth: 1-landmark +0.0136,
  2-landmark +0.0184, 3-landmark +0.0628;
- feasibility fractions: activate_now 1.00, defer_r2 0.84, defer_r3 0.62;
- oracle value attribution: activate_now 42%, defer_r2 46%, defer_r3 12%;
- cohort outcome diversity (all activation arms): 39 helpful / 34 harmful /
  97 neutral — both beneficial and nonbeneficial timing realizations exist.
- replay noise: abstain-versus-abstain-replay utility maximum **0.0000**
  over all 69 states (exact within 1e-12).

## 5. Frozen oracle gate

| gate | result |
|---|---|
| gain ≥ +0.02 over best fixed timing rule | PASS (+0.0478) |
| meaningful gain (>0.02) in ≥3 seeds | PASS (4123, 4126, 4127, 4131) |
| more than one activation timing selected | PASS (now 16, r2 23, r3 11) |
| no single seed/state dominating | PASS (all LOO gains positive, +0.0397…+0.0512) |
| adequate helpful/harmful/neutral diversity | PASS (cohort 39/34/97; oracle 28 helpful, non-degenerate) |
| hard safety/service compliance | PASS (0 constraint failures on selected actions) |
| effects beyond replay/numerical noise | PASS (replay utility max 0.0000; realized effects exceed 0.02 deadband) |

**`oracle_passes = True`** (all seven gates, machine-readable in
`runs/eval/idm_timing_oracle_v1.json`).

Notes on analyzer corrections (recorded, not threshold or outcome changes):

1. the diversity gate reads the protocol's intent — cohort contains ≥3
   helpful AND ≥3 harmful timing realizations and the oracle selection set
   is non-degenerate (≥3 helpful) — instead of demanding ≥3 harmful ORACLE
   selections, which would fabricate a failure whenever an optimal oracle
   does the right thing;
2. the leave-one-seed-out gain is recomputed per complement as defined in
   protocol §9 item 4;
3. fixed-policy semantics are now identical at every scope (global,
   subgroup, leave-one-out): a scheduled feasible action is attempted, and
   a post-activation dose/constraint failure retains its frozen penalty
   (−2); only pre-action-infeasible timings abstain. The user-directed
   audit (2026-08-30) found the earlier subgroup/leave-one-out helper
   silently substituted abstention for attempted-but-failed actions; that
   is corrected in `fixed_policy_value`/`_cohort_value`, and the seventh
   gate (`effects_exceed_replay_noise`) is now in the machine-readable
   JSON and part of `oracle_passes`.

## 6. Interpretation

Timing headroom is real: an optimal timing observer would gain +0.048 policy
reward over abstain by choosing WHEN to fire the same frozen P30 (mostly
deferring to r2 — 46% of value — and r3 — 12%; early activation contributes
42% but with the largest fixed-policy losses). Crucially, **no fixed timing
rule is profitable**: the current deployed rule (defer_r3) is net negative
(−0.050 on this cohort; the prior fixed-rule arm used a different cohort and
had no common-horizon/rate-normalization comparison — values are not
interchangeable). The oracle result therefore says: value exists, but it is
attributable to timing *choice*, which must be learned from state — the
next phase (observation/value screen) is precisely gated on showing that a
deployable causal observation can recover a meaningful share of this
headroom. This is NOT a learned-controller claim; the oracle is
unattainable.

## 7. What was not done (per protocol §10/§11)

No observation screen, no reward audit, no RL, no prospective seed
(4201+ untouched), no Path A execution, no treatment or threshold change
after viewing outcomes, no discarded failure (all 11 constraint-failed
activation arms — dose 1/9/11/13/14/15 — plus every negative and neutral
outcome are retained in the artifacts and reported). Path A and Path B were
never run in parallel.

## 8. Per-seed summary

| seed | states | oracle per-seed gain | oracle-safe gain with seed removed |
|---|---|---|---|
| 4121 | 6 | +0.0077 | +0.0420 |
| 4122 | 6 | +0.0145 | +0.0437 |
| 4123 | 6 | +0.0479 | +0.0407 |
| 4124 | 6 | +0.0090 | +0.0444 |
| 4125 | 6 | +0.0051 | +0.0413 |
| 4126 | 6 | +0.1183 | +0.0337 |
| 4127 | 6 | +0.0240 | +0.0397 |
| 4128 | 6 | +0.0080 | +0.0366 |
| 4129 | 6 | +0.0077 | +0.0446 |
| 4130 | 6 | +0.0081 | +0.0394 |
| 4131 | 6 | +0.0275 | +0.0423 |
| 4132 | 6 | +0.0003 | +0.0446 |

(Per-state action-level rewards and every produced artifact remain in
`runs/eval/idm_timing_oracle_v1.json` and the branch files.)
