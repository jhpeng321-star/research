# RL Timing-and-Candidate-Headroom No-Go Diagnosis

**Date:** 2026-08-30 (overnight goal)
**Status:** Phase C oracle-headroom gate **failed**; Phase D value screen run
as read-only diagnostic; **no RL training authorized, none started**. The
fixed 5–10% P30 rule remains the strongest honest damping arm
(`docs/IDM_FIXED_P30_LIMITED_DOMAIN_CONTROLLER.md`).
**Protocol:** `docs/IDM_RL_TIMING_CANDIDATE_HEADROOM_PROTOCOL.md`
**Sweep:** `runs/eval/idm_rl_timing_candidate_sweep_v1/` (18 cells, resumable,
all completed, no worker failure; no competing process at any point)
**Library:** `runs/eval/idm_rl_timing_option_library_v1/` (27 rows,
identity/effect/constraint validation passed for every root)
**Analysis:** `runs/eval/idm_rl_timing_oracle_headroom_v1.json`,
`runs/eval/idm_rl_timing_value_screen_v1.json`

## 0. Scope corrections (user-directed, 2026-08-30)

1. **Scope of the rejection.** This result rejects **candidate-1-versus-
   candidate-2 learning for the current fixed-P30 option library and the
   present causal observation families**. It is NOT a claim that all
   possible RL damping formulations are impossible (learned headway/speed-
   offset options, recurrent V2V residuals, RSU-assisted damping and
   multi-CAV absorption remain separately versioned candidates in §5).
2. **Sequential timing was not tested.** Recurrences 1/2/3 were analyzed as
   **separate states** (each its own predecision state with its own
   abstain/c1/c2 branches). A true sequential activate-now-versus-defer
   evaluation under a common predecision starting state and a common
   terminal horizon was **not** performed; that is the separate
   `IDM_SEQUENTIAL_TIMING_HEADROOM_PROTOCOL.md` proposal (Path B), which is
   review-only and not executed.
3. **Value caveat.** The held fixed-rule mean −0.0558 refers **only to the
   seven two-candidate-eligible held states** of this 27-state cohort. It
   does **not** replace the prior full limited-domain held value +0.04343
   (15-case, all-eligible-states cohort,
   `IDM_LIMITED_DOMAIN_P30_SCREEN_RESULTS.md`). The two cohorts differ
   (states with a second admissible candidate vs all eligible states), so
   their values are not interchangeable.
4. **Sweep size (exact artifact count).** The sweep comprises **18 cell
   runs** (penetration 0.05/0.10 × demand 1900/2000/2100 × recurrence
   1/2/3): 18 manifests, 216 discovery files (18 cells × 12 seeds), 108
   branch artifacts (27 states × 4 arms), 27 library rows. The "28" that
   appeared in the session summary was a typographical error; the inventory
   and this document use the manifest-derived 18.

## 1. Phase A — data gap and sweep

The inventory found: all existing 5–10% paired artifacts were rank-1 only and
p05 lacked recurrence-2. No candidate-2 branch existed anywhere. The frozen
protocol therefore specified the smallest development-only sweep: the
frozen runner at candidate-count 2 for p05/p10 × 1900/2000/2100 ×
recurrences 1–3, seeds 4121–4132, 4 workers, same treatment (P30), horizon
(60 s) and reward reference as every prior artifact.

## 2. Cohort size — the central finding

The 2-candidate cohort is **small**: 27 eligible states across 18 cells
(p05: 6, p10: 21). By cell: p05 m2000-r1 (1), p05 m2100-r1 (4), p05 m2100-r2
(1); p10 m1900-r1 (4), p10 m2000-r1 (7), p10 m2000-r2 (1), p10 m2100-r1 (6),
p10 m2100-r2 (3). **26 of 36 (cell, recurrence) slots produced no eligible
state**, and all 9 p05-m1900 and both mXX-r3 cells produced zero.

At 5% penetration, only about one third of the former rank-1 states have a
second admissible same-lane CAV at the same risk-onset step (6 of the former
candidate-count-1 p05 cohort of 19); at 10% it is about 91% (21 of the former
23). Candidate-selection is therefore mostly a **p10 phenomenon**, and the
headroom question is not even representable at p05 for most states (fixed
rule continues there). Split: train 20 rows (seeds 4121–4128), held 7 rows
(seeds 4129–4132, all p10; no p05 state survived into held).

Outcome diversity (all 27 rows): candidate-1 — 7 helpful / 11 harmful /
9 neutral; candidate-2 — 5 helpful / 6 harmful / 16 neutral. Exact
abstain-replay identity, decision identity, effects beyond replay noise and
hard constraints all validated on every root.

## 3. Phase C — oracle headroom results (held, n = 7)

Baselines (held mean paired reward): never 0.0000; always candidate-1
(= fixed 5–10% rule) **−0.0558** (0 helpful / 5 harmful / 2 neutral);
always candidate-2 +0.0146 (1/2/4); best fixed recurrence −0.0451 (0/4/3).

State-wise oracle (constrained): **+0.0265**, gain **+0.0716** over the best
fixed rule; selections: 1× candidate-1, 2× candidate-2, 4× abstain; outcome
mix 1 helpful / 0 harmful / 6 neutral; zero constraint failures; replay noise
zero. Per-row held detail:

| seed | cell | c1 reward | c2 reward | oracle |
|---|---|---|---|---|
| 4129 | m1900 r1 | −0.0003 (neutral) | −0.0209 (harmful) | abstain |
| 4129 | m2000 r1 | +0.0016 (neutral) | −0.0046 (neutral) | c1 (+0.0016) |
| 4130 | m2000 r1 | −0.0909 (harmful) | **+0.1827 (helpful)** | c2 |
| 4131 | m2000 r1 | −0.0767 (harmful) | −0.0420 (harmful) | abstain |
| 4132 | m2000 r1 | −0.1091 (harmful) | −0.0114 (neutral) | abstain |
| 4130 | m2100 r1 | −0.0748 (harmful) | −0.0024 (neutral) | abstain |
| 4130 | m2100 r2 | −0.0401 (harmful) | +0.0011 (neutral) | c2 |

Gate-by-gate (constrained oracle):

| gate | result |
|---|---|
| improves best fixed rule by ≥ 0.02 | PASS (+0.0716) |
| meaningful improvement in ≥ 3 seeds | PASS (4130, 4131, 4132) |
| > 1 non-abstain option used | PASS (c1 ×1, c2 ×2) |
| not a single outlier | **FAIL** — gain without seed 4130 = **−0.0583** (all positive gain is 4130; every other seed is ≈ 0 or negative) |
| adequate action/outcome diversity | **FAIL** — 1 helpful / 0 harmful / 6 neutral selections |
| preserves hard safety/service constraints | PASS (0 failures) |
| effects exceed replay noise | PASS (replay effect 0; realized deltas non-trivial) |

## 4. Phase D — value screen (read-only diagnostic, cannot authorize)

No family passes; held n = 7: `candidate_local` and
`candidate_local_rsu` (identical here): mean −0.0040, coverage 43%,
**0 helpful / 1 harmful / 6 neutral**, harmful fraction 33% (gate ≤ 25%
fails), not positive in two demands nor both penetrations; `rsu_pattern`
abstains everywhere (coverage 0). The model recovers the 4130 m2000-r1
candidate-2 selection (the only helpful held action) but adds a harmful
selection elsewhere — the "gain" is the same single-state dependence as the
oracle. There is no observable, deployable, generalizing signal in this
cohort.

## 5. Conclusion

Per the frozen gate chain, the oracle-headroom gate fails and the goal stops
here: **timing/candidate selection within fixed P30 is not a learnable RL
target at 5–10% penetration with this action space.** The library has value
only in one state (4130 m2000 r1, c2 +0.183 vs c1 −0.091) and the narrow
cohort (27 states) provides no separate evidence of a reproducible mechanism.
RL would overfit a single realization; per the protocol's §11 failure path:

- do not train; retain the fixed rule;
- propose (do not silently implement) a separately versioned actuator
  redesign.

Remaining material redesigns, ranked by expected scientific change (each
would require its own frozen treatment/mechanism protocol before any SUMO):

1. **Learned IDM desired-time-headway option** — changes the physical
   car-following response rather than only timing; new generation mechanism
   gate required (wave/braking effects directly testable in the existing
   paired runner).
2. **Learned bounded desired-speed offset** — small authority change around
   native IDM; mechanism gate plus dose/headroom audit.
3. **Recurrent local/V2V residual controller** — the failed screens suggest
   absence of local-only signal; costs a new state-contract and history data.
4. **RSU-assisted damping** — renames the architecture away from purely
   decentralized; merges into the wall direction.
5. **Multi-CAV coordinated absorption** — distinct treatment, closer to
   coordinated wall control than to decentralized damping.

## 6. What was not done

No PPO/MAPPO; no reward audit (Phase E never authorized); no future-frame,
dose, outcome or release-reason input into any model; no prospective seed
touched; no harmful or failed case discarded (all 27 states and all 54
action outcomes are retained and reported); no threshold changed after
opening held rows; no paper-scale data generated; the moving-wall predictor
was not touched. Exact authorization status: **no RL under any formulation
today**; the fixed-rule damping arm remains frozen pending prospective
acceptance.
