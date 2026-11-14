# Path A — Prospective Acceptance Protocol for the Frozen Rule-Based P30 Damping Arm (Revised)

**Status:** **frozen, user-authorized, and executed 2026-08-30; acceptance
fails.** No outcome from seeds 4201--4220 was opened before this freeze. Path A
was not run in parallel with Path B. Results are in
`docs/IDM_P30_RULE_ARM_PROSPECTIVE_ACCEPTANCE_RESULTS.md`.
**Date:** 2026-08-30
**Arm under test:** the frozen rule-based 5–10% P30 damping arm
(`docs/IDM_FIXED_P30_LIMITED_DOMAIN_CONTROLLER.md`): abstain outside 5–10%
CAV penetration; inside, activate the frozen P30 treatment
(`activation_speed_ceiling_v1`, runway rank-1 candidate, third risk-onset
decision per cell, 20 option steps, legacy smooth jerk-limited return to exact
native IDM, hard projection authoritative). The separately named projected-
release variants are not part of this prospective treatment.

## 1. Purpose and position in the paper

Prospectively verify the strongest honest decentralized damping arm on
untouched seeds before it can be presented as the paper's damping control
arm. Moving-wall revalidation (master plan Phase 6) and the escalation
predictor (Phase 7) are gated later protocols. Acceptance of this rule arm
does **not** authorize later testing of a redesigned learned controller on
the same prospective seeds.

## 2. Seeds and integrity

- First prospective acceptance set: **seeds 4201–4220** (20 seeds,
  untouched, never used for any tuning or design).
- Untouched confirmation set: **seeds 4221–4240** — opened only after the
  controller and acceptance analysis are frozen (rule arm today; learned
  checkpoint later, should Path B pass and a learned controller be selected).
- Final paper seeds **5001–5060 remain reserved; not opened**.
- Seeds 4107–4132 remain development-only; a seed ledger records the run.
- Every no-decision, constraint failure and harmful outcome is retained.

## 3. Cells and decision contract

- Cells: penetration 0.05, 0.10; mainline demand 1900, 2000, 2100 veh/h.
- Decision: frozen `risk_onset`, decision index 3, max 3 landmarks per cell,
  `runway` ranking, candidate count 1.
- Branches per accepted decision: abstain, abstain replay, P30 activation,
  from identical predecision state (exact replay and predecision identity
  required).
- Horizon: safety 60 controller steps (60 s); service 60 steps; reward
  reference `runs/eval/idm_decentralized_lower_reward_refs_v1.json`
  (SHA-256 `737b9ce4bb1945f7681b807941c86509ae3143128bd0729d2cb2aee5237c5b58`).

## 4. Metrics, deadband and constraints (exact)

- Primary effect: equal-weight reference-scaled reduction of corridor
  severe-braking and acceleration energy over the identical paired horizon,
  plus the **Path-A acceptance reward** (abstain 0; activation tanh(u) if all
  Path-A constraints, including the additional immediate-follower condition,
  pass; otherwise −2). This is reported separately from the reusable
  `IDM_BINARY_OPTION_REWARD_V1`, whose base contract does not include that
  additional follower condition.
- **Deadband: 0.02, fixed and never raised after prospective replay is
  observed.** If abstain-replay noise exceeded 0.02, the treatment is
  reported as not exceeding replay noise (no deadband adjustment).
- Constraints (each must hold; exact numbers):
  - complete horizon (exactly 60 service steps);
  - ≥ 16 of 20 treatment commands delivered;
  - zero unexpected commands;
  - added collision count ≤ 0 and added teleport count ≤ 0 (relative to
    abstention);
  - low-TTC exposure step delta ≤ 2 (corridor low-TTC count per step > 0
    steps, 1.5 s threshold);
  - immediate-follower severe-braking event delta ≤ 0 (follower accel
    < −3.0 m/s², per step, versus abstention);
  - mean corridor delay-rate increase ≤ 1.0 s/vehicle;
  - arrival loss ≤ 1 vehicle;
  - merge-discharge loss ≤ 1 vehicle;
  - no treatment safety abort when recorded.
- Classification: helpful = composite > +0.02 with all constraints pass;
  harmful = composite < −0.02 or any failure; neutral otherwise.

## 5. Acceptance criteria (predeclared, bounded-harm)

1. positive mean paired physical effect and mean paired reward over
   abstention;
2. helpful activations more frequent than harmful activations;
3. at least 70% helpful-or-neutral activations;
4. harmful fraction ≤ 25% of activations;
5. zero added collisions/teleports (constraint above);
6. low-TTC exposure increment ≤ 2 steps and immediate-follower
   severe-braking increment ≤ 0 (constraints above);
7. service noninferiority (delay/arrival/discharge constraints above);
8. effects exceed replay noise (abstain replay exact within 1e-12, and the
   composite genuinely exceeds the 0.02 deadband in qualified nonneutral
   cases);
9. no single seed responsible for the mean gain (every leave-one-seed-out
   opportunity-level mean remains strictly positive); the seed-cluster
   bootstrap 95% interval is reported as uncertainty, not used as an
   additional gate;
10. all outcomes reported per seed (no-decision, harmful, constraint-failed
    included).

Statistical analysis: seed-level paired effects, paired mean with 95% paired
bootstrap CI, per-seed table, failure list. The arm need not win every seed.
The deployment-policy mean is computed over all 20 seeds x 6 cells; a cell
with no third qualified decision is an explicit abstention with zero effect
and zero reward. Classification fractions are computed over attempted
activations. The bootstrap resamples the 20 seed-level six-cell means 20,000
times with RNG seed 20260830. Every leave-one-seed-out mean must remain
strictly positive.

## 6. Exact execution

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
$env:PYTHONPATH = "D:\research-dense"
$py = "C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe"
& $py tools/run_idm_paired_option_penetration_grid.py `
  --output-root runs/eval/idm_p30_rule_prospective_acceptance_v1 `
  --seeds 4201,4202,4203,4204,4205,4206,4207,4208,4209,4210,4211,4212,4213,4214,4215,4216,4217,4218,4219,4220 `
  --demands 1900,2000,2100 `
  --penetrations 0.05,0.10 `
  --decision-index 3 `
  --workers 4 `
  --reward-refs runs/eval/idm_decentralized_lower_reward_refs_v1.json `
  --python $py `
  --evidence-role damping_acceptance

& $py tools/analyze_idm_p30_rule_prospective_acceptance.py `
  --grid-root runs/eval/idm_p30_rule_prospective_acceptance_v1 `
  --output runs/eval/idm_p30_rule_prospective_acceptance_analysis_v1
```

## 7. Expected runtime

20 seeds × 6 cells gives 120 opportunities and at most one accepted decision
per seed-cell: at most 120 paired decisions and 360 branches. Prior development
timings imply approximately 40--75 minutes at four workers; budget **1--2
hours wall** conservatively. The run is resumable per branch.

## 8. Expected effect on the paper

- Acceptance passes: the frozen rule arm is the paper's low-authority
  decentralized damping arm and counterfactual control arm; wall
  revalidation and the escalation predictor follow their own frozen
  protocols.
- Acceptance fails: the arm is reported as a development-only conservative
  fallback; the paper's high-authority contribution becomes the coordinated
  moving wall; no learned-activation claim.

## 9. Risks

- Conservative arm: acceptance checks bounded harm, not large benefit.
- Metastable seeds may not reproduce the development profile; per-seed
  reporting and grouped criteria handle this honestly.
- A rule-arm acceptance does not imply anything about the wall; wall and
  predictor gates remain.

## 10. Prohibited actions

Opening 4221–4240 before confirmation, opening 5001–5060, tuning thresholds
after viewing acceptance results, relabeling prospective seeds as
development, using the same prospective seeds for a redesigned learned
controller, training any controller, wall/predictor experiments without
their own frozen protocols, running Path A in parallel with Path B.
