# Campaign 2: breaking the TTC/flow coupling — new action spaces & fresh basins

Date: 2026-08-05
Starting point (campaign-1 champion): `shepherd_fair_safe_v10c_softmask/iter_90`
+ step soft mask + recover sign gate = **28/50 domination, 26/50 win-rule,
14 TTC-worse** (1.5 s ruler, seeds 101–150, dt=0.5).
Campaign-1 conclusion to beat: the TTC/flow coupling is structural *within
the 5-mode discrete action space* — every metering change buys flow at TTC
cost. Campaign 2 attacks the coupling's mechanism directly.

## Mechanism hypothesis

The coupling lives in the action parameterization, not in metering per se:
discrete modes issue step-change speed commands (×0.90 slowdown etc.), and
step changes inject HDV–HDV compression waves (the TTC autopsy's dominant
source, ~85%). A **continuous acceleration action** (±1.5 m/s², per-second
updates) can meter *gradually* — same fairness effect, smaller disturbance.

## Findings so far

1. **The iteration-3 crash is an initialization pathology, not a reward
   problem.** Three flat_marl variants — binary control cost + TTC penalty
   (v1), magnitude-scaled cost + penalty (v2), magnitude cost + no penalty
   (v3) — ALL crashed at iteration ~3 (−667, −590, −677) despite three
   different reward designs. The invariant: RLlib's default Gaussian
   std=1.0 m/s² makes the untrained continuous policy jerk full-range from
   step 0 → instant gridlock → the critic sees no contrast → PPO flails.
   The discrete action space ships the same protection for free (20% coast
   at random init + bounded modes). Fix: bias the final layer's log_std
   half to −2.0 (std 0.14) — `_init_log_std_bias` in `models.py`,
   commit `0ae32ed`. With it (v4): iter_3 **recovered** to −402.6 and the
   run converges (−303, −436, −402, −422, −371, −357, −342…).
   Side fix: magnitude-scaled intervention costs on the continuous path
   (|a|/a_full), commit `c38674a` — the binary cost is an undescendable
   cliff for any Gaussian policy.
2. **The fresh discrete basin (B) learns fast and shows the same coupling
   early.** iter_25 smoke (12 seeds): pass 9/12, dL −2.30, dG −2.19 —
   near champion means at 25% trained — but TTC-worse on the first 4 rows
   (+6…+19). The coupling is not a property of the iter_90 checkpoint; it
   appears in every metering-capable policy early in training. The open
   question is whether a *converged* fresh policy resolves it better.
3. **Process lessons.** (a) eval drivers MUST have unique run_tag_prefix
   (tripinfo collision corrupted a parallel pair); (b) nohup-style
   backgrounding loses task tracking — use tracked tasks; (c) verify kills
   by process table, not assumption (an operator error killed the healthy
   Run B; it resumed from checkpoints with zero loss).
4. **The execution-context discriminator does not exist (double null).**
   The action-wake probe (champion, 6 seeds, ~2400 interventions logged)
   measured the corridor TTC wake of every fair_balance firing: 28% leave a
   +1 wake, 72% none. Neither follower gap (96.9m vs 100.3m means),
   corridor state, ego speed (14.6 vs 13.9 m/s), nor lane position (311 vs
   325m) separates the two groups — quantiles overlap throughout. The wake
   is stochastic/systemic, not deterministically triggered. Consequence:
   context-triggered execution modulation (incl. the follower-aware
   execution built behind `shepherd_follower_aware_exec`, kept with this
   evidence) cannot select the harmful cases; only ALWAYS-graded execution
   trained in can help — which is precisely what the continuous Run A
   learns. fair_balance also fires essentially never with a close follower
   (n=1/487), so that modulation would rarely activate anyway.

## Experiments

| # | Run | Config | Dom | Win | TTC-worse | Verdict |
|---|-----|--------|-----|-----|-----------|---------|
| A v1–v3 | flat_marl (reward variants) | continuous ±1.5 | — | — | — | crashed iter_3 (init pathology, diagnosed) |
| A v4 | flat_marl_dense_s600k | continuous ±1.5, log_std init −2.0, mag costs, penalty 0.0, 600k from scratch | — | — | +18/seed @iter_50 | KILLED at iter_51 (signed): iter_50 smoke TTC +18/seed mean (≈ +20 at iter_25, not improving), flow mixed, trails B-at-25% on every axis at equal budget. **The coupling is NOT parameterization-escapable** — continuous metering disturbs the corridor as much as discrete step-changes |
| B | shepherd_fair_safe_v11_fresh_s600k | discrete 5-mode, corridor_ttc_penalty 0.2, 600k from scratch | 0 | 0 | 50 | REJECTED (signed): worst TTC of the campaign. **Penalty indictment**: 0-for-3 in training (v10e, A v2, B); champion predates it |
| C | shepherd_fair_safe_v11_nopen_s600k | discrete 5-mode, penalty 0.0 (poison control), 600k from scratch | — | — | ~+12/seed @iter_50 | KILLED at iter_50 (signed): degraded from its iter_25 peak (8/12, dL −2.96 → 3/12, dL +0.09), TTC flat ~+12 — follows B's arc. **Penalty was NOT the poison** |

A-vs-B at identical reward isolates the action-space variable. B also tests
whether a fresh basin lands above iter_90 (all campaign-1 drift evidence was
warm-start; from-scratch is untested).

Eval: full 50-seed holdout under both rules, drivers
`_eval_flat_marl_dense_s600k_15_dt05_holdout101_150.py` (A) and
`_eval_fair_safe_v11_fresh_s600k_15_dt05_holdout101_150.py` (B), same noop
refs as campaign 1 (same 1.5 s ruler).

## Final conclusion (campaign 2)

**No configuration beat the champion. The iter_90 basin stands alone.**

1. **Continuous action space (A v1–v4)** — dead. Crashed on an
   initialization pathology (fixed: log_std init −2.0), then converged but
   trailed the discrete control on every axis at equal budget (TTC
   +18/seed at 50% trained). The TTC/flow coupling is **not
   parameterization-escapable** — smooth continuous metering disturbs the
   corridor as much as discrete step-changes.
2. **Fresh discrete basin with penalty (B)** — dead. Flow learned fast
   (champion-class by 25%), then TTC degraded to the campaign's worst:
   0/50 domination, 0/50 wins, 50/50 TTC-worse at iter_75.
3. **Fresh discrete basin without penalty (B-nopen)** — dead. The poison
   control: identical TTC-roughness at 25% (penalty exonerated), flow
   peaked champion-class at iter_25 then degraded by iter_50. Same arc
   as B: **peak ~25%, then drift.**
4. **The champion's iter_90 basin is not reproducible by recipe** — two
   independent fresh runs with the same recipe peaked early and degraded,
   while all warm-start continuations from iter_90 drift (campaign 1, 4×).
   Its TTC-efficiency is a property of its specific training history, not
   of any configuration in the recipe.
5. **The TTC/flow coupling is robust** — across action parameterizations
   (discrete/continuous), reward designs (±penalty, ±shaping), and basins.
   Every metering-capable policy pays TTC for flow; iter_90's exact
   operating point is the only observed exception, found by its training
   run, not by design.

### Recommended next directions (out of tonight's scope)

- **Multi-seed basin search**: N short fresh runs (e.g., 4 × 250k), keep
  the best at 25–50%, continue only that one — cheap lottery-ticket search.
- **Demand curriculum**: fresh run with density ramping light → dense, so
  the policy learns metering in calm traffic first — the one structural
  variation not tried tonight.
- **Keep the corridor-TTC penalty OFF for future training** (0-for-3 as a
  training term; chaos amplifier on untrained policies); revisit only as a
  warm finish on an already-converged policy.

## Checkpoint of record

`runs/rllib/shepherd_fair_safe_v10c_softmask/checkpoints/iter_90`
+ step soft mask + recover sign gate — **28/50 domination, 26/50 win-rule,
14 TTC-worse** at the 1.5 s ruler, seeds 101–150 (unchanged from campaign 1).
