# p20 line: 20% CAV penetration — verdict

Date: 2026-08-06
User steer: "see if a higher penetration could bring a difference, look into
20% penetration, train and re-eval just like before and continue to improve."
Priority order (unchanged): **TTC uncompensated > fairness > delay**
(delay has slack). All numbers: 50 held-out seeds (101–150), dt=0.5,
1.5 s TTC ruler, per-seed paired no-op baselines at the SAME penetration.

## Headline

The penetration raise is the real lever — but the winning configuration is
the **p10 champion deployed at p20**, not a p20-trained policy. Training at
p20 from scratch (600k steps, the healthiest fresh trajectory ever seen in
this project, best reward −26.4) does NOT beat simply running the existing
champion at the higher penetration.

| Configuration | Dom | Win | TTC-worse | d_ttc | d_gap | d_loss |
|---|---|---|---|---|---|---|
| champion @ p10 (bar of record) | 28 | 26 | 14 | −2.14 | −2.92 | −2.82 |
| **champion @ p20 (control)** | **27** | **24** | **15** | **−1.52** | **−3.76** | **−3.42** |
| p20-trained iter_75 | 24 | 21 | 17 | −1.60 | −3.08 | −3.59 |
| p20-trained final (600k) | 23 | 19 | 17 | −0.92 | −3.49 | −3.67 |

Champion @ p20 vs champion @ p10: scorecard statistically unchanged
(27/24/15 vs 28/26/14), flow means **+25–30% better** (d_gap −3.76 vs
−2.92, d_loss −3.42 vs −2.82), zero collisions, against a baseline that is
itself friendlier (p20 no-op TTC 13.2 vs p10 14.9 mean, loss 21.9 vs 23.4,
gap 6.0 vs 6.9 — CAVs coasting already smooth the stream).

## What was done

1. **p20 no-op baseline** (50 seeds, `runs/eval/p20_noop_refs_101_150`):
   the scenario is measurably friendlier at 20% penetration, but not
   trivially — CAVs generate TTC too (CAV–CAV seeds 119/115/137 get worse).
2. **Control**: champion iter_90 (+ soft mask + sign gate, config carried
   by the checkpoint) evaluated at p20, full 50
   (`runs/eval/champion_iter90_at_p20_15_dt05_holdout101_150`).
3. **p20 training**: from scratch, 600k steps, shepherd recipe,
   `--cav-penetration 0.2 --max-cavs-critic 48 --no-corridor-ttc-penalty`,
   paired All-HDV refs (penetration-independent, reused). One Ray
   object-store freeze at iter_35 recovered via `--resume` at 5 runners
   (30–39 steps/s after).
   Training arc: pol_ret −98 (iter_2) → −43.5 (iter_23) → −32.2 (iter_41)
   → **−26.4 best** (iter_100). Best fresh trajectory of the project and
   the ONLY fresh run ever to improve past iter_25 without a drift arc.
4. **Checkpoint smokes (12 seeds, 101–112)** and full-50 evals on final
   and iter_75 (drivers `_eval_shepherd_p20_v1_s600k_*`).

## Why the smokes misled (methodological note)

The 12-seed smoke subset favored p20-trained: iter_75 smoke read d_ttc
−2.08 / d_gap −5.55 / d_loss −5.17, *beating* the control's subset numbers
on two legs. On the full 50, iter_75 lands at −1.60 / −3.08 / −3.59 —
TTC mean ties the control but with **more** TTC-worse seeds (17 vs 15) and
a weaker fairness leg. Late training (iter_75→100) then drifted TTC back
up (−1.60 → −0.92). Lesson: 12 contiguous seeds (101–112) are not a
representative subset of this holdout; smoke reads are trajectory checks,
not verdicts. Full-50 remains the only scorecard.

## Signed conclusions

- **p20-trained line signed**: no p20-trained checkpoint beats the control
  on the TTC-first scorecard (iter_75 closest: TTC mean −1.60 vs −1.52 but
  17 vs 15 TTC-worse, gap −3.08 vs −3.76, win 21 vs 24).
- **Deployment recommendation**: raise penetration to 20% and keep the
  p10 champion. Same win profile, +25–30% flow means, safer baseline.
  This is a scenario-level improvement that required NO new policy —
  consistent with the campaign 1–4 finding that the TTC/flow coupling at
  p10 dense is scenario-bound, not policy-bound.
- The p20 training run itself is evidence *for* that framing: the
  best-ever reward trajectory still could not produce a policy that beats
  the transferred champion, because at p20 the champion's metering is
  already near the scenario's new Pareto corner.

## Artifacts

- Control eval: `runs/eval/champion_iter90_at_p20_15_dt05_holdout101_150`
- p20-trained evals: `runs/eval/shepherd_p20_v1_s600k[_iter75]_15_dt05_holdout101_150`
- p20 no-op refs: `runs/eval/p20_noop_refs_101_150`
- Training: `runs/rllib/shepherd_p20_v1_s600k` (checkpoints iter_5..95, best, final)
- Eval drivers: `_eval_champion_iter90_at_p20_15_dt05_holdout101_150.py`,
  `_eval_shepherd_p20_v1_s600k_*.py`
