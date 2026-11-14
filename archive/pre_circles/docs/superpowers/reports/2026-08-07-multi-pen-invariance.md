# Multi-penetration training: penetration-invariance verdict

Date: 2026-08-07
Question: does ONE policy trained across penetrations (5/10/20% sampled per
episode) work at all penetration levels, and does its absolute performance
improve monotonically with penetration?

Model: `runs/rllib/shepherd_multi_pen_v1` (600k, best reward −37.27,
127-dim obs — includes the metering-load V2V feature). Evaluated at p10, p20,
p30 vs each level's own no-op baseline, 50 held-out seeds (101–150).

## Result

| pen | win | TTC-worse | d_ttc | d_gap | d_loss | abs TTC noop→pol | abs loss | abs gap |
|---|---|---|---|---|---|---|---|---|
| p10 | 17 | 23 | −0.70 | −1.53 | −1.36 | 14.9→14.2 | 23.4→22.1 | 6.9→5.4 |
| p20 | 17 | 23 | −0.52 | −2.90 | −3.79 | 13.2→12.6 | 21.9→18.1 | 6.0→3.1 |
| p30 | 13 | 18 | −1.12 | −0.35 | −3.33 | 13.5→12.4 | 19.8→16.5 | 4.1→3.7 |

Reference: p20 specialist (champion) = 27 dom / 24 win / 15 TTC-worse,
d_ttc −1.52.

## Conclusions

1. **One model generalizes across penetrations**: at every level it improves
   TTC, delay, and fairness vs that level's no-op. No level makes it break.
2. **Monotonicity holds for the priority metric**: absolute TTC drops
   monotonically with penetration (14.2 → 12.6 → 12.4) and delay too
   (22.1 → 18.1 → 16.5). This matches the goal "higher penetration = better."
   Fairness dips slightly at p30 (3.1 → 3.7).
3. **The generalist does not beat the p20 specialist**: at p20 the multi-pen
   model scores 17 win / 23 TTC-worse vs the champion's 24/15. The generalist
   trades peak performance for coverage — a principled generalist/specialist
   tradeoff.

## Caveat

This model includes the metering-load feature (127-dim). B (fresh p20 +
feature only) was neutral-to-harmful, so the pure multi-pen contribution is
not isolated here. A clean multi-pen without the feature would separate
multi-pen's effect from the feature's.

## Artifacts

- Training: `runs/rllib/shepherd_multi_pen_v1`
- Evals: `runs/eval/shepherd_multi_pen_p{10,20,30}_15_dt05_holdout101_150`
- Drivers: `_eval_shepherd_multi_pen_p{10,20,30}_15_dt05_holdout101_150.py`
