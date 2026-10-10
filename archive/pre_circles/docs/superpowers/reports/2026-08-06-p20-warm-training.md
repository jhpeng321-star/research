# p20 warm-training (Lever A): verdict

Date: 2026-08-06
User steer (Tier 1): recalibrate the champion at p20 to improve the strongest
config. Warm-trained champion iter_90 (534k) at p20, lr 3e-5 (non-drifting
recipe), +150k target. Training froze at iter_106 (Ray object-store choke,
2nd p20 recurrence); harvested iter_100 (best-balanced smoke corner). Full-50
scored vs the p20 control bar.

## Result (full 50 seeds, dt=0.5, 1.5 s ruler, p20 no-op baseline)

| Config | Dom | Win | TTC-worse | d_ttc | d_gap | d_loss |
|---|---|---|---|---|---|---|
| champion @ p10 | 28 | 26 | 14 | −2.14 | −2.92 | −2.82 |
| champion @ p20 (control) | **27** | **24** | **15** | **−1.52** | **−3.76** | −3.42 |
| p20-warm iter_100 | 23 | 23 | 19 | −0.96 | −2.85 | **−3.95** |
| champion @ p30 | 15 | 14 | 23 | −0.80 | −1.31 | −2.68 |

## Reading

The p20 recalibration **improves only delay** (d_loss −3.95 vs −3.42) and is
worse on everything TTC-first: TTC-worse 19 vs 15, win 23 vs 24, dom 23 vs
27, d_ttc −0.96 vs −1.52, d_gap −2.85 vs −3.76. The 12-seed smoke had shown
a much better iter_100 (gap −5.43); the full-50 shows −2.85. This is the
**third recurrence of the 101–112 subset overstating** a candidate (after
p20-fresh and the p20 smokes) — full-50 is the only valid verdict.

## Conclusion

Warm recalibration at p20 does not beat simply deploying the p10 champion at
p20. Combined with the p30 result (inverted-U peaking at p20), **Tier 1 is
exhausted**: no penetration level and no p20-calibration improves on the
champion@p20 control (27/24/15). The plain champion@p20 deployment remains
the checkpoint of record — the same scorecard as p10 with 25–30% better flow
means against a friendlier baseline.

## The one consistent, citable finding across all levers

Every policy-side lever (reward, action space, density, calibration,
execution mechanics, mechanism, and now p20 recalibration) slides along the
TTC/flow tradeoff or makes it worse. The only lever that moves the *scenario*
(penetration, peaking at p20) improves it. The coupling is intrinsic to
metering at this merge; the headline result is **champion @ p20** (27/24/15).

## Artifacts

- Warm training: `runs/rllib/shepherd_p20_warm_v1` (iter_95/100/105 + best)
- Smokes: `runs/eval/shepherd_p20_warm_v1_iter{95,100,105}_15_dt05_smoke12`
- Full-50: `runs/eval/shepherd_p20_warm_v1_iter100_15_dt05_holdout101_150`
- Drivers: `_eval_shepherd_p20_warm_v1_*.py`
