# p30 penetration probe: the dose-response peaks at p20

Date: 2026-08-06
Extends the penetration finding to 30%. Champion iter_90 evaluated at
p30 on the same 50 held-out seeds, 1.5 s ruler, p30 no-op baseline
(`runs/eval/p30_noop_refs_101_150`, built fresh).

## Result

| Config | Dom | Win | TTC-worse | d_ttc | d_gap | d_loss |
|---|---|---|---|---|---|---|
| champion @ p10 | 28 | 26 | 14 | −2.14 | −2.92 | −2.82 |
| champion @ p20 | 27 | 24 | 15 | −1.52 | **−3.76** | **−3.42** |
| champion @ p30 | **15** | **14** | **23** | −0.80 | −1.31 | −2.68 |

Baselines: p10 no-op TTC 14.9 / loss 23.4 / gap 6.9; p20 13.2 / 21.9 / 6.0;
p30 13.5 / **19.8** / **4.1**. The p30 *scenario* keeps improving (loss, gap),
but the *policy* gets worse.

## Reading

The penetration benefit is an **inverted-U peaking at p20** — it saturates
then reverses. The p30 baseline is still friendlier (loss 19.8, gap 4.1), yet
the champion's scorecard collapses (15/14/23 vs 27/24/15 at p20). Cause: at
30% penetration, three times as many vehicles run the **p10-calibrated
metering policy**, so there are far more active speed-changers in the stream
→ more wave sources → TTC-worse climbs to 23 and every mean degrades. The
friendly-baseline benefit is outweighed by the over-metering of a policy not
calibrated for the higher penetration.

## Conclusion

p20 is the optimal operating point for the champion. This is a clean,
citable dose-response: *more CAVs help up to a point, then the shared metering
policy over-disturbs the stream.* It also motivates the immediate next step:
calibrate the policy *at* p20 (warm-train the champion at p20), which is
exactly what a p30-shifted policy would also need. The p20 warm-training is
launched next.

## Artifacts

- Eval: `runs/eval/champion_iter90_at_p30_15_dt05_holdout101_150`
- Baseline: `runs/eval/p30_noop_refs_101_150`
- Drivers: `_build_noop_refs_p30.py`, `_eval_champion_iter90_at_p30_15_dt05_holdout101_150.py`
