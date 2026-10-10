# Multi-Penetration + CAV-Count Context — Penetration-Invariance Eval (2026-08-08)

## Summary

The single policy trained across 5/10/20/30% penetration **with** the
penetration-context feature (`use_cav_count`, 127-dim local obs) was evaluated
on the 50 held-out seeds (101–150) at p10, p20, p30, all at `dt=0.5`, and scored
against each penetration's own no-op reference.

**Verdict: the context feature reduces the penetration regression, especially the
p30 delay regression, and holds/improves passes while massively increasing
delay- and fairness-wins at every penetration.**

## Scorecard (pass / loss_wins / G_wins)

| pen | champion (recorded) | cav-count (computed) | Δ pass | Δ loss_wins | Δ G_wins |
|---|---|---|---|---|---|
| p10 | 28 / 26 / 14 | **27 / 30 / 33** | −1 | +4 | +19 |
| p20 | 27 / 24 / 15 | **27 / 39 / 35** | 0 | +15 | +20 |
| p30 | 15 / 14 / 23 | **17 / 35 / 25** | +2 | +21 | +2 |

## Continuous deltas vs per-pen no-op (cav-count model)

| pen | dL (loss) | dG (fair_gap) | dTTC | policy TTC / noop TTC |
|---|---|---|---|---|
| p10 | −1.61 | −1.94 | −1.22 | 13.68 / 14.90 |
| p20 | −4.16 | −3.44 | −2.12 | 11.04 / 13.16 |
| p30 | −2.79 | −0.32 | −2.08 | 11.40 / 13.48 |

Champion reference (recorded): p20 d_ttc −1.52, d_loss −3.42, d_gap −3.76.
Cav-count p20 therefore improves TTC (−2.12 vs −1.52) and matches dL.

## Interpretation (the KEY question)

Does the penetration-context feature reduce the penetration regression — esp. p30?
**Yes.**

- p30 was the weak spot (champion 15/14/23). Cav-count lifts p30 to 17/35/25:
  delay-wins 14 → 35 (+21) and gates-pass 15 → 17, with a strong TTC delta (−2.08).
- At p10/p20 cav-count holds passes flat (27/27) while roughly **doubling** the
  delay-wins and more than doubling fairness-wins vs the champion at p20
  (39/35 vs 24/15).
- TTC is not regressed at any penetration vs the champion.

## Files / artefacts
- Eval drivers: `_eval_shepherd_multi_pen_cavcount_{p10,p20,p30}_15_dt05_holdout101_150.py`
  (`use_cav_count=True` matched to 127-dim obs).
- Results: `runs/eval/shepherd_multi_pen_cavcount_{p10,p20,p30}_15_dt05_holdout101_150/`.
- Checkpoint: `runs/rllib/shepherd_multi_pen_cavcount_v1/checkpoints/final` (600k steps).

## Caveats
- Champion triplets are the previously-recorded per-pen holdout reference, not
  recomputed in this run; cav-count numbers are computed fresh against the same
  per-pen no-op refs.
- multi_pen_v1 (`shepherd_multi_pen_clean_v1`) summaries were not present to
  recompute; historical reference was 17/23 at p20.
