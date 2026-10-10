# Medium-density coupling probe

Date: 2026-08-06
User steer: "lower the density a little to see if that would stop the
coupling effect." One step down from dense → **medium**, penetration kept at
p10 so density is the only variable. Same 50 held-out seeds (101–150),
dt=0.5, 1.5 s TTC ruler, paired per-seed no-op baselines built at medium
(`runs/eval/medium_noop_refs_101_150`).

## Headline

Lowering density does **not** break the TTC/flow coupling — it removes the
flow opportunity while the TTC cost stays. The coupling is bound to the
champion's speed-change actions, not to scenario density.

| | Dom | Win | TTC-worse | d_ttc | d_gap | d_loss |
|---|---|---|---|---|---|---|
| champion @ dense p10 (bar) | 28 | 26 | 14 | −2.14 | −2.92 | −2.82 |
| champion @ medium p10 | 5 | 6 | 23 | +0.32 | +0.09 | +0.21 |

Baselines (no-op, same seeds/ruler): medium TTC 10.5 / loss 11.2 / gap 7.6
vs dense 14.9 / 23.4 / 6.9. Medium halves delay; the fairness gap is
slightly *wider* (7.6 vs 6.9).

## Evidence it's not a coasting artifact

The near-zero means could be misread as "the policy did nothing." It
intervened plenty: 35/50 seeds have |d_ttc| ≥ 3 (range −12 … +10), 21/50
have |d_loss| ≥ 1.0 (range −5.6 … +3.8). The champion meters at medium —
the outcomes are just symmetric: TTC swings both ways, delay effects small,
net ≈ 0 on every leg.

## Interpretation

- **TTC cost per intervention is density-robust.** The same HDV–HDV wave
  mechanism fires at medium (+10 TTC on the worst seeds, same shape as
  dense).
- **What density controls is the flow opportunity.** At dense there is 23.4
  s of baseline delay to buy back; at medium only 11.2 s. The champion's
  metering buys ~nothing net (d_loss +0.21) because there is little to win.
- **Dense is this policy's best regime, not its worst.** Risk/benefit
  *worsens* as density drops: 14 TTC-worse at dense (with 26 wins) → 23
  TTC-worse at medium (with 6 wins). The dense congestion is exactly what
  makes the champion's flow gains large enough to (often) compensate the
  irreducible TTC cost.
- Consistent with campaigns 1–4 and the p20 line: the coupling follows the
  *policy's actions*. Scenario changes (penetration up, density down) move
  the baseline and the opportunity size; they do not decouple TTC from
  metering.

## Practical consequence

Do not present medium as "the regime where the policy is safe" — it is the
regime where the policy is *pointless* (5/50 domination). If a
medium-density variant is ever wanted, it needs its own training run with
recalibrated intervention thresholds (the champion's triggers are
dense-calibrated) — but the expected ceiling is low because the no-op is
already good at medium.

## Artifacts

- Eval: `runs/eval/champion_iter90_at_medium_15_dt05_holdout101_150`
- Baseline: `runs/eval/medium_noop_refs_101_150`
- Drivers: `_build_noop_refs_medium.py`,
  `_eval_champion_iter90_at_medium_15_dt05_holdout101_150.py`
