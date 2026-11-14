# Medium warm-training: does training at the density fix the coupling?

Date: 2026-08-06
User steer: "train my current model on medium to see if it would do better.
not just eval on medium." Follows the medium transfer probe (5 dom / 6 win /
23 TTC-worse — the dense-calibrated champion barely engages at medium).

## Setup

Warm-start of the champion (iter_90, 534k steps) at medium density:
`--density medium --paired-refs paired_noop_refs_medium.json (new, 200
seeds) --lr 3e-5 --no-corridor-ttc-penalty --total-timesteps 690000`
(+156k steps), 8 runners, full champion stack (soft mask + sign gate live
in env defaults). The trainer process died mid-run at iter_105 (630k) with
no visible crash; checkpoints iter_95/100/105 + `best` were retained and
used (no relaunch — enough adaptation signal).

## Results (full 50 seeds, dt=0.5, 1.5 s ruler, medium no-op baseline)

| Config | Dom | Win | TTC-worse | d_ttc | d_gap | d_loss |
|---|---|---|---|---|---|---|
| champion @ medium (transfer) | 5 | 6 | 23 | +0.32 | +0.09 | +0.21 |
| **medium warm iter_100 (+66k)** | **9** | **7** | **19** | **−0.18** | +0.47 | −0.17 |
| champion @ dense (bar of record) | 28 | 26 | 14 | −2.14 | −2.92 | −2.82 |

Smoke trajectory (12 seeds): iter_95 (+36k) d_ttc +2.00 (4/6) → iter_100
(+66k) −1.08 (7/4) → iter_105 (+96k) −0.50 (5/4). The adaptation is real
and non-monotonic; iter_100 is the best smoke point, and its full-50
confirms the direction.

## Interpretation

- **The recalibration hypothesis is confirmed.** Warm training at medium
  moves the policy off its dense-calibrated operating point: TTC-worse
  23 → 19, d_ttc +0.32 → −0.18, domination 5 → 9. Training at the scenario
  beats transferring to it, as the user suspected.
- **But the coupling is not calibration-bound either.** Even a
  medium-adapted policy leaves **19/50 seeds TTC-worse** and buys
  essentially zero net flow (d_gap +0.47, d_loss −0.17). The wave cost per
  intervention is reduced, not removed, and the flow opportunity at medium
  is too small to compensate it. Medium remains the weakest scenario for
  this policy stack; dense (28/26/14) and p20-deployed (27/24/15) remain
  the strongest.
- This is the third independent line (dense levers, p20 training, medium
  training) all landing on the same wall: the TTC cost lives in the
  speed-change mechanism itself, not in calibration, density, or
  penetration.

## Artifacts

- Warm training: `runs/rllib/shepherd_medium_warm_v1` (iter_95/100/105 + best)
- Medium paired refs: `runs/eval/paired_noop_refs_medium.json`
  (builder `_build_noop_baseline_refs_medium.py`)
- Evals: `runs/eval/shepherd_medium_warm_v1_iter100_15_dt05_holdout101_150`
  (full-50), `..._iter95_15_dt05_smoke12` (contains iter_95/100/105 rows
  by TAG — driver-generation incident, see progress log)
- Drivers: `_eval_shepherd_medium_warm_v1_iter{95,100,105}_smoke12.py`,
  `_eval_shepherd_medium_warm_v1_iter100_15_dt05_holdout101_150.py`

## Recommendation

Do not retrain at medium further — the ceiling is bounded (the no-op at
medium is already good, so a perfect policy has little to add, and the
residual TTC harm is mechanism-bound). The strongest result on the table
remains: **champion deployed at p20** (27/24/15 with +25-30% flow means).
The density axis is signed closed: dense is the policy's best regime, and
neither lowering density nor training at the lower density breaks the
coupling.
