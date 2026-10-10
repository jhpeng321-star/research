# Morning summary — 2026-10-05 (WM-N1 overnight: A8 -> A11)

## Models (all member 1, seed 101, 15x2000 steps, gated by a 400-step GPU smoke)
| model | what changed | checkpoint |
|---|---|---|
| A8 | label coverage (1,308 pairs), climatology-anchored paired head (state only), pair-dense K=4, loss scale | 8b023883 |
| A9 | + plan-aware head (10 s CAV plan) + 644 direct-speed runs (1,952 pairs) | 9c666a7c |
| A10 | + 263 re-aligned damping runs + supervised-reference fix (2,113 pairs) | 284b1b35 |
| A11 | + 120 s plan input + within-decision ranking loss | 41221852 |
Previous model M1FIX1 (94984cc0) lost to climatology at every tau by up to 10x.

## 1. All-controller evaluation slice (CAL-tune, sealed; 11 worlds x 5 taus; 8-9 candidate controllers per decision: native, PI, 4 direct-speed plans, fixed mix, switching mix, walls)
Decision regret (s of total delay vs the best controller per decision; lower = better). Always-native is the TRAIN fixed-best.
| tau | A11 | A10 | A9 | A8 | always-native | random |
|---|---|---|---|---|---|---|
| 30 | 0.5 | 2.1 | 1.0 | 102.0 | 0.5 | 136.5 |
| 60 | 3.3 | 13.6 | 2.1 | 148.0 | 3.3 | 367.1 |
| 120 | 105.1 | 179.8 | 104.8 | 105.1 | 105.1 | 949.5 |
| 300 | **972.3** | 1555.8 | 1618.0 | 1555.8 | 1555.8 | 2798.3 |
| 570 | **5408.9** | 5509.1 | 4845.7 | 6504.8 | 6504.8 | 7716.0 |
- **A11 never does worse than always-native and beats it at 300 s: 972 vs 1556 s, 90% CI of the difference [-1339, -3] (excludes 0)**, and at 570 s (5409 vs 6505, CI [-3288, 0]). It picks native at 30-120 s (correct: most direct-speed plans hurt there).
- First decision-quality win over the fixed-best baseline with a CI excluding 0. Caveats: 11 worlds per tau; 20 model x tau cells were tested, so one significant cell needs replication (more evaluation worlds) before it is a claim.

## 2. PI-vs-native only (CAL-tune, 11 decisions; like-for-like with earlier scores)
CRPS (lower better) and decision regret:
| tau | CRPS A11 / A10 / A9 / A8 / clim | regret A11 / A10 / A9 / native |
|---|---|---|
| 30 | 11.0 / 7.6 / 8.9 / 6.6 / 5.3 | 0.5 / 2.1 / 1.0 / 0.5 |
| 60 | 37.3 / 29.5 / 32.4 / 25.5 / 25.0 | 1.9 / 12.3 / 0.7 / 1.9 |
| 120 | 135.3 / 145.4 / 142.3 / 127.5 / 129.1 | 6.0 / 80.7 / 5.7 / 6.0 |
| 300 | **431.7** / 542.9 / 543.7 / 518.4 / 528.2 | **17.6** / 20.9 / 83.1 / 20.9 |
| 570 | 1567.4 / 1594.4 / 1616.1 / 1551.5 / 1563.6 | 257.5 / 257.5 / 394.8 / 257.5 |
- Accuracy: all models tie climatology at 60-570 s; A11 is best at 300 s (point 432 vs 528, CI [-266, 52]) but worst at 30 s (significantly worse than climatology). Calibration (10-90 coverage) 0.82-1.00 for A10/A11.

## 3. Noise floor (90 runs, analysis complete)
PI's PI-native spread is 5-27x the spread produced by a tiny one-CAV nudge at every horizon (all CIs > 1): long-horizon effects are real action effects, not amplified noise. Seed change after a snapshot restore changes nothing (RNG restored). Simulator reproduces native runs bit-exactly.

## 4. Other
- Direct-speed campaign: 644 TRAIN runs + 44 sealed CAL-tune runs, 0 failures, 0 collisions in smoke tests.
- Damping-only runs re-aligned (263) and in training from A10.
- Disk: E: ~77 GB free; D: 22.5 GB sits in the Recycle Bin (owner's rar files).
- gpuA can be powered off (all checkpoints copied and verified).

## Recommended next steps
1. Replicate the A11 300 s win with more evaluation decisions (more CAL-tune worlds / direct-speed plans per world; registered).
2. Closed-loop test: planner re-picks the controller every 60 s with A11 vs always-native / always-PI in SUMO (the result the headline needs).
3. Fix the 30 s accuracy regression (A11 worst at 30 s) — e.g. weight short-horizon cells or train a short-horizon head.
4. Add fixed/switching/trajectory contrasts to the slice CRPS table (scorer gap; they currently enter decision quality only).
Reports: runs/active/WM-N1-SCORE-A{8,9,10,11}-*/REPORT.md (A11: this file), runs/active/WM-N1-EVALSLICE-20261004/REPORT.md, runs/active/WM-N1-NOISEFLOOR-20261004/RESULT.md.
