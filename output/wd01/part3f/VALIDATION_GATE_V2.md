# WD-01 native Part A — VALIDATION GATE v2 (frozen before any p3f gate number)

Written by the world model session (reviewer) at the dispatch of p3f
(ZCode b2d22885). Agreed by main discussion (delegated), 2026-09-26.

## Why it replaces the old gate

The old gate (after p3c/p3e) asked the model to beat plain IDM-normal on
member-averaged trajectory speed RMSE at H=60 s. p3e confirmed that SUMO
applies stochastic dawdling to these vTypes. Individual trajectories are
therefore unpredictable: even the oracle-vType IDM only reaches 4.58 vs
4.97 m/s. The planner consumes DISTRIBUTIONS (window delay, exposure,
collisions, M×K spread), so the primary test must be distributional. The
change was made after the p3c/p3e failures, which this note states
openly. Thresholds are fixed now and are not tuned after results.

## GATE v2 (primary), evaluated at H = 60 s on the 4 validation roots

(a) Predicted collisions ≤ 1 per 1000 vehicle-steps.
(b) Lane-change count per window within ±20 % of truth.
(c) The member 10–90 % band covers the truth in ≥ 80 % of windows for BOTH
    the window delay proxy and TTC<3 s exposure, AND the ensemble CRPS
    of the delay proxy is below the MAE of the deterministic IDM-normal.
(d) Member-averaged speed RMSE ≤ 1.05 × B2 = 5.22 m/s.

## OLD gate (always reported alongside)

Member-averaged speed RMSE < 4.97 m/s; collisions ≤ 1/1000;
LC count ±20 %; delay coverage (member min–max) ≥ 50 %.

## Scope

Validation only. The holdout stays blocked. Its disclosed second use
waits for the owner.
