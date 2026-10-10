# DRAFT plan: G-2 relative-safety validation without replicates

Status: DRAFT (2026-09-29), docs only. It assumes a deterministic campaign
(H1; the seed test is pending).

## What G-2 validates

The planner uses Part A's **paired** safety deltas (plan − native) over the
commit window Δ = 30 s:

- ΔTTC<3 exposure;
- Δ hard-brake seconds;
- Δ min-TTC q05;
- Δ collision flag.

G-2 asks whether their sign and rank are right, and whether their error is
small enough for the relative-safety veto (`rel_safety_tol_s`,
`rel_safety_quantile` 0.90).

## Truth

At each restored branch point t0, every arm branch (pi, fixed1, fixed2,
switching1–3) is paired with the native branch of the same world. Under H1
each pair's metric difference over (t0, t0 + H] is EXACT.

- **Development:** chunks 1–3 TRAIN worlds. That is about 6 arms × about 140
  worlds per horizon H ∈ {30, 60}, and more if windows are also taken at
  later decision times inside the branch. Those later windows share the arm's
  history, so they are correlated; cluster by world.
- **Look:** one registered pass on fresh CAL worlds (chunk 3 or 4 CAL, which
  Part A has never seen). HOLDOUT is excluded.

## Predictions

Part A (the structural v2 once frozen, with v1 as the comparison) is rolled
with the action engine (idm_action + the A2 controller emulators + the
execution rule), for both arm and native, from the same state. That yields
predicted paired deltas with their M×K spread.

## Metrics (per metric and per H)

- sign agreement on |true Δ| > ε pairs;
- Spearman correlation over pairs;
- MAE against the zero-Δ baseline;
- coverage of the predicted 10–90% band on the true Δ;
- veto confusion: planner veto decision vs the veto that would be taken with
  the true Δ, giving false-veto and missed-veto rates.

Everything is reported per penetration and per arm.

## Pass criteria (proposal, to be frozen with a registration)

- sign agreement ≥ 0.8 on clearly nonzero pairs;
- MAE ≤ 0.7 × the zero-Δ MAE;
- 10–90% coverage ≥ 0.8;
- the false-veto rate reported and ≤ an owner-set bound.

## Prerequisites

1. Seed test confirms H1.
2. Part A v2 is frozen, including passing the oracle-vType exactness test.
3. The chunk-2 → RootArrays adapter from the reconstructor's RG-0 module.
4. A registration drafted and frozen before any CAL pass.
