# Native Part A v1 (working model): status and freeze record

Frozen 2026-09-27 on main discussion's approval. **This is NOT a passing model.**

## Label (use verbatim wherever the model is referenced)

> native Part A v1 (working model). Gate v2 NOT met (VAL look #8: physics-delay
> coverage 0.769 < 0.80 at H=30). TTC<3 forecasts UNVALIDATED (G-2; no better
> than climatology). Per-type lane changes inaccurate (merges +57%, discretionary
> changes have no density dependence).

## What it is

- Engine kind `idm_merge` (output/wd01/part3/engine.py), made of:
  - SUMO IDM plus fitted one-sided dawdle;
  - prior-sampled vType;
  - the lane-change hazard with per-type levers;
  - SUMO-like merge dynamics: lane-end obstacle, gap acceptance, end-force.
- Frozen parameters: output/wd01/part3o/results/candidate_p3o_iter2.json,
  sha256 458890d9c4b0ee6ee03de145be50498d9bd0873505a4e86f2bf8a1e224ebe45d.
- Frozen spread factors: output/wd01/part3o/results/calibration_p3o.json,
  sha256 c7ffa7540768fa67e75e99881eec6f801201990dea4ace90af93c202f789c22a.
- Both hashes are over the LF-normalised text; both frozen 2026-09-27T05:09:19Z.

## What it is good for

- The Part A delay term D_A over the commit window.
  - Physics-delay CRPS at H=30 is 37 vs 223 for climatology and 293 for deterministic IDM (VAL #8).
  - Mean bias is near zero at H=30.
- The end-of-window state s' that Part B consumes.

## What it must NOT be used for

- Absolute TTC<3 or hard-braking forecasts as safety evidence (G-2).
- Resolving the effect of individual per-CAV actions over 30 s (A2 step 2: the paired deltas are below single-realisation noise).
- Lane-change counts by type.

## Evidence trail

p3c–p3o, p3q, val8, A2 steps 1–2, deviation records DEV-1 and DEV-2 (8 validation looks).
