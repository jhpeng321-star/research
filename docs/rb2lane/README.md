# S-RB2-LANE evidence index

## Decision

Source/action-cadence and causal-inference fixes are delivered. The exact
`lane_raw_c5602102891e` mixture is **not adopted** by this lane workstream:
conditional false positives fall, but autonomous lane counts are suppressed
too far on development data. Keep the distinction between engineering readiness
and model qualification. INT owns the integrated-source comparison and final gate.

## Versions and protocols

- `a29116e9` / `6b4bfbd5`: physical kernel, own-causal LC posterior, sampled-profile
  warm memory, opt-in recipe and complete runtime staging dependencies.
- `fit_v2/`: 80 conditional episodes, native FIT32 H60/300 and development12
  five arms H60. K8 persistent LC profiles; same pooled IDM for both variants.
  Per-motivation recall is contribution to ALL-event recall, not observed class recall.
- `runtime_v1_SMOKE.json`: strict-loaded checkpoint, four60s closed-cohort native
  futures, two FIT starts; source6b4bfbd5, finite/conserved/no learned corrections.
- `free_v1/`: full native developmental K2/H300 pilot, entrants ON, 48 futures,
  sourcea60a6a62. Both variants causally warm; same IDM/arrival streams; LC block
  only factor. Per10s segment/kind counts/exposures match the independent target
  convention. Native CAVs share the same physics law in both variants.
- `INITIAL_LANE_AUDIT.json` and `prefix_v1_results/`: four legitimate initial
  junction-internal rows must not have their initialization counted as LC. Twelve
  one-step source-bound replays identify those transitions exactly; scorer excludes
  them without changing forecasts. Prefix sourceb5e53b3e has identical physical
  code/recipe/conditioning to a60a6a62 (verified by git diff).
- `free_v1/BOUNDARY_TARGETS.json`: independent native300→300.5 boundary check;
  zero missing target events on these12worlds. No future observation entered any
  forecast. Other conditional cohorts' seven known boundary events are separate.
- `*_DISPATCH.json`, `*_COMPLETE.json`, `*_results/`: immutable ownership,
  runtime/archive/input hashes and raw results. Every owned worker exited.
- `CPU_FINAL.xml`: 85 core targeted tests. Additional guard/scorer/ramp-cadence
  regression tests5PASS; Ruff PASS. No GPU or SUMO execution, no neural training.

## Reproduction

Postprocess saved results only (choose a NEW output name):

```powershell
python -m tools.rblane.round2_finalize --out docs/rb2lane/fit_recheck
python -m tools.rblane.round2_free_finalize --out docs/rb2lane/free_recheck
python -m tools.rblane.round2_release --verify
```

The free-pilot postprocessor rechecks permitted FIT boundary views on cpuQ; model
inference is not repeated. Launch receipts reject duplicate immutable jobs.
No reserved969432–969439 or WMDATA2 fitting/selection occurred. Sparse cells and
K2 preclude a full-panel confidence/qualification claim here.
