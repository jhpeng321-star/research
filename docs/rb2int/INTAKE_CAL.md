# CAL engineering intake (candidate, not final freeze)

Source snapshot `7c1cf53c`: design `c20b82ac`, fitted physical latent calibration
`0711a026`, source-bound decision API/audit `0782c9bf`, bounded continuation
receipts `7c1cf53c`. Uncommitted worker edits were not imported. The peer's own
live canary/collector and source identity remain unchanged.

- Physical artifact `docs/paper_results/rb2cal/v1/fit/calibration.json` declares
  FIT969400–969431 only, t0=300, learned corrections false, source22cc32f6.
  Parameters: posterior retention1, arrival variance1.667235195661008, entrant
  Dirichlet concentration0 (causal soft driver counts). Fixed population/support,
  ordinal/VID pairing and one future-owned mixture/rate; no per-tick noise.
- Native32 motion/count fit; physical outcome diagnosis restricted to dev12.
  Crossfit count likelihood improvement is weak; count CRPS worsens. No traffic
  coverage or controller-selection qualification is claimed.
- Source-bound FULL-v4 probability maps are preserved as evidence/API only;
  integrated v5 needs its own development forecasts and source-bound fit. The
  diagnostic output-scale experiments are not deployed physical corrections.
- Manual composition review caught a defect missed by textual merge: LC attach
  reconstructs FittedHiddenState and drops CAL's wrapper. INT now wraps the new
  joint base again, preserving entrant-mixture behavior/retention. Stored raw
  posteriors are not re-tempered; the wrapper is applied once on drawing.

INT verification:

`python -B -m pytest -q tests/test_rb2int_lane_cal_composition.py tests/test_rb2_spread_calibration.py tests/test_lane_continuation.py tests/test_rbint_recipe.py tests/test_rbint_motion_posterior.py tests/unit/test_hidden_state_futures.py tests/unit/test_fitted_hidden.py`

**69 PASS / 0 skips / 27.44s.** Regression explicitly tests causal LC attachment,
retained CalibratedHidden class, normalized entrant weights and LC-profile draws.
Only HANDOFF.md conflicted; both peers' provenance/status snapshots preserved.

Full model freeze/selection awaits BREAK/LPHYS and development evidence. No final
world outcome inspected and no new INT forecast dispatched in this step.
