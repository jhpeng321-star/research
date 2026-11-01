# LPHYS engineering intake — snapshotf6cf52ab

Imported committed fd768d54/cef6321c/06eaf9b3/1c5722e9/f6cf52ab only. Explicit
default-OFF `physics_params='learned_v1'` and frozen parameter file add a
documented SUMO departLane/departSpeed mixture; no residual, IDM or LC refit.
Draws are keyed by future/world/entrance/request ordinal, persistent across
admission retries and coupled across actions.

Artifact `artifacts/rb2lphys/learned_v1.json` declares development-only fit and
selection worlds. Learned lane-first0/mainline-last0/ramp-last0.01107518819.
The later development conditional birth objective is slightly worse; it does not
justify adoption. The worker's autonomous dev12/K2 comparison is still running.
Its K2 canaries are retained as development evidence, never counted as K8 or
reused as integrated FULL-v5-LP forecasts with a different recipe.

Two explicit integrated engineering recipes now exist:

- `configs/wm_phys_v5.toml`: causal fitted LC + physical spread calibration,
  current-lane IDM limit and CAV admission/execution-setting consistency.
- `configs/wm_phys_v5_lp.toml`: identical plus only the declared LP departure
  distribution. Both retain all learned corrections OFF and authentic causal
  conditioning. They are not final frozen/teacher-ready recipes yet.

INT CPU command:

`python -B -m pytest -q tests/test_rb2lphys.py tests/test_rb2int_candidate_recipes.py tests/test_rb2int_lane_cal_composition.py tests/test_rb2break_execution_settings.py tests/test_rb2break_lane_limits.py tests/test_rb2_spread_calibration.py tests/test_lane_continuation.py tests/test_rbint_recipe.py tests/test_rbint_departure.py tests/unit/test_hidden_state_futures.py tests/unit/test_fitted_hidden.py tests/unit/test_rb_merge.py tests/unit/test_rb_merge_encoded_fleet.py tests/test_rb_cf_sumo_idm.py`

**135 PASS / 0 skips / 20.11s.** This covers all four worker source families,
queue/admission/CF/posterior composition, and the paired-recipe invariant.
HANDOFF-only conflict resolved preserving peer provenance. Final exposure audit,
real-runtime staged smoke, integrated development forecasts and final comparison
remain pending. No INT inference was launched during this intake.
