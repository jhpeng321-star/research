# S-RB-UNC handoff

Branch `c4/rb-unc`, base `f0db3c82`. Owner WM-PHYS authorization read at
REGISTRATION.md:5683,5694; all four WHY reports and ENSPROBE read.

## Implementation (ready for integrator)
- NEW `wms.control.hidden_state` runtime adapter, opt-in
  `future_sampling='hidden_state'`; legacy futures remain default.
- Existing encoder exposes `uncertainty`, but `control/futures.py:94-99`
  documents that it has no training loss. It is NOT a calibrated posterior.
- Uses inferred population vectors as joint draws, history-conditioned motion
  likelihood for present drivers, independent future Poisson request streams;
  no generic acceleration/embedding/state/lane/merge noise.
- Integrator can call adapter directly; shared `wms.py/spec.py/factory.py`
  will not need edits. Adapter will require deterministic WM-PHYS physics.
- Entrants use persistent entrance/ordinal-keyed learned population draws,
  compatible with existing `entrant_driver_sampler` / `LearnedAdmission` hooks.
- Exact integration example and statistical caveats: `tools/rbunc/README.md`.
- API: `plan_hidden_state_futures(model, initial, history_kwargs,
  future_sampling='hidden_state', n_samples=8, seed=190, row_keys=world_keys,
  entrance_histories=histories, queues=queues)` -> list of `HiddenFuture`;
  each supplies `.kwargs` for rollout and `.receipt` for provenance. OFF -> None.
- Caller must pad first; freshly construct plans for alternatives/replay (mutable
  entries), paired rows have equal world keys. No need to edit shared model code.
- `sample_present=False` / `sample_arrivals=False` give exact source ablations.
- LC2013 defaults remain defaults: the old encoder's MOBIL outputs cannot
  identify LC2013 settings and are not reinterpreted as those parameters.

## Resources / checks
- Clean worktree at start; local NVIDIA driver unavailable, initial CPU load94%.
- Existing Python3.13 CPU runtime available per INT; use one CPU thread/low priority.
- No simulation, training, download, peer-process action or prohibited data opened.

## Tests
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 py -3.13 -B -m pytest -q -p no:cacheprovider
  tests/unit/test_hidden_state_futures.py tests/unit/test_c4_futures.py`:
  **20 passed / 21.42s**. One harmless pytest cache-plugin-disabled config warning.
- Covers exact replay of real IDM/MOBIL rollouts, varying outcomes from fixed
  driver draws; joint population prior for zero history, no global/embedding
  noise; paired-row CRN; admission/newborn identity equality; exact Poisson count
  mean/variance and zero demand; source ablations; legacy futures regression.

## Known gaps
- Recorded FIT source-attribution evaluation next.
- Existing untrained uncertainty must not be
  presented as posterior calibration. Across-world variability is descriptive,
  not conditional variability (one real continuation per FIT observation).
