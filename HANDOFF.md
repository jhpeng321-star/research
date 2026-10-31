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
- **SUMO insertion bridge added:** `PoissonEntries.insert` detects the per-future
  `cfg.sumo_insertion` installed by MERGE inside rollout and binds its `candidate`
  to the SAME weighted `BirthMix.candidate` used for newborn motion. Without this,
  SumoInsertion's legacy uniform `errsrc-mix` hash would test a different vector.
  This changes only the future-owned latent lookup, not placement or shared code.
  No patch to MERGE's files is required. Rejects a mismatched sampler object.

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
- Follow-up: **23 passed / 36.77s**, adding actual admission+newborn-motion
  replay, supplied posterior samples without arbitrary caps, and missing donor
  rejection. Physical-mode candidate support now uses the encoder's actual bounds.
- Integration-bridge follow-up: **11 new-suite tests passed / 18.23s**, including
  post-plan SUMO insertion configuration and train-mode/future-noise rejection.
  Earlier 13 legacy-futures regression tests remain passing; no shared code changed.

## FIT diagnostic running
- `docs/paper_results/RBUNC_DESIGN.md`: all32 native FIT969400..969431, K8,
  H120 deterministic legacy IDM/MOBIL attribution reference; additional three
  source ablations on first world969400. This is not integrated WM-PHYS efficacy.
- Uses same available frozen encoder as SUMOCORE (`5b256941...`), authentic60s
  motion; only pre-t0 request-demand/queue/CAV share, no future arrivals supplied.
- Own cpuQ canary211301/supervisor215439, max8 single-thread nice19 workers,
  queued-work resource gate. `/home/ubuntu/rbunc20261031/fit_probe_v1/`.
- Exact Poisson request-count spread additionally saved at120/300/600s.
- Initial remote load2.07/32 CPUs, panel check3.89; no GPU used.

## Known gaps
- Recorded FIT source-attribution outputs being collected; no distributional
  qualification or integrated WM-PHYS efficacy claim from this reference probe.
- Existing untrained uncertainty must not be
  presented as posterior calibration. Across-world variability is descriptive,
  not conditional variability (one real continuation per FIT observation).
