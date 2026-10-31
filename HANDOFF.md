# S-RB-UNC handoff

Branch `c4/rb-unc`, base `f0db3c82`. Owner WM-PHYS authorization read at
REGISTRATION.md:5683,5694; all four WHY reports and ENSPROBE read.

## In progress
- Building NEW `wms.control.hidden_state` runtime adapter, opt-in
  `future_sampling='hidden_state'`; legacy futures remain default.
- Existing encoder exposes `uncertainty`, but `control/futures.py:94-99`
  documents that it has no training loss. It is NOT a calibrated posterior.
- Use inferred population vectors as joint draws, history-conditioned motion
  likelihood for present drivers, independent future Poisson request streams;
  no generic acceleration/embedding/state/lane/merge noise.
- Integrator can call adapter directly; shared `wms.py/spec.py/factory.py`
  will not need edits. Adapter will require deterministic WM-PHYS physics.
- Entrants will use persistent entrance/ordinal-keyed learned population draws,
  compatible with existing `entrant_driver_sampler` / `LearnedAdmission` hooks.

## Resources / checks
- Clean worktree at start; local NVIDIA driver unavailable, initial CPU load94%.
- Existing Python3.13 CPU runtime available per INT; use one CPU thread/low priority.
- No simulation, training, download, peer-process action or prohibited data opened.

## Known gaps
- Implementation and tests pending. Existing untrained uncertainty must not be
  presented as posterior calibration. Across-world variability is descriptive,
  not conditional variability (one real continuation per FIT observation).
