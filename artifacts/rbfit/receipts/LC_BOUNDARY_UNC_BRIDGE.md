# LC boundary sampler bridge verification — 2026-11-01 ~02:15 +08

- New own opt-in helper:src/wms/physics/lc_fit_for_unc.py,
  `load_boundary_fitted_state(idm_directory,lane_directory=...)`.
- Inherits the actual committed UNC FittedHiddenState; explicit LC-only domain
  extension. No peer/source/shared-model/default-reader edits.
- Tested in existing cpuQ environment with previously frozen committed UNC code,
  not a mock.5tests PASS in0.98s.
- Source-zero behavioral settings sampled exactly; sparse observed driver support
  indices, unknown-ID and future-entrant population draws/replay supported.
- Invalid IDM zero, LC assertive zero, cooperative>1 correctly rejected.
- Actual plan_hidden_state_futures:4futures×2replays,8planned futures total;
  zero LC coordinates preserved, CAV/padding untouched, global RNG unchanged.
- This closes the boundary-artifact reader compatibility gap. It does not adopt
  a candidate or demonstrate free-running traffic/action-effect fidelity.
