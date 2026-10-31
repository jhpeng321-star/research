# S-RB-FIT handoff

Branch `c4/rb-fit`, base `f0db3c82`; owner amendment REGISTRATION.md:5694.
Status: Gaussian IDM, finite-profile IDM and LC fits complete; final publication
and all-row pooled point fit in progress. Own fit/lane jobs have exited.

## Frozen data/estimation contract
- Only FIT worlds 969400–969431, native arm, from cpuQ
  `/home/ubuntu/wmdata20261031/views`; explicit allowlist of `episode.json`,
  `time_s.npy`, `veh_kin.npy`, `veh_mask.npy`, `is_cav.npy`, `vid.npy`.
- Fit motion in [240,300] s only; driver identity = world + observable numeric VID.
  Do not combine the five counterfactual arms as independent drivers.
- Per-driver posterior uses at most the causal 60 s before t=300, shorter when
  absent; unseen/absent drivers use the learned population distribution.
- IDM fits mainline lanes 0/1 only: ramp/junction virtual-stop constraints are
  not a same-lane leader observation. Ramp-only histories use population prior.
- Acceleration target is forward speed difference, not a hidden driver setting.
  Exclude junction/no-lane snapshots, ego lane transitions, changed/absent leaders,
  invalid identities, and negative physical bumper gaps; retain stopped traffic.
- Positive IDM parameters in log space: v0, tau, accel, decel, minGap; delta=4
  (SUMO default, fixed). Population log-normal prior + per-driver Laplace
  posterior; variance/noise learned from motion, not labels.
- Validation is later native FIT motion only, conditional on recorded leaders and
  lanes; this does not establish free-running breakdown/traffic fidelity.

## Outputs / integration
Artifacts exist in `artifacts/rbfit/{gaussian_v1,finite_mixture_v4,lane_v1}`.
Finite-mixture posterior is lossless `idm_posteriors.json.gz` (32.8 MB); original
104 MB JSON preserved locally/remote and explicitly ignored by its own directory.
Runtime `wms.physics.motion_fit.MotionFit` supports both encodings.
No shared model/spec/factory edits; no switches enabled by this workstream.
- `tools/rbfit/idm_fit.py`: allowlisted extractor, source-backed NumPy SUMO IDM
  substepping/free-speed envelope, Gaussian acceleration likelihood, diagonal
  log-normal population EM, full-covariance per-driver Laplace posteriors.
- cpuQ run root `/home/ubuntu/rbfit20261031`, four nice19 workers, one BLAS thread.
- Tests: cpuQ `python -m pytest tests/unit/test_rbfit.py -q -p no:cacheprovider`
  **15 passed** (including six independent shared SUMO full-step numeric vectors).
  Artifact integrity/schema/numerics verification PASS. Local default Python
  lacks SciPy; use cpuQ existing environment. No new environment installed.

## Resources and gaps
- cpuQ load 0.07 on 32 CPUs at first check; existing miniconda Python available.
  Local CPU busy; no local GPU driver. CPU fitting, no GPU allocation required.
- Shared setdata not ready; independent extractor used. Shared idmspec numeric
  vectors arrived and agree. `rb-shared/fit/REFERENCE_COMPATIBILITY.md` flags
  CF source-default mismatches: NUMERICAL_EPS=0.001 and emergencyDecel=9.
- Gaussian posterior improves 10s position MAE but worsens RMSE tails.
  Finite-profile posterior 95% physical-acceleration coverage is only 68.8%; no
  posterior uncertainty qualification. Population coverage 95.0%, diagnostic only.
- LC later-FIT Brier 0.00306 loses to event-rate baseline 0.00241: not adopted.
- Student-t variants v2/v3 scale-collapsed; stopped verified OWN PIDs/children,
  STOP receipts and original partial logs preserved. Neither adopted.
- No new SUMO simulation or neural-network training.
