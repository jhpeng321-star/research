# S-RB-FIT handoff

Branch `c4/rb-fit`, base `f0db3c82`; owner amendment REGISTRATION.md:5701.
Status: RESUMED for owner00:40 addendum; prior outputs/commits preserved.
Pooled IDM, population/driver distributions and first LC parameters published.
Fresh cpuQ00:43 audit: all five receipted fit/lane PIDs absent; load8.55/32.
Remaining: UNC sample adapter, motion-only posterior shrinkage calibration,
LC refit against committed lane811cd7e8 (finite geometry/epsilon fixes), and
bounded SUMOCORE sign-reversal diagnosis. Report: `docs/paper_results/RBFIT.md`.

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
- Positive IDM parameters: v0, tau, accel, decel, minGap; delta=4 fixed. Gaussian
  empirical Bayes and finite-profile mixture ML; all fitted from motion.
- Validation is later native FIT motion only, conditional on recorded leaders and
  lanes; this does not establish free-running breakdown/traffic fidelity.

## Outputs / integration
- Point candidate: `artifacts/rbfit/pooled_full_v5/idm_pooled.json`, schema
  `wm_phys.idm_pooled.v1`; pass its `parameters` to WM-PHYS driver settings.
  v0/tau/a/b/s0 = 29.839994/0.770429/0.795080/1.772102/1.933667, delta=4.
- Population/posterior: `artifacts/rbfit/finite_mixture_v4`, schemas
  `wm_phys.idm_population.v1` / `wm_phys.idm_posteriors.v1`; order
  `[v0,time_headway,accel,decel,min_gap]`, units m/s,s,m/s²,m/s²,m.
  `support` + `weights` are joint physical profiles; per-driver sparse
  `support_indices` + `weights` select that same support, keys `world:VID`.
  `log_mean`/`log_cov` are summaries, NOT a new log-normal draw distribution.
  Missing driver weights -> population; t0 must equal `conditioning_end_s=300`.
- Posterior lossless `idm_posteriors.json.gz` (32.8 MB), original JSON preserved
  locally/remote; `MotionFit(directory).idm(world, present_vids, t0_s=300,
  sample=True, use_posterior=False, generator=g)` reads gzip or JSON. Draw ONCE,
  cache by observed identity for the entire future; entrants use population.
  No automatic integration switch; shared model/spec/factory untouched.
- v0 IS effective desired speed; speedFactor=v0/actual lane limit. No double scale.
- LC: `artifacts/rbfit/lane_v1/lc2013_{population,posteriors}.json`, schemas
  `wm_phys.lc2013_{population,posteriors}.v1`; joint support/weights, five physical
  keys. `draw_lane_parameters` draws one persistent vector. Emission probabilities
  are observation discrepancy, NEVER runtime event noise. LC not adopted.
- `tools/rbfit/README.md` contains reproduction commands and format details.
- cpuQ run root `/home/ubuntu/rbfit20261031`, four nice19 workers, one BLAS thread.
- Tests: cpuQ `python -m pytest tests/unit/test_rbfit.py -q -p no:cacheprovider`
  **15 passed** (including six independent shared SUMO full-step numeric vectors).
  Artifact integrity/schema/numerics verification PASS. Local default Python
  lacks SciPy; use cpuQ existing environment. No new environment installed.

## Resources and gaps
- CPU only; no GPU allocation, no new environment. Initial cpuQ load0.07/32 CPUs.
- Shared setdata not ready; independent extractor used. Shared idmspec numeric
  vectors arrived and agree. `rb-shared/fit/REFERENCE_COMPATIBILITY.md` flags
  CF source-default mismatches: NUMERICAL_EPS=0.001 and emergencyDecel=9.
- Gaussian posterior improves 10s position MAE but worsens RMSE tails.
  Finite-profile posterior 95% physical-acceleration coverage is only 68.8%; no
  posterior uncertainty qualification. Population coverage 95.0%, diagnostic only.
- LC later-FIT Brier 0.00306 loses to event-rate baseline 0.00241: not adopted.
- Point candidate acceleration RMSE 0.299 vs initial0.902 m/s², 10s position
  MAE5.58 m vs constant-speed13.81 m. Conditional FIT only; INT/EVAL must score
  free-running congestion/breakdown before adoption. Mainline-only CF fit and
  missing junction/LC constraints limit physical parameter identification.
- Student-t variants v2/v3 scale-collapsed; stopped verified OWN PIDs/children,
  STOP receipts and original partial logs preserved. Neither adopted.
- No new SUMO simulation or neural-network training.
