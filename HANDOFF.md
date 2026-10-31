# S-RB-FIT handoff

Branch `c4/rb-fit`, base `f0db3c82`; owner amendment REGISTRATION.md:5701.
Status: owner01:28 continuation ACTIVE; prior outputs/commits preserved.
Fresh01:30 resume at876b2433: both owned jobs exited; LANE source still811cd7e8.
Next: direction-aware LC decision likelihood; see receipts/RESUME_0128.md.
`tools/rbfit/lane_directional.py` adds source-lane categorical decision likelihood
and prefix-calibrated informative-evidence shrinkage;5new CPU tests PASS.
Smoke969400 PASS (9529fit/9779later intervals); no LC adoption change.
Full32-world `lane_directional_v3` running cpuQ557463 (01:37),2nice19CPU workers,
frozen estimator9f574a5c / LC811cd7e8; create-once receipts preserved.
New outputs `artifacts/rbfit/shrinkage_v1` and `lane_v2`; **22 CPU tests PASS**.
Direct UNC f8b6faea reader verified on ALL3,130 exported drivers and16
paired/replayed planned futures. `receipts/ADDENDUM_VERIFIED_v2.json` binds SHA;
generic causal117-interval motion posterior matches saved weights within5.6e-17.
cpuQ407617(LC)/426309(shrinkage) both exited; no outstanding owned jobs.
No action outcomes selected fits. Reports: `docs/paper_results/RBFIT.md` and
`docs/paper_results/RBFIT_SUMOCORE_DIAG.md`. No demonstrated600s sign repair.
Coordination receipt: `E:/agent-work/rb-shared/fit/ADDENDUM_0040.md` and
`ARTIFACTS_0040.json`; artifact commit4965c665, generic APIe60634b9.

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
- Preferred posterior candidate: `artifacts/rbfit/shrinkage_v1`, schemas
  `wm_phys.idm_population.v1` / `wm_phys.idm_posteriors.v1`; order
  `[v0,time_headway,accel,decel,min_gap]`, units m/s,s,m/s²,m/s²,m.
  `support` + `weights` are joint physical profiles; per-driver sparse
  `support_indices` + `weights` select that same support, keys `world:VID`.
  `log_mean`/`log_cov` are summaries, NOT a new log-normal draw distribution.
  Unknown identity -> population; t0 must equal `conditioning_end_s=300`.
- NEW shrinkage policy: n/(n+19.35568) posterior + remainder population,
  strength learned prefix240..270→270..300 CRPS. All3,130 records have explicit
  joint weights;174 zero-evidence records equal population. Policy ALREADY
  APPLIED in the gzip JSON; do not shrink again. Original raw finite_mixture_v4
  and Gaussian fits preserved. Population joint support unchanged.
- UNC saved-ID path: `FittedHiddenState.from_directory('.../shrinkage_v1')`, pass
  `fitted_state=reader,vehicle_ids=verified_tape_vids,t0_s=300,row_keys=worlds` to
  `plan_hidden_state_futures`. It uses fitted population for entrant settings
  too. Leave `lane_directory` unset: fitted LC still fails its baseline.
- `MotionFit(directory).idm(world,present_vids,t0_s=300,sample=True,generator=g)`
  reads gzip/JSON. Draw ONCE and cache by identity across future/chunks;
  entrants use population. Shared model/spec/factory untouched; no new switch.
- v0 IS effective desired speed; speedFactor=v0/actual lane limit. No double scale.
- Identity: never equate evaluator packed slots with tape numeric VIDs. NEW
  `MotionFit.condition_motion` infers from caller-supplied causal60s motion and
  frozen support, without saved-ID lookup. See tools README; missing motion ->
  explicit population fallback. No verified SUMO-string/tape-VID crosswalk.
  Worlds969432..439 have no saved posterior; do not claim one from slot indices.
- UNC: wrap the original model with `MotionFitForUNC(model, fit, worlds, ids,
  t0_s=300,n_particles=32,seed=...)` for `plan_hidden_state_futures` ONLY; use
  original model for rollout. Proxy supplies `encoded['posterior_samples']`
  [P,B,K,9] in exact encoder order. Five IDM keys replace present HDVs only;
  CAV/padding, four unrelated coordinates, embeddings and model stay intact.
  World+VID keyed samples are slot-order invariant and common across actions.
  Explicit `MotionFit(...,shrinkage_path=...)` enables the learned JSON policy;
  policy/population SHA and conditioning time must match. Default unchanged.
- LC: `artifacts/rbfit/lane_v2/lc2013_{population,posteriors}.json`, schemas
  `wm_phys.lc2013_{population,posteriors}.v1`; joint support/weights, five physical
  keys. `draw_lane_parameters` draws one persistent vector. Emission probabilities
  are observation discrepancy, NEVER runtime event noise. LC not adopted.
- `tools/rbfit/README.md` contains reproduction commands and format details.
- Old cpuQ root `/home/ubuntu/rbfit20261031`; addendum root
  `/home/ubuntu/rbfit20261101_addendum`,2LC+1calibration nice19/single-thread.
- Tests:18 FIT +4 committed UNC tests PASS; independent SUMO vectors, CRPS
  pairwise oracle, no-evidence/identity/CAV invariants, and actual exported
  posterior algebra/schema/PSD/SHA/UNC replay PASS. Local Python lacks SciPy;
  cpuQ existing environment used. No new environment installed.

## Resources and gaps
- CPU only; no GPU allocation, no new environment. Initial cpuQ load0.07/32 CPUs.
- Setdata arrived after original extraction; independent helper's pooled
  v0=29.87/tau=.753 corroborates our29.84/.770 (different filtering/optimizer).
  Shared idmspec vectors agree. `rb-shared/fit/REFERENCE_COMPATIBILITY.md` flags
  CF source-default mismatches: NUMERICAL_EPS=0.001 and emergencyDecel=9.
- Gaussian posterior improves 10s position MAE but worsens RMSE tails.
  Raw finite posterior nominal95% coverage68.8%; shrinkage90.48%, population95.0%.
  Shrunk/raw/population CRPS .14657/.16351/.16560; RMSE .3478/.3778/.3567m/s².
  Internal empirical Bayes calibration; still NO conditional coverage qualification.
- Updated LC811cd7e8 finite geometry/epsilon, pooled_full_v5: later-FIT Brier
  posterior.002828/population.003071 vs baseline.002414; not adopted.
- Six FIT SUMOCORE states: inferred-minus-default600s MAPPO/PI effect
  +4171/+3909veh-s, almost wholly road. Native road improves8523 vs controlled
  4351/4617veh-s. Initial HDVs AND entrant donors changed; cause unidentified.
- Point candidate acceleration RMSE 0.299 vs initial0.902 m/s², 10s position
  MAE5.58 m vs constant-speed13.81 m. Conditional FIT only; INT/EVAL must score
  free-running congestion/breakdown before adoption. Mainline-only CF fit and
  missing junction/LC constraints limit physical parameter identification.
- Student-t variants v2/v3 scale-collapsed; stopped verified OWN PIDs/children,
  STOP receipts and original partial logs preserved. Neither adopted.
- No new SUMO simulation or neural-network training.
