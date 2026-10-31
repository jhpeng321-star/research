# Observable-motion physical fitting

Offline requirements: NumPy/SciPy in the **existing** cpuQ miniconda environment.
Runtime reader requires only the application's existing Torch dependency.

## Reproduce

Use one BLAS/OpenMP thread, nice19, and check free load before dispatch. The root
below is the only data source; both fitters enforce native FIT world identities.
Each `--out` must be NEW; preserve every previous output.

```text
python tools/rbfit/idm_fit.py --root /home/ubuntu/wmdata20261031/views --out NEW_GAUSSIAN --workers 4 --iterations 25
python tools/rbfit/idm_mixture.py --root /home/ubuntu/wmdata20261031/views --initial NEW_GAUSSIAN --out NEW_MIXTURE --workers 4 --sobol-power 10
python tools/rbfit/pooled_fit.py --root /home/ubuntu/wmdata20261031/views --initial NEW_GAUSSIAN --out NEW_POOLED
python tools/rbfit/revalidate.py --root /home/ubuntu/wmdata20261031/views --fit NEW_GAUSSIAN
python tools/rbfit/predictive_check.py --root /home/ubuntu/wmdata20261031/views --fit NEW_MIXTURE
python tools/rbfit/lane_fit.py --root /home/ubuntu/wmdata20261031/views --idm NEW_GAUSSIAN --out NEW_LANE --workers 4 --sobol-power 6
python tools/rbfit/calibrate_shrinkage.py --root /home/ubuntu/wmdata20261031/views --fit NEW_MIXTURE --out NEW_SHRUNK
python tools/rbfit/package_artifacts.py NEW_MIXTURE
python tools/rbfit/verify_artifacts.py --artifacts artifacts/rbfit
python -m pytest tests/unit/test_rbfit.py -q -p no:cacheprovider
```

LC fitting requires `wms.physics.lc2013` from S-RB-LANE commit `c6b1edab` in
PYTHONPATH. Its highway-subset source SHA is frozen in the output. The own remote
bundle was archived from that commit; downloaded SUMO C++ was only read.
The optional experimental Student-t fitter mode is **rejected**: scale-degenerate
on these deterministic residuals. Reproduction should use default Gaussian.

## JSON contract

- Pooled schema `wm_phys.idm_pooled.v1`: `parameters` is the direct settings mapping.
- Population schema `wm_phys.idm_population.v1`, `parameter_order` exactly
  `[v0,time_headway,accel,decel,min_gap]`, fixed `delta=4`. Desired speed v0 is m/s,
  tau seconds, accel/decel m/s², minGap metres. Actual lane limit maps v0 to
  speedFactor; don't multiply fitted v0 by the resulting factor again.
- Gaussian version: `log_mean` + `log_cov` define a log-normal distribution.
- Finite version: `support[C,5]` + `weights[C]` define the distribution; moment
  summaries `log_mean`/`log_cov` must NOT be used to invent off-support draws.
- Driver schema `wm_phys.idm_posteriors.v1`: `drivers[world:observable_vid]`,
  `conditioning_end_s=300`. Gaussian uses the moments; finite posterior sparse
  `support_indices` and `weights` index POPULATION support. Missing weights or
  unknown identity use population. Short histories are not padded with motion.
- Finite original JSON is 104MB, preserved and ignored locally. Published
  `idm_posteriors.json.gz` is byte-lossless JSON; stdlib `gzip.open(...,'rt')`
  reads it. Publication receipt binds compressed/decompressed SHA256.
- LC schema `wm_phys.lc2013_population.v1`: joint `support`/`weights`, order
  `[lcStrategic,lcCooperative,lcSpeedGain,lcKeepRight,lcAssertive]`. Driver schema
  `wm_phys.lc2013_posteriors.v1` uses weights on the SAME population support.
  Emission probabilities explain observed model discrepancy, NOT runtime noise.

## Runtime

`MotionFit(path).idm(world, present_vid_tensor, t0_s=300, sample=True,
use_posterior=False, generator=g)` gives the physics settings mapping on the ID
tensor's device. Draw once at initialization, persist by identity across every
tick/chunk, and use paired generators/common draws for alternative controllers.
Entrants use the learned population. The driver posterior is causal only at its
conditioning time and is demonstrably underdispersed here; it is experimental.
The helper `draw_lane_parameters` also draws one persistent physical vector.
Neither reader enables a shared model/config switch or alters legacy behavior.

## Owner00:40 addendum

Preferred delivery is `artifacts/rbfit/shrinkage_v1`: population JSON and fully
materialized `idm_posteriors.json.gz` in the EXISTING v1 schemas. The fitted
shrinkage is already applied. On UNC commitf8b6faea or later:

```python
from wms.control.fitted_hidden import FittedHiddenState
reader = FittedHiddenState.from_directory("artifacts/rbfit/shrinkage_v1")
plans = plan_hidden_state_futures(model, state, history_kwargs,
    future_sampling="hidden_state", fitted_state=reader,
    vehicle_ids=observed_vids, t0_s=300, row_keys=worlds, n_samples=8, seed=190)
```

This direct interface uses the SAME learned population for future HDV entrants.
`lane_directory` stays unset because the updated LC fit fails the baseline.
`receipts/ADDENDUM_VERIFIED_v2.json` verifies all3,130 weights against the exact
raw-posterior/population mixture,174 no-evidence fallbacks, every covariance,
actual committed UNC draws,16paired/replayed planned futures and exited PIDs.
No full traffic rollout is implied by that interface check.

For older UNC consumers or an explicit policy applied to RAW artifacts:
`MotionFit(directory, shrinkage_path=POLICY_JSON)` optionally applies
`wm_phys.idm_shrinkage.v1`: mix the exact raw posterior with the SAME fitted
population, posterior fraction `n/(n+prior_count)`; zero usable acceleration
observations use population exactly. `population_only=true` represents the
infinite-shrinkage endpoint. Policy SHA-binds its population and causal time.
The scalar prior_count is fitted by physical acceleration CRPS using prefix
[240,270) to predict [270,300), then frozen before [300,360) scoring. Population
was estimated on the whole pre-t0 FIT sample: this is internal empirical Bayes,
not independent coverage qualification. No action outcomes select the policy.

```python
from wms.physics.motion_fit import MotionFit, MotionFitForUNC
fit = MotionFit("artifacts/rbfit/finite_mixture_v4", shrinkage_path=policy_path)
proxy = MotionFitForUNC(model, fit, worlds, observed_vids, t0_s=300,
                        n_particles=32, seed=190)
# UNC imports live on the integrator/UNC branch; no shared files are changed here.
plans = plan_hidden_state_futures(proxy, state, history_kwargs,
    future_sampling="hidden_state", row_keys=worlds, n_samples=8, seed=190)
# Roll out the ORIGINAL model with each plan.kwargs, retaining each joint draw.
```

`fit.unc_posterior_samples(...)` is also available directly. Shape [P,B,K,9]
uses `driver_encoder.PARAMETER_NAMES`, NOT IDM artifact order: min_gap is third.
The four non-IDM values and all CAV/padding values are preserved from the point
encoder; there is no MOBIL-to-LC conversion. This bridge supplies incumbent
samples; entrant pools still must explicitly use the fitted joint population.

Updated LC reproduction adds `--idm-pooled .../pooled_full_v5/idm_pooled.json`
and freezes lane811cd7e8 (two LC source files, finite route geometry and epsilon),
with identical65 numerical profiles. All old source/output versions remain.

Verification uses the existing cpuQ environment plus the frozen committed UNC
sources (not its working edits): `python tools/rbfit/verify_addendum.py --fit
RAW_MIXTURE --new NEW_SHRUNK --lane NEW_LANE --out NEW_RECEIPT.json`.
The local stdlib-only `verify_addendum_mirror.py` checks publication hashes and
the40-line paper-report limit against that receipt.

## Generic causal posterior when tape VID cannot be resolved

`MotionFit.condition_motion(predicted_next_speed[T,C], speed_before[T],
speed_after[T], dt_s[T], interval_end_s[T], t0_s=..., policy_path=...)`
returns a JSON-compatible driver distribution on the stored population support.
Caller evaluates the faithful CF step for each physical profile (columns in
EXACT support order) and passes only stable-mainline HDV/leader intervals.
The implementation derives observed accelerations from speeds, checks finite
physical motion/nonoverlapping intervals/causal60s, then uses the frozen fitted
Gaussian scale and learned population-contamination strength. Empty input is
exactly population. The population is never re-estimated from these inputs.

This runtime inference needs no tape VID and works at other causal start times.
It can populate `FittedHiddenState(fit.population, posterior)` in memory:
posterior header is the existing v1 schema/order/conditioning_end_s, and each
`drivers[world:local_observable_identity]` is the returned distribution. Caller
supplies the SAME stable numeric identity space to UNC's `vehicle_ids`; it must
not pretend that a packed slot or local identity is a saved tape VID. A string-ID
to local-numeric-ID map is only runtime bookkeeping, not a hidden type label.
Alternatively sample the returned weights directly and cache by original ID.
Unavailable/uninformative history -> population; no evaluation-world data files
were read or population parameters fitted to implement or verify this API.

`validation_recorded_endpoints.json` supersedes original Euler-reconstructed
position scores. `predictive_validation.json` scores physical-profile spread only,
without injecting fitted residual noise. These conditional FIT checks do not
qualify future traffic distributions; INT/EVAL must do the free-running checks.

## Owner01:28 LC continuation

`lane_directional.py` fits source-lane conditioned categorical LC decisions
(stay/target0/target1), so wrong-direction changes are failures. Driver evidence
is causal [240,300]; prefix240..270→270..300 estimates informative-evidence
shrinkage before later300..360 scoring. Emissions are discrepancy only; the
independent `verify_directional.py` rebuilds the likelihood directly from saved
decisions and reports RAW physical proposal scores separately.

`lane_directional_v3` retains the old65profile search for comparability; cooperative
values>1 are a known domain gap. Boundary continuation uses `--support-json` for
documented lcCooperative[0,1], nonnegative behavioral weights, assertive>0;
`--reuse COMPLETED_V3` reuses exact matching physical profile decisions after
SHA-checking LC source/geometry/config/pooled-IDM and exact observable alignment.
Zero-weight faces are legal SUMO parameter settings, not new behavioral rules.
Only new profiles are evaluated; previous outputs remain untouched.

LC JSON schemas remain v1; all driver weights already include shrinkage.
Boundary artifacts declare `requires_boundary_aware_lc_reader=true`. FIT's
`draw_lane_parameters` handles source-legal zero behavioral weights; UNC's
strict-positive reader requires an LC-only domain update before using those
artifacts. IDM always remains strictly positive. Neither version is adopted.

Latest source0a1e7e93 requires `--snapshot-version v2`: restore pre-change lane
origins before LC decisions with the published geometry's lane_position_shift.
`freeze_lc_version.py --lane-commit SHA --prepare` creates an immutable
explicit-file source bundle and an owned CPU-only remote root. Generic dispatcher
`dispatch_lc_version.py --root OWN_ROOT fit` claims exactly once; after independent
verification, action `raw` fits the physical profile mixture by categorical
least squares on the completed cache and learns driver evidence temperature
using causal prefix→suffix raw-decision Brier. Raw decisions and likelihood
emissions are separately scored, so probability calibration cannot hide too
many runtime lane proposals.

`watch_lc.py` is the durable owner01:28 continuation: inspect existing owned
receipts first, publish/commit each verified version, re-freeze only when committed
LANE physics bytes change, stop new starts08:40, normal polling09:00, and allow
existing jobs to drain/publish through09:25 before INT09:30. It has no
process-signalling code. Create-once watcher/job claims prohibit duplicate
restarts; staged user work/edited handoffs block automatic publication commits.
Live status is `receipts/LC_WATCH_0128_STATE.json`; terminal failures preserve
source/logs and require manual reconciliation. No automatic adoption occurs.

The explicit boundary adapter is now available without changing UNC's default
reader or peer files:

```python
from wms.physics.lc_fit_for_unc import load_boundary_fitted_state
reader = load_boundary_fitted_state("artifacts/rbfit/shrinkage_v1",
    lane_directory="artifacts/rbfit/lane_boundaries_v4")
# Pass reader as fitted_state to the existing plan_hidden_state_futures API.
```

This opts into testing that LC candidate. It inherits the real UNC reader and
changes ONLY LC support-domain validation: nonnegative behavioral weights,
cooperative<=1, assertive>0. IDM retains strict positivity. Five tests against
the committed UNC implementation pass, including sparse driver support,
population/entrant fallback, invalid-domain rejection,8paired/replayed planned
futures, unmodified CAV/padding and unchanged global RNG. No LC adoption implied.
UNC commit51ae6345 now also accepts source-valid LC boundaries directly; the FIT
adapter preserves compatibility with the older frozen reader used in our tests.

Explicit `--resume --run-tag UNIQUE` restores the same durable job state after
manual reconciliation; an OS singleton excludes concurrent watcher sessions.
It never restarts an existing remote job. UTF-8 subprocess capture is explicit
on Windows, and candidate names are not reused as artifact-loop variables.
