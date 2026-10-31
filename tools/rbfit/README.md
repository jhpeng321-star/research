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

`validation_recorded_endpoints.json` supersedes original Euler-reconstructed
position scores. `predictive_validation.json` scores physical-profile spread only,
without injecting fitted residual noise. These conditional FIT checks do not
qualify future traffic distributions; INT/EVAL must do the free-running checks.
