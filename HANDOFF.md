# S-RB-FIT handoff

Branch `c4/rb-fit`, base `f0db3c82`; owner amendment REGISTRATION.md:5694.
Status: observable-only extraction and hierarchical physical fitting in progress.

## Frozen data/estimation contract
- Only FIT worlds 969400–969431, native arm, from cpuQ
  `/home/ubuntu/wmdata20261031/views`; explicit allowlist of `episode.json`,
  `time_s.npy`, `veh_kin.npy`, `veh_mask.npy`, `is_cav.npy`, `vid.npy`.
- Fit motion in [240,300] s only; driver identity = world + observable numeric VID.
  Do not combine the five counterfactual arms as independent drivers.
- Per-driver posterior uses at most the causal 60 s before t=300, shorter when
  absent; unseen/absent drivers use the learned population distribution.
- Acceleration target is forward speed difference, not a hidden driver setting.
  Exclude junction/no-lane snapshots, ego lane transitions, changed/absent leaders,
  invalid identities, and negative physical bumper gaps; retain stopped traffic.
- Positive IDM parameters in log space: v0, tau, accel, decel, minGap; delta=4
  (SUMO default, fixed). Population log-normal prior + per-driver Laplace
  posterior; variance/noise learned from motion, not labels.
- Validation is later native FIT motion only, conditional on recorded leaders and
  lanes; this does not establish free-running breakdown/traffic fidelity.

## Outputs / integration
Pending: `artifacts/rbfit/idm_population.json`, `idm_posteriors.json`,
`validation.json`; schemas and measured results will be added after fitting.
No shared model/spec/factory edits; no switches enabled by this workstream.

## Resources and gaps
- cpuQ load 0.07 on 32 CPUs at first check; existing miniconda Python available.
  Local CPU busy; no local GPU driver. CPU fitting, no GPU allocation required.
- Shared `setdata/` and `idmspec/DONE.md` not ready at start.
- LC2013 module usable in c4-rb-lane; lane physical-parameter fitting follows IDM.
- No new SUMO simulation or neural-network training.
