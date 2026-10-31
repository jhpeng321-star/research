# S-RB-UNC — fitted hidden settings and learned entrance demand

Owner00:40 addendum; baseline `46c33fbe` preserved. Runtime changes:
`f8b6faea` fitted joint distributions, `e038da91` observed entrance process,
`f0d2a588` summary-only driver fallback. All remain behind
`future_sampling='hidden_state'`; shared model/spec/factory files untouched.

## Delivered contract
- Preferred runtime artifact is now S-RB-FIT `shrinkage_v1` (posterior SHA
  `578e1b2b...`); learned population mixing is already in its weights. FIT's
  independent3130-driver/16paired-future check PASS is reused rather than rerun.
  Current INT synthetic persistence/replay smoke with this artifact and optional
  `lane_v2` also PASS: `rbunc/preferred_integration_smoke.json`.
  The physical panel below stays frozen to RAW `finite_mixture_v4`; it is never
  relabeled a preferred-posterior traffic result.
  Subsequent preferred32-state reference comparison is published separately in
  `RBUNC_PREFERRED.md`: teamCRPS980.79 vs raw980.68, paired change+0.11
  [−24.19,+24.32]veh-s; no demonstrated improvement or integrated WM-PHYS verdict.
- Current HDVs: sample the exact S-RB-FIT finite joint posterior once by numeric
  observable world+VID. Missing/summary-only records use its fitted population.
  No summary-covariance log-normal substitution, clipping, or extra shrinkage.
- New HDVs: joint fitted population draws, keyed by world/future/entrance/ordinal.
  Insertion and subsequent motion use the SAME settings. CAV plant is retained.
- Optional LC2013: named joint fitted vectors persist for current/new HDVs.
  IDM and LC are independent fitted blocks because no cross-block posterior was
  supplied. FIT's LC candidate lost to baseline, so LC sampling is separate opt-in.
- Arrival fit: reconstruct request(t)=inserted(t)+queue(t) from every causal native
  FIT969400..969431 entrance counter, exact10s bins over0..300s. Destination exits
  are never treated as entrance demand. Per-world constant rates are ML estimates
  within the documented pre1200s phase; time-varying empirical profiles retained.
- Hidden demand rate: one Gamma(count+0.5,rate=observed exposure) posterior draw
  per future/entrance (Poisson Jeffreys prior), held for that phase. Conditional
  requests use exact integrated-hazard exponential clocks; zero-rate phases work.
- Mainline/ramp Pearson dispersion0.97656/1.07595 (928df); negative-binomial minus
  Poisson BIC+6.867/+6.271. No supported added overdispersion component.
  Time-varying common-profile minus constant BIC+166.133/+174.830 also favors
  the documented plateau, rather than extrapolating random pre-t0 fluctuations.
- Equal world/VID/future/seed share current settings, demand rates, requests and
  entrant settings across actions. Mutable future objects are rebuilt per action.
  No per-tick physical/event noise or untrained variance-head sampling.

## Verification scope and reproducibility
- CPU suites34PASS: original futures13, original hidden-source12, fitted5,
  arrival4. Tests include exact support/order, identity permutation, summary-only
  fallback, paired streams, observational conservation and integrated hazard.
- Actual current INT SUMO-IDM+LC2013+SUMO insertion synthetic smoke PASS: same
  admitted/newborn parameters, all five LC values persist over five ticks, two
  complete physical replays hash-identical, global RNG unchanged.
  Receipt: `rbunc/addendum_integration_smoke_v2.json` (exact runtime hashes).
- New full32/K8/H120 fitted-input physical probe uses the same deterministic
  legacy IDM/MOBIL reference/scoring as baseline v2. It is not integrated
  WM-PHYS efficacy. Native120s factual targets must equal the old targets exactly.
  First world adds driver-only, arrival-only, fixed-source ablations.
- Numeric identities bind by exact t0 recorder/start position+speed correspondence.
  `start.vehicle_ids` route-string ordinals are NOT the numeric FIT posterior IDs.
- New attemptv1 failed in its first planner before physical rollout: summary-only
  records lack weights. Four children rc1, zero completed worlds/futures. Original
  logs/terminal preserved. Explicit regression-tested population fallback repairs
  only this published-contract mismatch; v2 reuses the completed arrival fit bytes.
- cpuQ existing CPU environment, four disjoint nice19/one-thread workers, load
  admission<24/32, CUDAhidden. New SUMO0/GPU0/neural training0; no peer signals.
- Inputs/artifacts: `rbunc/observed_arrivals.json`, `rbunc/addendum_repair.json`,
  final panel `rbunc/fitted_source_probe.json`, independently checked summary
  `rbunc/fitted_summary.json`; `tools/rbunc/analyze_addendum.py` reproduces it.
- Runtime API and permitted-new-observation conditioning: `tools/rbunc/README.md`.
  Request forecasts beyond the fitted/documented phase end1200s fail closed.

## Complete32-state raw-posterior reference result
- All32 native FIT969400..969431, K8/H120, authentic60s histories; total280
  physical futures including first-state source ablations. All four owned v2
  workers rc0; exact numeric-identity binding, finite dynamics, no missing padding
  or dropped requests, exact queue conservation and global-RNG invariance PASS.
- Mean within-K team-delay SD **747.8 [668.7,826.9] veh-s**, versus real
  across-world SD **4200.5 veh-s**. Road **361.7 [315.3,415.6] vs1140.4**;
  source waiting **594.3 [519.2,669.1] vs3475.7**. Brackets are95% whole-world
  bootstrap CIs (10000 resamples, seed20261031), not conditional coverage bands.
- First-state team-delay SD: both929.5, drivers-only179.1, arrivals/newcomers-only
  893.8, both-fixed **0.0 veh-s**. All eight fixed complete physical tapes hash
  identically. Driver-only request streams match exactly; arrivals-only current
  driver spread is exactly zero. Nonlinear interactions prevent adding variances
  or interpreting these SDs as source fractions.
- Mean K8 request-count SD at120/300/600s:10.51/20.69/34.61 vehicles; mean
  Gamma-Poisson posterior-predictive SD11.26/21.28/36.87. These include hidden
  rate uncertainty as well as conditional Poisson requests. Only request clocks,
  not physical traffic delays, were advanced beyond120s in this diagnostic.
- Mean per-HDV input SD: v0 .982m/s, headway .160s, accel .134m/s²,
  decel .765m/s², minGap .397m. Optional raw-LC latent draws were also checked
  on all32 states, without adopting that candidate in physical dynamics.
- Team-delay bias+1142.1veh-s, empirical CRPS980.7veh-s, K8 range coverage53.1%.
  Earlier encoder-input reference bias+332.8/CRPS611.4/coverage59.4%: **wider
  spread did not improve this reference forecast**. Both posterior and rate
  uncertainty changed; this is not an isolated causal contrast or an integrated
  WM-PHYS verdict. Preferred shrinkage has a subsequent separately frozen
  free-running reference score in `RBUNC_PREFERRED.md`; it never inherits raw
  panel numbers.
- Independent pairwise variance and empirical-CDF CRPS checks PASS96 metric-world
  cells; original factual targets match exactly. Raw bundle SHA256
  `68b3f13c0b35e9983cdf82dcde041c084be04e45f98597ce6d0f37844c09d387`.
  Full worker/dispatch/terminal lineage is included in that bundle.

## Limits
S-RB-FIT's raw driver posterior nominal95% physical-acceleration coverage was68.8%;
its new motion-calibrated shrinkage improves that to90.48% and CRPS .16351→.14657,
but still has no conditional coverage qualification. This adapter samples either
selected artifact exactly, without silent inflation. Across-world real variation also contains observed
state/demand heterogeneity, so it is not a conditional ensemble-width target.
Integrated WM-PHYS120/300/600s distribution/action qualification stays INT/EVAL.
Saved-ID posteriors are only FIT969400..969431/t300. FIT's generic
`condition_motion` interface is needed for new worlds; its finite weights can now
bind via `FittedHiddenState.with_motion_posteriors` without inheriting saved IDs.
`MotionFitForUNC` itself supplies saved-ID/population particles. Unknown numeric
IDs alone fall back to population. Never interpret packed slots as numeric VIDs.
