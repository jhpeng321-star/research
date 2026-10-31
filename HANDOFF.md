# S-RB-UNC — owner00:40 implementation READY; fitted-input panel RUNNING

Fresh-session resume preserves clean baseline46c33fbe and all completed probes.
Next: fitted IDM/optional LC posterior + learned entrance-counter process +
identity-keyed action CRN. Frozen plan: `docs/paper_results/RBUNC_ADDENDUM_DESIGN.md`.
Remote terminal rechecked: old31children exited0; no old dispatch repeated.

Latest implementation `f0d2a588` (includes `f8b6faea`, `e038da91`): ready for
INT/DECIDE to integrate. CPU34PASS including original futures regression;
actual INT SUMO-IDM+LC2013 insertion smoke PASS, persistent fitted LC and exact
newborn admission/motion, bit-identical replays, global RNG unchanged.
Receipt: `docs/paper_results/rbunc/addendum_integration_smoke_v2.json`.
Fitted-input H120 panel runs ONCE in `cpuQ:rbunc20261031/addendum_v2`, own
supervisor457329/four disjoint nice19CPUworkers, immutable code62ad8c687e9a.
Old v1 failed BEFORE any rollout due to unidentified-driver missing weights;
population-fallback correction34CPUtestsPASS; original logs/terminal preserved.

## Owner00:40 fitted-posterior extension (CPU verified)
- `FittedHiddenState.from_directory(finite_mixture_v4, lane_directory=None)` in
  `wms.control.fitted_hidden`. Optional explicit `lane_directory=lane_v1` enables
  named joint LC draws; FIT LC is NOT adopted automatically.
- Pass `fitted_state=fit, vehicle_ids=observable_numeric_ids[B,slots], t0_s=300`
  plus numeric world `row_keys` to `plan_hidden_state_futures`. Existing switch
  remains default-OFF. Existing HDVs use fitted sparse posterior; missing IDs and
  new HDVs use fitted joint population. No extra encoder reliability mixture.
- Exact physical order is mapped, not clipped. CAV settings stay preserved.
  Current draws key on world/VID/future, invariant under slot permutations and
  candidate actions. Birth draws remain entrance/ordinal-keyed, shared admission
  and motion. LC block independent of IDM block (FIT supplied no joint coupling).
- CPU new+existing hidden suite16PASS.

## Owner00:40 learned arrivals (READY)
- `LearnedArrivalFit.from_file('docs/paper_results/rbunc/observed_arrivals.json')`
  in `wms.control.arrival_fit`; pass `arrival_fit=fit, t0_s=300` to planner.
  Optional `sample_demand_rate=False` isolates request-clock vs rate uncertainty.
- Actual FIT32 counters fitted: request(t)=inserted(t)+queue(t), exact10s bins.
  Mainline/ramp within-world Pearson dispersion0.9766/1.0759; NB minus Poisson
  BIC+6.87/+6.27. No supported excess-dispersion component. Empirical time-varying
  profile minus constant BIC+166.1/+174.8 supports documented constant phase
  before1200s; per-bin counts/rate profiles preserved, not spurious extrapolation.
- Sample one Gamma(count+0.5,rate=exposure) demand intensity per future/entrance,
  then exact exponential request clocks. Rate posterior uses Jeffreys Poisson
  prior; no behavioral threshold, clipping or per-tick physical perturbation.
- Exact integrated hazard supports piecewise rates/zero-rate intervals; refuse
  forecasting past end of fitted phase (here1200s absolute). K horizons<=600s
  from300s supported. Same world/future keys pair both rates and request clocks.
- `.condition(world, observed_entries, observed_queue_history)` supports new
  causal starts under the frozen family without reading future labels. Unknown
  worlds otherwise fail closed. Mutable future objects still recreated per arm.
- Combined CPU suite34PASS; new32-state fitted-source verification running.

Branch `c4/rb-unc`, base `f0db3c82`. Owner registration5683/5694 and WHY +
ENSPROBE read. Merge the branch; retain INT's own root HANDOFF on conflict.
Implementation commits: `1f918969`, `ebea8483`, `9739d304`, `73cb13be`.

## API / switch
- NEW `wms.control.hidden_state.plan_hidden_state_futures`:
  `(model, initial, history_kwargs, future_sampling='hidden_state', n_samples=8,
  seed=190, row_keys=world_keys, entrance_histories=histories, queues=queues)`.
  Returns K `HiddenFuture` objects: `.kwargs` for rollout, `.receipt` provenance.
- Default `future_sampling='off'` -> None, no encoding/RNG work.
- Caller pads first and supplies observed request-demand/backlog/CAV share.
  Create fresh plans/controllers per alternative or replay (entries are mutable).
  Equal row keys + equal starts/pools share drivers/demand/newcomers across arms.
- `sample_present=False` / `sample_arrivals=False` isolate the two latent sources.
- No shared wms/spec/factory/trainer changes. API example: `tools/rbunc/README.md`.

## Semantics / integration detail
- Present HDVs: calibrated supplied posterior samples, else empirical joint
  population spread around point estimates with pre-t0 motion likelihood.
  Existing encoder reliability mixes short histories toward population draws.
  Incumbent untrained uncertainty head ignored; embeddings stay fixed.
- Future requests: exact Poisson clocks from t0 at observed entrance demand.
  Newcomer settings drawn from motion-inferred joint population by birth ordinal.
- **SUMO bridge:** after MERGE configures insertion inside rollout,
  `PoissonEntries.insert` binds only its future-owned insertion `candidate` to
  its SAME weighted `BirthMix.candidate`. No legacy uniform-hash mismatch.
  Actual INT-runtime synthetic insertion/motion identity + replay smoke PASS.
- Known stochastic modules, train mode, runtime-noise/future-observation kwargs
  rejected. Physics remains deterministic within each conditional future.
- Old MOBIL outputs are NOT LC2013 settings; LC defaults stay defaults until
  motion-inferred LC uncertainty exists. Missing donor pools fail closed.

## Tests / complete FIT diagnostics
- Latest new-suite12CPUtestsPASS; earlier legacy-futures13PASS. Single thread.
- All32 native FIT969400..969431, K8/H120, authentic60s history COMPLETE.
  Reference deterministic IDM/MOBIL + frozen SUMOCORE encoder5b256941,
  NOT integrated WM-PHYS efficacy / main EVAL checkpointb8797242.
- Mean within-K team SD574.4 [506.6,644.9] vs real across-world SD4200.5veh-s.
  First-state drivers-only134.1, arrivals-only755.6, both696.7, fixed0.0veh-s;
  fixed futures have identical complete physical tape hashes.
- Queue conservation, finite dynamics, no missing padding/dropped requests,
  no global physics RNG consumption; all280v2 futures pass. Independent CRPS/
  variance/target/source-invariant checks PASS96metric-world cells.
- V1 retained; corrected raw-IDM likelihood to existing executed-plant bounds
  (max residual MSE3.18e10 ->42.51). No factual delay used to tune spread.
- Report `docs/paper_results/RBUNC.md`; final raw/summary/verification and own
  terminal receipts in `docs/paper_results/rbunc/`. All32 complete/31child rc0.

## Scope / remaining integration
- Reference spread is a source-plumbing diagnostic, not calibrated coverage or
  a WM-PHYS win. Full120/300/600s WM-PHYS action/distribution panel stays INT/EVAL.
- Request-clock spread checked120/300/600s; traffic-delay spread here only120s.
- Across-world SD includes observed-state heterogeneity: never tune conditional
  SD to equal it. Recorder `arrivals` are destination exits, not entrance demand.
- No SUMO/GPU/training, prohibited data, downloads, new venvs, deletions, push
  or peer-process actions. CPU nice19 owned workers; both versions preserved.
