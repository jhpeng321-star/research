# Hidden-state future sampling

`wms.control.hidden_state.plan_hidden_state_futures` is an inference-only adapter.
Default `future_sampling='off'` returns `None`, without encoding or RNG draws.
No edits to shared `wms.py`, `spec.py`, `factory.py` or trainer configuration.

Integration (A = existing motion encoder):

```python
from wms.control.hidden_state import plan_hidden_state_futures

futures = plan_hidden_state_futures(
    model, initial, history_kwargs, future_sampling='hidden_state',
    n_samples=8, seed=190, row_keys=[world_identity] * n_arms,
    entrance_histories=observed_histories, queues=observed_queues)
for future in futures:
    roll = model.rollout(initial_copy(), n_steps=1200,
                         cav_controller=fresh_controller(),
                         **history_kwargs, **future.kwargs)
```

Inputs are already padded by the caller, as with existing entry processes.
Each future owns its mutable entry process and birth sampler; reconstruct the
plan for replay/another action rather than reuse a consumed process. Paired
action rows use identical `row_keys`, state, history, demand and population pools.
`queues` holds OBSERVABLE t0 backlog, request-demand rates, and CAV shares.
If absent, entrance-history rates are explicitly a possibly censored insertion
proxy. No future request schedule, driver labels or assignment paths are read.

## Uncertainty contract

- Present HDVs: sample once at t0 and hold settings fixed. CAV present-driver
  settings and embeddings stay at their point estimates. No generic noise.
- A supplied calibrated encoder `posterior_samples [P,B,K,9]` is consumed
  directly as equally weighted joint samples. Incumbent `uncertainty` has no
  loss and is ignored; arbitrary +/-15%/50% caps are not reused.
- Fallback: empirical **joint** inferred HDV population deviations around each
  point estimate, constrained by the existing encoder parameter support, with
  acceleration-likelihood weights. The scale is empirical pre-t0 acceleration
  residual MSE; effective sample size accounts for serial residual correlation.
  The likelihood applies the model's existing plant acceleration bounds: raw
  IDM values at compressed/censored gaps are not observable executed motion.
  This calibrates a motion-likelihood scale, NOT held-back predictive coverage.
- Existing encoder reliability `n/(n+shrink_prior_steps)` mixes the local
  posterior and the population. At zero history it is exactly a population draw.
  Population donors must be present with real observed history; no padded donors.
  `Population` can instead be supplied from the learned motion-only newcomer mix.
- Nine-channel histories condition the IDM likelihood using observed speed,
  acceleration, gap and leader-relative speed/flag. Six-channel legacy history
  lacks leader speed and therefore uses its empirical spread without pretending
  to have a complete IDM likelihood.
- Longitudinal likelihood cannot identify lane parameters. MOBIL/merge outputs
  must NOT be reinterpreted as LC2013 settings; WM-PHYS keeps its documented LC
  defaults until a motion-inferred LC posterior becomes available.
- Future requests: exact exponential inter-arrival times from t0 at observed
  demand rates, independent entrances, no minimum headway or initial dt shift.
  This matches scenario source `traffic_shepherd/scenarios.py:49-74` throughout
  the t=300..900 interval (the scheduled demand change begins at t=1200).
- Entrant settings: weighted joint inferred population draws keyed by world,
  future, entrance and admission ordinal. Admission and subsequent motion use
  the SAME vector, including when a blocked queue changes slot/birth order.
  Lane/CAV identity draws are fixed latent entrant attributes, not process noise.
- WM-PHYS's `configure_entries` may replace admission inside rollout. This
  opt-in Poisson process then binds its own `SumoInsertion.candidate` to its
  own weighted birth sampler. The old uniform-mix hash is never allowed to
  silently substitute different admission settings; placement physics is intact.
- No stochastic lane/merge module or learned insertion hazard is permitted.
  `require_deterministic_physics` fails closed on known stochastic modules.
  Caller must not add `accel_noise`, `state_noise`, teachers or future masks.
  This is designed for rebuilt deterministic WM-PHYS, not unchanged WM2.

Attribution ablations: `sample_present=False`, `sample_arrivals=False`.
The latter fixes both request and newcomer-settings streams across futures;
it does not remove arrivals. With both false all K futures must agree exactly.

CPU verification:
`py -3.13 -B -m pytest -q -p no:cacheprovider tests/unit/test_hidden_state_futures.py`

No distributional qualification is implied by implementing the right latent
sources. Across-world real variation includes observable-state differences and
is not a target for conditional same-state ensemble SD.

## Owner00:40 fitted posterior and learned demand

```python
from wms.control.fitted_hidden import FittedHiddenState
from wms.control.arrival_fit import LearnedArrivalFit

fit = FittedHiddenState.from_directory("artifacts/rbfit/shrinkage_v1")
# LC is separately opt-in: lane_directory="artifacts/rbfit/lane_v2".
# FIT LC validation lost to its baseline; it is not silently adopted.
demand = LearnedArrivalFit.from_file("docs/paper_results/rbunc/observed_arrivals.json")
futures = plan_hidden_state_futures(
    model, initial, history_kwargs, future_sampling="hidden_state", n_samples=8,
    seed=20261031, row_keys=[observable_world] * n_arms,
    fitted_state=fit, vehicle_ids=observed_numeric_vids, t0_s=300,
    entrance_histories=observed_histories, queues=observed_queues,
    arrival_fit=demand)
```

- Numeric IDs are recorder `vid.npy` identities, **not** indices into route-string
  `start.vehicle_ids`. Example exact binding: `fitted_probe.observed_ids`.
- Current HDVs: exact sparse finite-support posterior; unknown or summary-only
  records use the population. No log-normal moment reinterpretation or additional
  encoder reliability shrinkage. Nine-coordinate IDM mapping is by name.
- Preferred S-RB-FIT `shrinkage_v1` already includes learned population mixing
  n/(n+19.35568); do not apply it again. Raw `finite_mixture_v4` remains supported
  and is the frozen input of this workstream's new H120 physical diagnostic.
- Saved-ID posteriors exist only for FIT969400..969431 at300s. For other worlds
  use FIT's causal `MotionFit.condition_motion` on the caller's stable-mainline
  motion intervals, with faithful next-speed predictions in the frozen support
  order and the learned shrinkage policy. Bind its returned distributions with
  `fit.with_motion_posteriors({(world, vid): distribution, ...}, t0_s=t0)` and
  pass the returned reader as `fitted_state=`. This samples the exact finite
  weights and the fitted entrant population without a finite-particle surrogate.
  Rebinding clears ALL saved-ID records and optional LC; missing IDs use population.
  The caller is responsible for support/likelihood compatibility and causal input
  intervals. No new recording is read by this adapter.
- `MotionFitForUNC` is an alternative supplied-particle proxy for saved-ID
  posteriors (or population-only draws), not a causal-motion inference wrapper.
  Use the supplied-posterior path plus fitted `Population` via `population=` for
  newcomers. Do not combine it with `fitted_state=`. Never fake a numeric VID
  from packed slots or route-string ordinals.
- New HDVs: fitted population. Current/new CAV plant settings remain the existing
  observed inference, outside the HDV physical fit. Optional named LC settings
  persist for current/new HDVs; no fitted joint IDM/LC dependence is claimed.
- Optional LC support follows the source domain: first four motivations may be
  zero; cooperative must be at most one; assertiveness must be strictly positive.
  No clipping. Old unadopted LC exports with cooperative>1 are rejected. Existing
  source-bound historical diagnostic receipts remain historical, not new results.
- Fit artifact records all32 nativeFIT entrance histories using insertions+queue
  conservation. It retains10s time-varying rate estimates and BIC diagnostics.
  Data and documented generator favor the constant pre1200s phase, with no
  supported added overdispersion. Arrival-rate estimation uncertainty uses one
  Gamma(count+.5, rate=observed exposure) posterior draw per future/entrance;
  requests are conditional Poisson with exact integrated-intensity inversion.
- `sample_demand_rate=False` fixes rates at their observed MLE, retaining request
  randomness. `sample_arrivals=False` fixes rate/request/newcomer draws across K.
- For a permitted new observation, first call
  `demand.condition(world, observed_entries, observed_queue_history)`; this only
  conditions the frozen family on causal counts. A missing world without causal
  counters fails closed. Requests beyond fitted phase end1200s are refused.
- Paired candidate actions must share initial observations, numeric IDs, row key,
  seed, and future index. Rebuild mutable future objects per arm. Hidden rates,
  incumbent settings, request clocks, and entrant settings then share CRN exactly.
- Pass `vehicle_ids=` also with supplied causal-motion posteriors: the final
  draw then keys by observable VID, preserving CRN under packed-slot permutations.
  Present IDs must be unique nonnegative integers per row; padding IDs are ignored.
  Legacy callers without IDs retain slot-keyed draws, identified in each receipt
  as `driver_draw_key='packed_slot'`. Supplying both `fitted_state=` and encoded
  `posterior_samples` is rejected rather than silently overriding either source.
- CPU suites: `test_fitted_hidden.py`, `test_arrival_fit.py`,
  `test_hidden_state_futures.py`, `test_c4_futures.py` (42 tests).
- New fitted-input physical verification and retained failure lineage:
  `docs/paper_results/RBUNC_ADDENDUM_DESIGN.md`, `rbunc/fitted_source_probe.json`.
  Complete-panel summaries: `python tools/rbunc/analyze_addendum.py`.
- Actual FIT API/artifact compatibility smoke (synthetic motion, no recording):
  `PYTHONPATH=src python -B tools/rbunc/motion_bridge_smoke.py --fit-source
  ../c4-rb-fit/src/wms/physics/motion_fit.py --directory
  ../c4-rb-fit/artifacts/rbfit/shrinkage_v1 --policy
  ../c4-rb-fit/artifacts/rbfit/shrinkage_v1/idm_shrinkage.json --out NEW_RECEIPT.json`.
- Preferred32-state panel is COMPLETE in `rbunc/preferred_source_probe.json`.
  Analysis only: `python -B tools/rbunc/analyze_addendum.py --raw
  docs/paper_results/rbunc/preferred_source_probe.json --compare-raw
  docs/paper_results/rbunc/fitted_source_probe.json --out NEW_SUMMARY.json`.
  Exact source/input equality is checked before reporting paired CRPS changes.
  Read `RBUNC_PREFERRED.md`; do not redispatch completed `preferred_v1`.
