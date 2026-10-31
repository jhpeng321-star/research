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
