"""Observable hidden-state futures for deterministic traffic physics.

No random acceleration, embedding, lane decision or merge event is introduced.
The existing encoder's variance head is untrained, so its output is deliberately
not interpreted as a posterior. Unless a calibrated posterior is supplied, joint
population deviations around the encoder point estimate are reweighted by an IDM
motion likelihood. Its residual scale is estimated from the supplied causal
history. This is empirical-Bayes *approximate* uncertainty, not coverage-certified
parameter inference. Lane settings retain their population joint prior: an IDM
likelihood does not identify LC2013 parameters.

Public entry point: ``plan_hidden_state_futures(..., future_sampling='hidden_state')``.
Default ``'off'`` returns None without encoding, allocating streams or touching RNG.
Each returned future must be constructed afresh for each alternative action. Equal
row_keys share hidden drivers, arrivals and entrant identities across alternatives.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Mapping

import numpy as np
import torch

from wms.model.driver_encoder import PARAMETER_LOWER, PARAMETER_NAMES, PARAMETER_UPPER
from wms.model.entries import EntranceQueue, EntryConfig, TorchEntryProcessV2
from wms.model.learned_admission import LearnedAdmission

from .futures import seed_of


def _generator(*key):
    return torch.Generator(device='cpu').manual_seed(seed_of('hidden-state', *key))


def _pick(weights, generator):
    return int(torch.multinomial(weights.detach().double().cpu(), 1, generator=generator))


def _stack(params):
    return torch.stack([params[name] for name in PARAMETER_NAMES], -1)


@dataclass
class Population:
    """Frozen JOINT inferred vectors; no parameter-wise independent type sampling.

    Pools can be supplied from a previously learned motion-only population. The
    convenience builder uses the same observed-driver pool as newcomer-mix, split
    into HDVs/CAVs, excluding padding, absent and history-free vehicles.
    """
    hdv: list
    cav: list
    weights: list

    @classmethod
    def from_motion(cls, params, state, mask):
        vectors = _stack(params).detach()
        count = mask.bool().sum(-1)
        present = state['present'].bool()
        cav = state['is_cav'].bool()
        hdv_pools, cav_pools, weights = [], [], []
        for b in range(present.shape[0]):
            observed = present[b] & (count[b] > 0)
            hdv = observed & ~cav[b]
            hdv_pools.append(vectors[b, hdv].clone())
            cav_pools.append(vectors[b, observed & cav[b]].clone())
            # More observed motion contributes proportionately to the empirical
            # population, with no arbitrary short-history cutoff.
            weights.append(count[b, hdv].to(vectors.dtype))
        return cls(hdv_pools, cav_pools, weights)

    def validate(self, batch):
        if not (len(self.hdv) == len(self.cav) == len(self.weights) == batch):
            raise ValueError('population requires one pool per batch row')
        for pool, weights in zip(self.hdv, self.weights):
            if pool.ndim != 2 or pool.shape[1] != len(PARAMETER_NAMES) or not len(pool):
                raise ValueError('no observed HDV population; supply a learned motion-only pool')
            if weights.shape != (len(pool),) or not torch.isfinite(weights).all() \
                    or (weights < 0).any() or weights.sum() <= 0:
                raise ValueError('invalid inferred population weights')
        for pool in self.hdv + self.cav:
            if pool.ndim != 2 or pool.shape[1] != len(PARAMETER_NAMES) \
                    or not torch.isfinite(pool).all() or (pool[:, :5] <= 0).any():
                raise ValueError('nonphysical inferred population')


def _prediction(history, particles, acceleration_limits=None):
    """IDM acceleration likelihood, [particles,H]; history is observable [H,9].

    Normalisations and leader flag are exactly data.history.long_history_features.
    This likelihood is not a physics-step perturbation or a SUMO parity claim.
    """
    v = history[None, :, 0] * 30
    closing = -history[None, :, 7] * 30
    has_leader = history[None, :, 8].bool()
    gap = history[None, :, 5] * 100
    v0, tau, gap0, accel, decel = [particles[:, j:j + 1] for j in range(5)]
    desired = gap0 + (v * tau + v * closing / (2 * (accel * decel).sqrt())).clamp_min(0)
    interaction = torch.where(has_leader, (desired / gap.clamp_min(
        torch.finfo(gap.dtype).eps)).square(), torch.zeros_like(desired))
    prediction = accel * (1 - (v / v0).pow(4) - interaction)
    # Observed acceleration is executed plant motion, not raw unbounded IDM.
    # Near a compressed/censored gap raw IDM can be enormous; using that as an
    # observational residual would falsely inflate uncertainty by many orders.
    return prediction if acceleration_limits is None else prediction.clamp(*acceleration_limits)


def _motion_weights(history, mask, particles, point, prior_weights, acceleration_limits=None):
    """Residual-scale calibrated likelihood with serial-correlation effective N.

    No supplied observations => prior exactly. Six-channel legacy histories lack
    leader-relative speed, hence are not silently treated as likelihood evidence.
    """
    if history is None or history.shape[-1] != 9 or int(mask.sum()) < 2:
        return prior_weights, None
    h = history[mask.bool()]
    if not torch.isfinite(h).all():
        raise ValueError('nonfinite observed history')
    actual = h[:, 1] * 3
    residual = actual - _prediction(h, point[None], acceleration_limits)[0]
    # Empirical residual MSE includes model discrepancy; NEVER used as runtime noise.
    variance = residual.square().mean().clamp_min(torch.finfo(h.dtype).eps)
    centered = residual - residual.mean()
    rho = ((centered[:-1] * centered[1:]).sum() /
           centered.square().sum().clamp_min(torch.finfo(h.dtype).eps)).clamp(0, 1)
    effective = (len(h) * (1 - rho) / (1 + rho)).clamp_min(1)
    error = (_prediction(h, particles, acceleration_limits) - actual[None]).square().mean(-1)
    log_weight = prior_weights.log() - 0.5 * effective * error / variance
    weights = log_weight.softmax(0)
    return weights, dict(residual_mse=float(variance), effective_n=float(effective),
                        particle_ess=float(1 / weights.square().sum()))


class BirthMix:
    """Ordinal-keyed population draw shared by admission and newborn motion.

    The candidate uses the historical LearnedAdmission key format, so its tested
    vector is byte-identical to the one assigned to that successful birth. This
    object is owned by one rollout; never share its counters between alternatives.
    """
    names = PARAMETER_NAMES

    def __init__(self, population, seed, row_keys, fitted_state=None):
        self.seed, self.row_keys = int(seed), list(row_keys)
        self.pools = [p.detach().clone() for p in population.hdv]
        self.weights = [w.detach().double().cpu().clone() for w in population.weights]
        self.cav_pools = [p.detach().clone() for p in population.cav]
        self.counts = [[0, 0] for _ in self.pools]
        self.assigned = [0] * len(self.pools)
        self.fitted_state = fitted_state

    def pool_for(self, b, cav):
        pool = self.cav_pools[b] if cav else self.pools[b]
        if not len(pool):
            raise ValueError('new CAV without an observed/learned CAV population')
        return pool

    def candidate(self, b, entrance, ordinal, cav=False):
        pool = self.pool_for(b, cav)
        weights = torch.ones(len(pool), dtype=torch.float64) if cav else self.weights[b]
        key = f'hidden-birth:{self.seed}:{self.row_keys[b]}:{entrance}:{ordinal}'
        u = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') / 2**64
        index = int(torch.searchsorted(weights.cumsum(0) / weights.sum(),
                                       torch.tensor(u, dtype=torch.float64)))
        values = dict(zip(self.names, pool[min(index, len(pool) - 1)].detach().cpu().tolist()))
        if self.fitted_state is not None and not cav:
            values.update(self.fitted_state.draw(self.row_keys[b], None, lane=True,
                generator=_generator(self.seed, self.row_keys[b], entrance, ordinal, 'birth-lc')))
        return values

    def __call__(self, state, newborn, params, birth_ordinals=None):
        result = {name: value.clone() for name, value in params.items()}
        for b, slot in newborn.nonzero().tolist():
            entrance = int(state['lane'][b, slot] == 2)
            if birth_ordinals is None or (b, slot) not in birth_ordinals:
                raise ValueError('hidden-state birth requires admission identity ordinal')
            ordinal = birth_ordinals[b, slot]
            values = self.candidate(
                b, entrance, ordinal, bool(state['is_cav'][b, slot]))
            for name in values:
                result[name][b, slot] = values[name]
            self.counts[b][entrance] += 1
            self.assigned[b] += 1
        return result


class PopulationAdmission(LearnedAdmission):
    """Same deterministic placement, with the exact weighted joint population draw."""
    def candidate(self, b, entrance, ordinal, cav=False):
        return self.mix.candidate(b, entrance, ordinal, cav)


class PoissonEntries(TorchEntryProcessV2):
    """Exact homogeneous Poisson requests, clocked from t0 (no initial dt shift).

    Scenario source: traffic_shepherd/scenarios.py:49-74 expovariate(rate/3600).
    Independent entrance/ordinal streams make demand invariant to blocking and
    alternative actions. No minimum headway floor, zero rate means zero arrivals.
    All request times are latent inputs; insertion itself remains deterministic.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.due_total = np.zeros((self.batch, 2), dtype=np.int64)
        self._request_rngs = [[np.random.default_rng(seed_of(
            'hidden-arrivals', self._pair_seeds[b], e)) for e in range(2)]
            for b in range(self.batch)]
        self.t_next = np.array([[self._next(b, e) for e in range(2)]
                                for b in range(self.batch)])

    def _next(self, b, e):
        rate = float(self._hist(b).rate[e])
        return float(self._request_rngs[b][e].exponential(1 / rate)) if rate > 0 else np.inf

    def _emit_due(self, b):
        self.now[b] += self.dt
        due = []
        for e in range(2):
            while self.t_next[b, e] <= self.now[b]:
                due.append(e)
                self.due_total[b, e] += 1
                self.t_next[b, e] += self._next(b, e)
        return due

    def insert(self, state):
        # WM-PHYS merge mode configures its deterministic SumoInsertion inside
        # model.rollout, AFTER the future is planned. Its legacy mix compatibility
        # reconstructs a uniform-pool hash; a calibrated/weighted posterior may
        # not use that hash. Bind ONLY this future-owned insertion object's latent
        # lookup to the exact draw used by newborn motion. Placement is unchanged.
        insertion = getattr(self.cfg, 'sumo_insertion', None)
        if insertion is not None:
            mix = getattr(self, 'birth_mix', None)
            if mix is None or insertion.mix is not mix:
                raise ValueError('SUMO insertion and newborn motion must share the future population')
            insertion.candidate = mix.candidate
        return super().insert(state)


def require_deterministic_physics(model):
    """Fail closed on known stochastic event modules; never mutate model switches."""
    if getattr(model, 'training', False):
        raise ValueError('hidden-state futures require evaluation mode')
    if getattr(model, '_samples_lanes', False):
        raise ValueError('hidden-state futures require deterministic lane physics')
    for name in ('merge_pivot', 'merge_threshold', 'merge_frailty', 'merge_core'):
        if getattr(model, name, None) is not None:
            raise ValueError(f'hidden-state futures require stochastic {name} OFF')
    if getattr(getattr(model, 'physics', None), 'entry_learned', False):
        raise ValueError('hidden-state futures require deterministic admission')


@dataclass
class HiddenFuture:
    kwargs: dict
    receipt: dict


@torch.no_grad()
def plan_hidden_state_futures(model, state: Mapping, history_args: Mapping, *,
                              future_sampling='off', n_samples=8, seed=0, row_keys=None,
                              population=None, entrance_histories=None, queues=None,
                              entry_config=None, sample_present=True, sample_arrivals=True,
                              fitted_state=None, vehicle_ids=None, t0_s=None,
                              arrival_fit=None, sample_demand_rate=True):
    """Prepare K deterministic conditional futures, with independent latent draws.

    Histories/queues MUST be observable at t0. queues.demand_rate is upstream
    request demand, not saturated insertion service. Without queues the supplied
    entrance-history rate is an explicit observed-service proxy. No future ledger
    or scenario driver identity is accepted. ``sample_*`` are attribution ablations.

    A caller may expose encoded ``posterior_samples [P,B,K,9]`` from calibrated
    motion inference. Otherwise use the empirical motion-likelihood posterior.
    ``uncertainty`` alone is not trusted (the incumbent head is untrained).
    """
    if future_sampling == 'off':
        return None
    if future_sampling != 'hidden_state':
        raise ValueError("future_sampling must be 'off' or 'hidden_state'")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError('n_samples must be a positive integer')
    require_deterministic_physics(model)
    forbidden = ('accel_noise', 'state_noise', 'lane_uniform', 'lane_seed',
                 'observe_states', 'observe_mask', 'cav_teacher', 'present_seq',
                 'junction_seq', 'is_cav_seq', 'lane_teacher', 'merge_lane_teacher')
    if any(history_args.get(key) is not None for key in forbidden) or any(
            history_args.get(key, 0) != 0 for key in ('state_noise_std_m', 'state_noise_std_mps')):
        raise ValueError('hidden-state futures reject process noise and future observation overrides')
    keys = ('history', 'history_mask', 'driver_history', 'driver_history_mask', 'driver_defaults')
    args = {key: history_args[key] for key in keys if key in history_args}
    encoded, embedding, params = model.encode_drivers(dict(state), **args)
    if params is None or set(params) != set(PARAMETER_NAMES):
        raise ValueError('hidden-state futures need physical inferred driver settings')
    point = _stack(params).detach()
    batch, slots, _ = point.shape
    row_keys = list(range(batch)) if row_keys is None else list(row_keys)
    if len(row_keys) != batch:
        raise ValueError('row_keys length mismatch')
    history = args.get('driver_history', args.get('history'))
    mask = args.get('driver_history_mask', args.get('history_mask'))
    if mask is None:
        mask = torch.zeros((*point.shape[:2], 1), dtype=torch.bool, device=point.device)
    if vehicle_ids is not None:
        if not isinstance(vehicle_ids, torch.Tensor) or vehicle_ids.shape != state['present'].shape:
            raise ValueError('observable vehicle_ids must match state slots')
        for b in range(batch):
            ids = vehicle_ids[b][state['present'][b].bool()].detach().cpu()
            if (ids.dtype == torch.bool or ids.is_complex() or not torch.isfinite(ids).all()
                    or (ids < 0).any() or (ids != ids.long()).any()
                    or len(ids.unique()) != len(ids)):
                raise ValueError('present vehicle_ids must be unique nonnegative integers per row')
    if fitted_state is not None:
        fitted_state.check_time(t0_s)
        if vehicle_ids is None or vehicle_ids.shape != state['present'].shape:
            raise ValueError('fitted posterior requires observable vehicle_ids matching slots')
        if any(isinstance(key, bool) or int(key) != key for key in row_keys):
            raise ValueError('fitted posterior requires numeric observed world row_keys')
        population = fitted_state.as_population(params, state, mask)
    population = population or Population.from_motion(params, state, mask)
    population.validate(batch)
    if entrance_histories is not None and len(entrance_histories) != batch:
        raise ValueError('entrance_histories length mismatch')
    if queues is not None and (entrance_histories is None or len(queues) != batch):
        raise ValueError('queues need entrance histories for every row')
    if arrival_fit is not None and entrance_histories is None:
        raise ValueError('learned arrivals require observable entrance histories')
    lower, upper = point.new_tensor(PARAMETER_LOWER), point.new_tensor(PARAMETER_UPPER)
    encoder = getattr(model, 'driver_encoder', None)
    if getattr(encoder, 'driver_param_mode', None) == 'physical':
        lower[:5], upper[:5] = encoder.physical_lower.to(point), encoder.physical_upper.to(point)
    prior_steps = float(getattr(encoder, 'shrink_prior_steps', 8))
    if not np.isfinite(prior_steps) or prior_steps <= 0:
        raise ValueError('encoder reliability prior must be positive')
    count = mask.bool().sum(-1).to(point)
    reliability = count / (count + prior_steps)
    physics = getattr(model, 'physics', None)
    limits = ((float(physics.accel_min_mps2), float(physics.accel_max_mps2))
              if hasattr(physics, 'accel_min_mps2') and hasattr(physics, 'accel_max_mps2') else None)
    posteriors, diagnostics = {}, []
    supplied = None if encoded is None else encoded.get('posterior_samples')
    if fitted_state is not None and supplied is not None:
        raise ValueError('choose fitted_state OR supplied motion posterior, not both')
    if supplied is not None and (supplied.ndim != 4 or supplied.shape[1:] != point.shape
                                or not len(supplied) or not torch.isfinite(supplied).all()
                                or (supplied[..., :5] <= 0).any()):
        raise ValueError('posterior_samples must be finite [P,B,K,9]')
    for b in range(batch):
        pool = population.hdv[b].to(point)
        weights = population.weights[b].to(point)
        mean = (pool * (weights / weights.sum())[:, None]).sum(0)
        for slot in (state['present'][b].bool() & ~state['is_cav'][b].bool()).nonzero().flatten().tolist():
            if fitted_state is not None:
                continue
            if supplied is not None:
                candidates = supplied[:, b, slot].to(point)
                weights_p = torch.ones(len(candidates), device=point.device, dtype=point.dtype)
                diag = None
            else:
                candidates = (point[b, slot] + pool - mean).clamp(lower, upper)
                weights_p, diag = _motion_weights(
                    None if history is None else history[b, slot], mask[b, slot],
                    candidates, point[b, slot], weights / weights.sum(), limits)
            posteriors[b, slot] = candidates, weights_p
            if diag is not None:
                diagnostics.append(dict(row=b, slot=slot, **diag))
    futures = []
    for sample in range(n_samples):
        values = point.clone()
        branches = dict(population=0, posterior=0)
        for (b, slot), (candidates, weights) in posteriors.items():
            if not sample_present:
                continue
            identity = slot if vehicle_ids is None else int(vehicle_ids[b, slot].item())
            rng = _generator(seed, row_keys[b], sample, 'present', identity)
            use_posterior = supplied is not None or float(torch.rand((), generator=rng)) < float(reliability[b, slot])
            pool = candidates if use_posterior else population.hdv[b].to(point)
            w = weights if use_posterior else population.weights[b]
            values[b, slot] = pool[_pick(w, rng)]
            branches['posterior' if use_posterior else 'population'] += 1
        kwargs = dict(driver_params={name: values[..., i] for i, name in enumerate(PARAMETER_NAMES)})
        if fitted_state is not None:
            from .fitted_hidden import LC_ORDER
            if fitted_state.lane_population is not None:
                kwargs['driver_params'].update({n: torch.ones_like(values[..., 0]) for n in LC_ORDER})
            for b, slot in (state['present'].bool() & ~state['is_cav'].bool()).nonzero().tolist():
                vid = vehicle_ids[b, slot].item()
                # Identity, not slot/order/action, controls the persistent draw.
                rng = _generator(seed, row_keys[b], sample if sample_present else 0, 'fit-idm', vid)
                draw = fitted_state.draw(row_keys[b], vid, generator=rng)
                if fitted_state.lane_population is not None:
                    draw.update(fitted_state.draw(row_keys[b], vid, lane=True,
                        generator=_generator(seed, row_keys[b], sample if sample_present else 0, 'fit-lc', vid)))
                for name, value in draw.items():
                    kwargs['driver_params'][name][b, slot] = value
                key = f'{int(row_keys[b])}:{int(vid)}'
                identified = 'weights' in fitted_state.posterior['drivers'].get(key, {})
                branches['posterior' if identified else 'population'] += 1
        if embedding is not None:
            kwargs['driver_embedding'] = embedding.detach().clone()
        arrival_sample = sample if sample_arrivals else 0
        # Entrant parameter uncertainty is part of future arrivals (b).
        birth_seed = seed_of(seed, arrival_sample, 'newcomers')
        mix = BirthMix(population, birth_seed, row_keys, fitted_state=fitted_state)
        if entrance_histories is not None:
            q = None if queues is None else [EntranceQueue(
                np.array(v.backlog, copy=True), np.array(v.demand_rate, copy=True),
                demand_cv=1., cav_share=None if v.cav_share is None else np.array(v.cav_share, copy=True))
                for v in queues]
            cfg = entry_config or EntryConfig(insert_rule='sumo_like_v2', fresh_no_credit=True,
                                              conserve_queue=True)
            if cfg.learned_entry is not None or not cfg.conserve_queue \
                    or cfg.insert_rule != 'sumo_like_v2':
                raise ValueError('hidden-state entries need conserved deterministic inferred admission')
            hist = [replace(h, headway_cv=np.ones(2)) for h in entrance_histories]
            row_seeds = [seed_of(seed, key, arrival_sample, 'arrival') for key in row_keys]
            process_type, extra = PoissonEntries, {}
            if arrival_fit is not None:
                from .arrival_fit import FittedEntries
                process_type = FittedEntries
                extra['demand_plans'] = [arrival_fit.draw(key, t0_s=t0_s, seed=seed,
                    sample=arrival_sample, sample_rate=sample_demand_rate) for key in row_keys]
            process = process_type(histories=hist, queues=q, batch=batch,
                dt_s=float(model.physics.dt_s), row_seeds=row_seeds, couple_draws=True,
                cfg=replace(cfg, learned_admission=PopulationAdmission(mix)), **extra)
            process.birth_mix = mix
            kwargs.update(entries=process, entrant_driver_sampler=mix)
        futures.append(HiddenFuture(kwargs, dict(future_sampling='hidden_state', future=sample,
            seed=int(seed), driver_branches=branches, likelihood=diagnostics,
            posterior_source=('rbfit_joint_finite_support' if fitted_state is not None else
                              'supplied_motion_posterior' if supplied is not None else 'empirical_motion_likelihood'),
            driver_draw_key='observable_vid' if vehicle_ids is not None else 'packed_slot',
            fitted_artifacts=None if fitted_state is None else fitted_state.provenance,
            fitted_lc=fitted_state is not None and fitted_state.lane_population is not None,
            idm_lc_joint_coupling='independent_fitted_blocks' if fitted_state is not None else None,
            likelihood_acceleration_limits=limits,
            population_sizes=[len(p) for p in population.hdv],
            arrival_sample=arrival_sample, arrivals='poisson' if entrance_histories is not None else 'closed_cohort',
            learned_arrival_sha256=getattr(arrival_fit, 'sha256', None),
            learned_arrival_observations=None if arrival_fit is None else [
                arrival_fit.observation_receipt(key, t0_s=t0_s) for key in row_keys],
            demand_rate_draws=None if arrival_fit is None else [p.rates.tolist() for p in process.demand_plans],
            rate_source=('fitted_entrance_counters' if arrival_fit is not None else
                         'observed_request_demand' if queues is not None else 'observed_insertion_proxy'),
            physics_noise=False, embedding_noise=False)))
    return futures
