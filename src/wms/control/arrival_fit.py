"""Observable entrance-counter inference and exact conditional Poisson clocks.

The scenario's pre1200s phase has constant rate. Empirical time-varying rates
are retained for diagnostics; they do not license extrapolating spurious trends
or inventing a process outside the documented generator family.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .futures import seed_of
from .hidden_state import PoissonEntries


def request_bins(past_entries, queue_history, *, t0_s=300., bin_s=10.):
    """Recover requests by conservation, not by saturated insertion throughput.

    Queue snapshots are post-step, and insertion events use that same timebase.
    All data must be causal. The documented generator has zero requests att=0.
    """
    queues = np.asarray(queue_history, dtype=float)
    if (queues.ndim != 2 or queues.shape[1] != 3 or not len(queues)
            or not np.isfinite(queues).all() or (queues[:, 1:] < 0).any()
            or not np.equal(queues[:, 1:], np.rint(queues[:, 1:])).all()
            or (np.diff(queues[:, 0]) <= 0).any() or queues[0, 0] <= 0
            or queues[-1, 0] != t0_s or bin_s <= 0):
        raise ValueError('invalid causal entrance queue counters')
    times = queues[:, 0]
    cumulative = queues[:, 1:].copy()
    for entrance in range(2):
        events = np.sort([float(e['time']) for e in past_entries if int(e['lane'] == 2) == entrance])
        if ((events <= 0).any() or (events > t0_s).any()
                or not np.isfinite(events).all()):
            raise ValueError('future/noncausal insertion observation')
        cumulative[:, entrance] += np.searchsorted(events, times, side='right')
    increment = np.diff(np.vstack((np.zeros((1, 2)), cumulative)), axis=0)
    if (increment < 0).any():
        raise ValueError('insertions + queue violate entrance request conservation')
    edges = np.arange(0, t0_s, bin_s).tolist() + [float(t0_s)]
    # Require actual snapshots at every bin boundary; never interpolate counts.
    selected = np.searchsorted(times, edges[1:])
    if not np.array_equal(times[selected], np.asarray(edges[1:])):
        raise ValueError('missing entrance counter at observation bin boundary')
    counts = np.diff(np.vstack((np.zeros((1, 2)), cumulative[selected])), axis=0).astype(int)
    return dict(edges_s=edges, counts=counts.tolist(), total=counts.sum(0).tolist(),
                exposure_s=float(t0_s), final_backlog=queues[-1, 1:].tolist())


@dataclass
class DemandPlan:
    """One future-owned rate realization; edges relative to rollout t0."""
    edges_s: np.ndarray
    rates: np.ndarray

    def __post_init__(self):
        self.edges_s = np.asarray(self.edges_s, dtype=float)
        self.rates = np.asarray(self.rates, dtype=float)
        if (self.edges_s.ndim != 1 or len(self.edges_s) < 2 or self.edges_s[0] != 0
                or not np.isfinite(self.edges_s).all() or (np.diff(self.edges_s) <= 0).any()
                or self.rates.shape != (len(self.edges_s)-1, 2)
                or not np.isfinite(self.rates).all() or (self.rates < 0).any()):
            raise ValueError('invalid learned piecewise demand plan')

    def next_time(self, time_s, entrance, hazard):
        """Invert integrated intensity, including zero-rate phases exactly."""
        for i, end in enumerate(self.edges_s[1:]):
            begin = max(float(time_s), float(self.edges_s[i]))
            if end <= begin:
                continue
            rate = self.rates[i, entrance]
            mass = rate * (end - begin)
            if rate > 0 and hazard < mass:
                return begin + hazard / rate
            hazard -= mass
        return np.inf


class LearnedArrivalFit:
    """Fit artifact only; never accepts original future requests or assignments."""
    def __init__(self, artifact):
        if artifact['schema'] != 'wm_phys.observed_arrivals.v1' \
                or artifact['generator_family'] != 'piecewise_constant_poisson':
            raise ValueError('unsupported learned arrival schema/family')
        self.artifact = artifact

    @classmethod
    def from_file(cls, path):
        from .fitted_hidden import read_artifact
        artifact, sha = read_artifact(path)
        instance = cls(artifact)
        instance.sha256 = sha
        return instance

    def condition(self, world, past_entries, queue_history, *, t0_s=300.):
        """Condition the frozen generator family on a new causal observation.

        Does not read a tape or refit a family on outcomes. INT may use this at
        a new permitted observation with its observed insertions and queues.
        """
        if t0_s != self.artifact['conditioning_end_s']:
            raise ValueError('arrival conditioning time mismatch')
        observation = request_bins(past_entries, queue_history, t0_s=t0_s,
                                   bin_s=self.artifact.get('bin_s', 10.))
        instance = LearnedArrivalFit({**self.artifact,
            'worlds': {**self.artifact['worlds'], str(world): observation}})
        instance.sha256 = getattr(self, 'sha256', None)
        return instance

    def draw(self, world, *, t0_s, seed, sample, sample_rate=True):
        if t0_s != self.artifact['conditioning_end_s']:
            raise ValueError('arrival conditioning time mismatch')
        observed = self.artifact['worlds'].get(str(world))
        if observed is None:
            raise ValueError('no causal entrance counters for this world; fit them first')
        counts = np.asarray(observed['total'], dtype=float)
        exposure = float(observed['exposure_s'])
        if counts.shape != (2,) or (counts < 0).any() or not np.isfinite(counts).all() \
                or not np.isfinite(exposure) or exposure <= 0:
            raise ValueError('invalid learned request counts/exposure')
        rates = counts / exposure
        if sample_rate:
            # Jeffreys prior for a Poisson intensity: p(lambda) ∝ lambda**(-1/2).
            # One posterior rate per future, held fixed for the documented phase.
            rates = np.array([np.random.default_rng(seed_of('hidden-demand-rate', seed,
                world, sample, e)).gamma(counts[e] + .5, 1 / exposure) for e in range(2)])
        end = float(self.artifact['observed_phase_end_s']) - t0_s
        return DemandPlan(np.array([0., end]), rates[None])


class FittedEntries(PoissonEntries):
    """Conditional deterministic placement, exact learned-demand hazard clock."""
    def __init__(self, *args, demand_plans, **kwargs):
        self.demand_plans = demand_plans
        self._hazard_time = np.zeros((len(demand_plans), 2))
        super().__init__(*args, **kwargs)
        if len(demand_plans) != self.batch:
            raise ValueError('one learned demand plan per batch row required')

    def _next(self, b, e):
        previous = self._hazard_time[b, e]
        nxt = self.demand_plans[b].next_time(previous, e, self._request_rngs[b][e].exponential())
        self._hazard_time[b, e] = nxt
        return nxt - previous

    def _emit_due(self, b):
        if self.now[b] + self.dt > self.demand_plans[b].edges_s[-1]:
            raise ValueError('forecast exceeds observed/documented fitted demand phase')
        return super()._emit_due(b)
