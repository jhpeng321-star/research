"""Persistent draws from S-RB-FIT's motion-only joint finite-support posterior."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import torch

from wms.model.driver_encoder import PARAMETER_NAMES

IDM_ORDER = ('v0', 'time_headway', 'accel', 'decel', 'min_gap')
LC_ORDER = ('lcStrategic', 'lcCooperative', 'lcSpeedGain', 'lcKeepRight', 'lcAssertive')
BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')


def read_artifact(path):
    path = Path(path)
    if any(word in str(path.resolve()).lower() for word in BAD):
        raise ValueError('prohibited physical-fit artifact path')
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw), hashlib.sha256(raw).hexdigest()


class FittedHiddenState:
    """No clipping/re-fitting/shrinkage of an already fitted joint posterior.

    LC is explicitly separate because the delivered LC candidate was not adopted.
    No cross-IDM/LC posterior covariance was estimated by FIT; these two joint
    blocks are sampled independently, with that approximation in the receipt.
    """
    def __init__(self, population, posterior, *, lane_population=None, lane_posterior=None,
                 provenance=None):
        self.population, self.posterior = population, posterior
        self.lane_population, self.lane_posterior = lane_population, lane_posterior
        self.provenance = provenance or {}
        self._cache = {}
        self._validate_headers(population, posterior, 'idm', IDM_ORDER)
        if population.get('distribution') != 'finite_support_mixture':
            raise ValueError('fitted hidden-state adapter requires joint finite support')
        self.support, self.weights = self._distribution(population, population, IDM_ORDER)
        if (lane_population is None) != (lane_posterior is None):
            raise ValueError('both LC population and posterior are required')
        if lane_population is not None:
            self._validate_headers(lane_population, lane_posterior, 'lc2013', LC_ORDER)
            self._distribution(lane_population, lane_population, LC_ORDER)

    @classmethod
    def from_directory(cls, directory, *, lane_directory=None):
        root = Path(directory)
        pop, a = read_artifact(root/'idm_population.json')
        path = root/'idm_posteriors.json.gz'
        if not path.exists():
            path = root/'idm_posteriors.json'
        post, b = read_artifact(path)
        kwargs = dict(provenance={'idm_population_sha256': a, 'idm_posterior_sha256': b})
        if lane_directory is not None:
            lp, c = read_artifact(Path(lane_directory)/'lc2013_population.json')
            ls, d = read_artifact(Path(lane_directory)/'lc2013_posteriors.json')
            kwargs.update(lane_population=lp, lane_posterior=ls)
            kwargs['provenance'].update(lc_population_sha256=c, lc_posterior_sha256=d)
        return cls(pop, post, **kwargs)

    @staticmethod
    def _validate_headers(pop, post, family, order):
        if (pop['schema'] != f'wm_phys.{family}_population.v1'
                or post['schema'] != f'wm_phys.{family}_posteriors.v1'
                or tuple(pop['parameter_order']) != order
                or tuple(post.get('parameter_order', pop['parameter_order'])) != order):
            raise ValueError('unsupported fitted posterior schema/order')

    @staticmethod
    def _distribution(pop, distribution, order):
        support = torch.as_tensor(pop['support'], dtype=torch.float64)
        if support.ndim != 2 or support.shape[1] != len(order):
            raise ValueError('invalid fitted joint support')
        indices = torch.as_tensor(distribution.get('support_indices', list(range(len(support)))))
        weights = torch.as_tensor(distribution['weights'], dtype=torch.float64)
        if (indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64)
                or (indices < 0).any() or (indices >= len(support)).any()
                or weights.shape != indices.shape or not torch.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0
                or not torch.isfinite(support).all() or (support <= 0).any()):
            raise ValueError('invalid fitted support indices/weights')
        return support[indices], weights

    def check_time(self, t0_s):
        for post in (self.posterior, self.lane_posterior):
            if post is not None and t0_s != post['conditioning_end_s']:
                raise ValueError('posterior conditioning time mismatch')

    def draw(self, world, vid, *, generator, lane=False):
        pop, post, order = ((self.lane_population, self.lane_posterior, LC_ORDER) if lane
                            else (self.population, self.posterior, IDM_ORDER))
        if pop is None:
            return {}
        if vid is not None and (int(vid) != vid or vid < 0):
            raise ValueError('present drivers require observable nonnegative numeric VID')
        distribution = pop if vid is None else post['drivers'].get(f'{int(world)}:{int(vid)}', pop)
        key = (lane, None if distribution is pop else f'{int(world)}:{int(vid)}')
        if key not in self._cache:
            self._cache[key] = self._distribution(pop, distribution, order)
        support, weights = self._cache[key]
        index = int(torch.multinomial(weights, 1, generator=generator))
        return dict(zip(order, support[index].tolist()))

    def as_population(self, params, state, mask):
        from .hidden_state import Population
        observed = Population.from_motion(params, state, mask)
        pools = []
        for b in range(state['present'].shape[0]):
            # Non-IDM coordinates remain inert legacy values, never LC proxies.
            base = torch.stack([params[n][b, 0] for n in PARAMETER_NAMES])
            pool = base.expand(len(self.support), -1).clone()
            for j, name in enumerate(IDM_ORDER):
                pool[:, PARAMETER_NAMES.index(name)] = self.support[:, j].to(pool)
            pools.append(pool)
        return Population(pools, observed.cav, [self.weights.clone() for _ in pools])
