"""Opt-in fitted uncertainty of physical latent inputs, never output/noise repair.

Artifacts are learned offline from permitted motion and counters. Incumbent
settings remain identity-keyed, entrants ordinal-keyed; one uncertain entrant
mixture and one demand intensity are drawn per future, not per physics tick.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from .arrival_fit import DemandPlan, LearnedArrivalFit
from .fitted_hidden import read_artifact, IDM_ORDER
from .futures import seed_of


class SpreadCalibration:
    def __init__(self, artifact, *, sha256=None):
        if (artifact.get('schema') != 'wm_phys.spread_calibration.v1'
                or artifact.get('fit_worlds') != list(range(969400, 969432))
                or artifact.get('conditioning_end_s') != 300
                or artifact.get('learned_corrections') is not False):
            raise ValueError('unsupported/non-development physical calibration')
        self.artifact, self.sha256 = copy.deepcopy(artifact), sha256
        parameters = artifact['parameters']
        retention = parameters['posterior_retention']
        variance = parameters['arrival_variance_multiplier']
        alpha = parameters['entrant_concentration']
        for value in (retention, variance, alpha):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int))
                                      or not math.isfinite(value) or value < 0):
                raise ValueError('invalid learned uncertainty parameter')
        if retention is None or retention > 1 or variance is None:
            raise ValueError('invalid learned retention/rate variance')
        self.retention, self.variance, self.alpha = retention, variance, alpha

    @classmethod
    def from_file(cls, path):
        artifact, digest = read_artifact(path)
        return cls(artifact, sha256=digest)

    def receipt(self):
        return dict(version=self.artifact['version'], sha256=self.sha256,
                    parameters=copy.deepcopy(self.artifact['parameters']),
                    fit_worlds=self.artifact['fit_worlds'])

    def hidden(self, fitted):
        if fitted.provenance.get('idm_population_sha256') != self.artifact['idm_population_sha256']:
            raise ValueError('spread calibration/population mismatch')
        return CalibratedHidden(fitted, self)

    def arrivals(self, fitted):
        if fitted.sha256 != self.artifact['arrivals_sha256']:
            raise ValueError('spread calibration/arrival-family mismatch')
        return CalibratedArrivals(fitted, self)


class CalibratedHidden:
    """A wrapper preserves the source finite support and shrunk causal posteriors."""
    def __init__(self, base, calibration):
        self.base, self.calibration = base, calibration
        self.provenance = dict(base.provenance, spread_calibration=calibration.receipt())
        self._weights, self._counts = {}, {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def with_motion_posteriors(self, drivers, *, t0_s):
        return type(self)(self.base.with_motion_posteriors(drivers, t0_s=t0_s), self.calibration)

    def dense_weights(self, world, vid):
        key = (int(world), None if vid is None else int(vid))
        if key not in self._weights:
            prior = self.base.weights.double().cpu()
            prior = prior/prior.sum()
            distribution = self.base.posterior['drivers'].get(f'{int(world)}:{int(vid)}', {}) \
                if vid is not None else {}
            if 'weights' not in distribution:
                weights = prior.clone()
            else:
                # Distribution indices refer to the original full population support.
                indices = torch.tensor(distribution.get('support_indices', range(len(prior))), dtype=torch.long)
                values = torch.tensor(distribution['weights'], dtype=torch.float64)
                if len(indices.unique()) != len(indices):
                    raise ValueError('duplicate physical support indices')
                dense = torch.zeros_like(prior)
                dense[indices] = values/values.sum()
                weights = prior+self.calibration.retention*(dense-prior)
                weights /= weights.sum()
            self._weights[key] = weights
        return self._weights[key]

    def draw(self, world, vid, *, generator, lane=False):
        if lane:
            return self.base.draw(world, vid, generator=generator, lane=True)
        if vid is not None and (isinstance(vid, bool) or int(vid) != vid or vid < 0):
            raise ValueError('observable nonnegative integer VID required')
        index = int(torch.multinomial(self.dense_weights(world, vid), 1, generator=generator))
        return dict(zip(IDM_ORDER, self.base.support[index].tolist()))

    def entrant_weights(self, world, *, seed):
        prior = self.base.weights.detach().double().cpu().numpy()
        prior = prior/prior.sum()
        alpha = self.calibration.alpha
        if alpha is None:  # Exact learned fixed-population endpoint.
            return torch.from_numpy(prior.copy())
        if int(world) not in self._counts:
            counts = np.zeros_like(prior)
            prefix = f'{int(world)}:'
            for key, distribution in self.base.posterior['drivers'].items():
                if key.startswith(prefix) and 'weights' in distribution:
                    counts += self.dense_weights(world, int(key[len(prefix):])).numpy()
            self._counts[int(world)] = counts
        shape = alpha*prior+self._counts[int(world)]
        positive = shape > 0
        if not positive.any():  # No observed motion and zero empirical prior count.
            return torch.from_numpy(prior.copy())
        weights = np.zeros_like(shape)
        rng = np.random.default_rng(seed_of('rb2-entrant-mixture', seed, int(world)))
        weights[positive] = rng.dirichlet(shape[positive])
        return torch.from_numpy(weights)


class CalibratedArrivals(LearnedArrivalFit):
    def __init__(self, base, calibration):
        super().__init__(base.artifact)
        self.calibration = calibration
        self.sha256 = base.sha256

    def condition(self, *args, **kwargs):
        return type(self)(super().condition(*args, **kwargs), self.calibration)

    def draw(self, world, *, t0_s, seed, sample, sample_rate=True):
        if not sample_rate:
            return super().draw(world, t0_s=t0_s, seed=seed, sample=sample, sample_rate=False)
        observed = self._observation(world, t0_s=t0_s)
        counts = np.asarray(observed['total'], dtype=float)
        exposure = float(observed['exposure_s'])
        variance = self.calibration.variance
        if variance == 1:
            return super().draw(world, t0_s=t0_s, seed=seed, sample=sample)
        if variance == 0:
            rates = (counts+.5)/exposure
        else:
            rates = np.array([np.random.default_rng(seed_of('hidden-demand-rate', seed,
                world, sample, e)).gamma((counts[e]+.5)/variance, variance/exposure)
                for e in range(2)])
        return DemandPlan([0., float(self.artifact['observed_phase_end_s'])-t0_s], rates[None])

    def observation_receipt(self, *args, **kwargs):
        return dict(super().observation_receipt(*args, **kwargs),
                    spread_calibration=self.calibration.receipt())
