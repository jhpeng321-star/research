"""Causal continuation of the frozen motion-fitted LC2013 joint mixture.

No population fitting or runtime decision noise. A physical profile is sampled
once per driver by FittedHiddenState; the discrepancy likelihood is used ONLY
to condition profile weights on the caller's own pre-start observed motion.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from wms.control.fitted_hidden import FittedHiddenState, LC_ORDER, read_artifact
from .lc2013 import STATE_KEYS, lane_change
from .lc2013_geometry import lane_position_shift


def postmotion_snapshot(before, after, physics):
    """Restore source-lane origin, preserving SUMO's raw positionOnLane."""
    moved = torch.as_tensor(np.array(before, copy=True), dtype=torch.float64)
    after = torch.as_tensor(np.array(after, copy=True), dtype=torch.float64)
    source, dest = moved[..., 1:4].argmax(-1), after[..., 1:4].argmax(-1)
    moved[..., 0] = after[..., 0] - lane_position_shift(after[..., 0], source, dest, physics)
    moved[..., 4:6] = after[..., 4:6]
    return moved


def causal_arrays(arrays, t0_s, physics):
    """Reject future/ambiguous identities before evaluating any LC profile."""
    required = ('time_s', 'veh_kin', 'veh_mask', 'vid', 'is_cav')
    a = {key: np.asarray(arrays[key]) for key in required}
    t, k, m, ids, cav = (a[key] for key in required)
    if (not np.isfinite(t0_s) or t.ndim != 1 or len(t) < 2
            or not np.isfinite(t).all() or t[-1] != t0_s or (t > t0_s).any()
            or not np.allclose(np.diff(t), physics.dt_s, rtol=0, atol=1e-8)
            or k.ndim != 3 or k.shape[0] != len(t) or k.shape[-1] != 7
            or m.shape != k.shape[:2] or ids.shape != m.shape or cav.shape != m.shape):
        raise ValueError('causal identity-aligned LC history required')
    for row in range(len(t)):
        visible = ids[row, m[row].astype(bool)]
        if ((visible < 0).any() or not np.isfinite(visible).all()
                or (visible != visible.astype(np.int64)).any()
                or len(np.unique(visible)) != len(visible)):
            raise ValueError('unique observable LC identities required')
    # Freeze exactly FIT's left-censoring window; extra older history is not
    # silently treated as additional evidence under its learned likelihood.
    selected = t >= t0_s - 60.
    if t[selected][0] != t0_s - 60.:
        raise ValueError('LC continuation requires complete causal60s')
    return {key: value[selected] for key, value in a.items()}


def snapshots(a, physics):
    k, m, ids, cav = (a[key] for key in ('veh_kin', 'veh_mask', 'vid', 'is_cav'))
    for i in range(len(k) - 1):
        present = (m[i].astype(bool) & m[i+1].astype(bool) & (ids[i] == ids[i+1])
            & (k[i, :, 1:4].sum(-1) == 1) & (k[i+1, :, 1:4].sum(-1) == 1))
        eligible = present & ~cav[i].astype(bool) & ~cav[i+1].astype(bool)
        yield i, postmotion_snapshot(k[i], k[i+1], physics), torch.as_tensor(present), torch.as_tensor(eligible)


class LaneContinuation:
    def __init__(self, population, *, provenance=None):
        self.population = copy.deepcopy(population)
        self.provenance = {'lc_conditioning': 'caller_causal_motion.v1',
            'saved_lc_driver_id_lookup': False, **(provenance or {})}
        if (population['schema'] != 'wm_phys.lc2013_population.v1'
                or tuple(population['parameter_order']) != LC_ORDER
                or population['fit_interval_s'] != [240, 300]
                or population['fit_worlds'] != list(range(969400, 969432))
                or population['snapshot_version'] != 'v2'
                or population.get('estimation') != 'physical categorical one-hot least squares; causal symmetric-likelihood driver Bayes'):
            raise ValueError('frozen raw-physical FIT LC continuation required')
        self.support, self.weights = FittedHiddenState._distribution(population, population, LC_ORDER)
        self.active = torch.nonzero(self.weights > 0).flatten()
        self.emission = torch.as_tensor(population['observation_emission']['probabilities'], dtype=torch.float64)
        if (self.emission.shape != (3, 3, 3) or not torch.isfinite(self.emission).all()
                or (self.emission <= 0).any()
                or not torch.allclose(self.emission.sum(-1), torch.ones((3, 3), dtype=torch.float64))):
            raise ValueError('invalid fitted LC observation likelihood')

    @classmethod
    def from_directory(cls, directory):
        population, digest = read_artifact(Path(directory)/'lc2013_population.json')
        return cls(population, provenance={'lc_population_sha256': digest,
            'lc_conditioning': 'caller_causal_motion.v1', 'saved_lc_driver_id_lookup': False})

    @torch.no_grad()
    def condition(self, arrays, *, world, t0_s, physics):
        a = causal_arrays(arrays, t0_s, physics)
        ids = a['vid']
        keys = np.unique(ids[a['veh_mask'].astype(bool) & ~a['is_cav'].astype(bool)])
        profiles = self.support[self.active]
        n, slots = len(profiles), ids.shape[1]
        params = {name: profiles[:, j:j+1].expand(n, slots) for j, name in enumerate(LC_ORDER)}
        params.update({name: torch.full((n, slots), value, dtype=torch.float64)
                       for name, value in self.population['idm_parameters'].items()})
        ll = torch.zeros((len(keys), n), dtype=torch.float64)
        count = np.zeros(len(keys), dtype=np.int64)
        memory = None
        for i, moved, present, eligible in snapshots(a, physics):
            identity = torch.as_tensor(np.array(ids[i], copy=True)).expand(n, slots)
            memory = lane_change(moved.expand(n, slots, 7), present.expand(n, slots), physics,
                memory=memory, identity=identity, decision_mask=eligible.expand(n, slots), driver_params=params)
            selected = torch.nonzero(eligible).flatten()
            if not len(selected):
                continue
            source = moved[:, 1:4].argmax(-1)[selected]
            dest = torch.as_tensor(a['veh_kin'][i+1, :, 1:4].argmax(-1))[selected]
            actual = torch.where(dest == source, 0, dest + 1)
            predicted = memory['next_lane'][:, selected].T
            predicted = torch.where(predicted == source[:, None], 0, predicted + 1)
            if (actual > 2).any() or (predicted > 2).any():
                raise ValueError('LC history contains unsupported ramp-target changes')
            rows = torch.as_tensor(np.searchsorted(keys, ids[i, selected.numpy()]))
            ll.index_add_(0, rows, self.emission[source[:, None], predicted, actual[:, None]].log())
            np.add.at(count, rows.numpy(), 1)
        log_prior = (self.weights[self.active] / self.weights.sum()).log()
        probabilities = (ll + log_prior).softmax(-1)
        records = {}
        for i, vid in enumerate(keys):
            if not count[i]:
                continue
            records[int(world), int(vid)] = dict(support_indices=self.active.tolist(),
                weights=probabilities[i].tolist(), n_observations=int(count[i]), conditioning_end_s=t0_s)
        return records, dict(self.provenance, conditioning_interval_s=[t0_s-60., t0_s],
            conditioned_lc_drivers=len(records), active_lc_profiles=n,
            lc_fit_source_sha256=self.population['lc2013_source_sha256'],
            lc_runtime_source_sha256=hashlib.sha256(Path(__file__).with_name('lc2013.py').read_bytes()).hexdigest())

    def attach(self, fitted, drivers, *, t0_s, receipt=None):
        """Attach *new* LC weights after IDM's independent causal conditioning."""
        fitted.check_time(t0_s)
        records = {}
        for (world, vid), record in drivers.items():
            if record.get('conditioning_end_s') != t0_s or int(world) != world or int(vid) != vid or world < 0 or vid < 0:
                raise ValueError('invalid causal LC posterior identity/time')
            FittedHiddenState._distribution(self.population, record, LC_ORDER)
            records[f'{int(world)}:{int(vid)}'] = copy.deepcopy(record)
        posterior = dict(schema='wm_phys.lc2013_posteriors.v1', parameter_order=LC_ORDER,
            conditioning_end_s=t0_s, drivers=records)
        digest = hashlib.sha256(json.dumps(posterior, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        provenance = dict(fitted.provenance, **(receipt or self.provenance), lc_posterior_sha256=digest)
        return FittedHiddenState(copy.deepcopy(fitted.population), copy.deepcopy(fitted.posterior),
            lane_population=copy.deepcopy(self.population), lane_posterior=posterior, provenance=provenance)


@torch.no_grad()
def warm_lane_memory(arrays, *, t0_s, physics, driver_params, vehicle_ids):
    """Replay causal observations under each sampled driver's persistent settings.

    Returns memory aligned by observable identity to caller's packed t0 slots.
    Replaying observed changes resets wishes; hypothetical successes never do.
    """
    a = causal_arrays(arrays, t0_s, physics)
    ids, batch = a['vid'], vehicle_ids.shape[0]
    slots = ids.shape[1]
    device = vehicle_ids.device
    defaults = dict(zip(('accel', 'decel', 'time_headway', 'min_gap'), physics.idm()[:4]))
    defaults.update(v0=physics.v0_mps, **{name: 1. for name in LC_ORDER})
    params = {}
    # Settings are known for the caller's t0 cohort; past departed neighbors
    # use the same inherited default physics, not their hidden settings.
    for name, value in driver_params.items():
        default = defaults.get(name)
        if default is None:
            continue
        params[name] = torch.full((batch, slots), float(default), dtype=torch.float64, device=device)
    memory = None
    for i, moved, present, eligible in snapshots(a, physics):
        history_ids = torch.as_tensor(np.array(ids[i], copy=True), device=device)
        match = (history_ids[None, :, None] == vehicle_ids[:, None, :]) & (history_ids[None, :, None] >= 0)
        found, index = match.any(-1), match.long().argmax(-1)
        for name in params:
            values = driver_params[name].to(device=device, dtype=torch.float64).gather(-1, index)
            params[name] = torch.where(found, values, float(defaults[name]))
        memory = lane_change(moved.to(device).expand(batch, slots, 7), present.to(device).expand(batch, slots), physics,
            memory=memory, identity=history_ids.expand(batch, slots),
            decision_mask=eligible.to(device).expand(batch, slots), driver_params=params)
    # Apply the reset caused by the *last observed* LC at t0, not one tick late.
    final_ids = torch.as_tensor(np.array(ids[-1], copy=True), device=device)
    final_lane = torch.as_tensor(a['veh_kin'][-1, :, 1:4].argmax(-1), device=device)
    final_present = torch.as_tensor(a['veh_mask'][-1].astype(bool)
        & (a['veh_kin'][-1, :, 1:4].sum(-1) == 1), device=device)
    same = memory['lc2013_present'] & final_present & (memory['lc2013_identity'] == final_ids)
    same &= memory['lc2013_lane'] == final_lane
    match = (vehicle_ids[..., None] == final_ids) & final_present & (vehicle_ids[..., None] >= 0)
    found, index = match.any(-1), match.long().argmax(-1)
    result = {}
    for name in STATE_KEYS:
        if name not in memory:
            continue
        value = memory[name]
        if name not in ('lc2013_identity', 'lc2013_lane', 'lc2013_present'):
            value = torch.where(same, value, 0)
        value = value.gather(-1, index)
        result[name] = torch.where(found, value, 0)
    result['lc2013_present'] = found & same.gather(-1, index)
    result['lc2013_identity'] = vehicle_ids.clone()
    result['lc2013_lane'] = final_lane.expand(batch, slots).gather(-1, index)
    return result
