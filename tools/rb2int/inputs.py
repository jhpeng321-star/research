"""Explicit, hash-bound new recorded starts for a frozen evaluator.

No fitting or outcome selection. Only the passed observation registry is allowed;
the default round-1 observable loader keeps its original world restriction.
"""
from functools import lru_cache
import json
from pathlib import Path

from tools.rbeval.core import digest, load, safe


def validate_panel(path):
    panel = load(path)
    if panel['schema'] != 'rb2int-panel-inputs/1':
        raise ValueError('unknown panel schema')
    registration_path = safe(panel['registration_path'])
    registration = load(registration_path)
    if digest(registration_path) != panel['registration_sha256']:
        raise ValueError('registration binding changed')
    seeds = panel['worlds']
    purpose = panel.get('purpose', 'final_comparison')
    if purpose not in ('final_comparison', 'development_bridge'):
        raise ValueError('unregistered execution purpose')
    allowed = registration['final_comparison_worlds'] if purpose == 'final_comparison' else registration['fitting_worlds']
    if len(set(seeds)) != len(seeds) or not seeds or any(
            s not in allowed for s in seeds):
        raise ValueError('unregistered execution world')
    if purpose == 'final_comparison' and any(969400 <= s <= 969431 for s in seeds):
        raise ValueError('development cannot enter final panel')
    root = safe(panel['observations_root'])
    if digest(root/'expected_hashes.json') != panel['observation_registry_sha256']:
        raise ValueError('observable input registry changed')
    design = load(root/'data/design.json')
    if [w['seed'] for w in design['worlds']] != seeds:
        raise ValueError('observable world roster changed')
    if digest(panel['real_observables']) != panel['real_observables_sha256']:
        raise ValueError('factual receipt changed')
    release_path = safe(panel['release_path'])
    release = load(release_path)
    if digest(release_path) != panel['release_sha256'] or release['status'] != 'READY':
        raise ValueError('explicit panel release required')
    if release['registration_sha256'] != panel['registration_sha256']:
        raise ValueError('release belongs to a different registration')
    if not release['exposure_audit_passed']:
        raise ValueError('exposure audit has not cleared this execution')
    if purpose not in release['execution_purposes']:
        raise ValueError('release does not authorize this execution purpose')
    return panel


class RegisteredObservations:
    """Read the exact observable files also consumed by registered start creation."""
    def __init__(self, root, seeds):
        self.root = safe(root)
        self.seeds = frozenset(seeds)
        self.expected = load(self.root/'expected_hashes.json')

    def job(self, seed):
        if isinstance(seed, bool) or seed not in self.seeds:
            raise ValueError('world absent from explicit observable registry')
        return self.root/'data/jobs'/f's{int(seed)}_ctrl0'

    @lru_cache(maxsize=64)
    def load(self, seed):
        import numpy as np
        import torch
        job = self.job(seed)
        hashes = {}
        for name in ('start.pt', 'start_history.npz', 'result.json'):
            hashes[name] = digest(job/name)
            if hashes[name] != self.expected[f'jobs/s{int(seed)}_ctrl0/{name}']:
                raise ValueError('observable input hash differs: '+name)
        start = torch.load(job/'start.pt', map_location='cpu', weights_only=False)
        result = load(job/'result.json')
        if (start['t0'] != 300 or start['seed'] != seed or
                start['initial_state_sha256'] != result['metrics']['initial_state_sha256']):
            raise ValueError('observable world/time/state differs')
        for name in ('start.pt', 'start_history.npz'):
            if result['output_hashes'][name] != hashes[name]:
                raise ValueError('exporter input receipt differs')
        with np.load(job/'start_history.npz', allow_pickle=False) as raw:
            vids = np.array(raw['vid_index'][-1], copy=True)
        present = start['state']['present'][0].bool()
        ids = torch.as_tensor(vids)[present].long()
        if not len(ids) or (ids < 0).any() or len(ids.unique()) != len(ids) or (
                ids.max() >= len(start['vehicle_ids'])):
            raise ValueError('invalid recorded numeric VID map')
        if not all(0 < e['time'] <= 300 for e in start['past_entries']):
            raise ValueError('noncausal entrance history')
        # The frozen arrival family must receive causal observed counters, never
        # seed-reconstructed future demands or an absent-queue zero substitute.
        if 'observed_queue_history' not in start:
            raise ValueError('missing observed source counters')
        return start, ids, hashes

    def motion(self, seed):
        import numpy as np
        start, _, hashes = self.load(seed)
        path = self.job(seed)/'start_history.npz'
        with np.load(path, allow_pickle=False) as data:
            history = {name: np.array(data[name], copy=True)
                       for name in ('kin', 'mask', 'vid_index')}
        if len(history['kin']) != 241 or digest(path) != hashes['start_history.npz']:
            raise ValueError('causal motion-history contract differs')
        history['time_s'] = np.arange(241)*.5+180.
        return start, history


def install_observations(root, seeds):
    """Opt-in only, in the owned evaluator process, before any candidate future."""
    from tools.rbint import observations
    reader = RegisteredObservations(root, seeds)
    observations.load_observations = reader.load
    observations.load_motion_history = reader.motion
    return reader
