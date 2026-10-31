"""Identity-bound pre-t0 observations for fitted drivers and learned arrivals.

Uses the existing observable exporter, including its recorded numeric VID map.
Never substitutes packed-slot ordinals for vehicle identities or reads new tapes.
"""
from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import torch

from tools.rbeval.core import digest


@lru_cache(maxsize=20)
def load_observations(seed):
    if isinstance(seed, bool) or int(seed) != seed or not 969420 <= seed <= 969439:
        raise ValueError('only registered observable starts are supported')
    root = Path('/root/wmcalib')
    job = root/'data/jobs'/f's{int(seed)}_ctrl0'
    expected = json.loads((root/'expected_hashes.json').read_text())
    hashes = {}
    for name in ('start.pt', 'start_history.npz', 'result.json'):
        hashes[name] = digest(job/name)
        if hashes[name] != expected[f'jobs/s{int(seed)}_ctrl0/{name}']:
            raise ValueError('observable start hash mismatch: '+name)
    start = torch.load(job/'start.pt', map_location='cpu', weights_only=False)
    result = json.loads((job/'result.json').read_text())
    if start['t0'] != 300 or start['seed'] != seed or start['initial_state_sha256'] != result['metrics']['initial_state_sha256']:
        raise ValueError('observable world/time/state binding differs')
    for name in ('start.pt', 'start_history.npz'):
        if result['output_hashes'][name] != hashes[name]:
            raise ValueError('observable exporter receipt differs')
    with np.load(job/'start_history.npz', allow_pickle=False) as raw:
        vids = np.array(raw['vid_index'][-1], copy=True)
    present = start['state']['present'][0].bool()
    ids = torch.as_tensor(vids)[present].long()
    if (ids < 0).any() or len(ids.unique()) != len(ids) or ids.max() >= len(start['vehicle_ids']):
        raise ValueError('invalid recorded numeric VID map')
    if not all(0 < e['time'] <= 300 for e in start['past_entries']):
        raise ValueError('noncausal entrance history')
    return start, ids, hashes


def fitted_inputs(initial, seed, initial_sha256, arrival_fit):
    start, ids, hashes = load_observations(seed)
    if start['initial_state_sha256'] != initial_sha256:
        raise ValueError('fitted inputs refer to a different registered start')
    selected = start['state']['present'][0].bool()
    n = int(selected.sum())
    mask = initial['present'].bool()
    if not mask[:, :n].all() or mask[:, n:].any():
        raise ValueError('fitted inputs require the registered packed present cohort')
    for name in ('s_m', 'v_mps', 'lane', 'length_m', 'is_cav'):
        expected = start['state'][name][:, selected].to(initial[name])
        if not torch.equal(initial[name][:, :n], expected.expand_as(initial[name][:, :n])):
            raise ValueError('observable cohort differs: '+name)
    vehicle_ids = torch.full_like(initial['lane'], -1, dtype=torch.long)
    vehicle_ids[:, :n] = ids.to(vehicle_ids)
    conditioned = arrival_fit.condition(int(seed), start['past_entries'],
                                        start['observed_queue_history'], t0_s=300.)
    return vehicle_ids, conditioned, dict(observable_hashes=hashes,
        identity_source='recorded start_history numeric vid_index, not packed slot',
        conditioning_end_s=300., present_identities=n)


def load_motion_history(seed):
    start, _, hashes = load_observations(seed)
    path = Path('/root/wmcalib/data/jobs')/f's{int(seed)}_ctrl0/start_history.npz'
    with np.load(path, allow_pickle=False) as data:
        history = {name: np.array(data[name], copy=True)
                   for name in ('kin', 'mask', 'vid_index')}
    # The registered exporter stores241 half-second frames ending at observable
    # t0=300 (180..300), independently bound by its saved artifact hashes.
    if len(history['kin']) != 241 or digest(path) != hashes['start_history.npz']:
        raise ValueError('registered causal history contract changed')
    history['time_s'] = np.arange(241)*.5+180.
    return start, history
