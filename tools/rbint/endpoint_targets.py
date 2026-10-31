"""Evaluation-only observed60s targets; never imported by a model/forecaster."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ARMS = ('native', 'ctrl0', 'mappo', 'pi', 'circles')
BAD = ('holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'sealed',
       'hetdata_truth', 'vehicle_params', 'driver_params', 'assignment', 'vtype')


def safe(path):
    path = Path(path)
    if any(token in str(path.resolve()).lower() for token in BAD):
        raise ValueError('prohibited endpoint input path')
    return path


def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def key(s, speed, lane, cav, junction=False):
    return float(np.float32(s)), float(np.float32(speed)), int(lane), bool(cav), bool(junction)


def causal_lane_carry(kin, present, vids):
    """Last OBSERVED normal lane per persistent VID, through the supplied t0."""
    last = {}
    for t in range(len(kin)):
        lane = np.where(kin[t, :, 1:4].sum(-1) == 1, kin[t, :, 1:4].argmax(-1), -1)
        ids = vids[t] if vids.ndim == 2 else vids
        for slot in np.flatnonzero(present[t] & (lane >= 0) & (ids >= 0)):
            last[int(ids[slot])] = int(lane[slot])
    ids = vids[-1] if vids.ndim == 2 else vids
    return np.array([last.get(int(v), -1) for v in ids])


def match_targets(start, initial_ids, initial_kin, initial_mask, initial_vid,
                  initial_cav, future_kin, future_mask, future_vid, future_cav):
    return match_targets_with_carry(start, initial_ids, initial_kin, initial_mask, initial_vid,
        initial_cav, future_kin, future_mask, future_vid, future_cav, initial_carry=None)


def match_targets_with_carry(start, initial_ids, initial_kin, initial_mask, initial_vid,
                            initial_cav, future_kin, future_mask, future_vid, future_cav,
                            initial_carry):
    state = start['state']
    selected = state['present'][0].bool().cpu().numpy()
    junction = state.get('junction', torch.zeros_like(state['present']))[0].bool().cpu().numpy()
    source_lane = state['lane'][0].cpu().numpy().copy()
    if initial_carry is None:
        source_lane[junction] = -1
    observed = {}
    raw_lane = np.where(initial_kin[:, 1:4].sum(-1) == 1,
                        initial_kin[:, 1:4].argmax(-1), -1)
    raw_junction = raw_lane < 0
    if initial_carry is not None:
        raw_lane = np.where(raw_junction, initial_carry, raw_lane)
    for slot in np.flatnonzero(initial_mask):
        ident = key(initial_kin[slot, 0], initial_kin[slot, 4], raw_lane[slot], initial_cav[slot], raw_junction[slot])
        observed.setdefault(ident, []).append(slot)
    current_ids = np.asarray(initial_ids)[selected]
    names = [str(start['vehicle_ids'][int(vid)]) for vid in current_ids]
    if len(set(names)) != len(names):
        raise ValueError('duplicate observed t0 vehicle name')
    rows = []
    for original_slot, name in zip(np.flatnonzero(selected), names):
        cav = bool(state['is_cav'][0, original_slot])
        ident = key(state['s_m'][0, original_slot], state['v_mps'][0, original_slot],
                    source_lane[original_slot], cav, junction[original_slot])
        found = observed.get(ident, [])
        if len(found) != 1:
            raise ValueError('t0 state matching must be EXACT and unique: '+name)
        vid = int(initial_vid[found[0]])
        if vid < 0:
            raise ValueError('invalid converted observable VID')
        future_slots = np.flatnonzero(future_mask & (future_vid == vid))
        if len(future_slots) > 1:
            raise ValueError('duplicate future VID')
        position, reason = None, 'absent_at60'
        if len(future_slots):
            slot = int(future_slots[0])
            if bool(future_cav[slot]) != cav:
                raise ValueError('observed ownership changed for the same vehicle')
            if future_kin[slot, 1:4].sum() == 1:
                position, reason = float(future_kin[slot, 0]), None
            else:
                reason = 'junction_carried_position'
        rows.append(dict(vehicle_id=name, is_hdv=not cav, s_m=position, censoring=reason))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root, output = safe(args.root), safe(args.out)
    if not output.parent.is_dir() or output.exists():
        raise ValueError('new output in an existing parent required')
    episodes = []
    for seed in range(969420, 969440):
        manifest_path = safe(root/f'world_{seed}.json')
        manifest = json.loads(manifest_path.read_text())
        job = safe(manifest['ctrl0']['job'])
        if job.name != f's{seed}_ctrl0':
            raise ValueError('unexpected observable start job')
        receipt = json.loads(safe(job/'result.json').read_text())
        for name in ('start.pt', 'start_history.npz'):
            if sha(job/name) != receipt['output_hashes'][name]:
                raise ValueError('observable start artifact hash differs')
        start = torch.load(safe(job/'start.pt'), map_location='cpu', weights_only=False)
        if (start['seed'] != seed or start['t0'] != 300.
                or start['initial_state_sha256'] != receipt['metrics']['initial_state_sha256']):
            raise ValueError('observable start clock/world differs')
        with np.load(safe(job/'start_history.npz'), allow_pickle=False) as history:
            ids = np.array(history['vid_index'][-1], copy=True)
        for arm in ARMS:
            name = manifest[arm]['episode']
            if not name.startswith(f'w{seed}m') or not name.endswith('__'+arm) or '/' in name or '\\' in name:
                raise ValueError('unexpected registered episode name')
            ep = safe(root/'episodes'/name)
            meta = json.loads(safe(ep/'episode.json').read_text())
            if meta['seed'] != seed or meta['arm'] != arm:
                raise ValueError('factual episode identity differs')
            files = {n:safe(ep/(n+'.npy')) for n in ('time_s', 'veh_kin', 'veh_mask', 'vid', 'is_cav')}
            arrays = {n:np.load(p, mmap_mode='r', allow_pickle=False) for n,p in files.items()}
            ticks = [np.flatnonzero(arrays['time_s'] == t) for t in (300., 360.)]
            if any(len(t) != 1 for t in ticks):
                raise ValueError('need exact recorded300/360s frames')
            a, b = [int(t[0]) for t in ticks]
            vid = arrays['vid']
            carry = causal_lane_carry(arrays['veh_kin'][:a+1], arrays['veh_mask'][:a+1].astype(bool),
                                     vid[:a+1] if vid.ndim == 2 else vid)
            rows = match_targets_with_carry(start, ids, arrays['veh_kin'][a], arrays['veh_mask'][a].astype(bool),
                vid[a] if vid.ndim == 2 else vid, arrays['is_cav'][a].astype(bool),
                arrays['veh_kin'][b], arrays['veh_mask'][b].astype(bool),
                vid[b] if vid.ndim == 2 else vid, arrays['is_cav'][b].astype(bool), carry)
            episodes.append(dict(seed=seed, arm=arm, initial_sha256=start['initial_state_sha256'],
                start_sha256=sha(job/'start.pt'), history_sha256=sha(job/'start_history.npz'),
                manifest_sha256=sha(manifest_path), metadata_sha256=sha(ep/'episode.json'),
                files_sha256={n:sha(p) for n,p in files.items()}, vehicles=rows))
    result = dict(schema='wmphys-factual-endpoints/1', horizon_s=60, t0_s=300,
        source_sha256=sha(Path(__file__)), root=str(root), episodes=episodes,
        matching='unique exact observed s/speed/lane-junction/ownership, then persistent converted VID',
        evaluation_only=True, model_inputs=False, training=False, sumo_runs=0)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print('FACTUAL ENDPOINTS', len(episodes), 'episodes', str(output))


if __name__ == '__main__':
    main()
