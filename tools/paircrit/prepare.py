"""Export the existing cycle0 CRN cohort. Fit native rule before policy labels."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from wms.control.breakdown import fit_native_rule, fit_otsu_rule, merge_speed, rule_sha256

OWN = Path('/root/autodl-tmp/c4/paircrit20261026')
ROOT = Path('/root/main14_loop')
DENY = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'vtype')


def safe(path):
    p = Path(path)
    assert not any(s in str(p).lower() or s in str(p.resolve()).lower() for s in DENY)
    assert p.resolve().is_relative_to(ROOT) or p.resolve().is_relative_to(OWN)
    return p


def load_trace(entry):
    assert entry['split_role'] == 'train' and not 964000 <= entry['seed'] <= 964199
    path = safe(ROOT / 'data/pairs_c00.a1' / entry['dir'])
    values = {key: np.load(safe(path / (key + '.npy')), allow_pickle=False)
              for key in ('veh_kin', 'veh_mask', 'is_cav', 'vid', 'time_s')}
    kin = torch.from_numpy(values['veh_kin'])
    lane = torch.where(kin[..., 1:4].sum(-1) > 0, kin[..., 1:4].argmax(-1), -1)
    state = {'s_m': kin[..., 0], 'v_mps': kin[..., 4], 'a_mps2': kin[..., 5],
             'length_m': kin[..., 6], 'lane': lane,
             'present': torch.from_numpy(values['veh_mask']).bool(),
             'is_cav': torch.from_numpy(values['is_cav']).bool()}
    times = values['time_s']
    start = entry['branch']['branch_t_s']
    use = np.flatnonzero(times >= start)
    states = {k: v[use] for k, v in state.items()}
    speed = merge_speed(states).numpy()
    t0 = use[0]
    digest = hashlib.sha256()
    for key in ('veh_kin', 'veh_mask', 'vid', 'is_cav'):
        digest.update(np.ascontiguousarray(values[key][t0]).tobytes())
    return times[use] - start, speed, digest.hexdigest()


def main():
    p = safe(ROOT / 'returns/returns_c00.a1.npz')
    with np.load(p, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    receipt = json.loads(safe(p.with_suffix('.receipt.json')).read_text())
    manifest = json.loads(safe(ROOT / 'data/pairs_c00.a1/C4_EPISODES_MANIFEST.json').read_text())
    entries = {(e['pair_key'], e['branch']['kind']): e for e in manifest['episodes']
               if e.get('branch', {}).get('kind') in ('B', 'POLICY')}
    records = [e for e in receipt['entries'] if 'pair_key' in e]
    native_j, speeds, times, hashes, rows = [], [], [], [], []
    for i, record in enumerate(records):
        b = entries[record['pair_key'], 'B']
        policy = entries[record['pair_key'], 'POLICY']
        assert b['seed'] == policy['seed'] == int(data['seeds'][i])
        assert b['branch']['branch_t_s'] == policy['branch']['branch_t_s']
        assert b['branch']['horizon_s'] == policy['branch']['horizon_s'] == 300
        path = safe(Path(b['source']) / 'summary.json')
        raw = path.read_bytes()
        d = json.loads(raw)
        assert d['seed'] == b['seed'] and d['requested_population'] == data['requested_population'][i]
        assert not (d['collisions'] or d['teleports'] or d['dropped_vehicle_steps'])
        native_j.append(d['ledger']['J_vehicle_seconds'])
        t, v, digest = load_trace(b)
        times.append(t)
        speeds.append(v)
        hashes.append(hashlib.sha256(raw).hexdigest())
        rows.append({'pair_key': record['pair_key'], 'seed': b['seed'], 'horizon_s': 300,
                     'observable_start_sha256': digest, 'same_saved_start': True,
                     'native_summary_sha256': hashes[-1], 'policy_summary_sha256': record['sha256']})
    fit = np.array([int(hashlib.sha256(f'965140:{int(s)}'.encode()).hexdigest(), 16) % 5 != 0
                    for s in data['seeds']])
    source_sha = hashlib.sha256(''.join(h for h, f in zip(hashes, fit) if f).encode()).hexdigest()
    try:
        rule = fit_native_rule(np.array(speeds)[fit], source_sha)
    except ValueError as error:
        rule = fit_otsu_rule(np.array(speeds)[fit], source_sha)
        rule['gaussian_identification_refused'] = str(error)
    # Publish before policy trace labels are even constructed.
    (OWN / 'NATIVE_RULE.json').write_text(json.dumps(rule, indent=2))
    np.savez_compressed(OWN / 'native_traces.npz', speed=np.array(speeds), time=np.array(times))
    np.savez_compressed(OWN / 'policy_returns.npz', **data)
    native = {**data, 'returns': -np.array(native_j), 'source_kind': np.array('native')}
    native.pop('policy_checkpoint')
    np.savez_compressed(OWN / 'native_returns.npz', **native)
    (OWN / 'policy_returns.receipt.json').write_text(json.dumps(receipt, indent=2))
    native_receipt = {**receipt, 'kind': 'B', 'source_kind': 'native', 'policy_checkpoint': '',
        'entries': [{'summary': str(Path(entries[row['pair_key'], 'B']['source']) / 'summary.json'),
                     'sha256': row['native_summary_sha256'], 'pair_key': row['pair_key'], 'J': float(j)}
                    for row, j in zip(rows, native_j)]}
    (OWN / 'native_returns.receipt.json').write_text(json.dumps(native_receipt, indent=2))
    proof = {'schema': 'paircrit_pairs/1', 'split': 'TRAIN', 'pairs': rows,
        'policy_sha256': hashlib.sha256((OWN / 'policy_returns.npz').read_bytes()).hexdigest(),
        'native_sha256': hashlib.sha256((OWN / 'native_returns.npz').read_bytes()).hexdigest(),
        'policy_checkpoint_sha256': str(data['policy_checkpoint'].item()),
        'manifest_sha256': hashlib.sha256(safe(ROOT / 'data/pairs_c00.a1/C4_EPISODES_MANIFEST.json').read_bytes()).hexdigest(),
        'note': 'common pair_key saved-state lineage; numeric initial arrays independently checked before scoring'}
    (OWN / 'PAIRING.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps({'rule': rule, 'rule_sha256': rule_sha256(rule), 'n_pairs': len(rows),
                      'fit_worlds': sorted(set(data['seeds'][fit].tolist()))}, indent=2))


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
