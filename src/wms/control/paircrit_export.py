"""Bind existing real POLICY/B exports to common saved starts; never runs SUMO."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import torch

from .real_returns14 import NPZRealReturnProvider
from .start_states import checked_path


def export_receipt(source, policy_returns, native_returns, output):
    root, output = checked_path(source), checked_path(output)
    policy, native = NPZRealReturnProvider(policy_returns), NPZRealReturnProvider(native_returns)
    manifest = json.loads(checked_path(root / 'C4_EPISODES_MANIFEST.json').read_text(encoding='utf-8'))
    entries = {(e.get('pair_key'), e.get('branch', {}).get('kind')): e
        for e in manifest['episodes'] if e.get('split_role') == 'train'}
    receipts = [json.loads(checked_path(p.path.with_suffix('.receipt.json')).read_text(encoding='utf-8'))
                for p in (policy, native)]
    rows = [[r for r in d['entries'] if 'pair_key' in r and not r.get('skipped')] for d in receipts]
    if len(rows[0]) != len(rows[1]) or len(rows[0]) != policy._returns.numel():
        raise ValueError('paired real exports are incomplete')
    if any(not torch.equal(v, native._states.get(k)) for k, v in policy._states.items()):
        raise ValueError('paired real initial state arrays differ')
    proof = []
    for i, (p, b) in enumerate(zip(*rows)):
        if not p['pair_key'] or p['pair_key'] != b['pair_key']:
            raise ValueError('paired real row order/saved-state IDs differ')
        both = [entries[p['pair_key'], kind] for kind in ('POLICY', 'B')]
        branch_p, branch_b = [e['branch'] for e in both]
        for field in ('base_world', 'base_episode', 'branch_t_s', 'horizon_s'):
            if branch_p.get(field) is None or branch_p[field] != branch_b.get(field):
                raise ValueError('paired saved-state lineage mismatch')
        if both[0]['seed'] != both[1]['seed'] or both[0]['seed'] != policy.provenance['seeds'][i]:
            raise ValueError('paired common seed mismatch')
        digests = []
        for e, r in zip(both, (p, b)):
            if hashlib.sha256(checked_path(r['summary']).read_bytes()).hexdigest() != r['sha256']:
                raise ValueError('paired summary changed since export')
            episode = checked_path(root / e['dir'])
            times = np.load(checked_path(episode / 'time_s.npy'), allow_pickle=False)
            tick = int(np.searchsorted(times, e['branch']['branch_t_s']))
            if tick >= len(times) or times[tick] != e['branch']['branch_t_s']:
                raise ValueError('paired saved-start tick missing')
            digest = hashlib.sha256()
            for name in ('veh_kin', 'veh_mask', 'vid', 'is_cav'):
                a = np.load(checked_path(episode / (name + '.npy')), mmap_mode='r', allow_pickle=False)
                digest.update(np.ascontiguousarray(a[tick]).tobytes())
            digests.append(digest.hexdigest())
        if digests[0] != digests[1]:
            raise ValueError('paired raw observable starts/vehicle IDs differ')
        proof.append({'pair_key': p['pair_key'], 'seed': int(both[0]['seed']),
            'horizon_s': float(branch_p['horizon_s']), 'same_saved_start': True,
            'observable_start_sha256': digests[0],
            'policy_summary_sha256': p['sha256'], 'native_summary_sha256': b['sha256']})
    doc = {'schema': 'paircrit_pairs/1', 'split': 'TRAIN', 'pairs': proof,
        'policy_sha256': hashlib.sha256(policy.path.read_bytes()).hexdigest(),
        'native_sha256': hashlib.sha256(native.path.read_bytes()).hexdigest(),
        'policy_checkpoint_sha256': policy.provenance['policy_checkpoint']}
    from .paircrit import PairedProvider
    # Persist only after all pair proofs succeed; never overwrite prior evidence.
    with output.open('x', encoding='utf-8') as stream:
        json.dump(doc, stream, indent=2)
    PairedProvider(policy, native, output, doc['policy_checkpoint_sha256'])
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'policy-returns', 'native-returns', 'output'):
        ap.add_argument('--' + name, required=True)
    args = ap.parse_args(argv)
    doc = export_receipt(args.source, args.policy_returns, args.native_returns, args.output)
    print(json.dumps({'pairs': len(doc['pairs']), 'output': args.output}))


if __name__ == '__main__':
    main()
