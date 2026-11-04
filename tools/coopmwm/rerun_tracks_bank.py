"""Observable, source-bound bank batches for the frozen-base cooperation learner."""
from copy import deepcopy
import math
import re

from .exact_grade import _seal


OPTIONS = dict(statistics='per_seed_v1', target_space='effective',
               mode='separate_full_pass', step_mode='kl_targeted', kl_cap=.045,
               effective_loss='tools.coopmwm.bank_imitation.pair_loss',
               base_frozen=True, separate_layer_optimizer=True)


def training_batch(bank, augmented, *, arm, encoder):
    from . import moves
    rows, ids = [], set()
    sources = {e['id']: e for e in bank}
    for example in bank+augmented:
        e = deepcopy(example)
        ident, source = e.get('id'), e.get('source_id')
        if (not isinstance(ident, str) or ident in ids or e.get('arm') != arm or
                source not in sources or sources[source]['arm'] != arm or
                type(e.get('seed')) is not int or e['seed'] < 0 or
                e.get('seed') != sources[source]['seed']):
            raise ValueError('REFUSED: bank training needs unique own-arm source/seed identity')
        ids.add(ident)
        obs = e.get('observation', {})
        if set(obs) != {'time_s', 'own_sensing', 'v2v_reports', 'cav_ids', 'bounds'}:
            raise ValueError('REFUSED: bank training needs its own observable context')
        if any(r['time_s'] > obs['time_s']-1 for r in obs['v2v_reports']):
            raise ValueError('REFUSED: bank training has current/future partner report')
        weight = e.get('weight')
        if (e.get('statistics') != OPTIONS['statistics'] or type(weight) not in (int, float)
                or not math.isfinite(weight) or weight <= 0):
            raise ValueError('REFUSED: bank training needs registered positive per-seed weight')
        provenance = e.get('provenance', {})
        for field in ('verification_sha256', 'weight_sha256', 'effective_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', provenance.get(field, '')):
                raise ValueError('REFUSED: bank training lacks '+field)
        features = encoder.encode(obs)
        sample = e.get('effective_sample', {})
        context = sample.get('context', {})
        members = context.get('members', [])
        if (not members or len(set(members)) != len(members) or
                not set(members).issubset(obs['cav_ids']) or
                context.get('observation') != obs or context.get('features') != features or
                set(context) != {'members', 'observation', 'features'}):
            raise ValueError('REFUSED: effective sample lost source-state/feature association')
        behaviour = sample.get('behaviour', {})
        proposed = behaviour.get('proposal_members', [])
        target = moves.from_plan(e['plan'])
        if (behaviour.get('executed') is not True or not proposed or
                set(proposed) != set(target.members) or not set(proposed).issubset(members) or
                len(sample.get('action', [])) != len(members) or
                set(sample.get('binding_limits', {})) != set(proposed)):
            raise ValueError('REFUSED: incomplete effective-action supervision')
        values = list(sample['action'])+list(sample['binding_limits'].values())
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError('REFUSED: nonfinite effective-action supervision')
        plan_sha = _seal(dict(plan=e['plan']), 'sha256')['sha256']
        if sample.get('plan_sha256') != plan_sha:
            raise ValueError('REFUSED: effective supervision differs from source plan')
        rows.append(dict(id=ident, source_id=source, arm=arm, seed=e['seed'],
            statistics=e['statistics'], weight=weight, provenance=provenance,
            observation=obs, features=features, target=target, plan=e['plan'],
            effective_sample=sample))
    return rows


def batch_receipt(batch):
    # JSON-native source plans bind move targets without assuming sibling Move internals.
    return _seal(dict(options=OPTIONS, rows=[{k: v for k, v in r.items() if k != 'target'}
                                           for r in batch]), 'batch_sha256')
