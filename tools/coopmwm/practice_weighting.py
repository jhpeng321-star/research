"""T2H opt-in practice share and T2G restart-manifest adapter (CPU only).

No new keys are emitted for legacy configs. Augmentation producers write a sealed
replay_bank/<ARM>/PRACTICE_AUGMENTED.json with schema SCHEMA, arm, and rows.
Each row names example_id, parent_example_id, verification_id, state_id,
seed_index, snapshot_sha256 (after CRN replication), and wm_cooperative=True.
The parent must be an own admitted cooperative example; the new decision link
and snapshot must also be in that arm's bank. Merely perturbing a plan at the
same restart state does not create additional practice capacity.
"""
import json
import math
from pathlib import Path
import re

KEYS = ('practice_weighting', 'practice_weighted_fraction')
FRACTIONS = (0.50,)  # T2H registered alternative: 12/24, versus legacy 6/24
SCHEMA = 'coopmwm-practice-augmented-v1'


def options(cfg):
    on = cfg.get('practice_weighting', False)
    fraction = cfg.get('practice_weighted_fraction', FRACTIONS[0])
    if type(on) is not bool:
        raise ValueError('practice_weighting must be a bool')
    if (type(fraction) is not float or not math.isfinite(fraction)
            or fraction not in FRACTIONS):
        raise ValueError('registered practice_weighted_fraction required')
    if on and cfg.get('practice_starts') is not True:
        raise ValueError('practice_weighting requires practice_starts=True')
    return dict(practice_weighting=on, practice_weighted_fraction=fraction)


def effective_fraction(cfg):
    values = options(cfg)
    return (values['practice_weighted_fraction'] if values['practice_weighting']
            else cfg.get('practice_fraction', 0.25))


def _state_key(row):
    # A seed-1 salt is unused; seed>=2 salts determine the actual CRN restart.
    return (row['state_id'], int(row['seed_index']),
            row['salt'] if int(row['seed_index']) > 1 else '')


def augmented_examples(root, arm, parents):
    """Return (distinct own-bank WM-cooperative restart rows, manifest hash).

    Missing manifest means no augmentation yet. Present invalid manifests refuse
    before dispatch; no foreign bank, arbitrary file path, or imported label is read.
    """
    from . import exact_grade as xg, practice_starts as ps
    path = Path(root)/'replay_bank'/arm/'PRACTICE_AUGMENTED.json'
    if not path.exists():
        return [], None
    record = json.loads(path.read_text(encoding='utf-8'))
    xg.verify_sealed(record, 'practice_augmented_sha256')
    if record.get('schema') != SCHEMA or record.get('arm') != arm:
        raise ValueError('practice augmentation schema/arm mismatch')
    if not isinstance(record.get('rows'), list):
        raise ValueError('practice augmentation rows must be a list')
    own = {r['example_id']: r for r in parents}
    links = ps._links(root, arm)
    seen_ids = set(own)
    seen_states = {_state_key(r) for r in parents}
    out = []
    for item in record['rows']:
        if not isinstance(item, dict):
            raise ValueError('practice augmentation row must be an object')
        parent = own.get(item.get('parent_example_id'))
        link = links.get(item.get('verification_id'))
        state = item.get('state_id')
        ident = item.get('example_id')
        if (parent is None or link is None or item.get('wm_cooperative') is not True
                or type(item.get('seed_index')) is not int
                or item['seed_index'] != parent['seed_index']
                or not isinstance(state, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', state)
                or link.get('state_id') != state
                or not isinstance(ident, str) or not ident or ident in seen_ids
                or not isinstance(item.get('snapshot_sha256'), str)
                or not re.fullmatch(r'[0-9a-f]{64}', item['snapshot_sha256'])):
            raise ValueError('invalid own-bank cooperative practice augmentation')
        row = dict(verification_id=item['verification_id'], example_id=ident,
                   seed_index=item['seed_index'], salt=item['verification_id'],
                   state_id=state, fallback=False, source='augmented',
                   parent_example_id=item['parent_example_id'],
                   snapshot_sha256=item['snapshot_sha256'])
        # Verify both the snapshot binding and the existing CRN derivation now,
        # before a plan is sealed or any simulation is dispatched.
        ps.load_snapshot(root, arm, row)
        seen_ids.add(ident)
        key = _state_key(row)
        if key not in seen_states:
            seen_states.add(key)
            out.append(row)
    return out, record['practice_augmented_sha256']
