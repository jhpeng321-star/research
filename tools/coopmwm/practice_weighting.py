"""T2H opt-in practice share and T2G restart-manifest adapter (CPU only).

No new keys are emitted for legacy configs. Augmentation producers write a sealed
replay_bank/<ARM>/PRACTICE_AUGMENTED.json with schema SCHEMA, arm, and rows.
Each row names example_id, parent_example_id, verification_id, state_id,
seed_index, snapshot_sha256 (after CRN replication), plan_sha256,
confirmation_sha256, and wm_cooperative=True (proposal provenance only).
Both parent and augmentation must be own admitted cooperative examples; the
augmentation needs its own sealed all8 confirmation and SUMO receipt. Merely
perturbing a plan at the same restart state does not create additional practice capacity.
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


def _verified_augmentation(root, arm, item):
    """Validate existing own all8 admission, plan, and original snapshot binding.

    Read fixed own-bank receipt paths only; never launch verification or follow a
    producer-supplied path. A parent's admission/WM proposal is not evidence.
    """
    from . import exact_grade as xg, replay_bank as rb, sumo_verify as sv
    from . import verify_perseed as vp
    from .protocol import digest
    import torch

    bank = rb.bank_root(root, arm)
    vid = item.get('verification_id')
    if any(not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value)
           for value in (vid, item.get('state_id'))):
        raise ValueError('verified own-bank all8 practice augmentation identifiers required')
    try:
        c = json.loads((bank/vp.RECORDS/(vid+'.json')).read_text(encoding='utf-8'))
        xg.verify_sealed(c, 'confirmation_sha256')
        if (c.get('schema') != vp.SCHEMA or c.get('verify_mode') != vp.ALL8
                or c.get('bank_rule') != 'per_seed_v1' or c.get('arm') != arm
                or c.get('verification_id') != vid or c.get('admitted') is not True
                or c['confirmation_sha256'] != item.get('confirmation_sha256')):
            raise ValueError('all8 confirmation binding/admission mismatch')
        matches = [e for e in c['per_seed_v1']['examples']
                   if e.get('example_id') == item['example_id']]
        if len(matches) != 1:
            raise ValueError('all8 augmented example missing or ambiguous')
        ex = matches[0]
        index = ex['candidate_index']
        plan_sha = item.get('plan_sha256')
        if (ex.get('admitted') is not True or ex.get('status') != 'admitted'
                or ex.get('cooperative') is not True
                or ex.get('seed_index') != item['seed_index']
                or type(index) is not int or index < 0
                or ex.get('plan_sha256') != plan_sha or digest(ex['plan']) != plan_sha
                or not any(p.get('index') == index and p.get('plan_sha256') == plan_sha
                           for p in c['candidates'])):
            raise ValueError('all8 augmented plan/seed admission mismatch')
        receipt = json.loads((bank/'sumo_verification'/'verifications'/(vid+'.json'))
                             .read_text(encoding='utf-8'))
        xg.verify_sealed(receipt, 'receipt_sha256')
        snap = torch.load(bank/'snapshots'/(item['state_id']+'.pt'), weights_only=False)
        # A rejected first seed can still yield an admitted seed-2 example.
        if (receipt.get('status') not in ('accepted', 'rejected')
                or receipt.get('verification_id') != vid
                or receipt.get('episode', {}).get('arm') != arm
                or c['first_seed'].get('verification_id') != vid
                or c['first_seed'].get('receipt_sha256') != receipt['receipt_sha256']
                or receipt['candidate_plan_sha256'][index] != plan_sha
                or receipt.get('snapshot_sha256') != sv.snapshot_sha(snap)):
            raise ValueError('all8 augmented SUMO receipt/state mismatch')
    except (OSError, KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError('verified own-bank all8 practice augmentation required: '+str(error)) from error


def augmented_examples(root, arm, parents):
    """Return (distinct own-bank all8-verified cooperative restart rows, manifest hash).

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
    manifest_ids = {r.get('example_id') for r in record['rows'] if isinstance(r, dict)
                    and isinstance(r.get('example_id'), str)}
    seen_ids = set()
    # Admitted all8 augmentations can already appear in the ordinary bank reader.
    seen_states = {_state_key(r) for r in parents if r['example_id'] not in manifest_ids}
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
                or ident == item.get('parent_example_id')
                or not isinstance(item.get('verification_id'), str)
                or not re.fullmatch(r'[A-Za-z0-9_-]+', item['verification_id'])
                or not isinstance(item.get('snapshot_sha256'), str)
                or not re.fullmatch(r'[0-9a-f]{64}', item['snapshot_sha256'])):
            raise ValueError('invalid own-bank cooperative practice augmentation')
        _verified_augmentation(root, arm, item)
        row = dict(verification_id=item['verification_id'], example_id=ident,
                   seed_index=item['seed_index'], salt=item['verification_id'],
                   state_id=state, fallback=False, source='augmented',
                   parent_example_id=item['parent_example_id'],
                   snapshot_sha256=item['snapshot_sha256'],
                   plan_sha256=item['plan_sha256'],
                   confirmation_sha256=item['confirmation_sha256'])
        # Verify both the snapshot binding and the existing CRN derivation now,
        # before a plan is sealed or any simulation is dispatched.
        ps.load_snapshot(root, arm, row)
        seen_ids.add(ident)
        key = _state_key(row)
        if key not in seen_states:
            seen_states.add(key)
            out.append(row)
    return out, record['practice_augmented_sha256']
