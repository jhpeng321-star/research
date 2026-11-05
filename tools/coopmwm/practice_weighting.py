"""T2H opt-in practice share and T2G restart-manifest adapter (CPU only).

No new keys are emitted for legacy configs. Augmentation producers write a sealed
replay_bank/<ARM>/PRACTICE_AUGMENTED.json with schema SCHEMA, arm, and rows.
Each row names example_id, parent_example_id, verification_id, state_id,
seed_index, snapshot_sha256 (after CRN replication), and wm_cooperative=True.
The parent AND child must be own admitted cooperative examples. Each child also
needs its own sealed SUMO receipt bound to its plan, snapshot, world and arm,
and the parent's pre-launch allocation/ledger provenance (<=200k sim-s).
WM-only children and v1 manifests cannot enter practice.
"""
import json
import math
from pathlib import Path
import re

KEYS = ('practice_weighting', 'practice_weighted_fraction')
FRACTIONS = (0.50,)  # T2H registered alternative: 12/24, versus legacy 6/24
SCHEMA = 'coopmwm-practice-augmented-v2'
VERIFIED_CHILDREN = True


def verify_child(root, arm, row, snapshot, prelaunch):
    """Recheck evidence even for a resumed frozen plan, before dispatch."""
    from . import replay_bank as rb, exact_grade as xg, sumo_verify as sv
    from .protocol import digest
    evidence = row.get('sumo_evidence') or {}
    relative = evidence.get('receipt_path', '')
    parts = relative.replace('\\', '/').split('/')
    if (not relative or Path(relative).is_absolute() or ':' in relative or '..' in parts
            or any(s in relative.lower() for s in ('holdout', 'heldout', 'cal_split', 'chunk5',
                'j4b', 'inc1', 'hetdata_truth', 'vtype', 'driver', 'vehicle_params'))
            or 'sealed' in [s.lower() for s in parts]):
        raise ValueError('invalid augmented child receipt path')
    bank = rb.bank_root(root, arm).resolve()
    registration = prelaunch if isinstance(prelaunch, dict) else {}
    ledger_root = Path(registration.get('ledger_root', root)).resolve()
    artifact_root = Path(root).resolve()
    if ledger_root != artifact_root and artifact_root.parent != ledger_root:
        raise ValueError('prelaunch ledger must own these smoke artifacts')
    receipt_path = (bank/relative).resolve()
    parent_bank = rb.bank_root(ledger_root, arm).resolve()
    if not (receipt_path.is_relative_to(bank) or
            (artifact_root.parent == ledger_root and receipt_path.is_relative_to(parent_bank))):
        raise ValueError('augmented child receipt escapes own bank')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    xg.verify_sealed(receipt, 'receipt_sha256')
    if (receipt.get('status') != 'accepted'
            or receipt.get('receipt_sha256') != evidence.get('receipt_sha256')
            or receipt.get('verification_id') != evidence.get('verification_id', row['verification_id'])
            or receipt.get('snapshot_sha256') != sv.snapshot_sha(snapshot)
            or receipt.get('episode', {}).get('arm') != arm
            or receipt['episode'].get('world') != snapshot['episode']['world']
            or digest(receipt.get('chosen_plan')) != evidence.get('plan_sha256')):
        raise ValueError('augmented child SUMO receipt binding mismatch')
    confirmation = rb.confirmations(root, arm).get(row['verification_id'], {})
    examples = (confirmation.get('per_seed_v1') or {}).get('examples', [])
    child = next((e for e in examples if e.get('example_id') == row['example_id']), {})
    tape = child.get('tape') or {}
    from .rerun_cpus import guard_path
    tape_result = Path(guard_path(Path(tape.get('branches_dir', ''))/
        str(tape.get('verification_id', ''))/str(tape.get('branch', {}).get('branch', ''))/'RESULT.json'))
    expected_result = Path(guard_path(receipt_path.parent.parent/'branches'/
        receipt['verification_id']/str(tape.get('branch', {}).get('branch', ''))/'RESULT.json'))
    if (not child.get('admitted') or child.get('cooperative') is not True
            or child.get('seed_index') != row['seed_index']
            or not tape_result.is_file() or tape_result.resolve() != expected_result.resolve()
            or tape.get('verification_id') != receipt['verification_id']
            or tape.get('branch', {}).get('result_sha256') !=
                receipt['branches'][receipt['chosen']+1].get('result_sha256')
            or (child.get('plan') is not None and digest(child['plan']) != evidence['plan_sha256'])
            or (child.get('plan_id') is not None
                and child['plan_id'] != receipt['chosen_plan']['id'])):
        raise ValueError('augmented child needs its own admitted cooperative SUMO example')
    if (not isinstance(prelaunch, dict) or not prelaunch.get('provenance')
            or type(prelaunch.get('allocation_sim_s')) is not int
            or not 0 < prelaunch['allocation_sim_s'] <= 200000
            or not isinstance(prelaunch.get('phase'), str) or not prelaunch['phase']):
        raise ValueError('augmented child prelaunch budget provenance required')
    charges = [json.loads(line) for line in (ledger_root/'CHARGES.jsonl').read_text().splitlines()]
    charges = [c for c in charges if c.get('phase') == prelaunch['phase']]
    if (any(type(c.get('sim_s')) is not int or c['sim_s'] < 0 for c in charges)
            or len({c['id'] for c in charges}) != len(charges)
            or sum(c['sim_s'] for c in charges) > prelaunch['allocation_sim_s']):
        raise ValueError('augmented child prelaunch allocation exceeded/invalid ledger')
    # Every policy/candidate branch actually graded by this receipt must have
    # been charged in the parent's registered pre-launch phase.
    ids = evidence.get('charge_ids')
    by_id = {c['id']: c for c in charges}
    scheme = evidence.get('charge_scheme', 'legacy_unqualified')
    if scheme == 'native_directory_v2':
        from .augmentation_charges import receipt_ids
        expected_ids = set(receipt_ids(receipt_path, receipt))
    elif scheme == 'legacy_unqualified':
        expected_ids = {f'{receipt["verification_id"]}_{b["branch"]}' for b in receipt['branches']}
        if evidence.get('adopted_ledger_binding_sha256'):
            resume = json.loads((ledger_root/'AUGMENTATION_RESUME.json').read_text())
            xg.verify_sealed(resume, 'receipt_sha256')
            if (resume['receipt_sha256'] != evidence['adopted_ledger_binding_sha256'] or not any(
                    Path(item['path']).resolve() == receipt_path and item['receipt_sha256'] == receipt['receipt_sha256']
                    and set(item['charge_ids']) == expected_ids for item in resume['completed_receipts'])):
                raise ValueError('completed branch adoption does not bind original charge proof')
    else:
        raise ValueError('unregistered augmented charge namespace scheme')
    if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
            or any(i not in by_id for i in ids)
            or set(ids) != expected_ids
            or type(receipt.get('branch_sim_s')) is not int or receipt['branch_sim_s'] <= 0
            or any(by_id[i]['sim_s'] != receipt['branch_sim_s'] for i in ids)
            or sum(by_id[i]['sim_s'] for i in ids) != receipt.get('charged_sim_s')):
        raise ValueError('augmented child missing prelaunch SUMO branch charges')


def options(cfg):
    on = cfg.get('practice_weighting', False)
    fraction = cfg.get('practice_weighted_fraction', FRACTIONS[0])
    if cfg.get('track2') is True and isinstance(on, dict):
        # T2C's approved launch profile uses a structured registration; legacy
        # callers retain the original strict boolean API.
        from .rerun_cpus import TRACK_SLOTS
        if on != TRACK_SLOTS['practice_weighting']:
            raise ValueError('Track 2 practice registration differs')
        fraction, on = float(on['fraction']), on['enabled']
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


def augmented_examples(root, arm, parents, *, deduplicate=True):
    """Return (distinct own-bank SUMO-verified restart rows, manifest hash).

    Missing manifest means no augmentation yet. Present invalid manifests refuse
    before dispatch; no foreign bank, arbitrary file path, or imported label is read.
    Breadth visitors use deduplicate=False to retain every validated child's
    provenance; restart-context deduplication happens only after that binding.
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
    child_ids = {r.get('example_id') for r in record['rows'] if isinstance(r, dict)}
    seen_ids = set(own)-child_ids
    seen_states = {_state_key(r) for r in parents if r['example_id'] not in child_ids}
    out = []
    for item in record['rows']:
        if not isinstance(item, dict):
            raise ValueError('practice augmentation row must be an object')
        parent = own.get(item.get('parent_example_id'))
        if parent is None:
            parent = _bound_training_parent(root, arm, item.get('parent_example_id'), record.get('prelaunch'))
        link = links.get(item.get('verification_id'))
        state = item.get('state_id')
        ident = item.get('example_id')
        if (parent is None or item.get('parent_example_id') in child_ids
                or link is None or item.get('wm_cooperative') is not True
                or type(item.get('seed_index')) is not int
                or (item['seed_index'] != parent['seed_index'] and
                    not _native_seed_mapping(root, arm, item, parent, link))
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
                   snapshot_sha256=item['snapshot_sha256'],
                   sumo_evidence=item.get('sumo_evidence'), prelaunch=record.get('prelaunch'))
        # Verify both the snapshot binding and the existing CRN derivation now,
        # before a plan is sealed or any simulation is dispatched.
        ps.load_snapshot(root, arm, row)
        seen_ids.add(ident)
        key = _state_key(row)
        if not deduplicate or key not in seen_states:
            seen_states.add(key)
            out.append(row)
    return out, record['practice_augmented_sha256']


def _bound_training_parent(root, arm, example_id, prelaunch):
    """Imported same-arm training-bank parents remain excluded from practice.

    Their *children* may enter only after the existing native all8 proof. The
    parent itself must be bound by the real full-bank reader and its immutable
    snapshot, not an arbitrary external example or an invented practice link.
    """
    from . import replay_bank as rb, practice_starts as ps, exact_grade as xg, sumo_verify as sv
    from .rerun_cpus import guard_path, sha
    artifact = Path(root).resolve()
    registration = prelaunch if isinstance(prelaunch, dict) else {}
    owner = Path(registration.get('ledger_root', root)).resolve()
    if owner != artifact and artifact.parent != owner:
        raise ValueError('training parent must belong to this experiment ledger')
    entry = None
    for suffix in ('_v3', ''):
        path = owner/f'AUGMENTATION_PARENTS_{arm}{suffix}.json'
        if not path.is_file():continue
        binding = json.loads(path.read_text())
        xg.verify_sealed(binding, 'parent_bank_sha256')
        if (binding.get('arm') != arm or binding.get('reader') != 'replay_bank.bank_pairs:cooperative_only:effective' or
                'augmentationParentBankSHA'+binding['parent_bank_sha256'] not in (owner/'AUTHORITY.md').read_text()):
            raise ValueError('unregistered/foreign training parent bank')
        entry = next((e for e in binding['parents'] if e['row']['example_id'] == example_id), None)
        if entry is not None:break
    if entry is None:
        return None
    row = entry['row']
    link = ps._links(root, arm).get(row['verification_id'], {})
    if not link.get('augmentation_parent_only') or link.get('state_id') != row['state_id']:
        return None
    receipt = entry['source_receipt'];xg.verify_sealed(receipt, 'receipt_sha256')
    snapshot_path = Path(guard_path(rb.bank_root(root, arm)/'snapshots'/(row['state_id']+'.pt')))
    if sha(snapshot_path) != entry['base_snapshot_file_sha256']:
        raise ValueError('registered training parent snapshot changed')
    snapshot = ps.load_snapshot(root, arm, row)
    if (snapshot['episode'].get('role') != 'train' or snapshot['episode'].get('arm') != arm or
            not 986000 <= snapshot['episode']['world']['seed'] <= 986071 or
            receipt['episode'].get('arm') != arm or sv.snapshot_sha(snapshot) != entry['snapshot_sha256']):
        raise ValueError('training parent role/arm/snapshot provenance mismatch')
    confirmation = rb.confirmations(root, arm).get(row['verification_id'], {})
    example = next((e for e in (confirmation.get('per_seed_v1') or {}).get('examples', [])
                    if e.get('example_id') == example_id), {})
    if (not example.get('admitted') or example.get('cooperative') is not True or
            example.get('seed_index') != row['seed_index']):
        raise ValueError('training parent must be own admitted cooperative per-seed evidence')
    return row


def _native_seed_mapping(root, arm, child, parent, link):
    """A new all8 decision numbers its own CRNs from one, independently of parent.

    An explicit native decision link and actual parent snapshot must bind this
    renumbering; a changed seed_index by itself remains invalid. The existing
    verify_child check still validates the child's actual snapshot, own-seed
    receipt, full tape, confirmation, plan and complete branch charges.
    """
    from . import practice_starts as ps, sumo_verify as sv
    mapping = (link or {}).get('augmentation', {})
    sha = child.get('source_snapshot_sha256')
    if (child.get('seed_index') not in (1, 2) or not sha or
            mapping.get('parent_example_id') != parent['example_id'] or
            mapping.get('source_snapshot_sha256') != sha):
        return False
    return sv.snapshot_sha(ps.load_snapshot(root, arm, parent)) == sha
