"""MAIN's one-use same-cap continuation; never replay the failed mechanics seed."""
import json
from pathlib import Path

from .protocol import digest, safe_path
from .storage import append

ORIGINAL = 'build_bc4c3aae72073204'
LEDGER_SHA = 'f94a2488879d79fb433e15a57e6d99710b62147d108eb6c2b19e772feda84253'
TERMINAL = 'build_999584a8cfacaf81'
TERMINAL_LEDGER_SHA = 'be3bcd873b401cce95b14a9659f33bcc8de38823eb8a6ca5a830119ca13347ce'
TRANSPORT = 'build_b3d13ccfae77de71'
TRANSPORT_LEDGER_SHA = '497e0c292a1acb757f4bd733fa4a60b9ad7e2853e6b16d16e8fe50c6c570ec88'


def validate(root, registration):
    root = safe_path(root)
    lines = safe_path(registration).read_text(encoding='utf-8').splitlines()
    authority = next((line for line in lines if line.startswith('- 00:29 MAIN:') and
                      "MAIN ACKNOWLEDGES S-SELMAPPO's no-retry stop under the same cap" in line and
                      'run only the original unused second mechanics episode' in line), None)
    if authority is None:
        raise PermissionError('specific MAIN same-cap continuation acknowledgement absent')
    if root.name != ORIGINAL:
        raise ValueError('continuation must bind the original stopped dispatch')
    stop = json.loads((root/'STOP.json').read_text())
    dispatch = json.loads((root/'DISPATCH.json').read_text())
    if ((Path('/proc')/str(dispatch['pid'])).exists() or
            stop['own_pid'] != dispatch['pid'] or not stop['own_pid_exited'] or
            stop['status'] != 'STOPPED_BEFORE_PHYSICS' or
            stop['charged_sim_s'] != 900 or stop['actual_sim_s'] != 0 or
            stop['training_episodes'] != 0 or stop['checkpoints'] != 0 or
            stop['ledger_sha256'] != LEDGER_SHA or
            digest(root/'training/ledger.jsonl') != LEDGER_SHA):
        raise ValueError('original stop/ledger/PID differs from acknowledged attempt')
    rows = [json.loads(line) for line in (root/'training/ledger.jsonl').read_text().splitlines()]
    if (len(rows) != 2 or rows[0]['event'] != 'START' or rows[1]['event'] != 'END' or
            any(row['id'] != 'fleet:18007998' for row in rows) or
            rows[0]['reserved_sim_s'] != 900 or rows[1]['actual_sim_s'] != 0):
        raise ValueError('continuation is not the exact unused second mechanics episode')
    return authority, rows


def inherit(out, original, registration):
    authority, rows = validate(original, registration)
    out = safe_path(out)
    with (out/'ledger.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True)+'\n')
        stream.flush()
        __import__('os').fsync(stream.fileno())
    append(out/'continuation.jsonl', dict(authority=authority, original=str(original),
                                        original_ledger_sha256=LEDGER_SHA,
                                        inherited_charge_sim_s=900, skipped_seed=18007998,
                                         only_unused_mechanics_seed=18007999))


def validate_production(root, registration):
    """Bind MAIN01:08 to the exact stopped second attempt, never another retry."""
    root = safe_path(root)
    authority = next((line for line in safe_path(registration).read_text(encoding='utf-8').splitlines()
                      if line.startswith('- 01:08 MAIN:') and
                      'ACKNOWLEDGES S-SELMAPPO production-only continuation' in line and
                      'retain both charges, no additional mechanics episode' in line and
                      '7,012,800' in line), None)
    if authority is None:
        raise PermissionError('MAIN01:08 production-only acknowledgement absent')
    if root.name != TERMINAL:
        raise ValueError('production continuation must bind the exact second stop')
    stop = json.loads((root/'STOP.json').read_text())
    dispatch = json.loads((root/'DISPATCH.json').read_text())
    if ((Path('/proc')/str(dispatch['pid'])).exists() or
            stop['own_pid'] != dispatch['pid'] or not stop['own_pid_exited'] or
            stop['status'] != 'STOPPED_AT_MECHANICS_TERMINAL' or
            stop['charged_sim_s'] != 1800 or stop['actual_sim_s'] != 900 or
            stop['training_episodes'] != 0 or stop['checkpoints'] != 0 or
            stop['result_complete'] or stop['mechanics_slots_remaining'] != 0 or
            stop['ledger_sha256'] != TERMINAL_LEDGER_SHA or
            digest(root/'training/ledger.jsonl') != TERMINAL_LEDGER_SHA or
            digest(root/'training.log') != stop['log_sha256'] or
            (root/'training/training.jsonl').exists() or
            list((root/'training').glob('checkpoint_*.pt'))):
        raise ValueError('second stop/ledger/log/PID differs from acknowledged attempt')
    rows = [json.loads(line) for line in (root/'training/ledger.jsonl').read_text().splitlines()]
    if (len(rows) != 4 or [r['event'] for r in rows] != ['START', 'END']*2 or
            [r['id'] for r in rows] != ['fleet:18007998']*2+['fleet:18007999']*2 or
            rows[0]['reserved_sim_s'] != 900 or rows[2]['reserved_sim_s'] != 900 or
            rows[1]['actual_sim_s'] != 0 or rows[3]['actual_sim_s'] != 900):
        raise ValueError('both original mechanics slots must remain consumed')
    return authority, rows


def inherit_production(out, original, registration):
    authority, rows = validate_production(original, registration)
    out = safe_path(out)
    with (out/'ledger.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True)+'\n')
        stream.flush()
        __import__('os').fsync(stream.fileno())
    append(out/'continuation.jsonl', dict(authority=authority, original=str(original),
                                        original_ledger_sha256=TERMINAL_LEDGER_SHA,
                                        inherited_charge_sim_s=1800, inherited_actual_sim_s=900,
                                         skipped_seeds=[18007998, 18007999], production_only=True))


def validate_transport(root, registration):
    """MAIN01:48 permits five unused worlds, never reconstruction/replay of the41."""
    root = safe_path(root)
    authority = next((line for line in safe_path(registration).read_text(encoding='utf-8').splitlines()
                      if line.startswith('- 01:48 MAIN ACKNOWLEDGEMENTS / DECISIONS:') and
                      'S-SELMAPPO same-cap continuation ACKNOWLEDGED' in line and
                      'skip the 41 consumed worlds' in line and
                      '5 unused worlds, then 168 x 46 episodes' in line and
                      'final-169-only' in line and '7,012,800' in line), None)
    if authority is None:
        raise PermissionError('MAIN01:48 exact transport schedule acknowledgement absent')
    if root.name != TRANSPORT:
        raise ValueError('transport continuation must bind the exact production stop')
    stop = json.loads((root/'STOP.json').read_text())
    dispatch = json.loads((root/'DISPATCH.json').read_text())
    out = root/'training'
    if ((Path('/proc')/str(dispatch['pid'])).exists() or
            stop['own_pid'] != dispatch['pid'] or not stop['own_pid_exited'] or
            stop['own_descendants_remaining'] != 0 or
            stop['status'] != 'STOPPED_AT_PRODUCTION_TRANSPORT' or
            stop['charged_sim_s'] != 38700 or stop['known_actual_sim_s'] != 30600 or
            stop['actual_upper_sim_s'] != 37800 or stop['training_updates'] != 0 or
            stop['ledger_open_reservations'] != 9 or stop['final_checkpoint'] or
            stop['evaluation_episodes'] != 0 or stop['ledger_sha256'] != TRANSPORT_LEDGER_SHA or
            digest(out/'ledger.jsonl') != TRANSPORT_LEDGER_SHA or
            digest(root/'training.log') != stop['log_sha256'] or
            digest(out/'failure.jsonl') != stop['failure_sha256'] or
            digest(out/'checkpoint_000.pt') != stop['initial_checkpoint_sha256'] or
            (out/'training.jsonl').exists() or
            [p.name for p in out.glob('checkpoint_*.pt')] != ['checkpoint_000.pt']):
        raise ValueError('production stop/ledger/checkpoint/log/PID differs from acknowledgement')
    for name, items in (('actual.jsonl', stop['worker_actual_receipts']),
                        ('result.jsonl', stop['successful_result_receipts'])):
        if len(items) != 33:
            raise ValueError('missing physical history receipts')
        for item in items:
            if not 18000000 <= item['seed'] <= 18000040:
                raise ValueError('foreign history receipt')
            if digest(out/str(item['seed'])/name) != item['receipt_sha256']:
                raise ValueError('changed physical history receipt')
    rows = [json.loads(s) for s in (out/'ledger.jsonl').read_text().splitlines()]
    starts = [r for r in rows if r['event'] == 'START']
    ids = {r['id'] for r in starts}
    expected = {f'fleet:{s}' for s in (*range(18000000, 18000041), 18007998, 18007999)}
    if (len(starts) != 43 or ids != expected or
            sum(r['reserved_sim_s'] for r in starts) != 38700):
        raise ValueError('consumed worlds differ from acknowledged41 plus both mechanics')
    return authority, stop, ids


def inherit_transport(out, original, registration):
    authority, stop, ids = validate_transport(original, registration)
    out = safe_path(out)
    # Preserve the open reservations byte-for-byte. No fictitious END/actual entries.
    with (out/'inherited_ledger.jsonl').open('xb') as stream:
        stream.write((safe_path(original)/'training/ledger.jsonl').read_bytes())
        stream.flush()
        __import__('os').fsync(stream.fileno())
    record = dict(event='HISTORY', charged_sim_s=38700, consumed_ids=sorted(ids),
                  ledger_sha256=TRANSPORT_LEDGER_SHA, known_actual_sim_s=30600,
                  actual_upper_sim_s=37800, unresolved_reservations=9,
                  unknown_actual_worlds=stop['actual_unknown_worlds'], authority=authority)
    with (out/'ledger.jsonl').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(record, sort_keys=True)+'\n')
        stream.flush()
        __import__('os').fsync(stream.fileno())
    append(out/'continuation.jsonl', dict(authority=authority, original=str(original),
           original_ledger_sha256=TRANSPORT_LEDGER_SHA, inherited_charge_sim_s=38700,
           gradient_episodes=7733, omitted_gradient_episodes=41, first_update_episodes=5,
           subsequent_updates=168, final_iteration=169, no_mechanics=True,
           evaluation_reserved_unstarted_sim_s=14400))


def initial_unchanged(model, original):
    import torch
    old = torch.load(safe_path(original)/'training/checkpoint_000.pt',
                     map_location='cpu', weights_only=False)
    if (old['iteration'] != 0 or set(old['model']) != set(model.state_dict()) or
            any(not torch.equal(value, old['model'][key])
                for key, value in model.state_dict().items()) or
            not torch.equal(old['rng'], torch.get_rng_state())):
        raise ValueError('initial model/RNG differ from the zero-update stopped attempt')
