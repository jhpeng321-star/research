"""MAIN's one-use same-cap continuation; never replay the failed mechanics seed."""
import json
from pathlib import Path

from .protocol import digest, safe_path
from .storage import append

ORIGINAL = 'build_bc4c3aae72073204'
LEDGER_SHA = 'f94a2488879d79fb433e15a57e6d99710b62147d108eb6c2b19e772feda84253'
TERMINAL = 'build_999584a8cfacaf81'
TERMINAL_LEDGER_SHA = 'be3bcd873b401cce95b14a9659f33bcc8de38823eb8a6ca5a830119ca13347ce'


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
