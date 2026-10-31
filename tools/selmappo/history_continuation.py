"""MAIN02:08: retain all46 consumed production worlds; 23+23+167x46 updates."""
import json
import os
from pathlib import Path

from .continuation import TRANSPORT, TRANSPORT_LEDGER_SHA, validate_transport
from .protocol import digest, safe_path, training_world
from .storage import append

HISTORY = 'build_3351504cece84e4b'
HISTORY_LEDGER_SHA = '7ae8d3b572fa4ab22298b9f0d161790afc7de3bd0dd13ded8804134fcf36f0d4'


def validate(root, registration):
    root = safe_path(root)
    authority = next((line for line in safe_path(registration).read_text(encoding='utf-8').splitlines()
                      if line.startswith('- 02:08 MAIN:') and
                      'S-SELMAPPO revised same-cap schedule ACKNOWLEDGED' in line and
                      '23 + 23 + 167 x 46 episodes, 169 genuine updates' in line and
                      'skip all 46 consumed worlds, total 7,012,800' in line), None)
    if authority is None:
        raise PermissionError('MAIN02:08 exact same-cap schedule acknowledgement absent')
    if root.name != HISTORY:
        raise ValueError('must bind the exact HISTORY-reader stop')
    _, old_stop, old_ids = validate_transport(root.parent/TRANSPORT, registration)
    out = root/'training'
    stop = json.loads((root/'STOP.json').read_text())
    dispatch = json.loads((root/'DISPATCH.json').read_text())
    if ((Path('/proc')/str(dispatch['pid'])).exists() or
            stop['own_pid'] != dispatch['pid'] or not stop['own_pid_exited'] or
            stop['own_descendants_remaining'] != 0 or
            stop['status'] != 'STOPPED_BEFORE_PHYSICS_HISTORY_READER' or
            stop['charged_sim_s'] != 43200 or stop['new_charge_sim_s'] != 4500 or
            stop['new_actual_sim_s'] != 0 or stop['training_updates'] != 0 or
            stop['known_actual_sim_s'] != 30600 or stop['actual_upper_sim_s'] != 37800 or
            stop['new_open_reservations'] != 5 or stop['final_checkpoint'] or
            not stop['failure_before_env_constructor'] or
            stop['failed_seeds'] != list(range(18000041, 18000046)) or
            stop['ledger_sha256'] != HISTORY_LEDGER_SHA or
            digest(out/'ledger.jsonl') != HISTORY_LEDGER_SHA or
            stop['inherited_ledger_sha256'] != TRANSPORT_LEDGER_SHA or
            digest(out/'inherited_ledger.jsonl') != TRANSPORT_LEDGER_SHA or
            digest(root/'training.log') != stop['log_sha256'] or
            digest(out/'failure.jsonl') != stop['failure_sha256'] or
            digest(out/'checkpoint_000.pt') != stop['initial_checkpoint_sha256'] or
            (out/'training.jsonl').exists() or
            [p.name for p in out.glob('checkpoint_*.pt')] != ['checkpoint_000.pt']):
        raise ValueError('HISTORY stop/ledger/checkpoint/log/PID changed')
    rows = [json.loads(s) for s in (out/'ledger.jsonl').read_text().splitlines()]
    if (len(rows) != 6 or rows[0]['event'] != 'HISTORY' or
            rows[0]['charged_sim_s'] != 38700 or set(rows[0]['consumed_ids']) != old_ids or
            rows[0]['ledger_sha256'] != TRANSPORT_LEDGER_SHA or
            any(row != dict(event='START', id=f'fleet:{seed}', world=training_world(seed),
                reserved_sim_s=900) for row, seed in zip(rows[1:], range(18000041, 18000046)))):
        raise ValueError('exact five retained reservations changed')
    for seed in range(18000041, 18000046):
        if not (out/str(seed)).is_dir() or any((out/str(seed)).iterdir()):
            raise ValueError('pre-physics episode directory changed')
    return authority, old_stop, old_ids | {f'fleet:{s}' for s in range(18000041, 18000046)}


def inherit(out, original, registration):
    authority, old_stop, ids = validate(original, registration)
    out, original = safe_path(out), safe_path(original)
    for target, source in (('inherited_ledger.jsonl', 'ledger.jsonl'),
                           ('inherited_transport_ledger.jsonl', 'inherited_ledger.jsonl')):
        with (out/target).open('xb') as stream:
            stream.write((original/'training'/source).read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    record = dict(event='HISTORY', charged_sim_s=43200, consumed_ids=sorted(ids),
                  ledger_sha256=HISTORY_LEDGER_SHA, known_actual_sim_s=30600,
                  actual_upper_sim_s=37800, unresolved_reservations=14,
                  unknown_actual_worlds=old_stop['actual_unknown_worlds'],
                  known_zero_actual_worlds=list(range(18000041, 18000046)),
                  transport_ledger_sha256=TRANSPORT_LEDGER_SHA, authority=authority)
    append(out/'ledger.jsonl', record)
    append(out/'continuation.jsonl', dict(authority=authority, original=str(original),
           original_ledger_sha256=HISTORY_LEDGER_SHA, inherited_charge_sim_s=43200,
           gradient_episodes=7728, omitted_gradient_episodes=46,
           update_episode_counts=[23, 23]+[46]*167, final_iteration=169,
           no_mechanics=True, evaluation_reserved_unstarted_sim_s=14400))
