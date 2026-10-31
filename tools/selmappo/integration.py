"""Validate peer integration BEFORE setup, allocation or simulator import."""
import json
import re

from .protocol import (BLOCKS, CADENCE, CAP, LIBRARY, PROTOCOL, audit_registration,
                       digest, safe_path)


def registered(registration, design_sha, *, variant='fleet'):
    if not re.fullmatch('[0-9a-f]{64}', design_sha):
        raise ValueError('invalid design SHA')
    lines = safe_path(registration).read_text(encoding='utf-8').splitlines()
    prefix = '- S-SELMAPPO DESIGN REGISTERED ' if variant == 'fleet' else '- S-SELMAPPO2 DESIGN REGISTERED '
    if not any(line.startswith(prefix) and
               f'SHA256 {design_sha}' in line and f'{variant}_cap={CAP}' in line and
               f'protocol={PROTOCOL}' in line for line in lines):
        raise PermissionError('exact v2 fleet design/budget not registered')


def load(path, registration, c1_root, design_sha, *, variant='fleet'):
    registered(registration, design_sha, variant=variant)
    if not audit_registration(registration, variant=variant, design_sha=design_sha)['passed']:
        raise ValueError('seed allocation conflict')
    record = json.loads(safe_path(path).read_text(encoding='utf-8'))
    if (record.get('schema') != 'selmappo.integration/2' or
            record.get('status') != 'CPU_INTEGRATION_PASS' or
            record.get('protocol') != PROTOCOL or
            record.get('cadence_s') != CADENCE or
            record.get('library') != list(LIBRARY) or
            record.get('design_sha256') != design_sha):
        raise ValueError('six-controller integration receipt absent/mismatched')
    root = safe_path(c1_root)
    files = record['files']
    if not files:
        raise ValueError('empty integration source binding')
    if not {'tools/c1switch/sumo.py', 'tools/c1switch/core.py',
            'tools/c1switch/design.py', 'tools/c1switch/storage.py'} <= files.keys():
        raise ValueError('C1SWITCH library/observer sources are not bound')
    for name, expected in files.items():
        path = safe_path(root/name)
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError('integration source binding mismatch')
    module = record['wall_module']
    if module != 'tools.c1switch.library':
        raise ValueError('invalid S-WALLSW module')
    if not {module.replace('.', '/')+'.py', 'WALLSW_HANDOFF.md',
            'traffic_shepherd/c4/switchable_wall.py', 'src/wms/control/wall.py'} <= files.keys():
        raise ValueError('WALL adapter/handoff not hash-bound')
    return record


def evaluation_panel(path, expected_sha, registration=None):
    if digest(path) != expected_sha:
        raise ValueError('evaluation panel hash mismatch')
    data = json.loads(safe_path(path).read_text(encoding='utf-8'))
    if data.get('protocol') != PROTOCOL:
        raise ValueError('wrong evaluation protocol')
    if registration is not None:
        lines = safe_path(registration).read_text(encoding='utf-8').splitlines()
        peer_sha = data.get('design_sha256')
        if (not isinstance(peer_sha, str) or not re.fullmatch('[0-9a-f]{64}', peer_sha) or
                not any('S-C1SWITCH2 DESIGN REGISTERED ' in line and peer_sha in line for line in lines)):
            raise PermissionError('evaluation panel peer design is not registered')
    worlds = data['worlds']
    if len(worlds) != 16 or len({w['seed'] for w in worlds}) != 16:
        raise ValueError('exactly16 paired evaluation worlds required')
    for world in worlds:
        seed = world['seed']
        if (type(seed) is not int or 964000 <= seed <= 964199 or seed//1000 in (962, 977) or
                any(lo <= seed <= hi for lo, hi in BLOCKS.values())):
            raise ValueError('unsafe evaluation seed')
    return worlds
