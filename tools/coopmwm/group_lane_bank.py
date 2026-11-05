"""MINED_MOVES.jsonl -> F3 source pointers, NEVER self-certified move labels.

v1 row: {schema: 'coopmwm-mined-move-v1', arm: 'B'|'R', run_root: str,
verification_id: str, plan_sha256: hex64, replay_sources?: {B: [str], R: [str]}}.
Paths are relative to the JSONL file. Native F3 admission, immutable receipts,
effective-command checks and own-seed tapes remain required. Both arms read
the same file; matched_pools/supervision still enforce common family support.
"""
import json

from .group_lane import enabled

SCHEMA = 'coopmwm-mined-move-v1'


def inputs(cfg):
    if not enabled(cfg):
        return []
    from .track2_backend import _path
    default = _path(cfg['run_root'])/'MINED_MOVES.jsonl'
    path = _path(cfg.get('mined_moves', default))
    if not path.exists():
        if 'mined_moves' in cfg:
            raise ValueError('configured MINED_MOVES.jsonl missing')
        return []
    grouped = {}
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row, dict) or row.get('schema') != SCHEMA or row.get('arm') not in ('B', 'R')
                or not isinstance(row.get('run_root'), str) or not row['run_root']
                or not isinstance(row.get('verification_id'), str) or not row['verification_id']
                or not isinstance(row.get('plan_sha256'), str) or len(row['plan_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in row['plan_sha256'])):
            raise ValueError(f'MINED_MOVES row {number} needs a native verification/plan binding')
        def resolve(value):
            p = _path(value)
            return str(_path(p if p.is_absolute() else path.parent/p).resolve())
        root = resolve(row['run_root'])
        sources = row.get('replay_sources', {})
        if not isinstance(sources, dict) or set(sources)-{'B', 'R'} or any(
                not isinstance(v, list) or any(not isinstance(p, str) for p in v) for v in sources.values()):
            raise ValueError('MINED_MOVES replay_sources requires B/R path lists')
        converted = {arm: [resolve(p) for p in paths] for arm, paths in sources.items()}
        key = (root, json.dumps(converted, sort_keys=True))
        source = grouped.setdefault(key, dict(run_root=root, replay_sources=converted, _g5_mined=[]))
        if row not in source['_g5_mined']:
            source['_g5_mined'].append(row)
    return list(grouped.values())
