"""MINED_MOVES.jsonl -> F3 source pointers, NEVER self-certified move labels.

v1 row: {schema: 'coopmwm-mined-move-v1', arm: 'B'|'R', run_root: str,
verification_id: str, plan_sha256: hex64, replay_sources?: {B: [str], R: [str]}}.
Paths are relative to the JSONL file. Native F3 admission, immutable receipts,
effective-command checks and own-seed tapes remain required. Both arms read
the same file; matched_pools/supervision still enforce common family support.

walls-only clean3 rows (owner ~07:2x) extend the pointer with their own verified
situation binding (MINED_BINDING_FIELDS): the row is one per-seed SUMO-verified
joint example whose decision snapshot comes from its own sealed SUMO verification,
never from a replay-bank decision record. See binding() and
replay_bank._decision_snapshot(mined=...).
"""
import json

from .group_lane import enabled

SCHEMA = 'coopmwm-mined-move-v1'
MINED_BINDING_FIELDS = ('seed', 'situation_id', 't0', 'world', 'members', 'duration_s',
                        'profile', 'policy_profile', 'perturbation_profile', 'synergy_veh_s',
                        'synergy_lower95', 'n_reps')
MINED_BINDING_RULE = ('walls-only clean3 (owner ~07:2x launch): a MINED_MOVES row may carry its own '
                      'verified situation binding - the per-seed CRN index, situation id, decision time '
                      't0, the registered TRAIN world, the executable plan fields, the measured per-seed '
                      'synergy with its one-sided 95% lower bound and the replication count. With no '
                      'replay-bank decision link the decision snapshot comes from that row\'s own sealed '
                      'SUMO verification. A partial binding is refused; ingestion never guesses.')


def binding(row):
    """The validated verified situation binding of a MINED_MOVES row (fail closed).

    Returns the normalized binding, or None for a v1 pointer row (no binding
    fields at all: it stays a pointer into a bank that owns its decision link).
    Any PARTIAL binding raises: a row that claims its own verified situation
    must carry every binding field.
    """
    import math
    if not isinstance(row, dict):
        raise ValueError('MINED_MOVES binding must be an object')
    if not any(key in row for key in MINED_BINDING_FIELDS):
        return None
    missing = [key for key in MINED_BINDING_FIELDS if key not in row]
    if missing:
        raise ValueError('MINED_MOVES row has a partial verified situation binding; missing: '
                         +','.join(missing))
    seed, t0, duration, n_reps = row['seed'], row['t0'], row['duration_s'], row['n_reps']
    for name, value in (('seed', seed), ('t0', t0), ('duration_s', duration), ('n_reps', n_reps)):
        if type(value) is not int or value <= 0:
            raise ValueError('MINED_MOVES binding '+name+' must be a positive integer')
    situation = row['situation_id']
    if not isinstance(situation, str) or not situation:
        raise ValueError('MINED_MOVES binding needs a situation id')
    world = row['world']
    if (not isinstance(world, dict) or type(world.get('seed')) is not int
            or not 986000 <= world['seed'] <= 986071):
        raise ValueError('MINED_MOVES binding needs its registered TRAIN world seed')
    members = row['members']
    if (not isinstance(members, list) or not members or len(set(members)) != len(members)
            or any(not isinstance(v, str) or not v for v in members)):
        raise ValueError('MINED_MOVES binding needs distinct member ids')

    def matrix(name):
        value = row[name]
        if not isinstance(value, list) or len(value) != len(members):
            raise ValueError('MINED_MOVES binding '+name+' must have one row per member')
        for line in value:
            if not isinstance(line, list) or len(line) != duration:
                raise ValueError('MINED_MOVES binding '+name+' must span duration_s')
            for x in line:
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
                    raise ValueError('MINED_MOVES binding '+name+' must be finite numbers')
        return [[float(x) for x in line] for line in value]

    def stat(name):
        value = row[name]
        if not isinstance(value, dict) or str(seed) not in value:
            raise ValueError('MINED_MOVES binding '+name+' must carry its own seed entry')
        out = {}
        for key, x in value.items():
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
                raise ValueError('MINED_MOVES binding '+name+' must be finite numbers')
            out[str(key)] = float(x)
        return out

    return dict(schema=SCHEMA, arm=row.get('arm'), verification_id=row.get('verification_id'),
                plan_sha256=row.get('plan_sha256'), seed=seed, situation_id=situation, t0=t0,
                world=dict(world), members=list(members), duration_s=duration,
                profile=matrix('profile'), policy_profile=matrix('policy_profile'),
                perturbation_profile=matrix('perturbation_profile'),
                synergy_veh_s=stat('synergy_veh_s'), synergy_lower95=stat('synergy_lower95'),
                n_reps=n_reps)



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
