"""Bounded, outcome-blind inputs and canonical executable grid (no model imports)."""
import hashlib
import json
import math
from pathlib import Path

FIELDS = {'vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m'}
BLOCKED = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
           'hetdata_truth', 'vehicle_params', 'vtype')
LANE_FAMILIES = ('yield_lane', 'open_gap_merge')
LONG_FAMILIES = ('merge_timed_wall', 'gap_open', 'ramp_assist', 'ease_off')
HORIZONS = (60, 120, 300, 600)
U0_SHA = '5ea2296470c2af2855f9e292b03eeaa62bafdfec08146712d7347d7ce117d734'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def checked_path(value):
    p = Path(value)
    for text in (str(p), str(p.resolve())):
        if any(word in text.lower() for word in BLOCKED):
            raise ValueError('restricted path')
    return p


def file_sha(path):
    h = hashlib.sha256()
    with checked_path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(checked_path(path).read_text(encoding='utf-8'))


def check_sources(manifest):
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('explicit nonempty source SHA manifest required')
    for path, expected in manifest.items():
        if file_sha(path) != expected:
            raise ValueError('source hash mismatch: ' + path)


def validate_query(query):
    """Envelope supplied by MAIN, never reconstructed from diagnostic outcomes.

    base_continuation is the actual frozen-u0 sampler state AFTER the decision
    at t0, plus that decision's own residuals. It contains no future samples.
    source_sha256 binds the original captured observable query/snapshot bytes.
    """
    if set(query) != {'record', 'ego', 'source_sha256', 'base_continuation'}:
        raise ValueError('query envelope whitelist')
    r = query['record']
    required = {'state_id', 'world_id', 'time_s', 'rows', 'group', 'commands',
                'policy_residuals', 'history'}
    if set(r) != required:
        raise ValueError('observable record whitelist; provide causal history explicitly')
    if type(r['world_id']) is not int or not 986000 <= r['world_id'] <= 986071:
        raise ValueError('registered primary TRAIN worlds only')
    if type(r['time_s']) is not int or not 300 <= r['time_s'] <= 600:
        raise ValueError('query must leave the complete H600 inside the 1200s episode')
    rows = r['rows']
    if not rows or any(set(row) != FIELDS for row in rows):
        raise ValueError('six-field sensor whitelist')
    by = {row['vid']: row for row in rows}
    if len(by) != len(rows) or query['ego'] not in by or not by[query['ego']]['is_cav']:
        raise ValueError('distinct identities and an observed CAV ego required')
    origin = by[query['ego']]['s_m']
    for row in rows:
        if (not isinstance(row['vid'], str) or type(row['is_cav']) is not bool
                or type(row['lane']) is not int or row['lane'] not in (0, 1, 2)
                or any(type(row[k]) not in (int, float) or not math.isfinite(row[k])
                       for k in ('s_m', 'v_mps', 'length_m'))
                or row['v_mps'] < 0 or row['length_m'] <= 0
                or abs(row['s_m'] - origin) > 100):
            raise ValueError('invalid or nonlocal sensor row')
    group = r['group']
    if len(group) != 2 or len(set(group)) != 2 or not all(by.get(v, {}).get('is_cav') for v in group):
        raise ValueError('explicit ordered two-CAV group required')
    cavs = {v for v, row in by.items() if row['is_cav']}
    for name in ('commands', 'policy_residuals'):
        if set(r[name]) != cavs or any(not math.isfinite(x) for x in r[name].values()):
            raise ValueError('complete finite current CAV commands/residuals required')
    b = query['base_continuation']
    if set(b) != {'checkpoint', 'checkpoint_sha256', 'deterministic',
                  'generator_state_after_t0', 'memory_after_t0'}:
        raise ValueError('base continuation whitelist')
    checked_path(b['checkpoint'])
    if b['checkpoint_sha256'] != U0_SHA or type(b['deterministic']) is not bool:
        raise ValueError('exact frozen-u0 checkpoint and sampling mode required')
    state = b['generator_state_after_t0']
    if not isinstance(state, list) or not state or any(type(x) is not int or not 0 <= x <= 255 for x in state):
        raise ValueError('captured CPU Torch generator state required')
    if (set(b['memory_after_t0']) != cavs
            or any(not math.isfinite(x) for x in b['memory_after_t0'].values())):
        raise ValueError('captured post-t0 AR memory for each CAV required')
    expected_current = ({v: 0. for v in cavs} if b['deterministic'] else b['memory_after_t0'])
    if r['policy_residuals'] != expected_current:
        raise ValueError('captured t0 residuals disagree with zero-actor u0 sampler memory/mode')
    sha = query['source_sha256']
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('captured input SHA required')
    # Validate history without importing the model or reading any other file.
    previous = -math.inf
    for frame in r['history']:
        if set(frame) != {'time_s', 'rows', 'entrance_counts'} or frame['entrance_counts'] is not None:
            raise ValueError('only sensed history; no source counters inferred from the full scene')
        t = frame['time_s']
        if not r['time_s'] - 60 <= t <= r['time_s'] or t <= previous:
            raise ValueError('noncausal history')
        previous = t
        for row in frame['rows']:
            if set(row) != FIELDS:
                raise ValueError('history row whitelist')
        # Past sensing must retain the same ego's actual presence and radius.
        past = {row['vid']: row for row in frame['rows']}
        if len(past) != len(frame['rows']) or query['ego'] not in past:
            raise ValueError('history lacks its actual sensor ego')
        s = past[query['ego']]['s_m']
        if any(abs(row['s_m'] - s) > 100 for row in frame['rows']):
            raise ValueError('nonlocal history')
    if r['history'] and (r['history'][-1]['time_s'] != r['time_s']
                         or r['history'][-1]['rows'] != rows):
        raise ValueError('history must end at the identical current roster')
    return query


def grid(query, families=LONG_FAMILIES, onsets=(0, 10, 30), holds=(20, 60), depths=(.3, 1.)):
    """At most 60 joint candidates + both solos; deterministic geometry only.

    Lane candidates retain canonical G5 instructions but are refused by the
    current longitudinal-only evaluator, never silently scored without lanes.
    """
    validate_query(query)
    if not families or len(set(families)) != len(families) or set(families) - set(LONG_FAMILIES + LANE_FAMILIES):
        raise ValueError('unknown/duplicate grid families')
    if (not onsets or not holds or not depths or len(onsets) > 3 or len(holds) > 3 or len(depths) > 2
            or any(type(x) is not int or not 0 <= x <= 30 for x in onsets)
            or any(type(x) is not int or not 20 <= x <= 90 for x in holds)
            or any(not .3 <= x <= 1. for x in depths)):
        raise ValueError('bounded registered onset/hold/depth grid required')
    r = query['record']; members = list(r['group']); by = {x['vid']: x for x in r['rows']}
    out = [dict(id='policy'), dict(id='MAPPO')]; rejected = []
    for family in families:
        roles = members
        if family in ('gap_open', 'yield_lane', 'open_gap_merge'):
            mains = [v for v in members if by[v]['lane'] == 1 and -100 <= by[v]['s_m'] <= 75.44]
            ramps = [v for v in members if by[v]['lane'] == 2 and -100 <= by[v]['s_m'] <= 75.44]
            if len(mains) != 1 or len(ramps) != 1:
                rejected.append(dict(family=family, reason='requires observed in-zone mainline/ramp roles')); continue
            roles = mains + ramps
            if family == 'open_gap_merge' and by[roles[0]]['s_m'] >= by[roles[1]]['s_m']:
                rejected.append(dict(family=family, reason='mainline yielder must be behind ramp')); continue
        if family == 'ramp_assist':
            if sorted(by[v]['lane'] for v in members) != [1, 2] or any(not -100 <= by[v]['s_m'] <= 75.44 for v in members):
                rejected.append(dict(family=family, reason='requires in-zone mainline/ramp pair')); continue
        if family == 'merge_timed_wall' and (any(by[v]['s_m'] >= -100 for v in members)
                or len({by[v]['lane'] for v in members}) < 2
                or abs(by[members[0]]['s_m'] - by[members[1]]['s_m']) > 75):
            rejected.append(dict(family=family, reason='requires upstream multi-lane wall within75m')); continue
        for onset in onsets:
            for hold in ((20,) if family == 'yield_lane' else holds):
                for depth in ((.3,) if family == 'yield_lane' else depths):
                    if family == 'merge_timed_wall':
                        arrival = min((-100 - by[v]['s_m']) / by[v]['v_mps']
                                      if by[v]['v_mps'] else math.inf for v in roles)
                        if onset + hold > arrival:
                            continue
                    canonical = 'wall' if family == 'merge_timed_wall' else family
                    # ease_off is genuinely unary; enumerate each actor as a control.
                    groups = [[v] for v in roles] if family == 'ease_off' else [roles]
                    for group in groups:
                        move = dict(family=canonical, members=group, onset=onset, hold=hold, depth=depth)
                        ident = 'M' + digest(move)[:16]
                        joint = dict(id=ident, wm_move=move, wm_active_members=group, mining_family=family)
                        if family in LANE_FAMILIES:
                            joint['g5_move'] = dict(move)
                            joint['uses_lane_actions'] = True
                            duration = onset + hold
                            joint['lane_actions'] = [[int(v == group[0] and t == onset) if family == 'yield_lane'
                                                      else int(v == group[1] and t == onset + 1)
                                                      for t in range(duration)] for v in group]
                        out.append(joint)
                        if len(group) == 2:
                            for i, v in enumerate(group):
                                out.append(dict(joint, id=ident + '_solo' + str(i),
                                                wm_active_members=[v], solo_of=ident))
    if len(out) > 200:
        raise ValueError('bounded grid exceeds 200 total action slots')
    return out, rejected
