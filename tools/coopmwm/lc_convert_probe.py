"""N9 part of the universal mined-row converter: COOPKINDS/PLANGEN2 lane-family rows (DRAFT API).

Reads a verified ``coopmwm-mined-move-v1`` row of the probe lane families (lane_balance,
early_merge, zipper / zipper_order; discharge where expressible) and returns its ACTUAL
TraCI command log in the WM frame, with the probe's per-member parameters, so LCEXEC's
free_program can execute exactly what the verified run issued. Every mapping goes through
the N1 descriptor (nm_map.LaneDrop): SUMO lane -> WM lane (approach-edge lane table) and
route x -> WM s (x - drop_x = s - drop_s). ``to_traci`` is the exact inverse used by the
command-sequence parity test. No family law is re-implemented or re-parameterised here.

Probe sources (role laws and command kinds):
  runs/active/C4-V7FT-20261017/coopkinds/code/ck_core.py Arm.control (sha256 003311cb...):
    changeLane(vid, lane, 2.0) each second (lane_to / merge_early / zip_merge),
    setLaneChangeMode 513 (policy) / 512 (zipper merger, TraCI-only), setTau (tau roles),
    slowDown(v + a*1 s) for 'a' (zip_yield, hold, pulse dither);
  runs/active/C4-V7FT-20261017/coopkinds/code/ck_plan.py / plangen2/code/pg2_plan.py
    (member rules and B/R parameters).
"""
import json

MINED_SCHEMA = 'coopmwm-mined-move-v1'
SCHEMA = 'n9-probe-lane-program-v1'
LANE_FAMILIES = ('lane_balance', 'early_merge', 'zipper', 'zipper_order', 'discharge')
KINDS = ('changeLane', 'lcMode', 'tau', 'a')
#: Probe constants of the issued commands (ck_core.Arm: changeLane duration 2.0 s; 1 s control).
CHANGE_LANE_DURATION_S = 2.0
_X_KEYS = ('trigger_x', 'release_x', 'giveup_x', 'merge_x', 'fallback_x')


def family_of(row):
    """Probe lane family of a row (B family or R ``r_kind``), or None."""
    kind = row.get('r_kind') if row.get('family') == 'R_random' else row.get('family')
    return kind if kind in LANE_FAMILIES else None


def _move(row):
    return row['move'] if isinstance(row['move'], dict) else json.loads(row['move'])


def _log(row):
    """Row command log as [(el, vid, kind, value)] in issue order (both probe formats)."""
    raw = row['commands']
    if isinstance(raw, dict):           # COOPKINDS: {vid: [[el, kind, value]]}
        items = [(int(el), vid, kind, value) for vid, rows in raw.items() for el, kind, value in rows]
    else:                               # PLANGEN2: [[el, vid, kind, value]]
        items = [(int(el), vid, kind, value) for el, vid, kind, value in raw]
    return sorted(items, key=lambda x: (x[0], x[1], KINDS.index(x[2]) if x[2] in KINDS else 99))


def convert(row, geometry):
    """EXACT probe lane-family row -> WM-frame command program (refuses anything else)."""
    if row.get('schema') != MINED_SCHEMA:
        raise ValueError('not a coopmwm-mined-move-v1 row')
    family = family_of(row)
    if family is None:
        raise ValueError(f'not a probe lane-family row: {row.get("family")!r}/{row.get("r_kind")!r}')
    move = _move(row)
    if move.get('map') != geometry.name:
        raise ValueError(f'row map {move.get("map")!r} is not the bound map {geometry.name!r}')
    members = list(row['members'])
    spec = {m['id']: m for m in move['members']}
    if set(spec) != set(members) or len(spec) != len(move['members']):
        raise ValueError('row members differ from the move member spec')
    if [int(x) for x in row['member_lanes']] != [int(spec[v]['lane']) for v in members]:
        raise ValueError('row member_lanes differ from the move member spec')
    dither = {d['id'] for d in (row.get('dither') or []) if d.get('role') == 'pulse'}
    commands, excluded = {v: [] for v in members}, set()
    for el, vid, kind, value in _log(row):
        if kind not in KINDS:
            raise ValueError(f'unknown probe command kind {kind!r}')
        if vid not in spec:
            if vid not in dither:
                raise ValueError('row commands a non-member that is not a registered dither vehicle')
            excluded.add(vid)
            continue
        if not 0 <= el < int(row['duration_s']):
            raise ValueError('command outside the recorded duration')
        commands[vid].append([el, kind, geometry.wm_lane(value) if kind == 'changeLane' else value])
    params = {}
    for vid in members:
        out = {}
        for key, value in spec[vid]['params'].items():
            if key in _X_KEYS:
                out[key[:-2]+'_s'] = geometry.x_to_s(value)
            elif key == 'zone':
                out['zone_s'] = [geometry.x_to_s(x) for x in value]
            elif key == 'target_lane':
                out[key] = geometry.wm_lane(value)
            else:
                out[key] = value
        params[vid] = out
    return dict(schema=SCHEMA, family=family, map=geometry.name, t0=int(row['t0']),
                duration_s=int(row['duration_s']), members=members,
                roles={v: spec[v]['role'] for v in members},
                lanes={v: geometry.wm_lane(spec[v]['lane']) for v in members},
                rows={v: spec[v].get('row') for v in members},
                member_params=params, family_params=dict(move.get('params') or {}),
                commands=commands, profile=row['profile'], policy_profile=row.get('policy_profile'),
                release_el=row.get('release_el'), dither_excluded=sorted(excluded),
                source=dict(verification_id=row.get('verification_id'), plan_sha256=row.get('plan_sha256'),
                            arm=row.get('arm'), family=row.get('family'), r_kind=row.get('r_kind'),
                            variant=row.get('variant'), world=row.get('world'), seed=row.get('seed')))


def to_traci(program, geometry):
    """Exact inverse: the program's member commands as the probe log [(el, vid, kind, value)] (SUMO lanes)."""
    inverse = {wm: sumo for sumo, wm in geometry.sumo_to_wm}
    out = [(el, vid, kind, inverse[value] if kind == 'changeLane' else value)
           for vid, rows in program['commands'].items() for el, kind, value in rows]
    return sorted(out, key=lambda x: (x[0], x[1], KINDS.index(x[2])))


def member_log(row):
    """The row's own member-only TraCI log (dither vehicles removed), for parity tests."""
    members = set(row['members'])
    return [x for x in _log(row) if x[1] in members]
