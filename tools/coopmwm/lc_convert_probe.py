"""N9 part of LCEXEC's universal converter: COOPKINDS / PLANGEN2 lane-family rows -> free_program.

``probe_free_program(row, frame)`` turns ONE verified ``coopmwm-mined-move-v1`` row whose
family (or R ``r_kind``) is lane_balance, early_merge, zipper / zipper_order or (COOPKINDS)
discharge into an :class:`lc_program.FreeProgram`:

* member order = the row's member order (the probe's), ``join_el`` all 0 (every probe member
  started at t0), ``duration_s`` = the row's, ``source`` = row identity;
* each member program = ``lc_roles.member_program(role, params, frame)`` on the member spec
  EXACTLY as the probe Arm received it (no role semantics re-implemented here);
* the row's own TraCI command log (both formats: COOPKINDS ``{vid: [[el, kind, value]]}``,
  PLANGEN2 ``[[el, vid, kind, value]]``) is validated against the programs
  (:func:`check_log`); commands on non-members are allowed only for the row's registered
  dither (``pulse``) vehicles, which are excluded from the program;
* every registered continuous value passes ``lc_program.check_registered``.

Frame: ``lc_program.frame_from_descriptor(map_descriptor.load(...))`` (N1), shared with
LCEXEC: s = x - anchor_x_m, SUMO approach lane -> WM lane via the N1 lane table.

Probe sources: runs/active/C4-V7FT-20261017/coopkinds/code/ck_core.py Arm.control
(sha256 003311cb...), plangen2/code/pg2_core.py Arm.control (COOPKINDS roles verbatim),
ck_plan.py / pg2_plan.py (member rules, B_PARAMS / R_GRID, CK_B / CK_R).
Logged command kinds: changeLane (SUMO lane; duration 2.0 s), lcMode (zip_merge 512 only),
tau (setTau, value only), a (COOPKINDS zip_yield/hold issued slowDown accel, round(a, 4);
PLANGEN2 keeps issued accels out of its command log).
"""
import json

from .lc_program import FreeProgram, check_registered
from .lc_roles import member_program

MINED_SCHEMA = 'coopmwm-mined-move-v1'
LANE_FAMILIES = ('lane_balance', 'early_merge', 'zipper', 'zipper_order', 'discharge')
LANE_ROLES = ('tau', 'merge_early', 'zip_merge', 'zip_yield', 'lane_to')
KINDS = ('changeLane', 'lcMode', 'tau', 'a')
A_ROUND = 1e-4 + 1e-12          # probe logs round(a, 4)


def family_of(row):
    """Probe lane family of a row (B family or R ``r_kind``), or None (not this converter's)."""
    kind = row.get('r_kind') if row.get('family') == 'R_random' else row.get('family')
    return kind if kind in LANE_FAMILIES else None


def _move(row):
    return row['move'] if isinstance(row['move'], dict) else json.loads(row['move'])


def _log(row):
    """Row command log [(el, vid, kind, value)] (both probe formats), stable order."""
    raw = row['commands']
    if isinstance(raw, dict):
        items = [(int(el), vid, kind, value) for vid, rows in raw.items() for el, kind, value in rows]
    else:
        items = [(int(el), vid, kind, value) for el, vid, kind, value in raw]
    return sorted(items, key=lambda x: (x[0], x[1], KINDS.index(x[2]) if x[2] in KINDS else len(KINDS)))


def member_log(row):
    """The row's member-only command log (dither vehicles removed), probe frame."""
    members = set(row['members'])
    return [x for x in _log(row) if x[1] in members]


def wm_log(row, frame):
    """Member-only command log in the WM frame (changeLane value -> WM lane)."""
    return [(el, vid, kind, frame.wm(value) if kind == 'changeLane' else value)
            for el, vid, kind, value in member_log(row)]


def probe_log(entries, frame):
    """Exact inverse of :func:`wm_log` (WM lane -> SUMO lane on the approach edge)."""
    return [(el, vid, kind, frame.sumo(value) if kind == 'changeLane' else value)
            for el, vid, kind, value in entries]


def _parse(row, frame):
    if row.get('schema') != MINED_SCHEMA:
        raise ValueError('not a coopmwm-mined-move-v1 row')
    family = family_of(row)
    if family is None:
        raise ValueError(f'not a probe lane-family row: {row.get("family")!r}/{row.get("r_kind")!r}')
    move = _move(row)
    if move.get('map') != frame.name:
        raise ValueError(f'row map {move.get("map")!r} is not the frame map {frame.name!r}')
    members = list(row['members'])
    spec = {m['id']: m for m in move['members']}
    if set(spec) != set(members) or len(spec) != len(move['members']) or len(set(members)) != len(members):
        raise ValueError('row members differ from the move member spec')
    if [int(x) for x in row['member_lanes']] != [int(spec[v]['lane']) for v in members]:
        raise ValueError('row member_lanes differ from the move member spec')
    if any(spec[v]['role'] not in LANE_ROLES for v in members):
        raise ValueError('lane-family row with a non-lane-family role')
    dither = {d['id'] for d in (row.get('dither') or []) if d.get('role') == 'pulse'}
    outsiders = {vid for _, vid, _, _ in _log(row) if vid not in spec}
    if outsiders-dither:
        raise ValueError('row commands a non-member that is not a registered dither vehicle')
    if any(kind not in KINDS for _, _, kind, _ in _log(row)):
        raise ValueError('unknown probe command kind')
    return family, move, members, spec, sorted(outsiders)


def check_log(program, row, frame):
    """Static consistency of the row's member command log with the converted programs."""
    duration = int(row['duration_s'])
    seen = {}
    for el, vid, kind, value in member_log(row):
        p = program.program(vid)
        if not 0 <= el < duration or el < p.start_el or (p.hold_s is not None and el >= p.start_el+p.hold_s):
            raise ValueError(f'{vid}: command at {el} outside the program window')
        count = seen.setdefault((vid, kind), [])
        count.append(el)
        if kind == 'changeLane':
            if p.lane is None or value != frame.sumo(p.lane.target):
                raise ValueError(f'{vid}: changeLane({value}) is not the program lane request')
        elif kind == 'tau':
            if p.tau is None or abs(float(value)-p.tau) > 1e-12 or len(count) > 1:
                raise ValueError(f'{vid}: tau command differs from the program tau')
        elif kind == 'lcMode':
            if not p.mode_logged or int(value) != p.lc_mode or len(count) > 1:
                raise ValueError(f'{vid}: lcMode command differs from the program mode')
        elif kind == 'a':
            lon = p.lon
            if lon.kind not in ('speed', 'track', 'accel') or not (
                    -float(lon.dmax)-A_ROUND <= float(value) <= float(lon.amax)+A_ROUND):
                raise ValueError(f'{vid}: issued accel {value} outside the program law bounds')
    release = row.get('release_el') or {}
    for vid in program.members:
        last = max((el for (v, _), els in seen.items() if v == vid for el in els), default=None)
        rel = release.get(vid)
        if last is not None and rel is not None and rel < last:
            raise ValueError(f'{vid}: release_el {rel} before its last command {last}')
        if rel is not None and rel > duration:
            raise ValueError(f'{vid}: release_el beyond the recorded duration')
    return True


def probe_free_program(row, frame, *, file=None):
    """EXACT verified lane-family row -> lc_program.FreeProgram (refuses anything else)."""
    family, move, members, spec, dither = _parse(row, frame)
    programs = tuple(member_program(spec[v]['role'], spec[v]['params'], frame) for v in members)
    program = FreeProgram(tuple(members), programs, map=frame.name, join_el=(0,)*len(members),
                          duration_s=int(row['duration_s']),
                          source=dict(schema=MINED_SCHEMA, converter='n9-lc-convert-probe-v1', file=file,
                                      verification_id=row.get('verification_id'),
                                      plan_sha256=row.get('plan_sha256'), arm=row.get('arm'),
                                      family=row.get('family'), r_kind=row.get('r_kind'), probe_family=family,
                                      variant=row.get('variant'), seed=row.get('seed'), t0=row.get('t0'),
                                      dither_excluded=dither, release_el=row.get('release_el')))
    violations = check_registered(program)
    if violations:
        raise ValueError(f'converted program outside the registered probe ranges: {violations}')
    check_log(program, row, frame)
    return program
