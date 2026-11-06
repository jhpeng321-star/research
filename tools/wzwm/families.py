"""Candidate lane-drop cooperation moves in the N3/PLANGEN variant format.

A move = (family, members, knobs). Knobs span STRENGTH (hold speed v_w, deceleration bound
dmax), TIMING (onset delay start_s, hold length hold_s, release rule) and MEMBERS (which CAVs).
Every move is executed by the unchanged N3 core (nm2_core.Arm role 'hold': each member tracks
a = clip((v_w - v)/1 s, -dmax, amax) with slowDown, lane-change mode 513, released at hold end,
at release_x, or when its anchor passes anchor_release_x). HDVs are never members.

Families
  abreast_wall   one zone CAV per lane (SUMO 0/1/2), spread <= WALL_SPREAD (pg_plan.pick_wall rule,
                 but every admissible triple is a candidate, not only the tightest)
  closing_pair   closing-lane (SUMO 0) CAV + adjacent-lane (SUMO 1) CAV, |dx| <= PAIR_DX; the
                 closing-lane CAV is the anchor (pg_plan pair_t semantics for release='early')
  harmonize_rows N9 (c4/nm-families) two abreast rows at a common speed released release_m before
                 the drop; generated only when tools.coopmwm.nm_families is importable (its SPECS
                 ranges). The other N9 families need CACC/lane executors that the N3 core lacks and
                 are reported as pending (never proposed for verification).

member_spec() is pg_plan.member_spec verbatim (the plug-in re-derives every spec with the vendored
pg_plan and refuses any mismatch).
"""
from __future__ import annotations

import hashlib
import itertools
import math

from . import geometry as G

REL = dict(release_x=G.X_B, amax=1.0)                  # pg_plan.REL
EARLY_RELEASE_M = 150.0                                 # pg_plan pair_t: anchor_release_x = X_B - 150
# pg_plan.VARIANTS (wall_l dropped at NEWMAP2 amendment 1) = the structured search's fixed variants
STRUCTURED = {
    'pair_g': ('closing_pair', dict(v_w=22.0, hold_s=120, dmax=1.0)),
    'pair_h': ('closing_pair', dict(v_w=15.0, hold_s=90, dmax=1.5)),
    'pair_t': ('closing_pair', dict(v_w=22.0, hold_s=150, dmax=1.0, anchor_release_x=G.X_B - EARLY_RELEASE_M)),
    'wall_g': ('abreast_wall', dict(v_w=22.0, hold_s=120, dmax=1.0)),
    'wall_h': ('abreast_wall', dict(v_w=15.0, hold_s=90, dmax=1.5)),
}
# Registered search grid (strength x timing). v_w/hold/dmax span the N3 variants and the R grid;
# start_s is the onset delay supported by nm2_core.Arm (params['start_s']).
DEFAULT_GRID = dict(v_w=(15.0, 18.0, 22.0), hold_s=(90, 120, 150, 180), dmax=(1.0, 1.5), start_s=(0, 10, 20),
                    release=('drop', 'early'))
KNOBS = ('v_w', 'hold_s', 'dmax', 'start_s', 'release')
BUILTIN_FAMILIES = ('abreast_wall', 'closing_pair')
N9_EXECUTABLE = ('harmonize_rows',)
MAX_SETS = 8


def member_spec(c, params, anchor=None):
    """pg_plan.member_spec (verbatim semantics)."""
    p = dict(REL)
    p.update(params)
    if 'anchor_release_x' in p:
        p['anchor'] = anchor
    return dict(id=c['id'], role='hold', params=p, lane=c['lane'], x=c['x'], v=c['v'])


def params_of(family, knob):
    """Variant params (pg_plan VARIANTS layout) of one knob setting."""
    p = dict(v_w=float(knob['v_w']), hold_s=int(knob['hold_s']), dmax=float(knob['dmax']))
    if int(knob.get('start_s') or 0):
        p['start_s'] = int(knob['start_s'])
    rel = knob.get('release', 'drop')
    if family == 'harmonize_rows':
        p['release_x'] = float(knob['release_x'])
        p['amax'] = float(knob['dmax'])
    elif rel == 'early':
        if family == 'closing_pair':
            p['anchor_release_x'] = G.X_B - EARLY_RELEASE_M
        else:
            p['release_x'] = G.X_B - EARLY_RELEASE_M
    elif rel != 'drop':
        raise ValueError(f'unknown release rule {rel!r}')
    return p


def knob_of_structured(vid):
    fam, prm = STRUCTURED[vid]
    return fam, dict(v_w=prm['v_w'], hold_s=prm['hold_s'], dmax=prm['dmax'], start_s=0,
                     release='early' if 'anchor_release_x' in prm else 'drop')


def move_key(family, ids, knob):
    k = '|'.join(f'{n}={knob[n]:g}' if isinstance(knob[n], (int, float)) else f'{n}={knob[n]}'
                 for n in sorted(knob))
    return f"{family}|{','.join(sorted(ids))}|{k}"


def set_key(family, ids):
    return f"{family}|{','.join(sorted(ids))}"


def make_move(family, members, knob, anchor=None):
    """members: CAV self-state dicts at t0 (id, edge, lane, x, v)."""
    params = params_of(family, knob)
    if family == 'closing_pair' and anchor is None:
        anchor = members[0]['id']
    specs = [member_spec(c, params, anchor=anchor) for c in members]
    ids = [c['id'] for c in members]
    return dict(key=move_key(family, ids, knob), set_key=set_key(family, ids), family=family, knob=dict(knob),
                params=params, member_ids=ids, members=specs, anchor=anchor if 'anchor_release_x' in params else None)


def to_variant(move):
    """N3 variant dict (family, params, members) consumed by pg_plan.bjob / pg_analyze."""
    return dict(family=move['family'], params=dict(move['params']), members=[dict(m) for m in move['members']])


# ---------------------------------------------------------------------------------------- member sets
def wall_sets(cavs, max_sets=MAX_SETS, spread=G.WALL_SPREAD):
    z = G.zone_cavs(cavs)
    lanes = [[c for c in z if int(c['lane']) == ln] for ln in (0, 1, 2)]
    if any(not x for x in lanes):
        return []
    out = []
    for combo in itertools.product(*lanes):
        xs = [float(c['x']) for c in combo]
        sp = max(xs) - min(xs)
        if sp <= spread:
            out.append(((sp, -max(xs), tuple(c['id'] for c in combo)), list(combo)))
    out.sort(key=lambda t: t[0])
    return [m for _, m in out[:max_sets]]


def pair_sets(cavs, max_sets=MAX_SETS, dx=G.PAIR_DX):
    z = G.zone_cavs(cavs)
    out = []
    for c0 in (c for c in z if int(c['lane']) == 0):
        for c1 in (c for c in z if int(c['lane']) == 1):
            d = abs(float(c0['x']) - float(c1['x']))
            if d <= dx:
                out.append(((d, -float(c0['x']), c0['id'], c1['id']), [c0, c1]))
    out.sort(key=lambda t: t[0])
    return [m for _, m in out[:max_sets]]


def n9_module():
    """tools.coopmwm.nm_families (c4/nm-families) if importable, else None."""
    try:
        from tools.coopmwm import nm_families
    except Exception:
        return None
    return nm_families


def n9_status(nf=None):
    nf = nf if nf is not None else n9_module()
    if nf is None:
        return dict(available=False, executable=[], pending_executor=[])
    names = list(getattr(nf, 'FAMILY_NAMES', ()))
    return dict(available=True, executable=[n for n in names if n in N9_EXECUTABLE],
                pending_executor=[n for n in names if n not in N9_EXECUTABLE],
                rule=getattr(nf, 'RULE', None))


def _linspace(lo, hi, n):
    return [lo] if n <= 1 or hi <= lo else [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def harmonize_grid(nf):
    spec = nf.SPECS['harmonize_rows']
    sp = spec.param('row_speed_mps')
    dlo, dhi = (float(x) for x in nf.DEPTH)
    release_x = G.X_B - float(spec.constant('release_m'))
    return dict(v_w=tuple(round(x, 2) for x in _linspace(sp.lo, sp.hi, 3)), hold_s=(180,),
                dmax=tuple(round(x, 3) for x in _linspace(dlo, dhi, 2)), start_s=(0,), release=('rows',),
                release_x=(release_x,))


def harmonize_sets(cavs, nf, max_sets=MAX_SETS):
    """Two abreast rows (>= 2 lanes each, span <= row_span_m), row 1 behind row 0 by row_spacing_m."""
    spec = nf.SPECS['harmonize_rows']
    span = float(spec.constant('row_span_m'))
    lo_sp, hi_sp = spec.param('row_spacing_m').lo, spec.param('row_spacing_m').hi
    cap = int(spec.cap[1])
    far = G.X_B - float(spec.constant('zone_far_m'))
    near = G.X_B - float(spec.constant('release_m'))
    z = [c for c in cavs if c['edge'] == 'up' and far <= float(c['x']) <= near]
    rows = []
    for lead in z:
        row = {int(lead['lane']): lead}
        for c in sorted(z, key=lambda c: abs(float(c['x']) - float(lead['x']))):
            ln = int(c['lane'])
            if ln not in row and abs(float(c['x']) - float(lead['x'])) <= span:
                row[ln] = c
        if len(row) >= 2:
            rows.append(sorted(row.values(), key=lambda c: int(c['lane'])))
    out, seen = [], set()
    for r0, r1 in itertools.permutations(rows, 2):
        ids0, ids1 = {c['id'] for c in r0}, {c['id'] for c in r1}
        if ids0 & ids1:
            continue
        gap = sum(float(c['x']) for c in r0) / len(r0) - sum(float(c['x']) for c in r1) / len(r1)
        if not lo_sp <= gap <= hi_sp or len(r0) + len(r1) > cap:
            continue
        key = frozenset(ids0 | ids1)
        if key in seen:
            continue
        seen.add(key)
        out.append(((abs(gap - (lo_sp + hi_sp) / 2), tuple(sorted(key))), r0 + r1))
    out.sort(key=lambda t: t[0])
    return [m for _, m in out[:max_sets]]


def grid_points(grid):
    names = [n for n in KNOBS if n in grid] + [n for n in grid if n not in KNOBS]
    for vals in itertools.product(*(grid[n] for n in names)):
        yield dict(zip(names, vals))


def member_sets(cavs, families=BUILTIN_FAMILIES, max_sets=MAX_SETS, nf=None):
    out = {}
    if 'abreast_wall' in families:
        out['abreast_wall'] = wall_sets(cavs, max_sets)
    if 'closing_pair' in families:
        out['closing_pair'] = pair_sets(cavs, max_sets)
    if 'harmonize_rows' in families and nf is not None:
        out['harmonize_rows'] = harmonize_sets(cavs, nf, max_sets)
    return out


def generate(cavs, grid=None, families=BUILTIN_FAMILIES, max_sets=MAX_SETS, nf=None):
    """Every (family, member set, knob) candidate move at one moment, deterministic order."""
    grid = dict(grid or DEFAULT_GRID)
    sets = member_sets(cavs, families, max_sets, nf)
    out = []
    for fam in families:
        g = harmonize_grid(nf) if fam == 'harmonize_rows' and nf is not None else grid
        for mem in sets.get(fam, []):
            for knob in grid_points(g):
                out.append(make_move(fam, mem, knob))
    return out


def grid_for(move, grid=None, nf=None):
    if move['family'] == 'harmonize_rows':
        return harmonize_grid(nf)
    return dict(grid or DEFAULT_GRID)


def neighbors(move, cavs, grid=None, nf=None):
    """One-step neighbours: one knob to an adjacent grid value, or one member swapped for another
    zone CAV of the same lane that keeps the family's geometry rule."""
    g = grid_for(move, grid, nf)
    out = []
    for n in g:
        vals = list(g[n])
        if len(vals) < 2 or move['knob'].get(n) not in vals:
            continue
        i = vals.index(move['knob'][n])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append(make_move(move['family'], _members(move, cavs), dict(move['knob'], **{n: vals[j]})))
    if move['family'] in BUILTIN_FAMILIES:
        z = G.zone_cavs(cavs)
        mem = _members(move, cavs)
        for i, m in enumerate(mem):
            for c in z:
                if c['id'] in move['member_ids'] or int(c['lane']) != int(m['lane']):
                    continue
                cand = mem[:i] + [c] + mem[i + 1:]
                xs = [float(x['x']) for x in cand]
                lim = G.WALL_SPREAD if move['family'] == 'abreast_wall' else G.PAIR_DX
                if max(xs) - min(xs) <= lim:
                    out.append(make_move(move['family'], cand, dict(move['knob'])))
    seen, uniq = {move['key']}, []
    for m in out:
        if m['key'] not in seen:
            seen.add(m['key'])
            uniq.append(m)
    return uniq


def _members(move, cavs):
    by = {c['id']: c for c in cavs}
    return [by[i] for i in move['member_ids']]


def structured_moves(cavs):
    """The N3/PLANGEN structured variants (geometry-only tightest pair/wall) as moves."""
    out = {}
    pairs, walls = pair_sets(cavs, 1), wall_sets(cavs, 1)
    for vid in STRUCTURED:
        fam, knob = knob_of_structured(vid)
        mem = (pairs if fam == 'closing_pair' else walls)
        if mem:
            out[vid] = make_move(fam, mem[0], knob)
    return out


# ---------------------------------------------------------------------------------------- WM actions
def open_loop_profile(members, H, cavs=None):
    """Approximate open-loop absolute accelerations (m/s^2, one per second) of the hold law for WMs
    that take profiles: members integrate v/x free of interactions; None = native (before onset).
    The list ends at the member's predicted release (hold end, release_x or anchor passing)."""
    state = {m['id']: [float(m['x']), float(m['v'])] for m in members}
    anchor_x = {}
    for m in members:
        a = m['params'].get('anchor')
        if a is not None and a not in state and cavs:
            c = next((c for c in cavs if c['id'] == a), None)
            if c is not None:
                anchor_x[a] = [float(c['x']), float(c['v'])]
    prof = {m['id']: [] for m in members}
    done = set()
    for el in range(int(H)):
        for m in members:
            vid, p = m['id'], m['params']
            if vid in done:
                continue
            x, v = state[vid]
            st = int(p.get('start_s', 0))
            if el < st:
                prof[vid].append(None)
                state[vid] = [x + v, v]
                continue
            anc = p.get('anchor')
            ax = (state.get(anc) or anchor_x.get(anc) or [None])[0] if anc is not None else None
            if (el >= st + int(p['hold_s']) or x >= float(p.get('release_x', G.X_B))
                    or (ax is not None and ax >= float(p['anchor_release_x']))):
                done.add(vid)
                continue
            a = max(-float(p['dmax']), min(float(p['amax']), float(p['v_w']) - v))
            prof[vid].append(round(a, 4))
            v = max(0.0, v + a)
            state[vid] = [x + v, v]
        for a, xv in anchor_x.items():
            anchor_x[a] = [xv[0] + xv[1], xv[1]]
    return prof


def crn_seed(sid, salt='wmc-crn-v1'):
    return int(hashlib.sha256(f'{salt}:{sid}'.encode()).hexdigest()[:12], 16)


def strength(params):
    """Monotone scalar move strength for the W3 head's 'depth' slot: dmax x relative speed cut."""
    return round(float(params['dmax']) * max(0.0, G.SPEED_LIMIT - float(params['v_w'])) / (G.SPEED_LIMIT - 15.0), 4)


def isfinite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)
