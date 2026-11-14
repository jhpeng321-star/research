"""Lane-change + speed cooperation families (owner 2026-11-06, S-LCMOVES) as proposer candidates.

The member builders ARE the S-LCMOVES code: lcm_pipeline/ holds verbatim, sha-pinned copies of lcm_plan.py
(94ebbc93), lcm_core.py (48aaae38), lcm_families.py (4df417f5) and population.py (6af5ee23). Every candidate is a
LCMOVES variant (family, params, members with role 'prog' built by lcm_plan.prog / _hold_prog), so SUMO verifies it
with lcm_core.Arm unchanged (verify_plugin + runner --core lcm) and the admission rule is the NEWMAP2 one.

Families and names (= LCMOVES):
  wall_fill          a CAV moves from a doubled lane into the empty adjacent lane, tracks the wall slot, then holds
  rolling_wall       front CAVs slow (hold) while a rear CAV 200-500 m behind accelerates / changes lane into the gap
  gap_open_merge     gom_1: one lane-1 gap maker tracks 25 m behind a closing-lane merger that accelerates and merges;
                     gom_multi: 2-3 gap makers (25/50/75 m) and an early merger trigger (X_B - 1200)
  lane_redistribute  2-3 CAVs of the locally most loaded lane move to the least loaded adjacent lane, matching speed
  free_program       per member an optional lane change (target, offset) + hold/accelerate/slow to v_w for a duration,
                     then release; 2-6 CAVs of one V2V-local group (open search samples these, see open_search.py)
The proposer searches, beyond the LCMOVES best-geometry member set, every geometric candidate the builders return
(<= max_per_family, in builder order) and registered knob variations: hold members' (v_w, hold_s, dmax) over the
NEWMAP2 R grid, lane-change onsets over OFFSETS (WMDATA 0/20/40/60 s). Every value comes from lcm_plan constants.
CAVs only (members are t0 CAV self-states on 'up'); lane_redistribute's lane counts use observable rows only
(CAV self-states + CAV-detected neighbours of tapes.enrich).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

from . import geometry as G

VENDOR = Path(__file__).resolve().parent / 'lcm_pipeline'
PINS = {
    'lcm_plan.py': 'f43c9e472bd61fb742af9f197bf00f1b659b00ad3e29dada213611a9c2ec14f4',      # raw 94ebbc93 (CRLF)
    'lcm_core.py': '48aaae384210e1bb97e9a2c028761e01ba7fd6d4c632d70946b007319cc330db',
    'lcm_families.py': '4df417f5f33f506011a3a1abdf1cfd5d29df96bb9521bcbcf6013343fe8d2c77',
    'population.py': '5e8c61a6c1d5195e2830a5f6825e7a0ed4fc0c3260e243880ab956ddbff5849b',     # raw 6af5ee23 (CRLF)
}
LC_FAMILIES = ('wall_fill', 'rolling_wall', 'gap_open_merge', 'lane_redistribute', 'free_program')
BUILDERS = (('wall_fill', 'wall_fill', 'cands_wall_fill'), ('rolling_wall', 'rolling_wall', 'cands_rolling_wall'),
            ('gom_1', 'gap_open_merge', 'cands_gom_1'), ('gom_multi', 'gap_open_merge', 'cands_gom_multi'),
            ('lane_redistribute', 'lane_redistribute', 'cands_lane_redistribute'))
PROG_KEYS = {'release_x', 'amax', 'speed', 'v_w', 'dmax', 'hold_s', 'sp_from', 'lc_target', 'lc_from', 'lc_from_x',
             'lc_giveup_x', 'fallback_x', 'merger', 'track_ref', 'track_off', 'kp', 'kv', 'settle_gap', 'tau',
             'tau_from', 'release_on_partner_merge', 'lc_mode'}
_MOD = {}


class LcmDriftError(RuntimeError):
    """A vendored LCMOVES file differs from its pinned bytes."""


def verify(vendor=VENDOR):
    for name, pin in PINS.items():
        got = hashlib.sha256((Path(vendor) / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if got != pin:
            raise LcmDriftError(f'{name}: lf_sha256 {got} != pinned {pin}')
    return dict(PINS)


def lcm():
    """The vendored lcm_plan module (imported once; its import has no side effects beyond constants)."""
    if 'plan' not in _MOD:
        verify()
        if str(VENDOR) not in sys.path:
            sys.path.append(str(VENDOR))
        import lcm_plan
        if Path(lcm_plan.__file__).resolve().parent != VENDOR:
            raise LcmDriftError('lcm_plan was imported from outside the vendored directory')
        _MOD['plan'] = lcm_plan
    return _MOD['plan']


def snapshot(m):
    """LCMOVES snapshot from a moment: CAV self-states + observable 'up' rows [lane, x, v] (CAVs + detected)."""
    cavs = [dict(c) for c in m['all_cavs_t0']]
    obs = (m.get('observed') or {}).get('vehicles') or []
    veh_up = [[int(r['lane']), float(r['x']), float(r['v'])] for r in list(cavs) + list(obs) if r['edge'] == 'up']
    return dict(cavs=cavs, veh_up=veh_up)


def prog_ref(c, params):
    """lcm_plan.prog re-derivation of one member spec (verify_plugin checks equality)."""
    return lcm().prog(c, **params)


def _key(family, members):
    blob = json.dumps([[mm['id'], mm['params']] for mm in members], sort_keys=True)
    return f"{family}|{','.join(sorted(mm['id'] for mm in members))}|{hashlib.sha256(blob.encode()).hexdigest()[:16]}"


def lc_move(family, origin, members, params=None, knob=None):
    ids = [mm['id'] for mm in members]
    return dict(key=_key(family, members), set_key=f"{family}|{','.join(sorted(ids))}", family=family,
                knob=dict(knob or {}, origin=origin), params=dict(params or {}), member_ids=ids,
                members=[dict(mm) for mm in members], anchor=None, role='prog', origin=origin)


def _vary_hold(members, v_w, hold_s, dmax):
    """Same move with every hold/settled-track member's (v_w, hold_s, dmax) replaced (registered R grid)."""
    out = []
    for mm in members:
        p = dict(mm['params'])
        if p.get('speed') in ('hold', 'track') and 'v_w' in p:
            p.update(v_w=float(v_w), hold_s=int(hold_s))
            if p.get('speed') == 'hold':
                p['dmax'] = float(dmax)
        out.append(dict(mm, params=p))
    return out


def _vary_lc_from(members, offset):
    out = []
    for mm in members:
        p = dict(mm['params'])
        if p.get('lc_target') is not None and 'lc_from_x' not in p:
            p['lc_from'] = int(offset)
        out.append(dict(mm, params=p))
    return out


def lc_moves(m, *, families=LC_FAMILIES, max_per_family=4, variations=True, n_free=4):
    """Every LCMOVES-family candidate at one moment (deterministic order: builder, geometry rank, variation)."""
    P = lcm()
    snap = snapshot(m)
    out = []
    for vid, fam, fname in BUILDERS:
        if fam not in families:
            continue
        cands = sorted(getattr(P, fname)(snap), key=lambda kv: kv[0])[:max_per_family]
        for rank, (_key0, mem) in enumerate(cands):
            out.append(lc_move(fam, f'{vid}#{rank}', mem, knob=dict(builder=vid, rank=rank)))
            if not variations:
                continue
            if fam in ('wall_fill', 'rolling_wall'):
                for v_w, hold_s, dmax in itertools.product(P.R_GRID['v_w'], P.R_GRID['hold_s'], P.R_GRID['dmax']):
                    out.append(lc_move(fam, f'{vid}#{rank}', _vary_hold(mem, v_w, hold_s, dmax),
                                       knob=dict(builder=vid, rank=rank, v_w=v_w, hold_s=hold_s, dmax=dmax)))
            if any(mm['params'].get('lc_target') is not None and 'lc_from_x' not in mm['params'] for mm in mem):
                for off in P.OFFSETS[1:]:
                    out.append(lc_move(fam, f'{vid}#{rank}', _vary_lc_from(mem, off),
                                       knob=dict(builder=vid, rank=rank, lc_from=off)))
    if 'free_program' in families and n_free:
        for name, mem, prm in P.free_programs(snap, m['sid'], n=int(n_free)):
            out.append(lc_move('free_program', name, mem, params=prm, knob=dict(builder='free_program', **prm)))
    seen, uniq = set(), []
    for mv in out:
        if mv['key'] not in seen:
            seen.add(mv['key'])
            uniq.append(mv)
    return uniq


def validate_prog(m, vid, v):
    """CAV-only, registered keys, and member specs identical to lcm_plan.prog re-derivations."""
    by = {c['id']: c for c in m['all_cavs_t0']}
    for mm in v['members']:
        c = by.get(mm['id'])
        if c is None or c['edge'] != 'up':
            raise ValueError(f"{m['sid']}/{vid}: member {mm['id']} is not a CAV on 'up' at t0")
        if mm['role'] != 'prog':
            raise ValueError(f"{m['sid']}/{vid}: role {mm['role']!r} is not an LCMOVES program")
        extra = set(mm['params']) - PROG_KEYS
        if extra:
            raise ValueError(f"{m['sid']}/{vid}: unregistered program keys {sorted(extra)}")
        ref = prog_ref(c, {k: val for k, val in mm['params'].items()})
        if ref != mm:
            raise ValueError(f"{m['sid']}/{vid}: member spec differs from lcm_plan.prog: {mm} != {ref}")
        for r in mm['params'].get('track_ref') or []:
            if r not in by:
                raise ValueError(f"{m['sid']}/{vid}: track reference {r} is not a CAV")
        bad = out_of_range(mm)
        if bad:
            raise ValueError(f"{m['sid']}/{vid}: member {mm['id']} outside the registered ranges: {bad}")
    return [mm['id'] for mm in v['members']]


def out_of_range(mm):
    """Registered-range check of one program (NEWMAP2 / COOPKINDS / PLANGEN2 / WMDATA constants of lcm_plan)."""
    P = lcm()
    p = mm['params']
    bad = []
    if p.get('release_x') != P.X_B or p.get('amax') != P.AMAX:
        bad.append('release_x/amax')
    tl = p.get('lc_target')
    if tl is not None and (tl not in (0, 1, 2) or abs(int(tl) - int(mm['lane'])) != 1):
        bad.append('lc_target')
    if p.get('speed') not in (None, 'hold', 'track', 'match', 'accel'):
        bad.append('speed')
    if 'v_w' in p and not min(P.R_GRID['v_w']) <= float(p['v_w']) <= max(P.R_GRID['v_w']):
        bad.append('v_w')
    if 'dmax' in p and not min(P.R_GRID['dmax']) <= float(p['dmax']) <= max(P.R_GRID['dmax'] + [P.TRACK['dmax']]):
        bad.append('dmax')
    if 'hold_s' in p and not 0 < int(p['hold_s']) <= max(P.OFFSETS) + 300:
        bad.append('hold_s')
    for k in ('lc_from', 'sp_from', 'tau_from'):
        if k in p and not 0 <= int(p[k]) <= max(P.OFFSETS):
            bad.append(k)
    if p.get('tau') not in P.TAU_SET:
        bad.append('tau')
    return bad


def descriptor_lanes(mm):
    """(lane at t0, lane after the program's lane change) of a member spec (hold or prog)."""
    lane = int(mm['lane'])
    tl = mm['params'].get('lc_target') if mm.get('role') == 'prog' else None
    return lane, (lane if tl is None else int(tl))


def zone_ok(mm):
    return G.ZONE[0] <= float(mm['x']) <= G.ZONE[1]
