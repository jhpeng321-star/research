"""FULLPOL free_program candidates for the trigger slate (clean5, DEFAULT OFF).

A ``free_program`` slate entry is a per-member program of LCEXEC primitives in LCEXEC's
``lcexec-free-program-v1`` dict schema (tools/coopmwm/lc_program.py FreeProgram.to_dict):
per member a longitudinal primitive (base / native / speed / track / match / accel), an
optional persistent lane request to an ADJACENT EXISTING lane (the closing lane included),
optional own tau, lcMode, start offset, hold and release conditions. This module only
PROPOSES programs; LCEXEC's executor runs them, the trigger chooses among them.

Slate rule ``fullpol-free-program-slate-v1`` (FULLPOL 2026-11-06, registration in
FULLPOL_READY.json): each program source class is its own F2 slot key (like N2's per-kind
initiate slots), so the F2 balanced rule gives every feasible class an equal share; the
option family recorded in the table is always ``free_program``. Classes:

* ``verified``   - the arm's OWN converted verified programs (B: verified rows; R: its own
                   matched-effort random-procedure rows, or none), re-bound to the acting
                   CAV's acquirable CAVs by lane role and longitudinal order;
* ``structured`` - registered structured families written as programs (closing_pair,
                   abreast_wall [NEWMAP2], wall_fill, gap_open_merge, lane_redistribute
                   [LCMOVES/COOPKINDS registrations]);
* ``random_r``   - LCMOVES' registered matched random-program procedure lcm_R_v1;
* ``random_prim``- uniform over LCEXEC's longitudinal primitive kinds, every parameter from a
                   registered grid (coverage of the primitive space).

Every number below is a registered value with its source (REGISTERED); no behavioural
threshold is introduced here. Every builder may return None for an infeasible draw (N11 F8
None-guard): callers drop None; the slate loop falls back to the pooled (feasible) template.

Observables only: builders read the acting CAV's own row, its sensed rows (HDVs are
geometry only, never members) and its acquirable observable CAV ids. Members are CAVs.
The regime / demand / penetration / any configuration label never enters a program, a
descriptor or a slate.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import combinations
import math

from . import moves

FAMILY = 'free_program'
RULE = 'fullpol-free-program-slate-v1'
PROVENANCE = ('FULLPOL 2026-11-06 (owner 17:55 "act any move that is allowed in the sumo"); '
              'MAIN 18:4x design approval; ranges: REGISTERED sources')
PROGRAM_SCHEMA = 'lcexec-free-program-v1'
CLASSES = ('verified', 'structured', 'random_r', 'random_prim')
SLOTS = tuple(f'{FAMILY}:{c}' for c in CLASSES)
STRUCTURED = ('closing_pair', 'abreast_wall', 'wall_fill', 'gap_open_merge', 'lane_redistribute')
LON_KINDS = ('base', 'native', 'speed', 'track', 'match', 'accel')      # LCEXEC lc_program.LON_KINDS
LANE_RULES = ('until_reached', 'right_of_target', 'from_lane')         # LCEXEC lc_program.LANE_RULES
RELEASES = ('hold_end', 'release_s', 'anchor', 'zone', 'partner_left', 'merged_zip', 'merger', 'fallback')
LC_MODE_POLICY, LC_MODE_TRACI_ONLY = 513, 512

_NM2 = 'NEWMAP2_REPORT.md / lane_drop.LANE_DROP_REGISTERED (NEWMAP2-20261106)'
_LCM = 'REGISTRATION.md 2026-11-06 18:22 S-LCMOVES'
_CK = 'COOPKINDS registration 15:06 (ck_plan B_PARAMS / R_GRID), cited by S-LCMOVES 18:22'
REGISTERED = dict(
    v_w=dict(values=(15., 18., 22.), source=_NM2+' R grid; '+_LCM+' R grid'),
    hold_s=dict(values=(90, 120, 180), source=_NM2+' R grid; '+_LCM+' R grid'),
    dmax=dict(values=(1.0, 1.5), source=_NM2+' R grid; '+_LCM),
    amax=dict(values=(1.0,), source=_NM2+' amax 1.0; '+_LCM+' amax 1.0'),
    tau=dict(values=(None, 0.6, 2.0), source=_LCM+' R own tau {none, 0.6, 2.0}'),
    offsets_s=dict(values=(0, 20, 40, 60), source=_LCM+' R LC offset and start offset {0,20,40,60}'),
    zone_to_drop_m=dict(values=(400., 1800.), source=_NM2+' zone [X_B-1800, X_B-400]'),
    lane_giveup_to_drop_m=dict(values=(300.,), source=_CK+' lane_balance give-up 300 m; '+_LCM),
    r_group_size=dict(values=(2, 3, 4), source=_LCM+' R random 2-4 zone CAVs'),
    r_spread_m=dict(values=(300.,), source=_LCM+' R within 300 m (NEWMAP2 R spread)'),
    member_arity=dict(values=(1, 2, 3, 4), source='tracks_spec.md T1A move members 1..4 nearby CAVs '
                      'incl. self (moves.MOVE_MAX_MEMBERS)'),
    lc_mode=dict(values=(513,), source=_LCM+' prog role lcMode 513 (probe LC_MODE_POLICY)'),
    pair_dx_m=dict(values=(100.,), source=_NM2+' pair |dx| <= 100 m'),
    wall_spread_m=dict(values=(200.,), source=_NM2+' wall spread <= 200 m; '+_LCM+' wall_fill within 200 m'),
    anchor_release_m=dict(values=(150.,), source=_NM2+' pair_t anchor 150 m before the drop'),
    wall_h=dict(values=(dict(v_w=15., hold_s=90, dmax=1.5),), source=_NM2+' wall_h; '+_LCM+' wall_fill hold'),
    track=dict(values=(dict(kp=.15, kv=.6, dmax=1.5, amax=1.0),), source=_CK+' slot tracking; '+_LCM),
    track_gap_m=dict(values=(25.,), source=_CK+' gap 25; '+_LCM+' gap_open_merge 25 m behind'),
    settle_m=dict(values=(25.,), source=_LCM+' wall_fill settles within 25 m'),
    zipper_merger_to_drop_m=dict(values=(200., 900.), source=_CK+' zipper window M [X_B-900, X_B-200]'),
    zipper_yield_rel_m=dict(values=(-120., 30.), source=_CK+' zipper window Y [x_M-120, x_M+30]'),
    zipper_fallback_to_drop_m=dict(values=(15.,), source=_CK+' zipper fallback X_B-15'),
    zipper_hold_s=dict(values=(dict(merge=300, yield_=240),), source=_CK+' zip_merge hold 300 / zip_yield 240'),
    merge_v_w=dict(values=(22.,), source=_LCM+' gap_open_merge M accelerates (hold 22, amax 1.0)'),
    redistribute_to_drop_m=dict(values=(600., 1800.), source=_CK+' lane_balance zone [X_B-1800, X_B-600]; '+_LCM),
    redistribute_size=dict(values=(2, 3), source=_LCM+' lane_redistribute 2-3 CAVs'),
    redistribute_hold_s=dict(values=(300,), source=_CK+' lane_balance hold_s 300'),
    accel_pulse=dict(values=(dict(a=-1.0, hold_s=2),), source=_NM2+' dither pulse -1.0 m/s^2 for 2 s '
                     '(LCEXEC REGISTRY accel)'),
    track_off_m=dict(values=(-35., -25., -15.), source=_CK+' zipper gap {15,25,35} (LCEXEC REGISTRY track_off_m)'),
    sensing_m=dict(values=(100.,), source='merge_features.SENSOR_RANGE_M (own sensing radius)'))


def registered(key):
    return REGISTERED[key]['values']


# ---------------------------------------------------------------- settings

def settings(cfg):
    """None when OFF (absent/False) -> every pre-existing output is byte-identical."""
    slot = (cfg or {}).get('fullpol', False)
    if slot is False or slot is None:
        return None
    if not isinstance(slot, dict) or slot.get('enabled') is not True or slot.get('rule') != RULE:
        raise ValueError('fullpol requires enabled=True and the registered slate rule')
    keys = {'enabled', 'rule', 'provenance', 'classes', 'templates', 'view', 'structured',
            'learn', 'regime_guidance', 'parity_families'}
    if set(slot)-keys:
        raise ValueError(f'unregistered fullpol keys {sorted(set(slot)-keys)}')
    if not isinstance(slot.get('provenance'), str) or not slot['provenance'].strip():
        raise ValueError('fullpol requires provenance')
    classes = tuple(slot.get('classes', CLASSES))
    if not classes or any(c not in CLASSES for c in classes) or len(set(classes)) != len(classes):
        raise ValueError(f'fullpol classes must be distinct entries of {CLASSES}')
    structured = tuple(slot.get('structured', STRUCTURED))
    if any(f not in STRUCTURED for f in structured):
        raise ValueError(f'fullpol structured families must be in {STRUCTURED}')
    templates = slot.get('templates', {}) or {}
    if not isinstance(templates, dict) or set(templates)-{'B', 'R'}:
        raise ValueError('fullpol templates are per arm {B: [...], R: [...]}')
    for rows in templates.values():
        for t in rows:
            check_template(t)
    out = deepcopy(slot)
    out.update(classes=classes, structured=structured)
    return out


def config_keys(*, provenance, view, templates=None, classes=CLASSES, structured=STRUCTURED, learn=None,
                regime_guidance=None, parity_families=None):
    """Common B = R config value for ``cfg['fullpol']`` (per-arm templates = guidance source)."""
    out = dict(enabled=True, rule=RULE, provenance=provenance, classes=list(classes),
               structured=list(structured), view=dict(view), templates=deepcopy(templates or {}))
    if learn is not None:
        out['learn'] = deepcopy(learn)
    if regime_guidance is not None:
        out['regime_guidance'] = deepcopy(regime_guidance)
    if parity_families is not None:
        out['parity_families'] = sorted(parity_families)
    return out


def view_of(fp):
    from .lane_drop import MapView
    return MapView.from_descriptor(fp['view'])


def runtime_state(fp, arm, view=None):
    """Per-CAV candidate state (identical for every CAV; templates = the arm's own guidance)."""
    return dict(view=view or view_of(fp), classes=fp['classes'], structured=fp['structured'],
                templates=tuple(fp.get('templates', {}).get(arm, ())))


# ---------------------------------------------------------------- program dicts (LCEXEC schema)

def lon(kind='base', *, v_w=None, amax=1., dmax=1., refs=(), off=0., kp=.15, kv=.6, settle_gap=None,
        from_el=0, a=None):
    if kind not in LON_KINDS:
        raise ValueError('unknown lon kind')
    return dict(kind=kind, v_w=v_w, amax=float(amax), dmax=float(dmax), refs=list(refs), off=float(off),
                kp=float(kp), kv=float(kv), settle_gap=settle_gap, from_el=int(from_el), a=a)


def lane_req(target, *, rule='until_reached', source_lane=None, from_el=0, from_s=None, giveup_s=None,
             duration_s=2.0):
    if rule not in LANE_RULES:
        raise ValueError('unknown lane rule')
    return dict(target=int(target), rule=rule, source_lane=source_lane, from_el=int(from_el),
                from_s=None if from_s is None else float(from_s),
                giveup_s=None if giveup_s is None else float(giveup_s), duration_s=float(duration_s))


def member(lon_=None, *, lane=None, start_el=0, hold_s=None, release_s=None, anchor=None,
           anchor_release_s=None, zone=None, partner=None, partner_lane=None, merger_lane=None,
           fallback_s=None, lc_mode=LC_MODE_POLICY, tau=None, mode_on='first_active',
           releases=('hold_end', 'release_s'), role='prog'):
    return dict(lon=lon_ or lon(), lane=lane, start_el=int(start_el), hold_s=hold_s,
                release_s=None if release_s is None else float(release_s), anchor=anchor,
                anchor_release_s=None if anchor_release_s is None else float(anchor_release_s),
                zone=None if zone is None else [float(z) for z in zone], partner=partner,
                partner_lane=partner_lane, merger_lane=merger_lane,
                fallback_s=None if fallback_s is None else float(fallback_s), lc_mode=int(lc_mode),
                tau=None if tau is None else float(tau), mode_on=mode_on, releases=list(releases),
                role=role, mode_logged=False)


def program(members, programs, *, duration_s=None, source=None, map_name='wz3to2', join_el=None):
    members = list(members)
    if not members or len(set(members)) != len(members) or len(programs) != len(members):
        raise ValueError('free_program needs distinct members with one program each')
    if duration_s is None:
        duration_s = max(int(p['start_el'])+int(p['hold_s'] or 0) for p in programs)
    return dict(schema=PROGRAM_SCHEMA, rule='lcexec-free-program-v1', map=map_name, members=members,
                programs=[deepcopy(p) for p in programs],
                join_el=list(join_el or [0]*len(members)), duration_s=max(1, int(duration_s)),
                source=deepcopy(source or {}))


def rebind(prog, mapping):
    """Program dict with member ids (and every id reference) renamed through ``mapping``."""
    out = deepcopy(prog)
    out['members'] = [mapping[v] for v in prog['members']]
    for p in out['programs']:
        p['lon']['refs'] = [mapping.get(r, r) for r in p['lon']['refs']]
        for key in ('anchor', 'partner'):
            if p.get(key) is not None:
                p[key] = mapping.get(p[key], p[key])
    return out


def lc_checked(prog):
    """LCEXEC validation of a program dict when lc_program is importable (else schema check)."""
    if prog.get('schema') != PROGRAM_SCHEMA:
        raise ValueError('not an lcexec free_program dict')
    try:
        from . import lc_program
    except ImportError:
        return prog
    lc_program.FreeProgram.from_dict(prog)
    return prog


# ---------------------------------------------------------------- the trigger option

@dataclass(frozen=True)
class FreeProgramOption:
    """Trigger table entry: one free_program for its members (initiator = acting CAV first).

    ``observed`` = {vid: [WM lane, s_m]} of the members from the acting CAV's own sensing at
    proposal time (descriptor geometry only). ``cls`` / ``source`` are provenance metadata;
    they are NEVER descriptor inputs (programs are judged by content)."""
    members: tuple
    program: dict
    cls: str
    source: dict = field(default_factory=dict)
    observed: dict = field(default_factory=dict)
    family: str = FAMILY
    self_validated_windows = True   # trigger_policy: registered program ranges, not variety windows

    def __post_init__(self):
        object.__setattr__(self, 'members', tuple(self.members))
        if self.family != FAMILY or self.cls not in CLASSES+('offered',):
            raise ValueError('free_program option family/class')
        if list(self.members) != list(self.program['members']):
            raise ValueError('option members must equal the program members (initiator first)')
        if not 1 <= len(self.members) <= moves.MOVE_MAX_MEMBERS:
            raise ValueError(f'free_program has 1..{moves.MOVE_MAX_MEMBERS} members')

    @property
    def onset(self):
        return 0

    @property
    def hold(self):
        return int(self.program['duration_s'])

    @property
    def onset_s(self):
        return 0

    @property
    def hold_s(self):
        return self.hold

    @property
    def depth(self):
        return 0.

    @property
    def actors(self):
        return self.members

    def to_dict(self):
        return dict(family=FAMILY, members=list(self.members), program=deepcopy(self.program), cls=self.cls,
                    source=deepcopy(self.source), observed={k: list(v) for k, v in self.observed.items()},
                    onset=0, hold=self.hold, depth=0.)

    @classmethod
    def from_dict(cls, data):
        if data.get('family') != FAMILY:
            raise ValueError('not a free_program option')
        return cls(tuple(data['members']), deepcopy(data['program']), data['cls'], deepcopy(data.get('source') or {}),
                   {k: tuple(v) for k, v in (data.get('observed') or {}).items()})


def is_free_program(move):
    inner = getattr(move, 'move', move)
    if isinstance(inner, dict):
        return inner.get('family') == FAMILY
    return isinstance(inner, FreeProgramOption)


def option_from_dict(data):
    return FreeProgramOption.from_dict(data)


def extension_families(fp):
    return (FAMILY,) if fp is not None else ()


# ---------------------------------------------------------------- geometry helpers (observable rows)

def _drop(view):
    return float(view.lane_drop_s_m)


def _to_drop(view, row):
    return _drop(view)-float(row['s_m'])


def lanes_at(view, s_m):
    """Existing lanes at road position s (the closing lane ends at the drop)."""
    return tuple(l for l in view.lane_ids if l != view.dropped_lane or float(s_m) < _drop(view))


def adjacent_targets(view, row):
    """Adjacent existing lanes of a member (closing lane allowed while a request can still act)."""
    lane, s = int(row['lane']), float(row['s_m'])
    giveup = _drop(view)-registered('lane_giveup_to_drop_m')[0]
    out = []
    for target in (lane-1, lane+1):
        if target not in lanes_at(view, s):
            continue
        if (target == view.dropped_lane or lane == view.dropped_lane) and s >= giveup:
            continue   # a closing-lane change past the registered give-up position cannot act
        out.append(target)
    return out


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _observed(rows):
    return {r['vid']: (int(r['lane']), float(r['s_m'])) for r in rows}


def _option(rows, prog, cls, source):
    if prog is None:
        return None
    return FreeProgramOption(tuple(prog['members']), prog, cls, source, _observed(rows))


def _speed_member(view, *, v_w, hold_s, dmax, start_el=0, lane=None, tau=None, mode_on='first_active',
                  extra_releases=(), **kwargs):
    amax = registered('amax')[0]
    return member(lon('speed', v_w=float(v_w), amax=amax, dmax=float(dmax)), lane=lane, start_el=start_el,
                  hold_s=int(hold_s), release_s=_drop(view), tau=tau, mode_on=mode_on,
                  releases=('hold_end', 'release_s')+tuple(extra_releases), **kwargs)


# ---------------------------------------------------------------- structured families (registered)

def _in(view, row, window):
    lo, hi = window
    return lo <= _to_drop(view, row) <= hi


def _structured_groups(family, view, own, partners, by_vid):
    zone = registered('zone_to_drop_m')
    if family in ('closing_pair', 'abreast_wall'):
        from .lane_drop import joint_groups
        return [tuple(g) for g in joint_groups(family, view, own, partners)]
    pool = [own]+list(partners)
    if family == 'wall_fill':
        spread = registered('wall_spread_m')[0]
        out = []
        for group in combinations(pool, 3):
            if own not in group or not all(_in(view, r, zone) for r in group):
                continue
            s = [float(r['s_m']) for r in group]
            if max(s)-min(s) > spread:
                continue
            lanes = [int(r['lane']) for r in group]
            doubled = [l for l in set(lanes) if lanes.count(l) == 2]
            if len(set(lanes)) != 2 or not doubled:
                continue
            empty = [l for l in view.lane_ids if l not in lanes]
            for filler in (r for r in group if int(r['lane']) == doubled[0]):
                targets = [l for l in adjacent_targets(view, filler) if l in empty]
                # the empty lane must hold no other acquirable CAV that already makes a static wall
                rest = [r for r in partners if r not in group and int(r['lane']) in targets
                        and abs(float(r['s_m'])-sum(s)/3.) <= spread]
                if targets and not rest:
                    anchors = tuple(r for r in group if r is not filler)
                    out.append((anchors[0], anchors[1], filler, targets[0]))
        return out
    if family == 'gap_open_merge':
        lo, hi = registered('zipper_yield_rel_m')
        out = []
        for m in pool:
            if int(m['lane']) != view.dropped_lane or not _in(view, m, registered('zipper_merger_to_drop_m')):
                continue
            for g in pool:
                if g is m or int(g['lane']) != view.adjacent_lane or own not in (m, g):
                    continue
                if lo <= float(g['s_m'])-float(m['s_m']) <= hi:
                    out.append((m, g))
        return out
    if family == 'lane_redistribute':
        sensed = [r for r in by_vid.values()]
        out = []
        for size in registered('redistribute_size'):
            for group in combinations(pool, size):
                if own not in group or len({int(r['lane']) for r in group}) != 1:
                    continue
                if not all(_in(view, r, registered('redistribute_to_drop_m')) for r in group):
                    continue
                s = [float(r['s_m']) for r in group]
                if max(s)-min(s) > registered('r_spread_m')[0]:
                    continue
                centre, radius = sum(s)/len(s), registered('sensing_m')[0]
                load = {l: sum(1 for r in sensed if int(r['lane']) == l and abs(float(r['s_m'])-centre) <= radius)
                        for l in lanes_at(view, centre)}
                lane = int(group[0]['lane'])
                if load.get(lane) != max(load.values()):
                    continue   # the group must be in the locally most loaded lane
                targets = [l for l in adjacent_targets(view, group[0]) if all(
                    l in adjacent_targets(view, r) for r in group)]
                if not targets:
                    continue
                least = min(load[l] for l in targets)
                out.append((group, min(l for l in targets if load[l] == least)))
        return out
    raise ValueError(f'unregistered structured family {family!r}')


def _structured_program(family, view, group, rng):
    """One program of a structured family for a geometric group (role order fixed)."""
    drop = _drop(view)
    if family in ('closing_pair', 'abreast_wall'):
        from .lane_drop import VARIANTS
        names = sorted(n for n, v in VARIANTS.items() if v['family'] == family)
        v = VARIANTS[_pick(rng, names)]
        rows = list(group)
        extra, kw = (), {}
        if v['anchor']:
            extra = ('anchor',)
            kw = dict(anchor=rows[0]['vid'], anchor_release_s=drop-registered('anchor_release_m')[0])
        progs = [_speed_member(view, v_w=v['target_mps'], hold_s=v['hold'], dmax=v['depth'],
                               extra_releases=extra, role=family, **kw) for _ in rows]
        return rows, program([r['vid'] for r in rows], progs,
                             source=dict(family=family, variant=[n for n in names if VARIANTS[n] is v][0]))
    if family == 'wall_fill':
        a, b, f, target = group
        wh = registered('wall_h')[0]
        tr = registered('track')[0]
        hold = wh['hold_s']
        giveup = drop-registered('lane_giveup_to_drop_m')[0]
        anchors = [_speed_member(view, v_w=wh['v_w'], hold_s=hold, dmax=wh['dmax'], role='wall') for _ in (a, b)]
        filler = member(lon('track', v_w=wh['v_w'], amax=tr['amax'], dmax=tr['dmax'], refs=(a['vid'], b['vid']),
                            off=0., kp=tr['kp'], kv=tr['kv'], settle_gap=registered('settle_m')[0]),
                        lane=lane_req(target, giveup_s=giveup), hold_s=hold, release_s=drop,
                        role='filler', mode_on='on_action')
        rows, progs = [f, a, b], [filler, *anchors]
        return rows, program([r['vid'] for r in rows], progs, source=dict(family=family))
    if family == 'gap_open_merge':
        m, g = group
        tr = registered('track')[0]
        holds = registered('zipper_hold_s')[0]
        fallback = drop-registered('zipper_fallback_to_drop_m')[0]
        merger = member(lon('speed', v_w=registered('merge_v_w')[0], amax=registered('amax')[0],
                            dmax=_pick(rng, registered('dmax'))),
                        lane=lane_req(view.adjacent_lane, rule='from_lane', source_lane=view.dropped_lane),
                        hold_s=holds['merge'], merger_lane=view.dropped_lane, fallback_s=fallback,
                        releases=('hold_end', 'merged_zip', 'fallback'), role='zip_merge')
        gap = registered('track_gap_m')[0]
        maker = member(lon('track', amax=tr['amax'], dmax=tr['dmax'], refs=(m['vid'],), off=-gap,
                           kp=tr['kp'], kv=tr['kv']),
                       hold_s=holds['yield_'], partner=m['vid'], partner_lane=view.dropped_lane,
                       fallback_s=fallback, releases=('hold_end', 'partner_left', 'fallback'), role='zip_yield')
        return [m, g], program([m['vid'], g['vid']], [merger, maker], source=dict(family=family))
    if family == 'lane_redistribute':
        rows, target = group
        giveup = drop-registered('lane_giveup_to_drop_m')[0]
        hold = registered('redistribute_hold_s')[0]
        progs = [member(lon('match', amax=registered('amax')[0], dmax=_pick(rng, registered('dmax'))),
                        lane=lane_req(target, giveup_s=giveup), hold_s=hold, release_s=drop,
                        role='redistribute', mode_on='on_action') for _ in rows]
        return list(rows), program([r['vid'] for r in rows], progs, source=dict(family=family))
    raise ValueError(f'unregistered structured family {family!r}')


def _ego_first(rows, prog, ego):
    """Reorder so the acting CAV is the initiator (members[0]); programs follow their member."""
    if prog['members'][0] == ego:
        return rows, prog
    i = prog['members'].index(ego)
    order = [i]+[j for j in range(len(rows)) if j != i]
    out = dict(prog, members=[prog['members'][j] for j in order],
               programs=[prog['programs'][j] for j in order], join_el=[prog['join_el'][j] for j in order])
    return [rows[j] for j in order], out


def structured(family, view, own, partners, by_vid, rng):
    groups = _structured_groups(family, view, own, partners, by_vid)
    if not groups:
        return None
    rows, prog = _structured_program(family, view, groups[int(rng.integers(len(groups)))], rng)
    rows, prog = _ego_first(rows, prog, own['vid'])
    return _option(rows, prog, 'structured', dict(prog['source'], family=family))


# ---------------------------------------------------------------- random programs (registered grids)

def _random_member(view, row, rng, *, kind=None, refs=()):
    """lcm_R_v1 member (kind None): LC target uniform over {none, adjacent lanes incl. closing},
    LC offset/start offset {0,20,40,60}, v_w/dmax/duration from the R grid, own tau
    {none,0.6,2.0}. ``kind`` given: the random_prim class with that longitudinal kind."""
    offsets = registered('offsets_s')
    targets = [None]+adjacent_targets(view, row)
    target = _pick(rng, targets)
    giveup = _drop(view)-registered('lane_giveup_to_drop_m')[0]
    lane = None if target is None else lane_req(target, from_el=_pick(rng, offsets), giveup_s=giveup)
    tau = _pick(rng, registered('tau'))
    start = _pick(rng, offsets)
    hold = _pick(rng, registered('hold_s'))
    dmax = _pick(rng, registered('dmax'))
    amax = registered('amax')[0]
    if kind is None or kind == 'speed':
        law = lon('speed', v_w=_pick(rng, registered('v_w')), amax=amax, dmax=dmax)
    elif kind == 'track':
        tr = registered('track')[0]
        law = lon('track', v_w=_pick(rng, (None,)+registered('v_w')), amax=tr['amax'], dmax=tr['dmax'],
                  refs=refs, off=_pick(rng, registered('track_off_m')), kp=tr['kp'], kv=tr['kv'],
                  settle_gap=registered('settle_m')[0])
    elif kind == 'match':
        law = lon('match', amax=amax, dmax=dmax)
    elif kind == 'accel':
        pulse = registered('accel_pulse')[0]
        law, hold = lon('accel', a=pulse['a']), pulse['hold_s']
    else:
        law = lon(kind)
    if kind == 'match' and lane is None:
        if len(targets) == 1:
            return None          # speed matching is defined only towards a lane change target
        lane = lane_req(_pick(rng, targets[1:]), from_el=_pick(rng, offsets), giveup_s=giveup)
    if kind in ('base', 'native') and lane is None and tau is None:
        return None              # no primitive acts: not a program member (None-guard)
    return member(law, lane=lane, start_el=start, hold_s=int(hold), release_s=_drop(view), tau=tau,
                  mode_on='on_action', role='R' if kind is None else 'prim')


def random_program(cls, view, own, partners, rng, *, size=None):
    """One random program (``random_r`` = lcm_R_v1, ``random_prim`` = primitive coverage)."""
    pool = list(partners)
    if cls == 'random_r':
        zone = registered('zone_to_drop_m')
        if not _in(view, own, zone):
            return None
        pool = [p for p in pool if _in(view, p, zone)
                and abs(float(p['s_m'])-float(own['s_m'])) <= registered('r_spread_m')[0]]
        sizes = [k for k in registered('r_group_size') if k-1 <= len(pool) and k <= moves.MOVE_MAX_MEMBERS]
    elif cls == 'random_prim':
        sizes = [k for k in registered('member_arity') if k-1 <= len(pool) and k <= moves.MOVE_MAX_MEMBERS]
    else:
        raise ValueError('unknown random class')
    if size is not None:
        sizes = [size] if size in sizes else []
    if not sizes:
        return None
    k = _pick(rng, sizes)
    picked = [pool[int(i)] for i in sorted(rng.choice(len(pool), size=k-1, replace=False))] if k > 1 else []
    rows = [own]+picked
    progs = []
    for row in rows:
        if cls == 'random_r':
            p = _random_member(view, row, rng)
        else:
            kind = _pick(rng, LON_KINDS if len(rows) > 1 else tuple(k for k in LON_KINDS if k != 'track'))
            refs = tuple(r['vid'] for r in rows if r is not row)[:1] if kind == 'track' else ()
            p = _random_member(view, row, rng, kind=kind, refs=refs)
        if p is None:
            return None
        progs.append(p)
    prog = program([r['vid'] for r in rows], progs, source=dict(family='random', procedure=(
        'lcm_R_v1' if cls == 'random_r' else 'fullpol_prim_v1')))
    return _option(rows, prog, cls, dict(prog['source']))


# ---------------------------------------------------------------- verified templates (own arm)

TEMPLATE_SCHEMA = 'fullpol-template-v1'


def check_template(t):
    if (not isinstance(t, dict) or t.get('schema') != TEMPLATE_SCHEMA
            or t.get('program', {}).get('schema') != PROGRAM_SCHEMA):
        raise ValueError('fullpol template needs schema fullpol-template-v1 and an lcexec program')
    members = t['program']['members']
    if (len(t['member_lanes']) != len(members) or len(t['member_s']) != len(members)
            or any(type(l) is not int for l in t['member_lanes'])
            or any(not math.isfinite(float(s)) for s in t['member_s'])):
        raise ValueError('template roles need one WM lane and one WM s per member')
    if not isinstance(t.get('source'), dict) or not t['source'].get('verification_id'):
        raise ValueError('template needs its verification source')
    return t


def template(prog, *, member_lanes, member_s, source):
    """Converted verified program -> template (roles = WM lane + t0 position per member)."""
    return check_template(dict(schema=TEMPLATE_SCHEMA, program=deepcopy(prog), member_lanes=list(member_lanes),
                               member_s=[float(s) for s in member_s], source=deepcopy(source)))


def _bindings(t, own, partners):
    """Role assignments (template member -> acquirable row), ego bound, lanes equal, order kept."""
    lanes, s0 = t['member_lanes'], t['member_s']
    k = len(lanes)
    rows = [own]+list(partners)
    order = sorted(range(k), key=lambda i: -s0[i])            # template front-to-back
    out = []
    for combo in combinations(rows, k):
        if own not in combo:
            continue
        ranked = sorted(combo, key=lambda r: -float(r['s_m']))
        assign = {order[j]: ranked[j] for j in range(k)}
        if any(int(assign[i]['lane']) != int(lanes[i]) for i in range(k)):
            continue
        ref = float(assign[order[0]]['s_m'])-s0[order[0]]
        cost = sum(abs(float(assign[i]['s_m'])-ref-s0[i]) for i in range(k))
        out.append((cost, [assign[i] for i in range(k)]))
    out.sort(key=lambda c: (c[0], [r['vid'] for r in c[1]]))
    return [rows for _, rows in out]


def _past_release(prog, rows):
    """A member already at/after its own position release would release at once (None-guard)."""
    for p, r in zip(prog['programs'], rows):
        s, rel = float(r['s_m']), set(p['releases'])
        if ('release_s' in rel and p.get('release_s') is not None and s >= float(p['release_s'])) or (
                'fallback' in rel and p.get('fallback_s') is not None and s >= float(p['fallback_s'])) or (
                'zone' in rel and p.get('zone') is not None and s > float(p['zone'][1])):
            return True
    return False


def adapt(t, view, own, partners, rng):
    """Re-bind one verified template to the acting CAV's acquirable CAVs (or None)."""
    options = _bindings(t, own, partners)
    if not options:
        return None
    rows = options[0]
    mapping = {old: r['vid'] for old, r in zip(t['program']['members'], rows)}
    prog = rebind(t['program'], mapping)
    if _past_release(prog, rows):
        return None
    for p, r in zip(prog['programs'], rows):
        lane = p.get('lane')
        if lane is not None and int(lane['target']) not in lanes_at(view, r['s_m']) and lane['rule'] == 'until_reached':
            return None
    rows, prog = _ego_first(rows, prog, own['vid'])
    source = dict(t['source'], template=True)
    prog['source'] = dict(prog.get('source') or {}, **source)
    return _option(rows, prog, 'verified', source)


# ---------------------------------------------------------------- slate hooks (feasible_proposals)

def families(structured_keys, state, event):
    """Slate pool keys with one F2 slot per enabled program class (none on event ticks)."""
    if event or state.get('only_join'):
        return tuple(structured_keys)
    return tuple(structured_keys)+tuple(f'{FAMILY}:{c}' for c in state['classes'])


def owns(key):
    return key in SLOTS


def _draw(key, state, own, partners, by_vid, rng):
    view = state['view']
    cls = key.split(':', 1)[1]
    if cls == 'verified':
        ts = list(state.get('templates') or ())
        if not ts:
            return None
        for i in rng.permutation(len(ts)):
            option = adapt(ts[int(i)], view, own, partners, rng)
            if option is not None:
                return option
        return None
    if cls == 'structured':
        fams = list(state['structured'])
        for i in rng.permutation(len(fams)):
            option = structured(fams[int(i)], view, own, partners, by_vid, rng)
            if option is not None:
                return option
        return None
    return random_program(cls, view, own, partners, rng)


def pool(key, state, own, partners, by_vid, rng):
    """Feasible templates of one class slot ([] = class infeasible for this CAV now)."""
    option = _draw(key, state, own, partners, by_vid, rng)
    return [] if option is None else [option]


def redraw(template_option, key, state, own, partners, by_vid, rng):
    """Fresh draw of the same class; None-guard: the caller keeps the feasible template."""
    return _draw(key, state, own, partners, by_vid, rng)


def offered(state, own, partners, rng):
    """Practice 'offer, don't force': offered verified programs re-bound to this CAV (slot 1)."""
    out = []
    for t in state.get('offered') or ():
        option = adapt(t, state['view'], own, partners, rng)
        if option is not None:
            out.append(FreeProgramOption(option.members, option.program, 'offered', option.source,
                                         option.observed))
    return out


__all__ = ['FAMILY', 'RULE', 'CLASSES', 'SLOTS', 'STRUCTURED', 'REGISTERED', 'settings', 'config_keys',
           'runtime_state', 'FreeProgramOption', 'is_free_program', 'option_from_dict', 'program', 'member',
           'lon', 'lane_req', 'template', 'adapt', 'structured', 'random_program', 'families', 'owns', 'pool',
           'redraw', 'offered', 'adjacent_targets', 'lanes_at', 'rebind', 'lc_checked']


def propose(state, own, rng):
    """LCEXEC lc_families generator hook (config ``lc_families.generator =
    'tools.coopmwm.fullpol_programs:propose'``): clean5b slot templates, see fullpol_templates."""
    from .fullpol_templates import propose as templates
    return templates(state, own, rng)
