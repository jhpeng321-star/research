"""clean5b: FULLPOL free_program TEMPLATE generator for LCEXEC's per-CAV initiate/join protocol.

LCEXEC contract (lc_families, 2026-11-06 ~21:3x): a template is an lc_program.FreeProgram whose
members are slot names '@0' (the initiator = the acting CAV) .. '@k', one MemberProgram per slot
(anchor/partner/refs may name '@k'), and ``join_rules`` per slot k >= 1 = {lane: WM lane | None,
ds: [lo, hi]} (the joiner's own lane and its s relative to the initiator's message position,
|ds| <= the V2V range). The hook ``propose(state, own, rng) -> list[FreeProgram]`` is wired by
config ``lc_families.generator = 'tools.coopmwm.fullpol_programs:propose'``.

Classes (FULLPOL slate rule, every value from lc_program.REGISTRY / registrations):
* ``verified``   - the arm's own converted verified templates (lc_families.template_from_program),
                   passed in ``state['fullpol']['templates']``; used as-is (joiners match by rule);
* ``structured`` - abreast_wall (wall_g / wall_h), closing_pair (pair_g / pair_h / pair_t),
                   solo hold (wall_g / wall_h, 1 slot), wall_fill, gap_open_merge, lane_redistribute;
* ``random_r``   - LCMOVES lcm_R_v1 (2-4 slots, per member: lane target uniform over {none,
                   adjacent lanes incl. closing}, LC/start offsets {0,20,40,60}, R grid v_w / hold /
                   dmax, own tau {none, 0.6, 2.0}); joiner lanes drawn uniformly so lane targets are
                   well defined;
* ``random_prim``- uniform over the longitudinal primitive kinds (1-4 slots).

Observables only: own row + own sensing rows (HDV geometry for lane loads), the map frame.
Every builder may return None (N11 F8 None-guard). No regime/config label is ever read.
"""
from . import fullpol_programs as fp

CLASSES = ('verified', 'structured', 'random_r', 'random_prim')
STRUCTURED = ('abreast_wall', 'closing_pair', 'solo_hold', 'wall_fill', 'rolling_wall', 'gap_open_merge',
              'lane_redistribute')


def _lc():
    from . import lc_program
    return lc_program


def _geo(frame):
    """(drop_s, lanes, closing, adjacent) from an lc_program.Frame or an N2 MapView."""
    if hasattr(frame, 'sumo_to_wm'):
        return (float(frame.drop_s_m), tuple(sorted(set(frame.sumo_to_wm.values()))), int(frame.closing_lane),
                int(frame.adjacent_lane))
    return float(frame.lane_drop_s_m), tuple(frame.lane_ids), int(frame.dropped_lane), int(frame.adjacent_lane)


def _view(frame):
    from .lane_drop import MapView
    drop, lanes, closing, _ = _geo(frame)
    return MapView(getattr(frame, 'name', 'frame') or 'frame', drop, lanes, closing)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _ds(lo, hi, range_m):
    lo, hi = max(float(lo), -range_m), min(float(hi), range_m)
    return None if lo > hi else (lo, hi)


def _template(programs, rules, *, source, duration_s=None):
    lc = _lc()
    members = tuple(f'@{k}' for k in range(len(programs)))
    progs = tuple(lc.MemberProgram.from_dict(p) for p in programs)
    if duration_s is None:
        duration_s = max(int(p['start_el'])+int(p['hold_s'] or 0) for p in programs)
    return lc.FreeProgram(members, progs, join_el=(0,)*len(members), duration_s=max(1, int(duration_s)),
                          source=dict(source), join_rules=(None,)+tuple(rules))


def _speed(view, *, v_w, hold_s, dmax, extra=(), **kw):
    return fp._speed_member(view, v_w=v_w, hold_s=hold_s, dmax=dmax, extra_releases=extra, **kw)


def _in_zone(view, own):
    lo, hi = fp.registered('zone_to_drop_m')
    return lo <= fp._to_drop(view, own) <= hi


# ---------------------------------------------------------------- structured templates

def _wall_like(family, view, own, rng, range_m):
    from .lane_drop import VARIANTS
    if not _in_zone(view, own):
        return None
    names = sorted(n for n, v in VARIANTS.items() if v['family'] == ('abreast_wall' if family != 'closing_pair'
                                                                        else 'closing_pair'))
    name = _pick(rng, names)
    v = VARIANTS[name]
    lane = int(own['lane'])
    if family == 'solo_hold':
        from .fullpol_ev import sumo_lane
        prog = _speed(view, v_w=v['target_mps'], hold_s=v['hold'], dmax=v['depth'], role='solo')
        # lane-specific strategy label (S-REGIME wall_{g,h}_solo{SUMO lane of the initiator}); offline only
        return _template([prog], [], source=dict(family='solo_hold', variant=f'{name}_solo{sumo_lane(view, lane)}'))
    if family == 'abreast_wall':
        others = [ln for ln in view.lane_ids if ln != lane]
        spread = fp.registered('wall_spread_m')[0]
        ds = _ds(-spread, spread, range_m)
        if ds is None:
            return None
        progs = [_speed(view, v_w=v['target_mps'], hold_s=v['hold'], dmax=v['depth'], role='wall')
                 for _ in range(1+len(others))]
        return _template(progs, [dict(lane=ln, ds=ds) for ln in others], source=dict(family=family, variant=name))
    # closing_pair: (closing-lane CAV, adjacent-lane CAV); the initiator takes its own role
    if lane not in (view.dropped_lane, view.adjacent_lane):
        return None
    other = view.adjacent_lane if lane == view.dropped_lane else view.dropped_lane
    dx = fp.registered('pair_dx_m')[0]
    ds = _ds(-dx, dx, range_m)
    if ds is None:
        return None
    anchor_slot = '@0' if lane == view.dropped_lane else '@1'
    kw, extra = {}, ()
    if v['anchor']:
        extra, kw = ('anchor',), dict(anchor=anchor_slot, anchor_release_s=fp._drop(view)-fp.registered('anchor_release_m')[0])
    progs = [_speed(view, v_w=v['target_mps'], hold_s=v['hold'], dmax=v['depth'], extra=extra, role='pair', **kw)
             for _ in range(2)]
    return _template(progs, [dict(lane=other, ds=ds)], source=dict(family=family, variant=name))


def _wall_fill(view, own, rows, rng, range_m):
    """Initiator = filler F in a lane it shares with a wall member; F changes into the empty adjacent
    lane (closing allowed) and slot-tracks the mean of @1 (same lane) and @2 (third lane)."""
    if not _in_zone(view, own):
        return None
    lane = int(own['lane'])
    spread = fp.registered('wall_spread_m')[0]
    cavs = [r for r in rows if r['is_cav'] and r['vid'] != own['vid']]
    targets = [ln for ln in fp.adjacent_targets(view, own)
               if not any(int(r['lane']) == ln and abs(float(r['s_m'])-float(own['s_m'])) <= spread for r in cavs)]
    if not targets:
        return None
    empty = _pick(rng, targets)
    third = [ln for ln in view.lane_ids if ln not in (lane, empty)]
    if not third:
        return None
    ds = _ds(-spread, spread, range_m)
    wh, tr = fp.registered('wall_h')[0], fp.registered('track')[0]
    giveup = fp._drop(view)-fp.registered('lane_giveup_to_drop_m')[0]
    filler = fp.member(fp.lon('track', v_w=wh['v_w'], amax=tr['amax'], dmax=tr['dmax'], refs=('@1', '@2'), off=0.,
                              kp=tr['kp'], kv=tr['kv'], settle_gap=fp.registered('settle_m')[0]),
                       lane=fp.lane_req(empty, giveup_s=giveup), hold_s=wh['hold_s'], release_s=fp._drop(view),
                       role='filler', mode_on='on_action')
    anchors = [_speed(view, v_w=wh['v_w'], hold_s=wh['hold_s'], dmax=wh['dmax'], role='wall') for _ in range(2)]
    return _template([filler]+anchors, [dict(lane=lane, ds=ds), dict(lane=third[0], ds=ds)],
                     source=dict(family='wall_fill'))


def _rolling_wall(view, own, rng, range_m):
    """S-LCMOVES rolling_wall with the initiator as the rear R: front A, B (distinct lanes) 200-500 m
    ahead hold 15 m/s (dmax 1.5, 120 s); R slot-tracks their mean (amax 1.0), changes into the lane
    the front leaves uncovered when its own lane is covered, settles within 25 m, then holds 15.
    The front window must intersect the V2V range (None otherwise: not representable)."""
    lo, hi = fp.registered('rolling_rear_m')
    ds = _ds(lo, hi, range_m)
    if ds is None or not _in_zone(view, own):
        return None
    lane = int(own['lane'])
    lanes = fp.lanes_at(view, own['s_m'])
    options = []
    for uncovered in lanes:
        front = [ln for ln in lanes if ln != uncovered]
        if len(front) != 2:
            continue
        if uncovered == lane:
            options.append((front, None))
        elif uncovered in fp.adjacent_targets(view, own):
            options.append((front, uncovered))
    if not options:
        return None
    front, target = options[int(rng.integers(len(options)))]
    h, tr = fp.registered('rolling_hold')[0], fp.registered('track')[0]
    giveup = fp._drop(view)-fp.registered('lane_giveup_to_drop_m')[0]
    rear = fp.member(fp.lon('track', v_w=h['v_w'], amax=fp.registered('amax')[0], dmax=tr['dmax'], refs=('@1', '@2'),
                            off=0., kp=tr['kp'], kv=tr['kv'], settle_gap=fp.registered('settle_m')[0]),
                     lane=None if target is None else fp.lane_req(target, giveup_s=giveup), hold_s=h['hold_s'],
                     release_s=fp._drop(view), role='rolling_rear', mode_on='on_action')
    fronts = [_speed(view, v_w=h['v_w'], hold_s=h['hold_s'], dmax=h['dmax'], role='rolling_front') for _ in front]
    return _template([rear]+fronts, [dict(lane=ln, ds=ds) for ln in front], source=dict(family='rolling_wall'))


def _gap_open_merge(view, own, rng, range_m):
    lane = int(own['lane'])
    lo, hi = fp.registered('zipper_yield_rel_m')
    holds, tr = fp.registered('zipper_hold_s')[0], fp.registered('track')[0]
    fallback = fp._drop(view)-fp.registered('zipper_fallback_to_drop_m')[0]
    mlo, mhi = fp.registered('zipper_merger_to_drop_m')
    merger = fp.member(fp.lon('speed', v_w=fp.registered('merge_v_w')[0], amax=fp.registered('amax')[0],
                              dmax=_pick(rng, fp.registered('dmax'))),
                       lane=fp.lane_req(view.adjacent_lane, rule='from_lane', source_lane=view.dropped_lane),
                       hold_s=holds['merge'], merger_lane=view.dropped_lane, fallback_s=fallback,
                       releases=('hold_end', 'merged_zip', 'fallback'), role='zip_merge')
    gap = fp.registered('track_gap_m')[0]

    def maker(ref):
        return fp.member(fp.lon('track', amax=tr['amax'], dmax=tr['dmax'], refs=(ref,), off=-gap, kp=tr['kp'],
                                kv=tr['kv']), hold_s=holds['yield_'], partner=ref, partner_lane=view.dropped_lane,
                         fallback_s=fallback, releases=('hold_end', 'partner_left', 'fallback'), role='zip_yield')
    if lane == view.dropped_lane and mlo <= fp._to_drop(view, own) <= mhi:
        ds = _ds(lo, hi, range_m)                  # yielder relative to the merger (initiator)
        return None if ds is None else _template([merger, maker('@0')], [dict(lane=view.adjacent_lane, ds=ds)],
                                                 source=dict(family='gap_open_merge', role='merger'))
    if lane == view.adjacent_lane:
        ds = _ds(-hi, -lo, range_m)                # merger relative to the yielder (initiator)
        to_drop = fp._to_drop(view, own)
        if ds is None or not mlo-hi <= to_drop <= mhi-lo:
            return None
        return _template([maker('@1'), merger], [dict(lane=view.dropped_lane, ds=ds)],
                         source=dict(family='gap_open_merge', role='yielder'))
    return None


def _lane_redistribute(view, own, rows, rng, range_m):
    lo, hi = fp.registered('redistribute_to_drop_m')
    if not lo <= fp._to_drop(view, own) <= hi:
        return None
    radius = fp.registered('sensing_m')[0]
    lanes = fp.lanes_at(view, own['s_m'])
    load = {ln: sum(1 for r in rows if int(r['lane']) == ln and abs(float(r['s_m'])-float(own['s_m'])) <= radius)
            for ln in lanes}
    lane = int(own['lane'])
    if load.get(lane) != max(load.values()):
        return None
    targets = fp.adjacent_targets(view, own)
    if not targets:
        return None
    least = min(load[ln] for ln in targets)
    target = min(ln for ln in targets if load[ln] == least)
    size = _pick(rng, fp.registered('redistribute_size'))
    ds = _ds(-fp.registered('r_spread_m')[0], fp.registered('r_spread_m')[0], range_m)
    giveup = fp._drop(view)-fp.registered('lane_giveup_to_drop_m')[0]
    progs = [fp.member(fp.lon('match', amax=fp.registered('amax')[0], dmax=_pick(rng, fp.registered('dmax'))),
                       lane=fp.lane_req(target, giveup_s=giveup), hold_s=fp.registered('redistribute_hold_s')[0],
                       release_s=fp._drop(view), role='redistribute', mode_on='on_action') for _ in range(size)]
    return _template(progs, [dict(lane=lane, ds=ds)]*(size-1), source=dict(family='lane_redistribute'))


def structured(family, view, own, rows, rng, range_m):
    if family in ('abreast_wall', 'closing_pair', 'solo_hold'):
        return _wall_like(family, view, own, rng, range_m)
    if family == 'wall_fill':
        return _wall_fill(view, own, rows, rng, range_m)
    if family == 'rolling_wall':
        return _rolling_wall(view, own, rng, range_m)
    if family == 'gap_open_merge':
        return _gap_open_merge(view, own, rng, range_m)
    if family == 'lane_redistribute':
        return _lane_redistribute(view, own, rows, rng, range_m)
    raise ValueError(f'unregistered structured template family {family!r}')


# ---------------------------------------------------------------- random templates

def _slot_row(view, own, lane, ds):
    """A representative joiner row (lane, s = own s + window centre) for lane-target feasibility."""
    return dict(vid='@', lane=lane, s_m=float(own['s_m'])+sum(ds)/2., v_mps=own['v_mps'], is_cav=True)


def random_template(cls, view, own, rng, range_m):
    if cls == 'random_r':
        if not _in_zone(view, own):
            return None
        sizes = fp.registered('r_group_size')
        spread = fp.registered('r_spread_m')[0]
    else:
        sizes = fp.registered('member_arity')
        spread = fp.registered('r_spread_m')[0]
    k = _pick(rng, sizes)
    ds = _ds(-spread, spread, range_m)
    if ds is None:
        return None
    slot_rows = [own]+[_slot_row(view, own, _pick(rng, fp.lanes_at(view, own['s_m'])), ds) for _ in range(k-1)]
    progs = []
    for i, row in enumerate(slot_rows):
        if cls == 'random_r':
            p = fp._random_member(view, row, rng)
        else:
            kinds = fp.LON_KINDS if k > 1 else tuple(x for x in fp.LON_KINDS if x != 'track')
            kind = _pick(rng, kinds)
            refs = (('@1' if i == 0 else '@0'),) if kind == 'track' else ()
            p = fp._random_member(view, row, rng, kind=kind, refs=refs)
        if p is None:
            return None
        progs.append(p)
    rules = [dict(lane=int(r['lane']), ds=ds) for r in slot_rows[1:]]
    return _template(progs, rules, source=dict(family='random', procedure=(
        'lcm_R_v1' if cls == 'random_r' else 'fullpol_prim_v1')))


# ---------------------------------------------------------------- the LCEXEC hook

def propose(state, own, rng):
    """LCEXEC lc_families generator hook -> list of lc_program.FreeProgram templates.

    ``state``: LCEXEC's per-CAV generator state (now, own row, own sensing ``rows``, ``frame``)
    plus FULLPOL's common ``fullpol`` slot {classes, structured, templates (this arm's own
    verified templates), range_m}. One draw per enabled class in a random class order; a class
    whose draw is infeasible contributes nothing (None-guard)."""
    slot = state.get('fullpol') or {}
    classes = tuple(slot.get('classes', CLASSES))
    families = tuple(slot.get('structured', STRUCTURED))
    range_m = float(slot.get('range_m', state.get('range_m', fp.registered('sensing_m')[0])))
    view = _view(state['frame'])
    rows = [r for r in state.get('rows', ()) if r.get('vid') != own['vid']]
    out = []
    for i in rng.permutation(len(classes)):
        cls = classes[int(i)]
        if cls == 'verified':
            # clean5c bank entries {template, offer_condition}: offered only where their verified trigger and
            # offer condition hold for this CAV now (own row + own sensing); plain template dicts as before.
            from . import fullpol_bank as fb
            ledger = state.get('ledger')

            def gate(x):
                fam = x['template'].get('source', {}).get('variant')
                entry = False if ledger is None else ledger.get(fam)
                return (fb.session_ok(x['offer_condition'], entry, state.get('now'))
                        and fb.offer_ok(x['offer_condition'], own, rows))
            own_t = slot.get('templates') or ()
            if isinstance(own_t, dict):          # clean5c per-arm {B: [...], R: []}: unknown arm -> none
                own_t = own_t.get(state.get('arm')) or ()
            ts = [x['template'] if 'offer_condition' in x else x for x in own_t
                  if 'offer_condition' not in x or gate(x)]
            t = _lc().FreeProgram.from_dict(ts[int(rng.integers(len(ts)))]) if ts else None
        elif cls == 'structured':
            t = None
            for j in rng.permutation(len(families)):
                t = structured(families[int(j)], view, own, rows, rng, range_m)
                if t is not None:
                    break
        else:
            t = random_template(cls, view, own, rng, range_m)
        if t is not None:
            out.append(t)
    return out


def labels(template):
    """Offline strategy labels of a template (EV matching only; never a policy input)."""
    src = dict(template.source if hasattr(template, 'source') else template.get('source') or {})
    return tuple(x for x in (src.get('variant'), src.get('family')) if x)
