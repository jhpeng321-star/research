"""AMENDMENT 75 (S-HOLDFORMS; MAIN 2026-11-13; pre-outcome, zero SUMO): hold-form VARIANT of the A44 executor.

Implementation change only. Spec: E:/research-dense/experiments/a75_holdforms/SPEC_HOLDFORMS.md. A separate executor
with its own digest (``a75_digest``); a44_lcvariant / a37_cellbank / oracle_plan / a32_arms are imported, never edited.

A plan with none of the new fields ("plain") is admitted, keyed and executed by the inherited A44 code objects
(V.LcStrategyCore, A32.AlwaysCore._gate, O.select_members): same commands, same lane requests, same strategy_key
(test_a75_holdforms_cpu). Only "extended" plans (any new field, or a zone outside oracle_plan.ZONES) run
``HoldFormCore`` + ``select_members`` below:
- short holds: hold_s 1..900 (no code change needed; range check only);
- zones: named 'mid' / 'merge' or parametric ``zone_d`` = [d_up, d_down] m upstream of X_B (zone 'custom');
- role ``start='gap'``: the member is armed (no command) from onset_s for gap_arm_s; a hold event fires when a
  closing-lane vehicle (SUMO lane 0 = WM lane 2, upstream of the drop) that THIS member detects with its own 100 m /
  +-1 lane sensing lies at dx = x_veh - x_cav in [gap_dx_lo, gap_dx_hi] (and within gap_drop_m of X_B when set);
  event length hold_s ('timer') or until the condition clears ('clear', <= hold_s); up to gap_events per session;
- ``dv_rel`` (target = max(v_w, v0 - dv_rel)), ``taper_in_s`` / ``taper_out_s`` (linear target ramps from / back to
  v0), ``pulse_on_s`` / ``pulse_off_s`` (law issued only in on-seconds; off-seconds = no override).
Law, single writer, lane pin, release at the drop, sessions / cooldown / slots / busy rule: unchanged.
Longitudinal requests go to CommandOwnership exactly as A37/A44 (``desired[vid] = requested_raw``); HDVs are never
commanded (gap inputs only READ detected neighbours' x / lane). No RNG, no clock.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter

from . import a32_arms as A32
from . import a37_cellbank as C
from . import a44_lcvariant as V
from . import oracle_plan as O

SCHEMA = 'coopmwm.a75_holdforms.v1'
A75_ARMS = ('A75_PLAN', 'A75_BANK')
ZONES = dict(O.ZONES, mid=(O.G.X_B - 1400.0, O.G.X_B - 600.0), merge=(O.G.X_B - 600.0, O.G.X_B - 100.0))
ZONE_D_MAX = 2400.0
#: new role fields and their defaults (default = today's behaviour; a default-valued field is never written)
ROLE_DEFAULTS = dict(start='onset', gap_dx_lo=None, gap_dx_hi=None, gap_drop_m=None, gap_arm_s=None,
                     gap_release='timer', gap_events=1, dv_rel=None, taper_in_s=0, taper_out_s=0,
                     pulse_on_s=None, pulse_off_s=None)
ROLE_TYPES = dict(start=str, gap_dx_lo=float, gap_dx_hi=float, gap_drop_m=float, gap_arm_s=int, gap_release=str,
                  gap_events=int, dv_rel=float, taper_in_s=int, taper_out_s=int, pulse_on_s=int, pulse_off_s=int)
OLD_ROLE_KEYS = ('lane', 'rank', 'v_w', 'dmax', 'hold_s', 'onset_s', 'lc_target')
HOLD_S_RANGE = (1, 900)
V_W_MAX = 33.0
GAP_SUMO_LANE = 1                       # the only role lane whose +-1 lane sensing covers the closing lane
CLOSING_WM_LANE = 2                     # = SUMO lane 0
SENSE_M = O.SENSOR_RANGE_M              # 100 m project sensing
SENSE_LANES = 1
GAP_DX_MAX = SENSE_M
HOLDFORM_REFUSALS = ('bad_holdform', 'gap_role_lane', 'lc_with_holdform', 'bad_zone')
REFUSALS = V.REFUSALS + HOLDFORM_REFUSALS


# ------------------------------------------------------------------------------------------------ plan schema

def _cast(k, v):
    return None if v is None else ROLE_TYPES[k](v)


def canonical(spec):
    """Copy of a strategy spec with new role fields cast to their types and default-valued ones dropped; zone_d
    as [float, float] (dropped when None). A plain spec comes back unchanged (same dict content)."""
    s = dict(spec)
    roles = []
    for r in spec['roles']:
        r2 = {k: v for k, v in r.items() if k not in ROLE_DEFAULTS}
        for k in ROLE_DEFAULTS:
            if k in r:
                v = _cast(k, r[k])
                if v != ROLE_DEFAULTS[k]:
                    r2[k] = v
        roles.append(r2)
    s['roles'] = roles
    if s.get('zone_d') is not None:
        s['zone_d'] = [float(x) for x in s['zone_d']]
    else:
        s.pop('zone_d', None)
    return s


def is_extended(spec):
    return (spec.get('zone_d') is not None or spec.get('zone') not in O.ZONES
            or any(k in r for r in spec['roles'] for k in ROLE_DEFAULTS))


def strategy_key(spec):
    """oracle_plan.strategy_key (identical for plain specs) + zone_d when set; roles carry their new fields."""
    body = {k: spec[k] for k in ('trigger', 'zone', 'spread_m', 'cooldown_s', 'slots', 'roles')}
    for k in ('gate', 'start_gate', 'phases', 'zone_d'):
        if spec.get(k) is not None:
            body[k] = spec[k]
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def spec_of(family, cond):
    """oracle_plan.spec_of + the new role fields / zone_d present in the offer_condition (canonical form)."""
    s = O.spec_of(family, cond)
    for r, c in zip(s['roles'], cond['roles']):
        for k in ROLE_DEFAULTS:
            if k in c:
                r[k] = c[k]
    if cond.get('zone_d') is not None:
        s['zone_d'] = cond['zone_d']
    return canonical(s)


def zone_bounds(s):
    if s.get('zone_d') is not None:
        d_up, d_down = s['zone_d']
        return (O.G.X_B - float(d_up), O.G.X_B - float(d_down))
    return ZONES[s['zone']]


def _int_ok(v, lo, hi=None):
    return isinstance(v, int) and not isinstance(v, bool) and v >= lo and (hi is None or v <= hi)


def holdform_refusal(spec):
    """None when the spec's hold forms / zone are executable, else one of HOLDFORM_REFUSALS."""
    zd = spec.get('zone_d')
    if zd is not None:
        if spec.get('zone') != 'custom' or len(zd) != 2 or not 0.0 <= float(zd[1]) < float(zd[0]) <= ZONE_D_MAX:
            return 'bad_zone'
    elif spec.get('zone') not in ZONES:
        return 'bad_zone'
    ext = is_extended(spec)
    for r in spec['roles']:
        if set(r) - set(OLD_ROLE_KEYS) - set(ROLE_DEFAULTS):
            return 'bad_holdform'
        if not _int_ok(r['hold_s'], *HOLD_S_RANGE) or not _int_ok(r['onset_s'], 0):
            return 'bad_holdform'
        if not 0.0 <= float(r['v_w']) <= V_W_MAX or not float(r['dmax']) > 0.0:
            return 'bad_holdform'
        f = dict(ROLE_DEFAULTS, **{k: r[k] for k in ROLE_DEFAULTS if k in r})
        if ext and any(x.get('lc_target') is not None for x in spec['roles']):
            return 'lc_with_holdform'
        if f['start'] not in ('onset', 'gap') or f['gap_release'] not in ('timer', 'clear'):
            return 'bad_holdform'
        gap_keys = ('gap_dx_lo', 'gap_dx_hi', 'gap_drop_m', 'gap_arm_s', 'gap_release', 'gap_events')
        if f['start'] == 'gap':
            if int(r['lane']) != GAP_SUMO_LANE:
                return 'gap_role_lane'
            if f['gap_dx_lo'] is None or f['gap_dx_hi'] is None or f['gap_arm_s'] is None:
                return 'bad_holdform'
            if not -GAP_DX_MAX <= float(f['gap_dx_lo']) < float(f['gap_dx_hi']) <= GAP_DX_MAX:
                return 'bad_holdform'
            if not _int_ok(f['gap_arm_s'], 1) or not _int_ok(f['gap_events'], 1):
                return 'bad_holdform'
            if f['gap_drop_m'] is not None and not float(f['gap_drop_m']) > 0.0:
                return 'bad_holdform'
        elif any(f[k] != ROLE_DEFAULTS[k] for k in gap_keys):
            return 'bad_holdform'
        if f['dv_rel'] is not None and not float(f['dv_rel']) > 0.0:
            return 'bad_holdform'
        if not _int_ok(f['taper_in_s'], 0) or not _int_ok(f['taper_out_s'], 0) \
                or f['taper_in_s'] + f['taper_out_s'] > r['hold_s']:
            return 'bad_holdform'
        if (f['pulse_on_s'] is None) != (f['pulse_off_s'] is None):
            return 'bad_holdform'
        if f['pulse_on_s'] is not None and not (_int_ok(f['pulse_on_s'], 1) and _int_ok(f['pulse_off_s'], 1)):
            return 'bad_holdform'
    return None


def executor_refusal(spec):
    """V.executor_refusal for plain specs (after the hold-form checks); extended specs: the same gate / trigger
    rules with the A75 zones."""
    s = canonical(spec)
    why = holdform_refusal(s)
    if why is not None:
        return why
    if not is_extended(s):
        return V.executor_refusal(s)
    if s.get('gate') not in O.GATES or s.get('start_gate') not in O.START_GATES or s.get('phases') is not None:
        return 'gate_or_phases'
    if s.get('trigger') not in ('wall', 'lanes'):
        return 'trigger_or_zone'
    return None


def a75_bank_plans(bank_doc, cell_key):
    """a44_lcvariant.lc_bank_plans (plan_source None) with the A75 spec / key / refusals; plan['extended']."""
    rows = [r for r in bank_doc['rows'] if r.get('regime') == cell_key]
    tpls = {(t['cell'], t['family']): t for t in (bank_doc.get('clean5c') or {}).get('templates', ())}
    plans, refused = [], {k: [] for k in REFUSALS}
    for r in rows:
        fam = r['family']
        t = tpls.get((cell_key, fam))
        if t is None:
            refused['no_template'].append(fam)
            continue
        cond = t['offer_condition']
        sit = cond.get('situation') or {}
        unconditional = O.row_is_unconditional(r, cond)
        if not unconditional and (sit.get('mode') != 'privileged' or not sit.get('bins')
                                  or sit.get('rule') != O.SITUATION_RULE):
            refused['bins_not_privileged'].append(fam)
            continue
        spec = spec_of(fam, cond)
        key = ((t['template'].get('source') or {}).get('strategy_key'))
        if key is None or strategy_key(spec) != key:
            refused['strategy_key_mismatch'].append(fam)
            continue
        why = holdform_refusal(spec) or V.lc_refusal(spec)
        if why is not None:
            refused[why].append(fam)
            continue
        if spec['gate'] not in O.GATES or spec['start_gate'] not in O.START_GATES or spec['phases'] is not None:
            refused['gate_or_phases'].append(fam)
            continue
        plans.append(dict(family=fam, lb95=float(r['lb95_veh_s']), key=key, spec=spec, extended=is_extended(spec),
                          row_class='unconditional' if unconditional else 'situational',
                          bins=None if unconditional else [dict(cong=b['cong'], lane_diff=b['lane_diff'])
                                                           for b in sit['bins']],
                          plan_source=r.get('plan_source'), template_sha256=r.get('template_sha256')))
    plans.sort(key=lambda p: (p['row_class'] != 'situational', -p['lb95'], p['key'], p['family']))
    return plans, {k: v for k, v in refused.items() if v}


# ------------------------------------------------------------------------------------------------ pure laws

def select_members(s, cavs, observed=None, phase=0):
    """oracle_plan.select_members with the zone window from zone_bounds (A75 zones / zone_d); otherwise the same."""
    seen_rows = list(cavs) + list(observed or [])
    if not O.gate_open(s, seen_rows):
        return None
    roles = O.active_roles(s, phase)
    lo, hi = zone_bounds(s)
    z = [c for c in cavs if c['edge'] == 'up' and lo <= float(c['x']) <= hi]
    if s['trigger'] == 'wall':
        wall = O._pick_wall(z)
        if wall is None:
            return None
        out = [wall[r['lane']] for r in roles]
    else:
        out = []
        for r in roles:
            lane = sorted((c for c in z if int(c['lane']) == r['lane']), key=lambda c: (-float(c['x']), c['id']))
            if len(lane) <= r['rank']:
                return None
            out.append(lane[r['rank']])
        if len({c['id'] for c in out}) != len(out):
            return None
        xs = [float(c['x']) for c in out]
        if max(xs) - min(xs) > float(s['spread_m']):
            return None
    sg = O.START_GATES[s.get('start_gate')]
    if sg is not None and any(O.behind_count(seen_rows, c, sg['behind_m']) > sg['max_behind'] for c in out):
        return None
    return out


def role_field(role, k):
    return role.get(k, ROLE_DEFAULTS[k])


def gap_candidate(role, own, rows):
    """(vid, dx) of the closing-lane vehicle (nearest |dx|, then vid) that the member ``own`` (WM row) detects with
    its own sensing (|ds| <= 100 m, |d lane| <= 1) inside the role window, else None. Vehicle type is never read."""
    s, lane = float(own['s_m']), int(own['lane'])
    lo, hi = float(role['gap_dx_lo']), float(role['gap_dx_hi'])
    drop_m = role_field(role, 'gap_drop_m')
    best = None
    for r in rows:
        if r['vid'] == own['vid'] or int(r['lane']) != CLOSING_WM_LANE or float(r['s_m']) >= O.RELEASE_S:
            continue
        dx = float(r['s_m']) - s
        if abs(dx) > SENSE_M or abs(int(r['lane']) - lane) > SENSE_LANES or not lo <= dx <= hi:
            continue
        if drop_m is not None and O.G.DROP_S - float(r['s_m']) > float(drop_m):
            continue
        key = (abs(dx), str(r['vid']))
        if best is None or key < best[0]:
            best = (key, (r['vid'], dx))
    return None if best is None else best[1]


def hold_target(role, h, v0):
    """Target speed at hold second h (0-based) of a hold that started at speed v0."""
    v_w = float(role['v_w'])
    dv = role_field(role, 'dv_rel')
    tgt = v_w if dv is None else max(v_w, float(v0) - float(dv))
    f = 1.0
    t_in, t_out = role_field(role, 'taper_in_s'), role_field(role, 'taper_out_s')
    if t_in:
        f = min(f, (h + 1) / float(t_in))
    if t_out:
        f = min(f, max(0, int(role['hold_s']) - 1 - h) / float(t_out))
    return tgt if f >= 1.0 else float(v0) + (tgt - float(v0)) * f


def pulse_on(role, h):
    on, off = role_field(role, 'pulse_on_s'), role_field(role, 'pulse_off_s')
    return on is None or (h % (int(on) + int(off))) < int(on)


def law(role, target, v):
    """oracle_plan StrategyCore.law with the target speed in place of v_w."""
    return max(-float(role['dmax']), min(1.0, float(target) - float(v)))


class HoldFormCore(V.LcStrategyCore):
    """StrategyCore session / slot / cooldown / release rules with the A75 hold timeline per member (extended plans
    only; never an lc role). Returns ({vid: requested accel}, {vid: own WM lane}) like StrategyCore.step."""

    def __init__(self, strategy, warmup_s=300):
        super().__init__(strategy, warmup_s)
        self.events = []                 # one record per hold event (onset holds and fired gap holds)
        self.sessions = []               # one record per member session

    def _end_event(self, m, t, why):
        ev = m['ev']
        if ev is not None:
            ev.update(end_t=t, end_why=why)
            m['ev'] = None

    def _release(self, vid, m, el, t, why):
        self._end_event(m, t, why)
        m['act']['release'][vid] = [el, why]
        m['sess'].update(released=[el, why])
        self.active.pop(vid)

    def _member(self, vid, m, row, rows, commands, t, accel, lanes):
        el = t - m['start']
        role = m['role']
        onset = int(role['onset_s'])
        if row is None or vid not in commands:
            return self._release(vid, m, el, t, 'left')
        gap = role_field(role, 'start') == 'gap'
        if not gap:
            if el >= onset + int(role['hold_s']):
                return self._release(vid, m, el, t, 'hold_end')
        else:
            ev = m['ev']
            if ev is not None:
                h = t - ev['fire_t']
                if h >= int(role['hold_s']):
                    self._end_event(m, t, 'timer')
                elif role_field(role, 'gap_release') == 'clear' and h >= 1 and gap_candidate(role, row, rows) is None:
                    self._end_event(m, t, 'clear')
            if m['ev'] is None:
                if m['n_ev'] >= role_field(role, 'gap_events'):
                    return self._release(vid, m, el, t, 'hold_end')
                if el >= onset + int(role['gap_arm_s']):
                    return self._release(vid, m, el, t, 'gap_arm_end')
        if float(row['s_m']) >= O.RELEASE_S:
            return self._release(vid, m, el, t, 'at_drop')
        lanes[vid] = int(row['lane'])
        if el < onset:
            return None
        if gap and m['ev'] is None:
            hit = gap_candidate(role, row, rows)
            if hit is None:
                return None
            m['n_ev'] += 1
            m['ev'] = dict(family=self.s['name'], vid=vid, start_kind='gap', armed_t=m['start'] + onset, fire_t=t,
                           end_t=None, end_why=None, trigger_vid=hit[0], dx_m=round(hit[1], 2),
                           v0=float(row['v_mps']), issued_s=0)
            self.events.append(m['ev'])
        if not gap and m['ev'] is None and el == onset:
            m['ev'] = dict(family=self.s['name'], vid=vid, start_kind='onset', armed_t=m['start'] + onset,
                           fire_t=t, end_t=None, end_why=None, trigger_vid=None, dx_m=None,
                           v0=float(row['v_mps']), issued_s=0)
            self.events.append(m['ev'])
        ev = m['ev']
        if ev is None:
            return None
        h = t - ev['fire_t']
        if not pulse_on(role, h):
            return None
        accel[vid] = law(role, hold_target(role, h, ev['v0']), row['v_mps'])
        ev['issued_s'] += 1
        self.issued.append([t, vid, round(accel[vid], 4)])
        return None

    def step(self, rows, commands, t):
        by = {r['vid']: r for r in rows if r.get('is_cav')}
        accel, lanes = {}, {}
        if t < self.warmup:
            return accel, lanes
        for vid, m in list(self.active.items()):
            self._member(vid, m, by.get(vid), rows, commands, t, accel, lanes)
        busy = len({id(m['act']) for m in self.active.values()})
        if t >= self.next_ok and busy < int(self.s.get('slots', 1)):
            free = [O.to_sumo_row(r) for vid, r in by.items() if vid not in self.active and vid in commands]
            phase = len(self.acts)
            mem = self.select_members(self.s, free, O.observed_rows(rows), phase=phase)
            if mem:
                act = dict(t=t, strategy=self.s['name'], members=[c['id'] for c in mem], phase=phase,
                           sumo_lanes=[c['lane'] for c in mem], x=[round(c['x'], 2) for c in mem], release={})
                self.acts.append(act)
                self.next_ok = t + int(self.s['cooldown_s'])
                for role, c in zip(O.active_roles(self.s, phase), mem):
                    sess = dict(family=self.s['name'], vid=c['id'], start=t,
                                start_kind=role_field(role, 'start'), released=None)
                    self.sessions.append(sess)
                    self.active[c['id']] = dict(start=t, role=role, act=act, ev=None, n_ev=0, sess=sess)
                    self._member(c['id'], self.active[c['id']], by[c['id']], rows, commands, t, accel, lanes)
        self.lane_held += len(lanes)
        return accel, lanes


def _form_gate(self, core, plan):
    """a32_arms.AlwaysCore._gate with the A75 select_members (extended plans only)."""
    def select(s, free, observed, phase=0):
        mem = select_members(s, free, observed, phase=phase)
        if not mem:
            return None
        if any(c['id'] in self._busy for c in mem):     # a member is in another family's session
            self.offers[(plan['family'], 'member_busy')] += 1
            return None
        self.offers[(plan['family'], 'fire')] += 1
        self._last = dict(family=plan['family'], holder=O.holder_of(mem)['id'],
                          row_class=plan['row_class'], plan_source=plan['plan_source'])
        return mem
    return select


class HoldFormCellBankCore(V.LcCellBankCore):
    """a44 LcCellBankCore (BANK_ALWAYS start rule, unconditional rows) with A75 admission; plain plans keep the
    A44 core class and the A32 gate function object, extended plans get HoldFormCore + _form_gate."""
    _form_gate = _form_gate

    def __init__(self, bank, cell_key, warmup_s=300):
        self.cell_key = str(cell_key)
        self.plan_source = None
        self.plans, self.refused = a75_bank_plans(bank, self.cell_key)
        self.cores = []
        for p in self.plans:
            if p['extended']:
                core = HoldFormCore(p['spec'], warmup_s)
                core.select_members = self._form_gate(core, p)
            else:
                core = V.LcStrategyCore(p['spec'], warmup_s)
                core.select_members = self._gate(core, p)
            self.cores.append(core)
        self.warmup = int(warmup_s)
        self.acts, self.offers = [], Counter()
        self._rows = None
        self._last = None
        self._busy = set()
        if any(p['row_class'] != 'unconditional' for p in self.plans):
            raise ValueError('A75: every row must be unconditional')

    def receipts(self):
        r = super().receipts()
        ev = [dict(e) for c in self.cores for e in getattr(c, 'events', ())]
        ss = [dict(x) for c in self.cores for x in getattr(c, 'sessions', ())]
        r.update(schema=SCHEMA, executor='cell_bank_always_holdforms',
                 extended=[p['family'] for p in self.plans if p['extended']], holdforms=ev, holdform_sessions=ss,
                 holdform_summary=dict(sessions=len(ss), events=len(ev),
                                       gap_sessions=sum(x['start_kind'] == 'gap' for x in ss),
                                       gap_fired=sum(e['start_kind'] == 'gap' for e in ev),
                                       gap_armed_no_event=sum(x['start_kind'] == 'gap' and x['released'] is not None
                                                              and x['released'][1] == 'gap_arm_end'
                                                              and not any(e['vid'] == x['vid'] and e['start_kind'] == 'gap'
                                                                          and e['fire_t'] >= x['start'] for e in ev)
                                                              for x in ss),
                                       clear_ends=sum(e['end_why'] == 'clear' for e in ev),
                                       issued_s=sum(e['issued_s'] for e in ev)))
        return r


def bank_doc(plans, cell_key):
    """a44_lcvariant.bank_doc (same rows / templates for plain plans) with the A75 fields and key."""
    rows, tpls = [], []
    for p in plans:
        s = canonical(p['strategy'])
        if s['name'] != p['family'] or executor_refusal(s) is not None:
            raise ValueError(f'{p["family"]}: not executable by the A75 executor ({executor_refusal(s)})')
        roles = []
        for r in s['roles']:
            d = dict(sumo_lane=int(r['lane']), rank=int(r['rank']), v_w=float(r['v_w']), dmax=float(r['dmax']),
                     hold_s=int(r['hold_s']), onset_s=int(r['onset_s']), lc_target_sumo=r['lc_target'])
            d.update({k: r[k] for k in ROLE_DEFAULTS if k in r})
            roles.append(d)
        cond = dict(trigger=s['trigger'], zone=s['zone'], spread_m=float(s['spread_m']), cooldown_s=int(s['cooldown_s']),
                    slots=int(s['slots']), gate=s['gate'], start_gate=s['start_gate'], phases=s['phases'],
                    situation=dict(unconditional=True), roles=roles)
        if s.get('zone_d') is not None:
            cond['zone_d'] = list(s['zone_d'])
        key = strategy_key(s)
        rows.append(dict(regime=cell_key, family=s['name'], lb95_veh_s=float(p.get('lb95_veh_s', 0.0)),
                         plan_source=C.PLAN_SOURCE, row_class='unconditional', situational=dict(unconditional=True),
                         template_sha256=None, strategy_key=key))
        tpls.append(dict(cell=cell_key, family=s['name'], offer_condition=cond,
                         template=dict(source=dict(strategy_key=key))))
    return dict(schema=SCHEMA+'.bank', rows=rows, clean5c=dict(templates=tpls))


def make_core(arm, cell_key, bank, family=None, warmup_s=300):
    if arm == 'A75_PLAN':
        core = HoldFormCellBankCore(C.restrict(bank, family), cell_key, warmup_s)
        if [p['family'] for p in core.plans] != [family]:
            raise ValueError(f'A75_PLAN: {family} refused by the executor {core.refused}')
        return core
    if arm == 'A75_BANK':
        core = HoldFormCellBankCore(bank, cell_key, warmup_s)
        want = {r['family'] for r in bank['rows'] if r['regime'] == cell_key}
        if not core.plans or {p['family'] for p in core.plans} != want:
            raise ValueError(f'A75_BANK: admitted plans {sorted(want)} not all executable {core.refused}')
        return core
    raise ValueError(f'unknown A75 arm {arm!r}')


def make_hook(arm, cell_key, **kw):
    """controller_hook (RULE-arm protocol), identical to a44_lcvariant.make_hook except the core."""

    class A75Hook:
        def __init__(self):
            self.arm = arm
            self.core = make_core(arm, cell_key, **kw)
            self.controller = None

        def __call__(self, controller, job, prefixes):
            self.controller = controller
            previous = controller.apply

            def apply(rows, commands, now):
                desired = previous(rows, commands, now)
                accel, lanes = self.core.step(rows, commands, int(now))
                cav = {r['vid'] for r in rows if r.get('is_cav')}
                bad = (set(accel) | set(lanes)) - cav
                if bad:
                    raise RuntimeError(f'{arm}: request for non-CAV rows {sorted(bad)} (HDVs never commanded)')
                owner = self.core.owners()
                for vid, a in accel.items():
                    desired[vid] = dict(requested_raw=a, acceleration_residual=a - commands[vid],
                                        oracle_plan=owner.get(vid))
                if lanes:
                    targets = dict(getattr(controller, 'g5_lane_targets', {}) or {})
                    targets.update(lanes)
                    controller.g5_lane_targets = targets
                return desired

            controller.apply = apply

        def receipts(self):
            return dict(self.core.receipts(), a75_arm=arm)

    return A75Hook()


EXECUTED = ('_cast', 'canonical', 'is_extended', 'strategy_key', 'spec_of', 'zone_bounds', '_int_ok',
            'holdform_refusal', 'executor_refusal', 'a75_bank_plans', 'select_members', 'role_field', 'gap_candidate',
            'hold_target', 'pulse_on', 'law', 'HoldFormCore', '_form_gate', 'HoldFormCellBankCore', 'bank_doc',
            'make_core', 'make_hook')
CONSTANTS = ('SCHEMA', 'A75_ARMS', 'ZONES', 'ZONE_D_MAX', 'ROLE_DEFAULTS', 'OLD_ROLE_KEYS', 'HOLD_S_RANGE', 'V_W_MAX',
             'GAP_SUMO_LANE', 'CLOSING_WM_LANE', 'SENSE_M', 'SENSE_LANES', 'GAP_DX_MAX', 'HOLDFORM_REFUSALS',
             'REFUSALS')


def a75_digest(module=None):
    """sha256 of the A75 executor as executed: a44_digest() (A44 + A37 executors, inherited) + every executed
    function/class here + the constants (ROLE_TYPES by type name)."""
    import sys
    m = module or sys.modules[__name__]
    parts = [V.a44_digest()]
    parts += [inspect.getsource(getattr(m, n)) for n in EXECUTED]
    consts = {n: getattr(m, n) for n in CONSTANTS}
    consts['ROLE_TYPES'] = {k: t.__name__ for k, t in m.ROLE_TYPES.items()}
    parts.append(json.dumps(consts, sort_keys=True, default=list))
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()
