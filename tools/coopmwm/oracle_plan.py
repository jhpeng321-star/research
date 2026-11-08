"""ORACLE-PLAN arm hook (N6 amendment 25; MAIN 2026-11-08 22:59 / 23:5x): U0 (frozen MAPPO + bound u0) plus privileged
execution of the admitted verified plans of the FROZEN end-of-training union bank (clean5c guidance doc, e.g.
GUIDANCE_C4SIT_priv_UNION_*.json) whenever, at a decision second, a plan's feasible CAV group is selected AND the
privileged onset bin of its acting CAV is one of the plan's admitted bins. Disclosed as a privileged upper bound, not a
deployable policy.

Executor = the S-WMVERIFY plant hold executor that verified these plans: StrategyCore + select_members (vendored
VERBATIM below from c4/wmverify 775cff21 tools/wzprop/strategy_search.py / strategy_hook.py; the only edits in
StrategyCore: the spec is the bank-rebuilt spec (strategy_key-checked, see spec_of) instead of SS.validate,
select_members is a class attribute so the oracle gate can wrap it per core, and SS.active_roles -> the vendored
active_roles; ORACLE_EDITS lists them and vendored_check undoes them against the origin digest). Hold law clip(v_w - v, -dmax, 1.0) as
the absolute pre-safety request through the plant single writer (MAPPO safety downstream), lane target = own lane
(G5 single-writer pin); CAV rows only, HDVs are never commanded.
Bin = fullpol_situational.onset_bin(holder row, all plant rows, 'privileged') (vendored VERBATIM from c4/p20-sit
d70caf9f), holder = the member on SUMO lane 2, else the first member (S-MULTICRIT-ACT / ops/sit_reduce.py rule); the
plant rows passed to controller.apply at second t are the rows the observable tape records at t (coopfree tick).
Session: one activation at a time over all plans of the cell; after an activation every plan waits the executed
plan's cooldown_s; candidates are tried in bank order lb95_veh_s descending (family name ties). No RNG, no clock;
reads only the bank doc, the plant rows and the U0 commands - never job / B / R outputs.
"""
from __future__ import annotations

import hashlib
import inspect
import itertools
import json
from collections import Counter


class G:                       # tools/wzprop/geometry.py constants (c4/wmverify 775cff21)
    X_B = 2696.0
    DROP_S = 75.44
    WALL_SPREAD = 200.0
    R_SPREAD = 300.0


# tools/wzprop/strategy_search.py constants (c4/wmverify 775cff21)
ZONES = {'full': (G.X_B - 1800.0, G.X_B - 400.0), 'near': (G.X_B - 1200.0, G.X_B - 400.0),
         'far': (G.X_B - 1800.0, G.X_B - 1000.0)}
GATES = (None, 5.0, 10.0, 15.0)
FIELD_X = (G.X_B - 454.10, G.X_B)
START_GATES = {None: None, 'empty100': dict(behind_m=100.0, max_behind=0)}
# tools/wzprop/strategy_hook.py constants
S_OFF = G.X_B - G.DROP_S
RELEASE_S = G.DROP_S
# tools/coopmwm/fullpol_situational.py constants (c4/p20-sit d70caf9f)
MODES = ('privileged', 'own_sensing')
S_DROP = 75.44
CONG_MPS, LANE_DIFF_MPS = 15.0, 5.0
SENSOR_RANGE_M = 100.0
UNOBSERVED = dict(cong='unobserved', lane_diff='unobserved')
SITUATION_RULE = 'fullpol-clean5c4-situational-v4'


# ------------------------------------------------------------------ vendored: strategy_search (VERBATIM)

def strategy_key(s):
    body = {k: s[k] for k in ('trigger', 'zone', 'spread_m', 'cooldown_s', 'slots', 'roles')}
    for k in ('gate', 'start_gate', 'phases'):           # only when set: older keys stay stable
        if s.get(k) is not None:
            body[k] = s[k]
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]

def _pick_wall(z):
    lanes = [[c for c in z if int(c['lane']) == ln] for ln in (0, 1, 2)]
    if any(not x for x in lanes):
        return None
    best = None
    for combo in itertools.product(*lanes):
        xs = [float(c['x']) for c in combo]
        sp = max(xs) - min(xs)
        if sp <= G.WALL_SPREAD:
            key = (sp, -max(xs))
            if best is None or key < best[0]:
                best = (key, list(combo))
    return None if best is None else {int(c['lane']): c for c in best[1]}

def field_speed(rows):
    sp = [float(r['v']) for r in rows if r['edge'] == 'up' and int(r['lane']) == 0
          and FIELD_X[0] <= float(r['x']) <= FIELD_X[1]]
    return sum(sp) / len(sp) if sp else None

def gate_open(s, observed):
    if s.get('gate') is None:
        return True
    fv = field_speed(observed)
    return fv is not None and fv <= float(s['gate'])

def behind_count(rows, member, behind_m):
    """Observed vehicles in the member's lane (same edge) within behind_m behind it (excluding itself)."""
    x, lane, edge = float(member['x']), int(member['lane']), member.get('edge', 'up')
    seen = set()
    for r in rows:
        if r['id'] == member['id'] or r['id'] in seen:
            continue
        if r.get('edge', 'up') == edge and int(r['lane']) == lane and x - behind_m <= float(r['x']) < x:
            seen.add(r['id'])
    return len(seen)

def active_roles(s, phase=0):
    ph = s.get('phases')
    return list(s['roles']) if not ph else [s['roles'][i] for i in ph[int(phase) % len(ph)]]

def select_members(s, cavs, observed=None, phase=0):
    """CAV rows (id, edge, lane, x, v) chosen by the strategy's trigger at this snapshot, in active-role order, else
    None. observed = every observable row (CAV self + CAV-detected) for the gates (default: the CAVs only)."""
    seen_rows = list(cavs) + list(observed or [])
    if not gate_open(s, seen_rows):
        return None
    roles = active_roles(s, phase)
    lo, hi = ZONES[s['zone']]
    z = [c for c in cavs if c['edge'] == 'up' and lo <= float(c['x']) <= hi]
    if s['trigger'] == 'wall':
        wall = _pick_wall(z)
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
    sg = START_GATES[s.get('start_gate')]
    if sg is not None and any(behind_count(seen_rows, c, sg['behind_m']) > sg['max_behind'] for c in out):
        return None
    return out


# ------------------------------------------------------------------ vendored: strategy_hook (VERBATIM)

def sumo_lane(wm_lane):
    return 2 - int(wm_lane)

def wm_lane_of(sumo):
    return 2 - int(sumo)

def to_sumo_row(r):
    return dict(id=r['vid'], edge='up' if float(r['s_m']) < RELEASE_S else 'wz', lane=sumo_lane(r['lane']),
                x=float(r['s_m']) + S_OFF, v=float(r['v_mps']))

def observed_rows(rows, report_range=100.0, lane_reach=1):
    """CAV rows + non-CAV rows a CAV detects (N1 cav_reports: |ds| <= 100 m, |dWM lane| <= 1), SUMO frame."""
    cav = [r for r in rows if r.get('is_cav')]
    out = [to_sumo_row(r) for r in cav]
    for r in rows:
        if r.get('is_cav'):
            continue
        if any(abs(float(r['s_m']) - float(c['s_m'])) <= report_range and abs(int(r['lane']) - int(c['lane'])) <= lane_reach
               for c in cav):
            out.append(to_sumo_row(r))
    return out



class StrategyCore:
    """Plant-independent rule state machine (tested with fake rows); the harness hook wraps it."""

    def __init__(self, strategy, warmup_s=300):
        self.s = dict(strategy)            # ORACLE: spec rebuilt from the bank + strategy_key check (oracle_plan.spec_of)
        self.warmup = int(warmup_s)
        self.active = {}             # vid -> dict(start, role, act)
        self.acts, self.issued = [], []
        self.next_ok = -1
        self.lane_held = 0
        self.lane_requests = 0

    select_members = staticmethod(select_members)     # ORACLE: per-core gate (OracleHook) may wrap it

    def law(self, role, v):
        return max(-float(role['dmax']), min(1.0, float(role['v_w']) - float(v)))

    def step(self, rows, commands, t):
        """rows: WM-frame plant rows (vid, s_m, v_mps, lane, is_cav); commands: base accel per CAV vid.
        Returns ({vid: requested accel}, {vid: target WM lane})."""
        by = {r['vid']: r for r in rows if r.get('is_cav')}
        accel, lanes = {}, {}
        if t < self.warmup:
            return accel, lanes
        for vid, m in list(self.active.items()):
            el = t - m['start']
            role = m['role']
            row = by.get(vid)
            end = int(role['onset_s']) + int(role['hold_s'])
            why = ('left' if row is None or vid not in commands else 'hold_end' if el >= end
                   else 'at_drop' if float(row['s_m']) >= RELEASE_S else None)
            if why is not None:
                m['act']['release'][vid] = [el, why]
                self.active.pop(vid)
                continue
            if el >= int(role['onset_s']):
                accel[vid] = self.law(role, row['v_mps'])
                self.issued.append([t, vid, round(accel[vid], 4)])
                if role['lc_target'] is not None:
                    lanes[vid] = wm_lane_of(role['lc_target'])
                    self.lane_requests += 1
                else:
                    lanes[vid] = int(row['lane'])
            else:
                lanes[vid] = int(row['lane'])
        busy = len({id(m['act']) for m in self.active.values()})
        if t >= self.next_ok and busy < int(self.s.get('slots', 1)):
            free = [to_sumo_row(r) for vid, r in by.items() if vid not in self.active and vid in commands]
            phase = len(self.acts)
            mem = self.select_members(self.s, free, observed_rows(rows), phase=phase)
            if mem:
                act = dict(t=t, strategy=self.s['name'], members=[c['id'] for c in mem], phase=phase,
                           sumo_lanes=[c['lane'] for c in mem], x=[round(c['x'], 2) for c in mem], release={})
                self.acts.append(act)
                self.next_ok = t + int(self.s['cooldown_s'])
                for role, c in zip(active_roles(self.s, phase), mem):
                    self.active[c['id']] = dict(start=t, role=role, act=act)
                    row = by[c['id']]
                    lanes[c['id']] = int(row['lane'])
                    if int(role['onset_s']) == 0:
                        accel[c['id']] = self.law(role, row['v_mps'])
                        self.issued.append([t, c['id'], round(accel[c['id']], 4)])
                        if role['lc_target'] is not None:
                            lanes[c['id']] = wm_lane_of(role['lc_target'])
                            self.lane_requests += 1
        self.lane_held += len(lanes)
        return accel, lanes


# ------------------------------------------------------------------ vendored: fullpol_situational (VERBATIM)

def situation_bin(sit):
    ds = sit.get('drop_speed')
    lanes = sit.get('v_lane_ahead500') or {}
    l2, l1 = lanes.get('0', lanes.get(0)), lanes.get('1', lanes.get(1))
    return dict(cong='congested' if (ds is not None and ds < CONG_MPS) else 'free',
                lane_diff='L2faster' if (l2 is not None and l1 is not None and l2-l1 > LANE_DIFF_MPS) else 'L2notfaster')

def _mean(x):
    return sum(x)/len(x) if x else None

def runtime_situation(own, rows):
    """The same onset features (reduce_act.situation) from the acting CAV's own row + own sensing rows."""
    s = float(own['s_m'])
    view = [own]+[r for r in rows if r.get('vid') != own.get('vid')]
    up = [(int(r['lane']), float(r['s_m']), float(r['v_mps'])) for r in view if float(r['s_m']) < S_DROP]
    return dict(drop_speed=_mean([v for _, x, v in up if -600 <= x]),
                v_lane_ahead500={str(k): _mean([v for ln, x, v in up if ln == k and s <= x <= s+500]) for k in (0, 1, 2)})

def onset_bin(own, rows, mode='own_sensing'):
    """THE bin rule, offline (ops/sit_reduce.py on the observable tape at onset) and at runtime
    (fullpol_bank.offer_ok): the acting CAV's own row + every vehicle within SENSOR_RANGE_M of it (the
    track2_backend own_sensing filter); drop-area speed or lane-0 / lane-1 ahead speed not observed in that
    view -> UNOBSERVED (its own bin; never admitted, never offered)."""
    if mode not in MODES:
        raise ValueError('onset_bin mode must be one of '+repr(MODES))
    if own is None or own.get('v_mps') is None:
        return dict(UNOBSERVED)
    if mode == 'privileged':
        # the full-tape S-MULTICRIT-ACT features (reduce_act.situation, verbatim): every vehicle, no range
        view = [r for r in rows if r.get('vid') != own.get('vid')]
        if any(r.get('v_mps') is None for r in view):
            return dict(UNOBSERVED)
        return situation_bin(runtime_situation(own, view))
    s = float(own['s_m'])
    view = [r for r in rows if r.get('vid') != own.get('vid') and abs(float(r['s_m'])-s) <= SENSOR_RANGE_M]
    if any(r.get('v_mps') is None for r in view):
        return dict(UNOBSERVED)
    sit = runtime_situation(own, view)
    lanes = sit['v_lane_ahead500']
    if sit['drop_speed'] is None or lanes['0'] is None or lanes['1'] is None:
        return dict(UNOBSERVED)
    return situation_bin(sit)



# ------------------------------------------------------------------ ORACLE (amendment 25)
#: sha256 of each vendored source (function / class text) at its origin commit; oracle_plan.vendored_check() verifies
#: the text here still matches (StrategyCore: origin text before the two documented ORACLE edits).
VENDORED = {
 "StrategyCore": {
  "origin": "c4/wmverify 775cff21 tools/wzprop/strategy_hook.py",
  "sha256": "b695252f793bf7eb3683cee0bfe5d138c3a51477573bc26a3a508f750101fccc"
 },
 "_mean": {
  "origin": "c4/p20-sit d70caf9f tools/coopmwm/fullpol_situational.py",
  "sha256": "410e3ed105d1c8d5e082c9948e8de8685b0022864b1afe30b44385b6b169f7b5"
 },
 "_pick_wall": {
  "origin": "c4/wmverify 775cff21 tools/wzprop/strategy_search.py",
  "sha256": "232bf0a35dfc75899e59db111484c230aec4f6de717cbdf5f42214b57f85e9ef"
 },
 "active_roles": {
  "origin": "c4/wmverify 775cff21 tools/wzprop/strategy_search.py",
  "sha256": "d9118f959c9587006d850a6c2aeec90cfa887297e6ad1b7e12ae62076614b0f3"
 },
 "behind_count": {
  "origin": "c4/wmverify 775cff21 tools/wzprop/strategy_search.py",
  "sha256": "3b0dddae8ccef3ebffcd452567567f4547305fa4fc3a667781545d41a9ba9e87"
 },
 "field_speed": {
  "origin": "c4/wmverify 775cff21 tools/wzprop/strategy_search.py",
  "sha256": "80df1c2adb1f1f6ef45baf320faa70d7a30480c830f20e75fd989c713d65df00"
 },
 "gate_open": {
  "origin": "c4/wmverify 775cff21 tools/wzprop/strategy_search.py",
  "sha256": "e7a2b967fb5340a231c265832953aacc7b4002b0c2717afbb415277e21823bca"
 },
 "observed_rows": {
  "origin": "c4/wmverify 775cff21 tools/wzprop/strategy_hook.py",
  "sha256": "42cc19e3a8d73543475581dfdc96b2d69d727fef6fa3e201ef1d8c0d35a1d9f2"
 },
 "onset_bin": {
  "origin": "c4/p20-sit d70caf9f tools/coopmwm/fullpol_situational.py",
  "sha256": "db21d0ec73865a4bdc5fa55cf60ac8dda2d5a86dc449f5d987812afbeb2de32e"
 },
 "runtime_situation": {
  "origin": "c4/p20-sit d70caf9f tools/coopmwm/fullpol_situational.py",
  "sha256": "3bcab1032086e4d4a782aa7caee07e59d95111608ddd89e7dcc281c90d3bbd4f"
 },
 "select_members": {
  "origin": "c4/wmverify 775cff21 tools/wzprop/strategy_search.py",
  "sha256": "8a19df37ea16d364a57fb0cb45b601a5bd7c3500dcd3248a9768ef4c4d7f03fc"
 },
 "situation_bin": {
  "origin": "c4/p20-sit d70caf9f tools/coopmwm/fullpol_situational.py",
  "sha256": "e0a6f3bae6c75b09e4c0bcc5ef4502ae7d3f7e10187fb341456872fe27a51baa"
 },
 "strategy_key": {
  "origin": "c4/wmverify 775cff21 tools/wzprop/strategy_search.py",
  "sha256": "b6d04d458ff6e13a1d6bd73ad6a93f15505c1aea92235ecb4fc17eb4fc4136dc"
 },
 "sumo_lane": {
  "origin": "c4/wmverify 775cff21 tools/wzprop/strategy_hook.py",
  "sha256": "73ad9a5e582bbaf4276ba3a08373b8a2e9e82a2352ba79dccb4d19d04d039b03"
 },
 "to_sumo_row": {
  "origin": "c4/wmverify 775cff21 tools/wzprop/strategy_hook.py",
  "sha256": "e9aed85850ec2c437612100c1be8549e87bb2d18f12df14b3c7c49f4ac7f3650"
 },
 "wm_lane_of": {
  "origin": "c4/wmverify 775cff21 tools/wzprop/strategy_hook.py",
  "sha256": "8c93be9268fbf5eb6f55ac2daa9bbd2ac57fd62fc6a7e6d65ec5d085e034459f"
 }
}
SCHEMA = 'coopmwm.oracle_plan.v1'
REFUSALS = ('strategy_key_mismatch', 'non_hold_plan', 'no_template', 'bins_not_privileged', 'gate_or_phases')


class OracleRefused(ValueError):
    pass


def spec_of(family, cond):
    """WM-C strategy spec rebuilt from a bank offer_condition (wmverify_hook.offer_condition fields)."""
    roles = []
    for r in cond['roles']:
        roles.append(dict(lane=int(r['sumo_lane']), rank=int(r['rank']), v_w=float(r['v_w']), dmax=float(r['dmax']),
                          hold_s=int(r['hold_s']), onset_s=int(r['onset_s']), lc_target=r.get('lc_target_sumo')))
    return dict(name=family, trigger=cond['trigger'], zone=cond['zone'], spread_m=float(cond['spread_m']),
                cooldown_s=int(cond['cooldown_s']), slots=int(cond['slots']), gate=cond.get('gate'),
                start_gate=cond.get('start_gate'), phases=cond.get('phases'), roles=roles)


def bank_plans(bank_doc, cell_key):
    """[(plan dict)] admitted for ``cell_key`` in priority order, and {refusal: [family]}.
    A plan = the bank row (lb95) + its template's offer_condition, rebuilt spec (strategy_key must equal the
    template's source.strategy_key), privileged admitted bins. Hold-only plans only (lane roles refused)."""
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
        if sit.get('mode') != 'privileged' or not sit.get('bins') or sit.get('rule') != SITUATION_RULE:
            refused['bins_not_privileged'].append(fam)
            continue
        spec = spec_of(fam, cond)
        key = ((t['template'].get('source') or {}).get('strategy_key'))
        if key is None or strategy_key(spec) != key:
            refused['strategy_key_mismatch'].append(fam)
            continue
        if any(x['lc_target'] is not None for x in spec['roles']):
            refused['non_hold_plan'].append(fam)
            continue
        if spec['gate'] not in GATES or spec['start_gate'] not in START_GATES or spec['phases'] is not None:
            refused['gate_or_phases'].append(fam)
            continue
        unconditional = bool((r.get('situational') or {}).get('unconditional') or sit.get('unconditional'))
        plans.append(dict(family=fam, lb95=float(r['lb95_veh_s']), key=key, spec=spec,
                          row_class='unconditional' if unconditional else 'situational',
                          bins=[dict(cong=b['cong'], lane_diff=b['lane_diff']) for b in sit['bins']],
                          plan_source=r.get('plan_source'), template_sha256=r.get('template_sha256')))
    # N6 2026-11-09: situational (bin-gated) rows first, lb95 descending, then strategy_key; unconditional rows (act
    # whenever feasible, episode-level lb95 - not comparable) only when no situational row activates at that step
    plans.sort(key=lambda p: (p['row_class'] != 'situational', -p['lb95'], p['key'], p['family']))
    return plans, {k: v for k, v in refused.items() if v}


def holder_of(members):
    """S-MULTICRIT-ACT / sit_reduce: the member on SUMO lane 2, else the first member (SUMO-frame rows)."""
    return next((c for c in members if int(c['lane']) == 2), members[0] if members else None)


def bin_admitted(got, bins):
    return got is not None and got != UNOBSERVED and any(
        all(got[k] == b[k] for k in ('cong', 'lane_diff')) for b in bins)


class OracleCore:
    """Plant-independent ORACLE state machine (fake-row testable): one StrategyCore per admitted plan, one global
    session; returns ({vid: accel}, {vid: WM lane}) like StrategyCore.step."""

    def __init__(self, bank_doc, cell_key, warmup_s=300):
        self.cell_key = str(cell_key)
        self.plans, self.refused = bank_plans(bank_doc, self.cell_key)
        self.cores = []
        for p in self.plans:
            core = StrategyCore(p['spec'], warmup_s)
            core.select_members = self._gate(core, p)
            self.cores.append(core)
        self.warmup = int(warmup_s)
        self.next_ok = -1
        self.acts, self.offers = [], Counter()
        self._rows = None
        self._last = None

    def _gate(self, core, plan):
        def select(s, free, observed, phase=0):
            mem = select_members(s, free, observed, phase=phase)
            if not mem:
                return None
            hold = holder_of(mem)
            own = next((r for r in self._rows if r.get('vid') == hold['id']), None)
            got = onset_bin(own, self._rows, 'privileged')
            self.offers[(plan['family'], json.dumps(got, sort_keys=True))] += 1
            if not bin_admitted(got, plan['bins']):
                return None
            self._last = dict(family=plan['family'], bin=got, holder=hold['id'], row_class=plan['row_class'])
            return mem
        return select

    def active(self):
        return next((c for c in self.cores if c.active), None)

    def step(self, rows, commands, t):
        self._rows = rows
        accel, lanes = {}, {}
        if t < self.warmup or not self.cores:
            return accel, lanes
        busy = self.active()
        if busy is not None:                        # the running session only (release / hold law)
            busy.next_ok = 10 ** 9                  # no new activation while one runs
            a, ln = busy.step(rows, commands, t)
            accel.update(a)
            lanes.update(ln)
            if busy.active or t < self.next_ok:
                return accel, lanes
        if t < self.next_ok:
            return accel, lanes
        for core, plan in zip(self.cores, self.plans):
            n = len(core.acts)
            core.next_ok = -1
            self._last = None
            a, ln = core.step(rows, commands, t)
            if len(core.acts) > n:                  # this plan activated: one session, its cooldown for all
                act = core.acts[-1]
                act.update(self._last or {}, priority=self.plans.index(plan))
                self.acts.append(act)
                self.next_ok = t + int(plan['spec']['cooldown_s'])
                accel.update(a)
                lanes.update(ln)
                break
        return accel, lanes

    def receipts(self):
        fam = Counter(a['family'] for a in self.acts)
        return dict(schema=SCHEMA, cell_key=self.cell_key, plans=[dict(family=p['family'], lb95=p['lb95'], key=p['key'],
                    bins=p['bins'], plan_source=p['plan_source'], row_class=p['row_class']) for p in self.plans],
                    refused=self.refused, refused_counts={k: len(v) for k, v in self.refused.items()},
                    activations=len(self.acts), activations_by_family=dict(fam),
                    issued=sum(len(c.issued) for c in self.cores),
                    activations_by_row_class=dict(Counter(a.get('row_class') for a in self.acts)),
                    acts=[dict(t=a['t'], family=a.get('family'), row_class=a.get('row_class'), members=a['members'],
                               holder=a.get('holder'), bin=a.get('bin'), release=a.get('release')) for a in self.acts],
                    offers={f'{k[0]}|{k[1]}': v for k, v in sorted(self.offers.items())})


def make_hook(bank_doc, cell_key, warmup_s=300):
    """controller_hook (RULE-arm protocol): hook(controller, job, prefixes) wraps controller.apply after the U0 hook.
    ``job`` and ``prefixes`` are never read."""

    class OracleHook:
        def __init__(self):
            self.core = OracleCore(bank_doc, cell_key, warmup_s)
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
                    raise RuntimeError(f'ORACLE: request for non-CAV rows {sorted(bad)} (HDVs never commanded)')
                for vid, a in accel.items():
                    desired[vid] = dict(requested_raw=a, acceleration_residual=a - commands[vid],
                                        oracle_plan=self.core.acts[-1]['family'] if self.core.acts else None)
                if lanes:
                    targets = dict(getattr(controller, 'g5_lane_targets', {}) or {})
                    targets.update(lanes)
                    controller.g5_lane_targets = targets
                return desired

            controller.apply = apply

        def receipts(self):
            return self.core.receipts()

    return OracleHook()


EXECUTED = ('strategy_key', '_pick_wall', 'field_speed', 'gate_open', 'behind_count', 'active_roles', 'select_members',
            'sumo_lane', 'wm_lane_of', 'to_sumo_row', 'observed_rows', 'StrategyCore', 'situation_bin', '_mean',
            'runtime_situation', 'onset_bin', 'spec_of', 'bank_plans', 'holder_of', 'bin_admitted', 'OracleCore',
            'make_hook')
CONSTANTS = ('ZONES', 'GATES', 'FIELD_X', 'START_GATES', 'S_OFF', 'RELEASE_S', 'MODES', 'S_DROP', 'CONG_MPS',
             'LANE_DIFF_MPS', 'SENSOR_RANGE_M', 'UNOBSERVED', 'SITUATION_RULE')


def oracle_digest(module=None):
    """sha256 of the ORACLE as executed: the source of every executed function/class + every constant + the
    geometry constants (like nm_eval_runner.rule_digest; a file edit elsewhere does not move it)."""
    import sys
    m = module or sys.modules[__name__]
    parts = [inspect.getsource(getattr(m, n)) for n in EXECUTED]
    consts = {n: getattr(m, n) for n in CONSTANTS}
    consts['G'] = {k: getattr(m.G, k) for k in ('X_B', 'DROP_S', 'WALL_SPREAD', 'R_SPREAD')}
    consts['START_GATES'] = {repr(k): v for k, v in consts['START_GATES'].items()}   # None key: not JSON-sortable
    parts.append(json.dumps(consts, sort_keys=True, default=list))
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()


ORACLE_EDITS = (
    ("        self.s = dict(strategy)            # ORACLE: spec rebuilt from the bank + strategy_key check (oracle_plan.spec_of)",
     "        self.s = SS.validate(dict(strategy), allow_banned=True)     # executing a banned comparator is allowed"),
    ("            mem = self.select_members(self.s, free, observed_rows(rows), phase=phase)",
     "            mem = SS.select_members(self.s, free, observed_rows(rows), phase=phase)"),
    ("    select_members = staticmethod(select_members)     # ORACLE: per-core gate (OracleHook) may wrap it\n\n", ""),
    ("                for role, c in zip(active_roles(self.s, phase), mem):",
     "                for role, c in zip(SS.active_roles(self.s, phase), mem):"),
)


def vendored_check(module=None):
    """{name: True} when each vendored text equals its origin digest (StrategyCore after undoing ORACLE_EDITS)."""
    import sys
    m = module or sys.modules[__name__]
    out = {}
    for name, pin in VENDORED.items():
        text = inspect.getsource(getattr(m, name)).rstrip('\n')
        if name == 'StrategyCore':
            for new, old in ORACLE_EDITS:
                text = text.replace(new, old)
        out[name] = hashlib.sha256(text.encode()).hexdigest() == pin['sha256']
    return out
