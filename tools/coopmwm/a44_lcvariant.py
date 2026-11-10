"""AMENDMENT 44 (MAIN 2026-11-10 22:50; pre-outcome): lane-change-capable VARIANT of the amendment-37 executor.

Implementation change only (no scientific question): the A37 executor (a37_cellbank.CellBankCore, digest 0a902437,
BANK_ALWAYS start rule on oracle_plan.OracleCore) refuses every plan with an ``lc_target`` role at bank admission
(oracle_plan.bank_plans 'non_hold_plan'). This module is a SEPARATE executor with its own digest (``a44_digest``);
a37_cellbank / oracle_plan / a32_arms are imported, never edited, so the A37 path and its digest stay unchanged.

Exactly what differs from the A37 path (everything else is the A37 code object itself, by inheritance):
1. admission (``lc_bank_plans`` = oracle_plan.bank_plans line for line, only the 'non_hold_plan' refusal replaced):
   a role with ``lc_target`` is admitted iff the target is a SUMO lane adjacent to the role lane
   (strategy_search.validate rule) AND the existing lane path can execute it: the G5 env writer
   (g5_runtime_env.pysrc ``_issue_lane_commands``) refuses every request into WM lane 2 (= SUMO lane 0, the closing
   lane: ``target == 2`` -> refused_not_allowed), so such roles are refused here ('lc_into_closing_lane') instead of
   being admitted and silently never executed. Non-adjacent targets: 'lc_not_adjacent'.
2. lane request: the vendored StrategyCore already turns ``role['lc_target']`` into the WM lane request from the
   role's onset (oracle_plan.py:203-204, 227-228) through ``controller.g5_lane_targets`` (the G5 single-writer lane
   path). ``LcStrategyCore`` adds ONLY the search's give-up point: the WM search executes an lc role as an LCMOVES
   'prog' with ``lc_from = onset_s`` and ``lc_giveup_x = X_B - GIVEUP_OFF`` (strategy_search.member_specs; lcm_core
   ``_prog``: the change is requested only while the member is on 'up' with x < lc_giveup_x, the member keeps lcMode
   513 and its speed program afterwards). Here: once the member's SUMO x >= lc_giveup_x, its lane request becomes its
   CURRENT lane (the A37 G5 pin of every other member; the G5 writer issues nothing for target == current, the CAV
   stays under policy lane control 513). No other change; GIVEUP_OFF is the lcm_plan constant the search used.
3. bookkeeping only: per lc member the first request, completion (WM row lane == target while active) and give-up
   seconds, in receipts.
For a plan without lc roles nothing in 1-3 applies: same admission, same commands (test_a44_lcvariant_cpu).

Longitudinal requests go to CommandOwnership exactly as A37 (``desired[vid] = requested_raw``); HDVs never commanded.
No RNG, no clock; reads only the bank doc, the plant rows and the U0 commands.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter

from . import a32_arms as A32
from . import a37_cellbank as C
from . import oracle_plan as O

SCHEMA = 'coopmwm.a44_lcvariant.v1'
A44_ARMS = ('A44_PLAN', 'A44_BANK')
#: lcm_plan.GIVEUP_OFF [COOPKINDS lane_balance giveup_off] (c4/wmc-p20 tools/wzprop/lcm_pipeline/lcm_plan.py, pinned
#: 94ebbc93 via families_lc.PINS); strategy_search.member_specs: lc_giveup_x = P.X_B - P.GIVEUP_OFF
GIVEUP_OFF = 300.0
LC_GIVEUP_X = O.G.X_B - GIVEUP_OFF
#: the G5 env writer refuses requests into WM lane 2 (g5_runtime_env.pysrc _issue_lane_commands: ``target == 2``)
G5_REFUSED_WM_TARGET = 2
LC_REFUSALS = ('lc_not_adjacent', 'lc_into_closing_lane')
REFUSALS = tuple(r for r in O.REFUSALS if r != 'non_hold_plan') + LC_REFUSALS


def lc_refusal(spec):
    """None when every lc role is executable through the existing (G5) lane path, else the refusal class."""
    for r in spec['roles']:
        tl = r.get('lc_target')
        if tl is None:
            continue
        if int(tl) not in (0, 1, 2) or abs(int(tl) - int(r['lane'])) != 1:
            return 'lc_not_adjacent'
        if O.wm_lane_of(tl) == G5_REFUSED_WM_TARGET:
            return 'lc_into_closing_lane'
    return None


def is_lc_plan(spec):
    return any(r.get('lc_target') is not None for r in spec['roles'])


def executor_refusal(spec):
    """a37_cellbank.executor_refusal with 'non_hold_plan' replaced by the lc rule above."""
    why = lc_refusal(spec)
    if why is not None:
        return why
    if spec.get('gate') not in O.GATES or spec.get('start_gate') not in O.START_GATES or spec.get('phases') is not None:
        return 'gate_or_phases'
    if spec.get('trigger') not in ('wall', 'lanes') or spec.get('zone') not in O.ZONES:
        return 'trigger_or_zone'
    return None


def lc_bank_plans(bank_doc, cell_key, plan_source=None):
    """oracle_plan.bank_plans, line for line, except the lc check: adjacent lc roles the G5 path executes are admitted."""
    if plan_source not in (None, 'wm'):
        raise ValueError("plan_source must be None (ORACLE: union) or 'wm' (ORACLE_WM)")
    rows = [r for r in bank_doc['rows'] if r.get('regime') == cell_key]
    tpls = {(t['cell'], t['family']): t for t in (bank_doc.get('clean5c') or {}).get('templates', ())}
    plans, refused = [], {k: [] for k in REFUSALS}
    for r in rows:
        fam = r['family']
        if plan_source == 'wm' and r.get('plan_source') != 'wm':
            refused['not_wm_source'].append(fam)
            continue
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
        spec = O.spec_of(fam, cond)
        key = ((t['template'].get('source') or {}).get('strategy_key'))
        if key is None or O.strategy_key(spec) != key:
            refused['strategy_key_mismatch'].append(fam)
            continue
        why = lc_refusal(spec)                                   # A44: replaces the 'non_hold_plan' refusal
        if why is not None:
            refused[why].append(fam)
            continue
        if spec['gate'] not in O.GATES or spec['start_gate'] not in O.START_GATES or spec['phases'] is not None:
            refused['gate_or_phases'].append(fam)
            continue
        plans.append(dict(family=fam, lb95=float(r['lb95_veh_s']), key=key, spec=spec,
                          row_class='unconditional' if unconditional else 'situational',
                          bins=None if unconditional else [dict(cong=b['cong'], lane_diff=b['lane_diff'])
                                                           for b in sit['bins']],
                          plan_source=r.get('plan_source'), template_sha256=r.get('template_sha256')))
    plans.sort(key=lambda p: (p['row_class'] != 'situational', -p['lb95'], p['key'], p['family']))
    return plans, {k: v for k, v in refused.items() if v}


class LcStrategyCore(O.StrategyCore):
    """oracle_plan.StrategyCore (A37 hold law, sessions, G5 lane pin, lc lane request from onset) + the search's
    lane-change give-up point; per-member lane-change bookkeeping. Roles without lc_target are untouched."""

    def __init__(self, strategy, warmup_s=300):
        super().__init__(strategy, warmup_s)
        self.lc = []                 # one record per (activation, lc member)
        self._lc_of = {}             # vid -> its open record

    def step(self, rows, commands, t):
        accel, lanes = super().step(rows, commands, t)
        by = {r['vid']: r for r in rows if r.get('is_cav')}
        for vid, rec in list(self._lc_of.items()):
            if vid not in self.active or self.active[vid]['act'] is not rec['_act']:
                rec['released'] = rec['_act']['release'].get(vid)
                self._lc_of.pop(vid)
        for vid, m in self.active.items():
            role = m['role']
            if role['lc_target'] is None:
                continue
            rec = self._lc_of.get(vid)
            if rec is None:
                rec = dict(_act=m['act'], vid=vid, start=m['start'], sumo_from=int(role['lane']),
                           sumo_target=int(role['lc_target']), onset_s=int(role['onset_s']), first_request=None,
                           requests=0, completed=None, gave_up=None, released=None)
                self.lc.append(rec)
                self._lc_of[vid] = rec
            if t - m['start'] < int(role['onset_s']) or vid not in lanes:
                continue
            row = by[vid]
            target = O.wm_lane_of(role['lc_target'])
            if rec['completed'] is None and int(row['lane']) == target:
                rec['completed'] = t
            if float(row['s_m']) + O.S_OFF >= LC_GIVEUP_X:          # lcm_core._prog: no request at x >= lc_giveup_x
                lanes[vid] = int(row['lane'])
                if rec['completed'] is None and rec['gave_up'] is None:
                    rec['gave_up'] = t
                continue
            if lanes[vid] != target:
                raise RuntimeError('A44: lc member without its lane request')
            rec['requests'] += 1
            if rec['first_request'] is None:
                rec['first_request'] = t
        return accel, lanes

    def lc_receipts(self):
        return [{k: v for k, v in r.items() if k != '_act'} for r in self.lc]


class LcOracleCore(O.OracleCore):
    """oracle_plan.OracleCore with lc_bank_plans admission and LcStrategyCore members (constructor only differs)."""

    def __init__(self, bank_doc, cell_key, warmup_s=300, plan_source=None):
        self.cell_key = str(cell_key)
        self.plan_source = plan_source
        self.plans, self.refused = lc_bank_plans(bank_doc, self.cell_key, plan_source)
        self.cores = []
        for p in self.plans:
            core = LcStrategyCore(p['spec'], warmup_s)
            core.select_members = self._gate(core, p)
            self.cores.append(core)
        self.warmup = int(warmup_s)
        self.acts, self.offers = [], Counter()
        self._rows = None
        self._last = None
        self._busy = set()


class LcCellBankCore(LcOracleCore):
    """a37_cellbank.CellBankCore with the lc admission: unconditional rows, BANK_ALWAYS start rule (same function)."""
    _gate = A32.AlwaysCore._gate

    def __init__(self, bank, cell_key, warmup_s=300):
        super().__init__(bank, cell_key, warmup_s, plan_source=None)
        if any(p['row_class'] != 'unconditional' for p in self.plans):
            raise ValueError('A44: every row must be unconditional')

    def receipts(self):
        r = super().receipts()
        lc = [dict(x, family=p['family']) for c, p in zip(self.cores, self.plans) for x in c.lc_receipts()]
        r.update(schema=SCHEMA, executor='cell_bank_always_lc', lc_members=lc,
                 lc_summary=dict(members=len(lc), requested=sum(x['first_request'] is not None for x in lc),
                                 completed=sum(x['completed'] is not None for x in lc),
                                 gave_up=sum(x['gave_up'] is not None for x in lc),
                                 never_requested=sum(x['first_request'] is None for x in lc)))
        return r


def bank_doc(plans, cell_key):
    """a37_cellbank.bank_doc with the A44 refusal rule (lc plans allowed); same row / template form and fields."""
    rows, tpls = [], []
    for p in plans:
        s = p['strategy']
        if s['name'] != p['family'] or executor_refusal(s) is not None:
            raise ValueError(f'{p["family"]}: not executable by the A44 executor ({executor_refusal(s)})')
        cond = dict(trigger=s['trigger'], zone=s['zone'], spread_m=float(s['spread_m']), cooldown_s=int(s['cooldown_s']),
                    slots=int(s['slots']), gate=s['gate'], start_gate=s['start_gate'], phases=s['phases'],
                    situation=dict(unconditional=True),
                    roles=[dict(sumo_lane=int(r['lane']), rank=int(r['rank']), v_w=float(r['v_w']), dmax=float(r['dmax']),
                                hold_s=int(r['hold_s']), onset_s=int(r['onset_s']), lc_target_sumo=r['lc_target'])
                           for r in s['roles']])
        rows.append(dict(regime=cell_key, family=s['name'], lb95_veh_s=float(p.get('lb95_veh_s', 0.0)),
                         plan_source=C.PLAN_SOURCE, row_class='unconditional', situational=dict(unconditional=True),
                         template_sha256=None, strategy_key=O.strategy_key(s)))
        tpls.append(dict(cell=cell_key, family=s['name'], offer_condition=cond,
                         template=dict(source=dict(strategy_key=O.strategy_key(s)))))
    return dict(schema=SCHEMA+'.bank', rows=rows, clean5c=dict(templates=tpls))


def make_core(arm, cell_key, bank, family=None, warmup_s=300):
    if arm == 'A44_PLAN':
        core = LcCellBankCore(C.restrict(bank, family), cell_key, warmup_s)
        if [p['family'] for p in core.plans] != [family]:
            raise ValueError(f'A44_PLAN: {family} refused by the executor {core.refused}')
        return core
    if arm == 'A44_BANK':
        core = LcCellBankCore(bank, cell_key, warmup_s)
        want = {r['family'] for r in bank['rows'] if r['regime'] == cell_key}
        if not core.plans or {p['family'] for p in core.plans} != want:
            raise ValueError(f'A44_BANK: admitted plans {sorted(want)} not all executable {core.refused}')
        return core
    raise ValueError(f'unknown A44 arm {arm!r}')


def make_hook(arm, cell_key, **kw):
    """controller_hook (RULE-arm protocol), identical to a37_cellbank.make_hook except the core."""

    class A44Hook:
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
            return dict(self.core.receipts(), a44_arm=arm)

    return A44Hook()


EXECUTED = ('lc_refusal', 'is_lc_plan', 'executor_refusal', 'lc_bank_plans', 'LcStrategyCore', 'LcOracleCore',
            'LcCellBankCore', 'bank_doc', 'make_core', 'make_hook')
CONSTANTS = ('SCHEMA', 'A44_ARMS', 'GIVEUP_OFF', 'LC_GIVEUP_X', 'G5_REFUSED_WM_TARGET', 'LC_REFUSALS', 'REFUSALS')


def a44_digest(module=None):
    """sha256 of the A44 executor as executed: a37_digest() (A37 executor incl. oracle_plan + a32_arms, which this
    variant inherits) + every executed function/class here + the constants."""
    import sys
    m = module or sys.modules[__name__]
    parts = [C.a37_digest()]
    parts += [inspect.getsource(getattr(m, n)) for n in EXECUTED]
    parts.append(json.dumps({n: getattr(m, n) for n in CONSTANTS}, sort_keys=True, default=list))
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()
