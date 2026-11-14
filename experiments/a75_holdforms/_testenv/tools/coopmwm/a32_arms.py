"""AMENDMENT 32 ablation arms (MAIN 2026-11-10, owner-approved; "what produces the gain"). Registered before any
outcome of these arms. Every arm = U0 (frozen MAPPO + bound u0) + the SAME plan executor as LEARNED_TRIGGER
(oracle_plan.OracleCore: the vendored S-WMVERIFY StrategyCore hold executor, per-family bank session/cooldown rule,
member exclusivity, priority order, HDVs never commanded); only the start rule or the plan set differs:

- SOLO_HOLD: the incumbent single-CAV hold wall_g_solo2 (c4/wmverify strategy_search.INCUMBENT: wall trigger, the
  SUMO lane-2 member holds 22 m/s for 120 s, dmax 1.0, cooldown 120 s), started whenever feasible.
- DUO_HOLD: the hand-written two-CAV hold duo12_v20 = the rule row of UNION_UNC 39f384a2 (unconditional row),
  started whenever feasible.
- BANK_ALWAYS: the LEARNED_TRIGGER bank (WM-only fd385a2b rows, the same classifier-family filter and priority
  order) with the classifier gate removed: a plan starts whenever its group is feasible.

No RNG, no clock; reads only the bank doc(s), the plant rows and the U0 commands.
"""
from __future__ import annotations

import hashlib
import inspect
import json

from . import learned_trigger as L
from . import oracle_plan as O

SCHEMA = 'coopmwm.a32_arms.v1'
A32_ARMS = ('SOLO_HOLD', 'DUO_HOLD', 'BANK_ALWAYS')
#: c4/wmverify daf08698 tools/wzprop/strategy_search.INCUMBENT (== ARMS_UNC_D2800.json D2800_p20/incumbent)
SOLO_SPEC = dict(name='wall_g_solo2', trigger='wall', zone='full', spread_m=200.0, cooldown_s=120, slots=1, gate=None,
                 start_gate=None, phases=None,
                 roles=[dict(lane=2, rank=0, v_w=22.0, dmax=1.0, hold_s=120, onset_s=0, lc_target=None)])
SOLO_KEY = 'c6b176478af8c50c'
DUO_FAMILY = 'duo12_v20'
DUO_KEY = '750a55c9439a2efc'
DUO_BANK_SHA256 = '39f384a29fb8e952a94f918234c52bb3ff2059f2155b99f17b2f5c26a38ec239'


def solo_bank(cell_key):
    """A one-row bank doc carrying SOLO_SPEC as an unconditional rule row (the form oracle_plan.bank_plans reads)."""
    s = SOLO_SPEC
    cond = dict(trigger=s['trigger'], zone=s['zone'], spread_m=s['spread_m'], cooldown_s=s['cooldown_s'],
                slots=s['slots'], gate=s['gate'], start_gate=s['start_gate'], phases=s['phases'],
                situation=dict(unconditional=True),
                roles=[dict(sumo_lane=r['lane'], rank=r['rank'], v_w=r['v_w'], dmax=r['dmax'], hold_s=r['hold_s'],
                            onset_s=r['onset_s'], lc_target_sumo=r['lc_target']) for r in s['roles']])
    return dict(rows=[dict(regime=cell_key, family=s['name'], lb95_veh_s=0.0, plan_source='rule',
                           situational=dict(unconditional=True), template_sha256=None)],
                clean5c=dict(templates=[dict(cell=cell_key, family=s['name'], offer_condition=cond,
                                             template=dict(source=dict(strategy_key=SOLO_KEY)))]))


def duo_bank(union_doc, cell_key):
    """UNION_UNC restricted to its duo12_v20 rule row (and template) for this cell."""
    rows = [r for r in union_doc['rows'] if r.get('regime') == cell_key and r['family'] == DUO_FAMILY]
    tpls = [t for t in (union_doc.get('clean5c') or {}).get('templates', ())
            if t['cell'] == cell_key and t['family'] == DUO_FAMILY]
    if len(rows) != 1 or len(tpls) != 1:
        raise ValueError(f'DUO_HOLD: {DUO_FAMILY} row/template not unique in the bank for {cell_key}')
    return dict(rows=rows, clean5c=dict(templates=tpls))


def _single_plan(core, family, key):
    if [p['family'] for p in core.plans] != [family] or core.plans[0]['key'] != key \
            or core.plans[0]['row_class'] != 'unconditional':
        raise ValueError(f'A32: expected exactly the unconditional plan {family} ({key}), got '
                         f'{[(p["family"], p["key"], p["row_class"]) for p in core.plans]} refused {core.refused}')
    return core


class AlwaysCore(L.LearnedCore):
    """LEARNED_TRIGGER (same WM rows, classifier-family filter, priority, sessions) with the classifier gate removed."""

    def _gate(self, core, plan):
        def select(s, free, observed, phase=0):
            mem = O.select_members(s, free, observed, phase=phase)
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

    def receipts(self):
        r = super().receipts()
        r.update(schema=SCHEMA, executor='bank_always', threshold=None, scores=0, fires=None)
        return r


def make_core(arm, cell_key, *, bank_doc=None, union_doc=None, model=None, warmup_s=300):
    if arm == 'SOLO_HOLD':
        return _single_plan(O.OracleCore(solo_bank(cell_key), cell_key, warmup_s), SOLO_SPEC['name'], SOLO_KEY)
    if arm == 'DUO_HOLD':
        return _single_plan(O.OracleCore(duo_bank(union_doc, cell_key), cell_key, warmup_s), DUO_FAMILY, DUO_KEY)
    if arm == 'BANK_ALWAYS':
        return AlwaysCore(bank_doc, cell_key, model, warmup_s)
    raise ValueError(f'unknown A32 arm {arm!r}')


def make_hook(arm, cell_key, **kw):
    """controller_hook (RULE-arm protocol), identical to oracle_plan.make_hook / learned_trigger.make_hook except the
    core. ``job`` and ``prefixes`` are never read."""

    class A32Hook:
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
            return dict(self.core.receipts(), a32_arm=arm)

    return A32Hook()


EXECUTED = ('solo_bank', 'duo_bank', '_single_plan', 'AlwaysCore', 'make_core', 'make_hook')
CONSTANTS = ('SCHEMA', 'A32_ARMS', 'SOLO_SPEC', 'SOLO_KEY', 'DUO_FAMILY', 'DUO_KEY', 'DUO_BANK_SHA256')


def a32_digest(module=None):
    """sha256 of the A32 code as executed: oracle_digest() (the shared executor) + LearnedCore (BANK_ALWAYS base) +
    every function/class here + the constants."""
    import sys
    m = module or sys.modules[__name__]
    parts = [O.oracle_digest(), inspect.getsource(L.LearnedCore), inspect.getsource(L.check_model)]
    parts += [inspect.getsource(getattr(m, n)) for n in EXECUTED]
    parts.append(json.dumps({n: getattr(m, n) for n in CONSTANTS}, sort_keys=True, default=list))
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()
