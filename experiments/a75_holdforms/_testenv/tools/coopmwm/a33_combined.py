"""AMENDMENT 33 COMBINED arm (MAIN 2026-11-10, owner "register option 1 and start it"). Registered before any
amendment-33 data exist.

COMBINED = the LEARNED_TRIGGER executor (oracle_plan.OracleCore: vendored S-WMVERIFY StrategyCore hold executor,
per-family bank session/cooldown rule, member exclusivity, U0 base, HDVs never commanded) on a bank of the 5 WM
plans of GUIDANCE_R6_WMONLY_PRIORITY fd385a2b PLUS the hand-written two-CAV hold duo12_v20 (the rule row of UNION_UNC
39f384a2), priority order = the LEARNED_TRIGGER order of the WM plans with duo12_v20 appended LAST. Start gate =
learned_trigger.LearnedCore's gate (p = classifier(features) >= threshold) with a classifier retrained by the EXACT
LEARNED_TRIGGER recipe (learned_trigger_train: labels, weights, seed-grouped 5-fold CV, C grid by onset AUC,
threshold = argmax OOF net verified benefit) on the original WM-plan rows (CANDS b2fc52f4 + LABELED a110af44,
reconstruction refit == model 8e0ddd3e bit-exact) plus duo12_v20 verification executions on fresh TRAIN-type seeds
(unconditional row: diff = episode J(duo12_v20) - J(u0), reference tape u0, lookback 10 s as DEMOX). The plan one-hot
grows from 5 to 6 families (69 features; disclosed).

No RNG, no clock at decision time; reads only the bank docs, the model doc, the plant rows and the U0 commands.
"""
from __future__ import annotations

import hashlib
import inspect
import json

from . import learned_trigger as L
from . import oracle_plan as O

SCHEMA = 'coopmwm.a33_combined.v1'
ARM = 'COMBINED'
WM_BANK_SHA256 = 'fd385a2b8b1cbaf33897f8617b6d4c0abeb9f2dea40c9b72dca91d718bfc3c82'
DUO_BANK_SHA256 = '39f384a29fb8e952a94f918234c52bb3ff2059f2155b99f17b2f5c26a38ec239'
DUO_FAMILY = 'duo12_v20'
DUO_KEY = '750a55c9439a2efc'
WM_FAMILIES = ('mut_006c290e03ad3d3c', 'mut_0d50e4e18041530d', 'mut_6c2bf382711cbc37', 'mut_b66c4207002b5a15',
               'mut_cfdbafe5f3316f1a')
FAMILIES = tuple(sorted(WM_FAMILIES + (DUO_FAMILY,)))
UNC_LOOKBACK_S = 10        # DEMOX 2b65aeab: unconditional rows look back 10 s
TRAIN_RANGE = (991600, 991699)   # amendment-33 fresh TRAIN-type block (audited; inside learned_trigger VERIFY_RANGES)


def combined_bank(wm_doc, union_doc, cell_key):
    """The WM rows (+ templates) of the WM-only doc for this cell + the duo12_v20 rule row (+ template) of UNION_UNC."""
    rows = [r for r in wm_doc['rows'] if r.get('regime') == cell_key]
    tpls = [t for t in (wm_doc.get('clean5c') or {}).get('templates', ()) if t['cell'] == cell_key]
    if any(r.get('plan_source') != 'wm' for r in rows):
        raise ValueError('COMBINED: the WM doc carries non-WM rows')
    duo = [r for r in union_doc['rows'] if r.get('regime') == cell_key and r['family'] == DUO_FAMILY]
    dt = [t for t in (union_doc.get('clean5c') or {}).get('templates', ())
          if t['cell'] == cell_key and t['family'] == DUO_FAMILY]
    if len(duo) != 1 or len(dt) != 1 or duo[0].get('plan_source') != 'rule':
        raise ValueError(f'COMBINED: {DUO_FAMILY} rule row/template not unique in the union doc for {cell_key}')
    return dict(rows=rows + duo, clean5c=dict(templates=tpls + dt))


def ordered_plans(doc, cell_key):
    """bank_plans(plan_source None) re-ordered: WM rows in their LEARNED_TRIGGER (bank_plans) order, rule rows last."""
    plans, refused = O.bank_plans(doc, cell_key, None)
    wm = [p for p in plans if p['plan_source'] == 'wm']
    rule = [p for p in plans if p['plan_source'] != 'wm']
    return wm + rule, refused


class CombinedCore(L.LearnedCore):
    """LearnedCore on the combined bank (plan_source None) with duo12_v20 appended last in the priority order."""

    def __init__(self, bank_doc, cell_key, model, warmup_s=300):
        self.model = L.check_model(model)
        self.threshold = float(model['threshold'])
        self.scores = []
        self._enc, self._t = None, None
        O.OracleCore.__init__(self, bank_doc, cell_key, warmup_s, plan_source=None)
        order, _ = ordered_plans(bank_doc, self.cell_key)
        pos = {p['family']: i for i, p in enumerate(order)}
        pairs = sorted(zip(self.cores, self.plans), key=lambda cp: pos[cp[1]['family']])
        keep, refused = [], {'not_in_classifier': [], 'classifier_key_mismatch': []}
        for core, p in pairs:
            if p['family'] not in model['families']:
                refused['not_in_classifier'].append(p['family'])
            elif model['strategy_keys'][p['family']] != p['key']:
                refused['classifier_key_mismatch'].append(p['family'])
            else:
                keep.append((core, p))
        self.cores, self.plans = [c for c, _ in keep], [p for _, p in keep]
        self.refused.update({k: v for k, v in refused.items() if v})
        if [p['family'] for p in self.plans][-1:] != [DUO_FAMILY] or len(self.plans) != len(FAMILIES):
            raise ValueError(f'COMBINED: expected 5 WM plans then {DUO_FAMILY}, got '
                             f'{[p["family"] for p in self.plans]} refused {self.refused}')

    def receipts(self):
        r = super().receipts()
        r.update(schema=SCHEMA, executor='combined', order=[p['family'] for p in self.plans])
        return r


def make_hook(bank_doc, cell_key, model, warmup_s=300):
    """controller_hook, identical to learned_trigger.make_hook except the core (CombinedCore)."""

    class CombinedHook:
        def __init__(self):
            self.core = CombinedCore(bank_doc, cell_key, model, warmup_s)
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
                    raise RuntimeError(f'COMBINED: request for non-CAV rows {sorted(bad)} (HDVs never commanded)')
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
            return self.core.receipts()

    return CombinedHook()


EXECUTED = ('combined_bank', 'ordered_plans', 'CombinedCore', 'make_hook')
CONSTANTS = ('SCHEMA', 'ARM', 'WM_BANK_SHA256', 'DUO_BANK_SHA256', 'DUO_FAMILY', 'DUO_KEY', 'WM_FAMILIES', 'FAMILIES')


def a33_digest(module=None):
    """sha256 of the COMBINED code as executed: learned_trigger_digest() (shared executor + featurizer + LearnedCore)
    + every function/class here + the constants. The model doc is pinned by its file sha256."""
    import sys
    m = module or sys.modules[__name__]
    parts = [L.learned_trigger_digest()] + [inspect.getsource(getattr(m, n)) for n in EXECUTED]
    parts.append(json.dumps({n: getattr(m, n) for n in CONSTANTS}, sort_keys=True, default=list))
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()
