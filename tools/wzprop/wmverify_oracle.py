"""S-WMVERIFY PAIR arms (owner chat 2026-11-09 01:38, MAIN pre-registration): two (or more) verified hold plans
executed unconditionally and CONCURRENTLY on different CAV groups, with the reviewed ORACLE hook as executor
(tools/coopmwm/oracle_plan.py = c4/oracle-arm 8bc436df, REV-SPD PASS; git blob ed6fd88a, oracle_digest 7409a7b4...):
per-family bank session rule (each family's own slots / cooldown), a CAV never in two sessions, families tried in
mini-bank priority order each second, unconditional rows (no bin gate).

A pair arm = a mini-bank (clean5c guidance-doc shape) of unconditional rows built from the verified WM-C specs:
offer_condition = wmverify_hook.offer_condition(spec) (the bank's own function), template.source.strategy_key =
strategy_search.strategy_key(spec), priority = the given order (first entry first). The harness hook is a Recorder
(per-second totals series, SUMMARY, CRN checks, vehicle log: the unchanged regimeu0 / wmverify worker path) whose
controller hook is oracle_plan.make_hook(mini_bank, cell_key).
"""
from __future__ import annotations

import hashlib
import json

from tools.coopmwm import oracle_plan as OP
from . import strategy_search as SS
from . import wmverify_hook as VH

EXECUTOR = 'oracle_bank'
ORACLE_BLOB = 'ed6fd88ad5fbf97ce42af36b5b8c2aa905562b66'        # git blob of oracle_plan.py at c4/oracle-arm 8bc436df
ORACLE_DIGEST = '7409a7b4765651813c4771462b7acbc16386d626900d9a031e19d3a65a9f9d22'


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def check_oracle():
    """The executed oracle_plan must be the reviewed one (digest of everything it executes)."""
    if OP.oracle_digest() != ORACLE_DIGEST or not all(OP.vendored_check().values()):
        raise ValueError('oracle_plan differs from the REV-SPD-reviewed 8bc436df (oracle_digest)')
    return ORACLE_DIGEST


def mini_bank(cell, specs, plan_sources=None):
    """Unconditional mini-bank for one pair arm: specs in priority order (first = highest)."""
    rows, tpls = [], []
    n = len(specs)
    for i, spec in enumerate(specs):
        s = SS.validate(dict(spec))
        if VH.has_lane_roles(s):
            raise ValueError(f'pair arms are hold-only: {s["name"]}')
        key = SS.strategy_key(s)
        src = (plan_sources or {}).get(s['name']) or ('wm' if s['name'].startswith('mut_') else 'rule')
        rows.append(dict(arm='B', regime=cell, family=s['name'], lb95_veh_s=float(n - i), plan_source=src,
                         situational=dict(unconditional=dict(unconditional=True, rule='PAIR (MAIN 2026-11-09 01:38)')),
                         template_sha256=key))
        tpls.append(dict(cell=cell, family=s['name'], template=dict(source=dict(strategy_key=key, strategy=s['name'])),
                         offer_condition=VH.offer_condition(s)))
    doc = dict(schema='wmverify.pair_minibank.v1', rows=rows, clean5c=dict(templates=tpls))
    plans, refused = OP.bank_plans(doc, cell)
    if refused or [p['family'] for p in plans] != [s['name'] for s in specs] or \
            any(p['row_class'] != 'unconditional' for p in plans):
        raise ValueError(f'mini-bank does not rebuild exactly (refused {refused})')
    return doc


def pair_arms(pairs_doc):
    """{cell: [(name, mini_bank)]} from {cell: [{name, specs: [...]}]}; names unique per cell."""
    out = {}
    for cell, items in pairs_doc.items():
        lst, seen = [], set()
        for it in items:
            if it['name'] in seen:
                raise ValueError(f'duplicate pair arm {it["name"]!r}')
            seen.add(it['name'])
            lst.append((it['name'], mini_bank(cell, it['specs'], it.get('plan_sources'))))
        out[cell] = lst
    return out


def bank_sha(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def make_recorder(bank_doc, cell_key):
    """Harness hook (regimeu0 Recorder protocol: acts / issued / lane_held / series / tick_wrapper) wrapping the
    reviewed ORACLE controller hook on the mini-bank."""
    from tools.coopmwm import imitreplay as ir
    check_oracle()

    class OracleRecorder(ir.Recorder):
        def __init__(self):
            super().__init__()
            self.oracle = OP.make_hook(bank_doc, cell_key)
            self.strategy = 'pair:' + '+'.join(p['family'] for p in self.oracle.core.plans)
            self.executor = EXECUTOR

        @property
        def acts(self):
            return self.oracle.core.acts

        @property
        def issued(self):
            return [x for c in self.oracle.core.cores for x in c.issued]

        @property
        def lane_held(self):
            return sum(c.lane_held for c in self.oracle.core.cores)

        @property
        def needs_lc_runtime(self):
            return False

        def receipts(self):
            return dict(executor=EXECUTOR, oracle_sha256=ORACLE_DIGEST, bank_sha256=bank_sha(bank_doc),
                        oracle=self.oracle.receipts())

        def __call__(self, controller, job, prefixes):
            self.controller = controller
            self.oracle(controller, job, prefixes)

    return OracleRecorder()
