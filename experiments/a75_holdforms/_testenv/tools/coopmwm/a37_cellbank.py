"""AMENDMENT 37 (MAIN 2026-11-10, condition tailoring on wz3to2; registered before any outcome of these arms).

Question: do plans the WM searched FOR a demand cell (STRATEGY_SEARCH_D3200_p20 / _D2400_p20, WM-E ft_r1, already run)
do better AT that cell than the frozen D2800 bank (LEARNED_TRIGGER) and the hand-written hold (DUO_HOLD)?

- Candidates (zero SUMO, fixed before any SUMO outcome): the search's ``final`` list ranked by the WM ``score`` only;
  comparator (reference) rows, banned rows and plans the registered plan executor refuses (oracle_plan.bank_plans:
  lane-change roles, phases, gate / start_gate outside the registered sets) are skipped; top K per cell.
- A37_PLAN (verification arm): ONE candidate as an unconditional bank row, started whenever feasible.
- CELL_BANK: the admitted candidates of the cell, started whenever feasible (no classifier), priority = verified lb95
  descending (oracle_plan.bank_plans order; the fd385a2b lb95 convention: gain = J(U0) - J(plan), lb95(gain) =
  -ub95(plan - U0)). No admitted plan -> CELL_BANK(cell) := U0 by definition (no episode runs).
- Executor = the BANK_ALWAYS executor exactly (a32_arms.AlwaysCore start rule on oracle_plan.OracleCore: vendored
  S-WMVERIFY StrategyCore hold law, per-family session/cooldown rule, member exclusivity, priority order; HDVs never
  commanded); only the plan set differs.
- Admission = the UNC rule that admitted the unconditional rows of bank fd385a2b (MAIN 2026-11-09 00:20, verbatim:
  "24 fresh seeds, unconditional (act whenever feasible), pass = (pick - u0) 95% CI upper < 0 AND ALLOWED"): CI =
  regimeu0.boot_ci (10,000 resamples, random.Random(20261106), per-seed paired diffs of J_veh_s); ALLOWED = the five
  registered groups present and each mean per-vehicle delay increase vs U0 <= 10 s (nm_finaleval.co_primary_groups,
  amendment 13, U0 labels).

No RNG, no clock in the executor; reads only the bank doc, the plant rows and the U0 commands.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import random

from . import a32_arms as A32
from . import oracle_plan as O

SCHEMA = 'coopmwm.a37_cellbank.v1'
A37_ARMS = ('A37_PLAN', 'CELL_BANK')
CELLS = ('D3200_p20', 'D2400_p20')
#: the existing WM searches (E:/research-dense/tmp/wmc_p20_r3/fetched; WM-E ft_r1; sha256 of the file bytes)
SEARCH_SHA256 = dict(D3200_p20='5674d95663678bac7926c2eee8b544982fdcdde29dbdb5832fea9ef2b073d92c',
                     D2400_p20='9fa5fa53bbe1343d4053c082a8ec63e35880a6db8be4292b24db84aa1648d75e')
K_MAX = 12                      # registration: <= 12 per cell
N_SEEDS = 24                    # UNC rule: 24 fresh seeds
EPISODE_SIM_S = 1200
VERIFY_CAP_SIM_S = 600_000      # registration: verification <= 600,000
VERIFY_RESERVE_SIM_S = 24_000   # 20 one-retry episodes kept inside the verification cap
GROUP_BOUND_S = 10.0
UNC_BOOT, UNC_BOOT_SEED = 10_000, 20261106
UNC_RULE = ('MAIN 2026-11-09 00:20 UNC rule (verbatim): 24 fresh seeds, unconditional (act whenever feasible), '
            'pass = (pick - u0) 95% CI upper < 0 AND ALLOWED')
PLAN_SOURCE = 'wm_search'


def k_per_cell(n_cells=len(CELLS)):
    """K fixed by the verification cap only: (U0 + K plans) x 24 seeds x 1,200 s per cell, all cells, plus the retry
    reserve, <= 600,000 -> K = 9 for 2 cells (never more than 12)."""
    per = N_SEEDS*EPISODE_SIM_S*n_cells
    return min(K_MAX, (VERIFY_CAP_SIM_S-VERIFY_RESERVE_SIM_S)//per-1)


def executor_refusal(spec):
    """The oracle_plan.bank_plans refusal class this spec would get from the registered executor, else None."""
    if any(r.get('lc_target') is not None for r in spec['roles']):
        return 'non_hold_plan'
    if spec.get('gate') not in O.GATES or spec.get('start_gate') not in O.START_GATES or spec.get('phases') is not None:
        return 'gate_or_phases'
    if spec.get('trigger') not in ('wall', 'lanes') or spec.get('zone') not in O.ZONES:
        return 'trigger_or_zone'
    return None


def spec_sha(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def select_candidates(search, cell, k=None):
    """Zero SUMO, WM ranking only. -> dict(candidates=[...], skipped=[...], ranking=[...])."""
    k = k_per_cell() if k is None else int(k)
    if search.get('cell') != cell:
        raise ValueError(f'search cell {search.get("cell")!r} is not {cell}')
    rows = sorted(search['final'], key=lambda x: (-float(x['score']), x['key']))
    ranking, skipped, cands = [], [], []
    for i, x in enumerate(rows):
        s = x['strategy']
        if O.strategy_key(s) != x['key']:
            raise ValueError(f'{s["name"]}: strategy_key differs from the search key')
        why = ('comparator' if x['comparator'] else 'banned' if x['banned'] else executor_refusal(s))
        entry = dict(wm_rank=i+1, name=s['name'], key=x['key'], score=x['score'], skip=why)
        ranking.append(entry)
        if why is not None:
            skipped.append(entry)
            continue
        if len(cands) < k:
            cands.append(dict(entry, family=s['name'], strategy=s, strategy_sha256=spec_sha(s),
                              wm=dict({f: x[f] for f in ('feasible', 'gain_vs_base', 'gain_vs_base_lcb', 'gain_vs_inc',
                                                        'worst_group_inc', 'cvar90_inc', 'bounded', 'fair_cal')},
                                      n_states=x['n_states']),
                              selected_rank=len(cands)+1))
    return dict(cell=cell, k=k, candidates=cands, skipped=skipped, ranking=ranking)


def bank_doc(plans, cell_key):
    """Unconditional bank rows + templates (the oracle_plan.bank_plans form) for ``plans`` =
    [dict(family, strategy, lb95_veh_s)]; template.source.strategy_key = strategy_key(spec)."""
    rows, tpls = [], []
    for p in plans:
        s = p['strategy']
        if s['name'] != p['family'] or executor_refusal(s) is not None:
            raise ValueError(f'{p["family"]}: not an executable hold plan')
        cond = dict(trigger=s['trigger'], zone=s['zone'], spread_m=float(s['spread_m']), cooldown_s=int(s['cooldown_s']),
                    slots=int(s['slots']), gate=s['gate'], start_gate=s['start_gate'], phases=s['phases'],
                    situation=dict(unconditional=True),
                    roles=[dict(sumo_lane=int(r['lane']), rank=int(r['rank']), v_w=float(r['v_w']), dmax=float(r['dmax']),
                                hold_s=int(r['hold_s']), onset_s=int(r['onset_s']), lc_target_sumo=r['lc_target'])
                           for r in s['roles']])
        rows.append(dict(regime=cell_key, family=s['name'], lb95_veh_s=float(p.get('lb95_veh_s', 0.0)),
                         plan_source=PLAN_SOURCE, row_class='unconditional', situational=dict(unconditional=True),
                         template_sha256=None, strategy_key=O.strategy_key(s)))
        tpls.append(dict(cell=cell_key, family=s['name'], offer_condition=cond,
                         template=dict(source=dict(strategy_key=O.strategy_key(s)))))
    return dict(schema=SCHEMA+'.bank', rows=rows, clean5c=dict(templates=tpls))


class CellBankCore(O.OracleCore):
    """oracle_plan.OracleCore over unconditional rows with the BANK_ALWAYS start rule (a32_arms.AlwaysCore._gate,
    the same function object): a plan starts whenever its group is feasible and no member is in another session."""
    _gate = A32.AlwaysCore._gate

    def __init__(self, bank, cell_key, warmup_s=300):
        super().__init__(bank, cell_key, warmup_s, plan_source=None)
        if any(p['row_class'] != 'unconditional' for p in self.plans):
            raise ValueError('A37: every row must be unconditional')

    def receipts(self):
        r = super().receipts()
        r.update(schema=SCHEMA, executor='cell_bank_always')
        return r


def restrict(bank, family):
    """The one-row bank of ``family`` (A37_PLAN verification arm)."""
    rows = [r for r in bank['rows'] if r['family'] == family]
    tpls = [t for t in bank['clean5c']['templates'] if t['family'] == family]
    if len(rows) != 1 or len(tpls) != 1:
        raise ValueError(f'A37_PLAN: family {family!r} not unique in the bank')
    return dict(bank, rows=rows, clean5c=dict(templates=tpls))


def make_core(arm, cell_key, bank, family=None, warmup_s=300):
    if arm == 'A37_PLAN':
        core = CellBankCore(restrict(bank, family), cell_key, warmup_s)
        if [p['family'] for p in core.plans] != [family]:
            raise ValueError(f'A37_PLAN: {family} refused by the executor {core.refused}')
        return core
    if arm == 'CELL_BANK':
        core = CellBankCore(bank, cell_key, warmup_s)
        want = {r['family'] for r in bank['rows'] if r['regime'] == cell_key}
        if not core.plans or {p['family'] for p in core.plans} != want:
            raise ValueError(f'CELL_BANK: admitted plans {sorted(want)} not all executable {core.refused}')
        return core
    raise ValueError(f'unknown A37 arm {arm!r}')


def make_hook(arm, cell_key, **kw):
    """controller_hook (RULE-arm protocol), identical to a32_arms.make_hook except the core."""

    class A37Hook:
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
            return dict(self.core.receipts(), a37_arm=arm)

    return A37Hook()


# ------------------------------------------------------------------ admission (zero physics)

def boot_ci(diffs, *, n=UNC_BOOT, seed=UNC_BOOT_SEED):
    """regimeu0.boot_ci verbatim (the S-WMVERIFY / UNC verdict bootstrap)."""
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k))/k for _ in range(n))
    return means[int(.025*n)], means[int(.975*n)-1]


def unc_verdict(per_seed_plan, per_seed_u0, groups):
    """per_seed_*: {seed: J_veh_s}; groups: nm_finaleval.co_primary_groups output (plan vs U0, U0 labels).
    PASS = (plan - U0) CI95 upper < 0 AND ALLOWED (all five groups present, each increase <= 10 s)."""
    seeds = sorted(per_seed_u0)
    if sorted(per_seed_plan) != seeds or len(seeds) != N_SEEDS:
        raise ValueError(f'UNC: needs the same {N_SEEDS} paired seeds')
    diffs = [float(per_seed_plan[s])-float(per_seed_u0[s]) for s in seeds]
    lo, hi = boot_ci(diffs)
    allowed = bool((groups or {}).get('all_pass') is True)
    worst = max(((g, v['increase_mean_s']) for g, v in (groups or {}).get('groups', {}).items()
                 if v.get('registered') and 'increase_mean_s' in v), key=lambda x: x[1], default=(None, None))
    return dict(rule=UNC_RULE, n=len(seeds), mean=sum(diffs)/len(diffs), ci95=[lo, hi], allowed=allowed,
                benefit=sum(d < 0 for d in diffs)/len(diffs), worst_group=worst[0], worst_group_inc=worst[1],
                PASS=bool(hi < 0 and allowed), lb95_gain_veh_s=-hi, mean_gain_veh_s=-sum(diffs)/len(diffs),
                per_seed={str(s): round(d, 1) for s, d in zip(seeds, diffs)},
                convention='diff = plan - U0 (negative = less delay); gain = -diff (fd385a2b lb95 convention)')


def admitted_plans(candidates, verdicts):
    """Admitted = PASS; bank order by verified lb95(gain) desc then strategy_key (oracle_plan.bank_plans)."""
    out = []
    for c in candidates:
        v = verdicts.get(c['family'])
        if v is not None and v['PASS']:
            out.append(dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=float(v['lb95_gain_veh_s'])))
    out.sort(key=lambda p: (-p['lb95_veh_s'], O.strategy_key(p['strategy'])))
    return out


EXECUTED = ('executor_refusal', 'bank_doc', 'CellBankCore', 'restrict', 'make_core', 'make_hook')
CONSTANTS = ('SCHEMA', 'A37_ARMS', 'PLAN_SOURCE')


def a37_digest(module=None):
    """sha256 of the A37 executor as executed: a32_digest() (shared executor + AlwaysCore) + every executed
    function/class here + the constants."""
    import sys
    m = module or sys.modules[__name__]
    parts = [A32.a32_digest()]
    parts += [inspect.getsource(getattr(m, n)) for n in EXECUTED]
    parts.append(json.dumps({n: getattr(m, n) for n in CONSTANTS}, sort_keys=True, default=list))
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()
