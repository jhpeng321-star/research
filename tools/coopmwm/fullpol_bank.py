"""clean5c guidance: WMVERIFY multi-CAV verified-plan bank -> EV-BC targets per cell (owner REGISTRATION, MAIN 2026-11-08).

Input: wmverify bank rows (schema ``wmverify.bank.v1``: one row per activation; family, members, cell,
seed, diff_vs_u0 [strategy - u0, veh-s, episode], cell_groups_vs_u0 [per-group mean per-vehicle delay
increase, s], executor, parity verdicts, template, offer_condition) plus, when present, the registered
S-WMVERIFY ANALYSIS (cells[cell][family]: mean_diff, ci95, n). Fallback: the F8S fair-rule guidance rows
(fullpol_fair.build_guidance, c4/fullpol fc36f646).

Per (cell, family) of MULTI-CAV plans only (>= 2 members in every activation):
* gain = -(strategy - u0); lb95 = -ci95_hi of the registered ANALYSIS (else the same paired seed bootstrap
  as rg_analyze on the bank's per-seed diff_vs_u0: 10,000 resamples, RNG 20261106);
* worst group increase = max over origin:L0/L1/L2, class:hdv, class:cav of the seed-mean point increase;
* congested cells (D3200_*): DELAY priority - eligible iff lb95 > 0; plans over the 10 s bound stay
  eligible but are labelled ``over_fair_bound`` (the training fairness penalty stays on);
* less congested cells (D2800_*, D2400_*): FAIRNESS priority - eligible iff lb95 > 0 AND worst <= 10 s.
A cell with no eligible verified multi-CAV plan uses the F8S rule targets for that cell. The policy, plant
and B/R code paths are unchanged; only the guidance (and the arm's verified templates) differ.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .fullpol_fair import THRESHOLD_S

RULE = 'fullpol-clean5c-bank-guidance-v1'
WM_GROUPS = ('wm', 'wm_fairer', 'wm_reserve')        # MAIN 2026-11-08: WM-guided targets come only from these
PARITY_PASS = 'identical'                            # N2 parity verdict (fake-plant, tick-by-tick) required
GROUPS = ('origin:L0', 'origin:L1', 'origin:L2', 'class:hdv', 'class:cav')
CONGESTED_DEMAND = 3200
PROVENANCE = ('owner REGISTRATION / MAIN 2026-11-08: clean5c EV-BC from the WMVERIFY multi-CAV bank; D3200 delay '
              'priority (over-bound plans labelled, fairness penalty on), D2800/D2400 fairness priority (<= 10 s); '
              'multi-CAV only; cells without a verified plan use the F8S rule targets; B = R except guidance')


def congested(cell):
    return int(str(cell).split('_')[0].lstrip('D')) >= CONGESTED_DEMAND


def _boot(d):
    d = np.asarray(d, float)
    idx = np.random.default_rng(20261106).integers(0, len(d), (10000, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def plans(rows, analysis=None):
    """{(cell, family): summary} over bank rows (multi-CAV families only; others listed in 'skipped')."""
    by = defaultdict(list)
    for r in rows:
        by[(r['cell'], r['family'])].append(r)
    out, skipped = {}, {}
    for (cell, fam), rs in sorted(by.items()):
        if any(len(r['members']) < 2 for r in rs):
            skipped[f'{cell}/{fam}'] = 'not multi-CAV'
            continue
        per_seed = {}
        for r in rs:
            per_seed.setdefault(r['seed'], r)          # episode-level fields repeat per activation
        seeds = sorted(per_seed)
        stats = ((analysis or {}).get('cells') or {}).get(cell, {}).get(fam)
        if stats is not None and {'mean_diff', 'ci95'} <= set(stats):
            mean, lo, hi, src = float(stats['mean_diff']), float(stats['ci95'][0]), float(stats['ci95'][1]), 'analysis'
        elif len(seeds) >= 2:
            mean, lo, hi = _boot([per_seed[s]['diff_vs_u0'] for s in seeds])
            src = 'bank_bootstrap'
        else:
            skipped[f'{cell}/{fam}'] = 'fewer than 2 seeds'
            continue
        groups = {}
        for g in GROUPS:
            vals = [per_seed[s]['cell_groups_vs_u0'][g] for s in seeds
                    if (per_seed[s].get('cell_groups_vs_u0') or {}).get(g) is not None]
            groups[g] = float(np.mean(vals)) if vals else None
        first = rs[0]
        out[(cell, fam)] = dict(cell=cell, family=fam, n_seeds=len(seeds), activations=len(rs),
                                mean_veh_s=-mean, lb95_veh_s=-hi, ub95_veh_s=-lo, gain_source=src,
                                group_increase_s=groups,
                                worst_group_increase_s=(max(groups.values()) if all(v is not None for v in groups.values())
                                                        else None),
                                executor=first.get('executor'), strategy_key=first.get('strategy_key'),
                                members=max(len(r['members']) for r in rs),
                                lcexec_parity=first.get('lcexec_parity'), n2_parity=first.get('n2_parity_verdict'),
                                template=first.get('template'), template_sha256=first.get('template_sha256'),
                                offer_condition=first.get('offer_condition'))
    return out, skipped


def eligible(p, threshold_s=THRESHOLD_S):
    """(eligible, label) per the registered per-cell priority."""
    if not p['lb95_veh_s'] > 0:
        return False, 'lb95<=0'
    worst = p['worst_group_increase_s']
    if congested(p['cell']):
        return True, ('over_fair_bound' if worst is None or worst > threshold_s else 'within_fair_bound')
    if worst is None:
        return False, 'groups_missing'
    return (worst <= threshold_s), ('within_fair_bound' if worst <= threshold_s else 'over_fair_bound')


def build(bank_paths, *, fallback, analysis_paths=(), provenance, arm='B', arms=None, verify=None,
          require_parity=True):
    """Guidance doc (fullpol-regime-guidance-v1): bank targets where verified, F8S rule rows elsewhere."""
    from .fullpol_learn import guidance_doc
    rows, sources, analysis = [], [], {'cells': {}}
    for path in analysis_paths:
        raw = Path(path).read_bytes()
        for cell, fams in json.loads(raw)['cells'].items():
            analysis['cells'].setdefault(cell, {}).update(fams)
        sources.append(dict(name=Path(path).name, sha256=hashlib.sha256(raw).hexdigest(), role='analysis'))
    bank = []
    for path in bank_paths:
        raw = Path(path).read_bytes()
        bank += load_rows(path)
        sources.append(dict(name=Path(path).name, sha256=hashlib.sha256(raw).hexdigest(), role='bank'))
    summary, skipped = plans(bank, analysis)
    verified_cells, rejected = set(), {}
    wm = wm_families(arms) if arms is not None else None
    parity = {f: (v.get('n2_parity') or {}).get('verdict') for f, v in ((verify or {}).get('families') or {}).items()}
    for (cell, fam), p in sorted(summary.items()):
        ok, label = eligible(p)
        verdict = p['n2_parity'] or parity.get(fam)
        if ok and wm is not None and fam not in wm.get(cell, set()):
            ok, label = False, 'not_wm_sourced'
        if ok and require_parity and verdict != PARITY_PASS:
            ok, label = False, f'n2_parity_{verdict}'
        p['n2_parity'] = verdict
        row = dict(arm=arm, regime=cell, family=fam, baseline='mappo', n=p['n_seeds'], mean_veh_s=p['mean_veh_s'],
                   lb95_veh_s=p['lb95_veh_s'], ub95_veh_s=p['ub95_veh_s'], fair=ok, fair_label=label,
                   worst_group_increase_s=p['worst_group_increase_s'], group_increase_s=p['group_increase_s'],
                   source='wm_bank', executor=p['executor'], members=p['members'],
                   lcexec_parity=p['lcexec_parity'], n2_parity=p['n2_parity'], template_sha256=p['template_sha256'])
        p['target'] = ok
        row['n2_parity'] = p['n2_parity']
        if ok:
            rows.append(row)                   # only eligible plans become guidance (safe without ev_filter)
            verified_cells.add(cell)
        else:
            rejected[f'{cell}/{fam}'] = dict(reason=label, lb95_veh_s=row['lb95_veh_s'],
                                             worst_group_increase_s=row['worst_group_increase_s'])
    fb = fallback if isinstance(fallback, dict) else json.loads(Path(fallback).read_text(encoding='utf-8'))
    for r in fb['rows']:
        if r['arm'] == arm and r['regime'] not in verified_cells:
            rows.append(dict(r, source='f8s_rule', fallback=True))
    sources.append(dict(name='F8S fallback', sha256=fb['source']['sha256'], role='fallback'))
    doc = guidance_doc(rows, sources=sources, provenance=provenance)
    doc['clean5c'] = dict(rule=RULE, provenance=PROVENANCE, threshold_s=THRESHOLD_S, verified_cells=sorted(verified_cells),
                          skipped=skipped, rejected=rejected,
                          templates=templates(summary, verified_cells))
    return doc


def templates(summary, cells=None):
    """Verified multi-CAV slot templates (lcexec-free-program-v1) labelled with their strategy name, for the
    arm's FULLPOL generator ('verified' class); offer_condition is carried for the runtime gate."""
    out = []
    for (cell, fam), p in sorted(summary.items()):
        if p['template'] is None or (cells is not None and cell not in cells) or not p.get('target'):
            continue
        t = json.loads(json.dumps(p['template']))
        t['source'] = dict(t.get('source') or {}, variant=fam, family=fam, cell=cell)
        out.append(dict(cell=cell, family=fam, template=t, offer_condition=p['offer_condition']))
    return out


def eligible_families(doc):
    """Parity list for EV-BC: every eligible bank family plus the F8S fallback's fair families."""
    return sorted({r['family'] for r in doc['rows'] if r.get('fair')})


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='clean5c bank guidance (CPU, no physics)')
    ap.add_argument('--bank', nargs='+', required=True)
    ap.add_argument('--analysis', nargs='*', default=())
    ap.add_argument('--fallback', required=True, help='F8S guidance json (fullpol_fair.build_guidance)')
    ap.add_argument('--arms', nargs='+', required=True, help='WM-C ARMS.json files or the WMVERIFY PLAN.json (WM group membership)')
    ap.add_argument('--verify', required=True, help='WMVERIFY VERIFY.json (N2 parity verdicts)')
    ap.add_argument('--provenance', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit('refusing to overwrite '+str(out))
    doc = build(a.bank, fallback=a.fallback, analysis_paths=a.analysis, provenance=a.provenance,
                arms=[json.loads(Path(p).read_text(encoding='utf-8')) for p in a.arms],
                verify=json.loads(Path(a.verify).read_text(encoding='utf-8')))
    out.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding='utf-8')
    print(json.dumps(dict(out=str(out), sha256=doc['source']['sha256'], verified_cells=doc['clean5c']['verified_cells'],
                          targets=eligible_families(doc))))



def wm_families(arms_docs):
    """{cell: WM-sourced family names} from WM-C ARMS.json docs ({cell: {group: [names]}})."""
    out = {}
    for doc in arms_docs:
        if 'families' in doc and 'cells' in doc:          # WMVERIFY PLAN.json: families[f].cells[cell] = group
            for fam, v in doc['families'].items():
                for cell, g in (v.get('cells') or {}).items():
                    if g in WM_GROUPS:
                        out.setdefault(cell, set()).add(fam)
            continue
        for cell, groups in doc.items():
            out.setdefault(cell, set()).update(n['name'] if isinstance(n, dict) else n
                                               for g in WM_GROUPS for n in (groups.get(g) or ()))
    return out


def offer_ok(cond, own, rows):
    """Verified trigger/offer condition at this step from the acting CAV's own row + own sensing (WM frame).

    Checked: own WM lane in own_wm_lanes; own s inside zone_s; trigger 'wall' = a CAV in every other WM lane
    within spread_m (own sensing); trigger 'lanes' = a CAV in each other role lane within spread_m. Gates /
    start gates are not evaluable from this view: fail closed (never offered). Cooldown / slots are session
    limits enforced by the protocol, not here."""
    if not cond or cond.get('gate') or cond.get('start_gate'):
        return False
    lane, s = int(own['lane']), float(own['s_m'])
    if int(lane) not in [int(x) for x in cond.get('own_wm_lanes') or ()]:
        return False
    zone = cond.get('zone_s')
    if zone is not None and not float(zone[0]) <= s <= float(zone[1]):
        return False
    spread = float(cond.get('spread_m', 0.))
    cavs = [r for r in rows if r.get('is_cav') and r['vid'] != own['vid'] and abs(float(r['s_m'])-s) <= spread]
    if cond.get('trigger') == 'wall':
        need = {int(r['wm_lane']) for r in cond.get('roles') or ()} | {0, 1, 2}
    elif cond.get('trigger') == 'lanes':
        need = {int(r['wm_lane']) for r in cond.get('roles') or ()}
    else:
        return False
    return all(any(int(r['lane']) == ln for r in cavs) for ln in need - {lane})


if __name__ == '__main__':
    main()
