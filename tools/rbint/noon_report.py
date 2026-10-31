"""Render the owner-requested <=60-line report from source-bound score receipts."""
import csv
from pathlib import Path

from tools.rbeval.core import load


def number(value, digits=0):
    return 'NA' if value is None else f'{value:.{digits}f}'


def ci(row, digits=0):
    if row is None or row.get('mean') is None:
        return 'PENDING;95%CI NA'
    if row.get('low') is None:
        return number(row['mean'], digits)+';95%CI NA'
    return f"{number(row['mean'], digits)}[{number(row['low'], digits)},{number(row['high'], digits)}]"


def rows(path):
    if path is None or not Path(path).is_file():
        return []
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def avg(data, key):
    values = [float(r[key]) for r in data if r.get(key) not in (None, '')]
    return sum(values)/len(values) if values else None


def comparison(item, split='reserved8'):
    if not item.get('comparison'):
        return None
    return load(item['comparison'])['summaries'][split]


def render(publication):
    variants = publication['variants']
    full = variants.get('FULL-v4', {})
    fallback = 'FULL-v3' if not (comparison(full) or {}).get('worlds') else 'FULL-v4'
    selected = variants.get(fallback, {})
    summary = comparison(selected) or {}
    comparisons = summary.get('comparisons', [])
    crps = [r for r in comparisons if r['metric'] == 'delay_crps_difference']
    rate = next((r for r in comparisons if r['metric'] == 'upstream_rate_distance_difference'
                 and r['horizon_s'] == 300), None)
    endpoint = load(selected['endpoints'])['summaries']['reserved8'] if selected.get('endpoints') else None
    distribution = Path(selected['distribution']) if selected.get('distribution') else None
    speed = [r for r in rows(distribution/'distribution_summary.csv' if distribution else None)
             if r['split'] == 'reserved8' and r['level'] == 'arm' and r['arm'] == 'native'
             and r['horizon_s'] == '600' and r['metric'] == 'speed_upstream']
    timing = [r for r in rows(distribution/'timing_worlds.csv' if distribution else None)
              if r['split'] == 'reserved8' and r['arm'] == 'native'
              and r['horizon_s'] == '600' and r['region'] == 'upstream']
    shifts = [r for r in rows(distribution/'expected_shifts_worlds.csv' if distribution else None)
              if r['split'] == 'reserved8' and r['reference'] == 'native'
              and r['horizon_s'] == '600' and r['quantity'] == 'delay']
    label = publication['label']
    lines = [f"# WM-PHYS morning report — {label}, {publication['local_time']}",
        'Physics candidates improve several point scores; adoption and calibrated action confidence remain unestablished.',
        f"Latest engineering source is FULL-v4/22cc32f6; current paired diagnostic below is {fallback}.", '',
        '## Model and engineering',
        '- SUMO IDM +LC2013 highway subset +merge/insertion; all learned corrections OFF; no neural training.',
        '- FULL-v4 uses final LANE advice and move→LC→emit, coupled entrant settings, no birth-tick motion/control.',
        '- FULL posterior: frozen FIT shrinkage_v1 joint support, strength19.35568 already applied, causal own-start motion;',
        '  unknown drivers/entrants use learned population; Gamma-Poisson arrivals use causal counters. Fitted LC OFF.',
        '- A encoder+hidden settings; B population default; C motion-fitted population point; FULL fitted posterior.',
        '- Touched modules287CPU PASS/4explicit skips; later endpoint/bootstrap21PASS/1skip and credibility18PASS.',
        '- A/B/C/FULL each passed60s on FIT969420/969421 (8rollouts): finite, conserved, no learned corrections.',
        '- FULL-v4 five-arm600s canary PASS370.48s/reused once; live sources/jobs retained, own-PID11:30 deadline.',
        '', '## Distributional evidence (per-variant available cases; not a cross-variant ranking)',
        '- WM2 baseline full20worlds/fivearms/K8; all candidates share registered969420..969439, H120/300/600.',
        '- Development12/reserved8 remain separate; incomplete ensembles never count as K8. Contrasts below reserved only.']
    for key in ('A2', 'B2', 'C2', 'FULL-v3', 'FULL-v4'):
        item = variants.get(key, {})
        value = comparison(item)
        if value and value.get('comparisons'):
            d = [r for r in value['comparisons'] if r['metric'] == 'delay_crps_difference']
            r = next(r for r in value['comparisons'] if r['metric'] == 'upstream_rate_distance_difference' and r['horizon_s'] == 300)
            lines.append(f"- {key}:n={len(value['worlds'])}, ΔCRPS120/300/600="+
                '/'.join(number(x['mean']) for x in d)+f"veh-s; Δ300s rate-distance={number(100*r['mean'],2)}pp.")
        else:
            lines.append(f"- {key}:PENDING complete K8 evidence; {item.get('tasks', 0)} saved tasks; no efficacy inferred.")
    lines += [f"- {fallback} native600s upstream speed CRPS={number(avg(speed,'wm_crps'),2)} vs cell={number(avg(speed,'cell_crps'),2)}m/s; clearing not qualified.",
        f"- Same native onset-distribution IBS600={number(avg(timing,'wm_ibs'),3)} vs cell={number(avg(timing,'cell_ibs'),3)}; onset-CDF gate retained."]
    for arm in ('mappo', 'pi'):
        data = [r for r in shifts if r['arm'] == arm]
        lines.append(f"- Expected600s {arm}−native delay={number(avg(data,'wm_effect'))} vs recorded={number(avg(data,'real_effect'))}veh-s (same available worlds).")
    lines += ['- Integrated ramp-delay target remains unavailable; separate nativeFIT32 H300 diagnostic2188 vs2273real is not this panel.',
        '- Final conditional LC FIT32:default precision3.006%/recall37.942%; encoder2.400%/40.492%; keep-right false positives dominate.',
        '- SUMOCORE DIFFERENT8world/3arm/K3:inferred CRPS647/4635/16728 vs defaults1087/5452/19343;',
        '  cell1433/4375/9131.300s rates inferred84.7/default88.9/real62.5%;18/144 startup collisions, validity FAIL.',
        '  Inferred600s MAPPO/PI effects+3450/+825 vs real−2064/−3858veh-s; not a clean information ceiling.', '',
        '## Switching-eligibility rule',
        f"- {fallback}−WM2 delay CRPS120/300/600="+'/'.join(ci(x) for x in crps)+'veh-s.' if crps else '- Delay CRPS comparison PENDING;95%CIs NA.',
        f"- 300s breakdown-distance difference={ci({**rate, 'mean':100*rate['mean'], 'low':None if rate['low'] is None else 100*rate['low'], 'high':None if rate['high'] is None else 100*rate['high']},2)}pp." if rate else '- Breakdown-distance comparison PENDING;95%CI NA.',
        f"- HDV60 mean-world-median error difference={ci(endpoint['primary'],2)}m; common HDV/arm endpoints={endpoint['counts'].get('common_hdv',0)}." if endpoint else '- HDV60 non-regression PENDING;95%CI NA (old A/B/C tasks lack endpoints).',
        '- Registered CIs resample only scored worlds within demand×penetration×split; singleton strata yield NA, never PASS.',
        '- Earlier04:00/first0545-labelled numeric CIs RETRACTED (unscored-peer bootstrap bug); point estimates unchanged.',
        '- ALL three conditions are required; no automatic S-C1SWITCH adoption. Sparse cells also prevent distributional qualification.', '',
        '## Credibility / schedule',
        '- Strict paired P(beats incumbent)>=0.8 plus lower expected delay is only a decision rule; calibration must also hold.']
    credibility = Path(selected['credibility']) if selected.get('credibility') else None
    # Raw report/figures retain all selected probabilities, outcomes and coverage;
    # avoid relabeling a forecast win probability as its own empirical validation.
    lines += [f"- Same-source selected-switch reliability and50/80/90%/range coverage: {selected.get('credibility','PENDING')}/CREDIBILITY.md.",
        '- Prior FULL-v3 matched4:0.8calls won6/11,6/10,9/11 vs94.3/92.5/97.7%forecast; range coverage20/40/25%.',
        '- ~04:00 interim retained:FULL-v1 n4 ΔCRPS−785/−3512/−11041; rate-distance+1.25pp; historical CIs retracted.',
        f"- ~08:00 interim: {publication.get('interim08','PENDING')}; final engineering freeze09:30: {publication.get('freeze','PENDING')}.",
        '- Final evaluation receipts through11:30, final report11:45; frozen source is not selected by reserved outcome tuning.',
        '- Remaining gaps:serial LC negotiation/internal junction state/cold memory; finish matched panels and validate action confidence.',
        f"- Audit/source manifests, per-world attrition, timing/speed/shift/gate/credibility files: {publication['receipt_directory']}."]
    if len(lines) > 60:
        raise AssertionError(f'report exceeds owner line limit: {len(lines)}')
    return '\n'.join(lines)+'\n'
