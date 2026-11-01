"""Render the owner-requested <=60-line report from source-bound score receipts."""
import csv
from collections import Counter
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


def replicated_world_cells(worlds, metadata):
    """All contributing strata need replication for a full world-level CI.

    Variation from a repeated stratum does not estimate missing within-stratum
    variance in the singleton strata. Preserve those conditional diagnostics in
    source reports, but never promote them to full-panel95% CIs here.
    """
    by_world = {int(r['seed']): (r['demand'], r['penetration'], r['split']) for r in metadata}
    counts = Counter(by_world[s] for s in worlds if s in by_world)
    return bool(worlds) and all(s in by_world for s in worlds) and all(n >= 2 for n in counts.values())


def world_ci(row, replicated):
    return row if replicated else dict(row, low=None, high=None)


def switching_point_status(summary, endpoint):
    """Owner's point-comparison rule, separate from confidence/qualification."""
    panel = set(range(969432, 969440))
    if (set(summary.get('worlds', [])) != panel or endpoint is None
            or set(endpoint.get('worlds', [])) != panel):
        return 'PENDING: complete reserved8 and HDV60 evidence required'
    comparisons = summary.get('comparisons', [])
    delay = {r['horizon_s']: r['mean'] for r in comparisons if r['metric'] == 'delay_crps_difference'}
    rate = next((r['mean'] for r in comparisons if r['metric'] == 'upstream_rate_distance_difference'
                 and r['horizon_s'] == 300), None)
    position = endpoint['primary'].get('mean')
    if set(delay) != {120, 300, 600} or rate is None or position is None:
        return 'PENDING: missing registered comparison'
    failures = []
    if not all(x < 0 for x in delay.values()): failures.append('delay CRPS not lower at every horizon')
    if rate >= 0: failures.append('300s breakdown rate not closer to real')
    if position > 0: failures.append('HDV60 position error worse')
    return 'FAIL: '+ '; '.join(failures) if failures else 'PASS on point comparisons; confidence and calibration are separate'


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
    metadata = rows(distribution/'distribution_worlds.csv') if distribution else []
    replicated = replicated_world_cells(summary.get('worlds', []), metadata)
    crps = [world_ci(r, replicated) for r in crps]
    rate = world_ci(rate, replicated) if rate else None
    if endpoint:
        endpoint = dict(endpoint, primary=world_ci(endpoint['primary'],
            replicated_world_cells(endpoint.get('worlds', []), metadata)))
    point_status = switching_point_status(summary, endpoint)
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
        'Registered switching rule '+point_status+'. Calibrated controller selection is not established.',
        f"Latest engineering source is FULL-v4/22cc32f6; current paired diagnostic below is {fallback}.", '',
        '## Model and engineering',
        '- SUMO IDM +LC2013 highway subset +merge/insertion; all learned corrections OFF; no neural training.',
        '- FULL-v4 uses final LANE advice and move→LC→emit, coupled entrant settings, no birth-tick motion/control.',
        '- FULL posterior: frozen FIT shrinkage_v1 joint support, strength19.35568 already applied, causal own-start motion;',
        '  unknown drivers/entrants use learned population; Gamma-Poisson arrivals use causal counters. Fitted LC OFF.',
        '- A encoder+hidden settings; B population default; C motion-fitted population point; FULL fitted posterior.',
        '- A2/B2/C2 use earlier4015b641; FULL-v4 includes later physical/native/source fixes: not a settings-only ablation.',
        '- Touched modules287CPU PASS/4explicit skips; later endpoint/bootstrap21PASS/1skip and credibility18PASS.',
        '- A/B/C/FULL each passed60s on FIT969420/969421 (8rollouts): finite, conserved, no learned corrections.',
        '- FULL-v4 five-arm600s canary PASS370.48s/reused once; live sources/jobs retained, own-PID11:30 deadline.',
        '- Superseded own FULL-v1/v3 retired after reserved8 completion; all outputs retained; A/B/C/FULL-v4 continued.',
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
    if distribution and crps:
        delay_rows = [r for r in rows(distribution/'distribution_worlds.csv')
                      if r['split'] == 'reserved8' and r['metric'] == 'team_delay_vs']
        candidate = [avg([r for r in delay_rows if r['horizon_s'] == str(h)], 'wm_crps')
                     for h in (120, 300, 600)]
        if all(x is not None for x in candidate):
            baseline = [value-row['mean'] for value, row in zip(candidate, crps)]
            lines.append('- Same-world absolute delay CRPS120/300/600:WM2 '+
                '/'.join(number(x) for x in baseline)+f"; {fallback} "+
                '/'.join(number(x) for x in candidate)+'veh-s (all five arms).')
            bias = avg([r for r in delay_rows if r['horizon_s'] == '600'], 'error')
            lines.append(f'- Same-world {fallback}600s mean delay bias={number(bias)}veh-s; severity alone does not establish clearing fidelity.')
    rate_rows = [r for r in comparisons if r['metric'] == 'upstream_rate_distance_difference']
    if ({r['horizon_s'] for r in rate_rows} == {120, 300, 600}
            and all(r.get(k) is not None for r in rate_rows
                    for k in ('real_rate', 'baseline_rate', 'candidate_rate'))):
        lines.append('- Same-world upstream rates120/300/600 (%):'+ '; '.join(
            label+' '+ '/'.join(number(100*r[key],1) for r in rate_rows)
            for label,key in (('real','real_rate'),('WM2','baseline_rate'),(fallback,'candidate_rate')))+'.')
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
        f"- {fallback}−WM2 delay CRPS differences (veh-s),120/300/600:"+' / '.join(ci(x) for x in crps)+'.' if crps else '- Delay CRPS comparison PENDING;95%CIs NA.',
        f"- 300s breakdown-distance difference (pp): {ci({**rate, 'mean':100*rate['mean'], 'low':None if rate['low'] is None else 100*rate['low'], 'high':None if rate['high'] is None else 100*rate['high']},2)}." if rate else '- Breakdown-distance comparison PENDING;95%CI NA.',
        f"- HDV60 mean-world-median error difference (m): {ci(endpoint['primary'],2)}; common HDV/arm endpoints={endpoint['counts'].get('common_hdv',0)}." if endpoint else '- HDV60 non-regression PENDING;95%CI NA (old A/B/C tasks lack endpoints).',
        '- Full-world95%CIs require replication in EVERY scored demand×penetration×split cell; any singleton gives NA here.',
        '- Detailed mixed-cell bootstrap intervals condition on singleton worlds; they are not full-panel confidence claims.',
        '- Earlier04:00/first0545-labelled numeric CIs RETRACTED (unscored-peer bootstrap bug); point estimates unchanged.',
        '- Point rule '+point_status+'; no automatic S-C1SWITCH adoption. All conditions required; sparse cells prevent qualification.', '',
        '## Credibility / schedule',
        '- Strict paired P(beats incumbent)>=0.8 plus lower expected delay is only a decision rule; calibration must also hold.']
    credibility = Path(selected['credibility']) if selected.get('credibility') else None
    data = [r for r in rows(credibility/'summary.csv' if credibility else None)
            if r['split'] == 'reserved8' and r['scope'] == 'pooled']
    def values(target, metric, *, range_only=False):
        selected_rows = [r for r in data if r['target'] == target and r['metric'] == metric
                         and (not range_only or r['interval'] != 'central')]
        return '/'.join(number(100*avg([r for r in selected_rows if r['horizon_s'] == str(h)], 'mean'),1)
                        if avg([r for r in selected_rows if r['horizon_s'] == str(h)], 'mean') is not None else 'NA'
                        for h in (120, 300, 600))
    # Raw report/figures retain all selected probabilities, outcomes and coverage;
    # avoid relabeling a forecast win probability as its own empirical validation.
    lines += [f"- Same-source selected-switch reliability and50/80/90%/range coverage: {selected.get('credibility','PENDING')}/CREDIBILITY.md.",
        f"- Same-source selected0.8calls observed win%={values('selected_preference','outcome')} vs forecast%={values('selected_preference','probability')}; range coverage%={values('delay','covered',range_only=True)}.",
        '- ~04:00 interim retained:FULL-v1 n4 ΔCRPS−785/−3512/−11041; rate-distance+1.25pp; historical CIs retracted.',
        '## ~08:00 interim',
        '- FULL-v4 reserved8: ΔCRPS120/300/600−393/−3040/−11647veh-s; HDV60−15.26m; breakdown-distance+5.63pp: point rule FAIL.',
        f"- Receipt: {publication.get('interim08','PENDING')}; primary95%CIs NA (singleton cells). Finalfreeze09:30: {publication.get('freeze','PENDING')}.",
        '- Final evaluation receipts through11:30, final report11:45; frozen source is not selected by reserved outcome tuning.',
        '- Remaining gaps:serial LC negotiation/internal junction state/cold memory; finish matched panels and validate action confidence.',
        f"- Audit/source manifests, per-world attrition, timing/speed/shift/gate/credibility files: {publication['receipt_directory']}."]
    if len(lines) > 60:
        raise AssertionError(f'report exceeds owner line limit: {len(lines)}')
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    assert out.parent.is_dir()
    out.write_text(render(load(args.publication)), encoding='utf-8')
