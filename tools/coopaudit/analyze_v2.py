"""Exact paired v2 credits; world bootstrap, explicit missing coverage, no new SUMO."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from tools.coopaudit.accounting import write
from tools.coopaudit.analyze import bootstrap, contrasts, SAFETY
from tools.coopaudit.design import interaction, sha, worlds

ARMS=('mappo','wall','sumocf')


def dump_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze():
    evidence=Path(__file__).resolve().parent/'evidence'
    root=evidence/'v2_complete'
    out=evidence/'v2_analysis'
    out.mkdir(exist_ok=True)
    selection=json.loads((root/'CORRECTED_DESIGN.json').read_text())
    assert json.loads((root/'V2_MAIN_COMPLETE.json').read_text())['ok']
    results=[]
    rows=[]
    branches=[]
    trace_manifest=[]
    for w in worlds():
        directory=root/'jobs'/f"v2_s{w['seed']}"
        r=json.loads((directory/'result.json').read_text())
        assert r['ok'] and r['frozen']['world']==w and r['branch_mode']=='full_prefix_rerun'
        planned=next(p for p in selection['panel'] if p['world']==w)
        assert r['frozen']['group']==planned['group'] and r['frozen']['wall_roster']==planned['wall_roster']
        assert r['actual_sim_s']==planned['charge_sim_s']
        if planned['group']:
            assert r['metrics']['native_B']==r['metrics']['native_B_repeat']
            assert r['trajectory_sha256']['native_B']==r['trajectory_sha256']['native_B_repeat']
            prefixes=set()
            for p in directory.glob('branch_*.json'):
                b=json.loads(p.read_text())
                assert b['prefix_verified'] and b['actual_sim_s']==900
                prefixes.add(b['prefix_trajectory_sha256'])
            assert len(prefixes)==1 and len(list(directory.glob('branch_*.json')))==11
        results.append(r)
        for c in ARMS:
            for outcome in ('300','600','local'):
                t=contrasts(r,c,outcome)
                rows.append(dict(seed=w['seed'],cell=w['cell'],controller=c,outcome=outcome,
                                 group_size=len(planned['group']),group='|'.join(planned['group']),
                                 wall_roster='|'.join(planned['wall_roster']),**t))
        for key,clocks in r['metrics'].items():
            for clock,m in clocks.items():
                assert abs(m['team_delay_vs']-m['road_delay_vs']-m['source_wait_vs'])<1e-6
                branches.append(dict(seed=w['seed'],cell=w['cell'],branch=key,clock=clock,
                                     **{k:m[k] for k in ('team_delay_vs','road_delay_vs','source_wait_vs')+SAFETY}))
        for p in sorted((directory/'traces').glob('*.jsonl.gz')):
            count=0
            lo=1e20
            hi=-1e20
            with gzip.open(p,'rt') as f:
                for line in f:
                    frame=json.loads(line)
                    assert isinstance(frame['member'],bool) and 'requested_accel' in frame
                    lo=min(lo,frame['t'])
                    hi=max(hi,frame['t'])
                    count+=1
            assert count and 270<=lo<=271 and hi<=390
            trace_manifest.append(dict(seed=w['seed'],file=str(p.relative_to(root)),sha256=sha(p),
                                       rows=count,min_t=lo,max_t=hi))
    eligible=[w for w in worlds() if next(r for r in results if r['frozen']['world']==w)['frozen']['group']]
    lookup={(r['seed'],r['controller'],r['outcome']):r for r in rows}
    summary=[]
    for cohort in ('all28','high_share24','eligible21','eligible_high_share20',*range(8)):
        chosen=[w for w in worlds() if cohort=='all28' or (cohort=='high_share24' and w['cell']<4) or
                (cohort=='eligible21' and w in eligible) or
                (cohort=='eligible_high_share20' and w in eligible and w['cell']<4) or w['cell']==cohort]
        for outcome in ('300','600','local'):
            keys=[(c,k) for c in ARMS for k in ('C_G','D_GROUP','sum_D_ONLY','frac_positive')]
            matrix=[[lookup[w['seed'],c,outcome][k] for c,k in keys] for w in chosen]
            for (c,k),ci in zip(keys,bootstrap(matrix,[w['cell'] for w in chosen])):
                valid=[lookup[w['seed'],c,outcome][k] for w in chosen if lookup[w['seed'],c,outcome][k] is not None]
                if len(chosen)==1:
                    ci['lo']=ci['hi']=None
                summary.append(dict(cohort=cohort,outcome=outcome,controller=c,metric=k,
                                    available_case_mean=float(np.mean(valid)) if valid else None,
                                    allocated_worlds=len(chosen),**ci))
    paired=[]
    for outcome in ('300','600','local'):
        matrix=[[lookup[w['seed'],'sumocf',outcome][k]-lookup[w['seed'],'mappo',outcome][k]
                 for k in ('C_G','D_GROUP','sum_D_ONLY')] for w in eligible]
        for k,ci in zip(('C_G','D_GROUP','sum_D_ONLY'),bootstrap(matrix,[w['cell'] for w in eligible])):
            paired.append(dict(contrast='sumocf_minus_mappo',outcome=outcome,metric=k,**ci))
    ledger=[json.loads(s) for s in (root/'accounting.jsonl').read_text().splitlines()]
    assert len({r['id'] for r in ledger})==len(ledger)
    charged=sum(r['sim_s'] for r in ledger)
    actual=21180+14400+sum(r['actual_sim_s'] for r in results)
    assert charged==selection['total_admitted_sim_s']==243780 and actual==243480 and actual<=charged<=500000
    assert len(results)==28 and len(eligible)==21 and len(trace_manifest)==231
    safety={}
    for c in ('native',*ARMS):
        key='native_B' if c=='native' else c+'_GROUP'
        safety[c]={k:sum(r['metrics'][key]['600'][k] for r in results if r['frozen']['group']) for k in SAFETY}
    summary_doc=dict(status='COMPLETE',worlds=28,high_share_worlds=24,eligible_worlds=21,
                     eligible_high_share_worlds=20,missing_groups=7,wall_feasible_worlds=3,
                     main_actual_sim_s=sum(r['actual_sim_s'] for r in results),actual_total_sim_s=actual,
                     charged_sim_s=charged,retained_unused_reservation_sim_s=300,cap_sim_s=500000,
                     trace_files=len(trace_manifest),branch_prefix_gates=231,summary=summary,paired=paired,safety=safety,
                     strict_positive_counts={c:sum(lookup[w['seed'],c,'600']['C_G']>0 for w in eligible) for c in ARMS},
                     material_positive_counts={c:sum(lookup[w['seed'],c,'600']['C_G']>1e-6 for w in eligible) for c in ARMS},
                     bootstrap='5000 paired stratified world draws, seed20261031; eligible-cell weights for conditional CIs',
                     full_panel_CI='undefined when stratum has no eligible group or undefined resamples; never impute missing=0')
    for name,data in (('COOPAUDIT_worlds.csv',rows),('COOPAUDIT_branches.csv',branches),
                      ('COOPAUDIT.csv',summary),('COOPAUDIT_paired.csv',paired),('TRACE_MANIFEST.csv',trace_manifest)):
        dump_csv(out/name,data)
    write(out/'SUMMARY.json',summary_doc)
    print(json.dumps({k:v for k,v in summary_doc.items() if k not in ('summary',)},indent=1))
    return summary_doc


if __name__=='__main__':
    analyze()
