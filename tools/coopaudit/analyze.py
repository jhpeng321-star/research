"""COOPMEAS2 stratified world-bootstrap convention; exact panel only, raw credit."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from tools.coopaudit.design import CONTROLLERS, CAP, interaction, worlds
from tools.coopaudit.accounting import write

SAFETY = ('hard_braking_vs','ttc_below_3_vs','collisions','teleports','dropped_vehicle_steps')


def bootstrap(values, cells, n_boot=5000):
    """Reuse COOPMEAS2 allocation-weighted strata and missing-stratum refusal."""
    values, cells = np.asarray(values,dtype=float),np.asarray(cells)
    rng = np.random.default_rng(20261031)
    result = np.zeros((n_boot,values.shape[1]))
    mean = np.zeros(values.shape[1])
    complete = np.ones(values.shape[1],bool)
    for cell in sorted(set(cells)):
        block = values[cells==cell]
        idx = rng.integers(0,len(block),size=(n_boot,len(block)))
        weight = len(block)/len(values)
        count = np.isfinite(block).sum(0)
        mean += weight*np.nansum(block,0)/np.maximum(count,1)
        sampled = block[idx]
        n = np.isfinite(sampled).sum(1)
        result += weight*np.nansum(sampled,1)/np.maximum(n,1)
        complete &= (count>0)&np.all(n>0,axis=0)
    return [dict(mean=float(mean[c]) if complete[c] else None,
                 lo=float(np.quantile(result[:,c],.025)) if complete[c] else None,
                 hi=float(np.quantile(result[:,c],.975)) if complete[c] else None,
                 n_worlds=int(np.isfinite(values[:,c]).sum()),
                 reason=None if complete[c] else 'missing stratum in original/bootstrap sample')
            for c in range(values.shape[1])]


def contrasts(result, controller, outcome):
    group = result['frozen']['group']
    if len(group)<2:
        return dict(D_GROUP=None,sum_D_ONLY=None,C_G=None,frac_positive=None)
    def cost(key):
        if outcome=='local':
            return result['effects'][key]['local_cohort_delay_vs']
        return result['metrics'][key][outcome]['team_delay_vs']
    t = interaction(cost('native_B'),cost(controller+'_GROUP'),
                    [cost(controller+f'_ONLY:{i}') for i in range(len(group))])
    return {k:t[k] for k in ('D_GROUP','sum_D_ONLY','C_G')} | dict(frac_positive=float(t['C_G']>0))


def analyze(root, out):
    root,out = Path(root),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if not (root/'MAIN_COMPLETE.json').exists():
        raise ValueError('complete fixed panel required')
    ledger = [json.loads(s) for s in (root/'accounting.jsonl').read_text().splitlines()]
    if len({r['id'] for r in ledger})!=len(ledger) or sum(r['sim_s'] for r in ledger)>CAP:
        raise ValueError('budget integrity failure')
    results, rows, raw = [],[],[]
    for w in worlds():
        r = json.loads((root/'jobs'/f"main_s{w['seed']}"/'result.json').read_text())
        if not r['ok'] or r['frozen']['world']!=w:
            raise ValueError('world identity failure')
        if r['trajectory_sha256']['native_B']!=r['trajectory_sha256']['native_B_repeat']:
            raise ValueError('native repeat failure')
        results.append(r)
        for c in CONTROLLERS:
            for outcome in ('300','600','local'):
                rows.append(dict(seed=w['seed'],cell=w['cell'],controller=c,outcome=outcome,
                                 group_size=len(r['frozen']['group']),**contrasts(r,c,outcome)))
        for branch,clocks in r['metrics'].items():
            for clock,m in clocks.items():
                if abs(m['team_delay_vs']-m['road_delay_vs']-m['source_wait_vs'])>1e-6:
                    raise ValueError('cost conservation failure')
                raw.append(dict(seed=w['seed'],branch=branch,clock=clock,
                                **{k:m[k] for k in ('team_delay_vs','road_delay_vs','source_wait_vs')+SAFETY}))
    summary=[]
    for cohort in ('all','high_share',*range(8)):
        for outcome in ('300','600','local'):
            chosen = [w for w in worlds() if cohort=='all' or
                      (cohort=='high_share' and w['cell']<4) or w['cell']==cohort]
            if not chosen:
                continue
            lookup = {(r['seed'],r['controller']):r for r in rows if r['outcome']==outcome}
            keys = [(c,k) for c in CONTROLLERS for k in ('C_G','D_GROUP','sum_D_ONLY','frac_positive')]
            values = [[lookup[w['seed'],c][k] for c,k in keys] for w in chosen]
            for (c,k),ci in zip(keys,bootstrap(values,[w['cell'] for w in chosen])):
                if len(chosen)==1:
                    ci['lo']=ci['hi']=None
                summary.append(dict(cohort=cohort,outcome=outcome,controller=c,metric=k,**ci))
    for name,data in (('COOPAUDIT_worlds.csv',rows),('COOPAUDIT_branches.csv',raw),('COOPAUDIT.csv',summary)):
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    write(out/'SUMMARY.json',dict(worlds=len(results),charged_sim_s=sum(r['sim_s'] for r in ledger),
                                 summary=summary,actual_main_sim_s=sum(r['actual_sim_s'] for r in results)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('out')
    a=p.parse_args()
    analyze(a.root,a.out)
