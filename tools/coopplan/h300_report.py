"""Separate H300 confirmation/wall analysis; immutable physics receipts only."""
import argparse
from copy import deepcopy
import csv
import json
from unittest.mock import patch

import numpy as np

from tools.coopbandit.ops import remote
from . import report,extension_report
from .h300 import HOSTS,EVIDENCE,allocation,WORLD_CAP
from .protocol import h300_worlds,digest,sha

DEST=EVIDENCE/'final'


def fetch(host):
    with patch.object(report,'HOSTS',HOSTS):snap=report.fetch(host)
    if not snap['complete'] or snap['stopped']:
        return dict(host=host,complete=snap['complete'],stopped=snap['stopped'],paired=len(snap['records']),charged=snap['charged'])
    with patch.object(extension_report,'HOSTS',HOSTS):snap.update(extension_report.details(host))
    root=HOSTS[host]['root']
    snap.update(json.loads(remote(host,f'''import pathlib,sys,json,gzip,hashlib
root=pathlib.Path({root!r});sys.path.insert(0,str(root/'pytools'))
from tools.coopplan.storage import read
charges=[json.loads(s) for s in (root/'CHARGES.jsonl').read_text().splitlines()]
per_world={{}};results={{}};hashes={{}}
for w in read(root/'ALLOCATION.json')['worlds']:
 seed=w['seed'];requests=overrides=0
 with gzip.open(root/'jobs'/f's{{seed}}_planner'/'observable.jsonl.gz','rt') as f:
  for line in f:
   a=json.loads(line)['actions'];requests+=len(a);overrides+=sum(x['clamped'] for x in a)
 per_world[str(seed)]=dict(requests=requests,overrides=overrides)
for job in (root/'jobs').iterdir():
 p=job/'RESULT.json'
 if p.exists() and '_t' in job.name:
  results[job.name]=read(p);hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(dict(charges=charges,per_world_safety=per_world,branch_results=results,branch_result_hashes=hashes)))
''')))
    path=EVIDENCE/f'RECEIPTS_{host}.json'
    report.save(path,snap)
    return dict(host=host,complete=True,worlds=len(snap['records']),charged=snap['charged'],new_collection=True)


def wall_counts(records):
    result=dict(decisions=0,available=0,evaluated=0,budget_pruned=0,chosen=0,effects=[])
    for r in records:
        for e in r['history']:
            result['decisions']+=1
            available=any(a['id']=='O2' for a in e['available_candidates'])
            evaluated='O2' in e['decision']['expected_delay']
            chosen=e['executed_action']['id']=='O2'
            assert not evaluated or available
            result['available']+=available;result['evaluated']+=evaluated
            result['budget_pruned']+=available and not evaluated
            result['chosen']+=chosen
            if evaluated:
                result['effects'].append(dict(seed=r['world']['seed'],time_s=e['time_s'],chosen=chosen,
                    D_O2=e['decision']['expected_delay']['O0']-e['decision']['expected_delay']['O2']))
    return result


def describe_wall(counts):
    effects=counts['effects']
    return dict(counts,evaluated_helpful=sum(e['D_O2']>0 for e in effects),
                helpful_but_not_chosen=sum(e['D_O2']>0 and not e['chosen'] for e in effects),
                evaluated_D_O2=dict(mean=float(np.mean([e['D_O2'] for e in effects])),
                    minimum=min(e['D_O2'] for e in effects),maximum=max(e['D_O2'] for e in effects)) if effects else None)


def prior_wall():
    original=extension_report.first_snapshot()
    extension=json.loads((EVIDENCE.parent/'extension/final/RECEIPTS.json').read_text())
    records=deepcopy([r for s in original+extension for r in s['records']])
    assert len(records)==36
    for r in records:
        for e in r['history']:
            assert e.get('admission',{}).get('reason')!='budget_pruned'
            # Original H120 had zero quota pruning; candidates are the availability roster.
            e['available_candidates']=e['candidates']
    return describe_wall(wall_counts(records))


def build(snapshot):
    expected=json.loads((EVIDENCE/'MANIFEST.json').read_text())
    frozen_free=json.loads((EVIDENCE/'FREE_FREEZE.json').read_text())
    assert len(snapshot)==3 and {s['host'] for s in snapshot}==set(HOSTS)
    assert len({digest(s['source_sha256s']) for s in snapshot})==1
    for s in snapshot:
        assert s['complete'] and not s['stopped'] and not s['incomplete']
        assert s['design_sha256']==expected['design_sha256'] and s['free']==frozen_free
        assert s['allocation']==allocation(s['host'])
        assert s['source_sha256s']==json.loads((EVIDENCE/f'PREPARE_{s["host"]}.json').read_text())['source_sha256s']
        assert s['actual']==s['charged']==sum(c['sim_s'] for c in s['charges'])<=s['allocation']['cap']
        assert not any(s['outstanding'].values()) and not s['backfill']
        assert all(b['prefix_pass'] for b in s['branches'])
        for w in s['allocation']['worlds']:
            prefix=f's{w["seed"]}_'
            assert sum(c['sim_s'] for c in s['charges'] if c['id'].startswith(prefix))<=WORLD_CAP
        for r in s['records']:
            seed=r['world']['seed']
            for e in r['history']:
                assert e['horizon_s']==300
                key=f's{seed}_t{e["time_s"]}'
                predicted=e['decision']['expected_delay']
                for name,value in predicted.items():
                    b=s['branch_results'][key+'_'+name]
                    assert b['prefix_pass'] and b['time_s']==e['time_s']
                    assert b['actual_sim_s']==e['time_s']+300
                    assert b['outcome']['team_delay']==value
            for c in r['credits']:
                assert c['horizon_s']==300
                key=f's{seed}_t{c["time_s"]}'
                j0=s['branch_results'][key+'_O0']['outcome']['team_delay']
                for i in range(2):
                    b=s['branch_results'][key+'_'+c['action']['id']+f'_solo{i}']
                    assert b['prefix_pass'] and b['actual_sim_s']==c['time_s']+300
                    assert c['member_alone_D'][i]==j0-b['outcome']['team_delay']
    records=sorted((r for s in snapshot for r in s['records']),key=lambda r:r['world']['seed'])
    assert [r['world'] for r in records]==h300_worlds()
    cohorts={name:[r for r in records if r['world']['cohort']==name] for name in ('confirmation18','wall6')}
    datasets={};summaries={}
    safety={int(k):v for s in snapshot for k,v in s['per_world_safety'].items()}
    for name,rs in cohorts.items():
        rows,credits,summary=extension_report.analyze(rs)
        summary['wall']=describe_wall(wall_counts(rs))
        summary['wall']['executed_credits']=[c for c in credits if c['action']['id']=='O2']
        summary['safety']['planner'].update({key:sum(safety[r['world']['seed']][key] for r in rs) for key in ('requests','overrides')})
        datasets[name]=(rows,credits);summaries[name]=summary
    charged=sum(s['charged'] for s in snapshot)
    branches=[b for s in snapshot for b in s['branches']]
    solos=sum(b['actual_sim_s'] for b in branches if '_solo' in b['id'])
    wall=np.array([b['wall_s'] for b in branches])
    summary=dict(complete=True,horizon_s=300,cohorts=summaries,prior_charged=565140,
        continuation_charged=charged,total_charged=565140+charged,total_actual=565140+charged,
        remaining=800000-565140-charged,approved_cap=800000,registered_upper=799860,
        continuation_costs=dict(paired_episodes=57600,all_group_twins=solos,lookahead=sum(b['actual_sim_s'] for b in branches)-solos),
        host_accounting={s['host']:dict(actual=s['actual'],charged=s['charged'],cap=s['allocation']['cap'],closed_jobs=s['closed_jobs']) for s in snapshot},
        prefix_passes=len(branches),free=frozen_free,h120_wall=prior_wall(),design_sha256=expected['design_sha256'],panel_sha256=digest(h300_worlds()),
        branch_cost=dict(median_wall_s=float(np.median(wall)),max_wall_s=float(wall.max()),scope='post_reset_excludes_warmup_initialization'))
    assert sum(summary['continuation_costs'].values())==charged<=234720
    return cohorts,datasets,summary


def figures(datasets,summary,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    prior=json.loads((EVIDENCE.parent/'extension/final/SUMMARY.json').read_text())['cohorts']['pooled36']
    labels=['H120 pooled36','H300 confirmation18','H300 wall6']
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,clock in zip(axes,('600','900')):
        for y,s in enumerate([prior,summary['cohorts']['confirmation18'],summary['cohorts']['wall6']]):
            c=s['savings'][clock]
            ax.errorbar(c['mean'],y,xerr=[[c['mean']-c['lower']],[c['upper']-c['mean']]],fmt='o',capsize=4,color='#286A83')
        ax.axvline(0,color='black',linewidth=.8);ax.set_yticks(range(3),labels);ax.invert_yaxis()
        ax.set(title=f'Paired J{clock}: MAPPO minus planner',xlabel='Mean team-delay saving (veh-s/world)',ylabel='Separate fresh development cohort')
    fig.suptitle('Exact-model planner / 95% world CIs / cross-cohort, not paired horizon effect')
    fig.savefig(dest/'horizons.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,name in zip(axes,('confirmation18','wall6')):
        c=summary['cohorts'][name]['wall'];keys=['available','evaluated','budget_pruned','chosen']
        bars=ax.bar(range(4),[c[k] for k in keys],color='#286A83')
        ax.bar_label(bars);ax.set_xticks(range(4),['Available','Evaluated','Budget-skipped','Chosen'])
        ax.set(title=f'O2 cross-lane wall / {name}',ylabel=f'Decisions (of {c["decisions"]})',xlabel='Availability before quota / subsequent planner disposition')
    fig.suptitle('H300 wall opportunity audit / all registered decisions, including fallback')
    fig.savefig(dest/'wall_opportunities.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,name in zip(axes,('confirmation18','wall6')):
        credits=datasets[name][1]
        if credits:ax.scatter([c['D_G'] for c in credits],[c['C_G'] for c in credits],color='#286A83')
        ax.axhline(0,color='black',linewidth=.8)
        ax.set(title=f'All executed groups / {name}',xlabel='Exact H300 joint saving D_G (veh-s)',ylabel='Exact H300 superadditive saving C_G (veh-s)')
    fig.savefig(dest/'h300_cooperation.png',dpi=180);plt.close(fig)


def collect():
    snapshot=[json.loads((EVIDENCE/f'RECEIPTS_{h}.json').read_text()) for h in HOSTS]
    cohorts,datasets,summary=build(snapshot)
    DEST.mkdir()
    report.save(DEST/'RECEIPTS.json',snapshot);report.save(DEST/'SUMMARY.json',summary)
    report.save(DEST/'ALL_GROUP_CREDITS.json',[c for rows,credits in datasets.values() for c in credits])
    report.save(DEST/'BEHAVIOURS.json',[dict(seed=r['world']['seed'],causal=r['causal_behaviours'],observational=r['observational_behaviours']) for rs in cohorts.values() for r in rs])
    for name,(rows,_) in datasets.items():
        with (DEST/(name+'.csv')).open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    figures(datasets,summary,DEST)
    return summary


def verify():
    snapshot=json.loads((DEST/'RECEIPTS.json').read_text())
    cohorts,datasets,summary=build(snapshot)
    assert summary==json.loads((DEST/'SUMMARY.json').read_text())
    assert [c for rows,credits in datasets.values() for c in credits]==json.loads((DEST/'ALL_GROUP_CREDITS.json').read_text())
    for name,(rows,_) in datasets.items():
        with (DEST/(name+'.csv')).open(newline='',encoding='utf-8') as f:written=list(csv.DictReader(f))
        assert len(written)==len(rows)
        for a,b in zip(written,rows):assert set(a)==set(b) and all(float(a[k])==v for k,v in b.items())
    result=dict(pass_=True,worlds=24,decisions=288,horizon_s=300,all_group_twins=True,
                availability_before_pruning=True,whole_world_bootstrap=True,accounting=summary['total_charged'],
                artifacts={p.name:sha(p) for p in sorted(DEST.iterdir()) if p.is_file() and p.name!='QA.json'})
    if (DEST/'QA.json').exists():assert result==json.loads((DEST/'QA.json').read_text())
    else:report.save(DEST/'QA.json',result)
    return result


def examples():
    credits=json.loads((DEST/'ALL_GROUP_CREDITS.json').read_text())
    selected=dict(largest_positive=max(credits,key=lambda c:c['C_G']),
                  most_negative=min(credits,key=lambda c:c['C_G']))
    selected.update({f'wall_{c["seed"]}':c for c in credits if c['action']['id']=='O2'})
    return {k:dict(seed=c['seed'],time_s=c['time_s'],action=c['action']['id'],
                   D_G=c['D_G'],C_G=c['C_G'],member_alone_D=c['member_alone_D']) for k,c in selected.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('fetch','collect','verify','examples'))
    p.add_argument('--host',choices=tuple(HOSTS),default='cpuQ');a=p.parse_args()
    print(json.dumps(fetch(a.host) if a.mode=='fetch' else globals()[a.mode]()))


if __name__=='__main__':main()
