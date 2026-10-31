"""Immutable 24-world and pooled36 reporting; all executed-group twins and world clusters."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.coopbandit.ops import remote
from . import report
from .extension import HOSTS,EVIDENCE,first_snapshot,missing,allocation
from .protocol import worlds,extension_worlds,sha,digest

DEST = EVIDENCE/'final'


def details(host):
    root=HOSTS[host]['root']
    return json.loads(remote(host,f'''import pathlib,sys,json,gzip,hashlib,os
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
root=pathlib.Path({root!r});sys.path.insert(0,str(root/'pytools'))
import numpy as np
from tools.coopplan.storage import read,verify
from tools.coopplan.quota import outstanding
from tools.coopbehav.io import Tape,Vehicle
from tools.coopbehav.detectors import detect
b=verify(root)
def tape(path,seed):
 rows=[];times=[]
 with gzip.open(path,'rt') as f:
  for line in f:
   r=json.loads(line);times.append(r['time_s']);row={{}}
   for v in r['vehicles']:
    control=v.get('controller');issued=v.get('issued_acceleration_mps2')
    row[v['vid']]=Vehicle(v['s_m'],v['v_mps'],v['length_m'],v['lane'],v['is_cav'],issued,
       'native' if control=='native_sumo' else control,issued if control=='policy_mean' else None)
   rows.append(row)
 return Tape(np.array(times),rows,dict(seed=seed),hashes=dict(trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),coverage=dict(trace_scope='full_observable_fleet'))
credits=read(root/'BACKFILL.json') if (root/'BACKFILL.json').exists() else []
evidence=[];hashes={{}}
for c in credits:
 key=f's{{c["seed"]}}_t{{c["time_s"]}}';name=c['action']['id']
 joint=tape(root/'original'/(key+'_'+name)/'observable.jsonl.gz',c['seed'])
 base=tape(root/'original'/(key+'_O0')/'observable.jsonl.gz',c['seed'])
 detected=detect(joint,base,c['members'],reference_kind='policy_mean_intervention')
 evidence.append(dict(seed=c['seed'],time_s=c['time_s'],events=detected['events'],coverage=detected['coverage'],
                      actual_hashes=joint.hashes,reference_hashes=base.hashes))
 j0=read(root/'original'/(key+'_O0')/'RESULT.json')['outcome']['team_delay']
 jg=read(root/'original'/(key+'_'+name)/'RESULT.json')['outcome']['team_delay']
 singles=[]
 for i in range(2):
  p=root/'jobs'/(key+'_'+name+f'_solo{{i}}')/'RESULT.json';r=read(p)
  assert r['prefix_pass'];singles.append(j0-r['outcome']['team_delay'])
  hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
 assert c['D_G']==j0-jg and c['member_alone_D']==singles and c['C_G']==j0-jg-sum(singles)
pending=outstanding(root)
print(json.dumps(dict(host={host!r},backfill=credits,backfill_behaviours=evidence,hashes=hashes,
 outstanding=pending,allocation=read(root/'ALLOCATION.json'),free=b['free_provenance'])))
''',timeout=120))


def fetch(host):
    with patch.object(report,'HOSTS',HOSTS):
        snap=report.fetch(host)
    snap.update(details(host))
    return snap


def credit_ci(records,resamples=10000,seed=20261101):
    """Event-weighted mean with WHOLE-world multiplicities, including zero-action worlds."""
    sums=np.array([sum(c['C_G'] for c in r['credits']) for r in records],float)
    counts=np.array([len(r['credits']) for r in records],int)
    if not counts.sum():return None
    draws=np.random.default_rng(seed).integers(0,len(records),(resamples,len(records)))
    denominators=counts[draws].sum(1)
    good=denominators>0
    boot=sums[draws].sum(1)[good]/denominators[good]
    return dict(mean=float(sums.sum()/counts.sum()),lower=float(np.quantile(boot,.025)),
                upper=float(np.quantile(boot,.975)),worlds=len(records),events=int(counts.sum()),
                positive=sum(c['C_G']>0 for r in records for c in r['credits']),
                active_worlds=int((counts>0).sum()),zero_event_bootstrap_draws=int((~good).sum()),
                unit='whole_world',estimand='event_weighted_mean_C_G')


def analyze(records):
    rows=[];credits=[];actions={};behaviours={};causal={};pruning=[]
    safety={arm:{k:0. for k in ('collisions','teleports','ttc_below_3_vs','hard_brake')}
            for arm in ('baseline','planner')}
    for r in records:
        w=r['world'];seed=w['seed']
        row=dict(seed=seed,cell=w['cell'],penetration=w['penetration'],demand=w['demand'])
        for clock in ('600','900'):
            b,p=r['baseline'][clock],r['planner'][clock]
            row.update({f'MAPPO_J{clock}':b['team_delay_vs'],f'planner_J{clock}':p['team_delay_vs'],
                        f'saving_J{clock}':b['team_delay_vs']-p['team_delay_vs']})
        rows.append(row)
        assert len(r['history'])==12
        executed={e['time_s']:e for e in r['history'] if e['group']}
        assert len({c['time_s'] for c in r['credits']})==len(r['credits'])
        assert set(executed)=={c['time_s'] for c in r['credits']},'ALL executed group credits required'
        for c in r['credits']:
            e=executed[c['time_s']]
            assert c['action']==e['executed_action'] and c['members']==e['group']
            assert c['C_G']==c['D_G']-sum(c['member_alone_D']) and c['D_G']>0
            credits.append(dict(seed=seed,**c))
        for e in r['history']:
            a=e['executed_action']['id'];d=e['decision'];actions[a]=actions.get(a,0)+1
            selected=min(d['eligible'],key=lambda a:(d['expected_delay'][a],a)) if d['eligible'] else 'O0'
            assert a==d['selected']==selected
            if a!='O0':
                assert len(e['group'])==2 and d['probability_beats'][a]==1.
                assert d['expected_delay'][a]<d['expected_delay']['O0']
            adm=e.get('admission',{})
            if adm.get('reason')=='budget_pruned':pruning.append(dict(seed=seed,time_s=e['time_s'],**adm))
        for event in r['observational_behaviours']['events']:
            key=event['behaviour'];behaviours[key]=behaviours.get(key,0)+1
        assert {c['time_s'] for c in r['causal_behaviours']} == set(executed)
        for group in r['causal_behaviours']:
            for event in group['events']:
                key=event['behaviour']+'/'+event['evidence'];causal[key]=causal.get(key,0)+1
        for arm in safety:
            for k in safety[arm]:safety[arm][k]+=r[arm]['900'][k]
    summary=dict(worlds=len(records),decisions=12*len(records),
        savings={clock:report.ci([r[f'saving_J{clock}'] for r in rows]) for clock in ('600','900')},
        credit=credit_ci(records),executed_actions=actions,all_group_credits=len(credits),
        causal_behaviour_evidence=causal,observational_behaviours=behaviours,safety=safety,
        budget_pruned_decisions=len(pruning),budget_pruning=pruning,
        bootstrap=dict(unit='whole_world',resamples=10000,seed=20261101))
    return rows,credits,summary


def build(snapshot):
    assert len(snapshot)==3 and {s['host'] for s in snapshot}==set(HOSTS)
    assert all(s['complete'] and not s['stopped'] and not s['incomplete'] for s in snapshot)
    expected=json.loads((EVIDENCE/'MANIFEST.json').read_text())
    assert {s['design_sha256'] for s in snapshot}=={expected['design_sha256']}
    assert len({digest(s['source_sha256s']) for s in snapshot})==1
    assert len({digest(s['free']) for s in snapshot})==1
    assert snapshot[0]['free']==json.loads((EVIDENCE/'FREE_FREEZE.json').read_text())
    for s in snapshot:
        assert s['allocation']==allocation(s['host'])
        frozen=json.loads((EVIDENCE/f'PREPARE_{s["host"]}.json').read_text())
        assert s['source_sha256s']==frozen['source_sha256s']
        assert s['charged']==s['actual']<=s['allocation']['cap']
        assert all(v==0 for v in s['outstanding'].values())
        assert all(b['prefix_pass'] for b in s['branches'])
        assert len(s['backfill'])==len(missing(s['host']))
    original=first_snapshot()
    old,_,_,old_summary=report.build(original)
    old=deepcopy(old)
    byseed={r['world']['seed']:r for r in old}
    for s in snapshot:
        for c in s['backfill']:
            r=byseed[c['seed']];r['credits'].append({k:v for k,v in c.items() if k!='seed'})
        for e in s['backfill_behaviours']:
            byseed[e['seed']]['causal_behaviours'].append({k:v for k,v in e.items() if k!='seed'})
    new=sorted((r for s in snapshot for r in s['records']),key=lambda r:r['world']['seed'])
    assert [r['world'] for r in new]==extension_worlds()
    assert [r['world'] for r in old]==worlds()
    cohorts=dict(original12=old,extension24=new,pooled36=old+new)
    summaries={};datasets={}
    for name,records in cohorts.items():
        rows,credits,summary=analyze(records);summaries[name]=summary;datasets[name]=(rows,credits)
    for name,hosts in [('original12',original),('extension24',snapshot),('pooled36',original+snapshot)]:
        summaries[name]['safety']['planner'].update(requests=sum(s['requests'] for s in hosts),overrides=sum(s['overrides'] for s in hosts))
    total=old_summary['charged']+sum(s['charged'] for s in snapshot)
    assert total<=800000
    summary=dict(complete=True,cohorts=summaries,original_charged=old_summary['charged'],
                 extension_charged=sum(s['charged'] for s in snapshot),total_charged=total,total_actual=total,
                 approved_cap=800000,remaining=800000-total,design_sha256=expected['design_sha256'],
                 free=snapshot[0]['free'],backfilled_original_groups=13,
                 extension_prefix_passes=sum(len(s['branches']) for s in snapshot),
                 original_prefix_passes=old_summary['exact_prefix_passes'])
    return cohorts,datasets,summary


def figures(datasets,summary,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    labels=['Original 12','Extension 24','Pooled 36']
    for ax,clock in zip(axes,('600','900')):
        for y,(name,label) in enumerate(zip(('original12','extension24','pooled36'),labels)):
            c=summary['cohorts'][name]['savings'][clock]
            ax.errorbar(c['mean'],y,xerr=[[c['mean']-c['lower']],[c['upper']-c['mean']]],fmt='o',color='#286A83',capsize=4)
        ax.axvline(0,color='black',linewidth=.8);ax.set_yticks(range(3),labels);ax.invert_yaxis()
        ax.set(title=f'Paired J{clock} saving: MAPPO minus planner',xlabel='Mean team-delay saving (veh-s/world)',ylabel='Fixed development cohort')
    fig.suptitle('Exact-model cooperative planning / 95% whole-world bootstrap CIs')
    fig.savefig(dest/'paired_cohorts.png',dpi=180);plt.close(fig)
    rows,credits=datasets['pooled36']
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,clock in zip(axes,('600','900')):
        ax.bar(range(36),[r[f'saving_J{clock}'] for r in rows],color=['#8B969A']*12+['#286A83']*24)
        ax.axhline(0,color='black',linewidth=.8);ax.axvline(11.5,color='black',linestyle=':',linewidth=.8)
        ax.set(title=f'Paired J{clock} by world (original12 | extension24)',xlabel='Development world index (fixed cell/repeat order)',ylabel='Team-delay saving (veh-s)')
    fig.savefig(dest/'paired_worlds.png',dpi=180);plt.close(fig)
    if credits:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        axes[0].scatter([c['D_G'] for c in credits],[c['C_G'] for c in credits],color='#286A83',alpha=.8)
        axes[0].axhline(0,color='black',linewidth=.8)
        axes[0].set(title='Every executed group: joint saving vs superadditivity',xlabel='D_G: exact H120 joint saving (veh-s)',ylabel='C_G: exact H120 superadditive saving (veh-s)')
        for y,name in enumerate(('original12','extension24','pooled36')):
            c=summary['cohorts'][name]['credit']
            if c:axes[1].errorbar(c['mean'],y,xerr=[[c['mean']-c['lower']],[c['upper']-c['mean']]],fmt='o',color='#286A83',capsize=4)
        axes[1].axvline(0,color='black',linewidth=.8);axes[1].set_yticks(range(3),labels);axes[1].invert_yaxis()
        axes[1].set(title='All-group C_G: 95% whole-world cluster CIs',xlabel='Event-weighted mean superadditive saving (veh-s/action)',ylabel='Fixed development cohort')
        fig.savefig(dest/'all_group_cooperation.png',dpi=180);plt.close(fig)


def collect(snapshot):
    cohorts,datasets,summary=build(snapshot)
    assert DEST.parent.is_dir();DEST.mkdir()
    report.save(DEST/'RECEIPTS.json',snapshot)
    report.save(DEST/'SUMMARY.json',summary)
    report.save(DEST/'ALL_GROUP_CREDITS.json',datasets['pooled36'][1])
    report.save(DEST/'BEHAVIOURS.json',[dict(seed=r['world']['seed'],causal=r['causal_behaviours'],observational=r['observational_behaviours']) for r in cohorts['pooled36']])
    for name,(rows,credits) in datasets.items():
        with (DEST/(name+'.csv')).open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    figures(datasets,summary,DEST)
    return summary


def verify():
    snapshot=json.loads((DEST/'RECEIPTS.json').read_text())
    cohorts,datasets,summary=build(snapshot)
    assert summary==json.loads((DEST/'SUMMARY.json').read_text())
    assert datasets['pooled36'][1]==json.loads((DEST/'ALL_GROUP_CREDITS.json').read_text())
    for name,(rows,_) in datasets.items():
        with (DEST/(name+'.csv')).open(newline='',encoding='utf-8') as f:recorded=list(csv.DictReader(f))
        assert len(recorded)==len(rows)
        for a,b in zip(recorded,rows):
            assert set(a)==set(b) and all(float(a[k])==v for k,v in b.items())
    result=dict(pass_=True,worlds=36,decisions=432,all_group_credits=summary['cohorts']['pooled36']['all_group_credits'],
                accounting=summary['total_charged'],cap=800000,source_and_design_bound=True,
                all_executed_groups_have_two_solos=True,whole_world_cluster_bootstrap=True,
                artifacts={p.name:sha(p) for p in sorted(DEST.iterdir()) if p.is_file() and p.name!='QA.json'})
    report.save(DEST/'QA.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--collect',action='store_true');p.add_argument('--verify',action='store_true')
    a=p.parse_args()
    if a.verify:print(json.dumps(verify()));return
    snapshot=[fetch(h) for h in HOSTS]
    if not all(s['complete'] for s in snapshot):
        print(json.dumps([dict(host=s['host'],complete=s['complete'],stopped=s['stopped'],paired=len(s['records']),
                              backfilled=len(s['backfill']),charged=s['charged'],actual=s['actual']) for s in snapshot]));return
    summary=collect(snapshot) if a.collect else build(snapshot)[2]
    print(json.dumps(summary))


if __name__=='__main__':main()
