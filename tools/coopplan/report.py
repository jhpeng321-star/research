"""Collect explicit public receipts, paired world bootstrap, and frozen behaviour evidence."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tools.coopbandit.ops import remote
from .ops import HOSTS, LOCAL
from .protocol import worlds, digest, budget

OUT = LOCAL/'docs/paper_results/coopplan'


def fetch(host):
    root = HOSTS[host]['root']
    return json.loads(remote(host,f'''import pathlib,sys,json,gzip,hashlib
root=pathlib.Path({root!r});sys.path.insert(0,str(root/'pytools'))
import numpy as np
from tools.coopbehav.io import Tape,Vehicle
from tools.coopbehav.detectors import detect
from tools.coopplan.storage import read,verify
binding=verify(root)
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
 return Tape(np.array(times),rows,dict(seed=seed),coverage=dict(trace_scope='full_observable_fleet'))
records=[];hashes={{}};complete=(root/'COMPLETE.json').exists();stopped=(root/'STOP.json').exists()
for world in read(root/'ALLOCATION.json')['worlds']:
 seed=world['seed'];base=root/'jobs'/f's{{seed}}_mappo'/'RESULT.json';live=root/'jobs'/f's{{seed}}_planner'/'RESULT.json'
 if not base.exists() or not live.exists():continue
 b,p=read(base),read(live)
 for path in (base,live):hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
 evidence=[]
 for credit in p['credits']:
  key=f's{{seed}}_t{{credit["time_s"]}}';action=credit['action']['id']
  actual=tape(root/'jobs'/(key+'_'+action)/'observable.jsonl.gz',seed)
  reference=tape(root/'jobs'/(key+'_O0')/'observable.jsonl.gz',seed)
  events=detect(actual,reference,credit['members'],reference_kind='policy_mean_intervention')
  evidence.append(dict(time_s=credit['time_s'],events=events))
 observational=detect(tape(live.parent/'observable.jsonl.gz',seed))
 records.append(dict(world=world,baseline=b['clocks'],planner=p['clocks'],history=p['history'],
                     credits=p['credits'],causal_behaviours=evidence,observational_behaviours=observational))
charges=[json.loads(s) for s in (root/'CHARGES.jsonl').read_text().splitlines()] if (root/'CHARGES.jsonl').exists() else []
closed=[];branches=[];overrides=0;requests=0;incomplete=[]
for p in (root/'jobs').iterdir():
 if not (p/'CLOSE.json').exists():incomplete.append(p.name);continue
 close=read(p/'CLOSE.json');closed.append(close)
 if (p/'RESULT.json').exists():
  result=read(p/'RESULT.json')
  if 'prefix_pass' in result:branches.append(dict(id=p.name,wall_s=result['wall_s'],actual_sim_s=result['actual_sim_s'],prefix_pass=result['prefix_pass']))
 if p.name.endswith('_planner') and (p/'observable.jsonl.gz').exists():
  with gzip.open(p/'observable.jsonl.gz','rt') as f:
   for line in f:
    actions=json.loads(line)['actions'];requests+=len(actions);overrides+=sum(a['clamped'] for a in actions)
print(json.dumps(dict(host={host!r},complete=complete,stopped=stopped,records=records,hashes=hashes,
 charged=sum(c['sim_s'] for c in charges),actual=sum(c['actual_sim_s'] for c in closed),
 closed_jobs=len(closed),incomplete=incomplete,branches=branches,requests=requests,overrides=overrides,
 design_sha256=binding['design_sha256'],source_sha256s=binding['sources'])))
''',timeout=240))


def ci(values):
    x = np.asarray(values,float)
    if not len(x):
        return None
    draws = np.random.default_rng(20261101).integers(0,len(x),(10000,len(x)))
    boot = x[draws].mean(1)
    return dict(n=len(x),mean=float(x.mean()),lower=float(np.quantile(boot,.025)),
                upper=float(np.quantile(boot,.975)))


def save(path,value):
    # Generated analytical artifacts are immutable snapshots, never overwritten.
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False)
        f.write('\n')


def build(snapshot):
    records = sorted((r for h in snapshot for r in h['records']),key=lambda r:r['world']['seed'])
    complete = all(h['complete'] and not h['stopped'] and not h['incomplete'] for h in snapshot)
    if not complete or [r['world'] for r in records] != worlds():
        raise ValueError('full fixed paired panel required; no success report from an incomplete panel')
    assert all(b['prefix_pass'] for h in snapshot for b in h['branches'])
    charged,actual = sum(h['charged'] for h in snapshot),sum(h['actual'] for h in snapshot)
    assert actual == charged and charged <= sum(budget().values())
    rows,credits = [],[]
    for r in records:
        row = dict(seed=r['world']['seed'],penetration=r['world']['penetration'],demand=r['world']['demand'])
        for clock in ('600','900'):
            b,p = r['baseline'][clock],r['planner'][clock]
            row.update({f'MAPPO_J{clock}':b['team_delay_vs'],f'planner_J{clock}':p['team_delay_vs'],
                        f'saving_J{clock}':b['team_delay_vs']-p['team_delay_vs']})
        rows.append(row)
        for credit in r['credits']:
            credits.append(dict(seed=r['world']['seed'],**credit))
        assert len(r['history']) == 12
        for e in r['history']:
            a = e['executed_action']['id']
            if a != 'O0':
                assert len(e['group']) == 2
                assert e['decision']['probability_beats'][a] == 1.
                assert e['decision']['expected_delay'][a] < e['decision']['expected_delay']['O0']
    savings = {clock:ci([r[f'saving_J{clock}'] for r in rows]) for clock in ('600','900')}
    action_counts = {}
    for r in records:
        for e in r['history']:
            a = e['executed_action']['id'];action_counts[a]=action_counts.get(a,0)+1
    behaviour_counts,causal = {},{}
    for r in records:
        for event in r['observational_behaviours']:
            b=event['behaviour'];behaviour_counts[b]=behaviour_counts.get(b,0)+1
        for group in r['causal_behaviours']:
            for event in group['events']:
                key=event['behaviour']+'/'+event['evidence'];causal[key]=causal.get(key,0)+1
    safety = {arm:{k:sum(r[arm]['900'][k] for r in records) for k in
                       ('collisions','teleports','ttc_below_3_vs','hard_brake')} for arm in ('baseline','planner')}
    safety['planner'].update(requests=sum(h['requests'] for h in snapshot),overrides=sum(h['overrides'] for h in snapshot))
    summary = dict(complete=True,worlds=12,savings=savings,credit=ci([r['C_G'] for r in credits]),
                   credit_positive=sum(r['C_G']>0 for r in credits),credit_sampled=len(credits),
                   executed_actions=action_counts,observational_behaviours=behaviour_counts,
                   release_supported_behaviours=causal,safety=safety,charged=charged,actual=actual,
                   budget_upper=704760,cap=800000,bootstrap=dict(seed=20261101,resamples=10000,unit='whole_world'),
                   exact_prefix_passes=sum(len(h['branches']) for h in snapshot))
    return records,rows,credits,summary


def figures(rows,credits,summary,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,clock in zip(axes,('600','900')):
        x=[r[f'saving_J{clock}'] for r in rows]
        ax.bar(range(12),x,color='#286A83')
        ax.axhline(0,color='black',linewidth=.8)
        ax.set(title=f'Paired J{clock}: MAPPO minus planner',xlabel='Fixed development world (cell order)',ylabel='Team-delay saving (veh-s)')
        c=summary['savings'][clock]
        ax.text(.02,.98,f'Mean {c["mean"]:.1f}\n95% world CI [{c["lower"]:.1f}, {c["upper"]:.1f}]',transform=ax.transAxes,va='top')
    fig.suptitle('S-COOPPLAN / fresh 983100–983111 / warmup excluded')
    fig.savefig(dest/'paired_delay.png',dpi=180)
    plt.close(fig)
    if credits:
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        ax.bar([str(r['seed']) for r in credits],[r['C_G'] for r in credits],color='#286A83')
        ax.axhline(0,color='black',linewidth=.8)
        ax.set(title='First executed group per world: exact H120 cooperation credit',xlabel='Fresh world identity (sampled worlds only)',ylabel='C_G: superadditive team-delay saving (veh-s)')
        ax.tick_params(axis='x',rotation=45)
        fig.savefig(dest/'cooperation_credit.png',dpi=180)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--collect',action='store_true')
    a=p.parse_args()
    snapshot=[fetch(host) for host in HOSTS]
    if not all(h['complete'] for h in snapshot):
        print(json.dumps([dict(host=h['host'],complete=h['complete'],stopped=h['stopped'],
                               paired=len(h['records']),charged=h['charged'],actual=h['actual']) for h in snapshot]))
        return
    records,rows,credits,summary=build(snapshot)
    if a.collect:
        dest=OUT/'final'
        assert dest.parent.is_dir()
        dest.mkdir()
        save(dest/'RECEIPTS.json',snapshot)
        save(dest/'SUMMARY.json',summary)
        save(dest/'BEHAVIOURS.json',[dict(seed=r['world']['seed'],causal=r['causal_behaviours'],
                                         observational=r['observational_behaviours']) for r in records])
        with (dest/'paired_worlds.csv').open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        save(dest/'CREDITS.json',credits)
        figures(rows,credits,summary,dest)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
