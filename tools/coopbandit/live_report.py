"""Paired fresh-dev analysis, exact event credit and optional frozen detectors."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .export import csvwrite
from .protocol import dev_worlds, sha
from .storage import read, write


def bootstrap(values):
    values=np.asarray(values,float)
    rng=np.random.default_rng(20261031)
    boot=rng.integers(0,len(values),(2000,len(values)))
    return dict(mean=float(values.mean()),ci95=np.quantile(values[boot].mean(axis=1),[.025,.975]).tolist())


def report(root, detector_root=None):
    root=Path(root)
    read(root/'LIVE_COMPLETE.json')
    dest=root/'report'
    dest.mkdir()
    rows,oracles,credits,sources,behaviour=[],[],[],[],[]
    if detector_root:
        import tools
        tools.__path__.append(str(Path(detector_root)/'tools'))
        from tools.coopbehav.report import load_trace
        from tools.coopbehav.detectors import detect,BEHAVIOURS
        detector_sources={n:sha(Path(detector_root)/'tools/coopbehav'/n)
                          for n in ('report.py','detectors.py','discovery.py','io.py')}
    for world in dev_worlds():
        results={}
        for arm in ('MAPPO','LEARNED'):
            path=root/'jobs'/f'dev_s{world["seed"]}_{arm}'
            results[arm]=read(path/'RESULT.json')
            sources.append(dict(path=str(path.relative_to(root)),result_sha256=sha(path/'RESULT.json'),
                                trace_sha256=sha(path/'observable.jsonl.gz')))
            if detector_root:
                tape=load_trace(path/'observable.jsonl.gz',dict(seed=world['seed'],arm=arm,
                    branch=dict(t0_s=300,horizon_s=900),trace_scope='full RSU/CAV-observable roster',
                    trace_limitations=['RSU sensing support only; 1 Hz; detector events descriptive, no paired release claim']))
                detected=detect(tape)
                for name in BEHAVIOURS:
                    behaviour.append(dict(world_id=world['seed'],arm=arm,behaviour=name,
                        events=sum(e['behaviour']==name for e in detected['events'])))
        base,learned=results['MAPPO'],results['LEARNED']
        for h in ('600','900'):
            row=dict(world_id=world['seed'],cell=world['cell'],horizon_s=int(h),
                     J_MAPPO=base['metrics'][h]['team_delay_vs'],J_LEARNED=learned['metrics'][h]['team_delay_vs'],
                     D=base['metrics'][h]['team_delay_vs']-learned['metrics'][h]['team_delay_vs'],
                     executed_options=sum(bool(d['plan']['members']) for d in learned['decisions']))
            for arm,result in results.items():
                for key in ('collisions','teleports','ttc_below_3_vs','hard_brake'):
                    row[f'{arm}_{key}']=result['metrics'][h][key]
                row[f'{arm}_episode_override_share']=result['safety']['override_share']
            rows.append(row)
        if not any(d['plan']['members'] for d in learned['decisions']):
            if base['clocks']!=learned['clocks']:
                raise ValueError('all-O0 paired episode differs bitwise')
        choices=[('O0',0.)]
        for oid in ('O1','O2','O3','O4'):
            path=root/'jobs'/f'dev_s{world["seed"]}_oracle_{oid}'/'RESULT.json'
            r=read(path)
            choices.append((oid,r['D']))
            oracles.append(dict(world_id=world['seed'],option=oid,available=r['plan']['available'],D600=r['D']))
            sources.append(dict(path=str(path.relative_to(root)),result_sha256=sha(path)))
        winner=max(choices,key=lambda item:item[1])
        oracles.append(dict(world_id=world['seed'],option='BEST_INCLUDING_O0',available=True,D600=winner[1],selected=winner[0]))
        reference=root/'jobs'/f'dev_s{world["seed"]}_credit_reference'/'RESULT.json'
        if reference.exists():
            ref=read(reference);event=ref['event'];t=event['time_s']
            joint=learned['clocks'][str(t+600)]['totals']['team_delay_vs']-learned['clocks'][str(t)]['totals']['team_delay_vs']
            jref=ref['metrics']['team_delay_vs']
            individual=[]
            for i in (0,1):
                p=root/'jobs'/f'dev_s{world["seed"]}_credit_solo{i}'/'RESULT.json'
                solo=read(p)
                individual.append(jref-solo['metrics']['team_delay_vs'])
                sources.append(dict(path=str(p.relative_to(root)),result_sha256=sha(p)))
            dg=jref-joint
            credits.append(dict(world_id=world['seed'],time_s=t,option=event['plan']['id'],
                D_G=dg,D_0=individual[0],D_1=individual[1],sum_D_i=sum(individual),C_G=dg-sum(individual)))
            sources.append(dict(path=str(reference.relative_to(root)),result_sha256=sha(reference)))
    charges=[json.loads(line) for line in (root/'CHARGES.jsonl').read_text().splitlines()]
    if len({r['id'] for r in charges})!=len(charges): raise ValueError('duplicate live charge')
    actual=0.
    for row in charges:
        p=root/'jobs'/row['id'];r,c=read(p/'RESULT.json'),read(p/'CLOSE.json')
        expected=r.get('event',{}).get('time_s',None)
        expected=expected+600 if '_credit_' in row['id'] else row['sim_s']
        if r['actual_sim_s']!=c['actual_sim_s'] or c['actual_sim_s']!=expected or c['charged_sim_s']!=row['sim_s']:
            raise ValueError('live charge/actual mismatch '+row['id'])
        actual+=c['actual_sim_s']
    summary=dict(worlds=12,paired={h:bootstrap([r['D'] for r in rows if r['horizon_s']==int(h)]) for h in ('600','900')},
        restricted_t300_oracle=bootstrap([r['D600'] for r in oracles if r['option']=='BEST_INCLUDING_O0']),
        executed_options=sum(r['executed_options'] for r in rows if r['horizon_s']==600),measured_event_credits=len(credits),
        charged_sim_s=sum(r['sim_s'] for r in charges),actual_sim_s=actual,
        oracle_scope='single baseline t300 decision, horizon600, NOT a closed-loop oracle',
        credit_scope='first qualifying event in first8 worlds, factual future targets frozen',
        detector_scope='S-COOPBEHAV descriptive events; no native-release attribution')
    if detector_root: summary['detector_source_sha256']=detector_sources
    if summary['charged_sim_s']>99360: raise ValueError('live cap exceeded')
    csvwrite(dest/'paired_worlds.csv',rows)
    csvwrite(dest/'oracle_options.csv',oracles)
    if credits: csvwrite(dest/'group_credit.csv',credits)
    if behaviour: csvwrite(dest/'behaviour.csv',behaviour)
    write(dest/'SUMMARY.json',summary)
    write(dest/'SOURCES.json',sources)
    write(dest/'MANIFEST.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()})
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--detector-root')
    args=parser.parse_args();print(json.dumps(report(args.root,args.detector_root)))
