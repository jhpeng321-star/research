"""Reproducible descriptive CSVs/figures for a verified EXAM and registered OOF fit."""
import argparse
import csv
from pathlib import Path

import numpy as np

from .protocol import sha,canonical
from .storage import read


def csvread(path):
    with Path(path).open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def publish(exam, learned, dest, refresh=False):
    exam,learned,dest=map(Path,(exam,learned,dest))
    for name,expected in read(exam/'MANIFEST.json').items():
        if Path(name).name!=name or sha(exam/name)!=expected:
            raise ValueError('EXAM checksum '+name)
    labels=[r for r in csvread(exam/'labels.csv') if r['horizon_s']=='600']
    ids=sorted({r['state_id'] for r in labels})
    bykey={(r['state_id'],r['option']):r for r in labels}
    oof={r['state_id']:r for r in csvread(learned/'oof_predictions.csv')}
    if set(oof)!=set(ids): raise ValueError('OOF/export mismatch')
    if refresh:
        if read(dest/'SUMMARY.json')['exam_manifest_sha256']!=sha(exam/'MANIFEST.json'):
            raise ValueError('refuse to refresh a different EXAM')
        for name,expected in read(dest/'MANIFEST.json').items():
            if sha(dest/name)!=expected: raise ValueError('derived output was edited: '+name)
    else:
        dest.mkdir()
    def emit_csv(name,rows):
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (dest/name).open('w' if refresh else 'x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,keys);writer.writeheader();writer.writerows(rows)
    def emit_json(name,value):
        with (dest/name).open('w' if refresh else 'x',encoding='utf-8') as stream:
            stream.write(canonical(value)+'\n')
    options=[]
    for oid in ('O1','O2','O3','O4'):
        rr=[r for r in labels if r['option']==oid and r['available']=='True']
        values=np.array([float(r['D']) for r in rr])
        options.append(dict(option=oid,available=len(rr),helpful=int((values>0).sum()),
            mean_D=float(values.mean()) if len(values) else None,
            max_D=float(values.max()) if len(values) else None,
            min_D=float(values.min()) if len(values) else None))
    states=[]
    for identity in ids:
        available=[bykey[identity,o] for o in ('O0','O1','O2','O3','O4') if bykey[identity,o]['available']=='True']
        best=max(available,key=lambda r:float(r['D']))
        chosen=oof[identity]['option'];r=bykey[identity,'O0']
        states.append(dict(state_id=identity,world_id=r['world_id'],cell=r['cell'],
            oracle_option=best['option'],oracle_D=float(best['D']),learned_option=chosen,
            learned_D=float(bykey[identity,chosen]['D'])))
    cells=[]
    for cell in sorted({r['cell'] for r in states},key=int):
        rr=[r for r in states if r['cell']==cell]
        cells.append(dict(cell=cell,states=len(rr),worlds=len({r['world_id'] for r in rr}),
            helpful=sum(r['oracle_D']>0 for r in rr),mean_oracle_D=np.mean([r['oracle_D'] for r in rr]),
            mean_learned_D=np.mean([r['learned_D'] for r in rr])))
    credits=[r for r in csvread(exam/'group_credit.csv') if r['horizon_s']=='600' and r['available']=='True']
    summary=dict(states=len(states),worlds=len({r['world_id'] for r in states}),cells=len(cells),
        helpful_states=sum(r['oracle_D']>0 for r in states),options=options,
        available_group_credit_cases=len(credits),
        positive_DG=sum(float(r['D_G'])>0 for r in credits),
        positive_DG_and_CG=sum(float(r['D_G'])>0 and float(r['C_G'])>0 for r in credits),
        CG_range=[min(float(r['C_G']) for r in credits),max(float(r['C_G']) for r in credits)] if credits else None,
        partial=read(exam/'provenance.json')['partial'],
        learned_summary=read(learned/'SUMMARY.json'),
        exam_manifest_sha256=sha(exam/'MANIFEST.json'),script_sha256=sha(__file__))
    emit_csv('option_summary.csv',options);emit_csv('state_metrics.csv',states)
    emit_csv('cell_summary.csv',cells)
    emit_json('SUMMARY.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    distributions=[[float(r['D']) for r in labels if r['option']==o and r['available']=='True'] for o in ('O1','O2','O3','O4')]
    ax.boxplot(distributions,tick_labels=['O1 metering','O2 wall','O3 ramp gap','O4 gap hold'])
    ax.axhline(0,color='0.5',lw=1);ax.set(ylabel='Exact D600 (vehicle-seconds saved)',xlabel='Available group option',
        title=f'{len(states)} TRAIN decision states: exact option effects')
    fig.tight_layout();fig.savefig(dest/'option_effects.png',dpi=150);plt.close(fig)
    importance=csvread(learned/'feature_importance.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for oid,ax in zip(('O1','O2','O3','O4'),axes.flat):
        rr=sorted([r for r in importance if r['option']==oid],key=lambda r:float(r['importance']))[-5:]
        ax.barh([r['feature'] for r in rr],[float(r['importance']) for r in rr],color='0.35')
        ax.set(title=oid,xlabel='Registered GBT impurity importance (fraction)')
    fig.suptitle('Observable-state model diagnostics; importance is not causal benefit')
    fig.tight_layout();fig.savefig(dest/'registered_importance.png',dpi=150);plt.close(fig)
    pdp=csvread(learned/'partial_dependence.csv')
    fig,axes=plt.subplots(4,3,figsize=(12,11))
    for row,oid in enumerate(('O1','O2','O3','O4')):
        features=list(dict.fromkeys(r['feature'] for r in pdp if r['option']==oid))
        for column,ax in enumerate(axes[row]):
            if column>=len(features): ax.set_visible(False);continue
            feature=features[column]
            rr=sorted([r for r in pdp if r['option']==oid and r['feature']==feature],key=lambda r:float(r['value']))
            unit=('s' if feature=='time_s' else
                  'm/s' if feature.endswith(('_v','_speed')) else
                  'm' if feature.endswith(('_s','_gap','_backgap')) else
                  'vehicles/km' if feature.endswith('_density') else
                  'count / code')
            ax.plot([float(r['value']) for r in rr],[float(r['mean_predicted_D']) for r in rr],color='0.25',marker='.')
            ax.axhline(0,color='0.7',lw=.7)
            ax.set(title=oid,xlabel=f'{feature} ({unit})',ylabel='Mean predicted D600 (veh-s)')
    fig.suptitle('Registered GBT partial dependence: marginal model predictions, not causal rules')
    fig.tight_layout();fig.savefig(dest/'registered_partial_dependence.png',dpi=150);plt.close(fig)
    emit_json('MANIFEST.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='MANIFEST.json'})
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('exam');parser.add_argument('learned');parser.add_argument('dest')
    parser.add_argument('--refresh',action='store_true',help='refresh only checksum-verified derived outputs; EXAM is immutable')
    args=parser.parse_args();publish(args.exam,args.learned,args.dest,args.refresh)
