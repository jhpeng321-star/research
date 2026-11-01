"""Use S-DISTILL's exact feature/model/selection ABI with WM targets only."""
import json
from pathlib import Path
import pickle

import numpy as np
from threadpoolctl import threadpool_limits

from tools.coopbandit.protocol import sha
from tools.distill.data import KEYS,OPTIONS
from tools.distill.model import fit_values,fit_classifier
from tools.distill.train import select,crossfit,evaluate,splits,residuals,choose,predict_values


def train(root,dest,last_setting):
    from tools.wmteach.run import save,read
    if not (root/'QUALITY.json').is_file(): raise ValueError('quality check must precede fitting')
    records=[]
    for setting in range(last_setting+1):
        for seed in range(969400,969432):
            r=read(root/'tasks'/f'w{seed}_t300_setting{setting}'/'LABEL_H300.json')
            if r['role']!='TRAIN' or r['world_id']!=seed or r['horizon_s']!=300:
                raise ValueError('FIT WM labels only; no diagnostic outcomes in fitting')
            records.append(r)
    return train_records(root,dest,records)


def train_records(root,dest,records):
    """Shared student fit, including the large-corpus continuation."""
    from tools.wmteach.run import save
    identity=json.loads((root/'IDENTITY.json').read_text())
    if not (root/'QUALITY.json').is_file():
        raise ValueError('quality check must precede fitting')
    if len({r['state_id'] for r in records}) != len(records):
        raise ValueError('duplicate decision states')
    if any(r['role']!='TRAIN' or r['world_id'] not in range(969400,969432) or
           r['horizon_s']!=300 for r in records):
        raise ValueError('FIT-only H300 WM targets required')
    x=np.array([[r['features'][k] for k in KEYS] for r in records],float)
    y=np.array([[r['effects'].get(o,0.) for o in OPTIONS] for r in records],float)
    av=np.array([r['available'] for r in records],bool)
    groups=np.array([r['world_id'] for r in records])
    with threadpool_limits(limits=1):
        oof=np.zeros(len(x),int); fold_receipts=[]
        for tr,te in splits(x,groups,4):
            best,scores,cache=select(x[tr],y[tr],av[tr],groups[tr],3)
            models=fit_values(x[tr],y[tr],av[tr],best['family'])
            pred=predict_values(models,x[te])
            oof[te]=choose(pred,av[te],cache[best['family']][1],best['confidence'])
            fold_receipts.append(dict(train_worlds=sorted(set(groups[tr].tolist())),
                test_worlds=sorted(set(groups[te].tolist())),selected=best))
        best,scores,cache=select(x,y,av,groups,4)
        models=fit_values(x,y,av,best['family'])
        classifier=fit_classifier(x,y,av)
    dest.mkdir()
    payload=dict(keys=KEYS,options=OPTIONS,models=models,errors=cache[best['family']][1],
        family=best['family'],confidence=best['confidence'],label_horizon_s=300,
        train_worlds=sorted(set(groups.tolist())),teacher=identity['backend']+'_WM_only',
        provenance=dict(identity_sha256=sha(root/'IDENTITY.json'),quality_sha256=sha(root/'QUALITY.json')))
    with (dest/'policy.pkl').open('xb') as f: pickle.dump(payload,f,protocol=5)
    with (dest/'classifier.pkl').open('xb') as f: pickle.dump(dict(model=classifier,keys=KEYS),f,protocol=5)
    save(dest/'SUMMARY.json',dict(states=len(x),worlds=len(set(groups.tolist())),primary_horizon=300,selected=best,
        inner_scores=scores,nested=evaluate(y,av,groups,oof),folds=fold_receipts,
        meaning='world-grouped nested CV against WM labels, NOT exact-SUMO saving or live efficacy',
        policy_sha256=sha(dest/'policy.pkl'),quality_before_fit=True,new_sumo_sim_s=0))
    save(dest/'COMPLETE.json',dict(policy_sha256=sha(dest/'policy.pkl'),new_sumo_sim_s=0))
