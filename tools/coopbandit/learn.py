"""Nested world-grouped regression, signed CV uncertainty and OOF policy exam."""
import csv
import json
import math
from pathlib import Path
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from .protocol import canonical, sha
from .storage import read, write
from .export import csvwrite

OPTIONS = ('O1','O2','O3','O4')


def estimator():
    return GradientBoostingRegressor(n_estimators=100,max_depth=2,min_samples_leaf=8,
                                     loss='huber',random_state=0)


def fit_models(x, y):
    return [estimator().fit(x,y[:,i]) for i in range(4)]


def predictions(models, x):
    return np.stack([m.predict(x) for m in models],axis=1)


def radius(errors, groups):
    maxima = np.array([np.max(errors[groups == g]) for g in np.unique(groups)])
    rank = min(len(maxima),math.ceil((len(maxima)+1)*.95))
    return max(0.,float(np.sort(maxima)[rank-1]))


def calibration(x, y, groups, folds):
    pred = np.zeros_like(y)
    for train, test in GroupKFold(folds).split(x,y,groups):
        pred[test] = predictions(fit_models(x[train],y[train]),x[test])
    return radius(pred-y,groups), pred


def choose(pred, available, uncertainty):
    values = np.where(available,pred,-np.inf)
    best = values.argmax(axis=1)
    positive = values[np.arange(len(values)),best] > uncertainty
    return np.where(positive,best+1,0)


def learn(root, partial=False, exam_name='exam', dest_name='learned'):
    root = Path(root)
    if not partial and not (root/'DATA_COMPLETE.json').exists():
        raise ValueError('complete verified dataset required before fitting')
    dest = root/dest_name
    dest.mkdir()
    states = [json.loads(s) for s in (root/exam_name/'states.jsonl').read_text().splitlines()]
    with (root/exam_name/'labels.csv').open(newline='') as f:
        labels = {(r['state_id'],r['option']):r for r in csv.DictReader(f) if r['horizon_s']=='600'}
    keys = sorted(states[0]['features'])
    x = np.array([[s['features'][k] for k in keys] for s in states])
    y = np.array([[float(labels[s['state_id'],o]['D']) for o in OPTIONS] for s in states])
    available = np.array([[labels[s['state_id'],o]['available']=='True' for o in OPTIONS] for s in states])
    groups = np.array([s['world_id'] for s in states])
    if len(np.unique(groups)) < 7:
        raise ValueError('at least seven complete training worlds required for nested CV')
    if any(981080 <= int(g) <= 981091 for g in groups):
        raise ValueError('live dev world leaked into fit/calibration')
    outer = np.zeros_like(y)
    selected = np.zeros(len(x),int)
    fold_rows, state_rows = [], []
    for fold,(train,test) in enumerate(GroupKFold(5).split(x,y,groups)):
        q, _ = calibration(x[train],y[train],groups[train],4)
        outer[test] = predictions(fit_models(x[train],y[train]),x[test])
        selected[test] = choose(outer[test],available[test],q)
        fold_rows.append(dict(fold=fold,radius=q,train_worlds=len(np.unique(groups[train])),test_worlds=len(np.unique(groups[test]))))
        for j in test:
            state_rows.append(dict(state_id=states[j]['state_id'],world_id=int(groups[j]),fold=fold,
                                   option=(('O0',)+OPTIONS)[selected[j]],radius=q,
                                   **{f'pred_{o}':float(outer[j,i]) for i,o in enumerate(OPTIONS)}))
    full_q, cv = calibration(x,y,groups,5)
    models = fit_models(x,y)
    with (dest/'policy.pkl').open('xb') as f:
        pickle.dump(dict(models=models,keys=keys,radius=full_q,options=OPTIONS),f)
    gains = np.column_stack((np.zeros(len(x)),y))
    oracle = np.max(np.where(np.column_stack((np.ones(len(x),bool),available)),gains,-np.inf),axis=1)
    actual = gains[np.arange(len(x)),selected]
    world_rows = []
    for g in np.unique(groups):
        m=groups==g
        world_rows.append(dict(world_id=int(g),oracle=float(oracle[m].mean()),
            learned=float(actual[m].mean()),regret=float((oracle-actual)[m].mean()),
            helpful=float((oracle[m]>0).mean()),cooperate=float((selected[m]>0).mean())))
    rng=np.random.default_rng(20261031)
    boot=rng.integers(0,len(world_rows),(2000,len(world_rows)))
    summary={}
    for k in ('oracle','learned','regret','helpful','cooperate'):
        values=np.array([r[k] for r in world_rows])
        summary[k]=dict(mean=float(values.mean()),ci95=np.quantile(values[boot].mean(axis=1),[.025,.975]).tolist())
    importances=[dict(option=o,feature=k,importance=float(m.feature_importances_[j]))
                 for o,m in zip(OPTIONS,models) for j,k in enumerate(keys)]
    pdp=[]
    for o,m in zip(OPTIONS,models):
        for j in np.argsort(m.feature_importances_)[-3:]:
            for value in np.unique(np.quantile(x[:,j],np.linspace(0,1,9))):
                modified=x.copy();modified[:,j]=value
                pdp.append(dict(option=o,feature=keys[j],value=float(value),mean_predicted_D=float(m.predict(modified).mean())))
    csvwrite(dest/'oof_predictions.csv',state_rows)
    csvwrite(dest/'world_metrics.csv',world_rows)
    csvwrite(dest/'feature_importance.csv',importances)
    csvwrite(dest/'partial_dependence.csv',pdp)
    write(dest/'SUMMARY.json',dict(metrics=summary,full_radius=full_q,folds=fold_rows,
           states=len(states),worlds=len(world_rows),partial=partial,policy_sha256=sha(dest/'policy.pkl'),
          confidence='95% finite-sample quantile of world-max signed overprediction; nested outer5/inner4 CV',
          limitations='CV-derived uncertainty under world exchangeability; no formal OOD or simultaneous live guarantee'))
    write(dest/'COMPLETE.json',dict(policy_sha256=sha(dest/'policy.pkl')))
    return summary


def load_policy(path):
    with Path(path).open('rb') as f:
        return pickle.load(f)


def decide(policy, features, available):
    x=np.array([[features[k] for k in policy['keys']]])
    pred=predictions(policy['models'],x)
    selected=int(choose(pred,np.array([available]),policy['radius'])[0])
    return (('O0',)+OPTIONS)[selected],pred[0].tolist()
