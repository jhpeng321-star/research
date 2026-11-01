"""Honest nested world CV. All saved artifacts are exclusive-create snapshots."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from tools.coopbandit.protocol import sha
from .data import load_exam, arrays, save, KEYS, OPTIONS, checked, planner_records
from .model import (FAMILIES, CONFIDENCES, fit_values, predict_values, residuals,
                    choose, teacher, fit_classifier, classify)


def splits(x, groups, folds):
    for train,test in GroupKFold(folds).split(x,groups=groups):
        if set(groups[train]) & set(groups[test]):
            raise ValueError('world leakage')
        yield train,test


def world_mean(values, groups):
    return float(np.mean([np.mean(values[groups == g]) for g in np.unique(groups)]))


def ci(values, groups):
    world = np.array([np.mean(values[groups == g]) for g in np.unique(groups)])
    draws = np.random.default_rng(20261101).integers(0,len(world),(10000,len(world)))
    return dict(mean=float(world.mean()), ci95=np.quantile(world[draws].mean(1),[.025,.975]).tolist(),
                worlds=len(world))


def crossfit(x,y,available,groups,family,folds):
    pred = np.zeros_like(y)
    for tr,te in splits(x,groups,folds):
        pred[te] = predict_values(fit_values(x[tr],y[tr],available[tr],family),x[te])
    return pred


def select(x,y,available,groups,folds):
    candidates, cache = [], {}
    for family in FAMILIES:
        pred = crossfit(x,y,available,groups,family,folds)
        errors = residuals(pred,y,available)
        cache[family] = (pred,errors)
        for level in CONFIDENCES:
            action = choose(pred,available,errors,level)
            gain = y[np.arange(len(y)),action]
            candidates.append(dict(family=family,confidence=level,saving=world_mean(gain,groups),
                                   moves=int((action != 0).sum())))
    best = max(candidates,key=lambda c:(c['saving'],-c['confidence'],-FAMILIES.index(c['family'])))
    return best, candidates, cache


def evaluate(y,available,groups,actions):
    truth = teacher(y,available)
    gain = y[np.arange(len(y)),actions]
    helpful = truth != 0
    return dict(saving=ci(gain,groups), teacher_agreement=world_mean(actions == truth,groups),
                active_teacher_agreement=float((actions[helpful] == truth[helpful]).mean()) if helpful.any() else None,
                teacher_moves=int(helpful.sum()), moves=int((actions != 0).sum()),
                action_counts=dict(Counter(OPTIONS[a] for a in actions)),
                oracle=ci(y[np.arange(len(y)),truth],groups))


def run(dest, exam=None):
    dest = checked(dest)
    if not dest.parent.is_dir():
        raise ValueError('output parent must already exist')
    dest.mkdir()  # Never overwrite/resume/retry an existing fit snapshot.
    records, provenance = load_exam() if exam is None else load_exam(exam)
    x,y,av,groups = arrays(records)
    selected = np.zeros(len(x),int)
    raw = {f:np.zeros_like(y) for f in FAMILIES}
    raw_actions = {f:np.zeros(len(x),int) for f in FAMILIES}
    direct = np.zeros(len(x),int)
    folds, oof = [], []
    for fold,(tr,te) in enumerate(splits(x,groups,5)):
        best, scores, cache = select(x[tr],y[tr],av[tr],groups[tr],4)
        fold_actions = {}
        for f in FAMILIES:
            models = fit_values(x[tr],y[tr],av[tr],f)
            pred = predict_values(models,x[te])
            raw[f][te] = pred
            # Per-family confidence is selected on INNER data, not outer labels.
            winner = max((s for s in scores if s['family'] == f),
                         key=lambda c:(c['saving'],-c['confidence']))
            errors = cache[f][1]
            fold_actions[f] = choose(pred,av[te],errors,winner['confidence'])
            raw_actions[f][te] = fold_actions[f]
        selected[te] = fold_actions[best['family']]
        direct[te] = classify(fit_classifier(x[tr],y[tr],av[tr]),x[te],av[te])
        folds.append(dict(fold=fold,train_worlds=sorted(set(groups[tr].tolist())),
                          test_worlds=sorted(set(groups[te].tolist())),selected=best,inner_scores=scores))
        for j in te:
            oof.append(dict(state_id=records[j]['state_id'],world_id=int(groups[j]),fold=fold,
                            chosen=OPTIONS[selected[j]],teacher=OPTIONS[teacher(y[j:j+1],av[j:j+1])[0]],
                            exact_D=float(y[j,selected[j]]),family=best['family'],confidence=best['confidence'],
                            predictions={f:raw[f][j].tolist() for f in FAMILIES}))
        print(json.dumps(dict(fold=fold,selected=best,outer_moves=int((selected[te]!=0).sum()))),flush=True)
    final, scores, cache = select(x,y,av,groups,5)
    models = fit_values(x,y,av,final['family'])
    payload = dict(keys=KEYS,options=OPTIONS,models=models,errors=cache[final['family']][1],
                   family=final['family'],confidence=final['confidence'],label_horizon_s=600,
                   train_worlds=sorted(set(groups.tolist())),provenance=provenance)
    with (dest/'policy.pkl').open('xb') as f:
        pickle.dump(payload,f,protocol=5)
    with (dest/'classifier.pkl').open('xb') as f:
        pickle.dump(dict(model=fit_classifier(x,y,av),keys=KEYS,train_worlds=payload['train_worlds']),f,protocol=5)
    # Frozen before opening planner DEV outcomes. No downstream policy reselection.
    freeze = dict(policy_sha256=sha(dest/'policy.pkl'),selected=final,train_worlds=payload['train_worlds'],
                  source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
                  versions=dict(python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__),
                  provenance=provenance)
    save(dest/'FREEZE.json',freeze)
    summary = dict(states=len(x),worlds=len(set(groups)),primary_horizon=600,
                   nested=evaluate(y,av,groups,selected),
                   per_family={f:evaluate(y,av,groups,raw_actions[f]) for f in FAMILIES},
                   direct_classifier=evaluate(y,av,groups,direct),folds=folds,
                   final_selection=final,final_inner_scores=scores,
                   final_model_convergence_warnings=[m.distill_convergence_warnings_ for m in models],
                   policy_sha256=freeze['policy_sha256'],
                   meaning='offline exact-label saving/state, not live J600/J900; no pre-gate')
    importance = []
    # Final GBT impurity importance is descriptive TRAIN fit; explicitly labelled.
    for k,m in enumerate(models,1):
        if hasattr(m,'feature_importances_'):
            importance.extend(dict(option=OPTIONS[k],feature=key,importance=float(v),
                                   method='final_TRAIN_GBT_impurity')
                              for key,v in zip(KEYS,m.feature_importances_))
    # Model-agnostic importance for both models: TRAIN-only permuted prediction sensitivity.
    baseline = predict_values(models,x)
    rng = np.random.default_rng(20261101)
    for j,key in enumerate(KEYS):
        perm = x.copy()
        perm[:,j] = perm[rng.permutation(len(x)),j]
        changed = predict_values(models,perm)
        for k in range(1,5):
            mask = av[:,k]
            importance.append(dict(option=OPTIONS[k],feature=key,
                                   importance=float(np.mean(abs(changed[mask,k]-baseline[mask,k]))) if mask.any() else 0.,
                                   method='TRAIN_prediction_sensitivity_veh_s_not_causal'))
    trigger_rows = []
    for family,actions in {**raw_actions,'nested':selected,'direct_classifier':direct}.items():
        for k in range(1,5):
            mask = actions == k
            trigger_rows.append(dict(model=family,option=OPTIONS[k],states=int(mask.sum()),
                observed_feature_means={key:float(x[mask,j].mean()) for j,key in enumerate(KEYS)} if mask.any() else {},
                exact_mean_D=float(y[mask,k].mean()) if mask.any() else None))
    save(dest/'SUMMARY.json',summary)
    save(dest/'OOF.json',sorted(oof,key=lambda r:r['state_id']))
    save(dest/'FEATURE_IMPORTANCE.json',importance)
    save(dest/'OBSERVABLE_TRIGGERS.json',trigger_rows)
    save(dest/'TRAIN_RECORDS.json',records)
    audit(dest,payload)
    save(dest/'COMPLETE.json',dict(policy_sha256=freeze['policy_sha256'],new_sumo_sim_s=0))
    print(json.dumps(dict(final=final,nested=summary['nested'],policy_sha256=freeze['policy_sha256'])),flush=True)
    return summary


def audit(dest,payload):
    records,sources = planner_records()
    output = []
    for r in records:
        x = np.array([[r['features'][k] for k in KEYS]])
        av = np.array([[o in r['available'] for o in OPTIONS]])
        pred = predict_values(payload['models'],x)
        a = OPTIONS[int(choose(pred,av,payload['errors'],payload['confidence'])[0])]
        effects = r['evaluated_effects']
        output.append(dict(**r,student=a,predicted_mean_D=pred[0].tolist(),
                           student_exact_D=effects.get(a),teacher_agreement=a==r['teacher']))
    cohorts = {}
    for h in (120,300):
        rows = [r for r in output if r['horizon_s'] == h]
        known = [r for r in rows if r['student_exact_D'] is not None]
        cohorts[str(h)] = dict(states=len(rows),worlds=len({r['world_id'] for r in rows}),
            agreement=float(np.mean([r['teacher_agreement'] for r in rows])),
            student_moves=sum(r['student'] != 'O0' for r in rows),
            selected_effect_measured=len(known),selected_effect_missing=len(rows)-len(known),
            conditional_measured_saving=ci(np.array([r['student_exact_D'] for r in known]),
                                         np.array([r['world_id'] for r in known])) if known else None,
            limitation='DEV after freeze; horizon differs from trained D600; missing quota-pruned labels never imputed')
    save(dest/'PLANNER_DEV_RECORDS.json',output)
    save(dest/'PLANNER_DEV_AUDIT.json',dict(source_hashes=sources,cohorts=cohorts,
                                           frozen_policy_sha256=sha(dest/'policy.pkl'),never_fit=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dest',type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.dest)


if __name__ == '__main__':
    main()
