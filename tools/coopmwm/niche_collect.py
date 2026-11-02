"""Outcome-blind admission alternation and logging for MAIN01:33 niche comparison."""
import copy
import itertools
import numpy as np


def search_config(ordinal):
    if int(ordinal)!=ordinal or ordinal<0:raise ValueError('nonnegative niche admission ordinal')
    return dict(iterations=2,screen_K=2,niches=(1,4)[int(ordinal)%2])


def validate(config):
    if config not in (search_config(0),search_config(1)):
        raise ValueError('registered it2_K2 niches=1/4 configuration required')
    return dict(config)


def annotate(result,ranked,record,config):
    """Keep ALL original K8 finalists, including negatives; never re-score/reselect."""
    if 'niches' not in (config or {}):return ranked
    mode=int(config['niches'])
    plans=copy.deepcopy(ranked)
    byid={p['id']:p['niche'] for p in result.get('finalists',[]) if 'niche' in p}
    byid.update({p['id']:p['niche'] for p in (result.get('topk') or []) if 'niche' in p})
    for p in plans:
        if p['id'] in byid:p['niche']=byid[p['id']]
    from .mppi_search import plan_distance
    distances=[plan_distance(a,b) for a,b in itertools.combinations(plans,2)]
    braking=[bool((np.asarray(p['profile'],float).mean(axis=1)<0).any()) for p in plans]
    members=record.get('group',[])
    result['variation']=dict(niches=mode,candidates=len(plans),
        mean_pairwise_distance=float(np.mean(distances)) if distances else None,
        occupied_niches=len({p['niche'] for p in plans if 'niche' in p}) if mode>1 else 1,
        share_with_net_deceleration=sum(braking)/len(braking) if braking else None,
        sign_rule='any member mean acceleration < 0; exact sign, no fitted threshold',
        members=[{k:r[k] for k in ('vid','s_m','v_mps','lane')} for r in record.get('rows',[]) if r['vid'] in members],
        shortlist_contract='all eight K8 finalists, unfiltered; niche search allocates finalists across niches; ordered by WM mean')
    return plans
