"""Outcome-blind admission alternation and logging for MAIN01:33 niche comparison."""
import copy
import itertools
import numpy as np


def search_config(ordinal,coop=None,alternate=False,variety=None):
    """it2_K2 niches 1/4 alternation; optional additive coop_objective keys (default none);
    optional additive search_families keys (``variety`` dict of registered variety_* keys),
    merged on EVERY ordinal: the variety mix is independent of the coop alternation parity.

    alternate: coop only on ordinals with (ordinal//2)%2==0, crossing the niche parity
    (outcome-blind 2x2: niches1/4 x coop on/off)."""
    if int(ordinal)!=ordinal or ordinal<0:raise ValueError('nonnegative niche admission ordinal')
    out=dict(iterations=2,screen_K=2,niches=(1,4)[int(ordinal)%2])
    if coop and coop.get('coop_objective') and not (alternate and (int(ordinal)//2)%2):
        out.update(coop)
    if variety:
        from .variety import split_config
        split_config(dict(variety))   # registered keys and value checks only
        out.update(variety)
    return out


def validate(config):
    base={k:v for k,v in config.items() if not (k.startswith('coop_') or k.startswith('variety_'))}
    if base not in (search_config(0),search_config(1)):
        raise ValueError('registered it2_K2 niches=1/4 configuration required')
    coop={k:v for k,v in config.items() if k.startswith('coop_')}
    if coop:
        from .coop_search import split_config
        if not coop.get('coop_objective'):raise ValueError('coop keys require coop_objective=True')
        if 'coop_novelty_bank' in coop:raise ValueError('reference the novelty bank by coop_novelty_bank_sha256, never embed it')
        sha=coop.get('coop_novelty_bank_sha256')
        if sha is not None and not (isinstance(sha,str) and len(sha)==64 and all(c in '0123456789abcdef' for c in sha)):
            raise ValueError('coop_novelty_bank_sha256 must be a sha256 hex digest')
        coop=dict(coop,coop_novelty_bank_sha256=None)   # the digest is resolved on the GPU host
        split_config(dict(coop))   # registered keys and value checks only
    variety={k:v for k,v in config.items() if k.startswith('variety_')}
    if variety:
        from .variety import split_config
        if not variety.get('variety_families'):raise ValueError('variety keys require variety_families=True')
        split_config(dict(variety))   # registered keys and value checks only
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
