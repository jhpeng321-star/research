"""Glue between the delegated discovery-pure search and qualified WM evaluator."""
import time
import torch
from . import mppi_search as search
from .protocol import digest
from .plan_wm import evaluate
from .exploration import regime


def posthoc(plan,record,bounds,policy):
    """Labels are descriptive only; unsupported prototypes cannot kill a search."""
    if len(record['group']) not in (2,3):
        return dict(label='new',reason='no inherited prototypes for this group size')
    try:
        from .plans import family_seeds
        prototypes=family_seeds(record['rows'],record['group'],record['commands'],bounds)
        return search.posthoc_label(plan,prototypes,policy)
    except Exception as error:
        import traceback
        return dict(label='unlabelled',error=repr(error),traceback=traceback.format_exc())


def imagined_centres(rt,record,memory,width=2,speed2=True,transfers=False):
    trace=[]
    acts=search.comparators(record,600)
    begin=time.monotonic()
    scores=evaluate(rt,record,acts,1,300,chunk_actions=2,trim=False,
                    pad_actions=width,speed2=speed2,memory=memory,trace=trace,transfers=transfers)
    if len(trace)!=300:raise ValueError('incomplete comparator acceleration trace')
    ids=[r['vid'] for r in record['rows']]
    slots=[ids.index(v) for v in record['group']]
    values=torch.stack(trace)[:,:2,slots].permute(1,2,0).cpu().tolist()
    return values[0],values[1],dict(wall_s=time.monotonic()-begin,K=1,H=300,
        source='future0 pre-safety current-policy/MAPPO acceleration',costs=scores)


def label_list8_search(run):
    """Live c4/coopnative ls_gpu.capture_search(label_list8=True), same arithmetic.

    top_k=8 with one finalist per k keeps exactly the original eight final K8 WM calls;
    'topk' becomes those eight candidates ordered by WM mean cost, UNFILTERED (no
    positive-saving filter), each labelled with its K8 savings vs MAPPO and policy.
    """
    import copy

    def wrapped(*args,**kwargs):
        values=list(args);evaluate=values[4];last=[]
        def score(record,actions,k,h):
            costs=evaluate(record,actions,k,h)
            if k==8:
                def mean(key):
                    xs=costs[key]
                    return sum(x['team_delay'] if isinstance(x,dict) else x for x in xs)/len(xs)
                plans=[a for a in actions if 'profile' in a]
                last[:]=[dict(copy.deepcopy(p),saving_vs_mappo=mean('MAPPO')-mean(p['id']),
                    saving_vs_policy=mean('policy')-mean(p['id']))
                    for p in sorted(plans,key=lambda p:(mean(p['id']),p['id']))]
            return costs
        values[4]=score
        kwargs['config']=dict(kwargs.get('config') or {},top_k=8,finalists_per_k=1)
        out=run(*values,**kwargs)
        if 'niches' in kwargs['config']:
            from .niche_collect import annotate
            last[:]=annotate(out,last,values[0],kwargs['config'])
        out['collection_top8']=last[:8]
        out['topk']=copy.deepcopy(last[:8]);out['available_k']=len(out['topk'])
        return out
    return wrapped


def plan(rt,record,memory,binding):
    from .plan_service import batch_width
    from .committed import members
    groups=search.group_components(record['rows'],exclude=members(record.get('committed_plans',[])))
    group=next((g for g in groups if g['members']==record['group']),None)
    if group is None:raise ValueError('group does not match registered observable components')
    policy,mappo,receipt=imagined_centres(rt,record,memory,speed2=binding.get('speed2',False),transfers=binding.get('speed2_transfers',False))
    learning=record.get('search_learning')
    covariance=search.CovarianceModel.from_json(learning['covariance']) if learning else None
    library=search.PlanLibrary.from_json(learning['library']) if learning else None
    def score(rec,actions,k,h):
        width=batch_width(binding,k)
        return evaluate(rt,rec,actions,k,h,chunk_actions=width,trim=False,
                        pad_actions=width,speed2=binding.get('speed2',False),memory=memory,transfers=binding.get('speed2_transfers',False))
    bounds=(float(rt.mp.policy.accel_min),float(rt.mp.policy.accel_max))
    from .plan_proposer import RecordingEvaluator
    recording=RecordingEvaluator(score)
    run=label_list8_search(search.search) if binding.get('label_list8') else search.search
    config=dict(binding.get('search_config') or {})
    if 'search_config' in record:
        from .niche_collect import validate
        config.update(validate(record['search_config']))
    result=run(record,group,policy,bounds,recording,regime(record['rows']),
        mappo_traj=mappo,covariance=covariance,library=library,
        seed=int(digest(record)[:15],16),config=config)
    from .plan_proposer import receipt_from_search
    try:
        result['proposer_receipt']=receipt_from_search(record,result,recording,arm='B',
            initial_covariance=covariance.get(regime(record['rows']),len(record['group'])) if covariance else None)
    except Exception as error:
        # MAIN (harden-int): logging-only receipt, isolated on its own; a logging
        # failure never discards (declines) a valid search result.
        import traceback
        result['proposer_receipt']=None
        result['proposer_receipt_error']=dict(error=repr(error),traceback=traceback.format_exc())
    from .overlap import recheck
    searched=result
    result=recheck(record,result,score)
    if record.get('committed_plans') and result.get('topk'):
        # MAIN (harden-int): each verified candidate gets the overlap truncation of ITS
        # OWN recheck (same rule/K8/H), never the Thompson plan's.
        topk=[]
        for c in result['topk']:
            own=recheck(record,dict(searched,plan=c,accepted=True),score)['overlap_recheck']
            topk.append(dict(c,overlap_truncated_plan_ids=own['truncated_plan_ids'],
                overlap_keeps_existing=own['chosen']=='overlap_drop_new'))
        result=dict(result,topk=topk)
    result.update(record_sha256=digest(record),centre_rollout=receipt,
        memory=memory.receipt if memory is not None else {'mode':'cold'})
    if result['accepted']:
        # Inherited families are used only AFTER the candidate has been selected.
        label=posthoc(result['plan'],record,bounds,policy)
        result.update(predicted_rank=next(i+1 for i,p in enumerate(result['topk'])
            if p['id']==result['plan']['id']),known_family=label['label'] not in ('new','unlabelled'),posthoc=label)
    return result
