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
    result=search.search(record,group,policy,bounds,recording,regime(record['rows']),
        mappo_traj=mappo,covariance=covariance,library=library,
        seed=int(digest(record)[:15],16),config=binding.get('search_config'))
    from .plan_proposer import receipt_from_search
    result['proposer_receipt']=receipt_from_search(record,result,recording,arm='B',
        initial_covariance=covariance.get(regime(record['rows']),len(record['group'])) if covariance else None)
    from .overlap import recheck
    result=recheck(record,result,score)
    result.update(record_sha256=digest(record),centre_rollout=receipt,
        memory=memory.receipt if memory is not None else {'mode':'cold'})
    if result['accepted']:
        # Inherited families are used only AFTER the candidate has been selected.
        label=posthoc(result['plan'],record,bounds,policy)
        result.update(predicted_rank=next(i+1 for i,p in enumerate(result['topk'])
            if p['id']==result['plan']['id']),known_family=label['label'] not in ('new','unlabelled'),posthoc=label)
    return result
