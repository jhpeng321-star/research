"""Thin bounded adapter around the frozen WMTEACH batched K8 code path."""
import ast
import contextlib
import copy
import inspect
import json
import math
import types
from .protocol import digest,HORIZONS
from .credit import paired


def candidates(record):
    chosen=record['targets']; members=record['group']
    if len(members)!=2 or len(set(members))!=2:
        raise ValueError('complete two-member RSU group required')
    return [dict(id=name,window=10,targets={v:t for v,t in chosen.items() if v not in revert})
            for name,revert in [('joint',[]),*[(f'revert:{i}',[v]) for i,v in enumerate(members)],
                                ('all',members)]]


def validate(record):
    from tools.coopnative.policy import encode
    encode(record['rows'])
    allowed={'state_id','world_id','time_s','rows','targets','group'}
    if set(record) not in (allowed,allowed|{'mode','setting'}) or not 300<=record['time_s']<=1190:
        raise ValueError('observable replay whitelist/clock violation')
    if 'mode' in record and (record['mode']!='proposal' or record['setting'] not in range(4)):
        raise ValueError('unregistered proposal mode/setting')
    ids={r['vid'] for r in record['rows'] if r['is_cav']}
    if not set(record['group'])<=ids or not set(record['targets'])<=ids:
        raise ValueError('non-CAV target/group')
    for target in record['targets'].values():
        if set(target)!={'speed','tau'} or not all(math.isfinite(v) for v in target.values()) or target['tau']<=0:
            raise ValueError('invalid setpoint')
    return record


@contextlib.contextmanager
def executor_adapter():
    """Use the original option executor; only its target/request decoder changes."""
    from tools.leverprobe import intervention
    from tools.coopfree.policy import requests
    from tools.wmteach import runtime
    old_targets,old_requests=intervention.targets,intervention.requests
    old_prepare=runtime.prepare_batch
    tree=ast.parse(inspect.getsource(old_prepare));counts={300:0,960:0}
    class Capacity(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int and node.value in counts:
                counts[node.value]+=1
                return ast.copy_location(ast.Constant({300:600,960:1600}[node.value]),node)
            return node
    tree=Capacity().visit(tree);ast.fix_missing_locations(tree)
    if counts!={300:1,960:1}:raise ValueError('frozen entrant capacity ABI changed')
    ns=dict(old_prepare.__globals__);exec(compile(tree,'<coopmwm-H600-entrant-capacity>','exec'),ns)
    runtime.prepare_batch=ns['prepare_batch']
    intervention.targets=lambda action,rows:action['targets']
    intervention.requests=lambda action,rows,fixed:(requests(fixed,rows,zone=None),'setpoint')
    try:yield
    finally:
        intervention.targets,intervention.requests=old_targets,old_requests
        runtime.prepare_batch=old_prepare


def adapted_futures(horizons=HORIZONS):
    """Minimal AST parameterization: H60/120/300/600, same K8 implementation.

    Frozen upstream sources are staged/hash-bound. Fail closed on changed structure.
    No physical rollout, sampling, costs, safety, or option executor is reimplemented.
    """
    from tools.wmteach import scale,runtime
    tree=ast.parse(inspect.getsource(scale.futures))
    counts={'steps':0,'horizons':0}
    class Horizon(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value==600:
                counts['steps']+=1
                return ast.copy_location(ast.Constant(2*max(horizons)),node)
            return node
        def visit_Tuple(self,node):
            if all(isinstance(v,ast.Constant) for v in node.elts) and [v.value for v in node.elts]==[120,300]:
                counts['horizons']+=1
                return ast.copy_location(ast.Tuple([ast.Constant(h) for h in horizons],ast.Load()),node)
            return self.generic_visit(node)
    tree=Horizon().visit(tree);ast.fix_missing_locations(tree)
    if counts!={'steps':1,'horizons':2}:
        raise ValueError('frozen batched K8 ABI changed')
    def source(rt,r):
        b=runtime.diagnostic_source(rt,r,0)
        # Identical population-marginal source prior used by the qualified teacher.
        # These values are inferred priors, never read from SUMO RequestLedger.
        r.update(prior_counters(b.entry_queue,r['time_s']))
        return b
    namespace=dict(scale.futures.__globals__,actions=lambda rows,setting:namespace['_actions'],
                   source_for=source,BACKEND='FULL-v4')
    exec(compile(tree,'<coopmwm-frozen-batched-multihorizon>','exec'),namespace)
    return namespace


def prior_counters(entry_queue,time_s):
    """Integer encode inferred FIT-prior counts, never actual SUMO counters.

    The frozen learned-arrival API requires integer count sufficient statistics.
    Its population-prior mean rate times a new observable clock is not generally
    integral; nearest-integer encoding preserves that rate to <=0.5/time_s.
    Exposure remains the disclosed inferred current-clock prior, not measured
    requests. It must not be described as causal source-counter conditioning.
    """
    backlog=entry_queue[0,:2].cpu().tolist()
    raw=(entry_queue[0,2:4]*time_s).cpu().tolist()
    if time_s<=0 or any(not math.isfinite(x) or x<0 for x in backlog+raw):
        raise ValueError('invalid inferred FIT source prior')
    return dict(backlog=backlog,request_total=[int(round(x)) for x in raw])


def proposal_candidates(record):
    """Decode inherited O1--O4 executors into the learner's W10 set-point ABI.

    Original geometric availability, magnitudes and option requests are reused.
    At each decision the initial executor target is frozen as a speed/time gap;
    the inherited common safety layer grades actual execution in both plants.
    """
    from tools.wmteach.labels import actions
    from tools.leverprobe.intervention import targets as fixed, requests as decode
    from tools.coopfree.policy import leader_gap
    rows=record['rows'];byid={r['vid']:r for r in rows};chosen=record['targets']
    result=[dict(id='policy',window=10,targets=chosen),dict(id='MAPPO',window=10,targets={})]
    for option in actions(rows,record['setting']):
        if option['id']=='O0':continue
        requests,_=decode(option,rows,fixed(option,rows))
        if not requests:continue
        changed={}
        for vid,r in requests.items():
            speed=r['target_mps'];gap=leader_gap(byid[vid],rows)
            tau=1. if gap is None else max(1e-6,gap/max(speed,1e-6))
            changed[vid]=dict(speed=speed,tau=tau)
        targets={**chosen,**changed}
        result.append(dict(id=option['id'],window=10,targets=targets,proposal_targets=changed))
    return result


def propose(rt,record):
    validate(record);acts=proposal_candidates(record)
    if len(acts)==2:
        return dict(accepted=False,option=None,targets={},expected_saving=0.,record_sha256=digest(record))
    ns=adapted_futures((300,));ns['_actions']=acts
    adapted=dict(record)
    with executor_adapter():
        chunks=[ns['futures'](rt,adapted,indices=tuple(range(i,i+2))) for i in range(0,8,2)]
    if any(not c['population_audit']['finite'] or not c['population_audit']['conserved'] for c in chunks):
        raise ValueError('proposal conservation/finite check failed')
    futures={a['id']:[v for c in chunks for v in c['futures']['300'][a['id']]] for a in acts}
    means={a:sum(v['team_delay'] for v in fs)/8 for a,fs in futures.items()}
    best=min(acts[2:],key=lambda a:(means[a['id']],a['id']))
    saving=min(means['policy'],means['MAPPO'])-means[best['id']]
    return dict(accepted=saving>0,option=best['id'],targets=best['proposal_targets'] if saving>0 else {},
        expected_saving=saving,means=means,futures=futures,K=8,horizon_s=300,
        record_sha256=digest(record),contract='inherited-options/W10-setpoint-adapter/observable-only',
        source_prior='uniform32FIT/current-clock inferred integer pseudo-counts; not observed source requests')


def canary(rt,record,result):
    """Same indexed future alone vs batched; no new SUMO or model selection."""
    proposal=record.get('mode')=='proposal'
    acts=proposal_candidates(record) if proposal else candidates(record)
    if proposal and not result.get('futures'):return dict(deferred=True,reason='no available inherited option')
    horizons=(300,) if proposal else HORIZONS
    ns=adapted_futures(horizons);ns['_actions']=acts
    with executor_adapter():single=ns['futures'](rt,dict(record,setting=record.get('setting',0)),indices=(0,))
    futures={'300':result['futures']} if proposal else result['replay']['futures']
    worst=max(abs(single['futures'][str(h)][a['id']][0]['team_delay']-
        futures[str(h)][a['id']][0]['team_delay']) for h in horizons for a in acts)
    if worst>1e-3:raise ValueError('frozen K8 batched-vs-single forecast mismatch '+str(worst))
    return dict(pass_=True,max_team_cost_error=worst,serial_future=0,indices=list(range(8)),new_sumo_sim_s=0)


def cost_adapter(fn,source=None):
    """Add H60 to the qualified reducer, without introspecting exec/linecache.

    The frozen producer execs reference snippets under a shared synthetic
    filename. inspect.getsource can silently retrieve a different snippet.
    Bind explicit qualified source to the actual function bytecode first.
    """
    source=inspect.getsource(fn) if source is None else source
    original=dict(fn.__globals__)
    exec(compile(source,'<coopmwm-qualified-cost-source>','exec'),original)
    def signature(code):
        return (code.co_code,code.co_names,code.co_varnames,
                tuple(signature(x) if isinstance(x,types.CodeType) else x for x in code.co_consts))
    if signature(original[fn.__name__].__code__)!=signature(fn.__code__):
        raise ValueError('qualified cost source/function bytecode mismatch')
    tree=ast.parse(source);count=0
    class Horizons(ast.NodeTransformer):
        def visit_Tuple(self,node):
            nonlocal count
            if all(isinstance(v,ast.Constant) for v in node.elts) and [v.value for v in node.elts]==[90,120,300,600]:
                count+=1
                return ast.copy_location(ast.Tuple([ast.Constant(h) for h in HORIZONS],ast.Load()),node)
            return self.generic_visit(node)
    tree=Horizons().visit(tree);ast.fix_missing_locations(tree)
    if count!=1:raise ValueError('frozen team-cost horizon ABI changed')
    ns=dict(fn.__globals__);exec(compile(tree,'<coopmwm-frozen-cost-horizons>','exec'),ns)
    return ns[fn.__name__]


def evaluate(rt,record):
    validate(record)
    ns=adapted_futures();ns['_actions']=candidates(record)
    adapted=dict(record,setting=0)
    with executor_adapter():
        # Same producer's indexed batching, complete K8 assembled without changing
        # draw identities. Two futures/batch bounds H600 memory under the 8GiB cap.
        chunks=[ns['futures'](rt,adapted,indices=tuple(range(i,i+2))) for i in range(0,8,2)]
    if any(not r['population_audit']['finite'] or not r['population_audit']['conserved'] for r in chunks):
        raise ValueError('WM finite/conservation failure')
    result=dict(futures={str(h):{a['id']:[v for r in chunks for v in r['futures'][str(h)][a['id']]]
                for a in candidates(record)} for h in HORIZONS},chunks=chunks,indices=list(range(8)))
    from .credit import noise
    credits={};variances={}
    for h in HORIZONS:
        f=result['futures'][str(h)]
        draws=[{a:values[k] for a,values in f.items()} for k in range(8)]
        credits[str(h)]=paired(draws);variances[str(h)]=noise(draws)
    return dict(credits=credits,noise=variances,replay=result,record_sha256=digest(record),horizons_s=HORIZONS,
                observation_contract='roster-only/cold-memory/uniform32FIT-source-prior')
