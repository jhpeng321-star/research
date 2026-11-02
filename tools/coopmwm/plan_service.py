"""Persistent FULL-v4 acceleration planner, qualified once per worker."""
import json
import os
from pathlib import Path
import sys
import time
from .storage import read,write
from .protocol import sha,digest


def configured_runtime(bootstrap,fraction):
    # The inherited bootstrap installs its own 8GiB cap. Worker-specific limits
    # must be applied AFTER it, otherwise concurrent service limits are ignored.
    rt=bootstrap()
    import torch
    torch.cuda.set_per_process_memory_fraction(fraction)
    return rt


def batch_width(binding,k):
    widths=binding.get('batch_widths',binding.get('batch_width_by_K',{}))
    return int(widths.get(str(k),binding.get('batch_width',4)))


def request_memory(rt,record,binding):
    diagnostic=record.get('mode')=='diagnostic'
    allowed={'state_id','world_id','time_s','rows','group','commands','policy_residuals'}|({'mode','plan'} if diagnostic else set())
    if 'warm_memory' in record:allowed.add('warm_memory')
    if 'history' in record:allowed.add('history')
    allowed|={k for k in ('policy_state','search_learning','committed_plans') if k in record}
    if set(record)!=allowed:raise ValueError('observable plan request whitelist')
    from .committed import validate
    validate(record.get('committed_plans',[]),record['group'])
    if 'history' in record and 'warm_memory' in record:
        raise ValueError('ambiguous warm memory payload')
    if binding.get('memory_mode')=='warm' and not {'history','warm_memory'} & set(record):
        raise ValueError('warm worker requires an explicit memory payload (empty allowed)')
    if 'history' in record:
        from .warm_runtime import prepare
        return prepare(rt,record)
    from .speed2_memory import WarmQuery
    return WarmQuery.from_record(rt,record)


def learned_gate_problems(search_fn,kwargs_fn,record,group,policy_traj,bounds,regime,search_config=None,k=8):
    """MAIN decision (c), harden-2: dry learned search on a fixture with a constant fake
    evaluator (no GPU, no SUMO). Any exception (e.g. TypeError for a missing ``warm_start``
    or a duplicate ``config=``), a per-query decline or fewer than k ranked candidates is
    a problem, so a worker can never go READY and then silently decline every query."""
    def evaluate(rec,actions,K,H):
        # Candidates strictly beat both comparators, so an accepted search must rank a topk.
        return {a['id']:[float(i)+(100. if a['id'] in ('policy','MAPPO') else 0.) for i in range(K)] for a in actions}
    try:
        kw=dict(kwargs_fn(record,group,policy_traj,bounds,regime))
        config=dict(search_config or {},**kw.pop('config',{}))
        result=search_fn(record,group,policy_traj,bounds,evaluate,regime,config=config,
            seed=int(digest(record)[:15],16),**kw)
    except Exception as error:
        return ['learned search raised '+repr(error)]
    problems=[]
    if result.get('query_error'):problems.append('learned search returned a per-query decline')
    from .mppi_search import REGISTERED
    top_k=int(dict(REGISTERED,**config)['top_k'])
    if top_k<k:problems.append(f'learned config top_k={top_k} < verification k={k}')
    if not isinstance(result.get('topk'),list):problems.append('learned search returned no ranked topk list')
    return problems


def learned_gate(binding,problems):
    """Fail loudly BEFORE READY when the learned search is bound but not conformant."""
    if binding.get('learned_search') and problems:
        raise ValueError('learned-search startup gate failed: '+'; '.join(problems))


def startup_learned_problems(root,binding,rt):
    """Static interface checks + one dry learned search on the worker's own INPUTS fixture."""
    from . import mppi_search
    from .sumo_verify import resume_preflight
    spec=binding['learned_search']
    problems=resume_preflight(dict(k=spec.get('k',8),phase='sumo_verification_k8'),
        binding.get('search_config'),learned=True)
    try:
        from .plan_proposer import learned_search_kwargs,load_checkpoint
        proposer=load_checkpoint(spec['checkpoint'])
    except Exception as error:
        return problems+['learned proposer unavailable: '+repr(error)]
    source=read(root/'INPUTS.json')['records'][0]
    group=next(g for g in mppi_search.group_components(source['rows']) if g['members']==source['group'])
    bounds=(float(rt.mp.policy.accel_min),float(rt.mp.policy.accel_max))
    policy=[[0.]*600 for _ in group['members']]
    return problems+learned_gate_problems(mppi_search.search,
        lambda r,g,pt,b,reg:learned_search_kwargs(r,g,proposer,reg,mode=spec.get('mode','warm'),policy_traj=pt,bounds=b),
        source,group,policy,bounds,'ordinary',binding.get('search_config'),k=spec.get('k',8))


def main(root):
    import fcntl
    lock=(root/'SERVICE.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    binding=read(root/'GPU_BINDING.json')
    for name,h in binding['hashes'].items():
        if sha(root/name)!=h:raise ValueError('immutable GPU source mismatch '+name)
    assert binding['qualification']['pass_']
    import torch
    from tools.wmteach.runtime import bootstrap
    from .wm import cost_adapter
    from .plan_wm import evaluate
    rt=configured_runtime(bootstrap,binding.get('memory_fraction',.65))
    rt.frozen.costs=cost_adapter(rt.frozen.costs,read(root/'FROZEN_COSTS.json')['code'],
        horizons=(60,120,300,360,420,600))
    speed2=binding.get('speed2',False)
    transfers=binding.get('speed2_transfers',False)
    if speed2 and not binding.get('startup_qualification'):
        raise ValueError('speed2 requires source-bound startup qualification')
    if binding.get('startup_qualification'):
        # Once per persistent worker, never per query. Same registered geometry
        # and source-bound full roster as the concurrency qualification.
        from .plans import baseline_actions,family_seeds
        source=read(root/'INPUTS.json')['records'][0]
        commands={r['vid']:0. for r in source['rows'] if r['is_cav']}
        record=dict(state_id=source['state_id'],world_id=source['world_id'],rows=source['rows'],group=source['group'],
            time_s=source['time_s'],commands=commands,policy_residuals={v:-.25 for v in commands})
        bounds=(float(rt.mp.policy.accel_min),float(rt.mp.policy.accel_max))
        acts=baseline_actions(record)+family_seeds(record['rows'],record['group'],commands,bounds)[:2]
        for k,h in ((4,120),(8,300)):
            width=batch_width(binding,k)
            if speed2:
                from .speed2_runtime import qualify
                receipt=dict(qualify(rt,record,acts,k,h,width,transfers=transfers),pid=os.getpid())
            else:
                ref=evaluate(rt,record,acts,k,h,chunk_actions=1,trim=False,pad_actions=width)
                batch=evaluate(rt,record,acts,k,h,chunk_actions=width,trim=False,pad_actions=width)
                error=max(abs(a['team_delay']-b['team_delay']) for key in ref for a,b in zip(ref[key],batch[key]))
                receipt=dict(pass_=error<=.001,max_cost_error=error,K=k,H=h,width=width,pid=os.getpid())
            write(root/f'STARTUP_PARITY_K{k}_H{h}.json',receipt)
            if not receipt['pass_']:raise ValueError('persistent worker batch parity failed')
        if speed2 and binding.get('memory_mode')=='warm':
            from .speed2_memory import native_fixture
            warm_record,memory=native_fixture(rt)
            warm_acts=baseline_actions(warm_record)
            for k,h in ((4,120),(8,300)):
                width=batch_width(binding,k)
                receipt=dict(qualify(rt,warm_record,warm_acts,k,h,width,memory=memory,transfers=transfers),pid=os.getpid(),memory=memory.receipt)
                write(root/f'STARTUP_WARM_PARITY_K{k}_H{h}.json',receipt)
                if not receipt['pass_']:raise ValueError('persistent worker warm speed2 parity failed')
    if binding.get('learned_search'):
        learned_gate(binding,startup_learned_problems(root,binding,rt))
    def emit(event,**kw):print('COOPMWM_JSON '+json.dumps(dict(event=event,**kw)),flush=True)
    ready=dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
    write(root/'SERVICE_OWNER.json',ready);emit('READY',**ready)
    for line in sys.stdin:
        if time.time()>binding['deadline_unix']:raise ValueError('registered GPU deadline')
        r=json.loads(line);key=r['state_id'];dest=root/'forecasts'/(key+'.json')
        if dest.exists():raise ValueError('duplicate inference attempt; inspect owned receipt')
        diagnostic=r.get('mode')=='diagnostic'
        centre=r.get('mode')=='centre'
        if centre:r={k:v for k,v in r.items() if k!='mode'}
        from .query_errors import execute
        def prepared(r=r):
            # Per-request validation (whitelist, warm history, roster) is a
            # per-query fault for plan/centre requests: decline, never die.
            memory=request_memory(rt,r,binding)
            from tools.coopnative.policy import encode
            encode(r['rows'])
            ids={v['vid'] for v in r['rows'] if v['is_cav']}
            if not set(r['commands'])<=ids or not set(r['group'])<=ids:raise ValueError('non-CAV request')
            return memory
        if diagnostic:memory=prepared()
        start=time.monotonic()
        if centre:
            from .centre_service import result as centre_result
            result=execute(r,lambda:centre_result(rt,r,prepared()))
            if result.get('query_error'):result.update(unavailable=True,candidate_scores=0)
        elif diagnostic:
            from .accel_diagnostic import evaluate as credit
            result=credit(rt,r,speed2=speed2,memory=memory,transfers=transfers)
        else:
            from .mppi_runtime import plan
            result=execute(r,lambda:plan(rt,r,prepared(),binding))
        payload=dict(key=key,result=result,wall_s=time.monotonic()-start,unix=time.time())
        write(dest,payload);emit('RESULT',**payload)


if __name__=='__main__':main(Path(sys.argv[1]))
