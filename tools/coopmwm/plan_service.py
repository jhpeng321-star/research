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
    allowed|={k for k in ('policy_state','search_learning') if k in record}
    if set(record)!=allowed:raise ValueError('observable plan request whitelist')
    if 'history' in record and 'warm_memory' in record:
        raise ValueError('ambiguous warm memory payload')
    if binding.get('memory_mode')=='warm' and not {'history','warm_memory'} & set(record):
        raise ValueError('warm worker requires an explicit memory payload (empty allowed)')
    if 'history' in record:
        from .warm_runtime import prepare
        return prepare(rt,record)
    from .speed2_memory import WarmQuery
    return WarmQuery.from_record(rt,record)


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
    def emit(event,**kw):print('COOPMWM_JSON '+json.dumps(dict(event=event,**kw)),flush=True)
    ready=dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
    write(root/'SERVICE_OWNER.json',ready);emit('READY',**ready)
    for line in sys.stdin:
        if time.time()>binding['deadline_unix']:raise ValueError('registered GPU deadline')
        r=json.loads(line);key=r['state_id'];dest=root/'forecasts'/(key+'.json')
        if dest.exists():raise ValueError('duplicate inference attempt; inspect owned receipt')
        diagnostic=r.get('mode')=='diagnostic'
        memory=request_memory(rt,r,binding)
        from tools.coopnative.policy import encode
        encode(r['rows'])
        ids={v['vid'] for v in r['rows'] if v['is_cav']}
        if not set(r['commands'])<=ids or not set(r['group'])<=ids:raise ValueError('non-CAV request')
        start=time.monotonic()
        if diagnostic:
            from .accel_diagnostic import evaluate as credit
            result=credit(rt,r,speed2=speed2,memory=memory,transfers=transfers)
        else:
            from .mppi_runtime import plan
            result=plan(rt,r,memory,binding)
        payload=dict(key=key,result=result,wall_s=time.monotonic()-start,unix=time.time())
        write(dest,payload);emit('RESULT',**payload)


if __name__=='__main__':main(Path(sys.argv[1]))
