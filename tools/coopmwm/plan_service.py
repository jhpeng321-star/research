"""Persistent FULL-v4 acceleration planner, qualified once per worker."""
import json
import os
from pathlib import Path
import sys
import time
from .storage import read,write
from .protocol import sha,digest


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
    from .plans import search
    rt=bootstrap();torch.cuda.set_per_process_memory_fraction(.65)
    rt.frozen.costs=cost_adapter(rt.frozen.costs,read(root/'FROZEN_COSTS.json')['code'])
    def emit(event,**kw):print('COOPMWM_JSON '+json.dumps(dict(event=event,**kw)),flush=True)
    ready=dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
    write(root/'SERVICE_OWNER.json',ready);emit('READY',**ready)
    for line in sys.stdin:
        if time.time()>binding['deadline_unix']:raise ValueError('registered GPU deadline')
        r=json.loads(line);key=r['state_id'];dest=root/'forecasts'/(key+'.json')
        if dest.exists():raise ValueError('duplicate inference attempt; inspect owned receipt')
        diagnostic=r.get('mode')=='diagnostic'
        allowed={'state_id','world_id','time_s','rows','group','commands','policy_residuals'}|({'mode','plan'} if diagnostic else set())
        if set(r)!=allowed:raise ValueError('observable plan request whitelist')
        from tools.coopnative.policy import encode
        encode(r['rows'])
        ids={v['vid'] for v in r['rows'] if v['is_cav']}
        if not set(r['commands'])<=ids or not set(r['group'])<=ids:raise ValueError('non-CAV request')
        start=time.monotonic()
        if diagnostic:
            from .accel_diagnostic import evaluate as credit
            result=credit(rt,r)
        else:
            result=search(r,(float(rt.mp.policy.accel_min),float(rt.mp.policy.accel_max)),
                lambda a,k,h:evaluate(rt,r,a,k,h,chunk_actions=34 if k==4 else 10,trim=False,pad_actions=34 if k==4 else 10))
        payload=dict(key=key,result=result,wall_s=time.monotonic()-start,unix=time.time())
        write(dest,payload);emit('RESULT',**payload)


if __name__=='__main__':main(Path(sys.argv[1]))
