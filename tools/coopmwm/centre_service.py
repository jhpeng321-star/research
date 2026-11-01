"""Baseline-only warm trajectory server for R; no candidate score/search API."""
import json
import os
from pathlib import Path
import sys
import time
from .storage import read,write
from .protocol import digest,sha


def result(rt,record,memory):
    from .mppi_runtime import imagined_centres
    policy,unused,receipt=imagined_centres(rt,record,memory,speed2=True,transfers=True)
    return dict(record_sha256=digest(record),policy_traj=policy,receipt=receipt,candidate_scores=0)


def main(root):
    import fcntl
    lock=(root/'CENTRE.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from tools.wmteach import runtime
    from .plan_service import configured_runtime,request_memory
    from .wm import cost_adapter
    binding=read(root/'CENTRE_BINDING.json')
    for name,h in binding['hashes'].items():assert sha(root/name)==h,name
    assert binding['cpu']['pass_'] and binding['core_gpu_audit']['pass_']
    # Launch only after B's early registered pause; never compete with B search.
    rt=configured_runtime(runtime.bootstrap,.25)
    rt.frozen.costs=cost_adapter(rt.frozen.costs,read(root/'FROZEN_COSTS.json')['code'],horizons=(300,))
    owner=dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
    write(root/'CENTRE_OWNER.json',owner)
    print('COOPMWM_JSON '+json.dumps(dict(event='READY',**owner)),flush=True)
    for line in sys.stdin:
        assert time.time()<binding['deadline_unix']
        record=json.loads(line);key=record['state_id'];target=root/'forecasts'/(key+'.json')
        assert not target.exists(),'duplicate centre attempt'
        memory=request_memory(rt,record,dict(memory_mode='warm'))
        begin=time.monotonic();value=result(rt,record,memory)
        payload=dict(event='RESULT',key=key,result=value,wall_s=time.monotonic()-begin,unix=time.time())
        write(target,payload);print('COOPMWM_JSON '+json.dumps(payload),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
