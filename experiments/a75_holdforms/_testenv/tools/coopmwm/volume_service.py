"""Registered multi-window GPU service; once-per-worker parity and private draws."""
import contextlib
import json
import os
from pathlib import Path
import sys
import time
from .storage import read,write


def main(root):
    os.nice(19)
    from .protocol import sha
    identity=read(root/'GPU_BINDING.json')
    for name,h in identity['hashes'].items():assert sha(root/name)==h,name
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from tools.wmteach.runtime import bootstrap
        from .wm import cost_adapter
        rt=bootstrap();torch.cuda.set_per_process_memory_fraction(.65)
        rt.frozen.costs=cost_adapter(rt.frozen.costs,read(root/'FROZEN_COSTS.json')['code'])
    owner=dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
               boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    write(root/'GPU_READY.json',owner)
    print('COOPMWM_JSON '+json.dumps(dict(event='READY',owner=owner)),flush=True)
    for line in sys.stdin:
        assert time.time()<identity['deadline_unix'],'original GPU wall cap reached'
        records=json.loads(line)['batch'];assert 1<=len(records)<=identity['batch_windows']
        start=time.monotonic()
        with contextlib.redirect_stdout(sys.stderr):
            from .wm import validate,proposal_result,canary
            from .fused import futures
            for r in records:
                validate(r)
                used=sum(1 for _ in root.glob('forecasts/*/CLAIM.json'))
                assert used<identity['proposal_cap'],'registered proposal cap reached'
                assert not (root/'forecasts'/r['state_id']/'CLAIM.json').exists(),'no duplicate inference'
                write(root/'forecasts'/r['state_id']/'CLAIM.json',dict(owner=owner,unix=time.time()))
            values=futures(rt,records,tuple(range(identity['K'])),trim_absent=True)
            results=[]
            for r,v in zip(records,values):
                assert v['population_audit']['finite'] and v['population_audit']['conserved']
                results.append(proposal_result(r,v['futures']['300']))
            if not (root/'GPU_PARITY.json').exists():
                checks=[canary(rt,r,v) for r,v in zip(records,results)]
                write(root/'GPU_PARITY.json',dict(pass_=all(c['pass_'] for c in checks),checks=checks,new_sumo_sim_s=0))
            for r,v in zip(records,results):write(root/'forecasts'/r['state_id']/'RESULT.json',v)
        print('COOPMWM_JSON '+json.dumps(dict(event='BATCH_RESULT',results=results,
            keys=[r['state_id'] for r in records],wall_s=time.monotonic()-start)),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
