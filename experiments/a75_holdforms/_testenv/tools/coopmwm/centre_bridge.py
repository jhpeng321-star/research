"""R-only centre queue, admitted after B-u4; no candidate scoring or GPU competition."""
import json
import subprocess
import time
from . import ops
from .storage import read,write
from .plan_bridge import Service as OriginalService
from .warm_deploy import ROOT

GPU_ROOT='/root/coopmwm20261102_warmcentre1'


class Service(OriginalService):
    def __init__(self):
        self.host='gpuA'
        command=f'nice -n 19 /root/miniconda3/bin/python -B -u -c "import sys;sys.path.insert(0,\'{GPU_ROOT}/pytools\');from pathlib import Path;from tools.coopmwm.centre_service import main;main(Path(\'{GPU_ROOT}\'))"'
        log=(ops.EVIDENCE/'WARM_CENTRE_GPU.log').open('x',encoding='utf-8')
        self.process=subprocess.Popen(['ssh','-o','BatchMode=yes','gpuA',command],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1)
        write(ops.EVIDENCE/'WARM_CENTRE_SSH.json',dict(pid=self.process.pid,unix=time.time()))
        write(ops.EVIDENCE/'WARM_CENTRE_READY.json',self.receive('READY'))


def run():
    # B pause is a scheduling rule, not an outcome-dependent choice of plans.
    service=None
    try:
        while True:
            state=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({ROOT!r});assert r.is_dir()
records=[json.loads(p.read_text()) for p in (r/'centre_requests').glob('*.json') if not (r/'centre_responses'/p.name).exists()]
print(json.dumps(dict(stop=(r/'STOP.json').exists() or ((r/'WARM_R_PAUSED.json').exists() and not records),
 B_paused=(r/'WARM_B_PAUSED.json').exists(),records=records)))''',timeout=30))
            if state['stop']:break
            if not state['B_paused']:time.sleep(20);continue
            if service is None:service=Service()
            if not state['records']:time.sleep(2);continue
            record=sorted(state['records'],key=lambda r:r['state_id'])[0];value=service.call(record)
            ops.ssh.remote('cpuQ',f'''import pathlib,json,sys
r=pathlib.Path({ROOT!r});sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.storage import write
write(r/'centre_responses'/{(record['state_id']+'.json')!r},json.loads({json.dumps(value['result'])!r}))
print('persisted')''',timeout=40)
    except BaseException as error:
        write(ops.EVIDENCE/'WARM_CENTRE_FAILED.json',dict(error=repr(error),unix=time.time()));raise
    finally:
        if service:service.process.stdin.close()


if __name__=='__main__':run()
