"""MAIN23:25 recovery operations; preserve every original deployment/STOP."""
import argparse
import json

from .ops import EVIDENCE, ssh
from .storage import write


def inspect():
    out = {}
    for host in ('cpuQ', 'gpuA', 'gpuR'):
        out[host] = ssh(host, '''import json,os,subprocess
from pathlib import Path
out=dict(load=os.getloadavg())
if HOST != 'cpuQ':
 out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 out['base_exists']=Path('/root/c1switch20261031').exists()
 out['wm_source_exists']=Path('/root/c1switch20261031/main_v7').exists()
 out['inference_base_exists']=Path('/root/autodl-tmp/c4/speedhead20261030').exists()
out['recovery_exists']=Path('/home/ubuntu/c1switch20261031/recovery_v8' if HOST=='cpuQ' else '/root/c1switch20261031/recovery_v8').exists()
print(json.dumps(out))
'''.replace('HOST', repr(host)))
    write(EVIDENCE/'V8_INSPECT.json', out)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('inspect',))
    globals()[parser.parse_args().mode]()
