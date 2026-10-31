"""Short sequential, exclusive, synchronous CPU addendum operations."""
from pathlib import Path
import argparse
import json

from remote import PYTHON, REMOTE, ROOT, ssh, stage

OUT = REMOTE + '/addendum_v1'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('inspect', 'stage', 'fit', 'fetch-fit', 'probe', 'fetch-probe'))
    p.add_argument('--code')
    a = p.parse_args()
    if a.command == 'stage':
        print(stage())
    elif a.command == 'inspect':
        print(ssh('''
from pathlib import Path
import os,json,torch
torch.set_num_threads(1)
p=Path('/home/ubuntu/wmdata20261031/jobs/s969400_native/start.pt')
s=torch.load(p,map_location='cpu',weights_only=False)
roots=['/home/ubuntu/rbfit20261031/fit_mixture_v4',
 '/home/ubuntu/rbfit20261031/lane_v1',
 '/home/ubuntu/rbfit20261031/artifacts/rbfit/finite_mixture_v4']
print(json.dumps(dict(load=os.getloadavg(),ids_type=str(type(s['vehicle_ids'])),
 ids=s['vehicle_ids'][:8],artifacts={r:Path(r).is_dir() for r in roots}),default=str))
'''))
    elif a.command.startswith('fetch-'):
        name = 'observed_arrivals.json' if a.command == 'fetch-fit' else 'fitted_source_probe.json'
        text = ssh(f"from pathlib import Path;print((Path({OUT!r})/{name!r}).read_text())")
        target = ROOT/'docs/paper_results/rbunc'/name
        with target.open('x', encoding='utf-8') as stream:
            stream.write(text)
        print('FETCHED', target, len(text))
    else:
        assert a.code and a.code.startswith(REMOTE+'/code/')
        script = 'fit_arrivals.py' if a.command == 'fit' else 'fitted_probe.py'
        print(ssh(f'''
from pathlib import Path
import subprocess,os,json,time
root=Path({OUT!r});assert root.parent.is_dir();root.mkdir(exist_ok=True)
assert os.getloadavg()[0]<24,'insufficient free CPU'
lock=root/{(a.command+'_DISPATCH.json')!r}
with lock.open('x') as f:json.dump(dict(start_unix=time.time(),code={a.code!r},mode={a.command!r}),f)
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',
 PYTHONPATH={a.code!r}+'/src:'+{a.code!r})
with (root/{(a.command+'.log')!r}).open('x') as log:
 child=subprocess.Popen([{PYTHON!r},'-B',{a.code!r}+'/tools/rbunc/{script}',
  '--out',str(root)],cwd={a.code!r},env=env,stdout=log,stderr=subprocess.STDOUT)
 with (root/{(a.command+'_PID.json')!r}).open('x') as f:json.dump(dict(pid=child.pid),f)
 rc=child.wait()
 with (root/{(a.command+'_TERMINAL.json')!r}).open('x') as f:json.dump(dict(pid=child.pid,rc=rc,end_unix=time.time()),f)
print((root/{(a.command+'.log')!r}).read_text())
assert rc==0,rc
''', timeout=600))


if __name__ == '__main__':
    main()
