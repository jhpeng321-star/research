"""Short sequential, exclusive, synchronous CPU addendum operations."""
from pathlib import Path
import argparse
import json

from remote import PYTHON, REMOTE, ROOT, ssh, stage

OUT = REMOTE + '/addendum_v1'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('inspect', 'stage', 'fit', 'fetch-fit', 'probe', 'status', 'fetch-probe'))
    p.add_argument('--code')
    a = p.parse_args()
    if a.command == 'stage':
        print(stage())
    elif a.command == 'inspect':
        print(ssh('''
from pathlib import Path
import os,json,torch,numpy as np
torch.set_num_threads(1)
p=Path('/home/ubuntu/wmdata20261031/jobs/s969400_native/start.pt')
s=torch.load(p,map_location='cpu',weights_only=False)
roots=['/home/ubuntu/rbfit20261031/fit_mixture_v4',
 '/home/ubuntu/rbfit20261031/lane_v1',
 '/home/ubuntu/rbfit20261031/artifacts/rbfit/finite_mixture_v4']
view=Path('/home/ubuntu/wmdata20261031/views')
ep=json.loads((view/'world_969400.json').read_text())['native']['episode']
assert ep.startswith('w969400m') and ep.endswith('__native') and '/' not in ep
base=view/'episodes'/ep
times=np.load(base/'time_s.npy',allow_pickle=False)
ti=int(np.flatnonzero(times==300)[0])
vid=np.load(base/'vid.npy',mmap_mode='r',allow_pickle=False)[ti]
mask=np.load(base/'veh_mask.npy',mmap_mode='r',allow_pickle=False)[ti].astype(bool)
kin=np.load(base/'veh_kin.npy',mmap_mode='r',allow_pickle=False)[ti]
keep=s['state']['present'][0].nonzero().flatten().tolist()[:5]
print(json.dumps(dict(load=os.getloadavg(),ids_type=str(type(s['vehicle_ids'])),
 ids=s['vehicle_ids'][:8],artifacts={r:Path(r).is_dir() for r in roots},
 present_slots=keep,start_s=s['state']['s_m'][0,keep].tolist(),
 tape_vid=vid[mask][:5].tolist(),tape_s=kin[mask,0][:5].tolist(),
 tape_shape=list(kin.shape)),default=str))
'''))
    elif a.command == 'status':
        print(ssh(f'''
from pathlib import Path
import os,json
root=Path({OUT!r})
print(json.dumps(dict(load=os.getloadavg(),complete=[s for s in range(969400,969432)
 if (root/f'fitted_w{{s}}.json').exists()])))
for i in range(4):
 p=root/f'probe_worker{{i}}.log'
 if p.exists():print(p.name,p.read_text())
p=root/'probe_TERMINAL.json'
if p.exists():print(p.read_text())
'''))
    elif a.command.startswith('fetch-'):
        name = 'observed_arrivals.json' if a.command == 'fetch-fit' else 'fitted_source_probe.json'
        if a.command == 'fetch-probe':
            text = ssh(f'''
from pathlib import Path
import json
root=Path({OUT!r})
terminal=json.loads((root/'probe_TERMINAL.json').read_text())
assert all(r['rc']==0 for r in terminal['results'])
records=[json.loads((root/f'fitted_w{{s}}.json').read_text()) for s in range(969400,969432)]
receipts={{n:json.loads((root/n).read_text()) for n in
 ['fit_DISPATCH.json','fit_PID.json','fit_TERMINAL.json','probe_DISPATCH.json','probe_PID.json','probe_TERMINAL.json']
 +[f'probe_worker{{i}}_PID.json' for i in range(4)]}}
print(json.dumps(dict(records=records,receipts=receipts)))
''')
        else:
            text = ssh(f"from pathlib import Path;print((Path({OUT!r})/{name!r}).read_text())")
        target = ROOT/'docs/paper_results/rbunc'/name
        with target.open('x', encoding='utf-8') as stream:
            stream.write(text)
        print('FETCHED', target, len(text))
    else:
        assert a.code and a.code.startswith(REMOTE+'/code/')
        script = 'fit_arrivals.py' if a.command == 'fit' else 'fitted_panel.py'
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
  '--out',str(root)],cwd={a.code!r},env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 with (root/{(a.command+'_PID.json')!r}).open('x') as f:json.dump(dict(pid=child.pid),f)
 if {a.command!r}=='fit':
  rc=child.wait()
  with (root/'fit_TERMINAL.json').open('x') as f:json.dump(dict(pid=child.pid,rc=rc,end_unix=time.time()),f)
  print((root/'fit.log').read_text())
  assert rc==0,rc
 else:print(json.dumps(dict(pid=child.pid,status='dispatched_once',workers=4)))
''', timeout=600))


if __name__ == '__main__':
    main()
