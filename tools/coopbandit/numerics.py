"""CPU-only frozen-executor bitwise comparison on synthetic observable traffic."""
import argparse
import json
from .ops import remote,HOSTS
from .dispatch import EVIDENCE,once


def probe(host, mode='default'):
    config=HOSTS[host]
    envs={'default':{},'compatible':{'MKL_CBWR':'COMPATIBLE'},
          'avx2':{'MKL_CBWR':'AVX2','ATEN_CPU_CAPABILITY':'avx2','ONEDNN_MAX_CPU_ISA':'AVX2'},
          'sse':{'MKL_CBWR':'SSE4_2','ATEN_CPU_CAPABILITY':'default','ONEDNN_MAX_CPU_ISA':'SSE41'}}[mode]
    code=f'''import os,sys,json,random,hashlib
os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
os.environ.update({envs!r})
sys.path[:0]=[{config['root']!r}+'/pytools']+{config['deps']!r}
from tools.leverprobe.backend import setup,exact
setup({config['runtime']!r},{config['checkpoint']!r})
from tools.coopbandit.protocol import canonical,digest
from wms.control.policy import load_policy_fn
import torch
from unittest.mock import patch
hashes={{}}
with patch('subprocess.Popen',side_effect=AssertionError('SUMO forbidden in numerical test')):
 for n in (8,32,64,128,192):
  exe=load_policy_fn({config['checkpoint']!r},device='cpu');exe.reset()
  rng=random.Random(31965+n);out=[]
  for t in range(20):
   s=torch.tensor([[rng.uniform(-100,75) for _ in range(n)]],dtype=torch.float32)
   v=torch.tensor([[rng.uniform(0,25) for _ in range(n)]],dtype=torch.float32)
   state=dict(s_m=s,v_mps=v,a_mps2=torch.zeros(1,n),lane=torch.tensor([[i%3 for i in range(n)]]),
    present=torch.ones(1,n,dtype=torch.bool),is_cav=torch.tensor([[i%5==0 for i in range(n)]]),
    length_m=torch.full((1,n),5.),junction=torch.zeros(1,n,dtype=torch.bool))
   cmd,lanes=exe(state,with_lanes=True)
   out.append(digest(exact(dict(command=cmd,memory=vars(exe.memory),dwell=vars(exe.dwell)))))
  hashes[str(n)]=out
print(json.dumps(dict(host={host!r},mode={mode!r},environment={envs!r},hashes=hashes,sim_s=0)))
'''
    result=json.loads(remote(host,code,timeout=30))
    once(EVIDENCE/f'NUMERICS_{host}_{mode}.json',result)
    print(json.dumps(dict(host=host,mode=mode,hashes={n:row[0] for n,row in result['hashes'].items()},sim_s=0)))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('host',choices=tuple(HOSTS));p.add_argument('--mode',default='default',choices=('default','compatible','avx2','sse'))
    a=p.parse_args();probe(a.host,a.mode)
