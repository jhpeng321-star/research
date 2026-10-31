"""Reduce exact registered observable tapes remotely; no model or simulator imports."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]/'docs/paper_results/wmbreakdown'
DETECTOR = Path('E:/agent-work/c4-entrant/tools/lrdiag/replay.py')
BAD = ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1',
       'hetdata_truth','vehicle_params','driver_params','vtype')

REMOTE = r'''
import os
os.nice(10)
def safe(p):
 p=Path(p)
 assert not any(x in str(p).lower() for x in BAD),str(p)
 return p
def read(p): return safe(p).read_bytes()
def digest(p): return hashlib.sha256(read(p)).hexdigest()
def load(p): return json.loads(read(p))
def region_masks(s,lane):
 valid=(lane>=0)&(lane<=2)
 return {'upstream':valid&(lane<2)&(s<0),
         'merge':valid&(lane<2)&(s>=0)&(s<=75.44), 'ramp':valid&(lane==2)}
def means(v,mask,idx,n):
 count=np.bincount(idx[mask],minlength=n)[:n]
 total=np.bincount(idx[mask],weights=v[mask].astype(float),minlength=n)[:n]
 return [float(total[i]/count[i]) if count[i] else None for i in range(n)],count.tolist()
rows=[]; checks=[]
for seed in range(LO,HI):
 w=load(Path(ROOT)/f'world_{seed}.json')
 assert set(w)==set(ARMS)
 for arm in ARMS:
  record=w[arm]; ep=safe(Path(ROOT)/'episodes'/record['episode'])
  meta=load(ep/'episode.json'); job=safe(record['job'])
  assert meta['seed']==seed and meta['arm']==arm
  assert meta['source']['initial_state_sha256']==record['initial_state_sha256']
  source=job/'traj_base.npz'; sha=digest(source)
  assert sha==meta['source']['traj_sha256']
  assert digest(job/'result.json')==meta['source']['result_sha256']
  times=np.load(ep/'time_s.npy',allow_pickle=False)
  kin=np.load(ep/'veh_kin.npy',mmap_mode='r',allow_pickle=False)
  present=np.load(ep/'veh_mask.npy',mmap_mode='r',allow_pickle=False).astype(bool)
  present &= kin[:,:,1:4].sum(-1)>0
  lane=kin[:,:,1:4].argmax(-1); pos=kin[:,:,0]; speed=kin[:,:,4]
  assert times[0]==240 and times[-1]==600 and meta['t0_index']==120
  with np.load(source,allow_pickle=False) as f:
   # Explicit observable allowlist: never deserialize the NPZ metadata.
   t,s,v,l=[f[k] for k in ('t','s','v','lane')]
  assert t.min()==300.5 and t.max()==900 and set(np.unique(l))<=set((-1,0,1,2))
  # Restore the t=300 observation omitted by the raw post-step recorder.
  m=present[120]
  t=np.r_[np.full(int(m.sum()),300.),t]
  s=np.r_[pos[120][m],s]; v=np.r_[speed[120][m],v]; l=np.r_[lane[120][m],l]
  clock={}; hashes={}
  for h in (120,300,600):
   p=job/f'cumulative_{h}.json'; hashes[str(h)]=digest(p)
   assert hashes[str(h)]==meta['source']['clocks_sha256'][str(h)]
   c=load(p)
   assert abs(c['team_delay_vs']-record['clocks'][str(h)]['team_delay_vs'])<1e-6
   clock[str(h)]={k:float(c[k]) for k in ('team_delay_vs','road_delay_vs','source_wait_vs')}
   clock[str(h)]['objective_J']=float(load(job/f'objective_{h}.json')['J'])
  base=dict(corpus=CORPUS,split=('reserved8' if 969432<=seed<=969439 else 'dev32' if seed<969440 else 'replication50'),
            seed=seed,arm=arm,demand=meta['demand_vph'],penetration=meta['penetration'],
            initial_sha256=record['initial_state_sha256'],clocks=clock)
  bins={}; maxdiff=0.
  masks=region_masks(pos,lane); raw_masks=region_masks(s,l)
  for region in ('upstream','merge','ramp'):
   prefix=[]
   for j in range(6):
    keep=present[j*20:(j+1)*20]&masks[region][j*20:(j+1)*20]
    val=speed[j*20:(j+1)*20][keep]
    prefix.append(float(val.astype(float).mean()) if len(val) else None)
   use=(t>=300)&(t<900)&raw_masks[region]
   idx=np.floor((t-300)/10).astype(int)
   future,counts=means(v,use,idx,60)
   # Independent converted-vs-raw agreement through 300s; no cohort masks.
   for j in range(30):
    a=120+j*20; b=a+20
    keep=present[a:b]&masks[region][a:b]
    val=speed[a:b][keep]
    cv=float(val.astype(float).mean()) if len(val) else None
    if cv is not None and future[j] is not None: maxdiff=max(maxdiff,abs(cv-future[j]))
    else: assert cv is future[j]
   bins[region]={'speed':prefix+future,'post_counts':counts,
                 'events':{str(h):breakdown(prefix+future[:h//10]) for h in (120,300,600)}}
  assert maxdiff<1e-5,(seed,arm,maxdiff)
  rows.append(dict(**base,bins=bins))
  checks.append(dict(seed=seed,arm=arm,trajectory_sha256=sha,clocks_sha256=hashes,
    episode_sha256=digest(ep/'episode.json'),converted_raw_max_mps=maxdiff))
print(json.dumps(dict(corpus=CORPUS,root=ROOT,episodes=rows,checks=checks),allow_nan=False))
'''

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    text=DETECTOR.read_text()
    tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='breakdown')
    detector=ast.get_source_segment(text,node)
    provenance={'detector_source':str(DETECTOR),'detector_sha256':hashlib.sha256(detector.encode()).hexdigest(),
                'detector_function':detector,'analysis':'S-WMBREAKDOWN','hosts':[]}
    for host, root, python, lo, hi, corpus, arms in (
        ('cpuQ','/home/ubuntu/wmdata20261031/views','/home/ubuntu/miniconda3/bin/python',969400,969440,'WMDATA',
         ['native','ctrl0','mappo','pi','circles']),
        ('gpuR','/root/wmdata2/views','/root/miniconda3/bin/python',969550,969600,'WMDATA2',
         ['native','ctrl0','mappo','pi','circles','generic_random'])):
        target=OUT/f'{corpus.lower()}_observables.json'
        if target.exists():
            data=json.loads(target.read_text())
            assert len(data['episodes'])==(hi-lo)*len(arms)
            print('Existing verified receipt',target)
        else:
            header='from pathlib import Path\nimport json, hashlib\nimport numpy as np\n'
            header+='\n'.join(f'{k}={v!r}' for k,v in dict(ROOT=root,LO=lo,HI=hi,CORPUS=corpus,ARMS=arms,BAD=BAD).items())+'\n'
            p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,python+' -B -'],
                input=header+detector+'\n'+REMOTE,text=True,capture_output=True,timeout=600)
            if p.returncode: raise RuntimeError(f'{host}: {p.stderr}\n{p.stdout[:1000]}')
            data=json.loads(p.stdout)
            assert len(data['episodes'])==(hi-lo)*len(arms)
            target.write_text(json.dumps(data,indent=1,allow_nan=False))
        provenance['hosts'].append(dict(host=host,root=root,episodes=len(data['episodes']),
            local_receipt_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        print(host,len(data['episodes']),'episodes; max converted/raw difference',
              max(c['converted_raw_max_mps'] for c in data['checks']))
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=1))

if __name__=='__main__': main()
