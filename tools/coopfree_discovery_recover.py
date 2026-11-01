"""CPU-only recovery of completed u001 discovery; no SUMO dispatch or original mutation."""
import json
from tools.coopbandit.ops import remote
from tools.coopfree_continue_ops import ROOT, LOCAL


def recover():
    source='''import pathlib,json,os,sys,time,hashlib,fcntl
os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
r=pathlib.Path(ROOT);old=pathlib.Path('/home/ubuntu/coopfree20261101_build5')
assert r.is_dir() and old.is_dir()
lock=(r/'discovery_recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if os.getloadavg()[0]>12:
 print(json.dumps(dict(deferred='load above12; no job/physics started',load=os.getloadavg())))
 sys.exit(0)
sys.path.insert(0,str(r/'pytools'))
import numpy as np
from tools.coopfree.storage import read,write,completed
from tools.coopfree.protocol import sha,digest
from tools.coopfree.credit import credit
from tools.coopfree.analysis import tape
from tools.coopfree_hold import solo_hold,summary
import torch
torch.set_num_threads(1)
from tools.coopbehav.detectors import detect
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
output=r/'RECOVERED_HOLD_DISCOVERY_001.json'
if output.exists():
 print(json.dumps(dict(already_complete=True,sha256=sha(output))))
 sys.exit(0)
plan=read(old/'HOLD_DISCOVERY_PLAN_001.json')
write(r/f'DISCOVERY_RECOVERY_OWNED_{os.getpid()}.json',dict(pid=os.getpid(),start_ticks=pathlib.Path('/proc/self/stat').read_text().split()[21],sim_s=0,nice=19))
groups=[];x=[]
for i,entry in enumerate(plan):
 event=entry['event'];sample=event['sample'];members=sample['context']['members']
 prefix=f'hold_disc_{i:02}'
 base=completed(old,prefix+'_baseline');joint=completed(old,prefix+'_joint')
 solos=[completed(old,prefix+f'_solo{j}') for j in range(len(members))]
 assert base is not None and joint is not None and all(s is not None for s in solos),'missing closed arm; no retry'
 assert base['sample']==entry['mean'] and joint['sample']==sample
 assert all(s['sample']==solo_hold(entry['mean'],sample,m) for s,m in zip(solos,members))
 assert all(s['prefix_sha256']==digest(event['state']['prefix']) for s in [base,joint]+solos)
 metrics={str(h):credit(base['metrics'][str(h)]['team_delay_vs'],joint['metrics'][str(h)]['team_delay_vs'],[s['metrics'][str(h)]['team_delay_vs'] for s in solos]) for h in (120,300,600)}
 partial=r/f'RECOVERED_HOLD_GROUP_{i:02}.json'
 if partial.exists():
  group=read(partial);assert group['credit']==metrics
 else:
  path=old/'jobs'/(prefix+'_joint')/'observable.jsonl.gz'
  assert sha(path)==joint['trace_sha256']
  actual=tape(path,entry['world']['seed'])
  found=detect(actual)
  group=dict(world=entry['world'],t0=event['time_s'],sample=sample,credit=metrics,
   events=found['events'],coverage=found['coverage'],trace_sha256=sha(path),
   reference_sha256=base['trace_sha256'],joint_id=prefix+'_joint',baseline_id=prefix+'_baseline')
  write(partial,group)
 residual=np.asarray(sample['action'])-np.asarray(sample['context']['anchors'])
 end=str(max(map(int,joint['motions'])))
 common=[v for v in members if v in joint['motions'][end] and v in base['motions'][end]]
 motion=np.asarray([[joint['motions'][end][v][d]-base['motions'][end][v][d] for d in (0,1)] for v in common]).reshape(-1,2)
 movement=motion.mean(0).tolist() if len(motion) else [0.,0.]
 x.append([*residual.mean(0),*residual.std(0),sample['window'],len(members),*movement])
 groups.append(group)
clusters=[]
if len(x)>=3 and len(np.unique(x,axis=0))>=3:
 z=StandardScaler().fit_transform(x)
 for k in range(2,min(8,len(x)-1,len(np.unique(z,axis=0))-1)+1):
  model=KMeans(n_clusters=k,n_init=10,random_state=20261101).fit(z)
  if 1<len(set(model.labels_))<len(z):clusters.append(dict(k=k,silhouette=float(silhouette_score(z,model.labels_)),labels=model.labels_.tolist()))
cluster=max(clusters,key=lambda c:c['silhouette']) if clusters else None
counts={}
for g in groups:
 for event in g['events']:counts[event['behaviour']]=counts.get(event['behaviour'],0)+1
cg=[g['credit']['600']['C_G'] for g in groups]
result=dict(update=1,checkpoint_sha256=sha(old/'policy_hold_u001.pt'),groups=groups,
 group_count=len(groups),D_G=summary([g['credit']['600']['D_G'] for g in groups]),
 sum_D_i=summary([g['credit']['600']['sum_D_i'] for g in groups]),C_G=summary(cg),positive_C_G_share=sum(v>0 for v in cg)/len(cg) if cg else None,
 observational_counts=counts,clustering=cluster,sim_s=0,new_physics=0,
 selection='original first eligible DEV event/world and complete affordable groups, unchanged',
 limitation='Exact HOLD CG for learned group under learned prefix. Observational detectors only: recorded setpoint HOLD is not a native/policy-command release. No paired-release detector claim; unknown clusters are descriptive, not a novel helpful mechanism.')
write(output,result)
print(json.dumps({k:v for k,v in result.items() if k not in ('groups',)}))
'''
    return remote('cpuQ','ROOT='+repr(ROOT)+'\n'+source,timeout=45)


if __name__=='__main__':
    output=recover();print(output)
    (LOCAL/'coopfree/evidence/RECOVERED_DISCOVERY_CPU.json').write_text(output,encoding='utf-8')
