"""Observable FIT kinematic identity audit, not a new WM/SUMO rollout."""
import json

from .operate import RECEIPTS, remote


def main():
    out = RECEIPTS/'POSITION_INPUT_AUDIT.json'
    assert out.parent.is_dir() and not out.exists()
    code = r'''
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
assert os.getloadavg()[0]<20
os.nice(19)
import pathlib,re,json,math,numpy as np
from collections import Counter
root=pathlib.Path('/home/ubuntu/wmdata20261031/views/episodes')
tot=Counter(); rows=[]
for world in range(969400,969432):
    for p in sorted(root.glob(f'w{world}m*')):
        if not re.fullmatch(rf'w{world}m[0-9]+p[0-9]+t300__(native|ctrl0|mappo|pi|circles)',p.name): continue
        meta=json.loads((p/'episode.json').read_text()); assert int(meta['seed'])==world
        a={name:np.load(p/(name+'.npy'),allow_pickle=False,mmap_mode='r')
           for name in ('time_s','veh_kin','veh_mask','vid','is_cav')}
        t=a['time_s']; ix=np.flatnonzero((t>=240)&(t<=360)); lo,hi=int(ix[0]),int(ix[-1])+1
        k=np.asarray(a['veh_kin'][lo:hi],dtype=np.float64); m=a['veh_mask'][lo:hi].astype(bool)
        ids=a['vid'][lo:hi]; cav=a['is_cav'][lo:hi].astype(bool)
        dt=np.diff(t[lo:hi])[:,None]; assert np.all(dt==.5)
        normal=k[:,:,1:4].sum(-1)>0; lane=k[:,:,1:4].argmax(-1)
        segment=np.where(k[:,:,0]<0,0,np.where(k[:,:,0]<=75.44,1,2))
        v0,v1=k[:-1,:,4],k[1:,:,4]
        take=(m[:-1]&m[1:]&(ids[:-1]==ids[1:])&normal[:-1]&normal[1:]
              &~cav[:-1]&(lane[:-1]<2)&(lane[:-1]==lane[1:])
              &(segment[:-1]==segment[1:])&(v0>0)&(v1>0)&(v0!=v1))
        ds=k[1:,:,0]-k[:-1,:,0]
        e=(ds-v1*dt)[take]; b=(ds-.5*(v0+v1)*dt)[take]; w=((v1-v0)**2)[take]
        row=dict(n=int(take.sum()),euler_abs=float(np.abs(e).sum()),ballistic_abs=float(np.abs(b).sum()),
            euler_sse=float((e*e).sum()),ballistic_sse=float((b*b).sum()),
            dv_weight=float(w.sum()),euler_weighted_sse=float((e*e*w).sum()),
            ballistic_weighted_sse=float((b*b*w).sum()),euler_closer=int((np.abs(e)<np.abs(b)).sum()),
            ballistic_closer=int((np.abs(b)<np.abs(e)).sum()))
        tot.update(row); rows.append(dict(episode=p.name,statistics=row))
assert len(rows)==160
n=tot['n']; w=tot['dv_weight']
print(json.dumps(dict(status='COMPLETE',episodes=160,worlds=32,statistics=tot,
    cohort='240..360s, HDV mainline, same-ID/lane/normal edge, both speeds positive and changed',
    euler_mae_m=tot['euler_abs']/n,ballistic_mae_m=tot['ballistic_abs']/n,
    euler_rmse_m=math.sqrt(tot['euler_sse']/n),ballistic_rmse_m=math.sqrt(tot['ballistic_sse']/n),
    euler_dv_weighted_rmse_m=math.sqrt(tot['euler_weighted_sse']/w),
    ballistic_dv_weighted_rmse_m=math.sqrt(tot['ballistic_weighted_sse']/w),
    wm_inference=False,sumo_simulation=False,by_episode=rows)))
'''
    result = json.loads(remote(code, timeout=90))
    result['public_source_reference'] = 'SUMO1.27.1 MSVehicle.cpp:4954-4967'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='by_episode'}, indent=2))


if __name__ == '__main__':
    main()
