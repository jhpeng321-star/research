"""Summarize existing TRAIN action/kinematic arrays; no simulator imports."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('/root/main14_loop/data/pairs_c00.a1')
OUT=Path('/root/c0diag_s1')
entries=json.loads((ROOT/'C4_EPISODES_MANIFEST.json').read_text())['episodes']
rows=[]
def stats(a):
    a=np.asarray(a); a=a[np.isfinite(a)]
    return {'n':int(a.size),'mean':float(a.mean()) if a.size else None,
            'std':float(a.std()) if a.size else None,
            'q':np.quantile(a,[0,.1,.5,.9,1]).tolist() if a.size else []}
for e in entries:
    assert 968129 <= e['seed'] <= 968141
    d=ROOT/e['dir']
    load=lambda n:np.load(d/(n+'.npy'),mmap_mode='r',allow_pickle=False)
    kin,p,cav,vid,t,ct=(load(n) for n in ('veh_kin','veh_mask','is_cav','vid','time_s','control_time_s'))
    ctrl,req,issued,tgt,lr,lc=(load(n) for n in ('controller','requested_acceleration_mps2','issued_acceleration_mps2','issued_target_speed_mps','lane_request_target','lane_command_target'))
    start=e['warmup_s']; ix=np.searchsorted(t,ct)
    cm=p.astype(bool)&cav.astype(bool)&(kin[:,:,1:4].sum(-1)>0)&(t[:,None]>start)
    dm=(ctrl==3)&(ct[:,None]>=start)
    speed=kin[:,:,4][cm]
    lane=np.where(kin[:,:,1:4].sum(-1)>0,kin[:,:,1:4].argmax(-1),-1)
    same=p[1:].astype(bool)&p[:-1].astype(bool)&(vid[1:]==vid[:-1])&(vid[1:]>=0)&cav[:-1].astype(bool)&(t[1:,None]>start)
    changes=same&(lane[1:]!=lane[:-1])&(lane[1:]>=0)&(lane[:-1]>=0)
    valid=dm & (ix[:,None]+2<len(t))
    # Identity-safe full one-second responses only.
    idx=np.minimum(ix+2,len(t)-1)
    valid &= p[ix].astype(bool)&p[idx].astype(bool)&(vid[ix]==vid[idx])
    realised=kin[idx,:,4]-kin[ix,:,4]
    r={'seed':e['seed'],'arm':e['branch']['kind'],'time':start,'share':e['penetration'],
       'speed':stats(speed),'crawl_fraction':float((speed<2).mean()),'stop_fraction':float((speed<.1).mean()),
       'observed_accel':stats(kin[:,:,5][cm]),'requested':stats(req[dm]),'issued':stats(issued[dm]),
       'requested_negative_fraction':float((req[dm]<0).mean()) if dm.any() else None,
       'issued_target_identity_maxerr':float(np.max(np.abs((tgt-kin[ix,:,4]-issued)[dm]))) if dm.any() else None,
       'response_mean':stats(realised[valid]),'response_minus_issued':stats((realised-issued)[valid]),
       'lane_requests':int(((lr>=0)&dm).sum()),'lane_issued':int(((lc>=0)&dm).sum()),
       'lane_changes':int(changes.sum()),'ramp_merges':int((changes&(lane[:-1]==2)&(lane[1:]==1)).sum()),
       'time_bins':[],'regions':{}}
    for lo,hi in ((0,10),(10,60),(60,180),(180,300)):
        mask=cm&(t[:,None]>start+lo)&(t[:,None]<=start+hi)
        a=kin[:,:,4][mask]
        r['time_bins'].append({'seconds':[lo,hi],'speed':stats(a),'crawl_fraction':float((a<2).mean())})
    for name,mask in (('ramp',lane==2),('downstream',(lane!=2)&(kin[:,:,0]>100)),('upstream',(lane!=2)&(kin[:,:,0]<0))):
        a=kin[:,:,4][cm&mask]
        r['regions'][name]={'speed':stats(a),'crawl_fraction':float((a<2).mean()) if a.size else None}
    rows.append(r)
(OUT/'tapes.json').write_text(json.dumps(rows,indent=2))
for arm in ('B','POLICY'):
    rs=[r for r in rows if r['arm']==arm]
    def weighted(key,stat='mean'):
        return sum(r[key]['n']*r[key][stat] for r in rs if r[key]['n'])/sum(r[key]['n'] for r in rs)
    n=sum(r['speed']['n'] for r in rs)
    print(arm,'speed',weighted('speed'),'crawl',sum(r['speed']['n']*r['crawl_fraction'] for r in rs)/n,
          'stop',sum(r['speed']['n']*r['stop_fraction'] for r in rs)/n,
          'requested',weighted('requested') if arm=='POLICY' else None,
          'issued',weighted('issued') if arm=='POLICY' else None,
          'LC',sum(r['lane_changes'] for r in rs),'ramp merges',sum(r['ramp_merges'] for r in rs))
print('example',json.dumps(rows[1]))
