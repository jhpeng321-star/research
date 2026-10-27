"""Full-recording sensor/command coverage audit on E3 validation only."""
import json
from pathlib import Path
import numpy as np
from replay import safe,breakdown,regions
from summarize import write_csv


def main(root,out):
    m=json.loads(safe(root/'e3data/C4_EPISODES_MANIFEST.json').read_text())
    split=json.loads(safe(root/'e3data/SPLIT.json').read_text())
    direct_id=m['controller_vocab'].index('direct')
    rows=[];sensors=[];events=[]
    for e in m['episodes']:
        assert split['worlds'][str(e['seed'])]=='validation'
        assert not 964000<=e['seed']<=964199
        p=root/'e3data'/e['dir']
        t=np.load(safe(p/'time_s.npy'),mmap_mode='r');t0=int(np.searchsorted(t,e['warmup_s']))
        a=np.load(safe(p/'veh_kin.npy'),mmap_mode='r')
        present=np.load(safe(p/'veh_mask.npy'),mmap_mode='r').astype(bool)
        cav=np.load(safe(p/'is_cav.npy'),mmap_mode='r').astype(bool)
        ids=np.load(safe(p/'vid.npy'),mmap_mode='r')
        control_times=np.load(safe(p/'control_time_s.npy'),mmap_mode='r')
        controls=np.load(safe(p/'controller.npy'),mmap_mode='r')
        # Frozen converter vocabulary: native, damping, PI-external-u, direct.
        control_index=np.searchsorted(control_times,t,side='right')-1
        assert control_index[t0:].min()>=0
        issued=controls[np.clip(control_index,0,len(control_times)-1)]
        valid=(a[...,1:4].sum(-1)>0)&present
        t0_ids=ids[t0][present[t0]]
        original=np.isin(ids,t0_ids)&(ids>=0)
        for start in range(0,600,20):
            sl=slice(t0+start+1,t0+start+21)
            allc=valid[sl]&cav[sl];old=allc&original[sl]
            for region,reg in regions(a[sl,...,0],a[sl,...,1:4].argmax(-1)).items():
                total=int((allc&reg).sum());cohort=int((old&reg).sum())
                direct=allc&reg&(issued[sl]==direct_id)
                direct_n=int(direct.sum());new_direct=int((direct&~original[sl]).sum())
                rows.append({'role':'validation','world':e['world'],'arm':e['arm'],'region':region,
                    'bin_end_s':start//2+10,'cav_vehicle_seconds':total*.5,
                    't0_cav_vehicle_seconds':cohort*.5,'post_t0_cav_vehicle_seconds':(total-cohort)*.5,
                    'post_t0_cav_fraction':(total-cohort)/total if total else None,
                    'direct_cav_vehicle_seconds':direct_n*.5,
                    'post_t0_direct_cav_vehicle_seconds':new_direct*.5,
                    'post_t0_fraction_of_direct_cav_seconds':new_direct/direct_n if direct_n else None,
                    'replay_note':'Production paired replay assigns synthetic entrant CAVs native commands; no identity bridge to these recorded future CAVs.'})
        for region,reg in regions(a[...,0],a[...,1:4].argmax(-1)).items():
            if region not in ('upstream','merge'):continue
            vals=[]
            for start in range(t0-120,t0+600,20):
                sl=slice(start,start+20);keep=valid[sl]&reg[sl]
                speed=float(a[sl,...,4][keep].mean()) if keep.any() else None;vals.append(speed)
                sensors.append({'role':'validation','world':e['world'],'arm':e['arm'],'region':region,
                    'bin_start_s':(start-t0)*.5,'mean_speed_mps':speed,'n_vehicle_ticks':int(keep.sum())})
            events.append({'role':'validation','world':e['world'],'arm':e['arm'],'region':region,**breakdown(vals)})
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'recorded_controller_coverage.csv',rows)
    write_csv(out/'recorded_speed_sensors.csv',sensors)
    write_csv(out/'recorded_breakdown.csv',events)
    receipt={'status':'complete','version':2,'role':'validation','worlds':36,
        'episodes':108,'source':'full raw converted arrays, not t0-cohort loader masks',
        'code_evidence':'src/wms/train/inputs.py:_device_timeline defaults synthetic entrant columns to native',
        'interpretation':'Coverage establishes a replay confound, not the causal share of paired-error bias.'}
    (out/'OBSERVABLE_DONE.json').write_text(json.dumps(receipt,indent=1))
    (out/'CONTROLLER_COVERAGE.json').write_text(json.dumps(receipt,indent=1))


if __name__=='__main__':main(Path.home()/'lrdiag',Path.home()/'lrdiag/evidence')
