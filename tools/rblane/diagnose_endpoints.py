"""FIT input-only audit of exact represented ramp endpoints; no WM inference."""
import json

from .operate import RECEIPTS, remote


def main():
    out = RECEIPTS/'ENDPOINT_INPUT_AUDIT.json'
    assert out.parent.is_dir() and not out.exists()
    source = json.loads((RECEIPTS/'PANEL9_COMPLETE.json').read_text())
    assert source['complete'] and not source['alive']
    code = r'''
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
assert os.getloadavg()[0]<20
os.nice(19)
import json,re,pathlib,numpy as np
from collections import Counter
root=pathlib.Path('/home/ubuntu/wmdata20261031/views/episodes')
total=Counter(); population=Counter(); rows=[]; keys=set()
end=float(75.44); encoded=np.float32(end)
for world in range(969400,969432):
    for p in sorted(root.glob(f'w{world}m*')):
        if not re.fullmatch(rf'w{world}m[0-9]+p[0-9]+t300__(native|ctrl0|mappo|pi|circles)',p.name): continue
        meta=json.loads((p/'episode.json').read_text()); keys.update(meta)
        assert int(meta['seed'])==world and meta['branch']['t0_s']==300
        pop=meta.get('population')
        for field in ('world','source','scenario'):
            if pop is None and isinstance(meta.get(field),dict): pop=meta[field].get('population')
        population[str(pop)]+=1
        a={name:np.load(p/(name+'.npy'),allow_pickle=False,mmap_mode='r')
           for name in ('time_s','veh_kin','veh_mask','vid','is_cav')}
        t=a['time_s']; window=np.flatnonzero((t>=240)&(t<=360))
        lo,hi=int(window[0]),int(window[-1])+1
        k=a['veh_kin'][lo:hi]; m=a['veh_mask'][lo:hi].astype(bool)
        ids=a['vid'][lo:hi]; cav=a['is_cav'][lo:hi].astype(bool); times=t[lo:hi-1]
        assert k.dtype==np.float32
        normal=k[:,:,1:4].sum(-1)>0
        present=m[:-1]&m[1:]&(ids[:-1]==ids[1:])&normal[:-1]&normal[1:]
        lane=k[:,:,1:4].argmax(-1); ramp=present&(lane[:-1]==2)
        # The ramp-to-through origin is identical on the merge edge, so thev2
        # source-frame coordinate is exactly this observed post-motion value.
        at=ramp&(k[1:,:,0]==encoded)
        row=Counter()
        for phase,clock in (('warmup',times<300),('scoring',times>=300)):
            scope=clock[:,None]
            row[phase+'/all_ramp_endpoint_frames']=int((at&scope).sum())
            row[phase+'/hdv_ramp_endpoint_frames']=int((at&scope&~cav[:-1]).sum())
            row[phase+'/recorded_hdv_changes_at_endpoint']=int((at&scope&~cav[:-1]&(lane[1:]!=2)).sum())
            row[phase+'/all_recorded_hdv_ramp_changes']=int((ramp&scope&~cav[:-1]&(lane[1:]!=2)).sum())
        total.update(row); rows.append(dict(episode=p.name,counts=row))
assert len(rows)==160 and total['scoring/all_recorded_hdv_ramp_changes']==801
print(json.dumps(dict(status='PASS',episodes=160,worlds=32,counts=total,
    geometric_end_m=end,float32_end_m=float(encoded),rounding_error_m=float(encoded)-end,
    metadata_population=population,metadata_keys=sorted(keys),by_episode=rows,
    parameter_inference=False,wm_inference=False)))
'''
    result = json.loads(remote(code, timeout=90))
    result['input_identity_reference'] = source['root']+'/results (completed PANEL9 input SHA receipts)'
    result['source_panel_bundle_sha256'] = source['bundle_sha256']
    result['fresh_byte_hashing'] = False
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='by_episode'},indent=2))


if __name__ == '__main__':
    main()
