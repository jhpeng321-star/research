"""Cross-check scored recorded events against completed GLM targets; no inference."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

from .operate import RECEIPTS, remote


def main():
    target = Path('E:/agent-work/rb-shared/targets/lc_events.csv')
    done = target.parent / 'DONE.md'
    if not done.is_file():
        raise RuntimeError('reference helper not done')
    reference = {}
    with target.open(newline='') as f:
        for row in csv.DictReader(f):
            seed = int(row['seed'])
            if not 969400 <= seed <= 969431:
                raise ValueError('reference contains non-FIT world')
            key = (seed, row['arm'].lower(), int(row['vid']), float(row['t_s']),
                   int(row['from_lane']), int(row['to_lane']))
            if key in reference:
                raise ValueError('duplicate target event')
            reference[key] = float(row['s_m'])
    inputs = []
    for path in sorted((RECEIPTS / 'panel_results').glob('w*.json')):
        row = json.loads(path.read_text())
        assert 969400 <= row['seed'] <= 969431
        inputs.append({key: row[key] for key in ('seed','arm','episode','input_sha256')})
    assert len(inputs) == 160
    code = r'''
import pathlib,json,numpy as np,hashlib
root=pathlib.Path('/home/ubuntu/wmdata20261031/views/episodes')
inputs=INPUTS
events=[]
for r in inputs:
    assert 969400<=r['seed']<=969431
    p=root/r['episode']
    for name in ('episode.json','time_s.npy','veh_kin.npy','veh_mask.npy','is_cav.npy','vid.npy'):
        assert hashlib.sha256((p/name).read_bytes()).hexdigest()==r['input_sha256'][name]
    t,k,m,c,vid=[np.load(p/(n+'.npy'),allow_pickle=False,mmap_mode='r') for n in
               ('time_s','veh_kin','veh_mask','is_cav','vid')]
    lane=k[:,:,1:4].argmax(-1); normal=k[:,:,1:4].sum(-1)>0
    good=(m[:-1].astype(bool)&m[1:].astype(bool)&normal[:-1]&normal[1:]
       &(vid[:-1]==vid[1:])&~c[:-1].astype(bool)&(lane[:-1]!=lane[1:]))
    good &= ((t[:-1]>=300)&(t[:-1]<360))[:,None]
    for i,j in zip(*np.nonzero(good)):
        events.append([r['seed'],r['arm'].lower(),int(vid[i,j]),float(t[i+1]),
                       int(lane[i,j]),int(lane[i+1,j]),float(k[i+1,j,0])])
print(json.dumps(events))
'''.replace('INPUTS', repr(inputs))
    events = json.loads(remote(code))
    missing, errors = [], []
    for event in events:
        key = tuple(event[:6])
        if key not in reference:
            missing.append(event)
        else:
            errors.append(abs(event[6] - reference[key]))
    first = json.loads((RECEIPTS/'panel_results/SUMMARY.json').read_text())
    assert len(events) == first['aggregate']['lc2013']['exact']['recorded']
    receipt = dict(helper_done_sha256=hashlib.sha256(done.read_bytes()).hexdigest(),
        target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        helper_events=len(reference), scoring_events=len(events), matched=len(errors),
        missing=missing, position_abs_max_m=max(errors) if errors else None,
        position_abs_mean_m=sum(errors)/len(errors) if errors else None,
        status='PASS' if not missing else 'MISMATCH', new_inference=0,
        definition='Our HDV/same-slot-ID/adjacent-normal-frame/300..360s cohort is a subset of helper all-vehicle/connector-tolerant/300..900s events')
    out = RECEIPTS/'TARGET_CROSSCHECK.json'
    if out.exists():
        raise RuntimeError('crosscheck already retained')
    out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
