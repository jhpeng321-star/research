"""Independent full-panel accounting, pairing and metric verification; zero SUMO."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import xml.etree.ElementTree as ET
import numpy as np
from admission import amendments,effective,ROOT


def main():
    plan=json.loads((ROOT/'plan.json').read_text());mapping=amendments()
    hashes=defaultdict(set);actual=0;maxroad=maxsource=0.;invalid=[]
    for original in plan['jobs']:
        job=effective(original,mapping);out=ROOT/'jobs'/job['id']
        path=out/'result.json' if (out/'result.json').exists() else out/'recovered_result.json'
        r=json.loads(path.read_text());a=r['initial_audit']
        assert a['all_lanes_match'] and a['max_position_error']<1e-5 and a['max_speed_error']<1e-5
        assert r['actual_sim_s']==600;actual+=r['actual_sim_s']
        valid=not (r['safety'].get('collision_involvements') or r['safety'].get('collision_involvements_at_least') or r['safety'].get('starting_teleports'))
        assert r.get('scientific_valid',True)==valid
        if not valid:invalid.append(job['id'])
        hashes[job['world'],job['rng']].add(r['input_receipt']['future_request_sha256'])
        with np.load(out/'trajectory.npz',allow_pickle=False) as z:
            t=z['t'];v=z['v'];ids=z['vid'];names=z['vehicle_ids']
            times,starts=np.unique(t,return_index=True)
            assert len(times)==1201 and np.allclose(np.diff(times),.5)
            frames=[dict(zip(ids[i:j].tolist(),v[i:j].tolist())) for i,j in zip(starts,np.r_[starts[1:],len(t)])]
            first={}
            for ti,frame in zip(times,frames):
                for vid in frame:first.setdefault(str(names[vid]),float(ti))
            road=0.;road_at={}
            for i in range(1,len(times)):
                before,after=frames[i-1],frames[i]
                road+=.5*sum(max(0.,1.-after.get(vid,speed)/31.29) for vid,speed in before.items())
                h=int(times[i]-300)
                if times[i]-300 in (120,300,600):road_at[str(h)]=road
            # Bound saved-float32 quantization conservatively, not an arbitrary
            # numerical pass threshold. SUMO sums use float64 unsaved speeds.
            bound=float(np.spacing(np.maximum(np.abs(v),1.)).sum())*.5/31.29+1e-6
        requests=[(x.attrib['id'],float(x.attrib['depart'])) for x in ET.parse(out/'generated.rou.xml').getroot().findall('vehicle')]
        for h in (120,300,600):
            source=sum(max(0.,min(first.get(vid,300+h),300+h)-max(300,depart)) for vid,depart in requests)
            c=r['cumulative'][str(h)]
            er=abs(road_at[str(h)]-c['road']);es=abs(source-c['source'])
            assert er<=bound,(job['id'],h,er,bound)
            assert es<1e-7,(job['id'],h,es)
            assert abs(c['team']-c['source']-c['road'])<1e-7
            maxroad=max(maxroad,er);maxsource=max(maxsource,es)
    assert actual==86400 and all(len(h)==1 for h in hashes.values())
    ledger=[json.loads(x) for x in (ROOT/'ledger.jsonl').read_text().splitlines()]
    assert len(ledger)==150 and sum(x['charge'] for x in ledger)==90150
    failures=[json.loads(p.read_text()) for p in (ROOT/'jobs').glob('*/failure.json')]
    assert len(failures)==12
    assert sum(r['actual_sim_s']==0 for r in failures)==4
    assert sum(r['actual_sim_s']==600 for r in failures)==6
    partial=sum(r['actual_sim_s'] for r in failures if 0<r['actual_sim_s']<600)
    assert partial==1117.5
    delay=list(csv.DictReader((ROOT/'scores/delay.csv').open()))
    score=json.loads((ROOT/'scores/summary.json').read_text())
    for row in score['summary']:
        chosen=[r for r in delay if r['variant']==row['variant'] and int(r['horizon'])==row['horizon']
                and (row['split']=='all8' or r['split']==row['split']) and (row['arm']=='all' or r['arm']==row['arm'])]
        assert abs(np.mean([float(r['crps']) for r in chosen])-row['delay_crps'])<1e-8
    receipt=dict(status='PASS',panel_jobs=144,preserved_zero_step_failures=4,
                 actual_sim_s=actual+partial,panel_sim_s=actual,preserved_partial_sim_s=partial,
                 precharged_sim_s=90150,hard_cap=100000,
                 max_road_reconstruction_error=maxroad,max_source_reconstruction_error=maxsource,
                 paired_arrival_cav_stream_cells=len(hashes),scored_delay_rows=len(delay),
                 scientific_validity='FAIL' if invalid else 'PASS',invalid_attempts=invalid,
                 core_sha256=score['references']['core_sha256'])
    with (ROOT/'scores/verification.json').open('x') as f:json.dump(receipt,f,indent=1)
    print(json.dumps(receipt))


if __name__=='__main__':main()
