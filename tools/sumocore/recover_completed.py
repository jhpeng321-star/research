"""Recover finished collision-flagged data WITHOUT re-simulation or retuning.

Original failures remain. Reconstructed road clocks carry a float32 rounding
bound; source clocks are recovered exactly from observable own request times.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
from worker import ROOT,write
from admission import digest


def main():
    recovered=[];failure_hashes={}
    for failure in sorted((ROOT/'jobs').glob('*/failure.json')):
        f=json.loads(failure.read_text())
        if f['actual_sim_s']!=600:continue
        assert "assert not env.totals.get('collision_involvements')" in f['traceback']
        out=failure.parent
        assert not (out/'result.json').exists()
        with np.load(out/'trajectory.npz',allow_pickle=False) as z:
            times,starts=np.unique(z['t'],return_index=True)
            assert len(times)==1201 and times[0]==300 and times[-1]==900
            frames=[dict(zip(z['vid'][i:j].tolist(),z['v'][i:j].tolist())) for i,j in zip(starts,np.r_[starts[1:],len(z['t'])])]
            first={}
            for t,frame in zip(times,frames):
                for vid in frame:first.setdefault(str(z['vehicle_ids'][vid]),float(t))
            bound=float(np.spacing(np.maximum(abs(z['v']),1.)).sum())*.5/31.29+1e-6
        requests=[(x.attrib['id'],float(x.attrib['depart'])) for x in ET.parse(out/'generated.rou.xml').getroot().findall('vehicle')]
        cumulative={};road=0.
        for i in range(1,len(times)):
            before,after=frames[i-1],frames[i]
            road+=.5*sum(max(0.,1.-after.get(vid,speed)/31.29) for vid,speed in before.items())
            if times[i]-300 in (120,300,600):
                h=int(times[i]-300)
                source=sum(max(0.,min(first.get(vid,300+h),300+h)-max(300,depart)) for vid,depart in requests)
                cumulative[str(h)]=dict(road=road,source=source,team=road+source)
        result=dict(ok=True,scientific_valid=False,recovered_after_postrun_assert=True,
            job=f['job'],actual_sim_s=600.,cumulative=cumulative,
            initial_audit=json.loads((out/'initial_audit.json').read_text()),
            input_receipt=json.loads((out/'input_receipt.json').read_text()),
            safety=dict(collision_involvements_at_least=1,exact_collision_count_unavailable=True),
            clock_reconstruction=dict(road_float32_error_bound_vs=bound,source_exact=True),
            original_failure_sha256=digest(failure))
        write(out/'recovered_result.json',result)
        recovered.append(out.name);failure_hashes[out.name]=digest(failure)
    assert len(recovered)==6
    stops={p.name:digest(p) for p in [ROOT/'STOP',*ROOT.glob('STOP.*')] if p.exists()}
    write(ROOT/'STATUS_AMENDMENT.json',dict(recovered=recovered,stops=stops,
        full_rollout_failures=failure_hashes,extra_sim_s=0,extra_precharge=0,
        interpretation='retain collision-flagged forecasts in diagnostic scores; they FAIL physical validity, never rerun or tune'))
    print(json.dumps(dict(recovered=recovered,extra_sim_s=0)))


if __name__=='__main__':main()
