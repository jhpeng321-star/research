"""Consume the completed shared obsstate/targets helpers without new simulation."""
from pathlib import Path
import csv
import gzip
import importlib.util
import json
import sys
import numpy as np
R=Path('/home/ubuntu/e3final/runtime');sys.path[:0]=[str(R/'src'),str(R)]
HERE=Path(__file__).resolve().parent
from admission import ROOT
from prepare_bundle import network


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,HERE/file)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def main():
    nf=module('wms.eval.netframe','helper_netframe.py')
    obs=module('wms.eval.obsstate','helper_obsstate.py')
    net=Path('/home/ubuntu/e3final/runtime/scenarios/idm_v5_7_extended_zone/osm.v5_7.extended.net.xml')
    # Helper parser accepts the recorded gzip packaging, while the frozen
    # simulator installs the same network as plain XML. Repackage only.
    packed=ROOT/'helper_net.xml.gz'
    if not packed.exists():
        with packed.open('xb') as f:f.write(gzip.compress(net.read_bytes(),mtime=0))
    assert gzip.decompress(packed.read_bytes())==net.read_bytes()
    frame=nf.load_netframe(packed)
    _,route=network()
    plan=json.loads((ROOT/'plan.json').read_text())
    worlds=sorted({j['world'] for j in plan['jobs']});rows=[]
    for world in worlds:
        p=Path('/home/ubuntu/wmdata20261031/jobs')/f's{world}_native/deploy/episode/episode.npz'
        tape=obs.load_tape(p);tick=obs.resolve_tick(tape,t0_s=300)
        # Keep only causal fields and <=t0 rows before invoking helper export.
        arrays={k:np.asarray(tape.arrays[k])[:tick+1] for k in
            ('time_s','s_m','lane','lane_carry','v_mps','a_mps2','length_true_m','present','is_cav','vid_index')}
        tape=obs.Tape(tape.path,arrays,tape.meta,tape.vehicle_ids)
        vehicles,excluded,summary=obs.export_state(tape,tick,frame)
        assert not excluded and summary['n_exported']==summary['n_present']
        assert all(list(v.placement.route_edges)==route(v.placement.edge) for v in vehicles)
        assert all(v.length_m==5. for v in vehicles)
        rows.append(dict(world=world,helper_present=len(vehicles),routes_match=True,lengths5m=True,
                         helper_tight_pairs=len(obs.overlap_warnings(vehicles,frame))))
    real=json.loads((HERE/'real_observables.json').read_text())['episodes']
    lookup={(e['seed'],e['arm'],int(h)):c['team_delay_vs'] for e in real for h,c in e['clocks'].items()}
    count=0;error=0.
    with (HERE/'helper_clocks.csv').open() as f:
        for row in csv.DictReader(f):
            w=int(row['seed'])
            if w in worlds and row['arm'] in ('native','mappo','pi'):
                diff=abs(float(row['team_delay_vs'])-lookup[w,row['arm'],int(row['horizon_s'])])
                assert diff<1e-8;count+=1;error=max(error,diff)
    receipt=dict(status='PASS',extra_sim_s=0,obsstate_helper_commit='541f109f',states=rows,
                 total_present=sum(r['helper_present'] for r in rows),target_clock_comparisons=count,max_clock_error=error)
    with (ROOT/'helper_crosscheck.json').open('x') as f:json.dump(receipt,f,indent=1)
    print(json.dumps(receipt))


if __name__=='__main__':main()
