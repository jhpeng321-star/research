"""Offline numerical and identity checks; never dispatch or load a WM."""
import ast
import csv
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import analyze as a


def main():
    provenance=json.loads(a.read(a.OUT/'provenance.json'))
    source=Path(provenance['detector_source']).read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='breakdown')
    detector=ast.get_source_segment(source,node)
    assert hashlib.sha256(detector.encode()).hexdigest()==provenance['detector_sha256']
    scope={'np':np}; exec(detector,scope)
    episodes=[]
    for name,host in zip(('wmdata','wmdata2'),provenance['hosts']):
        raw=a.read(a.OUT/(name+'_observables.json'))
        assert hashlib.sha256(raw).hexdigest()==host['local_receipt_sha256']
        data=json.loads(raw); assert len(data['episodes'])==host['episodes']
        assert len(data['checks'])==host['episodes']
        assert max(r['converted_raw_max_mps'] for r in data['checks'])<1e-5
        episodes+=data['episodes']
    checked=0
    for e in episodes:
        for b in e['bins'].values():
            assert len(b['speed'])==66
            for h in a.HS:
                actual=scope['breakdown'](b['speed'][:6+h//10])
                saved=b['events'][str(h)]
                assert actual.keys()==saved.keys()
                for key,value in actual.items():
                    if isinstance(value,float):
                        assert np.isclose(value,saved[key],rtol=1e-8,atol=1e-8),(e['seed'],e['arm'],h,key)
                    else:
                        assert value==saved[key],(e['seed'],e['arm'],h,key)
                assert actual['event'] is not None
                checked+=1
    assert checked==4500
    assert a.crps([0,2],1)==.5 and a.crps([0,2],1,True)==0
    assert a.crps([3],1)==2 and a.crps([1]*8,1)==0
    # Independent exact CDF integration check for a nontrivial finite ensemble.
    x=np.array([-3.,1.,2.,7.]); y=4.
    knots=np.unique(np.r_[x,y]); mid=(knots[:-1]+knots[1:])/2
    integrated=np.sum(np.diff(knots)*((x[:,None]<=mid).mean(0)-(y<=mid))**2)
    assert np.isclose(integrated,a.crps(x,y))
    tables={p.stem:list(csv.DictReader(p.open(newline=''))) for p in a.OUT.glob('*.csv')}
    assert len(tables['real_events'])==4500 and len(tables['forecast_draws'])==13440
    assert len(tables['distribution_worlds'])==1680
    for rows in tables.values():
        assert all(v.lower() not in ('nan','inf','-inf') for r in rows for v in r.values())
    scores=tables['distribution_worlds']
    assert all(969400<=int(r['seed'])<=969431 for r in scores if r['split']=='dev32')
    assert all(969432<=int(r['seed'])<=969439 for r in scores if r['split']=='reserved8')
    assert all(int(r['baseline_worlds'])==(31 if r['split']=='dev32' else 32) for r in scores)
    # Conservative planning: real Bernoulli plus K-future Monte Carlo uncertainty.
    nd=NormalDist(); K=32; epsilon=.1; M=12*5*3
    n=int(np.ceil(nd.inv_cdf(1-.05/(2*M))**2*.25*(1+1/K)/epsilon**2))
    receipt=dict(detector_rechecks=checked,real_episodes=len(episodes),numerical_crps_checks='PASS',
        source_receipt_hashes='PASS',split_identity_checks='PASS',finite_csv_checks='PASS',
        gate_rate_design=dict(cells=12,arms=5,horizons=3,tests=M,K=K,epsilon=epsilon,
            approximate_worlds_per_cell=n,assumptions='Normal planning bound; world-cluster inference required; exact/binomial or bootstrap power recheck before registration'),
        outputs={p.name:hashlib.sha256(a.read(p)).hexdigest() for p in sorted(a.OUT.iterdir())
                 if p.suffix in ('.csv','.png','.json') and p.name!='verification.json'})
    (a.OUT/'verification.json').write_text(json.dumps(receipt,indent=1,allow_nan=False))
    print('PASS:',checked,'detector rechecks; CRPS integral; receipt hashes; split IDs; finite CSVs')
    print('Prospective 12-cell x 5-arm x 3-horizon rate gate: approx',n,'worlds/cell x K32 for 10pp simultaneous precision')


if __name__=='__main__':main()
