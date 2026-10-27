"""Independent numerical and provenance checks on gpuA CPU at nice19."""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp/c4/restart20261026')
HOME = ROOT/'c2real'


def main():
    assert os.getpriority(os.PRIO_PROCESS,0)==19
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    assert all(os.environ[k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
    result=json.loads((HOME/'RESULTS.json').read_text())
    rows=list(csv.DictReader((HOME/'C2REAL.csv').open()))
    starts=[r for r in rows if r['level']=='start']
    assert len(starts)==60
    counts={c:sum(r['cycle']==str(c) for r in starts) for c in range(3)}
    assert counts=={0:8,1:26,2:26}
    assert len({(r['cycle'],r['pair_key']) for r in starts})==60
    for r in starts:
        assert np.isclose(float(r['policy_J'])-float(r['native_J']),float(r['value']))
        assert np.isclose(float(r['source_delta'])+float(r['road_delta']),float(r['value']))
        assert int(r['horizon_s'])==300
    vectors={}
    for c in range(3):
        sample=[r for r in starts if r['cycle']==str(c)]
        worlds=sorted({r['world'] for r in sample})
        values=np.array([sum(float(r['value']) for r in sample if r['world']==w)/2 for w in worlds])
        vectors[c]=values
        expected=result['cycles'][c]
        indices=np.random.default_rng(20261027).integers(0,len(values),(5000,len(values)))
        bounds=np.quantile(values[indices].mean(1),[.025,.975])
        assert np.allclose([values.mean(),*bounds],[expected['value'],expected['ci_low'],expected['ci_high']])
        # Original smoke sums the two starts; reproduce it independently.
        journal=json.loads((ROOT/f'arm/loop/worlds_c{c:02}.json').read_text())
        observed=np.array(journal['native_smoke']['world_deltas'])
        assert np.allclose(values[:4]*2,observed)
        ix=np.random.default_rng(20261026).integers(0,4,(5000,4))
        assert np.allclose(np.quantile(observed[ix].mean(1),[.025,.975]),journal['native_smoke']['ci95'])
    rng=np.random.default_rng(20261027)
    draws=vectors[2][rng.integers(0,13,(5000,13))].mean(1)
    draws-=vectors[1][rng.integers(0,13,(5000,13))].mean(1)
    contrast=result['cycle2_minus_cycle1']
    assert np.allclose(np.quantile(draws,[.025,.975]),[contrast['ci_low'],contrast['ci_high']])
    assert np.isclose(vectors[2].mean()-vectors[1].mean(),contrast['value'])
    beh=[r for r in rows if r['level']=='behaviour_start']
    assert len(beh)==480
    behaviour_worlds={}
    for c in range(3):
        for metric in ('metering','multi_slow','side_by_side','vacate'):
            worlds=sorted({r['world'] for r in beh if r['cycle']==str(c)})
            grouped=np.zeros((len(worlds),2,2))
            for r in beh:
                if r['cycle']==str(c) and r['metric']==metric:
                    grouped[worlds.index(r['world']),int(r['arm']=='policy')]+=[int(r['count']),float(r['exposure_cav_h'])]
            behaviour_worlds[c,metric]=grouped
            indices=np.random.default_rng(20261027).integers(0,len(worlds),(5000,len(worlds)))
            sums=grouped[indices].sum(1)
            rates=sums[:,:,0]/sums[:,:,1]
            raw={}
            for arm in ('native','policy'):
                selected=[r for r in beh if r['cycle']==str(c) and r['metric']==metric and r['arm']==arm]
                count=sum(int(r['count']) for r in selected)
                exp=sum(float(r['exposure_cav_h']) for r in selected)
                rate=count/exp
                raw[arm]=rate
                est=next(r for r in rows if r['level']=='behaviour_estimate' and r['cycle']==str(c)
                         and r['metric']==metric and r['arm']==arm)
                assert np.isclose(rate,float(est['value']))
                assert float(est['ci_low'])<=rate<=float(est['ci_high'])
                bounds=np.quantile(rates[:,int(arm=='policy')],[.025,.975])
                assert np.allclose(bounds,[float(est['ci_low']),float(est['ci_high'])])
            est=next(r for r in rows if r['level']=='behaviour_estimate' and r['cycle']==str(c)
                     and r['metric']==metric and r['arm']=='policy_minus_native')
            assert np.isclose(raw['policy']-raw['native'],float(est['value']))
            assert np.allclose(np.quantile(rates[:,1]-rates[:,0],[.025,.975]),
                               [float(est['ci_low']),float(est['ci_high'])])
    for est in result['metering']:
        rng=np.random.default_rng(20261027)
        draws=[]
        for c in (2,1):
            x=behaviour_worlds[c,est['metric']]
            sums=x[rng.integers(0,len(x),(5000,len(x)))].sum(1)
            draws.append(sums[:,1,0]/sums[:,1,1]-sums[:,0,0]/sums[:,0,1])
        assert np.allclose(np.quantile(draws[0]-draws[1],[.025,.975]),[est['ci_low'],est['ci_high']])
    for stat in result['wm_existing']:
        c,h=stat['cycle'],stat['horizon_s']
        rr=[r for r in rows if r['level']=='wm_start' and r['cycle']==str(c) and r['horizon_s']==str(h)]
        assert len(rr)==stat['available_starts']
        for r in rr:
            assert np.isclose(float(r['realised_dJ'])-float(r['predicted_dJ']),float(r['value']))
        ws=sorted({r['world'] for r in rr})
        bias=np.mean([np.mean([float(r['value']) for r in rr if r['world']==w]) for w in ws])
        assert np.isclose(bias,stat['bias']['value'])
        assert np.isclose(np.corrcoef([[float(r['predicted_dJ']) for r in rr],
                                      [float(r['realised_dJ']) for r in rr]])[0,1],stat['pearson'])
    source=(HOME/'analyze.py').read_bytes()
    ast.parse(source)
    assert 'import torch' not in source.decode()
    inputs=json.loads((HOME/'INPUTS.json').read_text())
    for name,sha in inputs.items():
        path=ROOT/name
        assert path.resolve().is_relative_to(ROOT.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,name
    assert inputs['c2real/thresholds.json'].startswith('9119cd77')
    coverage=json.loads((HOME/'COVERAGE.json').read_text())
    assert coverage['cycle1_cycle2_worlds_disjoint']
    assert coverage['source_sha256']==hashlib.sha256((HOME/'coverage.py').read_bytes()).hexdigest()
    for name,sha in coverage['input_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    report=(HOME/'report.md').read_text()
    assert len(report.split())<=400, len(report.split())
    normalized=report.replace('\N{MINUS SIGN}','-')
    for stat in result['cycles']+[result['pooled'],result['cycle2_minus_cycle1']]:
        phrase=f"{stat['value']:,.0f} [{stat['ci_low']:,.0f},{stat['ci_high']:+,.0f}]"
        assert phrase in normalized,phrase
    assert all(phrase in normalized for phrase in ('unpaired','unidentified at 60 s','bias CI unavailable'))
    owned=json.loads((HOME/'PID.json').read_text())
    proc=Path('/proc')/str(owned['pid'])/'stat'
    if proc.exists():
        fields=proc.read_text().rsplit(')',1)[1].split()
        assert fields[19]!=owned['start_ticks'] or fields[0]=='Z','own analysis still running'
    receipt=dict(status='PASS',start_pairs=60,worlds=30,tapes=120,behaviour_start_rows=480,
                 input_files_verified=len(inputs),nice=19,numerical_threads=1,gpu_compute=False,
                 simulation_calls=0,analysis_pid=owned,
                 source_sha256=hashlib.sha256(source).hexdigest(),
                 csv_sha256=hashlib.sha256((HOME/'C2REAL.csv').read_bytes()).hexdigest(),
                  results_sha256=hashlib.sha256((HOME/'RESULTS.json').read_bytes()).hexdigest(),
                  verification_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  coverage_sha256=hashlib.sha256((HOME/'COVERAGE.json').read_bytes()).hexdigest(),
                  report_sha256=hashlib.sha256((HOME/'report.md').read_bytes()).hexdigest(),
                  report_words=len(report.split()),coverage_input_files_verified=len(coverage['input_hashes']),
                 own_analysis_exited=True,tests=['pairing/accounting/clocks','three guard reproductions',
                    'world bootstrap','unpaired cycle contrast','frozen behaviour pooled ratios',
                     'same-start prediction residuals/correlation','all input hashes',
                     'all behaviour rate and unpaired contrast bootstrap CIs','coverage input hashes',
                     'report word count and rounded real-effect CIs'])
    (HOME/'VERIFIED.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
