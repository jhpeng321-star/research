"""Analytical and receipt checks for the diagnostic (no model execution)."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import numpy as np
from analyze import ci, paired

ROOT=Path(__file__).resolve().parents[2]


def analytical_checks():
    # Exact support of resampling two unequal-sized clusters:
    # AA -> median 1.5, AB/BA -> 2, BB -> 100. Independent of code's
    # vectorized representation; treating starts as independent is different.
    result=ci([('A',1.),('A',2.),('B',100.)])
    assert result['point']==2. and result['lo']==1.5 and result['hi']==100.
    assert result['equal_world_mean']==50.75
    def row(values):return {'endpoints':{'60':{'ids':{k:{'ds':v} for k,v in values.items()}}}}
    a=row({'one':10.,'two':100.,'private':0.})
    b=row({'one':20.,'two':50.,'other':0.})
    assert paired(a,b,60)==dict(a=55.,b=35.,delta=20.,n_ids=2)
    assert paired(a,b,60,row({'one':0.}))['delta']==-10.
    assert paired(row({'a':1.}),row({'b':2.}),60) is None
    return dict(unequal_world_cluster_bootstrap=True,common_id_before_pairing=True,
                three_way_anchor_ids=True,empty_endpoints_unavailable=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw');ap.add_argument('--summary');args=ap.parse_args()
    value={'analytical':analytical_checks()}
    if args.raw:
        path=Path(args.raw)
        with gzip.open(path,'rt') as f:rows=[json.loads(line) for line in f]
        assert all(not r['error'] for r in rows)
        assert len({r['_source'] for r in rows})==len(rows)
        assert {r['tag'] for r in rows}=={'c1struct','wm13e','wm2','bbtraj'}
        for r in rows:
            if r['_panel']=='aux':
                d=r['derived']
                assert d['t0']==d['original_t0']+60. and d['history_max']==d['t0']
                assert d['history_min']==d['original_t0'] and d['n_history_ticks']==121
                assert d['future_noninterference']
                if r['cohort']=='wall_same':assert r['endpoints']['120'].get('unavailable')
        checks=[];legacy_checks=[]
        refresh=Path('E:/agent-work/c4-c1refresh/_out/c1refresh/cpuy/out/units')
        for r in rows:
            if r['_panel'] not in ('original','legacy') or r['mode']!='tf' or r.get('component'):continue
            if r['tag']=='wm2' or r['population'] or not r['residual'] or r['lanes']!='original':continue
            if r['history']!=('h5' if r['cohort']=='wall_same' else 'snapshot'):continue
            e=r['unit'][0];p=refresh/f"{r['tag']}__tf__H120"/f"{r['sid']}__{e['arm']}.json"
            if not p.is_file():continue
            old=json.loads(p.read_text())
            for h in (60,120):
                got=r['endpoints'][str(h)]['median'];want=old['car'].get(str(float(h)),{}).get('hdv',{}).get('ds_med')
                entry=dict(sid=r['sid'],tag=r['tag'],horizon=h,new=got,archived=want,
                    abs_difference=abs(got-want) if got is not None and want is not None else None,
                    missing_agrees=(got is None)==(want is None))
                (legacy_checks if r['_panel']=='legacy' else checks).append(entry)
        finite=[c['abs_difference'] for c in checks if c['abs_difference'] is not None]
        value.update(units=len(rows),raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            refresh_reproduction=dict(comparisons=len(checks),missing_agree=all(c['missing_agrees'] for c in checks),
                max_abs_difference=max(finite) if finite else None,
                median_abs_difference=float(np.median(finite)) if finite else None,
                within_1mm=sum(v<=.001 for v in finite),finite=len(finite),details=checks))
        if legacy_checks:
            finite=[c['abs_difference'] for c in legacy_checks if c['abs_difference'] is not None]
            value['archived_source_reproduction']=dict(comparisons=len(legacy_checks),
                max_abs_difference=max(finite),median_abs_difference=float(np.median(finite)),
                missing_agree=all(c['missing_agrees'] for c in legacy_checks),details=legacy_checks)
    if args.summary:
        summary=Path(args.summary)
        doc=json.loads(summary.read_text())
        assert doc['raw_sha256']==value['raw_sha256']
        # C1's original residual mode must not be re-described as learned
        # physical-style identification. Check its inert population switch.
        groups={(r['panel'],r['mode'],r['cohort'],r['config']):r for r in doc['summary']}
        inert=[]
        for key,r in groups.items():
            if r['tag']!='c1struct' or not r['population']:continue
            other=groups.get((*key[:3],key[3].replace('__pop__','__inf__')))
            if other:
                for h in ('60','120'):
                    a=r['horizons'][h]['own']['point'];b=other['horizons'][h]['own']['point']
                    inert.append((a is None and b is None) or abs(a-b)<1e-8)
        assert all(inert)
        if not inert:assert not any(r['population'] for r in rows)
        value['c1_population_inert_cells']=len(inert)
        value['summary_sha256']=hashlib.sha256(summary.read_bytes()).hexdigest()
        target=summary.with_name('VERIFICATION.json')
        with target.open('x') as f:json.dump(value,f,indent=1,allow_nan=False)
        sources=[Path('E:/agent-work/c4-c1refresh')/p for p in ('C1REFRESH.md','docs/c1refresh/C1REFRESH_RESULTS.json',
            'docs/c1refresh/C1REFRESH_PER_STATE.csv','docs/c1refresh/INPUTS_MANIFEST.json','tools/c1refresh/c1r_run.py','tools/c1refresh/c1r_analyze.py')]
        sources += [Path('E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md'),
            Path('E:/agent-work/c4-c1gap/docs/paper_results/c1gap/exposure.json'),
            Path('E:/agent-work/c4-c1closed/C1CLOSED_DESIGN.md'),ROOT/'DIAG_PLAN.md',ROOT/'docs/paper_results/c1pidiag/PANEL.json']
        receipt={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        with target.with_name('SOURCE_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({k:v for k,v in value.items() if k not in ('refresh_reproduction','archived_source_reproduction')},indent=2))
    if 'refresh_reproduction' in value:
        print('refresh comparison', {k:v for k,v in value['refresh_reproduction'].items() if k!='details'})
    if 'archived_source_reproduction' in value:
        print('archived-source comparison',{k:v for k,v in value['archived_source_reproduction'].items() if k!='details'})


if __name__=='__main__':main()
