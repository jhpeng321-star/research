"""Reuse C1CLOSED forecast/error/bootstrap code; adapt only panel size and schema."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
from core import digest, write, audit

sys.addaudithook(audit)
HERE=Path(__file__).resolve().parent
UPSTREAM=HERE/'evidence/C1CLOSED_analyze_source.py'


def load_upstream(path):
    spec=importlib.util.spec_from_file_location('c1closed_analysis',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def task(row):
    values={}
    for h,value in row['values'].items():
        v={k:[value[k]] for k in ('source','upstream','merge','discharge','J',
                                 'crossings','merge_queue','mean_merge_queue')}
        ep=dict(value['endpoints'])
        ep['present']=[True]*len(ep['ids'])
        v['endpoints']=[ep]
        v['fields']={k:[x] for k,x in value['fields'].items()}
        values[h]=v
    return dict(arms=['native'],values=values)


def panel_summarizer(module,n):
    # Preserve every upstream statistic and missing-ID rule. Only the registered
    # dimension changes; do not pad to 40 worlds or bootstrap draws separately.
    source=inspect.getsource(module.summarize)
    assert source.count('rng.integers(0,40,(2000,40))')==1
    assert source.count('np.full(40,np.nan)')==1
    adapted=source.replace('rng.integers(0,40,(2000,40))',f'rng.integers(0,{n},(2000,{n}))')
    adapted=adapted.replace('np.full(40,np.nan)',f'np.full({n},np.nan)')
    namespace=dict(module.__dict__)
    exec(compile(adapted,'<C1CLOSED summarize: panel dimension only>','exec'),namespace)
    return namespace['summarize'],dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                      adapted_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
                                      only_change=f'world panel dimension 40->{n}')


def analyze(labels_path,out,upstream=UPSTREAM,chaos=False):
    A=load_upstream(upstream)
    data=json.loads(labels_path.read_text())
    by={(r['seed'],r['variant']):r for r in data['rows']}
    seeds=sorted({s for s,v in by})
    n=6 if chaos else 18
    model='SUMO_tiny_perturbation_chaos' if chaos else 'SUMO_RNG_floor'
    assert seeds==list(range(969400,969400+n)) and len(by)==3*n
    assert all((s,v) in by for s in seeds for v in ('reference','alt0','alt1'))
    raw,draws=[],[]
    for w,s in enumerate(seeds):
        ref=by[s,'reference']
        hdv=[i for i,c in zip(ref['original_ids'],ref['original_cav']) if not c]
        if chaos:
            excluded=set(ref['excluded_target_indices'])
            assert len(excluded)==2 and excluded<=set(hdv)
            hdv=[i for i in hdv if i not in excluded]
        rows=[task(by[s,v]) for v in ('alt0','alt1')]
        for h in A.HS:
            real=ref['values'][str(h)]
            pred=A.forecast(rows,'native',h)
            ep,_=A.endpoint(pred,real['endpoints'],hdv)
            raw.append(dict(model=model,plant='recorded',world=w,seed=s,
                            arm='native',horizon_s=h,**A.errors(pred,real),**ep,initial_hdv=len(hdv)))
            for k,row in enumerate(rows):
                p=A.forecast([row],'native',h)
                e,_=A.endpoint(p,real['endpoints'],hdv)
                draws.append(dict(model=model,plant='recorded',world=w,seed=s,
                                   arm='native',horizon_s=h,draw=k,rng_seed=s if chaos else 969050+2*w+k,
                                  **A.errors(p,real),**e,initial_hdv=len(hdv)))
    summarize,adapter=panel_summarizer(A,len(seeds))
    summary=[r for r in summarize(raw) if r['arm']=='native']
    assert all(r['model']==model for r in summary)
    out.mkdir(parents=True,exist_ok=True)
    A.write_csv(out/'C1FLOOR_per_world.csv',raw)
    A.write_csv(out/'C1FLOOR_per_draw.csv',draws)
    A.write_csv(out/'C1FLOOR_error_horizon.csv',summary)
    receipt=dict(schema='c1floor_analysis/1',worlds=n,draws=2,horizons=list(A.HS),method=model,
                 labels_sha256=digest(labels_path),upstream_analyzer_sha256=digest(upstream),
                 upstream_labels_sha256=data['source_sha256'],upstream_aggregates_sha256=data['aggregates_sha256'],
                 adapter=adapter,analyzer_sha256=digest(__file__),
                 outputs={p.name:digest(p) for p in out.glob('C1FLOOR*.csv')})
    write(out/'C1FLOOR_ANALYSIS.json',receipt)
    print(json.dumps(receipt,indent=1))


def blocked(receipt_path,out,upstream=UPSTREAM):
    A=load_upstream(upstream)
    receipt=json.loads(receipt_path.read_text())
    assert receipt['status']=='BLOCKED_SAVED_STATE_CANARY'
    assert receipt['alternative_branches_run']==receipt['panel_worlds_run']==0
    metrics=('position','speed','J_ae','source_ae','upstream_ae','merge_ae','discharge_ae',
             'crossings_ae','throughput_ae','merge_queue_ae','mean_merge_queue_ae',
             'density_field_ae','speed_field_ae','field_occupancy_mismatch')
    # Ask the inherited summarizer to emit its exact empty-cell schema. None
    # means unmeasured, never a zero-valued observation or a computed interval.
    rows=[dict(model='SUMO_RNG_floor',plant='recorded',arm='native',world=w,
               seed=969400+w,horizon_s=h,**{m:None for m in metrics})
          for w in range(18) for h in A.HS]
    summarize,adapter=panel_summarizer(A,18)
    summary=[r for r in summarize(rows) if r['arm']=='native']
    for r in summary:
        assert r['estimate']==r['lo']==r['hi']=='' and r['n_worlds']==0
        r['status']='blocked_saved_state_canary'
    A.write_csv(out/'C1FLOOR_error_horizon.csv',summary)
    write(out/'C1FLOOR_ANALYSIS.json',dict(status=receipt['status'],scientific_estimates=0,
          scientific_intervals=0,rows=len(summary),blocked_receipt_sha256=digest(receipt_path),
          upstream_analyzer_sha256=digest(upstream),analyzer_sha256=digest(__file__),
          adapter=adapter,csv_sha256=digest(out/'C1FLOOR_error_horizon.csv')))
    print(json.dumps(dict(status=receipt['status'],empty_horizon_rows=len(summary),
                         actual_sim_s=receipt['actual_sim_s'],charged_sim_s=receipt['charged_sim_s'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--labels',type=Path)
    p.add_argument('--blocked-receipt',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--upstream',type=Path,default=UPSTREAM)
    p.add_argument('--chaos',action='store_true')
    a=p.parse_args()
    if a.blocked_receipt:
        assert a.labels is None
        blocked(a.blocked_receipt,a.out,a.upstream)
    else:
        assert a.labels is not None
        analyze(a.labels,a.out,a.upstream,a.chaos)
