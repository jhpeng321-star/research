"""Immutable scoring-only real observables/endpoint joins; never a model input."""
import argparse
from pathlib import Path

from tools.rb2int.audit import ROOT, WORK
from tools.rbeval.core import ARMS, HS, REGIONS, breakdown, digest, load, save_new


def join_real(first,second):
    rows = [r for data in (first,second) for r in data['episodes'] if r['arm'] in ARMS]
    keys = {(r['seed'],r['arm']) for r in rows}
    expected = {(s,a) for s in (*range(969400,969440),*range(969550,969600)) for a in ARMS}
    if keys != expected or len(rows) != len(keys):
        raise ValueError('complete disjoint40+50 five-arm recorded corpus required')
    for row in rows:
        for region in REGIONS:
            bins = row['bins'][region]
            if len(bins['speed']) != 66 or len(bins['post_counts']) != 60:
                raise ValueError('recorded bin protocol differs')
            for h in HS:
                event = breakdown(bins['speed'][:6+h//10])
                stored = bins['events'][str(h)]
                if (event['event'],event['onset_s'],event.get('preexisting',False)) != (stored['event'],stored['onset_s'],stored.get('preexisting',False)):
                    raise ValueError('recorded detector differs from frozen BIC')
    return dict(schema='rb2int-real-observables/1',episodes=rows,
        baseline_fit_worlds=list(range(969400,969432)),evaluation_only=True,model_inputs=False)


def join_endpoints(old,new):
    if any(d['schema'] != 'wmphys-factual-endpoints/1' or d['horizon_s'] != 60 or d['t0_s'] != 300 for d in (old,new)):
        raise ValueError('factual endpoint protocol differs')
    rows = old['episodes']+new['episodes']
    if len(rows) != 350 or len({(r['seed'],r['arm']) for r in rows}) != 350:
        raise ValueError('reuse original100 plus new250 unique endpoint targets')
    return dict(schema='wmphys-factual-endpoints/1',horizon_s=60,t0_s=300,episodes=rows,
        evaluation_only=True,model_inputs=False,training=False,sumo_runs=0,
        matching='registered exact observable t0 then persistent converted VID; originals reused')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if not out.parent.is_dir() or out.exists():
        raise ValueError('exclusive scoring-only output root required')
    real_paths = [ROOT/'tools/rbeval/real_observables.json',
        WORK/'c4-wmbreakdown/docs/paper_results/wmbreakdown/wmdata2_observables.json']
    endpoint_paths = [ROOT/'docs/rbint/hdv60/FACTUAL_065e9853.json',ROOT/'docs/rb2int/FACTUAL50.json']
    real = join_real(*(load(p) for p in real_paths))
    endpoints = join_endpoints(*(load(p) for p in endpoint_paths))
    out.mkdir()
    save_new(out/'REAL_OBSERVABLES.json',real)
    save_new(out/'FACTUAL_ENDPOINTS.json',endpoints)
    save_new(out/'SCORING_SOURCE.json',dict(real_inputs={str(p):digest(p) for p in real_paths},
        endpoint_inputs={str(p):digest(p) for p in endpoint_paths},
        real_sha256=digest(out/'REAL_OBSERVABLES.json'),endpoint_sha256=digest(out/'FACTUAL_ENDPOINTS.json'),
        source_sha256=digest(__file__),parameter_fitting=False,model_inputs=False))
    print('SCORING DATA READY',len(real['episodes']),len(endpoints['episodes']))


if __name__ == '__main__':
    main()
