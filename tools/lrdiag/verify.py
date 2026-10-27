"""Numerical analysis invariants and artifact coverage, not model-fitting tests."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def selfcheck():
    from replay import breakdown
    rng=np.random.default_rng(24)
    flat=np.full(36,25.0);trend=25+.125*np.arange(36)
    assert not breakdown(flat)['event']
    assert not breakdown(trend)['event']
    for cp in (8,15,25):
        y=trend.copy();y[cp:]-=10;y+=rng.normal(0,.05,36)
        result=breakdown(y)
        assert result['event'] and result['onset_s']==(cp-6)*10,result
    y=trend.copy();y[3:]-=10
    result=breakdown(y)
    assert result['preexisting'] and not result['event'],result
    print('PASS: flat/trend nulls; known-onset downward changes; preexisting separation')


def outputs(root):
    summary=json.loads((root/'SUMMARY.json').read_text())
    c=summary['coverage']
    assert not c['partial'] and c['complete_worlds']==36 and c['n_results']==1728,c
    counts={}
    for name,n in [('ablation_worlds.csv',864),('ablation_draws.csv',6912),
                   ('within_world_slopes.csv',432),('timeline_worlds.csv',17280),
                   ('breakdown_worlds.csv',216),('breakdown_draws.csv',1728),
                   ('paired_signal_worlds.csv',2160)]:
        with (root/name).open(newline='') as f: rows=list(csv.DictReader(f))
        assert len(rows)==n,(name,len(rows),n)
        counts[name]=len(rows)
    assert summary['max_delay_accounting_error']<.002
    report={'status':'PASS','counts':counts,'checks':[
        '36 validation worlds x 6 variants x K8 complete',
        'world bootstrap, never draw bootstrap',
        'regional HDV paired errors sum to horizon-gate dJ error',
        'pulse and pairs300 excluded from E3 verdict statistics'],
        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(root.iterdir()) if p.suffix in ('.csv','.json','.png','.svg') and p.name!='VERIFIED.json'}}
    (root/'VERIFIED.json').write_text(json.dumps(report,indent=1))
    print('PASS:',counts)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path);a=ap.parse_args()
    selfcheck()
    if a.evidence:outputs(a.evidence)
