"""Independently reconcile every displayed table number with the analysis JSON."""
from pathlib import Path
import argparse
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/paper_results/c1pidiag'


def numbers(text):return [float(x.replace('−','-')) for x in re.findall(r'[+−-]?\d+(?:\.\d+)?',text)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true',help='Verify without writing a new receipt')
    args=parser.parse_args()
    path=ROOT/'docs/paper_results/C1PIDIAG.md';text=path.read_text(encoding='utf-8');lines=text.splitlines()
    assert len(lines)<=60
    summary=json.loads((OUT/'SUMMARY.json').read_text())
    keys={(r['panel'],r['mode'],r['cohort'],r['config']):r for r in summary['summary']}
    phase='original';checked=0
    for line in lines:
        if line.startswith('## Genuine-history'):phase='aux'
        if line.startswith('## Feedback'):break
        if not (line.startswith('| PI /') or line.startswith('| Wall /')):continue
        label,absolute,delta60,delta120=[v.strip() for v in line.strip('|').split('|')]
        cohort='pi_u_same' if label.startswith('PI') else 'wall_same'
        name=label.split(' / ',1)[1]
        tag='c1struct' if name=='old C1' else ('bbtraj' if name=='BB-traj' else ('wm13e' if name.startswith('13E') else 'wm2'))
        hist='h60' if phase=='aux' else ('h5' if cohort=='wall_same' else 'snapshot')
        lane='mobil' if 'MOBIL' in name else 'original'
        component='residual_net_off' if 'residual-net' in name else ('yield_off' if 'yield off' in name else 'baseline')
        cfg='__'.join((tag,hist,'inf','A1',lane,component))
        row=keys[(phase,'aux_tf' if phase=='aux' else 'tf',cohort,cfg)]
        for h,cell in zip(('60','120'),absolute.split('/')):
            point=row['horizons'][h]['own']['point'];got=numbers(cell)
            assert (point is None and not got) or (len(got)==1 and abs(got[0]-point)<=.0051),(label,h,cell)
        for h,cell in (('60',delta60),('120',delta120)):
            s=row['horizons'][h]['paired_wm_minus_bb'];got=numbers(cell)
            want=[] if s['point'] is None else ([s['point']] if tag=='bbtraj' else [s['point'],s['lo'],s['hi']])
            assert len(got)==len(want) and all(abs(a-b)<=.0051 for a,b in zip(got,want)),(label,h,cell)
        checked+=1
    assert checked==20
    audit=json.loads((OUT/'FINAL_RUN_AUDIT.json').read_text())
    assert audit['canonical_units']==4730 and audit['owned_registered_processes_active']==0
    for p,h in [(OUT/'RAW_UNITS.jsonl.gz',audit['raw_sha256']),
                (OUT/'aux_feedback/AUX_FEEDBACK.jsonl.gz',audit['supplement_sha256'])]:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==h
    coverage=json.loads((OUT/'COVERAGE.json').read_text())
    for cohort,n in [('pi_u_same',40),('wall_same',13)]:
        c=coverage['cohorts'][cohort]
        assert c['c1struct_and_frozen_BBs']['overlap_count']==n
        assert c['13e']['overlap_count']==c['wm2']['overlap_count']==0
    value=dict(report_lines=len(lines),table_rows_reconciled=checked,all_table_values_match=True,
        canonical_units=4730,raw_hashes_verified=True,coverage_verified=True,
        report_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if args.check_only:
        assert value==json.loads((OUT/'FINAL_REPORT_CHECK.json').read_text()),'saved report receipt differs'
    else:
        with (OUT/'FINAL_REPORT_CHECK.json').open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
