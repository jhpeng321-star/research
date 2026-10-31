"""Independent delivery arithmetic, population, provenance and file checks."""
from pathlib import Path
import csv
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'docs/paper_results'


def main():
    receipt=json.loads((OUT/'C1CLOSED_ANALYSIS.json').read_text())
    assert receipt['tasks']==480 and receipt['worlds']==40 and receipt['K']==8
    assert receipt['analyzer_sha256']==hashlib.sha256((HERE/'analyze.py').read_bytes()).hexdigest()
    for name,digest in receipt['outputs'].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest
    raw=list(csv.DictReader((OUT/'C1CLOSED_per_world.csv').open(encoding='utf-8')))
    paired=list(csv.DictReader((OUT/'C1CLOSED_paired.csv').open(encoding='utf-8')))
    summary=list(csv.DictReader((OUT/'C1CLOSED_bootstrap.csv').open(encoding='utf-8')))
    assert len(raw)==receipt['raw_rows'] and len(paired)==receipt['paired_rows']
    assert len(summary)==receipt['summary_rows']
    keys=[(r['model'],r['plant'],r['world'],r['arm'],r['horizon_s']) for r in raw]
    assert len(keys)==len(set(keys)) and {int(r['world']) for r in raw}==set(range(40))
    for r in raw:
        if not r.get('J_bias'): continue
        assert math.isclose(float(r['J_bias']),sum(float(r[c+'_bias']) for c in ('source','upstream','merge','discharge')),abs_tol=1e-3)
    for r in paired:
        if r.get('position'):
            assert int(r['n_hdv'])>0
            assert math.isclose(float(r['position']),float(r['wm_position'])-float(r['bb_position']),abs_tol=1e-8)
        elif r.get('n_hdv'):
            assert int(r['n_hdv'])==0 and not r['wm_position'] and not r['bb_position']
    for r in summary:
        if r['estimate']:
            assert all(math.isfinite(float(r[k])) for k in ('estimate','lo','hi'))
            assert float(r['lo'])<=float(r['hi']) and 0<int(r['n_worlds'])<=40
        else:
            assert not r['lo'] and not r['hi'] and int(r['n_worlds'])==0
    assert len((OUT/'C1CLOSED.md').read_text(encoding='utf-8').splitlines())<=60
    from PIL import Image
    with Image.open(OUT/'C1CLOSED_error_horizon.png') as image:
        assert image.format=='PNG' and image.width>1000 and image.height>1500
        image.verify()
    check=dict(schema='c1closed_delivery_check/1',passed=True,worlds=40,tasks=480,
        analysis_sha256=hashlib.sha256((OUT/'C1CLOSED_ANALYSIS.json').read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checks=['unique_world_arm_horizon_keys','complete_480_task_K8_panel','input_output_hashes',
                'additive_signed_components','paired_common_cohort_differences','unavailable_not_zero',
                'finite_cluster_CIs','report_60_line_limit','PNG_decode_dimensions'])
    target=HERE/'evidence/DELIVERY_VERIFIED.json'
    with target.open('x') as f:json.dump(check,f,indent=1)
    print(json.dumps(check))


if __name__=='__main__':main()
