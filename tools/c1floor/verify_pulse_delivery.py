"""Independent delivery checks: ledger conservation, executed amplitude and CSV scope."""
from pathlib import Path
import csv
import json
import math
from core import digest,write

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OUT=REPO/'docs/paper_results'
E=HERE/'evidence/fallback_1793439838485955000'


def main():
    complete=json.loads((E/'pulse/COMPLETE.json').read_text())
    assert complete['ok'] and complete['worlds']==6 and complete['draws']==12
    assert all(p['state'] in ('exited','Z') for p in complete['owners'])
    prior=HERE/'evidence/evidence_1793437482207139800'
    old=[json.loads(p.read_text()) for p in (prior/'jobs').glob('*/failure.json')]
    fallback=[json.loads(p.read_text()) for p in (E/'jobs').glob('fallback_*/*.json') if p.name in ('result.json','failure.json')]
    pulses=[json.loads(p.read_text()) for p in (E/'jobs').glob('pulse_*/result.json')]
    assert len(old)==6 and len(fallback)==8 and len(pulses)==6
    actual=sum(r['actual_sim_s'] for r in old+fallback+pulses)
    charged=sum(json.loads(s)['sim_s'] for p in (prior/'accounting.jsonl',E/'fallback/accounting.jsonl',E/'pulse/accounting.jsonl') for s in p.read_text().splitlines())
    assert actual==complete['actual_sim_s']==21900
    assert charged==complete['charged_sim_s']==29700<=30000
    assert sorted(r['world']['seed'] for r in pulses)==list(range(969400,969406))
    for r in pulses:
        assert all(v['identical'] for v in r['reproduction'].values())
        assert len(r['pulses'])==2
        for p in r['pulses']:
            assert abs(p['delta_mps']-.01)<1e-10 and not p['persistent_speed_override']
            assert p['api']=='setPreviousSpeed'
            assert p['other_vehicle_speeds_unchanged']>0
    analysis=json.loads((OUT/'C1FLOOR_ANALYSIS.json').read_text())
    assert analysis['method']=='SUMO_tiny_perturbation_chaos' and analysis['worlds']==6
    assert digest(HERE/'analyze.py')==analysis['analyzer_sha256']
    assert digest(HERE/'evidence/C1CLOSED_analyze_source.py')==analysis['upstream_analyzer_sha256']
    assert digest(HERE/'evidence/PULSE_LABELS.json')==analysis['labels_sha256']==complete['labels_sha256']
    for name,h in analysis['outputs'].items():assert digest(OUT/name)==h
    with (OUT/'C1FLOOR_error_horizon.csv').open(newline='') as f:
        reader=csv.DictReader(f);rows=list(reader)
        assert reader.fieldnames==['model','plant','arm','horizon_s','metric','estimate','lo','hi','n_worlds','world_stat','units','bootstrap_draws','bootstrap_seed','status']
    assert len(rows)==266
    assert {int(r['horizon_s']) for r in rows}=={10,30,60,90,120,180,300}
    assert {r['model'] for r in rows}=={'SUMO_tiny_perturbation_chaos'}
    assert {r['arm'] for r in rows}=={'native'}
    for r in rows:
        assert r['bootstrap_draws']=='2000' and r['bootstrap_seed']=='0'
        for key in ('estimate','lo','hi'):
            assert r[key]=='' or math.isfinite(float(r[key]))
        if r['metric'] in ('position','speed') and r['horizon_s']=='300':
            assert r['n_worlds']=='0' and r['status']=='no_surviving_ids'
            assert all(r[k]=='' for k in ('estimate','lo','hi'))
    with (OUT/'C1FLOOR_per_world.csv').open(newline='') as f:worlds=list(csv.DictReader(f))
    with (OUT/'C1FLOOR_per_draw.csv').open(newline='') as f:draws=list(csv.DictReader(f))
    assert len(worlds)==42 and len(draws)==84
    for r in worlds+draws:
        assert abs(sum(float(r[c+'_pred']) for c in ('source','upstream','merge','discharge'))-float(r['J_pred']))<1e-8
    report=OUT/'C1FLOOR.md';assert len(report.read_text(encoding='utf-8').splitlines())<=40
    entry=(HERE/'evidence/PULSE_REGISTRATION_ENTRY.md').read_text(encoding='utf-8').strip()
    reg=Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    assert reg.read_text(encoding='utf-8').count(entry)==1
    receipt=dict(ok=True,worlds=6,draws=12,horizon_rows=266,world_rows=42,draw_rows=84,
                 verified_instantaneous_pulses=12,actual_sim_s=actual,charged_sim_s=charged,
                 report_lines=len(report.read_text(encoding='utf-8').splitlines()),report_sha256=digest(report),
                 analysis_sha256=digest(OUT/'C1FLOOR_ANALYSIS.json'),registration_entry_sha256=digest(HERE/'evidence/PULSE_REGISTRATION_ENTRY.md'))
    target=HERE/'evidence/PULSE_DELIVERY_VERIFIED_FINAL.json'
    if target.exists():
        assert json.loads(target.read_text())==receipt
    else:
        write(target,receipt)
    print(json.dumps(receipt,indent=1))


if __name__=='__main__':main()
