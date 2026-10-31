"""Verify the terminal blocker, complete accounting, binding and honest empty CSV."""
from pathlib import Path
import csv
import json
import argparse
import hashlib
from core import digest, write

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EVIDENCE=HERE/'evidence/evidence_1793437482207139800'
OUT=ROOT/'docs/paper_results'


def main(output=None):
    stop=json.loads((EVIDENCE/'BLOCKED.json').read_text())
    ledger=[json.loads(s) for s in (EVIDENCE/'accounting.jsonl').read_text().splitlines()]
    failures=[json.loads(p.read_text()) for p in (EVIDENCE/'jobs').glob('*/failure.json')]
    assert len(ledger)==len({r['id'] for r in ledger})==len(failures)==6
    assert {r['id'] for r in ledger}=={f['job']['id'] for f in failures}
    assert sum(r['sim_s'] for r in ledger)==stop['charged_sim_s']==9000
    assert sum(f['actual_sim_s'] for f in failures)==stop['actual_sim_s']==2400
    assert all(f['actual_sim_s']<=f['job']['charge'] for f in failures)
    assert stop['alternative_branches_run']==stop['panel_worlds_run']==0
    assert all(r['identical'] for r in stop['original_reproduction'].values())
    assert stop['first_drift_time_s']==300.5 and stop['first_drift_hdv_ids']==[101,104]
    assert all(p['state'] in ('exited','Z') for p in stop['processes'])
    binding=json.loads((EVIDENCE/'PREFLIGHT_BINDING.json').read_text())
    for n in ('worker.py','run.py','core.py','design.json'):
        assert digest(HERE/n)==binding['source_hashes'][n],n
    assert binding['guards_passed']==54 and not binding['filename_exposure']
    assert not binding['exact_alternative_exposure']
    with (OUT/'C1FLOOR_error_horizon.csv').open(newline='') as f:
        reader=csv.DictReader(f);columns=reader.fieldnames;rows=list(reader)
    expected=['model','plant','arm','horizon_s','metric','estimate','lo','hi','n_worlds',
              'world_stat','units','bootstrap_draws','bootstrap_seed','status']
    assert columns==expected
    assert len(rows)==98 and {int(r['horizon_s']) for r in rows}=={10,30,60,90,120,180,300}
    assert len({(r['horizon_s'],r['metric']) for r in rows})==len(rows)
    assert all(r['estimate']==r['lo']==r['hi']=='' and r['n_worlds']=='0' and
               r['status']=='blocked_saved_state_canary' for r in rows)
    upstream=Path('E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED_bootstrap.csv')
    if upstream.exists():
        with upstream.open(newline='') as f:assert next(csv.reader(f))==columns
    lines=(OUT/'C1FLOOR.md').read_text(encoding='utf-8').splitlines()
    assert len(lines)<=40,len(lines)
    analysis=json.loads((OUT/'C1FLOOR_ANALYSIS.json').read_text())
    assert analysis['scientific_estimates']==analysis['scientific_intervals']==0
    assert analysis['csv_sha256']==digest(OUT/'C1FLOOR_error_horizon.csv')
    registration=Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    entries=[s for s in registration.read_text(encoding='utf-8').splitlines()
             if s.startswith('- S-C1FLOOR BLOCKED AT BITWISE RESTORE GATE')]
    assert len(entries)==1
    result=dict(status='PASS_BLOCKED_DELIVERY_NOT_FLOOR',actual_sim_s=2400,
                charged_sim_s=9000,attempts=6,alternative_branches=0,csv_rows=len(rows),
                csv_columns=columns,report_lines=len(lines),source_bindings_match=True,
                existing_c1closed_export_present=upstream.exists(),schema_matches_pinned_c1closed=True,
                registration_entry_sha256=hashlib.sha256(entries[0].encode()).hexdigest(),
                outputs={p.name:digest(p) for p in OUT.glob('C1FLOOR*') if p.is_file()})
    print(json.dumps(result,indent=1))
    if output is not None:write(output,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path)
    args=parser.parse_args();main(args.out)
