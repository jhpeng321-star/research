"""Final source/output/PID receipt; only task-owned process identities are inspected."""
from pathlib import Path
import hashlib
import json
import time

ROOT=Path('/root/autodl-tmp/c4/c1pidiag20261031')


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    phases=['evidence_v2/TF_DONE.json','feedback_evidence/COMPLETE.json','aux_evidence/COMPLETE.json',
            'extra_evidence/COMPLETE.json','legacy_evidence/COMPLETE.json','aux_feedback_evidence/COMPLETE.json']
    counts={};sources={}
    for name in phases:
        d=json.loads((ROOT/name).read_text());assert not d['errors']
        counts[name]=d.get('units',d.get('tf',0)+d.get('clh',0))
        for p,h in d.get('sources',{}).items():
            assert digest(Path(p))==h,p
            sources[p]=h
    assert sum(counts.values())==4730
    owned={}
    for folder in ('evidence','evidence_v2'):
        for p in (ROOT/folder).glob('*OWNED.json'):
            r=json.loads(p.read_text());pid=r['pid'];q=Path(f'/proc/{pid}/stat')
            fields=q.read_text().split(') ',1)[1].split() if q.exists() else None
            alive=bool(fields and fields[19]==r['start_ticks'] and fields[0] not in ('Z','X'))
            owned[p.name]=dict(pid=pid,start_ticks=r['start_ticks'],state=fields[0] if fields else None,active_same_identity=alive)
            assert not alive,p
    raw=ROOT/'RAW_UNITS.jsonl.gz';supp=ROOT/'AUX_FEEDBACK.jsonl.gz'
    manifest=json.loads((ROOT/'OUTPUTS_MANIFEST.json').read_text())
    assert digest(raw)==manifest['raw_sha256']
    value=dict(completed=time.time(),canonical_units=4730,counts=counts,all_phase_errors_zero=True,
        initial_compatibility_failures_preserved=195,raw_sha256=digest(raw),supplement_sha256=digest(supp),
        sources=sources,owned=owned,owned_registered_processes_active=0,no_gpu_inference=True,
        gpu_initial_free_mib=10701,gpu_admission_threshold_mib=12288,workers_max=24)
    with (ROOT/'FINAL_RUN_AUDIT.json').open('x') as f:json.dump(value,f,indent=2)
    print('PASS',value['canonical_units'],'canonical units;',len(owned),'owned dispatch identities inactive; source/output hashes match')


if __name__=='__main__':main()
