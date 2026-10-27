"""No-dispatch handoff verification; fetch compact SHA-bound S-C2H60 receipts."""
import hashlib
import json
from pathlib import Path
import subprocess

from tools.loopresume.c1collect import PRELUDE

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = PRELUDE + '''
from tools.loopresume.c2h60_box import verify_updates
binding=loop.read_json(r/'C2H60_BINDINGS.json')
launch=loop.read_json(r/'C2H60_LAUNCH.json')
progress=loop.read_json(r/'C2H60_PROGRESS.json')
assert previous.live(launch) and previous.live(progress['child'])
assert not previous.live(binding['superseded']['coordinator'])
assert not previous.live(binding['superseded']['child'])
assert loop.sha256(binding['warm_start']['checkpoint'])==binding['warm_start']['sha256']
for name,sha in binding['source'].items(): assert loop.sha256(root/name)==sha,name
assert loop.sha256(a/'horizon_gate_current.json')==binding['gate_sha256']
out=a/'loop/policy/c02.h60'
rows=[]
for line in (out/'updates.jsonl').read_text().splitlines():
 try: rows.append(json.loads(line))
 except json.JSONDecodeError: pass
assert len(rows)>=10
verify_updates(rows,233,len(rows))
gate=loop.read_json(out/'gate_panel_result.json')
assert gate['passed'] is True and gate['phase_update']==10
assert gate['sha256']==loop.sha256(a/'gate_panel.json')
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg)
assert state['sim_s_charged']==176473
assert all(state['cycles'][2]['phases'][name]['status']=='pending' for name in ('collect','convert','wm_ft','check'))
assert cfg['c2h60']['time_limit_waived'] is True
os.chdir(root); os.environ['C4_ALLOWED_ROOTS']=':'.join(cfg['policy']['allowed_roots'])
from wms.control.policy14_cli import _read_config
raw=_read_config(cfg['policy']['config'])
assert raw['run']['gpu_memory_fraction']==0.65
assert raw['paircrit']['enabled'] is True
assert raw['horizon']['curriculum_updates']==[0,150,233]
samples=[json.loads(line) for line in (r/'C2H60_MEMORY.jsonl').read_text().splitlines()]
own_samples=[s for s in samples if s.get('update',0)>=238 and 'used_memory_mib' in s]
assert own_samples
observation=loop.read_json(r/'C2H60_OBSERVATION.json')
assert observation['updates']>=10 and observation['coordinator_live'] and observation['child_live']
names=('C2H60_SOURCE.json','C2H60_TESTS.json','C2H60_BINDINGS.json','C2H60_LAUNCH.json',
 'C2H60_STOP_REQUEST.json','C2H60_OBSERVATION.json')
receipts={name:{'sha256':loop.sha256(r/name),'document':loop.read_json(r/name)} for name in names}
result={'t':loop.now(),'verified':True,'updates':len(rows),'last_update':rows[-1]['update'],
 'phase10_gate':gate,'charged':176473,'pending':['collect','convert','wm_ft','check'],
 'source':binding['source'],'memory_samples':own_samples,'receipts':receipts,
 'registration_sha256':loop.sha256(cfg['registration']),
 'effective_config':{key:raw[key] for key in ('run','horizon','paircrit','exploration')},
 'prerequisite_config_sha256':loop.sha256(root/'configs/c4_policy14_c2prep.toml')}
loop.write_atomic(r/'C2H60_VERIFIED.json',result)
loop.append(a/'loop/cycle_ledger.jsonl',{'event':'S-C2H60_running_verified','t':loop.now(),
 'updates':len(rows),'last_update':rows[-1]['update'],'receipt_sha256':loop.sha256(r/'C2H60_VERIFIED.json')})
loop.append(r/'C1_EVENTS.jsonl',{'event':'S-C2H60_running_verified','t':loop.now(),
 'updates':len(rows),'last_update':rows[-1]['update'],'receipt_sha256':loop.sha256(r/'C2H60_VERIFIED.json')})
print(json.dumps({'sha256':loop.sha256(r/'C2H60_VERIFIED.json'),'document':result}))
'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'gpuA',
                             '/root/miniconda3/bin/python -B -'],
                            input=source, text=True, encoding='utf-8', capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    doc = json.loads(result.stdout)
    for name, sha in doc['document']['source'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    register = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    text = register.read_text(encoding='utf-8')
    amendment = text[text.index('## S-C2H60 — prospective'):]
    doc['markdown_registration'] = {'path': str(register),
        'sha256': hashlib.sha256(register.read_bytes()).hexdigest(), 'amendment': amendment}
    target = ROOT / 'tools/loopresume/receipts_c2h60/VERIFIED.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(doc, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'verified': True, 'updates': doc['document']['updates'],
                      'last_update': doc['document']['last_update'],
                      'remote_receipt_sha256': doc['sha256'], 'local_receipt': str(target)}))


if __name__ == '__main__':
    main()
