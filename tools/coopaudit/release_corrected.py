"""One explicit, registered release; no simulator launch and no stale-worker resume."""
import hashlib
import json
from pathlib import Path
import subprocess
from tools.coopaudit.accounting import write
from tools.coopaudit.design import sha
from tools.coopaudit.remote import run


def main():
    evidence=Path(__file__).resolve().parent/'evidence'
    selection=json.loads((evidence/'CORRECTED_DESIGN.json').read_text())
    release=dict(status='PASS',registered=True,controllers=['mappo','wall'],
                 primary_controllers=['mappo','wall','sumocf'],reserved_controller_count=3,
                 artifact_document_sha256=selection['artifact_document_sha256'],
                 selected_design_sha256=hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest(),
                 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 registration_entry='S-COOPMEASURE CORRECTED v2 REGISTERED BEFORE NEW SUMO (2026-11-01 fresh-session recovery)')
    registration=Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md').read_text(encoding='utf-8')
    assert release['registration_entry'] in registration
    release['registration_document_sha256']=hashlib.sha256(registration.encode()).hexdigest()
    code='''
import fcntl,json,os,signal,time
from pathlib import Path
root=Path('/root/coopaudit20261031')
selection=__SELECTION_LITERAL__
release=__RELEASE_LITERAL__
hold=json.loads((root/'ARTIFACT_HOLD.json').read_text())
assert hold['active'] is True
worker=hold['worker']
proc=Path('/proc')/str(worker['pid'])
fields=(proc/'stat').read_text().split()
assert fields[21]==worker['proc_start_ticks'] and fields[2]=='T'
assert str(root).encode() in (proc/'cmdline').read_bytes() or b'tools.coopaudit.runner' in (proc/'cmdline').read_bytes()
assert not (proc/'task'/str(worker['pid'])/'children').read_text().strip()
assert not (root/'ARTIFACT_RELEASE.json').exists()
assert not (root/'STOP').exists()
charged=sum(json.loads(s)['sim_s'] for s in (root/'accounting.jsonl').read_text().splitlines())
assert charged==selection['retained_charged_sim_s']
assert charged+selection['canary_charge_sim_s']+selection['main_charge_sim_s']<=500000
storage=Path(selection['storage_root'])
assert storage.parent.is_dir() and not storage.exists()
assert __import__('shutil').disk_usage(storage.parent).free>=1024**3
storage.mkdir()
# SIGKILL only the receipt-identified OWN stopped obsolete worker. Never SIGCONT.
os.kill(worker['pid'],signal.SIGKILL)
for _ in range(100):
    if not (proc/'stat').exists() or (proc/'stat').read_text().split()[2]=='Z':
        break
    time.sleep(.05)
else:
    raise RuntimeError('owned stale worker did not exit')
with (root/'RUN.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'SELECTED_DESIGN_PRE_V2.json').write_bytes((root/'SELECTED_DESIGN.json').read_bytes())
    (root/'SELECTED_DESIGN.json').write_text(json.dumps(selection,sort_keys=True,indent=1))
    (root/'ARTIFACT_HOLD_PRE_V2.json').write_bytes((root/'ARTIFACT_HOLD.json').read_bytes())
    (root/'ARTIFACT_HOLD.json').write_text(json.dumps(dict(hold,active=False,released_unix=time.time(),release=release)))
    (root/'ARTIFACT_RELEASE.json').write_text(json.dumps(release,sort_keys=True,indent=1))
    # Preserve the old priority flag as a receipt; clear via rename, not deletion.
    (root/'YIELD').rename(root/'YIELD_PRE_V2')
print(json.dumps(dict(ok=True,release=release,retired_worker=worker,charged_sim_s=charged,storage=str(storage)),indent=1))
'''.replace('__SELECTION_LITERAL__',repr(selection)).replace('__RELEASE_LITERAL__',repr(release))
    result=run('gpuR',code,timeout=30)
    print(result.stdout,end='')
    print(result.stderr,end='')
    if result.returncode:
        raise SystemExit(result.returncode)
    write(evidence/'ARTIFACT_HOLD_PRE_V2.json',json.loads((evidence/'ARTIFACT_HOLD.json').read_text()))
    write(evidence/'SELECTED_DESIGN_PRE_V2.json',json.loads((evidence/'SELECTED_DESIGN.json').read_text()))
    write(evidence/'SELECTED_DESIGN.json',selection)
    write(evidence/'ARTIFACT_HOLD.json',dict(active=False,release=release))
    write(evidence/'ARTIFACT_RELEASE.json',release)
    write(evidence/'V2_RELEASE_VERIFIED.json',json.loads(result.stdout))


if __name__=='__main__':
    main()
