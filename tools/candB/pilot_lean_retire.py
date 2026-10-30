"""Retire only receipt-bound superseded S-PILOTB processes, retaining all data."""
import json
from pathlib import Path
import subprocess
import time

CODE = r'''
import ctypes,json,os,signal,time
from pathlib import Path
root=Path('/root/autodl-tmp/c4/candB_pilot_final_run_EXPLORATORY_20261030')
for name in ('STOP','continue_STOP','dry_EXPLORATORY/STOP'):
    (root/name).touch(exist_ok=True)
doc=dict(epoch=time.time(),reason='MAIN23:50 EXPLORATORY-LEAN relaunch',owners={},files={})
for name in ('continue_trainer_OWNERSHIP.json','pilot_OWNERSHIP.json'):
    if (root/name).exists(): raise RuntimeError('Unexpected dispatched successor; inspect before retirement')
for name in ('continue_OWNERSHIP.json','dry_OWNERSHIP.json','OWNERSHIP.json'):
    owner=json.loads((root/name).read_text()); proc=Path('/proc')/str(owner['pid'])
    entry=dict(owner=owner)
    if (proc/'stat').exists():
        fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if fields[19]!=str(owner['start_ticks']) or 'tools.candB.pilot_' not in argv:
            raise RuntimeError('Owned incarnation mismatch')
        libc=ctypes.CDLL(None,use_errno=True)
        fd=libc.syscall(434,owner['pid'],0)
        if fd<0: raise OSError(ctypes.get_errno(),'pidfd_open')
        # Verify again AFTER opening the stable process handle.
        assert (proc/'stat').read_text().rsplit(')',1)[1].split()[19]==str(owner['start_ticks'])
        entry.update(argv=argv,cpu_ticks=int(fields[11])+int(fields[12]),state=fields[0])
        if libc.syscall(424,fd,signal.SIGTERM,0,0)!=0:
            raise OSError(ctypes.get_errno(),'pidfd_send_signal')
        os.close(fd)
        entry['signal']='SIGTERM via verified pidfd'
    doc['owners'][name]=entry
time.sleep(3)
for name,entry in doc['owners'].items():
    proc=Path('/proc')/str(entry['owner']['pid'])
    entry['exited']=not (proc/'stat').exists() or (proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
    if not entry['exited']: raise RuntimeError('Owned process did not exit: '+name)
for name in ('dry_EXPLORATORY/updates.jsonl','dry_EXPLORATORY/frozen_scoring_baseline.json','dry.log','GPU_ADMISSION.jsonl'):
    p=root/name
    doc['files'][name]=dict(exists=p.exists())
    if p.exists(): doc['files'][name].update(size=p.stat().st_size,tail=p.read_text()[-7000:])
doc['sunk_wall_s']=doc['epoch']-1793366535.895446
(root/'LEAN_RETIREMENT.json').write_text(json.dumps(doc,indent=2))
print(json.dumps(doc))
'''

if __name__ == '__main__':
    result = subprocess.run(['ssh', 'gpuA', '/root/autodl-tmp/c4/candB_rankcheck_0264b967_20261030/venv/bin/python', '-'],
        input=CODE, text=True, capture_output=True, timeout=30)
    if result.returncode:
        print(result.stderr)
        raise SystemExit(result.returncode)
    doc = json.loads(result.stdout)
    target = Path(__file__).with_name(f'pilot_lean_retirement_{int(time.time())}.json')
    with target.open('x') as stream:
        json.dump(doc, stream, indent=2)
    print(json.dumps(doc))
