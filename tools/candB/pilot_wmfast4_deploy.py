"""Prepare S4 CPU/authority proof, then exclusively hand off the waiting controller."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from tools.candB.pilot_wmfast_dispatch import ROOT
from tools.candB.pilot_lean_deploy import PYTHON, RUNTIME as PARENT_RUNTIME

RUNTIME = '/root/autodl-tmp/c4/candB_pilot_WMFAST_S4_v3_EXPLORATORY-LEAN_20261031'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'cpu', 'status', 'handoff'))
    args = parser.parse_args()
    code = f'''
import hashlib,json,os,subprocess,time
from pathlib import Path
runtime=Path({RUNTIME!r}); root=Path({ROOT!r}); python={PYTHON!r}
os.chdir(runtime)
env=dict(os.environ,PYTHONPATH='src:.',C4_ALLOWED_ROOTS='/root/autodl-tmp/c4')
cpuenv=dict(env,CUDA_VISIBLE_DEVICES='')
def alive(owner):
    proc=Path('/proc')/str(owner['pid'])/'stat'
    return proc.exists() and proc.read_text().rsplit(')',1)[1].split()[19]==str(owner['start_ticks'])
if {args.action!r}=='prepare':
    manifest=json.loads(Path('SOURCE_MANIFEST.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    Path('authority_gen').mkdir(exist_ok=True)
    command=[python,'-B','-m','tools.candB.pilot_wmfast4_authority','--exploratory-owner-20261030']
    subprocess.run(command+['build','authority_gen/PILOT_WMFAST4_AUTHORITY.json',
        '--parent-authority',{PARENT_RUNTIME!r}+'/authority_gen/PILOT_EXPLORATORY_LEAN_AUTHORITY.json'],
        env=cpuenv,check=True)
    subprocess.run(command+['verify','authority_gen/PILOT_WMFAST4_AUTHORITY.json'],env=cpuenv,check=True)
    print(json.dumps(dict(source=json.loads(Path('SOURCE.json').read_text())['commit'],files=len(manifest))))
elif {args.action!r}=='cpu':
    command=[python,'-B','-u','-m','tools.candB.pilot_wmfast4_cpu']
    with Path('pilot_WMFAST4_CPU_OWNERSHIP.json').open('x') as receipt:
        with Path('pilot_WMFAST4_CPU.log').open('x') as log:
            child=subprocess.Popen(command,env=cpuenv,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        owner=dict(pid=child.pid,command=command,epoch=time.time())
        json.dump(owner,receipt)
    print(json.dumps(owner))
elif {args.action!r}=='status':
    for name in ('pilot_WMFAST4_CPU_OWNERSHIP.json','pilot_WMFAST4_CPU_RESULT.json','pilot_WMFAST4_CPU.log'):
        if Path(name).exists(): print(name,Path(name).read_text())
else:
    assert json.loads(Path('pilot_WMFAST4_CPU_RESULT.json').read_text())['returncode']==0
    assert not (root/'WMFAST4_CONTROL_OWNERSHIP.json').exists(),'recover existing S4 waiter'
    assert not (root/'continue_TRAINER_OWNERSHIP.json').exists(),'recover existing continuation'
    assert not (root/'STOP').exists()
    old=json.loads((root/'WMFAST_CONTROL_OWNERSHIP.json').read_text())
    assert old['pid']==37982 and str(old['start_ticks'])=='876126751'
    if alive(old):
        argv=(Path('/proc')/str(old['pid'])/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
        assert 'tools.candB.pilot_wmfast_control --root '+str(root) in argv
        with (root/'WMFAST_CONTROL_STOP').open('x') as stop:
            stop.write('Own waiting controller only; S4 successor. Trainer/baseline untouched.\\n')
            stop.flush(); os.fsync(stop.fileno())
        until=time.monotonic()+15
        while alive(old) and time.monotonic()<until: time.sleep(.2)
    assert not alive(old),'old controller still live; NO dispatch'
    assert not (root/'continue_TRAINER_OWNERSHIP.json').exists()
    retired=dict(epoch=time.time(),previous=old,old_incarnation_exited=True,
        action='own cooperative controller STOP only; trainer and supervisor preserved')
    with (root/'WMFAST4_CONTROL_HANDOFF.json').open('x') as stream: json.dump(retired,stream,indent=1)
    with (root/'WMFAST4_CONTROL.log').open('x') as log:
        child=subprocess.Popen([python,'-B','-u','-m','tools.candB.pilot_wmfast4_control',
            '--root',str(root),'--python',python],env=env,stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(10):
        if (root/'WMFAST4_CONTROL_OWNERSHIP.json').exists() or child.poll() is not None: break
        time.sleep(1)
    assert child.poll() is None,'S4 waiter exited; inspect, do not repeat dispatch'
    owner=json.loads((root/'WMFAST4_CONTROL_OWNERSHIP.json').read_text())
    assert alive(owner)
    print(json.dumps(dict(retirement=retired,successor=owner)))
'''
    result = subprocess.run(['ssh', '-o', 'ConnectTimeout=10', 'gpuA',
        '/root/miniconda3/bin/python', '-'], input=code, text=True,
        capture_output=True, timeout=60)
    receipt = Path(__file__).with_name(f'pilot_wmfast4_{args.action}_{int(time.time())}.json')
    with receipt.open('x') as stream:
        json.dump(dict(action=args.action,runtime=RUNTIME,returncode=result.returncode,
            stdout=result.stdout,stderr=result.stderr),stream,indent=1)
    print(json.dumps(dict(receipt=str(receipt), returncode=result.returncode)))
    print(result.stdout+result.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
