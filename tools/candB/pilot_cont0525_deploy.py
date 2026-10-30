"""Prepare the owner05:25 CONT0525 runtime/CPU proof/dispatch. No unowned training."""
import argparse
import json
from pathlib import Path
import subprocess

from tools.candB.pilot_lean_deploy import PYTHON

PARENT = '/root/autodl-tmp/c4/candB_pilot_run_RETRY0300_v2_20261031/pilot_EXPLORATORY-LEAN_RETRY0300'
PARENT_AUTHORITY = '/root/autodl-tmp/c4/candB_pilot_RETRY0300_v5_EXPLORATORY-LEAN_20261031/authority_gen/PILOT_RETRY_AUTHORITY.json'
RUNTIME = '/root/autodl-tmp/c4/candB_pilot_CONT0525_v2_EXPLORATORY-LEAN_20261031'
ROOT = '/root/autodl-tmp/c4/candB_pilot_run_CONT0525_v2_20261031'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'cpu', 'status', 'dispatch'))
    args = parser.parse_args()
    code = f'''
import hashlib,json,os,subprocess,time
from pathlib import Path
runtime=Path({RUNTIME!r}); root=Path({ROOT!r}); python={PYTHON!r}; os.chdir(runtime)
env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='',C4_ALLOWED_ROOTS='/root/autodl-tmp/c4')
if {args.action!r}=='prepare':
    manifest=json.loads(Path('SOURCE_MANIFEST.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    Path('authority_gen').mkdir(exist_ok=True)
    subprocess.run([python,'-B','-m','tools.candB.pilot_cont0525_authority','--exploratory-owner-20261030',
        'build','authority_gen/PILOT_CONT0525_AUTHORITY.json','--parent-authority',{PARENT_AUTHORITY!r}],env=env,check=True)
elif {args.action!r}=='cpu':
    tag=str(int(time.time()))
    assert not Path(f'pilot_CONT0525_CPU_OWNERSHIP_{{tag}}.json').exists(),'recover existing CPU job'
    command=[python,'-B','-m','tools.candB.pilot_cont0525_cpu']
    with Path(f'pilot_CONT0525_CPU_{{tag}}.log').open('x') as log:
        child=subprocess.Popen(command,env=dict(env,PILOT_CONT0525_CPU_TAG=tag),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    Path(f'pilot_CONT0525_CPU_OWNERSHIP_{{tag}}.json').write_text(json.dumps(dict(pid=child.pid,command=command,epoch=time.time())))
    print('CPU_TAG',tag)
elif {args.action!r}=='status':
    for name in sorted(Path('.').glob('pilot_CONT0525_CPU_OWNERSHIP_*.json')):
        print(name.name,Path(name).read_text()[:400])
    for result in sorted(Path('.').glob('pilot_CONT0525_CPU_RESULT_*.json')):
        doc=json.loads(result.read_text())
        print(result.name,json.dumps(dict(returncode=doc['returncode'],seconds=doc['seconds'])))
        if doc['returncode']!=0:
            print(doc['stdout'][-1500:]); print(doc['stderr'][-1500:])
else:
    result=max(Path('.').glob('pilot_CONT0525_CPU_RESULT_*.json'))
    assert json.loads(result.read_text())['returncode']==0,'latest CPU proof must pass before dispatch'
    if root.exists():
        assert not any((root/name).exists() for name in
            ('OWNERSHIP.json','TRAINER_OWNERSHIP.json','EXIT.json')),'recover existing dispatch first'
    else:
        root.mkdir()
    with (root/'supervisor.log').open('a' if (root/'supervisor.log').exists() else 'x') as log:
        child=subprocess.Popen([python,'-B','-u','-m','tools.candB.pilot_cont0525_supervise','--root',str(root),
            '--python',python],env=dict(env,PILOT_CONT0525_CPU_RESULT=result.name),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(15):
        if (root/'TRAINER_OWNERSHIP.json').exists() or child.poll() is not None: break
        time.sleep(1)
    for name in ('OWNERSHIP.json','TRAINER_OWNERSHIP.json','EXIT.json'):
        if (root/name).exists(): print(name,(root/name).read_text())
    assert child.poll() is None, 'supervisor exited; recover receipts'
parent=Path({PARENT!r})
if parent.is_dir():
    doc=json.loads((parent/'RUN_SUMMARY.json').read_text())
    print('PARENT',json.dumps(dict(status=doc.get('status'),error=doc.get('error'),
        label=doc.get('label'),rows=len(doc.get('updates',[])),
        last_update=doc.get('updates',[-1])[-1].get('update') if doc.get('updates') else None,
        phase_start=doc.get('operational_recovery',{{}}).get('phase_start'))))
    for name in ('policy_latest.pt','frozen_scoring_baseline.json'):
        data=(parent/name).read_bytes()
        print('ARTIFACT',name,len(data),hashlib.sha256(data).hexdigest())
'''
    result = subprocess.run(['ssh', 'gpuA', '/root/miniconda3/bin/python', '-'],
        input=code, text=True, capture_output=True, timeout=120)
    import time
    receipt = Path(__file__).with_name(f'pilot_cont0525_{args.action}_{int(time.time())}.json')
    with receipt.open('x') as stream:
        json.dump(dict(action=args.action, runtime=RUNTIME, root=ROOT, returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr), stream, indent=1)
    print(result.stdout+result.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
