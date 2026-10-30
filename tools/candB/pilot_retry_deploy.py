"""Prepare the owner03:00 retry runtime/CPU proof. No training dispatch here."""
import argparse
import json
from pathlib import Path
import subprocess

from tools.candB.pilot_retry import OLD_RUNTIME, PARENT
from tools.candB.pilot_lean_deploy import PYTHON

RUNTIME = '/root/autodl-tmp/c4/candB_pilot_RETRY0300_EXPLORATORY-LEAN_20261031'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'cpu', 'status'))
    args = parser.parse_args()
    code = f'''
import hashlib,json,os,subprocess,time
from pathlib import Path
runtime=Path({RUNTIME!r}); python={PYTHON!r}; os.chdir(runtime)
env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='',C4_ALLOWED_ROOTS='/root/autodl-tmp/c4')
if {args.action!r}=='prepare':
    manifest=json.loads(Path('SOURCE_MANIFEST.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    Path('authority_gen').mkdir(exist_ok=True)
    subprocess.run([python,'-B','-m','tools.candB.pilot_retry_authority','--exploratory-owner-20261030',
        'build','authority_gen/PILOT_RETRY_AUTHORITY.json','--parent-authority',
        {OLD_RUNTIME!r}+'/authority_gen/PILOT_EXPLORATORY_LEAN_AUTHORITY.json'],env=env,check=True)
elif {args.action!r}=='cpu':
    assert not Path('pilot_RETRY_CPU_OWNERSHIP.json').exists(),'recover existing CPU job'
    command=[python,'-B','-m','tools.candB.pilot_retry_cpu']
    with Path('pilot_RETRY_CPU.log').open('x') as log:
        child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    Path('pilot_RETRY_CPU_OWNERSHIP.json').write_text(json.dumps(dict(pid=child.pid,command=command,epoch=time.time())))
else:
    for name in ('pilot_RETRY_CPU_OWNERSHIP.json','pilot_RETRY_CPU.log'):
        if Path(name).exists(): print(name,Path(name).read_text()[:400])
    result=Path('pilot_RETRY_CPU_RESULT.json')
    if result.exists():
        doc=json.loads(result.read_text())
        print('CPU',json.dumps(dict(returncode=doc['returncode'],seconds=doc['seconds'])))
        print(doc['stdout'][-2000:]); print(doc['stderr'][-2000:])
    parent=Path({PARENT!r})
    doc=json.loads((parent/'RUN_SUMMARY.json').read_text())
    print('PARENT',json.dumps(dict(status=doc.get('status'),error=doc.get('error'),
        label=doc.get('label'),seed=doc.get('seed'),rows=len(doc.get('updates',[])),
        zero_start=doc.get('zero_start'),resumed_from=doc.get('resumed_from'),
        has_score_before='independent_score_before' in doc,
        score_before=doc.get('independent_score_before',{{}}).get('J') if isinstance(doc.get('independent_score_before'),dict) else doc.get('independent_score_before'),
        native_curve_updates=[r.get('update') for r in doc.get('native_curve',[])],
        gate_panel=dict(passed=doc.get('gate_panel',{{}}).get('passed'),
            n_before=len(doc.get('gate_panel',{{}}).get('J_before',[]) or []),
            sha256=doc.get('gate_panel',{{}}).get('sha256','')[:16]),
        wm_members=len(doc.get('wm_members',[])),
        operational_recovery=doc.get('operational_recovery'))))
    import os as os2
    for name in ('policy_native_best.pt','frozen_scoring_baseline.json'):
        data=(parent/name).read_bytes()
        print('ARTIFACT',name,len(data),hashlib.sha256(data).hexdigest())
'''
    result = subprocess.run(['ssh', 'gpuA', '/root/miniconda3/bin/python', '-'],
        input=code, text=True, capture_output=True, timeout=120)
    import time
    receipt = Path(__file__).with_name(f'pilot_retry_{args.action}_{int(time.time())}.json')
    with receipt.open('x') as stream:
        json.dump(dict(action=args.action, runtime=RUNTIME, returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr), stream, indent=1)
    print(result.stdout+result.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
