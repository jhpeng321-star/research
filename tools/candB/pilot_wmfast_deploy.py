"""Prepare a NEW WMFAST runtime/CPU proof. No training dispatch in this tool."""
import argparse
import json
from pathlib import Path
import subprocess

from tools.candB.pilot_lean_deploy import ROOT, RUNTIME as PARENT_RUNTIME, PYTHON

RUNTIME = '/root/autodl-tmp/c4/candB_pilot_WMFAST_EXPLORATORY-LEAN_20261031'


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
    subprocess.run([python,'-B','-m','tools.candB.pilot_wmfast_authority','--exploratory-owner-20261030',
        'build','authority_gen/PILOT_WMFAST_AUTHORITY.json','--parent-authority',
        {PARENT_RUNTIME!r}+'/authority_gen/PILOT_EXPLORATORY_LEAN_AUTHORITY.json'],env=env,check=True)
    subprocess.run([python,'-B','-m','tools.candB.pilot_lean_authority','--exploratory-owner-20261030',
        'build','authority_gen/PILOT_EXPLORATORY_LEAN_AUTHORITY.json','--parent-authority',
        {PARENT_RUNTIME!r}+'/authority_gen/PILOT_EXPLORATORY_LEAN_AUTHORITY.json'],env=env,check=True)
elif {args.action!r}=='cpu':
    assert not Path('pilot_WMFAST_CPU_OWNERSHIP.json').exists(),'recover existing CPU job'
    command=[python,'-B','-m','tools.candB.pilot_wmfast_cpu']
    with Path('pilot_WMFAST_CPU.log').open('x') as log:
        child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    Path('pilot_WMFAST_CPU_OWNERSHIP.json').write_text(json.dumps(dict(pid=child.pid,command=command,epoch=time.time())))
else:
    for name in ('pilot_WMFAST_CPU_OWNERSHIP.json','pilot_WMFAST_CPU.log','pilot_WMFAST_CPU_RESULT.json'):
        if Path(name).exists(): print(name,Path(name).read_text())
    parent=Path({ROOT!r})/'pilot_EXPLORATORY-LEAN'
    for name in ('RUN_SUMMARY.json','frozen_scoring_baseline.json','pilot_preserved_startup.json'):
        if (parent/name).exists():
            doc=json.loads((parent/name).read_text())
            print(name,json.dumps(dict(status=doc.get('status'),error=doc.get('error'),
                saved_score=doc.get('independent_score_before'),baseline=doc.get('frozen_scoring_baseline'),
                rows=len(doc.get('updates',[])))))
'''
    result = subprocess.run(['ssh', 'gpuA', '/root/miniconda3/bin/python', '-'],
        input=code, text=True, capture_output=True, timeout=60)
    receipt = Path(__file__).with_name(f'pilot_wmfast_{args.action}_receipt.json')
    if receipt.exists():
        import time
        receipt = receipt.with_name(receipt.stem+f'_{int(time.time())}.json')
    with receipt.open('x') as stream:
        json.dump(dict(action=args.action, runtime=RUNTIME, returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr), stream, indent=1)
    print(result.stdout+result.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
