"""Prepare/dispatch NEW immutable lean runtime; never alter original runtimes."""
import argparse
import json
from pathlib import Path
import subprocess

RUNTIME='/root/autodl-tmp/c4/candB_pilot_EXPLORATORY-LEAN_20261030'
ROOT='/root/autodl-tmp/c4/candB_pilot_run_EXPLORATORY-LEAN_20261030'
PYTHON='/root/autodl-tmp/c4/candB_rankcheck_0264b967_20261030/venv/bin/python'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','dispatch'))
    args=parser.parse_args()
    code=f'''
import hashlib,json,os,subprocess,time
from pathlib import Path
runtime=Path({RUNTIME!r}); root=Path({ROOT!r}); python={PYTHON!r}
assert runtime.is_dir()
env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='',C4_ALLOWED_ROOTS='/root/autodl-tmp/c4')
os.chdir(runtime)
if {args.action!r}=='prepare':
    manifest=json.loads(Path('SOURCE_MANIFEST.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    Path('authority_gen').mkdir(exist_ok=True)
    subprocess.run(['bash','-n','tools/candB/pilot_lean_launch.sh'],check=True)
    subprocess.run([python,'-B','-m','tools.candB.pilot_lean_authority','--exploratory-owner-20261030',
        'build','authority_gen/PILOT_EXPLORATORY_LEAN_AUTHORITY.json','--parent-authority',
        '/root/autodl-tmp/c4/candB_pilot_final_EXPLORATORY_20261030/authority_gen/PILOT_EXPLORATORY_AUTHORITY.json'],env=env,check=True)
else:
    assert not root.exists(),'existing root; recover, never duplicate dispatch'
    root.mkdir()
    with (root/'supervisor.log').open('x') as log:
        child=subprocess.Popen([python,'-B','-u','-m','tools.candB.pilot_lean_supervise','--root',str(root),
            '--python',python],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(15):
        if (root/'TRAINER_OWNERSHIP.json').exists() or child.poll() is not None: break
        time.sleep(1)
    for name in ('OWNERSHIP.json','TRAINER_OWNERSHIP.json','EXIT.json'):
        if (root/name).exists(): print(name,(root/name).read_text())
    assert child.poll() is None, 'supervisor exited; recover receipts'
'''
    result=subprocess.run(['ssh','gpuA','/root/miniconda3/bin/python','-'],input=code,
        text=True,capture_output=True,timeout=90)
    receipt=Path(__file__).with_name('pilot_lean_'+args.action+'_receipt.json')
    with receipt.open('x') as stream:
        json.dump(dict(action=args.action,runtime=RUNTIME,root=ROOT,returncode=result.returncode,
            stdout=result.stdout,stderr=result.stderr),stream,indent=1)
    print(result.stdout+result.stderr)
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
