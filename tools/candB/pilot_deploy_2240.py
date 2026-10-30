"""Stage only committed, new operational overlay files; retain immutable runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


FILES = ('pilot_handoff.py', 'pilot_recipe.py', 'pilot_continue.py',
         'pilot_continue_launch.sh', 'pilot_continue_supervise.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--receipt', default='pilot_deploy_2240_receipt.json')
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    sources = {name: subprocess.check_output(['git', 'show', commit+':tools/candB/'+name]).decode()
               for name in FILES}
    receipt = dict(label='EXPLORATORY', source_commit=commit,
        sha256={name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()})
    code = f'''import hashlib,json,os,subprocess,time
from pathlib import Path
runtime=Path({args.runtime!r})
root=Path({args.root!r})
assert runtime.is_dir() and root.is_dir()
sources={sources!r}
receipt={receipt!r}
for name in sources:
    path=runtime/'tools/candB'/name
    assert not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256'][name], name
assert not (root/'continue_OWNERSHIP.json').exists()
for name,value in sources.items():
    if not (runtime/'tools/candB'/name).exists():
        with (runtime/'tools/candB'/name).open('x',newline='\\n') as stream: stream.write(value)
    assert hashlib.sha256((runtime/'tools/candB'/name).read_bytes()).hexdigest()==receipt['sha256'][name]
subprocess.run(['bash','-n','tools/candB/pilot_continue_launch.sh'],cwd=runtime,check=True)
env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='',C4_ALLOWED_ROOTS='/root/autodl-tmp/c4')
subprocess.run([{args.python!r},'-B','-m','tools.candB.pilot_authority','--exploratory-owner-20261030',
    'verify','authority_gen/PILOT_EXPLORATORY_AUTHORITY.json'],cwd=runtime,env=env,check=True)
with (root/'continue_OVERLAY.json').open('x') as stream: json.dump(receipt,stream,indent=1)
with (root/'continue_supervisor.log').open('x') as log:
    child=subprocess.Popen([{args.python!r},'-B','-u','-m','tools.candB.pilot_continue_supervise',
        '--exploratory-owner-20261030','--root',str(root),'--python',{args.python!r}],cwd=runtime,
        env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(30):
    if (root/'continue_OWNERSHIP.json').is_file(): break
    time.sleep(1)
assert (root/'continue_OWNERSHIP.json').is_file()
print((root/'continue_OWNERSHIP.json').read_text())
'''
    result = subprocess.run(['ssh', 'gpuA', '/root/miniconda3/bin/python', '-'], input=code,
                            text=True, capture_output=True, timeout=60)
    receipt.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    with (Path(__file__).parent / args.receipt).open('x') as stream:
        json.dump(receipt, stream, indent=1)
    print(result.stdout + result.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
