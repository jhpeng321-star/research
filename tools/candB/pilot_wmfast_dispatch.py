"""Single exclusive dispatch into a fresh chain; source and CPU checks already durable."""
import json
from pathlib import Path
import subprocess

from tools.candB.pilot_wmfast_deploy import RUNTIME
from tools.candB.pilot_lean_deploy import PYTHON

ROOT = '/root/autodl-tmp/c4/candB_pilot_run_WMFAST_EXPLORATORY-LEAN_20261031'


def main():
    code = f'''
import json,os,subprocess,time
from pathlib import Path
runtime=Path({RUNTIME!r}); root=Path({ROOT!r}); python={PYTHON!r}
assert runtime.is_dir() and not root.exists(),'recover existing dispatch; never duplicate'
os.chdir(runtime)
assert json.loads(Path('pilot_WMFAST_CPU_RESULT.json').read_text())['returncode']==0
root.mkdir()
env=dict(os.environ,PYTHONPATH='src:.',C4_ALLOWED_ROOTS='/root/autodl-tmp/c4')
with (root/'supervisor.log').open('x') as log:
    child=subprocess.Popen([python,'-B','-u','-m','tools.candB.pilot_wmfast_supervise',
        '--root',str(root),'--python',python],env=env,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(12):
    if (root/'repair_TRAINER_OWNERSHIP.json').exists() or child.poll() is not None: break
    time.sleep(1)
for name in ('OWNERSHIP.json','repair_TRAINER_OWNERSHIP.json','supervisor.log','EXIT.json'):
    if (root/name).exists(): print(name,(root/name).read_text())
assert child.poll() is None,'recover failed dispatch receipts; no automatic retry'
'''
    result = subprocess.run(['ssh','gpuA','/root/miniconda3/bin/python','-'], input=code,
        text=True, capture_output=True, timeout=35)
    receipt = Path(__file__).with_name('pilot_wmfast_DISPATCH.json')
    with receipt.open('x') as stream:
        json.dump(dict(root=ROOT, runtime=RUNTIME, returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr), stream, indent=1)
    print(result.stdout+result.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
