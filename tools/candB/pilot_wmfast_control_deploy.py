"""Dispatch only the exclusive boundary waiter, never another fresh trainer."""
import json
from pathlib import Path
import subprocess

from tools.candB.pilot_wmfast_dispatch import ROOT
from tools.candB.pilot_wmfast_deploy import RUNTIME
from tools.candB.pilot_lean_deploy import PYTHON


def main():
    code = f'''
import json,os,subprocess,time
from pathlib import Path
runtime=Path({RUNTIME!r}); root=Path({ROOT!r}); python={PYTHON!r}; os.chdir(runtime)
assert not (root/'WMFAST_CONTROL_OWNERSHIP.json').exists(),'recover existing exclusive controller'
assert json.loads(Path('pilot_WMFAST_CPU_RESULT.json').read_text())['returncode']==0
env=dict(os.environ,PYTHONPATH='src:.',C4_ALLOWED_ROOTS='/root/autodl-tmp/c4')
with (root/'WMFAST_CONTROL.log').open('x') as log:
    child=subprocess.Popen([python,'-B','-u','-m','tools.candB.pilot_wmfast_control',
        '--root',str(root),'--python',python],env=env,stdin=subprocess.DEVNULL,stdout=log,
        stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(10):
    if (root/'WMFAST_CONTROL_OWNERSHIP.json').exists() or child.poll() is not None: break
    time.sleep(1)
print((root/'WMFAST_CONTROL_OWNERSHIP.json').read_text())
assert child.poll() is None,'controller exited; inspect receipts, do not repeat dispatch'
'''
    result = subprocess.run(['ssh','-o','ConnectTimeout=10','gpuA','/root/miniconda3/bin/python','-'],
        input=code,text=True,capture_output=True,timeout=30)
    with Path(__file__).with_name('pilot_wmfast_CONTROL_DISPATCH.json').open('x') as stream:
        json.dump(dict(returncode=result.returncode,runtime=RUNTIME,root=ROOT,
            stdout=result.stdout,stderr=result.stderr),stream,indent=1)
    print(result.stdout+result.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
