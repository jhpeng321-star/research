"""Short, explicit-file S-C4PREP transport; never stages into peer runtimes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/longint'))
from operate import remote


def inspect():
    remote('''import pathlib,json,shutil,subprocess,re
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026')
bad=re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype',re.I)
def safe(p):
 assert not bad.search(str(p)) and not bad.search(str(p.resolve()));return p
for rel in ('AUTONOMY_STATUS.json','arm/loop/longint/c3wmft/check_repair1/RESULTS.json',
 'arm/loop/checks/c03.a2/ACCEPT.json'):
 p=safe(r/rel)
 if p.is_file(): print(rel,p.read_text())
state=json.loads((r/'arm/loop/LOOP14_STATE.json').read_text())
print('state',json.dumps({k:state.get(k) for k in ('accepted_wm','policy','stopped','looplong_boundary')}))
for rel in ('arm/loop/data','arm/loop/checks/c03.a2'):
 p=safe(r/rel)
 if p.is_dir(): print(rel,[q.name for q in p.iterdir() if not bad.search(q.name)])
print('gpu',subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used','--format=csv,noheader'],text=True))
print('free_bytes',shutil.disk_usage(r).free)
''', timeout=25)


def query(source, host='gpuA'):
    """Run an owned, reviewed source file via stdin (avoids Windows command limits)."""
    data = Path(source).read_bytes()
    python = '/root/miniconda3/bin/python' if host.startswith('gpu') else '/home/ubuntu/miniconda3/bin/python'
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
                        python + ' -B -'], input=data, capture_output=True, timeout=35)
    print(p.stdout.decode(), end='')
    print(p.stderr.decode(), end='', file=sys.stderr)
    if p.returncode:
        raise SystemExit(p.returncode)


def runtime(entry, args):
    assert entry in ('calibrate', 'launch')
    # Preparation transport intentionally cannot dispatch a full policy phase.
    assert args and args[0] in ('preflight', 'templates', 'validate', 'prepare')
    remote(f'''import pathlib,json,os,subprocess
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026')
code=pathlib.Path(json.loads((r/'arm/loop/c4prep/CURRENT_RUNTIME.json').read_text())['runtime'])
env={{**os.environ,'PYTHONPATH':str(code/'src')+':'+str(code),
 'C4_ALLOWED_ROOTS':'/root/autodl-tmp/c4','OMP_NUM_THREADS':'1',
 'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}}
p=subprocess.run(['/root/miniconda3/bin/python','-B',str(code/'tools/c4prep/{entry}.py'),*{args!r}],
 cwd=code,env=env,timeout=25)
raise SystemExit(p.returncode)
''', timeout=35)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['inspect', 'query', 'runtime'])
    ap.add_argument('--source')
    ap.add_argument('--host', default='gpuA')
    ap.add_argument('--entry', choices=('calibrate', 'launch'))
    ap.add_argument('--args', nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()
    if a.action == 'inspect':
        inspect()
    elif a.action == 'query':
        query(a.source, a.host)
    else:
        runtime(a.entry, a.args)
