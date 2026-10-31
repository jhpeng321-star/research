"""Explicit short staging/audit/CPU operations; no automatic training dispatch."""
import argparse
import json
from pathlib import Path
from tools.coopbandit import ops as ssh
from .protocol import checked, sha, worlds, digest
from .storage import write

LOCAL = Path(__file__).resolve().parents[2]
ROOT = '/home/ubuntu/coopfree20261101_build1'
REG = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
ssh.PYTHON['gpuA'] = '/root/miniconda3/bin/python'


def audit(host):
    # Reuse exact range+safe-name audit without opening any assignment/truth files.
    oldworlds, olddev = ssh.worlds, ssh.dev_worlds
    ssh.worlds, ssh.dev_worlds = worlds, lambda:worlds(True)
    try:
        return ssh.audit(host)
    finally:
        ssh.worlds, ssh.dev_worlds = oldworlds, olddev


def sources():
    files = {}
    for p in sorted((LOCAL/'tools/coopfree').glob('*.py')):
        files['tools/coopfree/'+p.name] = checked(p).read_text(encoding='utf-8')
    for name in ('__init__.py','protocol.py'):
        p = LOCAL/'tools/coopbandit'/name
        files['tools/coopbandit/'+name] = checked(p).read_text(encoding='utf-8')
    for name in ('__init__.py','backend.py','design.py','intervention.py','selection.py','storage.py'):
        p = Path('E:/agent-work/c4-leverprobe/tools/leverprobe')/name
        files['tools/leverprobe/'+name] = checked(p).read_text(encoding='utf-8')
    return files


def stage(host,root):
    files = sources()
    design = checked(LOCAL/'docs/paper_results/COOPFREE_DESIGN.md').read_text(encoding='utf-8')
    return ssh.remote(host,f'''import pathlib,json,hashlib
root=pathlib.Path({root!r});assert root.parent.is_dir()
root.mkdir(exist_ok=True)
files=json.loads({json.dumps(files)!r})
design=json.loads({json.dumps(design)!r})
for name,text in files.items():
 p=root/'pytools'/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_text()!=text: raise ValueError('immutable stage '+name)
 if not p.exists(): p.write_text(text)
p=root/'COOPFREE_DESIGN.md'
if p.exists() and p.read_text()!=design: raise ValueError('immutable design')
if not p.exists(): p.write_text(design)
print(json.dumps(dict(root=str(root),files=len(files),sources={{n:hashlib.sha256((root/'pytools'/n).read_bytes()).hexdigest() for n in files}},design_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
''')


def cpu(host,root):
    if host != 'cpuQ':
        raise ValueError('CPU build uses existing cpuQ frozen environment')
    return ssh.remote(host,f'''import sys,os,unittest,json,hashlib,time
from unittest.mock import patch
os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,{root!r}+'/pytools')
from tools.leverprobe.backend import setup
checkpoint='/home/ubuntu/mappospeed/production2/policy_latest.pt'
setup('/home/ubuntu/e3final',checkpoint)
os.environ['COOPFREE_CPU_CHECKPOINT']=checkpoint
begin=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('SUMO forbidden during CPU build')):
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tools.coopfree.test_cpu'))
receipt=dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,wall_s=time.monotonic()-begin,root={root!r})
print(json.dumps(receipt))
assert result.wasSuccessful() and not result.skipped
''')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=('inspect','audit','stage','cpu'))
    p.add_argument('--host',default='cpuQ',choices=('cpuQ','gpuR','gpuA'))
    p.add_argument('--root',default=ROOT)
    p.add_argument('--receipt',type=Path)
    a = p.parse_args()
    checked(a.root)
    if a.mode == 'inspect':
        result = ssh.inspect(a.host)
    elif a.mode == 'audit':
        result = json.dumps(audit(a.host))
    elif a.mode == 'stage':
        result = stage(a.host,a.root)
    else:
        result = cpu(a.host,a.root)
    print(result)
    if a.receipt:
        write(a.receipt,dict(operation=a.mode,host=a.host,root=a.root,result=result,
                             design_sha256=sha(LOCAL/'docs/paper_results/COOPFREE_DESIGN.md')))


if __name__ == '__main__':
    main()
