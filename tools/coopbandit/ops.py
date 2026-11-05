"""Explicit, short SSH operations. All remote reads prune restricted paths."""
import argparse
import json
from pathlib import Path
import subprocess
from .protocol import checked, digest, sha, worlds, dev_worlds, FORBIDDEN

LOCAL = Path(__file__).resolve().parents[2]
REG = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
PROBE = Path('E:/agent-work/c4-leverprobe/tools/leverprobe')
ROOT = '/home/ubuntu/coopbandit20261031_v1'
PYTHON = dict(cpuQ='/home/ubuntu/miniconda3/bin/python', gpuR='/root/miniconda3/bin/python')


def command(host, remote_command, *, timeout=None):
    if str(host).lower() == 'cpuv':
        from tools.coopmwm.cpuv_ssh import command as multiplexed
        return multiplexed(host, remote_command, timeout=timeout)
    return ['ssh', '-o', 'BatchMode=yes', host, remote_command]


def remote(host, source, timeout=180):
    code = 'import sys;exec(compile(sys.stdin.read(),\'<coopbandit-remote>\',\'exec\'))'
    r = subprocess.run(command(host, PYTHON[host]+' -c "'+code+'"', timeout=timeout),
                       input=source, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(r.stdout+r.stderr)
    return r.stdout


def inspect(host):
    return remote(host, '''import os,json,pathlib,shutil,time
p=pathlib.Path('/sys/fs/cgroup/cpu.max')
print(json.dumps(dict(host=os.uname().nodename,load=os.getloadavg(),cpus=os.cpu_count(),
 quota=p.read_text() if p.exists() else None,disk_free=shutil.disk_usage('.').free,
 meminfo=pathlib.Path('/proc/meminfo').read_text().splitlines()[:3],unix=time.time())))
''')


def audit(host):
    import re
    seeds = [w['seed'] for w in worlds()+dev_worlds()]
    text = checked(REG).read_text(encoding='utf-8')
    hits = []
    for line in text.splitlines():
        ranges = re.findall(r'(?<![0-9a-fA-F.])(9\d{5})\s*[-–锟?]+\s*(9?\d{4,5})(?![0-9a-fA-F])', line)
        for a, b in ranges:
            if len(b) < 6:
                b = a[:6-len(b)]+b
            if any(int(a) <= s <= int(b) for s in seeds):
                hits.append(line)
        if re.search(r'(?<![0-9a-fA-F.])(?:s|w)?(?:'+'|'.join(map(str,seeds))+r')(?![0-9a-fA-F])',line):
            hits.append(line)
    if hits:
        raise ValueError('registration collision: '+str(hits))
    home = '/home/ubuntu' if host == 'cpuQ' else '/root'
    result = json.loads(remote(host, f'''import os,pathlib,re,json
forbidden=re.compile({FORBIDDEN.pattern!r},re.I)
ident=re.compile(r'(?:^|[^0-9a-fA-F.])(?:s|w)?(?:'+ '|'.join(map(str,{seeds!r}))+r')(?![0-9a-fA-F])')
hits=[];count=0
for parent,dirs,files in os.walk({home!r}):
 dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','miniconda3','.cache','.npm','.rustup','.local') and not forbidden.search(str(pathlib.Path(parent)/d)) and not (pathlib.Path(parent)/d).is_symlink()]
 for name in dirs+files:
  if forbidden.search(str(pathlib.Path(parent)/name)): continue
  count+=1
  if ident.search(name): hits.append(str(pathlib.Path(parent)/name))
print(json.dumps(dict(host={host!r},safe_names=count,hits=hits,pass_=not hits)))
'''))
    result['registration_sha256'] = sha(REG)
    result['worlds_sha256'] = digest(worlds()+dev_worlds())
    return result


def stage(root):
    checked(root)
    files = {}
    for p in sorted((LOCAL/'tools/coopbandit').glob('*.py')):
        files['tools/coopbandit/'+p.name] = p.read_text(encoding='utf-8')
    for name in ('__init__.py','backend.py','design.py','intervention.py','selection.py','storage.py'):
        files['tools/leverprobe/'+name] = (PROBE/name).read_text(encoding='utf-8')
    files['tools/coopbandit/probe_quantiles.json'] = (PROBE/'evidence/RECORDED_QUANTILES.json').read_text(encoding='utf-8')
    design = (LOCAL/'docs/paper_results/COOPBANDIT_DESIGN.md').read_text(encoding='utf-8')
    return remote('cpuQ', f'''import pathlib,json
root=pathlib.Path({root!r});assert root.parent.is_dir()
root.mkdir(exist_ok=True)
files=json.loads({json.dumps(files)!r})
for name,text in files.items():
 p=root/'pytools'/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_text()!=text: raise ValueError('immutable stage '+name)
 if not p.exists(): p.write_text(text)
p=root/'COOPBANDIT_DESIGN.md'
if p.exists() and p.read_text()!={design!r}: raise ValueError('immutable design')
if not p.exists(): p.write_text({design!r})
print(json.dumps(dict(root=str(root),files=len(files))))
''')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=('inspect','audit','stage','cpu'))
    p.add_argument('--host',default='cpuQ',choices=tuple(PYTHON))
    p.add_argument('--root',default=ROOT)
    a = p.parse_args()
    if a.mode == 'inspect':
        print(inspect(a.host))
    elif a.mode == 'audit':
        print(json.dumps(audit(a.host),indent=2))
    elif a.mode == 'stage':
        print(stage(a.root))
    else:
        print(stage(a.root))
        print(remote('cpuQ', f'''import sys,os,unittest,json
from unittest.mock import patch
os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,{a.root!r}+'/pytools')
from tools.leverprobe.backend import setup
setup('/home/ubuntu/e3final','/home/ubuntu/mappospeed/production2/policy_latest.pt')
with patch('subprocess.Popen',side_effect=AssertionError('SUMO disabled')):
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tools.coopbandit.test_cpu'))
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0)))
assert result.wasSuccessful() and not result.skipped
'''))


if __name__ == '__main__':
    main()
