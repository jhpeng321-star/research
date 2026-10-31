"""CPU-only preparation/export. These commands never dispatch SUMO or inference."""
import argparse
import base64
import io
import json
from pathlib import Path
import tarfile

from tools.c1switch.design import FORBIDDEN, sha
from tools.c1switch.ops import ssh
from tools.c1switch.storage import write

from .audit import registration_audit
from .design import REGISTRATION, canary_world, guard, manifest, worlds

EVIDENCE = Path('tools/c1switch2/evidence')


def audit(host=None):
    report = registration_audit(REGISTRATION)
    for world in worlds()+[canary_world()]:
        guard(world)
    report['local_seed_guards'] = len(worlds())+1
    if host:
        parent = '/home/ubuntu' if host == 'cpuQ' else '/root'
        # Source names only; do not open, parse or stat a prohibited entry.
        code = f'''import json,os,re
bad=re.compile({FORBIDDEN.pattern!r},re.I)
wanted={set(w['seed'] for w in worlds()+[canary_world()])!r}
hits=[];count=0;pruned=0
stack=[{parent!r}]
while stack:
 here=stack.pop()
 try:
  with os.scandir(here) as entries:
   for entry in entries:
    if bad.search(entry.path) or entry.name.startswith('.') or entry.name in ('miniconda3','miniconda','venv','__pycache__'):
     pruned+=1;continue
    if entry.is_symlink():continue
    count+=1
    if any(int(s) in wanted for s in re.findall(r'(?<![0-9])(969[0-9]{{3}})(?![0-9])',entry.name)):
     hits.append(entry.path)
    if entry.is_dir(follow_symlinks=False):stack.append(entry.path)
 except PermissionError:raise RuntimeError('incomplete filename audit')
print(json.dumps(dict(host={host!r},safe_names_checked=count,pruned=pruned,hits=hits,new_sim_s=0,new_gpu_s=0)))
'''
        report['remote'] = ssh(host, code, timeout=120)
    report['pass'] = not report['hits'] and not report.get('remote', {}).get('hits')
    output = EVIDENCE/('EXPOSURE_'+(host or 'local')+'_v3.json')
    if output.exists():
        raise ValueError('preserve prior exposure receipt; use a new prospective version')
    write(output, report)
    print(json.dumps(report, indent=2))


def cpu(host):
    files = sorted(Path('tools/c1switch').glob('*.py'))+sorted(Path('tools/c1switch2').glob('*.py'))
    files += [Path('tests/test_c1switch2.py')]
    sources = {p.as_posix(): sha(p) for p in files}
    identity = __import__('hashlib').sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:12]
    parent = '/home/ubuntu' if host == 'cpuQ' else '/root'
    root = parent+'/c1switch2_cpu_'+identity
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for path in files:
            tar.add(path, arcname=path.as_posix())
    raw = base64.b64encode(buf.getvalue()).decode()
    runtime = '/home/ubuntu/e3final/runtime' if host == 'cpuQ' else '/root/autodl-tmp/c4/speedhead20261030/e3final/runtime'
    result = ssh(host, f'''import base64,contextlib,hashlib,io,json,os,sys,tarfile
from pathlib import Path
root=Path({root!r});assert root.parent.is_dir();assert not root.exists(),'no CPU stage overwrite'
root.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({raw!r})),mode='r:gz') as tar:tar.extractall(root,filter='data')
sources={sources!r}
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==value for name,value in sources.items())
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[str(root),{runtime!r}+'/src',{runtime!r}]
import subprocess
def forbidden(*a,**kw):raise AssertionError('CPU preparation attempted SUMO or subprocess')
subprocess.Popen=forbidden
try:
 import traci
 traci.start=forbidden;traci.simulationStep=forbidden
except ImportError:pass
import pytest
log=io.StringIO()
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
 code=pytest.main(['-q','-p','no:cacheprovider','-c','/dev/null',str(root/'tests/test_c1switch2.py')])
out=dict(host={host!r},root=str(root),returncode=int(code),output=log.getvalue(),sources=sources,new_sim_s=0,new_gpu_s=0)
(root/'CPU.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
''', timeout=120)
    write(EVIDENCE/('CPU_'+host+'.json'), result)
    print(json.dumps({k: v for k, v in result.items() if k != 'sources'}, indent=2))


def export():
    proposal = dict(base=manifest(), conditional_wm_phys=manifest(True))
    output = Path('docs/paper_results/c1switch2')
    if (output/'BUDGET_PROPOSAL.json').exists():
        raise ValueError('preserve prior design export')
    write(output/'BUDGET_PROPOSAL.json', proposal)
    write(output/'PANEL.json', dict(protocol=proposal['base']['protocol'], worlds=worlds(),
        canary=canary_world(), measured=proposal['base']['measured'],
        decisions_s=list(range(300,900,20)), J600_window_s=[300,900], J900_window_s=[300,1200],
        evaluation_owner='S-C1SWITCH2'))
    print(json.dumps(dict(design_sha256=proposal['base']['design_sha256'],
        base_sim_s=proposal['base']['total_sim_s'],
        conditional_sim_s=proposal['conditional_wm_phys']['total_sim_s'],
        new_sim_s=0,new_gpu_s=0), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('audit', 'cpu', 'export'))
    parser.add_argument('--host', choices=('cpuQ', 'gpuR', 'gpuA'))
    args = parser.parse_args()
    if args.command == 'audit':
        audit(args.host)
    elif args.command == 'cpu':
        if not args.host:
            raise ValueError('CPU host required')
        cpu(args.host)
    else:
        export()
