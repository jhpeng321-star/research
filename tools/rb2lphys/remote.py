"""Own immutable source stages and bounded jobs; never signal another process."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from tools.rbint.remote import ssh

ROOT = Path(__file__).resolve().parents[2]
PARENT = '/root/rbeval20261031'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('audit', 'stage', 'run', 'status', 'fetch'))
    p.add_argument('--host', default='gpuR', choices=('gpuR', 'gpuA', 'cpuQ'))
    p.add_argument('--stage')
    p.add_argument('--label')
    p.add_argument('--command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    parent = '/home/ubuntu/rb2lphys20261101' if args.host == 'cpuQ' else PARENT
    python = '/home/ubuntu/miniconda3/bin/python' if args.host == 'cpuQ' else '/root/miniconda3/bin/python'
    if args.action == 'audit':
        print(ssh(args.host, '''import os,json,subprocess,importlib.util
from pathlib import Path
print('LOAD',os.getloadavg(),'CPUS',os.cpu_count())
import shutil
if shutil.which('nvidia-smi'):print(subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout)
print('MODULES',[(m,bool(importlib.util.find_spec(m))) for m in ('torch','scipy','pytest')])
for p in ('/root/wmdata20261031/views','/root/wmdata20261031','/home/ubuntu/wmdata20261031/views','/root/rbeval20261031/phys_code_22cc32f65a80'):
 try:print('PATH',p,Path(p).is_dir())
 except PermissionError:print('PATH',p,'inaccessible')
'''))
        return
    if args.action == 'stage':
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        data = subprocess.check_output(['git', 'archive', '--format=tar', sha,
            'src/wms', 'tools/rbint', 'tools/rbeval', 'tools/rbfit', 'tools/rb2lphys',
            'configs/wm_phys.toml', 'configs/execparity_cav_idm_v1.json',
            'configs/native_cav_head_v5.json', 'artifacts/rbfit/shrinkage_v1',
            'docs/paper_results/rbunc/observed_arrivals.json', 'tests/test_rb2lphys.py'], cwd=ROOT)
        digest = hashlib.sha256(data).hexdigest()
        dest = f'{parent}/lphys_code_{sha[:12]}'
        code = f'''import sys,io,tarfile,hashlib,json
from pathlib import Path
data=sys.stdin.buffer.read();assert hashlib.sha256(data).hexdigest()=={digest!r}
parent=Path({parent!r});assert parent.parent.is_dir();parent.mkdir(exist_ok=True)
dest=Path({dest!r})
if dest.exists():
 receipt=json.loads((dest/'LP_STAGE.json').read_text());assert receipt['sha256']=={digest!r};print(json.dumps(receipt));raise SystemExit(0)
dest.mkdir()
with tarfile.open(fileobj=io.BytesIO(data)) as t:
 for m in t.getmembers():
  assert not m.issym() and not m.islnk() and '..' not in Path(m.name).parts and not Path(m.name).is_absolute()
 t.extractall(dest,filter='data')
receipt=dict(commit={sha!r},sha256={digest!r},stage=str(dest))
with (dest/'LP_STAGE.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))'''
        result = subprocess.run(['ssh', args.host, python, '-B', '-c', shlex.quote(code)],
                                input=data, capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr.decode()+result.stdout.decode())
        print(result.stdout.decode())
        return
    if not args.label or not args.label.replace('_', '').isalnum():
        p.error('safe label required')
    work = f'{parent}/lphys_{args.label}'
    if args.action == 'run':
        if not args.stage or not args.command:
            p.error('stage and command required')
        code = f'''import os,json,subprocess,time
from pathlib import Path
stage=Path({args.stage!r});assert (stage/'LP_STAGE.json').is_file()
parent=Path({parent!r});assert parent.is_dir()
work=Path({work!r});assert not work.exists();work.mkdir()
assert os.getloadavg()[0] < os.cpu_count()/2
cmd=[{python!r},'-B',*{args.command!r}]
env=dict(os.environ,PYTHONPATH=str(stage/'src')+':'+str(stage),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (work/'stdout.log').open('x') as log:
 child=subprocess.Popen(cmd,cwd=stage,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
receipt=dict(owner='S-RB2-LPHYS',pid=child.pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=Path('/proc/%s/stat'%child.pid).read_text().split()[21],stage=str(stage),work=str(work),command=cmd,started_unix=time.time())
with (work/'PROCESS.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))'''
        print(ssh(args.host, code))
    elif args.action == 'status':
        print(ssh(args.host, f'''import json,os
from pathlib import Path
w=Path({work!r});r=json.loads((w/'PROCESS.json').read_text());print(json.dumps(r))
p=Path('/proc/%s/stat'%r['pid']);print('PROCESS',p.read_text().split()[2] if p.exists() else 'EXITED')
print((w/'stdout.log').read_text())
for name in ('SUMMARY.json','DIAGNOSIS.json','SEARCH.json'):
 if (w/name).exists():print(name,(w/name).read_text())'''))
    else:
        dest = ROOT/'docs/rb2lphys'/args.label
        assert dest.parent.is_dir() and not dest.exists()
        # Explicitly own work directory only. No dataset traversal or peer artifacts.
        code = f'''import io,sys,tarfile
from pathlib import Path
w=Path({work!r});assert (w/'PROCESS.json').is_file()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
 for p in sorted(w.rglob('*')):
  if p.is_file() and not p.is_symlink():t.add(p,arcname=str(p.relative_to(w)))'''
        result = subprocess.run(['ssh', args.host, python, '-B', '-c', shlex.quote(code)], capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr.decode())
        import io
        import tarfile
        dest.mkdir()
        with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
            for m in tar.getmembers():
                assert not m.issym() and not m.islnk() and '..' not in Path(m.name).parts and not Path(m.name).is_absolute()
            tar.extractall(dest, filter='data')
        print(dest)


if __name__ == '__main__':
    main()
