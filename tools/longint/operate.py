"""S-LONGINT explicit-file transport; isolated gpuA runtime, no SUMO dispatch."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
RELEASE = '/root/autodl-tmp/c4/restart20261026'
RUNTIME = RELEASE + '/runtime_longint'
BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'hetdata_truth', 'vtype')


def remote(code, data=None, timeout=30):
    encoded = base64.b64encode(code.encode()).decode()
    command = 'import base64;exec(base64.b64decode(' + repr(encoded) + '))'
    for attempt in range(3):
        proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
            'gpuA', shlex.join(['/root/miniconda3/bin/python', '-B', '-c', command])],
            input=data, capture_output=True, timeout=timeout)
        if proc.returncode != 255 or b'connect to host' not in proc.stderr:
            break
    print(proc.stdout.decode('utf-8'), end='')
    print(proc.stderr.decode('utf-8'), end='', file=__import__('sys').stderr)
    if proc.returncode:
        raise SystemExit(proc.returncode)
    return proc.stdout


def stage():
    files = subprocess.check_output(['git', 'ls-files', '--cached', '--others',
        '--exclude-standard', 'src', 'tests/unit', 'configs', 'scripts/c4', 'tools', 'traffic_shepherd',
        'pyproject.toml', 'LOOPLONG.md', 'LONGMEM.md', 'PROBE.md', 'GROUPGRAD.md'],
        cwd=ROOT, text=True).splitlines()
    files = sorted({f for f in files if not any(t in f.lower() for t in BAD)
                    and (f.endswith(('.py', '.toml'))
                         or ('/' not in f and f.endswith('.md'))
                         or (f.startswith(('configs/', 'tests/unit/'))
                             and f.endswith(('.json', '.yaml', '.yml'))))})
    # Required packaged physics constants; do not copy datasets or prior receipts.
    files += subprocess.check_output(['git', 'ls-files', 'src/wms/model/*.json'],
                                     cwd=ROOT, text=True).splitlines()
    files = sorted(set(files))
    # Unmodified tracked files come from the index (canonical LF bytes, so hash
    # bindings such as WM configs match the deployed loop); only modified or
    # untracked files are read from disk, with Windows CRLF normalised to LF.
    dirty = set(subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT,
                                        text=True).splitlines())
    dirty |= set(subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'],
                                         cwd=ROOT, text=True).splitlines())
    missing = [f for f in files if not (ROOT / f).is_file() or f not in dirty]
    blobs = {}
    if missing:
        raw = subprocess.check_output(['git', 'cat-file', '--batch'], cwd=ROOT,
            input=''.join(':' + f + '\n' for f in missing).encode())
        stream = io.BytesIO(raw)
        for f in missing:
            header = stream.readline().split()
            if header[1] != b'blob':
                raise ValueError(f'Missing tracked source: {f}')
            blobs[f] = stream.read(int(header[2]))
            assert stream.read(1) == b'\n'
    payload = io.BytesIO()
    manifest = {}
    with tarfile.open(fileobj=payload, mode='w:gz') as archive:
        for f in files:
            assert not any(t in f.lower() for t in BAD)
            data = blobs[f] if f in blobs else (ROOT / f).read_bytes().replace(b'\r\n', b'\n')
            manifest[f] = hashlib.sha256(data).hexdigest()
            entry = tarfile.TarInfo(f)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
        for name in ('policy', 'bridge', 'dreamer'):
            f = 'baseline/src/wms/control/' + name + '.py'
            data = subprocess.check_output(['git', 'show', 'a2fc2c18^:src/wms/control/' + name + '.py'], cwd=ROOT)
            manifest[f] = hashlib.sha256(data).hexdigest()
            entry = tarfile.TarInfo(f)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    remote(f'''import pathlib,shutil,sys,tarfile,json,hashlib,io
parent=pathlib.Path({RELEASE!r}); assert parent.is_dir()
assert shutil.disk_usage(parent).free > 3*1024**3 + {len(payload.getvalue())}*10
root=pathlib.Path({RUNTIME!r}); root.mkdir(exist_ok=True)
manifest={{}}
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read({len(payload.getvalue())})),mode='r:gz') as archive:
 for e in archive:
  p=pathlib.PurePosixPath(e.name)
  assert e.isfile() and not p.is_absolute() and '..' not in p.parts
  assert not any(t in e.name.lower() for t in {BAD!r})
  data=archive.extractfile(e).read(); manifest[e.name]=hashlib.sha256(data).hexdigest()
  target=root/e.name; target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists() and target.read_bytes()!=data:
   old=target.read_bytes(); backup=target.with_name(target.name+'.'+hashlib.sha256(old).hexdigest()+'.before')
   if not backup.exists(): backup.write_bytes(old)
  target.write_bytes(data)
assert hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()=={hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()!r}
(root/'SOURCE.json').write_text(json.dumps(manifest,sort_keys=True,indent=2))
print(json.dumps({{'files':len(manifest),'free_bytes':shutil.disk_usage(root).free,'runtime':str(root)}}))
''', payload.getvalue(), timeout=90)


def tests(name, full=False):
    assert name.replace('-', '').isalnum()
    targets = ['tests/unit'] if full else ['tests/unit/test_longmem.py',
        'tests/unit/test_looplong.py', 'tests/unit/test_policy14_integration.py',
        'tests/unit/test_probe.py', 'tests/unit/test_native_full_actor.py',
        'tests/unit/test_nativeinit.py']
    code = f'''import pathlib,subprocess,os,json,time
r=pathlib.Path({RUNTIME!r}); assert r.is_dir()
name={name!r}; receipt=r/(name+'.json'); assert not receipt.exists()
env={{**os.environ,'PYTHONPATH':str(r/'src')+':'+str(r), 'CUDA_VISIBLE_DEVICES':'',
 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
 'NATIVEINIT_BASELINE':str(r/'baseline'), 'PATH':'/usr/bin:/bin'}}
import shutil
assert shutil.which('sumo',path=env['PATH']) is None
start=time.time()
with (r/(name+'.log')).open('xb') as log:
 p=subprocess.run(['/root/miniconda3/bin/python','-B','-m','pytest','-q',
  *{targets!r},'--basetemp='+str(r/('pytest-'+name))],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT)
doc={{'exit':p.returncode,'seconds':time.time()-start,'targets':{targets!r}}}
receipt.write_text(json.dumps(doc,indent=2)); print(json.dumps(doc)); print((r/(name+'.log')).read_text())
'''
    if not full:
        remote(code, timeout=200)
    else:
        remote(f'''import pathlib,subprocess,json
r=pathlib.Path({RUNTIME!r}); name={name!r}
assert not (r/(name+'.pid.json')).exists() and not (r/(name+'.json')).exists()
with (r/(name+'.driver.log')).open('xb') as log:
 p=subprocess.Popen(['/root/miniconda3/bin/python','-B','-c',{code!r}],cwd=r,
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
doc={{'pid':p.pid,'start_ticks':pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21]}}
(r/(name+'.pid.json')).write_text(json.dumps(doc)); print(json.dumps(doc))
''')


def status(name=None):
    if name:
        assert name.replace('-', '').isalnum()
        remote(f'''import pathlib,json
r=pathlib.Path({RUNTIME!r}); name={name!r}
for suffix in ('.pid.json','.json','.driver.log','.log'):
 p=r/(name+suffix)
 if p.exists(): print(p.name,p.read_text()[-16000:])
''')
    else:
        remote(f'''import pathlib,shutil,json
r=pathlib.Path({RELEASE!r})
for f in ('AUTONOMY_STATUS.json','AUTONOMY_HOLD_C3_VERIFIED.json','AUTONOMY_HOLD_C3.json'):
 p=r/f
 if p.exists(): print(f,p.read_text())
p=r/'arm/loop/wm/c02.log'; print('WM2 log',p.read_text()[-2000:])
print('free bytes',shutil.disk_usage(r).free)
''')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['stage', 'tests', 'full-tests', 'status', 'test-status'])
    ap.add_argument('--name', default='merge-longmem')
    args = ap.parse_args()
    if args.action == 'stage':
        stage()
    elif args.action in ('status', 'test-status'):
        status(args.name if args.action == 'test-status' else None)
    else:
        tests(args.name, args.action == 'full-tests')
