"""Create an immutable, owned gpuA source snapshot; never update peer runtimes."""
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import base64
import shlex

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/longint'))
from operate import remote

BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', re.I)


def main():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True)
    if dirty.strip():
        raise ValueError('commit source changes before staging')
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'HEAD', '--',
                                    'src', 'tools', 'configs', 'tests/unit', 'traffic_shepherd',
                                    'scripts/c4', 'scripts/round13c_waiter.py', 'pyproject.toml'],
                                    cwd=ROOT, text=True).splitlines()
    names = [n for n in names if not BAD.search(n) and
             (n.endswith(('.py', '.toml')) or n.startswith('src/wms/model/') and n.endswith('.json'))]
    data = io.BytesIO()
    manifest = {}
    blobs = subprocess.check_output(['git', 'cat-file', '--batch'], cwd=ROOT,
        input=''.join(f'HEAD:{name}\n' for name in names).encode())
    stream = io.BytesIO(blobs)
    base = '/root/autodl-tmp/c4/restart20261026/runtime_longint'
    code = f'import pathlib;print((pathlib.Path({base!r})/"SOURCE.json").read_text())'
    encoded = base64.b64encode(code.encode()).decode()
    command = shlex.join(['/root/miniconda3/bin/python', '-B', '-c',
                          f'import base64;exec(base64.b64decode({encoded!r}))'])
    source = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             'gpuA', command], capture_output=True, check=True, timeout=25)
    existing = json.loads(source.stdout)
    transferred = 0
    with tarfile.open(fileobj=data, mode='w:gz') as archive:
        for name in names:
            header = stream.readline().split()
            assert header[1] == b'blob'
            blob = stream.read(int(header[2]))
            assert stream.read(1) == b'\n'
            manifest[name] = hashlib.sha256(blob).hexdigest()
            if existing.get(name) == manifest[name]:
                continue
            transferred += 1
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            archive.addfile(info, io.BytesIO(blob))
        blob = json.dumps({'commit': commit, 'files': manifest}, indent=1).encode()
        info = tarfile.TarInfo('SOURCE.json')
        info.size = len(blob)
        archive.addfile(info, io.BytesIO(blob))
    runtime = f'/root/autodl-tmp/c4/restart20261026/runtime_c4prep_{commit[:12]}'
    remote(f'''import pathlib,sys,tarfile,io,json,hashlib
r=pathlib.Path({runtime!r}); assert r.parent.is_dir()
base=pathlib.Path({base!r})
raw=sys.stdin.buffer.read({len(data.getvalue())})
assert hashlib.sha256(raw).hexdigest()=={hashlib.sha256(data.getvalue()).hexdigest()!r}
if r.exists():
 assert json.loads((r/'SOURCE.json').read_text())['commit']=={commit!r}
 for name,digest in json.loads((r/'SOURCE.json').read_text())['files'].items():
  assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest
else:
 r.mkdir()
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
  for item in archive:
   name=pathlib.PurePosixPath(item.name)
   assert item.isfile() and not name.is_absolute() and '..' not in name.parts
   target=r/item.name; target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as out: out.write(archive.extractfile(item).read())
 manifest=json.loads((r/'SOURCE.json').read_text())['files']
 for name,digest in manifest.items():
  target=r/name
  if not target.exists():
   blob=(base/name).read_bytes(); assert hashlib.sha256(blob).hexdigest()==digest
   target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as out: out.write(blob)
  assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
print(json.dumps({{'runtime':str(r),'commit':{commit!r},'files':{len(manifest)},'transferred':{transferred}}}))
''', data.getvalue(), timeout=60)


if __name__ == '__main__':
    main()
