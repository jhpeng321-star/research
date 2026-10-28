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
import argparse
import os

ROOT = Path(__file__).resolve().parents[2]


def remote(source, data, timeout=60):
    """One fresh SSH call; retry decisions remain explicit and >=60s apart."""
    encoded = base64.b64encode(source.encode()).decode()
    command = shlex.join(['/root/miniconda3/bin/python', '-B', '-c',
        f'import base64;exec(base64.b64decode({encoded!r}))'])
    ssh = ['ssh']
    if os.environ.get('C4_SSH_HOSTNAME'):
        ssh += ['-o', 'HostName=' + os.environ['C4_SSH_HOSTNAME'],
                '-o', 'HostKeyAlias=[connect.bjb1.seetacloud.com]:11000',
                '-o', 'StrictHostKeyChecking=yes']
    if os.environ.get('C4_SSH_JUMP'):
        ssh += ['-J', os.environ['C4_SSH_JUMP']]
    result = subprocess.run([*ssh, '-v', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'ControlPersist=no',
        'gpuA', command], input=data, capture_output=True, timeout=timeout,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    print(result.stdout.decode(), end='')
    print(result.stderr.decode(), end='', file=sys.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)

BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', re.I)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checks', action='store_true',
                        help='detach source-bound qualification in the same SSH transaction')
    parser.add_argument('--base-commit', help='reuse a receipt-confirmed immutable runtime without an SSH probe')
    args = parser.parse_args(argv)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True)
    if dirty.strip():
        raise ValueError('commit source changes before staging')
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'HEAD', '--',
                                    'src', 'tools', 'configs', 'tests/unit', 'traffic_shepherd',
                                    'scripts/c4', 'scripts/round13c_waiter.py', 'pyproject.toml'],
                                    cwd=ROOT, text=True).splitlines()
    names = [n for n in names if not BAD.search(n) and
             (n.endswith(('.py', '.toml')) or n.startswith('src/wms/model/') and n.endswith('.json')
              or n == 'configs/native_cav_head_v5.json')]
    data = io.BytesIO()
    manifest = {}
    blobs = subprocess.check_output(['git', 'cat-file', '--batch'], cwd=ROOT,
        input=''.join(f'HEAD:{name}\n' for name in names).encode())
    stream = io.BytesIO(blobs)
    base = '/root/autodl-tmp/c4/restart20261026/runtime_longint'
    # A full compressed source-only bundle avoids a separate manifest SSH probe.
    # No datasets/checkpoints enter this allowlisted archive.
    existing = {}
    if args.base_commit:
        assert re.fullmatch(r'[0-9a-f]{40}', args.base_commit), 'full receipt-bound source SHA required'
        base = f'/root/autodl-tmp/c4/restart20261026/runtime_c4prep_{args.base_commit[:12]}'
        old_names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', args.base_commit, '--',
            'src', 'tools', 'configs', 'tests/unit', 'traffic_shepherd', 'scripts/c4',
            'scripts/round13c_waiter.py', 'pyproject.toml'], cwd=ROOT, text=True).splitlines()
        old_names = [n for n in old_names if n in names]
        old_blobs = io.BytesIO(subprocess.check_output(['git', 'cat-file', '--batch'], cwd=ROOT,
            input=''.join(f'{args.base_commit}:{name}\n' for name in old_names).encode()))
        for name in old_names:
            header = old_blobs.readline().split()
            assert header[1] == b'blob'
            blob = old_blobs.read(int(header[2]))
            assert old_blobs.read(1) == b'\n'
            existing[name] = hashlib.sha256(blob).hexdigest()
    transferred = 0
    # Fast source-only compression keeps local preparation bounded too; the
    # remote installer already accepts either archive codec without a probe.
    with tarfile.open(fileobj=data, mode='w:gz', compresslevel=1) as archive:
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
    install = f'''import pathlib,sys,tarfile,io,json,hashlib
r=pathlib.Path({runtime!r}); assert r.parent.is_dir()
base=pathlib.Path({base!r})
if {bool(args.base_commit)!r}:
 assert json.loads((base/'SOURCE.json').read_text())['commit']=={args.base_commit!r}
raw=sys.stdin.buffer.read({len(data.getvalue())})
assert hashlib.sha256(raw).hexdigest()=={hashlib.sha256(data.getvalue()).hexdigest()!r}
if r.exists():
 assert json.loads((r/'SOURCE.json').read_text())['commit']=={commit!r}
 for name,digest in json.loads((r/'SOURCE.json').read_text())['files'].items():
  assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest
else:
 r.mkdir()
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:*') as archive:
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
out=r.parent/'arm/loop/c4prep'; assert out.parent.is_dir();out.mkdir(exist_ok=True)
receipt={{'runtime':str(r),'commit':{commit!r},'files':{len(manifest)},'transferred':{transferred}}}
path=out/('STAGING_'+{commit[:12]!r}+'.json')
if not path.exists(): path.write_text(json.dumps(receipt,indent=1))
tmp=out/'CURRENT_RUNTIME.tmp'; tmp.write_text(json.dumps(receipt,indent=1));tmp.replace(out/'CURRENT_RUNTIME.json')
print(json.dumps({{'runtime':str(r),'commit':{commit!r},'files':{len(manifest)},'transferred':{transferred}}}))
'''
    if args.checks:
        install += f'''\nimport runpy
from datetime import datetime
authority=json.loads((out/'MAIN_1700.json').read_text())
assert authority['authority']=='MAIN2026-10-28T17:00'
assert datetime.now().astimezone()<datetime.fromisoformat(authority['deadline'])
receipt=out/('CHECK_COORDINATOR_'+{commit[:12]!r}+'_PID.json')
if receipt.exists():
 print(json.dumps({{'qualification_already_dispatched':json.loads(receipt.read_text())}}))
else:
 sys.argv=['launch_remote','checks']
 runpy.run_path(str(r/'tools/c4prep/launch_remote.py'),run_name='__main__')
'''
    compile(install, '<c4prep-stage-install>', 'exec')
    print(json.dumps({'stage': 'bundle_ready', 'source_commit': commit,
                      'files': len(manifest), 'bytes': len(data.getvalue())}), flush=True)
    remote(install, data.getvalue(), timeout=60)


if __name__ == '__main__':
    main()
