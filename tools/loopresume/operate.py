"""Small source-only transport; heavy data and computations stay on authorized boxes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
RELEASE = '/root/autodl-tmp/c4/restart20261026'
CPU = '/home/ubuntu/c4/main14/restart20261026'


def remote(host, source, timeout=120):
    python = '/home/ubuntu/miniconda3/bin/python' if host.startswith('cpu') else '/root/miniconda3/bin/python'
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, python+' -B -'],
                            input=source, text=True, encoding='utf-8', capture_output=True, timeout=timeout)
    print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='')
    if result.returncode:
        raise SystemExit(result.returncode)


def stage(host):
    changed = subprocess.check_output(['git', 'diff', '--name-only', '--diff-filter=ACMRT', 'daac879c'], cwd=ROOT, text=True).splitlines()
    changed += subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
    names = sorted(set(n for n in changed if n.startswith(('src/', 'configs/', 'scripts/', 'tests/', 'tools/'))))
    payload, manifest = io.BytesIO(), {}
    with tarfile.open(fileobj=payload, mode='w') as archive:
        for name in names:
            if any(x in name.lower() for x in ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'vtype')):
                raise ValueError('refused source path')
            data = ((ROOT/name).read_bytes() if (ROOT/name).is_file() else
                    subprocess.check_output(['git', 'show', ':'+name], cwd=ROOT))
            manifest[name] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    release = CPU if host.startswith('cpu') else RELEASE
    source = '''import hashlib,json,pathlib,sys,tarfile,shutil
release=pathlib.Path(sys.argv[1]); assert release.is_dir()
root=release/'runtime'; assert root.is_dir()
manifest=json.loads(sys.argv[2]); seen=set()
backup=release/'loopresume_source_before'; backup.mkdir(exist_ok=True)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
 for entry in archive:
  path=pathlib.PurePosixPath(entry.name)
  assert entry.isfile() and not path.is_absolute() and '..' not in path.parts and entry.name in manifest
  data=archive.extractfile(entry).read(); assert hashlib.sha256(data).hexdigest()==manifest[entry.name]
  target=root/entry.name
  if target.exists() and not (backup/entry.name).exists():
   (backup/entry.name).parent.mkdir(parents=True,exist_ok=True); shutil.copy2(target,backup/entry.name)
  target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data); seen.add(entry.name)
assert seen==set(manifest)
out=release/'LOOPRESUME_SOURCE.json'; out.write_text(json.dumps(manifest,sort_keys=True))
print(json.dumps({'files':len(seen),'manifest_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
'''
    python = '/home/ubuntu/miniconda3/bin/python' if host.startswith('cpu') else '/root/miniconda3/bin/python'
    command = ['ssh', host, shlex.join([python, '-B', '-c', source, release, json.dumps(manifest)])]
    subprocess.run(command, input=payload.getvalue(), check=True, timeout=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'register', 'tests', 'install', 'launch', 'status', 'wait-gates'])
    parser.add_argument('--host', default='gpuA', choices=['gpuA', 'cpuP', 'cpuQ'])
    args = parser.parse_args()
    if args.action == 'stage':
        stage(args.host)
        return
    source = f"import runpy,sys; sys.argv=['box.py',{args.action!r}]; runpy.run_path({RELEASE+'/runtime/tools/loopresume/box.py'!r},run_name='__main__')"
    remote('gpuA', source, timeout=1100 if args.action in ('tests', 'wait-gates') else 180)


if __name__ == '__main__':
    main()
