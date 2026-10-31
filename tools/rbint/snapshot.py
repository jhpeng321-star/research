"""Read-only, hash-bound snapshot of registered result shards; no dispatch."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', choices=('gpuA', 'gpuR'), required=True)
    p.add_argument('--roots', nargs='+', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists() or not out.parent.is_dir():
        raise ValueError('new snapshot in an existing parent required')
    for name in args.roots:
        if not name.startswith('/root/rbeval20261031/') or any(x in name.lower() for x in
                ('..', 'sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1')):
            raise ValueError('registered result roots only')
    code = '''import hashlib,io,json,sys,tarfile,time
from pathlib import Path
roots=ROOTS
files={}; skipped=[]
for n,root in enumerate(roots):
 root=Path(root);assert root.is_dir()
 paths=[root/'IDENTITY.json']+sorted(root.glob('w*_k*.json'))
 for path in paths:
  data=path.read_bytes()
  try: json.loads(data)
  except json.JSONDecodeError:
   skipped.append(str(path));continue
  files[f'shard{n}/'+path.name]=data
receipt=dict(unix=time.time(),roots=roots,read_only=True,skipped_incomplete=skipped,
 hashes={name:hashlib.sha256(data).hexdigest() for name,data in files.items()})
files['SNAPSHOT_RECEIPT.json']=json.dumps(receipt,indent=2).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name,data in files.items():
  info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
'''.replace('ROOTS', repr(args.roots))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        args.host, '/root/miniconda3/bin/python -B -c '+shlex.quote(code)],
        capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile() or member.name.startswith('/') or '..' in Path(member.name).parts:
                raise ValueError('invalid snapshot member')
        receipt = json.loads(tar.extractfile('SNAPSHOT_RECEIPT.json').read())
        for name, expected in receipt['hashes'].items():
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == expected
        out.mkdir()
        tar.extractall(out, filter='data')
    print(json.dumps(dict(out=str(out), files=len(receipt['hashes']),
                          skipped_incomplete=receipt['skipped_incomplete'])))


if __name__ == '__main__':
    main()
