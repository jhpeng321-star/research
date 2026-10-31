"""Stage only committed INT inference code in a NEW immutable remote directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from tools.rbint.remote import ssh


def main():
    p = argparse.ArgumentParser()
    p.add_argument('host', choices=('gpuA', 'gpuR'))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    paths = ['src/wms', 'tools/rbint', 'tools/rbeval',
             'configs/execparity_cav_idm_v1.json', 'configs/wm_phys.toml']
    archive = subprocess.check_output(['git', 'archive', '--format=tar', sha, *paths], cwd=root)
    archive_sha = hashlib.sha256(archive).hexdigest()
    parent = '/root/rbeval20261031'
    dest = f'{parent}/phys_code_{sha[:12]}'
    # This archive consists solely of this repository's committed code/data receipts.
    code = f'''
import io, json, hashlib, tarfile
from pathlib import Path
parent=Path({parent!r})
assert parent.parent.is_dir()
parent.mkdir(exist_ok=True)
dest=Path({dest!r})
if dest.exists():
 receipt=json.loads((dest/'INT_STAGE.json').read_text())
 assert receipt['commit']=={sha!r} and receipt['archive_sha256']=={archive_sha!r}
 print(json.dumps(receipt)); raise SystemExit(0)
dest.mkdir()
'''
    prepared = ssh(args.host, code + '\nprint("NEW_STAGE",str(dest))')
    print(prepared)
    if 'NEW_STAGE' not in prepared:
        return
    # Send trusted local git bytes, never a remote download or peer runtime edit.
    unpack = f'''import sys,io,tarfile,hashlib,json
from pathlib import Path
data=sys.stdin.buffer.read();assert hashlib.sha256(data).hexdigest()=={archive_sha!r}
dest=Path({dest!r});assert dest.is_dir() and not any(dest.iterdir())
with tarfile.open(fileobj=io.BytesIO(data)) as tar:
 for m in tar.getmembers():
  assert not m.issym() and not m.islnk() and '..' not in Path(m.name).parts and not Path(m.name).is_absolute()
 tar.extractall(dest,filter='data')
receipt=dict(host={args.host!r},commit={sha!r},archive_sha256={archive_sha!r},runtime=str(dest))
with (dest/'INT_STAGE.json').open('x') as f: json.dump(receipt,f,indent=2)
print(json.dumps(receipt))'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host,
        '/root/miniconda3/bin/python', '-B', '-c', __import__('shlex').quote(unpack)],
        input=archive, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr.decode() + result.stdout.decode())
    print(result.stdout.decode())


if __name__ == '__main__':
    main()
