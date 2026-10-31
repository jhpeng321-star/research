"""Install an immutable code snapshot under a task-owned existing-parent root."""
import argparse
import subprocess

from .audit import digest, safe


def main():
    p = argparse.ArgumentParser()
    p.add_argument('host', choices=('gpuR', 'gpuA', 'cpuQ'))
    p.add_argument('archive')
    p.add_argument('tag')
    a = p.parse_args()
    archive = safe(a.archive)
    if not a.tag.replace('_', '').isalnum():
        raise ValueError('plain snapshot tag required')
    home = '/home/ubuntu' if a.host == 'cpuQ' else '/root'
    root = home+'/wmvalue20261031'
    python = home+'/miniconda3/bin/python'
    check = subprocess.run(['ssh', a.host, 'test -d '+home], check=True)
    assert check.returncode == 0
    target = home+'/wmvalue_code_'+a.tag+'.zip'
    subprocess.run(['scp', str(archive), a.host+':'+target], check=True)
    code = f'''
from pathlib import Path
import hashlib, zipfile
parent=Path({home!r})
assert parent.is_dir()
archive=Path({target!r})
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={digest(archive)!r}
root=Path({root!r})
root.mkdir(exist_ok=True)
dest=root/{a.tag!r}
assert not dest.exists(), 'existing snapshot: inspect, never overwrite'
dest.mkdir()
with zipfile.ZipFile(archive) as z:
    assert all(n.startswith(('tools/wmvalue/', 'configs/')) and '..' not in n for n in z.namelist())
    z.extractall(dest)
print('DEPLOYED', dest, flush=True)
'''
    subprocess.run(['ssh', a.host, python+' -B -'], input=code, text=True, check=True, timeout=30)


if __name__ == '__main__':
    main()
