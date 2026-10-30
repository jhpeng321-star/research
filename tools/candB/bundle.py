"""Build a fresh LF, source-only runtime archive from committed allowlisted files."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOTS = ['src', 'traffic_shepherd', 'configs', 'assets', 'baselines', 'scenarios',
         'output/controller_verification_v2/pi_saturation_v1.py',
         'output/controller_verification_v4/pi_external_u.py',
         'output/controller_verification_v7/pi_owner_adapter.py',
         'tools/candA', 'tools/candB', 'tools/execparity', 'tools/errsrc',
         'tools/distill', 'tools/setupaudit', 'tests/unit', 'tests/__init__.py',
         'authority', 'pyproject.toml', 'requirements.txt']
BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', re.I)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--commit', help='explicit immutable Git commit; exclude all working-tree changes')
    a = p.parse_args()
    if a.out.exists() or not a.out.parent.is_dir():
        raise ValueError('archive must be fresh, with existing parent')
    if not a.commit and subprocess.check_output(['git', 'status', '--porcelain']).strip():
        raise ValueError('commit before source bundling')
    commit = subprocess.check_output(['git', 'rev-parse', '--verify',
                                     (a.commit or 'HEAD')+'^{commit}'], text=True).strip()
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit,
                                     '--', *ROOTS], text=True).splitlines()
    names = [name for name in names if not BAD.search(name)
             and not name.startswith(('tools/setupaudit/evidence/',))]
    manifest = {}
    blobs = subprocess.Popen(['git', 'cat-file', '--batch'], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE)
    with tarfile.open(a.out, 'x:gz') as archive:
        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if name.endswith('.sh') else 0o644
            archive.addfile(info, io.BytesIO(data))
        for name in names:
            blobs.stdin.write(f'{commit}:{name}\n'.encode())
            blobs.stdin.flush()
            header = blobs.stdout.readline().split()
            if len(header) != 3 or header[1] != b'blob':
                raise ValueError('runtime source is not a Git blob: '+name)
            data = blobs.stdout.read(int(header[2]))
            if blobs.stdout.read(1) != b'\n':
                raise ValueError('Git blob stream corrupted')
            manifest[name] = hashlib.sha256(data).hexdigest()
            add(name, data)
        add('SOURCE.json', json.dumps(dict(commit=commit)).encode())
        add('SOURCE_MANIFEST.json', json.dumps(manifest, sort_keys=True, indent=1).encode())
    blobs.stdin.close()
    if blobs.wait() != 0:
        raise ValueError('Git source stream failed')
    print(json.dumps(dict(commit=commit, files=len(manifest), archive=str(a.out))))


if __name__ == '__main__':
    main()
