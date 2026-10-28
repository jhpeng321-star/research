"""Incrementally fetch only entrant-owned JSON evidence; preserve conflicting files."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def command(source):
    encoded = base64.b64encode(source.encode()).decode()
    return "python3 -c 'import base64;exec(base64.b64decode(\"" + encoded + "\"))'"


def gather(host, destination, receipts_only=False):
    assert host in ('cpuP', 'cpuQ')
    root = destination / host
    root.mkdir(parents=True, exist_ok=True)
    index = {str(f.relative_to(root)).replace('\\', '/'): hashlib.sha256(f.read_bytes()).hexdigest()
             for f in root.rglob('*.json') if 'snapshots' not in f.parts}
    source = '''
import hashlib,io,json,sys,tarfile,time
from pathlib import Path
p=Path.home()/'entrant';assert p.is_dir();skip=json.load(sys.stdin)
files=[] if ONLY else list((p/'results_v4').glob('*_s600.json'))
if not ONLY:
    files += list((p/'results').glob('e3_g01_*_k0_s600.json'))
    files += list((p/'results_v2').glob('e3_g01_*_k0_s600.json'))
files += [f for f in p.glob('*.json') if f.name not in ('DONE.json','CANARY_DONE.json')]
threshold=p/'runtime/configs/coopw_horizon_gate_13E_b2.json'
if threshold.exists():files.append(threshold)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
    for f in files:
        assert f.is_file() and not f.is_symlink()
        rel=f.relative_to(p).as_posix()
        assert not any(s in rel.lower() for s in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth'))
        if skip.get(rel)==hashlib.sha256(f.read_bytes()).hexdigest():continue
        tar.add(f,arcname=rel,recursive=False)
'''.replace('ONLY', repr(receipts_only))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, command(source)],
                            input=json.dumps(index).encode(), capture_output=True, check=True)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:gz') as tar:
        for item in tar:
            path = root / item.name
            assert item.isfile() and path.resolve().is_relative_to(root.resolve())
            data = tar.extractfile(item).read()
            if path.exists():
                previous = path.read_bytes()
                if previous == data:
                    continue
                assert not item.name.startswith('results'), 'scientific receipt conflict'
                digest = hashlib.sha256(previous).hexdigest()
                archive = root / 'snapshots' / digest / item.name
                archive.parent.mkdir(parents=True, exist_ok=True)
                if not archive.exists():
                    archive.write_bytes(previous)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            count += 1
    print(json.dumps({'host': host, 'new_files': count, 'bytes_received': len(result.stdout)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('host', choices=('cpuP', 'cpuQ'))
    parser.add_argument('--out', type=Path, default=ROOT/'evidence_entrant/raw')
    parser.add_argument('--receipts-only', action='store_true')
    args = parser.parse_args()
    gather(args.host, args.out, args.receipts_only)
