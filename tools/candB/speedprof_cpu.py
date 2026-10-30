"""Stage this worktree's named source/CPU fixtures in an owned remote probe directory."""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import shlex
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='gpuR')
    parser.add_argument('--root', default='/root/autodl-tmp/c4/speedprof_CPU_20261030')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.root
    if not root.startswith('/root/autodl-tmp/c4/speedprof_CPU_') or any(s in root.lower() for s in FORBIDDEN):
        raise ValueError('probe must use its own named root')
    subprocess.run(['ssh', args.host, 'test -d /root/autodl-tmp/c4'], check=True)
    paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '--',
        'src/wms', 'tools/candB', 'tools/candA', 'tools/gated300', 'tools/distill',
        'tests/unit/test_eslong.py', 'tests/unit/test_gated300.py', 'tests/unit/test_gatemeas.py',
        'tests/unit/test_longgrad.py', 'tests/unit/test_longmem.py', 'tests/unit/test_longskip.py',
        'tests/unit/test_polspeed2_parallel.py', 'tests/unit/test_speedprof.py',
        'tests/unit/test_candb_integration.py', 'configs', 'pyproject.toml'], cwd=ROOT, text=True).splitlines()
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as archive:
        for path in sorted(set(paths)):
            if any(s in path.lower() for s in FORBIDDEN) or not (ROOT/path).is_file():
                continue
            if path.startswith('tools/candB/evidence/') or path.endswith(('.log', '.jsonl')):
                continue
            archive.add(ROOT/path, arcname=path, recursive=False)
    subprocess.run(['ssh', args.host, f'mkdir -p {shlex.quote(root)} && tar -xzf - -C {shlex.quote(root)}'],
                   input=data.getvalue(), check=True)
    command = args.command
    if command[:1] == ['--']:
        command = command[1:]
    command = command or ['-m', 'pytest', '-q', 'tests/unit/test_speedprof.py']
    remote = (f'cd {shlex.quote(root)} && env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 '
              'MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. '
              'nice -n 19 /root/miniconda3/bin/python -B ' + shlex.join(command))
    result = subprocess.run(['ssh', args.host, remote], capture_output=True, text=True)
    print(result.stdout, end='')
    print(result.stderr, end='')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
