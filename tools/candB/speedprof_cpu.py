"""Stage this worktree's named source/CPU fixtures in an owned remote probe directory."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'vtype')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='gpuR')
    parser.add_argument('--root', default='/root/autodl-tmp/c4/speedprof_CPU_20261030')
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--gpu-probe', action='store_true',
                        help='only an idle gpuR, explicitly bounded synthetic forward probe')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.gpu_probe and args.host != 'gpuR':
        raise ValueError('GPU probes are restricted to gpuR')
    root = args.root
    if not root.startswith('/root/autodl-tmp/c4/speedprof_CPU_') or any(s in root.lower() for s in FORBIDDEN):
        raise ValueError('probe must use its own named root')
    subprocess.run(['ssh', args.host, 'test -d /root/autodl-tmp/c4'], check=True)
    paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '--',
        'src/wms', 'tools/candB', 'tools/candA', 'tools/gated300', 'tools/distill',
        'tools/setupaudit/summarize2.py',
        'tests/unit/test_eslong.py', 'tests/unit/test_gated300.py', 'tests/unit/test_gatemeas.py',
        'tests/unit/test_longgrad.py', 'tests/unit/test_longmem.py', 'tests/unit/test_longskip.py',
        'tests/unit/test_polspeed2_parallel.py', 'tests/unit/test_speedprof.py',
        'tests/unit/test_candb_integration.py', 'tests/__init__.py',
        'configs', 'pyproject.toml'], cwd=ROOT, text=True).splitlines()
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
    admission = ''
    if args.gpu_probe:
        admission = ('test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" && ')
    visible = '0' if args.gpu_probe else ''
    remote_receipt = f'PROBE_{time.time_ns()}.json'
    remote = (f'cd {shlex.quote(root)} && {admission}flock -n -E 75 CPU.lock '
              f'env CUDA_VISIBLE_DEVICES="{visible}" OMP_NUM_THREADS=1 '
              'MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=deps:src:. '
              'nice -n 19 /root/miniconda3/bin/python -B -m tools.candB.speedprof_run '
              f'--receipt {remote_receipt} -- ' + shlex.join(command))
    start = time.time()
    result = subprocess.run(['ssh', args.host, remote], capture_output=True, text=True)
    if args.receipt:
        args.receipt.write_text(json.dumps(dict(host=args.host, root=root, start_epoch=start,
            seconds=time.time()-start, command=command, returncode=result.returncode,
            remote_receipt=remote_receipt,
            stdout=result.stdout, stderr=result.stderr), indent=1)+'\n', encoding='utf-8')
    print(result.stdout, end='')
    print(result.stderr, end='')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
