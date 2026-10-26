"""Task-owned, bounded source staging and remote Python execution (no simulator)."""
import argparse
import io
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/root/autodl-tmp/c4/paircrit20261026'
DENY = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', re.I)


def ssh(code, host='gpuR'):
    result = subprocess.run(['ssh', host, '/root/miniconda3/bin/python -'], input=code,
        capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def stage():
    # Read tracked source only; exclude restricted filenames BEFORE opening.
    files = subprocess.check_output(['git', 'ls-files', 'src', 'configs', 'scripts/c4', 'traffic_shepherd', 'tools/fastloop', 'tools/main14',
        'tests/unit'], cwd=ROOT, text=True).splitlines()
    files += subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard',
        'src', 'configs', 'tests/unit'], cwd=ROOT, text=True).splitlines()
    files += [str(p.relative_to(ROOT)).replace('\\', '/')
              for p in (ROOT / 'tools/paircrit').glob('*.py')]
    files += ['tests/unit/test_paircrit.py']
    files += ['tests/__init__.py']
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for name in sorted(set(files)):
            if not DENY.search(name) and (ROOT / name).is_file():
                tar.add(ROOT / name, arcname=name, recursive=False)
    print(ssh(f"from pathlib import Path; p=Path('{REMOTE}'); "
              "assert p.parent.is_dir(); p.mkdir(exist_ok=True); print(p)"))
    subprocess.run(['ssh', 'gpuR', f'tar -xzf - -C {REMOTE}'], input=buf.getvalue(), check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stage', 'run', 'probe'])
    parser.add_argument('script', nargs='?')
    args = parser.parse_args()
    if args.action == 'stage':
        stage()
    elif args.action == 'probe':
        assert args.script and not DENY.search(args.script)
        print(ssh((ROOT / args.script).read_text(encoding='utf-8')))
    else:
        assert args.script and not DENY.search(args.script)
        code = ("import os,runpy,sys; "
            f"os.chdir('{REMOTE}'); "
            "os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',"
            "OPENBLAS_NUM_THREADS='4',C4_ALLOWED_ROOTS='/root'); "
            f"sys.path.insert(0,'{REMOTE}/src'); "
            f"runpy.run_path('{REMOTE}/{args.script}',run_name='__main__')")
        print(ssh(code))
