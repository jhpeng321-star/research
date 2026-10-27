"""Stage source and run CPU-only tests in ~/looplong; never writes peer runtimes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'vtype')


def ssh(source, *, data=None, timeout=180):
    command = ['/home/ubuntu/miniconda3/bin/python', '-B', '-c', source]
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'cpuP',
                           shlex.join(command)], input=data, capture_output=True, timeout=timeout)
    print(proc.stdout.decode('utf-8'), end='')
    print(proc.stderr.decode('utf-8'), end='', file=__import__('sys').stderr)
    if proc.returncode:
        raise SystemExit(proc.returncode)
    return proc.stdout


def stage():
    files = subprocess.check_output(['git', 'ls-files', 'src', 'configs', 'scripts/c4/loop14.py',
        'scripts/c4/loop14_fast.py', 'scripts/c4/loop14_resume.py', 'tools/r10b_checks/run_onestep.py'],
        cwd=ROOT, text=True).splitlines()
    files += subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'],
                                     cwd=ROOT, text=True).splitlines()
    files += ['pyproject.toml', 'LOOPLONG.md', 'tests/unit/test_looplong.py',
              'tests/unit/test_policy14_integration.py', 'tests/unit/test_coopw.py',
              'tests/unit/test_paircrit.py', 'tools/loop14_checks/accept.py',
              'tools/r13e_checks/offline.py', 'tools/wm_eval_common.py',
              'tools/looplong/dry_run.py', 'scripts/c4/loop14_long.py']
    files = sorted({f for f in files if not any(t in f.lower() for t in BAD)
                    and f.endswith(('.py', '.json', '.toml', '.md'))})
    payload = io.BytesIO()
    manifest = {}
    with tarfile.open(fileobj=payload, mode='w') as archive:
        for f in files:
            path = ROOT / f
            data = path.read_bytes() if path.is_file() else subprocess.check_output(['git', 'show', 'HEAD:' + f], cwd=ROOT)
            manifest[f] = hashlib.sha256(data).hexdigest()
            entry = tarfile.TarInfo(f)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    ssh("import pathlib; p=pathlib.Path.home(); assert p.is_dir(); (p/'looplong').mkdir(exist_ok=True)")
    source = '''import pathlib,tarfile,sys,hashlib,json
root=pathlib.Path.home()/'looplong'; assert root.is_dir()
manifest={{}}
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
 for e in archive:
  p=pathlib.PurePosixPath(e.name)
  assert e.isfile() and not p.is_absolute() and '..' not in p.parts
  assert not any(t in e.name.lower() for t in {bad!r})
  data=archive.extractfile(e).read(); manifest[e.name]=hashlib.sha256(data).hexdigest()
  target=root/e.name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
(root/'SOURCE.json').write_text(json.dumps(manifest,sort_keys=True))
assert hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()=={expected!r}
print(json.dumps({{'files':len(manifest),'source_sha256':hashlib.sha256((root/'SOURCE.json').read_bytes()).hexdigest()}}))
'''.format(bad=BAD, expected=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())
    ssh(source, data=payload.getvalue())


def run():
    data = ssh("import pathlib,subprocess,os; r=pathlib.Path.home()/'looplong'; assert r.is_dir(); "
        "env={**os.environ,'PYTHONPATH':str(r/'src')+':'+str(r),'CUDA_VISIBLE_DEVICES':'',"
        "'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}; "
        "p=subprocess.run(['/home/ubuntu/miniconda3/bin/python','-B','-m','pytest','-q',"
        "'tests/unit/test_looplong.py','tests/unit/test_policy14_integration.py',"
        "'tests/unit/test_coopw.py','--basetemp='+str(r/'pytest_tmp')],cwd=r,env=env); "
        "raise SystemExit(p.returncode)", timeout=900)
    (ROOT / 'tools/looplong/TESTS.log').write_bytes(data)


def evidence():
    ssh("import pathlib,subprocess,os; r=pathlib.Path.home()/'looplong'; assert r.is_dir(); "
        "env={**os.environ,'PYTHONPATH':str(r/'src')+':'+str(r),'CUDA_VISIBLE_DEVICES':'',"
        "'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}; "
        "p=subprocess.run(['/home/ubuntu/miniconda3/bin/python','-B','tools/looplong/dry_run.py'],cwd=r,env=env); "
        "raise SystemExit(p.returncode)", timeout=900)


def fetch():
    data = ssh("import pathlib,sys; p=pathlib.Path.home()/'looplong/tools/looplong/DRY_RUN.json'; "
               "sys.stdout.write(p.read_text())")
    target = ROOT / 'tools/looplong/DRY_RUN.json'
    assert target.parent.is_dir()
    target.write_bytes(data)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('stage', 'tests', 'dry-run', 'fetch'))
    args = ap.parse_args()
    {'stage': stage, 'tests': run, 'dry-run': evidence, 'fetch': fetch}[args.action]()
