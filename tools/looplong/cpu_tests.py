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


def ssh(source, *, data=None, timeout=180, echo=True):
    command = ['/home/ubuntu/miniconda3/bin/python', '-B', '-c', source]
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'cpuP',
                           shlex.join(command)], input=data, capture_output=True, timeout=timeout)
    if echo:
        print(proc.stdout.decode('utf-8'), end='')
    print(proc.stderr.decode('utf-8'), end='', file=__import__('sys').stderr)
    if proc.returncode:
        raise SystemExit(proc.returncode)
    return proc.stdout


def stage(delta=False):
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
    if delta:
        files = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'],
                                        cwd=ROOT, text=True).splitlines()
    files = sorted({f for f in files if not any(t in f.lower() for t in BAD)
                    and f.endswith(('.py', '.json', '.toml', '.md'))})
    payload = io.BytesIO()
    manifest = {}
    # Sparse files come from Git, in one batch rather than hundreds of Windows
    # subprocess startups. Protected names have already been excluded.
    missing = [f for f in files if not (ROOT / f).is_file()]
    blobs = {}
    if missing:
        raw = subprocess.check_output(['git', 'cat-file', '--batch'], cwd=ROOT,
            input=''.join('HEAD:' + f + '\n' for f in missing).encode())
        stream = io.BytesIO(raw)
        for f in missing:
            header = stream.readline().split()
            assert header[1] == b'blob', f
            blobs[f] = stream.read(int(header[2]))
            assert stream.read(1) == b'\n'
    with tarfile.open(fileobj=payload, mode='w') as archive:
        for f in files:
            path = ROOT / f
            data = path.read_bytes() if path.is_file() else blobs[f]
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


def start(action, run_id='resume'):
    """Detach only our CPU child and return immediately; poll by explicit receipt."""
    command = (['-m', 'pytest', '-q', 'tests/unit/test_looplong.py',
                'tests/unit/test_policy14_integration.py', 'tests/unit/test_coopw.py',
                '--basetemp=pytest_resume_tmp'] if action == 'tests'
               else ['tools/looplong/dry_run.py', '--checkpoint-parity'] if action == 'speed-checkpoint'
               else ['tools/looplong/dry_run.py'])
    if not run_id.replace('-', '').isalnum():
        raise ValueError('run id must be alphanumeric with optional dashes')
    name = action + '_' + run_id
    worker = ("import subprocess,json,pathlib,time; start=time.time(); "
              f"p=subprocess.run({['/home/ubuntu/miniconda3/bin/python', '-B', *command]!r}); "
              f"pathlib.Path({name + '.json'!r}).write_text(json.dumps("
              "{'status':'done','exit':p.returncode,'seconds':time.time()-start}))")
    ssh("import pathlib,subprocess,os,json; r=pathlib.Path.home()/'looplong'; assert r.is_dir(); "
        "env={**os.environ,'PYTHONPATH':str(r/'src')+':'+str(r),'CUDA_VISIBLE_DEVICES':'',"
        "'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}; "
        f"name={name!r}; log=r/(name+'.log'); status=r/(name+'.json'); "
        "assert not status.exists(), 'preserve previous run; use a new receipt name'; "
        f"code={worker!r}; "
        "status.write_text(json.dumps({'status':'running'})); "
        "p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-c',code],"
        "cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=log.open('xb'),"
        "stderr=subprocess.STDOUT,start_new_session=True); print(json.dumps({'pid':p.pid,'log':str(log)}))")


def poll(action, run_id='resume'):
    ssh("import pathlib; r=pathlib.Path.home()/'looplong'; "
        f"print((r/{action + '_' + run_id + '.json'!r}).read_text()); "
        f"print((r/{action + '_' + run_id + '.log'!r}).read_text())")


def collect(action, run_id):
    if action == 'tests':
        remote, local = f'tests_{run_id}.log', 'tools/looplong/TESTS.log'
    elif action == 'speed-checkpoint':
        remote = local = 'tools/looplong/CHECKPOINT_SPEED.json'
    else:
        remote = local = 'tools/looplong/DRY_RUN.json'
    data = ssh("import pathlib,sys,json; r=pathlib.Path.home()/'looplong'; "
        f"s=json.loads((r/{action + '_' + run_id + '.json'!r}).read_text()); "
        "assert s['status']=='done' and s['exit']==0; "
        f"sys.stdout.write((r/{remote!r}).read_text())", echo=False)
    target = ROOT / local
    assert target.parent.is_dir()
    target.write_bytes(data)
    print(json.dumps({'collected': local, 'sha256': hashlib.sha256(data).hexdigest()}))


def background_stage():
    import sys
    log = ROOT / 'tools/looplong/stage_resume_background.log'
    assert log.parent.is_dir()
    proc = subprocess.Popen([sys.executable, '-B', str(Path(__file__)), 'stage'], cwd=ROOT,
        stdin=subprocess.DEVNULL, stdout=log.open('ab'), stderr=subprocess.STDOUT,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    print(json.dumps({'pid': proc.pid, 'log': str(log)}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('stage', 'stage-delta', 'tests', 'dry-run', 'fetch',
        'start-tests', 'start-dry-run', 'poll-tests', 'poll-dry-run', 'background-stage',
        'start-speed-checkpoint', 'poll-speed-checkpoint', 'collect-tests', 'collect-dry-run',
        'collect-speed-checkpoint'))
    ap.add_argument('--run-id', default='resume')
    args = ap.parse_args()
    {'stage': stage, 'stage-delta': lambda: stage(True), 'tests': run, 'dry-run': evidence, 'fetch': fetch,
     'start-tests': lambda: start('tests', args.run_id), 'start-dry-run': lambda: start('dry-run', args.run_id),
      'poll-tests': lambda: poll('tests', args.run_id), 'poll-dry-run': lambda: poll('dry-run', args.run_id),
      'start-speed-checkpoint': lambda: start('speed-checkpoint', args.run_id),
      'poll-speed-checkpoint': lambda: poll('speed-checkpoint', args.run_id),
      'collect-speed-checkpoint': lambda: collect('speed-checkpoint', args.run_id),
      'collect-tests': lambda: collect('tests', args.run_id),
      'collect-dry-run': lambda: collect('dry-run', args.run_id),
     'background-stage': background_stage}[args.action]()
