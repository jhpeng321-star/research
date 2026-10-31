"""Stage our code and manage one owned CPU diagnostic; never signals processes.

Usage: python -m tools.rblane.operate smoke|panel|panel2|status|collect
Runs are new immutable directories; completed smoke/panel cannot be redispatched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = ROOT / 'tools/rblane/receipts'
PYTHON = '/home/ubuntu/miniconda3/bin/python'


def remote(code):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             'cpuQ', PYTHON + ' -B -'], input=code, text=True,
                            capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def compatible_mobil_baseline(parameter_sha):
    """Reuse only an already completed identical MOBIL experiment/transform."""
    from dataclasses import asdict
    from wms.physics.config import PhysicsConfig
    from .snapshot import SNAPSHOT_CONTRACT
    physics = asdict(PhysicsConfig(sumo_fidelity=True, open_road=True))
    dependencies = ('mobil.py', 'neighbours.py', 'idm.py', 'config.py')
    current = {name: hashlib.sha256((ROOT/'src/wms/physics'/name).read_bytes()).hexdigest()
               for name in dependencies}
    matches = []
    for receipt in RECEIPTS.glob('PANEL*_COMPLETE.json'):
        completed = json.loads(receipt.read_text())
        summary_path = RECEIPTS / (receipt.name.removesuffix('_COMPLETE.json').lower() + '_results/SUMMARY.json')
        if not completed['complete'] or completed['alive'] or not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        if (summary['status'] != 'COMPLETE' or summary['episodes'] != 160
                or summary['physics'] != physics or summary['horizon_s'] != 60
                or summary['snapshot'] != SNAPSHOT_CONTRACT
                or summary.get('motion_parameter_sha256') != parameter_sha):
            continue
        previous = {Path(name).name: digest for name, digest in summary['source_sha256'].items()}
        if all(previous.get(name) == digest for name, digest in current.items()):
            matches.append(completed)
    return max(matches, key=lambda r: r['collected_at'])['root'] + '/results' if matches else None


def launch(kind, idm_parameters=None, encoder_checkpoint=None):
    from .validate import FORBIDDEN
    RECEIPTS.mkdir(exist_ok=True)
    receipt = RECEIPTS / (kind.upper() + '_DISPATCH.json')
    if receipt.exists():
        raise RuntimeError('dispatch receipt already exists: inspect status, do not duplicate')
    parent = Path('E:/Temp/opencode')
    if not parent.is_dir():
        raise RuntimeError('approved temporary parent missing')
    fd, name = tempfile.mkstemp(prefix='rblane_', suffix='.zip', dir=parent)
    import os
    os.close(fd)
    bundle = Path(name)
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for folder in ('src/wms', 'tools/rblane'):
            for path in sorted((ROOT / folder).rglob('*.py')):
                if any(term in str(path).lower() for term in FORBIDDEN):
                    continue
                archive.write(path, path.relative_to(ROOT).as_posix())
        if idm_parameters is not None:
            from .validate import load_motion_point
            load_motion_point(idm_parameters)
            archive.write(idm_parameters, 'inputs/idm_pooled.json')
        if encoder_checkpoint is not None:
            from .encoder import REGISTERED_WM2_SHA
            assert not any(term in str(encoder_checkpoint.resolve()).lower() for term in FORBIDDEN)
            assert hashlib.sha256(encoder_checkpoint.read_bytes()).hexdigest() == REGISTERED_WM2_SHA
            archive.write(encoder_checkpoint, 'inputs/wm_s.pt')
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    run = f'/home/ubuntu/rblane20261031_{kind}_{digest[:12]}'
    selected = encoder_checkpoint if encoder_checkpoint is not None else idm_parameters
    parameter_sha = hashlib.sha256(selected.read_bytes()).hexdigest() if selected is not None else None
    reuse = compatible_mobil_baseline(parameter_sha) if kind != 'smoke' else None
    preflight = remote(f'''import os,pathlib,json
p=pathlib.Path({run!r})
assert p.parent.is_dir() and not p.exists(), 'run exists: do not duplicate'
assert os.getloadavg()[0] < 20, 'load admission'
p.mkdir()
print(json.dumps(dict(root=str(p),load=os.getloadavg())))
''')
    # Persist the staging reservation before any dispatch, including failed stages.
    record = dict(kind=kind, root=run, bundle_sha256=digest,
                  parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  preflight=json.loads(preflight), status='STAGING', staged_at=time.time())
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    subprocess.run(['scp', '-q', str(bundle), 'cpuQ:' + run + '/code.zip'],
                   check=True, timeout=45)
    launched = remote(f'''import os,pathlib,zipfile,hashlib,json,subprocess,time
p=pathlib.Path({run!r})
assert hashlib.sha256((p/'code.zip').read_bytes()).hexdigest()=={digest!r}
assert not (p/'PID.json').exists()
with zipfile.ZipFile(p/'code.zip') as z: z.extractall(p)
env=dict(os.environ,PYTHONPATH=str(p/'src'),CUDA_VISIBLE_DEVICES='',
         OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
cmd=['/usr/bin/nice','-n','19',{PYTHON!r},'-B','-m','tools.rblane.validate',
     '--out',str(p/'results')]+(['--smoke'] if {kind=='smoke'!r} else [])
if {reuse!r} is not None: cmd += ['--reuse-mobil-from',{reuse!r}]
if {idm_parameters is not None!r}: cmd += ['--idm-parameters',str(p/'inputs/idm_pooled.json')]
if {encoder_checkpoint is not None!r}: cmd += ['--encoder-checkpoint',str(p/'inputs/wm_s.pt')]
with (p/'run.log').open('x') as log:
    child=subprocess.Popen(cmd,cwd=p,env=env,stdin=subprocess.DEVNULL,
       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
start=pathlib.Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21]
r=dict(pid=child.pid,start_ticks=start,command=cmd,root=str(p),started_at=time.time())
(p/'PID.json').write_text(json.dumps(r,indent=2)+'\\n')
print(json.dumps(r))
''')
    record.update(status='DISPATCHED', process=json.loads(launched))
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


def status(kind, collect=False):
    record = json.loads((RECEIPTS / (kind.upper() + '_DISPATCH.json')).read_text())
    run = record['root']
    code = f'''import pathlib,json,os
p=pathlib.Path({run!r}); proc=json.loads((p/'PID.json').read_text())
f=pathlib.Path('/proc/'+str(proc['pid'])+'/stat')
alive=f.exists() and f.read_text().split()[21]==proc['start_ticks'] and f.read_text().split()[2]!='Z'
summary=p/'results/SUMMARY.json'
lines=(p/'run.log').read_text().splitlines()
print(json.dumps(dict(alive=alive,complete=summary.exists(),
  progress=lines[-4:],load=os.getloadavg(),
  summary=json.loads(summary.read_text()) if summary.exists() else None)))
'''
    report = json.loads(remote(code))
    brief = {key: report[key] for key in ('alive', 'complete', 'load')}
    if report['summary']:
        brief.update(episodes=report['summary']['episodes'],
                     elapsed_s=report['summary']['elapsed_s'],
                     aggregate={model: {key: value for key, value in scores.items()
                          if key in ('exact', 'by_motivation', 'positive_runs')}
                          for model, scores in report['summary']['aggregate'].items()})
    else:
        brief['progress'] = report['progress']
    print(json.dumps(brief, indent=2))
    if collect:
        if report['alive'] or not report['complete']:
            raise RuntimeError('cannot collect incomplete/live job as complete')
        out = RECEIPTS / (kind + '_results')
        if out.exists():
            raise RuntimeError('results already collected')
        out.mkdir()
        subprocess.run(['scp', '-q', 'cpuQ:' + run + '/results/*.json', str(out)],
                       check=True, timeout=60)
        (RECEIPTS / (kind.upper() + '_COMPLETE.json')).write_text(json.dumps(
            dict(root=run, process=record['process'], bundle_sha256=record['bundle_sha256'],
                 complete=True, alive=False, collected_at=time.time()), indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command')
    ap.add_argument('--kind', default='smoke')
    ap.add_argument('--idm-parameters', type=Path)
    ap.add_argument('--encoder-checkpoint', type=Path)
    args = ap.parse_args()
    if args.idm_parameters is not None and args.encoder_checkpoint is not None:
        ap.error('choose one frozen parameter source')
    import re
    if not re.fullmatch(r'smoke|panel(?:[2-9]|[1-9][0-9]+)?', args.kind):
        ap.error('invalid run kind')
    if re.fullmatch(r'smoke|panel(?:[2-9]|[1-9][0-9]+)?', args.command):
        launch(args.command, args.idm_parameters, args.encoder_checkpoint)
    elif args.command in ('status', 'collect'):
        status(args.kind, args.command=='collect')
    else:
        ap.error('unknown command')


if __name__ == '__main__':
    main()
